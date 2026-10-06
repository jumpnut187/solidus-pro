"""Deep surrogate inference: ``DeepSurrogateModel`` (``SurrogateModel`` ABC).

The deep model drops into the shared ``SurrogateModel`` interface, so the
existing ``run_surrogate_pipeline``, ``SurrogateWorker``, and UI paths run it
unchanged.  Prediction is backend-aware:

1. Core ML — exported ``.mlpackage`` (Apple ANE / GPU / CPU) when present.
2. ONNX Runtime — exported ``.onnx`` graph on the best execution provider
   (Qualcomm / Intel / AMD NPU, NVIDIA GPU, DirectML, or ONNX Runtime CPU).
   On the first prediction the ONNX output is checked against PyTorch, and in
   ``auto`` mode both are timed and the faster one is kept.
3. PyTorch MPS — Apple GPU.
4. PyTorch CPU — universal fallback.

The network predicts ``log1p(start)`` and ``log1p(local)`` and derives
``end = start + local`` in-graph (``UNet3D.forward``).  ``reconstruct_schedule``
then applies the active-cell mask and the ``max_simulation_time`` clamp; it is
the single reconstruction used by ``DeepSurrogateModel.predict`` and by
``training._evaluate`` so reported accuracy measures what inference returns.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np

from core.deep_model.backend import DeepBackend, onnx_provider_description, torch_device_for
from core.deep_model.model import (
    DEFAULT_BASE_CHANNELS,
    DEFAULT_CONDITIONING_SIZE,
    END_TIME_FLOOR,
    UNet3D,
    levels_for_resolution,
    read_deep_manifest,
)
from core.surrogate.conditioning import Conditioning, raw_from_conditioning
from core.surrogate.model import ModelLoadResult, ModelManifest, SurrogateModel

DEFAULT_MAX_SIMULATION_TIME = 14400.0

#: The ONNX output must match PyTorch on active cells within
#: ``atol + rtol * |torch|`` seconds before ONNX Runtime is used.  The
#: tolerance covers fp16 execution on an NPU; it matches the Core ML export
#: regression test.
ONNX_MATCH_RTOL = 0.03
ONNX_MATCH_ATOL_S = 2.0
#: Timed predictions per backend when ``auto`` chooses between ONNX Runtime
#: and PyTorch (median of these, after the validation run as warm-up).
BENCHMARK_RUNS = 3

_LOG = logging.getLogger(__name__)


def max_simulation_time_from(conditioning: Conditioning) -> float:
    """Recover ``max_simulation_time`` (seconds) from a conditioning vector."""
    raw = raw_from_conditioning(conditioning)
    return float(raw.get("max_simulation_time", DEFAULT_MAX_SIMULATION_TIME))


def reconstruct_schedule(
    network_output: np.ndarray, channels: np.ndarray, max_time: float
) -> np.ndarray:
    """Turn a ``UNet3D.forward`` output into the ``SurrogateModel`` schedule.

    Args:
        network_output: ``(3, R, R, R)`` physical seconds ``[start, end, local]``
            as returned by ``UNet3D.forward`` (or an exported graph of it).
        channels: ``(7, R, R, R)`` input channels; ``channels[0] > 0.5`` marks
            active cells.
        max_time: ``max_simulation_time`` for the run (seconds).

    Returns:
        ``(3, R, R, R)`` float32 ``[start, end, local]`` with
        ``end == start + local`` on active cells (bounded by ``max_time`` and
        floored at ``END_TIME_FLOOR``) and 0 on inactive cells.
    """
    pred = np.asarray(network_output, dtype=np.float32)
    if pred.shape[0] != 3 or pred.ndim != 4:
        raise ValueError(f"Expected (3, R, R, R) network output, got {pred.shape}")
    active = np.asarray(channels)[0] > 0.5
    max_time = float(max_time)

    start = np.clip(pred[0], 0.0, max_time)
    local = np.maximum(pred[2], 0.0)
    end = np.minimum(start + local, max_time)
    end = np.maximum(end, END_TIME_FLOOR)
    local = np.maximum(end - start, 0.0)

    out = np.stack([start, end, local]).astype(np.float32)
    out[:, ~active] = 0.0
    return out


def build_unet_from_artifact(model_dir: str, manifest: ModelManifest) -> UNet3D:
    """Build a ``UNet3D`` from ``weights.pt`` + manifests in ``model_dir``.

    The deep manifest's ``target_mean`` / ``target_std`` are authoritative
    for denormalization and are copied onto the model buffers after the
    weights load.  Shared by inference and export.
    """
    import torch

    weights_path = os.path.join(model_dir, "weights.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing weights.pt in {model_dir}")
    deep = read_deep_manifest(model_dir)
    cond_size = int(getattr(deep, "conditioning_size", 0) or DEFAULT_CONDITIONING_SIZE)
    base_channels = int(getattr(deep, "base_channels", 0) or DEFAULT_BASE_CHANNELS)
    resolution = int(getattr(manifest, "grid_resolution", 32) or 32)

    model = UNet3D(
        conditioning_size=cond_size,
        base_channels=base_channels,
        levels=levels_for_resolution(resolution),
    )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    if deep is not None and deep.target_mean and deep.target_std:
        model.set_target_normalization(deep.target_mean, deep.target_std)
    model.eval()
    return model


class DeepSurrogateModel(SurrogateModel):
    """A trained 3D U-Net exposed through the :class:`SurrogateModel` ABC."""

    def __init__(
        self,
        manifest: ModelManifest,
        model_dir: str,
        backend: Optional[str] = None,
        backend_state: Optional[Dict[str, Any]] = None,
    ):
        import torch

        if manifest.kind != "torch_unet":
            raise ValueError(f"DeepSurrogateModel requires kind='torch_unet', got {manifest.kind}")
        self.manifest = manifest
        self.model_dir = os.path.abspath(model_dir)

        self._torch = torch
        self.model = build_unet_from_artifact(self.model_dir, manifest)

        from core.deep_model.backend import detect_torch_backend
        from core.deep_model.onnx_runner import find_onnx_artifact

        chosen = (backend or "auto").lower()
        self.requested_backend = chosen
        if chosen == "mps":
            self._torch_backend = DeepBackend.MPS
        elif chosen == "cpu":
            self._torch_backend = DeepBackend.CPU
        else:
            self._torch_backend = detect_torch_backend()

        self.mlpackage_path = None
        self.onnx_path: Optional[str] = None
        self._coreml_model: Optional[Any] = None
        self._onnx: Optional[Any] = None
        #: Set until the first prediction checks the ONNX graph against PyTorch.
        self._onnx_check_pending = False
        #: ``auto`` only: also time ONNX Runtime against PyTorch and keep the faster.
        self._benchmark_pending = False
        #: What the ONNX check and benchmark found; shown by ``describe``.
        self.selection: Dict[str, Any] = {}

        if chosen in ("auto", "coreml"):
            self.mlpackage_path = self._find_mlpackage()
        if self.mlpackage_path is None and chosen in ("auto", "onnx"):
            self.onnx_path = find_onnx_artifact(self.model_dir)

        if self.mlpackage_path is not None:
            self._backend = DeepBackend.COREML
        elif self.onnx_path is not None:
            # Explicit "onnx" starts on ONNX; "auto" starts on PyTorch and
            # switches only if ONNX Runtime matches it and is faster.
            self._backend = DeepBackend.ONNX if chosen == "onnx" else self._torch_backend
            self._onnx_check_pending = True
            self._benchmark_pending = chosen == "auto"
        else:
            if chosen == "onnx":
                self.selection = {"onnx_rejected": "no exported SolidusPro_Deep.onnx found"}
            self._backend = self._torch_backend
        self._backend_state = backend_state or {}
        self._device = torch.device(torch_device_for(self._torch_backend))
        self._model_device: Optional[Any] = None

    # -- helpers -------------------------------------------------------------
    def _find_mlpackage(self) -> Optional[str]:
        # export_for_target defaults to <model_dir>/export; the desktop settings
        # default (models/deep_model/export) is the sibling of the trained dir.
        candidates = []
        for base in (
            self.model_dir,
            os.path.join(self.model_dir, "export"),
            os.path.join(self.model_dir, "..", "export"),
        ):
            base = os.path.abspath(base)
            if os.path.isdir(base):
                for name in os.listdir(base):
                    if name.endswith(".mlpackage"):
                        candidates.append(os.path.join(base, name))
        return candidates[0] if candidates else None

    def _coreml(self):
        """Load the Core ML model once and reuse it across predict calls."""
        if self._coreml_model is None:
            try:
                import coremltools as ct
            except ImportError as exc:
                raise RuntimeError("coremltools not installed for Core ML inference") from exc
            self._coreml_model = ct.models.MLModel(self.mlpackage_path)
        return self._coreml_model

    def _predict_coreml(self, channels: np.ndarray, conditioning: Conditioning) -> np.ndarray:
        cond_vec = np.asarray(conditioning.vector, dtype=np.float32).reshape(1, -1)
        out = self._coreml().predict(
            {"channels": channels.astype(np.float32)[None], "conditioning": cond_vec}
        )
        return np.asarray(out["schedule"], dtype=np.float32)[0]

    # -- ONNX Runtime ---------------------------------------------------------
    def _reject_onnx(self, reason: str) -> None:
        """Stay on (or return to) PyTorch and record why ONNX Runtime was not used."""
        _LOG.warning("Deep model: not using ONNX Runtime: %s", reason)
        self._backend = self._torch_backend
        self._onnx = None
        self.selection = {"chosen": self._backend.value, "onnx_rejected": reason}
        self._backend_state.update(self.selection)

    def _settle_onnx(self, channels: np.ndarray, conditioning: Conditioning) -> np.ndarray:
        """Check the ONNX graph against PyTorch and pick the backend.

        Runs once, on the first prediction, because the check needs a real
        input.  ONNX Runtime is used only when its output matches PyTorch on the
        active cells; in ``auto`` mode it must also be faster.  Returns the raw
        network output from the backend that was kept.
        """
        from core.deep_model.onnx_runner import OnnxRunner, median_seconds

        self._onnx_check_pending = False
        cond_vec = np.asarray(conditioning.vector, dtype=np.float32)
        try:
            runner = OnnxRunner(self.onnx_path)
        except Exception as exc:
            self._reject_onnx(f"ONNX Runtime could not open {self.onnx_path}: {exc}")
            return self._predict_torch(channels, conditioning)
        if not runner.accepts(channels):
            self._reject_onnx(
                f"ONNX graph expects channels {runner._channels_shape[1:]}, got {list(channels.shape)}"
            )
            return self._predict_torch(channels, conditioning)
        try:
            onnx_raw = runner.run(channels, cond_vec)
        except Exception as exc:
            self._reject_onnx(f"ONNX Runtime failed on {runner.provider}: {exc}")
            return self._predict_torch(channels, conditioning)

        torch_raw = self._predict_torch(channels, conditioning)
        max_time = max_simulation_time_from(conditioning)
        onnx_schedule = reconstruct_schedule(onnx_raw, channels, max_time)
        torch_schedule = reconstruct_schedule(torch_raw, channels, max_time)
        active = np.asarray(channels)[0] > 0.5
        max_diff = float(np.max(np.abs(onnx_schedule[:, active] - torch_schedule[:, active]), initial=0.0))
        if not np.allclose(
            onnx_schedule[:, active], torch_schedule[:, active],
            rtol=ONNX_MATCH_RTOL, atol=ONNX_MATCH_ATOL_S,
        ):
            self._reject_onnx(
                f"{runner.device_name} output differs from PyTorch by up to {max_diff:.2f} s"
            )
            return torch_raw

        self.selection = {
            "onnx_provider": runner.provider,
            "onnx_device": runner.device_name,
            "onnx_max_abs_diff_s": round(max_diff, 4),
            "onnx_skipped_providers": list(runner.skipped),
        }
        keep_onnx = True
        if self._benchmark_pending:
            self._benchmark_pending = False
            onnx_s = median_seconds(lambda: runner.run(channels, cond_vec), BENCHMARK_RUNS)
            torch_s = median_seconds(lambda: self._predict_torch(channels, conditioning), BENCHMARK_RUNS)
            self.selection.update(
                onnx_ms=round(onnx_s * 1000.0, 2),
                torch_ms=round(torch_s * 1000.0, 2),
                torch_device=self._torch_backend.value,
            )
            keep_onnx = onnx_s < torch_s
        if keep_onnx:
            self._backend = DeepBackend.ONNX
            self._onnx = runner
        else:
            self._backend = self._torch_backend
            self._onnx = None
        self.selection["chosen"] = self._backend.value
        self._backend_state.update(self.selection)
        _LOG.info("Deep model backend: %s", self.device_summary)
        return onnx_raw if keep_onnx else torch_raw

    @property
    def device_summary(self) -> str:
        """One line naming the hardware predictions run on."""
        torch_name = "PyTorch MPS (Apple GPU)" if self._torch_backend == DeepBackend.MPS else "PyTorch CPU"
        if self._backend == DeepBackend.COREML:
            return "Core ML (Apple Neural Engine / GPU / CPU)"
        if self._onnx_check_pending:
            if self._benchmark_pending:
                return f"ONNX Runtime or {torch_name}, chosen on the first prediction"
            return "ONNX Runtime, checked against PyTorch on the first prediction"
        if self._backend == DeepBackend.ONNX:
            provider = self.selection.get("onnx_provider", "")
            return f"ONNX Runtime on {onnx_provider_description(provider)}"
        return torch_name

    # -- SurrogateModel ABC --------------------------------------------------
    def predict(self, channels: np.ndarray, conditioning: Conditioning) -> np.ndarray:
        if channels.ndim != 4:
            raise ValueError(f"Expected 4D channels, got shape {channels.shape}")
        if self._onnx_check_pending:
            pred = self._settle_onnx(channels, conditioning)
        elif self._backend == DeepBackend.COREML:
            pred = self._predict_coreml(channels, conditioning)
        elif self._backend == DeepBackend.ONNX and self._onnx is not None and self._onnx.accepts(channels):
            pred = self._onnx.run(channels, np.asarray(conditioning.vector, dtype=np.float32))
        else:
            pred = self._predict_torch(channels, conditioning)
        return reconstruct_schedule(pred, channels, max_simulation_time_from(conditioning))

    def _predict_torch(self, channels: np.ndarray, conditioning: Conditioning) -> np.ndarray:
        torch = self._torch
        if self._model_device is None:
            self.model.to(self._device)
            self._model_device = self._device
        with torch.no_grad():
            ch = torch.from_numpy(np.asarray(channels, dtype=np.float32)).unsqueeze(0)
            cond = torch.from_numpy(np.asarray(conditioning.vector, dtype=np.float32)).unsqueeze(0)
            pred = self.model(ch.to(self._device), cond.to(self._device))[0].cpu().numpy()
        return pred

    def close(self) -> None:
        self._coreml_model = None
        self._onnx = None

    def describe(self) -> str:
        lines = super().describe() + f"\n  Deep device: {self.device_summary}"
        if self.selection.get("onnx_rejected"):
            lines += f"\n  ONNX Runtime not used: {self.selection['onnx_rejected']}"
        if "onnx_ms" in self.selection:
            lines += (
                f"\n  Timed: ONNX Runtime {self.selection['onnx_ms']} ms, "
                f"PyTorch {self.selection['torch_ms']} ms"
            )
        return lines


def load_deep_model(path: str, backend: Optional[str] = None) -> ModelLoadResult:
    """Load a ``torch_unet`` artifact directory into a :class:`DeepSurrogateModel`.

    Args:
        path: Directory containing ``manifest.json`` + ``weights.pt``.
        backend: Optional override of the inference backend: ``"auto"``,
            ``"coreml"``, ``"onnx"``, ``"mps"`` or ``"cpu"``.  ``None`` uses
            ``settings.deep.inference_backend``.

    Returns:
        :class:`ModelLoadResult` with the model or an error message.
    """
    from core.surrogate.model import _read_manifest

    path = os.fspath(path)
    manifest = _read_manifest(path)
    if manifest is None:
        return ModelLoadResult(error=f"No manifest.json found at {path}")
    if manifest.kind != "torch_unet":
        return ModelLoadResult(
            error=f"Model kind '{manifest.kind}' is not a deep model (expected torch_unet)"
        )
    if backend is None:
        try:
            from config.settings import get_settings

            backend = str(get_settings().deep.inference_backend or "auto")
        except Exception:
            backend = "auto"
    try:
        model = DeepSurrogateModel(manifest, path, backend=backend)
        return ModelLoadResult(model=model)
    except Exception as exc:
        return ModelLoadResult(error=f"Failed to load deep model: {exc}")
