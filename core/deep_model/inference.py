"""Deep surrogate inference: ``DeepSurrogateModel`` (``SurrogateModel`` ABC).

The deep model drops into the shared ``SurrogateModel`` interface, so the
existing ``run_surrogate_pipeline``, ``SurrogateWorker``, and UI paths run it
unchanged.  Prediction is backend-aware:

1. Core ML — exported ``.mlpackage`` (Apple ANE / GPU / CPU) when present.
2. PyTorch MPS — Apple GPU.
3. PyTorch CPU — universal fallback.

The schedule is reconstructed so ``end == start + local`` (bounded by
``max_simulation_time``), the same invariant the numpy surrogate enforces.
Without reconstruction, the independently-trained end head drifts from the
start/local heads and inflates the fraction-liquid timeline with outlier end
times far past the true solidification completion.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np

from core.deep_model.backend import DeepBackend, torch_device_for
from core.deep_model.model import UNet3D, levels_for_resolution, read_deep_manifest
from core.surrogate.conditioning import Conditioning, raw_from_conditioning
from core.surrogate.model import ModelLoadResult, ModelManifest, SurrogateModel

END_TIME_FLOOR = 0.1


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
        weights_path = os.path.join(self.model_dir, "weights.pt")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing weights.pt in {self.model_dir}")

        deep = read_deep_manifest(self.model_dir)
        cond_size = int(getattr(deep, "conditioning_size", 0) or 23)
        resolution = int(getattr(manifest, "grid_resolution", 32) or 32)

        self._torch = torch
        self.model = UNet3D(conditioning_size=cond_size, levels=levels_for_resolution(resolution))
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

        chosen = backend or "auto"
        self.mlpackage_path = None
        if chosen == "auto" or chosen == "coreml":
            self.mlpackage_path = self._find_mlpackage()
        if self.mlpackage_path is not None:
            self._backend = DeepBackend.COREML
        elif chosen == "coreml":
            from core.deep_model.backend import detect_torch_backend

            self._backend = detect_torch_backend()
        elif chosen == "auto":
            from core.deep_model.backend import detect_torch_backend

            self._backend = detect_torch_backend()
        elif chosen == "mps":
            self._backend = DeepBackend.MPS
        else:
            self._backend = DeepBackend.CPU
        self._backend_state = backend_state or {}
        self._device = torch.device(torch_device_for(self._backend))
        self._model_device: Optional[Any] = None

    # -- helpers -------------------------------------------------------------
    def _find_mlpackage(self) -> Optional[str]:
        candidates = []
        for base in (self.model_dir, os.path.join(self.model_dir, "..", "export")):
            base = os.path.abspath(base)
            if os.path.isdir(base):
                for name in os.listdir(base):
                    if name.endswith(".mlpackage"):
                        candidates.append(os.path.join(base, name))
        return candidates[0] if candidates else None

    def _predict_coreml(self, channels: np.ndarray, conditioning: Conditioning) -> np.ndarray:
        try:
            import coremltools as ct
        except ImportError as exc:
            raise RuntimeError("coremltools not installed for Core ML inference") from exc
        model = ct.models.MLModel(self.mlpackage_path)
        cond_vec = np.asarray(conditioning.vector, dtype=np.float32)
        out = model.predict({"channels": channels.astype(np.float32), "conditioning": cond_vec})
        pred = np.asarray(out["schedule"], dtype=np.float32)
        return pred

    # -- SurrogateModel ABC --------------------------------------------------
    def predict(self, channels: np.ndarray, conditioning: Conditioning) -> np.ndarray:
        if channels.ndim != 4:
            raise ValueError(f"Expected 4D channels, got shape {channels.shape}")
        if self._backend == DeepBackend.COREML:
            pred = self._predict_coreml(channels, conditioning)
        else:
            pred = self._predict_torch(channels, conditioning)

        raw = raw_from_conditioning(conditioning)
        max_time = float(raw.get("max_simulation_time", 14400.0))
        start = pred[0].astype(np.float32)
        local = pred[2].astype(np.float32)

        active = channels[0] > 0.5
        start = np.where(active, np.clip(start, 0.0, max_time), 0.0)
        local = np.where(active, np.maximum(local, 0.0), 0.0)

        # Reconstruct the schedule so end == start + local (the invariant the
        # numpy surrogate enforces and every fraction-solid computation relies
        # on). The three heads are trained independently; trusting the raw end
        # head lets outliers inflate the fraction-liquid timeline far past the
        # true completion time, so the end head is only used as a lower bound
        # for the completion time via local.
        end = np.minimum(start + local, max_time)
        end = np.maximum(end, END_TIME_FLOOR)
        local = np.maximum(end - start, 0.0)

        start[~active] = 0.0
        end[~active] = 0.0
        local[~active] = 0.0
        return np.stack([start, end, local]).astype(np.float32)

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
        pass

    def describe(self) -> str:
        lines = super().describe()
        return lines + f"\n  Deep device: {self._backend.value}"


def load_deep_model(path: str, backend: Optional[str] = None) -> ModelLoadResult:
    """Load a ``torch_unet`` artifact directory into a :class:`DeepSurrogateModel`.

    Args:
        path: Directory containing ``manifest.json`` + ``weights.pt``.
        backend: Optional override of the inference backend ("auto" default).

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
    try:
        model = DeepSurrogateModel(manifest, path, backend=backend)
        return ModelLoadResult(model=model)
    except Exception as exc:
        return ModelLoadResult(error=f"Failed to load deep model: {exc}")
