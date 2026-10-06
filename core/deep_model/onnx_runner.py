"""ONNX Runtime inference for the exported deep model.

Runs ``SolidusPro_Deep.onnx`` (written by ``export.py``) through the best
ONNX Runtime execution provider on this machine: an NPU (Qualcomm QNN, Intel
OpenVINO, AMD Vitis AI), a GPU (TensorRT, CUDA, DirectML), or the ONNX
Runtime CPU provider.  Provider order and per-provider options live in
``backend.py``.

ONNX Runtime keeps the CPU provider in every session, so an operator the
accelerator cannot run is executed on the CPU instead of failing.  For the 3D
U-Net that can mean most of the graph, which is why ``inference.py`` times
this runner against PyTorch before choosing it in ``auto`` mode.

``onnxruntime`` is imported lazily: the app must start without it.
"""

from __future__ import annotations

import os
import time
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from core.deep_model.backend import (
    ONNX_CPU_PROVIDER,
    available_onnx_providers,
    onnx_provider_description,
    onnx_provider_options,
    onnx_session_options,
)

#: File name written by ``export.py`` for the fp32 graph.  The fp32 graph is
#: the one loaded: QNN runs it at fp16 on the NPU (``enable_htp_fp16_precision``)
#: and every other provider accepts fp32 inputs.
ONNX_ARTIFACT_NAME = "SolidusPro_Deep.onnx"


def find_onnx_artifact(model_dir: str) -> Optional[str]:
    """Locate the exported fp32 ONNX graph for a trained model directory.

    Searches the same places as the Core ML lookup: the model directory, its
    ``export`` subdirectory, and the sibling ``export`` directory used by the
    desktop settings default.
    """
    for base in (
        model_dir,
        os.path.join(model_dir, "export"),
        os.path.join(model_dir, "..", "export"),
    ):
        candidate = os.path.abspath(os.path.join(base, ONNX_ARTIFACT_NAME))
        if os.path.isfile(candidate):
            return candidate
    return None


def create_session(path: str, providers: Optional[Sequence[str]] = None) -> Tuple[Any, str, List[str]]:
    """Open an ONNX Runtime session on the first provider that accepts the model.

    Args:
        path: The ``.onnx`` file.
        providers: Provider names to try in order.  Defaults to
            :func:`backend.available_onnx_providers`.

    Returns:
        ``(session, provider, errors)``: the session, the provider it runs on,
        and why each earlier provider was skipped.

    Raises:
        RuntimeError: No provider could open the model.
    """
    import onnxruntime as ort

    order = list(providers) if providers else available_onnx_providers()
    errors: List[str] = []
    for name in order:
        try:
            chain: List[Any] = [(name, onnx_provider_options(name))]
            if name != ONNX_CPU_PROVIDER:
                chain.append(ONNX_CPU_PROVIDER)
            session = ort.InferenceSession(
                path, sess_options=onnx_session_options(ort, name), providers=chain
            )
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue
        # ONNX Runtime can drop a provider that fails to initialize and carry
        # on with the CPU provider; that is not the session that was asked for.
        active = session.get_providers()
        if not active or active[0] != name:
            errors.append(f"{name}: not initialized (session runs on {active[:1]})")
            continue
        return session, name, errors
    raise RuntimeError("No ONNX Runtime provider could open the model: " + "; ".join(errors or ["none installed"]))


class OnnxRunner:
    """One ONNX Runtime session for the exported ``UNet3D.forward`` graph.

    Inputs are ``channels`` ``(1, 7, R, R, R)`` and ``conditioning``
    ``(1, C)``; the output ``schedule`` is ``(1, 3, R, R, R)`` physical seconds,
    the same contract as the PyTorch model and the Core ML package.
    """

    def __init__(self, path: str, providers: Optional[Sequence[str]] = None):
        self.path = path
        self.session, self.provider, self.skipped = create_session(path, providers)
        inputs = {item.name: item for item in self.session.get_inputs()}
        if "channels" not in inputs or "conditioning" not in inputs:
            raise ValueError(f"{path} does not have the deep model inputs (channels, conditioning)")
        self._channels_shape = list(inputs["channels"].shape)
        self._channels_dtype = np.float16 if "float16" in str(inputs["channels"].type) else np.float32
        self._cond_dtype = np.float16 if "float16" in str(inputs["conditioning"].type) else np.float32

    @property
    def device_name(self) -> str:
        return onnx_provider_description(self.provider)

    def accepts(self, channels: np.ndarray) -> bool:
        """Whether ``channels`` ``(7, R, R, R)`` fits the graph's fixed input shape."""
        expected = self._channels_shape[1:]
        actual = list(np.asarray(channels).shape)
        if len(expected) != len(actual):
            return False
        return all(not isinstance(e, int) or e == a for e, a in zip(expected, actual))

    def run(self, channels: np.ndarray, conditioning_vector: np.ndarray) -> np.ndarray:
        """Run the graph and return ``(3, R, R, R)`` float32 seconds."""
        feed = {
            "channels": np.asarray(channels, dtype=self._channels_dtype)[None],
            "conditioning": np.asarray(conditioning_vector, dtype=self._cond_dtype).reshape(1, -1),
        }
        out = self.session.run(["schedule"], feed)[0]
        return np.asarray(out, dtype=np.float32)[0]


def median_seconds(fn, runs: int) -> float:
    """Median wall time of ``runs`` calls to ``fn``."""
    times = []
    for _ in range(max(1, runs)):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times))
