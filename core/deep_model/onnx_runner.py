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

ONNX_ARTIFACT_NAME = "SolidusPro_Deep.onnx"

def find_onnx_artifact(model_dir: str) -> Optional[str]:
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
        active = session.get_providers()
        if not active or active[0] != name:
            errors.append(f"{name}: not initialized (session runs on {active[:1]})")
            continue
        return session, name, errors
    raise RuntimeError("No ONNX Runtime provider could open the model: " + "; ".join(errors or ["none installed"]))

class OnnxRunner:

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
        expected = self._channels_shape[1:]
        actual = list(np.asarray(channels).shape)
        if len(expected) != len(actual):
            return False
        return all(not isinstance(e, int) or e == a for e, a in zip(expected, actual))

    def run(self, channels: np.ndarray, conditioning_vector: np.ndarray) -> np.ndarray:
        feed = {
            "channels": np.asarray(channels, dtype=self._channels_dtype)[None],
            "conditioning": np.asarray(conditioning_vector, dtype=self._cond_dtype).reshape(1, -1),
        }
        out = self.session.run(["schedule"], feed)[0]
        return np.asarray(out, dtype=np.float32)[0]

def median_seconds(fn, runs: int) -> float:
    times = []
    for _ in range(max(1, runs)):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times))
