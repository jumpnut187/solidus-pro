from __future__ import annotations

import platform
from enum import Enum
from typing import Dict, List, Optional, Tuple

class Backend(Enum):
    NUMPY = "numpy"
    MLX = "mlx"
    COREML = "coreml"
    TORCH = "torch"

_BACKEND_STATE: Dict[str, bool] = {}
_DEVICE_NAME: Dict[str, str] = {}

def _probe_coreml() -> Tuple[bool, str]:
    if platform.system() != "Darwin":
        return False, "Core ML requires macOS"
    try:
        import coremltools
    except ImportError:
        return False, "coremltools not installed (pip install coremltools)"
    except Exception as exc:
        return False, f"coremltools failed to import: {exc}"
    return True, "Core ML (ANE/GPU/CPU)"

def _probe_mlx() -> Tuple[bool, str]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False, "MLX requires Apple Silicon"
    try:
        import mlx.core as mx
    except ImportError:
        return False, "mlx not installed (pip install mlx)"
    except Exception as exc:
        return False, f"mlx failed to import: {exc}"
    return True, "MLX (GPU/CPU)"

def _probe_numpy() -> Tuple[bool, str]:
    return True, "NumPy reference"

def _probe_torch() -> Tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch not installed (pip install torch)"
    try:
        if torch.backends.mps.is_available():
            return True, "PyTorch (MPS on Apple GPU)"
        return True, "PyTorch (CPU)"
    except Exception as exc:
        return False, f"torch probe failed: {exc}"

_PROBES = {
    Backend.COREML: _probe_coreml,
    Backend.MLX: _probe_mlx,
    Backend.NUMPY: _probe_numpy,
    Backend.TORCH: _probe_torch,
}

_PREFERENCE_ORDER = [Backend.COREML, Backend.MLX, Backend.NUMPY]

def _detect() -> None:
    if _BACKEND_STATE:
        return
    for backend in _PREFERENCE_ORDER:
        ok, desc = _PROBES[backend]()
        _BACKEND_STATE[backend.value] = ok
        _DEVICE_NAME[backend.value] = desc

def get_available_backends() -> List[str]:
    _detect()
    return [name for name, ok in _BACKEND_STATE.items() if ok]

def detect_backend(preference: str = "auto") -> Backend:
    _detect()
    if preference == "auto":
        for backend in _PREFERENCE_ORDER:
            if _BACKEND_STATE.get(backend.value, False):
                return backend
        return Backend.NUMPY
    try:
        backend = Backend(preference.lower())
    except ValueError:
        return detect_backend("auto")
    if _BACKEND_STATE.get(backend.value, False):
        return backend
    return detect_backend("auto")

def backend_device_name(backend: Backend) -> str:
    _detect()
    return _DEVICE_NAME.get(backend.value, backend.value)

def supports_ane(backend: Backend) -> bool:
    return backend == Backend.COREML

def is_coreml_active() -> bool:
    _detect()
    return bool(_BACKEND_STATE.get(Backend.COREML.value, False))
