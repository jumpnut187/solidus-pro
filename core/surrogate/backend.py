"""Surrogate inference backends with automatic detection and fallback.

Mirrors ``core.gpu_compute.GPUComputeManager``: try the best backend first,
lazy-import optional libraries, and always fall back to the pure-NumPy path.

Priority ladder on Apple Silicon:

1. ``COREML`` — Apple Neural Engine / GPU / CPU via ``coremltools``.  This is
   the only path that can actually use the Neural Engine.
2. ``MLX`` — Apple's array framework (CPU/GPU; not the ANE).
3. ``NUMPY`` — universal reference backend (testable on any machine).

Phase 0 ships the NumPy backend with a bundled reference model; Core ML / MLX
inference wrappers are scaffolded and raise a clear error until their model
artifacts are produced by the trainer.
"""

from __future__ import annotations

import platform
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Backend(Enum):
    NUMPY = "numpy"
    MLX = "mlx"
    COREML = "coreml"
    # Deep-model backend (core/deep_model): manifests of kind="torch_unet"
    # carry backend="torch" so SurrogateModel.backend can construct.  NOT in
    # _PREFERENCE_ORDER — the numpy-surrogate ladder is untouched; deep
    # backend selection lives in core/deep_model/backend.py.
    TORCH = "torch"


_BACKEND_STATE: Dict[str, bool] = {}
_DEVICE_NAME: Dict[str, str] = {}


def _probe_coreml() -> Tuple[bool, str]:
    if platform.system() != "Darwin":
        return False, "Core ML requires macOS"
    try:
        import coremltools  # noqa: F401
    except ImportError:
        return False, "coremltools not installed (pip install coremltools)"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"coremltools failed to import: {exc}"
    return True, "Core ML (ANE/GPU/CPU)"


def _probe_mlx() -> Tuple[bool, str]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False, "MLX requires Apple Silicon"
    try:
        import mlx.core as mx  # noqa: F401
    except ImportError:
        return False, "mlx not installed (pip install mlx)"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"mlx failed to import: {exc}"
    return True, "MLX (GPU/CPU)"


def _probe_numpy() -> Tuple[bool, str]:
    return True, "NumPy reference"


def _probe_torch() -> Tuple[bool, str]:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch not installed (pip install torch)"
    try:
        if torch.backends.mps.is_available():
            return True, "PyTorch (MPS on Apple GPU)"
        return True, "PyTorch (CPU)"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"torch probe failed: {exc}"


_PROBES = {
    Backend.COREML: _probe_coreml,
    Backend.MLX: _probe_mlx,
    Backend.NUMPY: _probe_numpy,
    Backend.TORCH: _probe_torch,
}

_PREFERENCE_ORDER = [Backend.COREML, Backend.MLX, Backend.NUMPY]


def _detect() -> None:
    """Populate cached detection state once."""
    if _BACKEND_STATE:
        return
    for backend in _PREFERENCE_ORDER:
        ok, desc = _PROBES[backend]()
        _BACKEND_STATE[backend.value] = ok
        _DEVICE_NAME[backend.value] = desc


def get_available_backends() -> List[str]:
    """Return the list of backend names that are usable on this machine."""
    _detect()
    return [name for name, ok in _BACKEND_STATE.items() if ok]


def detect_backend(preference: str = "auto") -> Backend:
    """Select the best available backend.

    Args:
        preference: One of ``"auto"``, ``"coreml"``, ``"mlx"``, ``"numpy"``.
            ``"auto"`` walks the priority ladder (Core ML -> MLX -> NumPy).

    Returns:
        The selected :class:`Backend`.
    """
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
    """Human-readable device description for a backend."""
    _detect()
    return _DEVICE_NAME.get(backend.value, backend.value)


def supports_ane(backend: Backend) -> bool:
    """Return True when the backend can target the Apple Neural Engine."""
    return backend == Backend.COREML


def is_coreml_active() -> bool:
    """Return True when a Core ML model could run on this machine."""
    _detect()
    return bool(_BACKEND_STATE.get(Backend.COREML.value, False))