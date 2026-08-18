"""Deep-learning inference backend ladder.

Mirrors ``core.surrogate.backend``: best backend first, lazy imports, CPU
fallback.  The ladder for deep inference on this macOS desktop app:

1. ``COREML`` — Apple ANE / GPU / CPU via ``coremltools`` (exported
   ``.mlpackage``).  The only path that can use the Neural Engine.
2. ``TORCH_MPS`` — PyTorch on the Apple GPU (MPS backend).
3. ``TORCH_CPU`` — PyTorch CPU fallback.

Unlike the surrogate ladder, this is *not* wired into the shared
``Backend`` preference order — ``core/surrogate/backend.py`` only gains the
``TORCH`` enum value so ``SurrogateModel.backend`` can construct from a deep
manifest; the deep backend selection lives here.
"""

from __future__ import annotations

import platform
from enum import Enum
from typing import Dict, Tuple


class DeepBackend(Enum):
    COREML = "coreml"
    MPS = "mps"
    CPU = "cpu"


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


def _probe_mps() -> Tuple[bool, str]:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch not installed (pip install torch)"
    try:
        if torch.backends.mps.is_available():
            return True, "PyTorch MPS (Apple GPU)"
        return False, "MPS unavailable (not Apple Silicon)"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"MPS probe failed: {exc}"


def _probe_cpu() -> Tuple[bool, str]:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch not installed (pip install torch)"
    return True, "PyTorch CPU"


_PROBES = {
    DeepBackend.COREML: _probe_coreml,
    DeepBackend.MPS: _probe_mps,
    DeepBackend.CPU: _probe_cpu,
}

_PREFERENCE_ORDER = [DeepBackend.COREML, DeepBackend.MPS, DeepBackend.CPU]


def _detect() -> None:
    """Populate cached detection state once."""
    if _BACKEND_STATE:
        return
    for backend in _PREFERENCE_ORDER:
        ok, desc = _PROBES[backend]()
        _BACKEND_STATE[backend.value] = ok
        _DEVICE_NAME[backend.value] = desc


def get_available_deep_backends() -> list:
    """Return the deep backend names usable on this machine."""
    _detect()
    return [name for name, ok in _BACKEND_STATE.items() if ok]


def detect_deep_backend(preference: str = "auto") -> DeepBackend:
    """Select the best available deep inference backend.

    Args:
        preference: One of ``"auto"``, ``"coreml"``, ``"mps"``, ``"cpu"``.

    Returns:
        The selected :class:`DeepBackend`.  Falls back down the ladder when a
        preferred backend is unavailable.
    """
    _detect()
    if preference in ("auto", ""):
        for backend in _PREFERENCE_ORDER:
            if _BACKEND_STATE.get(backend.value, False):
                return backend
        return DeepBackend.CPU
    try:
        backend = DeepBackend(preference.lower())
    except ValueError:
        return detect_deep_backend("auto")
    if _BACKEND_STATE.get(backend.value, False):
        return backend
    return detect_deep_backend("auto")


def deep_backend_device_name(backend: DeepBackend) -> str:
    """Human-readable device description for a deep backend."""
    _detect()
    return _DEVICE_NAME.get(backend.value, backend.value)


def detect_torch_backend() -> DeepBackend:
    """Torch-only ladder: MPS when available, else CPU.

    Used at model load time — Core ML is only chosen when an exported
    ``.mlpackage`` artifact actually exists next to the weights.
    """
    _detect()
    if _BACKEND_STATE.get(DeepBackend.MPS.value, False):
        return DeepBackend.MPS
    return DeepBackend.CPU


def torch_device_for(backend: DeepBackend) -> str:
    """Map a deep backend to a torch device string."""
    if backend == DeepBackend.MPS:
        return "mps"
    return "cpu"
