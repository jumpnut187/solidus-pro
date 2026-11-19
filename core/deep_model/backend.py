"""Deep-learning inference backend ladder.

Mirrors ``core.surrogate.backend``: best backend first, lazy imports, CPU
fallback.  The ladder for deep inference:

1. ``COREML`` — Apple ANE / GPU / CPU via ``coremltools`` (exported
   ``.mlpackage``).  The only path that can use the Apple Neural Engine.
2. ``ONNX`` — ONNX Runtime on the exported ``.onnx`` graph, through the
   best execution provider installed: Qualcomm Hexagon NPU (QNN), Intel NPU
   (OpenVINO), AMD Ryzen AI NPU (Vitis AI), NVIDIA GPU (TensorRT / CUDA),
   DirectML, then the ONNX Runtime CPU provider.  An NPU is not assumed to be
   faster: in ``auto`` mode the model validates the ONNX output against
   PyTorch and times both on the first prediction (``inference.py``).
3. ``MPS`` — PyTorch on the Apple GPU.
4. ``CPU`` — PyTorch CPU fallback.

Unlike the surrogate ladder, this is *not* wired into the shared
``Backend`` preference order — ``core/surrogate/backend.py`` only gains the
``TORCH`` enum value so ``SurrogateModel.backend`` can construct from a deep
manifest; the deep backend selection lives here.
"""

from __future__ import annotations

import os
import platform
from enum import Enum
from typing import Any, Dict, List, Tuple


class DeepBackend(Enum):
    COREML = "coreml"
    ONNX = "onnx"
    MPS = "mps"
    CPU = "cpu"


#: ONNX Runtime execution providers tried for the deep model, in order, with
#: the hardware each one reaches.  NPUs first, then GPUs.  The CPU provider is
#: always appended last as the in-session fallback for operators an
#: accelerator does not support (3D convolutions are the usual ones).
#: ``CoreMLExecutionProvider`` is left out on purpose: on a Mac the native
#: ``.mlpackage`` path (``COREML``) is used instead.
ONNX_PROVIDER_PREFERENCE: Tuple[Tuple[str, str], ...] = (
    ("QNNExecutionProvider", "Qualcomm Hexagon NPU (QNN)"),
    ("OpenVINOExecutionProvider", "Intel NPU (OpenVINO)"),
    ("VitisAIExecutionProvider", "AMD Ryzen AI NPU (Vitis AI)"),
    ("TensorrtExecutionProvider", "NVIDIA GPU (TensorRT)"),
    ("CUDAExecutionProvider", "NVIDIA GPU (CUDA)"),
    ("DmlExecutionProvider", "DirectML GPU/NPU"),
)
ONNX_CPU_PROVIDER = "CPUExecutionProvider"
_ONNX_PROVIDER_NAMES = dict(ONNX_PROVIDER_PREFERENCE)
_ONNX_PROVIDER_NAMES[ONNX_CPU_PROVIDER] = "ONNX Runtime CPU"

#: Environment override for the provider order, comma separated, e.g.
#: ``CASTSIM_DEEP_ONNX_PROVIDERS=OpenVINOExecutionProvider``.  Takes priority
#: over ``settings.deep.onnx_providers``.
ONNX_PROVIDERS_ENV = "CASTSIM_DEEP_ONNX_PROVIDERS"


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


def _probe_onnx() -> Tuple[bool, str]:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False, "onnxruntime not installed (pip install onnxruntime)"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"onnxruntime failed to import: {exc}"
    names = [onnx_provider_description(p) for p in available_onnx_providers()]
    return True, "ONNX Runtime (" + ", ".join(names) + ")"


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
    DeepBackend.ONNX: _probe_onnx,
    DeepBackend.MPS: _probe_mps,
    DeepBackend.CPU: _probe_cpu,
}

_PREFERENCE_ORDER = [DeepBackend.COREML, DeepBackend.ONNX, DeepBackend.MPS, DeepBackend.CPU]


# ---------------------------------------------------------------------------
# ONNX Runtime providers
# ---------------------------------------------------------------------------

def _configured_provider_order() -> List[str]:
    """The provider order requested by env or settings; empty for the default."""
    raw = os.environ.get(ONNX_PROVIDERS_ENV, "").strip()
    if not raw:
        try:
            from config.settings import get_settings

            raw = str(getattr(get_settings().deep, "onnx_providers", "") or "").strip()
        except Exception:
            raw = ""
    return [name.strip() for name in raw.split(",") if name.strip()]


def available_onnx_providers() -> List[str]:
    """ONNX Runtime providers to try for the deep model, best first.

    Only providers the installed ``onnxruntime`` build registers are returned.
    The order is :data:`ONNX_PROVIDER_PREFERENCE` unless overridden by
    ``CASTSIM_DEEP_ONNX_PROVIDERS`` or ``settings.deep.onnx_providers``.  The
    CPU provider is always last.  Empty when onnxruntime is not installed.
    """
    try:
        import onnxruntime as ort
    except Exception:
        return []
    installed = set(ort.get_available_providers())
    requested = _configured_provider_order() or [name for name, _ in ONNX_PROVIDER_PREFERENCE]
    order = [name for name in requested if name in installed and name != ONNX_CPU_PROVIDER]
    if ONNX_CPU_PROVIDER in installed:
        order.append(ONNX_CPU_PROVIDER)
    return order


def accelerated_onnx_providers() -> List[str]:
    """The NPU/GPU providers from :func:`available_onnx_providers`."""
    return [name for name in available_onnx_providers() if name != ONNX_CPU_PROVIDER]


def onnx_provider_description(provider: str) -> str:
    """Human-readable hardware name for an ONNX Runtime provider."""
    return _ONNX_PROVIDER_NAMES.get(provider, provider)


def onnx_provider_options(provider: str) -> Dict[str, Any]:
    """Provider options that point each provider at its NPU.

    - QNN: the HTP (Hexagon NPU) backend, running the fp32 graph at fp16
      precision, which is what HTP executes natively.
    - OpenVINO: the NPU device.  Session creation fails on machines without
      one, and the next provider is tried.
    """
    if provider == "QNNExecutionProvider":
        return {"backend_type": "htp", "enable_htp_fp16_precision": "1"}
    if provider == "OpenVINOExecutionProvider":
        return {"device_type": "NPU"}
    return {}


def onnx_session_options(ort: Any, provider: str) -> Any:
    """Session options for ``provider``.

    DirectML does not support memory patterns or parallel execution; ONNX
    Runtime returns an error unless both are off.
    """
    options = ort.SessionOptions()
    if provider == "DmlExecutionProvider":
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return options


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
        preference: One of ``"auto"``, ``"coreml"``, ``"onnx"``, ``"mps"``,
            ``"cpu"``.

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
