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

ONNX_PROVIDERS_ENV = "CASTSIM_DEEP_ONNX_PROVIDERS"

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

def _probe_onnx() -> Tuple[bool, str]:
    try:
        import onnxruntime
    except ImportError:
        return False, "onnxruntime not installed (pip install onnxruntime)"
    except Exception as exc:
        return False, f"onnxruntime failed to import: {exc}"
    names = [onnx_provider_description(p) for p in available_onnx_providers()]
    return True, "ONNX Runtime (" + ", ".join(names) + ")"

def _probe_mps() -> Tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch not installed (pip install torch)"
    try:
        if torch.backends.mps.is_available():
            return True, "PyTorch MPS (Apple GPU)"
        return False, "MPS unavailable (not Apple Silicon)"
    except Exception as exc:
        return False, f"MPS probe failed: {exc}"

def _probe_cpu() -> Tuple[bool, str]:
    try:
        import torch
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

def _configured_provider_order() -> List[str]:
    raw = os.environ.get(ONNX_PROVIDERS_ENV, "").strip()
    if not raw:
        try:
            from config.settings import get_settings

            raw = str(getattr(get_settings().deep, "onnx_providers", "") or "").strip()
        except Exception:
            raw = ""
    return [name.strip() for name in raw.split(",") if name.strip()]

def available_onnx_providers() -> List[str]:
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
    return [name for name in available_onnx_providers() if name != ONNX_CPU_PROVIDER]

def onnx_provider_description(provider: str) -> str:
    return _ONNX_PROVIDER_NAMES.get(provider, provider)

def onnx_provider_options(provider: str) -> Dict[str, Any]:
    if provider == "QNNExecutionProvider":
        return {"backend_type": "htp", "enable_htp_fp16_precision": "1"}
    if provider == "OpenVINOExecutionProvider":
        return {"device_type": "NPU"}
    return {}

def onnx_session_options(ort: Any, provider: str) -> Any:
    options = ort.SessionOptions()
    if provider == "DmlExecutionProvider":
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return options

def _detect() -> None:
    if _BACKEND_STATE:
        return
    for backend in _PREFERENCE_ORDER:
        ok, desc = _PROBES[backend]()
        _BACKEND_STATE[backend.value] = ok
        _DEVICE_NAME[backend.value] = desc

def get_available_deep_backends() -> list:
    _detect()
    return [name for name, ok in _BACKEND_STATE.items() if ok]

def detect_deep_backend(preference: str = "auto") -> DeepBackend:
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
    _detect()
    return _DEVICE_NAME.get(backend.value, backend.value)

def detect_torch_backend() -> DeepBackend:
    _detect()
    if _BACKEND_STATE.get(DeepBackend.MPS.value, False):
        return DeepBackend.MPS
    return DeepBackend.CPU

def torch_device_for(backend: DeepBackend) -> str:
    if backend == DeepBackend.MPS:
        return "mps"
    return "cpu"
