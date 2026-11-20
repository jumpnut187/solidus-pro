from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.deep_model.backend import (
    DeepBackend,
    deep_backend_device_name,
    detect_deep_backend,
)

__all__ = [
    "DeepBackend",
    "DeepModelManifest",
    "deep_backend_device_name",
    "detect_deep_backend",
    "export_for_target",
    "load_deep_model",
    "train_deep_model",
]

def _require_torch() -> None:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is not installed. Install the deep-learning extras with "
            "`pip install -r requirements.txt` (DEEP LEARNING MODULE section)."
        ) from exc

def train_deep_model(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    _require_torch()
    from core.deep_model.training import train_deep_model as _train

    return _train(*args, **kwargs)

def load_deep_model(path: str, backend: Optional[str] = None):
    _require_torch()
    from core.deep_model.inference import load_deep_model as _load

    return _load(path, backend=backend)

def export_for_target(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    _require_torch()
    from core.deep_model.export import export_for_target as _export

    return _export(*args, **kwargs)

def DeepModelManifest(*args: Any, **kwargs: Any):
    from core.deep_model.model import DeepModelManifest as _Manifest

    return _Manifest(*args, **kwargs)
