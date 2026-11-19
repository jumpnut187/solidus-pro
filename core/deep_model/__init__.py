"""Deep learning solidification surrogate (``core/deep_model``).

A PyTorch 3D U-Net (with FiLM conditioning) that predicts the solidification
schedule fields (start / end / local times) from geometry channels plus
alloy/config conditioning.  Separate from the lightweight NumPy linear-log
model in ``core/surrogate``; reuses its data capture, field encoding,
conditioning vectors, and pipeline interface so both modules coexist.

Public API (all torch imports are lazy — the app must start without torch):

* ``train_deep_model`` — full training run from captured records.
* ``load_deep_model`` — build a :class:`DeepSurrogateModel` from an artifact.
* ``export_for_target`` — export one trained model to coreml / onnx / tflite.
* ``detect_deep_backend`` — inference backend ladder
  (Core ML -> ONNX Runtime -> MPS -> CPU).
* ``DeepModelManifest`` — deep artifact metadata (targets, quantization).
"""

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
    """Fail with a clear message when torch is not installed."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PyTorch is not installed. Install the deep-learning extras with "
            "`pip install -r requirements.txt` (DEEP LEARNING MODULE section)."
        ) from exc


def train_deep_model(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Train the 3D U-Net on captured records (lazy import)."""
    _require_torch()
    from core.deep_model.training import train_deep_model as _train

    return _train(*args, **kwargs)


def load_deep_model(path: str, backend: Optional[str] = None):
    """Load a trained deep model artifact (lazy import)."""
    _require_torch()
    from core.deep_model.inference import load_deep_model as _load

    return _load(path, backend=backend)


def export_for_target(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Export a trained model to an NPU target (lazy import)."""
    _require_torch()
    from core.deep_model.export import export_for_target as _export

    return _export(*args, **kwargs)


def DeepModelManifest(*args: Any, **kwargs: Any):
    """Lazy wrapper around the deep manifest dataclass."""
    from core.deep_model.model import DeepModelManifest as _Manifest

    return _Manifest(*args, **kwargs)
