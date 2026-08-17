"""Surrogate solidification backend.

Public API:

* ``run_surrogate_pipeline`` — run a surrogate simulation end-to-end.
* ``load_model`` / ``save_numpy_demo`` — manage model artifacts.
* ``detect_backend`` / ``Backend`` / ``get_available_backends`` — backend ladder.
* ``capture_run`` — opt-in training-data capture from traditional sims.

Phase 0 ships the NumPy reference model; Core ML (Apple Neural Engine) and
MLX inference are scaffolded behind the same interface and activated once the
trainer produces those artifacts.
"""

from core.surrogate.backend import (
    Backend,
    backend_device_name,
    detect_backend,
    get_available_backends,
    is_coreml_active,
    supports_ane,
)
from core.surrogate.conditioning import (
    Conditioning,
    build_conditioning_vector,
    raw_from_conditioning,
)
from core.surrogate.dataset import (
    DEFAULT_DATASET_DIR,
    capture_run,
    dataset_summary,
    load_dataset,
)
from core.surrogate.fields import CHANNELS, GridMeta, encode_channels, unproject_grid
from core.surrogate.metrics import (
    aggregate_schedule_metrics,
    compute_field_metrics,
    field_confidence,
    format_metrics,
    schedule_metrics,
)
from core.surrogate.model import (
    ModelLoadResult,
    ModelManifest,
    NumpyDemoModel,
    NumpyWeightsModel,
    SurrogateModel,
    load_model,
    save_numpy_demo,
)
from core.surrogate.pipeline import run_surrogate_pipeline

__all__ = [
    "Backend",
    "CHANNELS",
    "Conditioning",
    "DEFAULT_DATASET_DIR",
    "GridMeta",
    "ModelLoadResult",
    "ModelManifest",
    "NumpyDemoModel",
    "NumpyWeightsModel",
    "SurrogateModel",
    "aggregate_schedule_metrics",
    "backend_device_name",
    "build_conditioning_vector",
    "capture_run",
    "compute_field_metrics",
    "dataset_summary",
    "detect_backend",
    "encode_channels",
    "field_confidence",
    "format_metrics",
    "get_available_backends",
    "is_coreml_active",
    "load_dataset",
    "load_model",
    "raw_from_conditioning",
    "run_surrogate_pipeline",
    "save_numpy_demo",
    "schedule_metrics",
    "supports_ane",
    "unproject_grid",
]