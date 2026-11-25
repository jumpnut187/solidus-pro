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
    masked_schedule_metrics,
    schedule_evaluation_mask,
    schedule_metrics,
    skill_score,
    train_mean_baseline_rmse,
)
from core.surrogate.split import geometry_group_key, grouped_train_val_indices, split_records
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
    "geometry_group_key",
    "get_available_backends",
    "grouped_train_val_indices",
    "masked_schedule_metrics",
    "schedule_evaluation_mask",
    "skill_score",
    "split_records",
    "train_mean_baseline_rmse",
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
