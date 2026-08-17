"""Programmatic training entry points for surrogate models.

Shared by the desktop UI (Surrogate > Retrain Surrogate Model) and the CLI
(``tools/train_surrogate_model.py``).

Two data sources feed training:

* ``generate_dataset`` — run the full solver over synthetic geometries x
  catalog alloys (offline batch), used to seed a dataset from scratch.
* ``retrain_from_dataset`` — fit on records captured from real simulations
  (``surrogate.auto_capture``), optionally generating a starter dataset first.
"""

from __future__ import annotations

import contextlib
import io
import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from core.materials import get_all_alloy_names
from core.simulation_pipeline import PipelineInputs, run_pipeline
from core.surrogate.dataset import DEFAULT_DATASET_DIR, capture_run, load_dataset
from core.surrogate.metrics import aggregate_schedule_metrics, compute_field_metrics
from core.surrogate.model import ModelManifest, NumpyWeightsModel, _save_manifest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_OUTPUT = os.path.join(PROJECT_ROOT, "models", "surrogate", "trained")

ProgressCallback = Optional[Callable[[str], None]]
DetailedProgressCallback = Optional[Callable[[int, int], None]]
CancelCallback = Optional[Callable[[], bool]]


class TrainingCancelledError(RuntimeError):
    """Raised when the user requests cancellation during training."""


def _check_cancel(cancel: CancelCallback) -> None:
    if cancel is not None and cancel():
        raise TrainingCancelledError("Training cancelled")


def _loader_for_trimesh(mesh):
    """Wrap a trimesh object as a minimal stl_loader for the pipeline."""
    bounds = np.stack([mesh.bounds[0], mesh.bounds[1]], axis=1)

    class _Loader:
        trimesh_obj = mesh
        is_valid = lambda self: True  # noqa: E731
        volume = float(getattr(mesh, "volume", 0.0) or 0.0)

        def get_bounds_array(self):
            return bounds

    return _Loader()


def synthetic_geometries():
    """Return a list of (name, trimesh mesh) pairs covering basic shapes."""
    import trimesh

    geoms = []
    geoms.append(("box_2", trimesh.creation.box(extents=(2.0, 2.0, 2.0))))
    geoms.append(("box_4", trimesh.creation.box(extents=(4.0, 4.0, 4.0))))
    geoms.append(("plate_6x4x1", trimesh.creation.box(extents=(6.0, 4.0, 1.0))))
    geoms.append(("bar_8x2x2", trimesh.creation.box(extents=(8.0, 2.0, 2.0))))
    geoms.append(("cylinder", trimesh.creation.cylinder(radius=1.5, height=4.0)))
    return geoms


def generate_dataset(
    alloys: Optional[List[str]] = None,
    out_dir: str = DEFAULT_DATASET_DIR,
    grid: int = 64,
    progress_cb: ProgressCallback = None,
    detailed_cb: DetailedProgressCallback = None,
    cancel_cb: CancelCallback = None,
) -> int:
    """Run the full solver over synthetic geometries x alloys and capture runs.

    Returns:
        Number of records generated.
    """
    if alloys is None or not alloys:
        alloys = get_all_alloy_names()
    os.makedirs(out_dir, exist_ok=True)
    geoms = synthetic_geometries()
    total = len(geoms) * len(alloys)
    generated = 0
    for idx, (name, mesh) in enumerate(geoms):
        voxel_size = round(0.5 if (idx % 2) == 0 else 0.4, 2)
        for alloy in alloys:
            _check_cancel(cancel_cb)
            if progress_cb:
                progress_cb(f"Generating sample: {name} / {alloy} ...")
            loader = _loader_for_trimesh(mesh)
            inputs = PipelineInputs(
                alloy_name=alloy,
                voxel_size=voxel_size,
                refinement_level=1,
                memory_mode="ram",
                analysis_options={"shrinkage": False, "hotspot": False, "fraction_liquid": False},
                show_progress=False,
            )
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    result = run_pipeline(inputs, stl_loader=loader)
                captured = capture_run(
                    result.engine,
                    voxel_size=voxel_size,
                    resolution=grid,
                    out_dir=out_dir,
                )
                if captured:
                    generated += 1
            except TrainingCancelledError:
                raise
            except Exception as exc:
                if progress_cb:
                    progress_cb(f"  Sample {name} / {alloy} skipped: {exc}")
            if detailed_cb:
                detailed_cb(min(generated, total), total)
    if progress_cb:
        progress_cb(f"Generated {generated} sample records into {out_dir}")
    return generated


def train_numpy(records: List[Dict[str, Any]], grid: int) -> Dict[str, Any]:
    """Fit the linear-log surrogate and return its manifest metadata + weights."""
    channels = [r["channels"] for r in records]
    conditioning = [r["conditioning"] for r in records]
    ends = [r["end"] for r in records]
    starts = [r["start"] for r in records]
    locals_ = [r["local"] for r in records]

    weights = NumpyWeightsModel._fit_weights(channels, conditioning, ends)

    split = max(1, int(len(records) * 0.8))
    test_idx = list(range(split, len(records)))

    def _predict(ch, cond):
        dummy = NumpyWeightsModel(
            ModelManifest(name="eval", version="0", kind="numpy_weights", backend="numpy", grid_resolution=grid),
            weights,
        )
        return dummy.predict(ch, cond)

    pred_start = np.concatenate([_predict(channels[i], conditioning[i])[0].reshape(-1) for i in test_idx])
    pred_end = np.concatenate([_predict(channels[i], conditioning[i])[1].reshape(-1) for i in test_idx])
    pred_local = np.concatenate([_predict(channels[i], conditioning[i])[2].reshape(-1) for i in test_idx])
    ref_start = np.concatenate([starts[i].reshape(-1) for i in test_idx])
    ref_end = np.concatenate([ends[i].reshape(-1) for i in test_idx])
    ref_local = np.concatenate([locals_[i].reshape(-1) for i in test_idx])

    metric_dict = {
        "start_time": compute_field_metrics(pred_start, ref_start),
        "end_time": compute_field_metrics(pred_end, ref_end),
        "local_time": compute_field_metrics(pred_local, ref_local),
    }
    headline = aggregate_schedule_metrics(metric_dict)
    return {
        "weights": weights,
        "train_records": len(records) - len(test_idx),
        "test_records": len(test_idx),
        "metrics": metric_dict,
        "headline": headline,
    }


def save_trained(
    weights: np.ndarray,
    headline: Dict[str, Any],
    name: str,
    grid: int,
    alloys: List[str],
    output_dir: str,
    notes: str,
) -> str:
    """Persist the trained weights + manifest."""
    os.makedirs(output_dir, exist_ok=True)
    manifest = ModelManifest(
        name=name,
        version="1.0.0",
        kind="numpy_weights",
        backend="numpy",
        grid_resolution=int(grid),
        supported_analyses=[
            "shrinkage",
            "porosity",
            "hotspot",
            "niyama",
            "sand_burn_in",
            "gas_porosity",
            "fraction_liquid",
            "chvorinov",
        ],
        alloy_coverage=list(alloys),
        accuracy={
            "rmse_s": round(float(headline.get("rmse_s", 0.0)), 2),
            "mae_s": round(float(headline.get("mae_s", 0.0)), 2),
            "max_error_s": round(float(headline.get("max_error_s", 0.0)), 2),
            "confidence": round(float(headline.get("confidence", 0.0)), 3),
        },
        trained_on=time.strftime("%Y-%m-%d"),
        notes=notes,
    )
    _save_manifest(manifest, output_dir)
    np.savez_compressed(os.path.join(output_dir, "weights.npz"), weights=weights)
    return output_dir


def retrain_from_dataset(
    data_dir: str = DEFAULT_DATASET_DIR,
    output_dir: str = DEFAULT_MODEL_OUTPUT,
    grid: int = 64,
    name: str = "Trained Linear Surrogate",
    limit: Optional[int] = None,
    generate_first: bool = False,
    progress_cb: ProgressCallback = None,
    detailed_cb: DetailedProgressCallback = None,
    cancel_cb: CancelCallback = None,
) -> Dict[str, Any]:
    """Fit a surrogate model from captured records and persist it.

    Returns:
        Dict with ``records``, ``train``, ``test``, ``headline``, ``saved_to``,
        and ``alloys``.

    Raises:
        ValueError: When no usable training records are available.
        TrainingCancelledError: When cancellation is requested.
    """
    if generate_first:
        _check_cancel(cancel_cb)
        generate_dataset(
            None,
            data_dir,
            grid=grid,
            progress_cb=progress_cb,
            detailed_cb=detailed_cb,
            cancel_cb=cancel_cb,
        )

    if progress_cb:
        progress_cb("Loading captured training data...")
    records = load_dataset(data_dir, limit=limit)
    if len(records) < 4:
        raise ValueError(
            f"Only {len(records)} usable training record(s) found in {data_dir}. "
            "Run more traditional simulations (with Surrogate training capture "
            "enabled), or enable 'generate starter data' and try again."
        )

    _check_cancel(cancel_cb)
    if progress_cb:
        progress_cb(f"Fitting surrogate model on {len(records)} records...")
    result = train_numpy(records, grid)
    _check_cancel(cancel_cb)

    alloys = sorted({r["alloy_name"] for r in records})
    saved_to = save_trained(
        weights=result["weights"],
        headline=result["headline"],
        name=name,
        grid=grid,
        alloys=alloys,
        output_dir=output_dir,
        notes="Linear-log least-squares surrogate fit on captured runs. "
              "Upgrade path: torch 3D U-Net + FiLM (next phase) for higher fidelity.",
    )
    if progress_cb:
        progress_cb(f"✓ Model saved to {saved_to}")
    return {
        "records": len(records),
        "train": result["train_records"],
        "test": result["test_records"],
        "headline": result["headline"],
        "saved_to": saved_to,
        "alloys": alloys,
    }