#!/usr/bin/env python3
"""
Deep surrogate vs full-solver benchmark (standalone verification script).

Trains (or loads) the deep 3D U-Net surrogate and compares its schedule
predictions against the full solver on held-out geometries (boxes, plates,
bars).  Reports per-field RMSE / MAE / relative error for start / end /
local times.

Usage:
    python3 tests/verify_deep_benchmarks.py                 # load models/deep_model/trained
    python3 tests/verify_deep_benchmarks.py --train --grid 16 --epochs 20
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import tempfile

import numpy as np
import trimesh

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    import torch  # noqa: F401
except ImportError:
    print("torch not installed — run: venv/bin/pip install torch")
    sys.exit(2)

from core import surrogate  # noqa: E402
from core.materials import get_alloy  # noqa: E402
from core.simulation_pipeline import PipelineInputs, run_pipeline  # noqa: E402
from core.surrogate import run_surrogate_pipeline  # noqa: E402
from core.surrogate.dataset import DEFAULT_DATASET_DIR  # noqa: E402
from core.surrogate.metrics import compute_field_metrics  # noqa: E402


def _loader_for(mesh):
    bounds = np.stack([mesh.bounds[0], mesh.bounds[1]], axis=1)

    class _Loader:
        trimesh_obj = mesh
        is_valid = lambda self: True  # noqa: E731
        volume = float(getattr(mesh, "volume", 0.0) or 0.0)

        def get_bounds_array(self):
            return bounds

    return _Loader()


def _inputs(loader, alloy="WCB", voxel_size=0.5):
    return PipelineInputs(
        alloy_name=alloy,
        voxel_size=voxel_size,
        refinement_level=1,
        memory_mode="ram",
        analysis_options={"shrinkage": False, "hotspot": False, "fraction_liquid": False},
        max_voxels_for_visualization=50_000,
        show_progress=False,
        mesh_type="voxel",  # benchmarks surrogates trained on voxel records against the voxel solver
    )


def _geometries():
    return [
        ("box_2", trimesh.creation.box(extents=(2.0, 2.0, 2.0))),
        ("plate_6x4x1", trimesh.creation.box(extents=(6.0, 4.0, 1.0))),
        ("bar_8x2x2", trimesh.creation.box(extents=(8.0, 2.0, 2.0))),
    ]


def _train_fresh_model(grid, epochs, min_records):
    """Train on the captured dataset (fallback: tiny synthetic dataset)."""
    from core.deep_model.training import train_deep_model

    data_dir = DEFAULT_DATASET_DIR
    try:
        from core.surrogate.dataset import load_dataset

        if len(load_dataset(data_dir)) < min_records:
            raise ValueError("captured dataset too small")
    except ValueError:
        tmp = tempfile.mkdtemp()
        data_dir = tmp
        from tests.test_deep_model import _write_synthetic_record

        for seed in range(6):
            _write_synthetic_record(data_dir, seed, resolution=16)
        print(f"Using synthetic dataset (no captured records): {data_dir}")

    out = tempfile.mkdtemp(prefix="deep_bench_")
    summary = train_deep_model(
        data_dir=data_dir,
        output_dir=out,
        grid=grid,
        epochs=epochs,
        batch_size=2,
        learning_rate=1e-3,
        min_records=min_records,
        progress_cb=lambda m: print(f"  {m}"),
    )
    return summary["saved_to"], summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Deep surrogate vs full-solver benchmark.")
    parser.add_argument("--model", default=None,
                        help="Model directory (default: models/deep_model/trained).")
    parser.add_argument("--train", action="store_true",
                        help="Train a fresh model on the captured dataset instead.")
    parser.add_argument("--grid", type=int, default=16, help="Training grid for --train.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs for --train.")
    parser.add_argument("--min-records", type=int, default=4, help="Min records to train.")
    args = parser.parse_args()

    model_dir = args.model
    trained_summary = None
    if args.train or model_dir is None:
        default_dir = os.path.join(_PROJECT_ROOT, "models", "deep_model", "trained")
        if args.train or not os.path.isdir(default_dir):
            model_dir, trained_summary = _train_fresh_model(
                args.grid, args.epochs, args.min_records
            )
        else:
            model_dir = default_dir
            print(f"Loading existing deep model: {model_dir}")

    result = surrogate.load_model(model_dir)
    if result.error or result.model is None:
        print(f"Failed to load deep model: {result.error}")
        return 1
    model = result.model
    print(f"Model: {model.manifest.name} (grid {model.manifest.grid_resolution}^3, "
          f"backend {model.manifest.backend})")

    print("\nRunning full solver + deep surrogate on held-out geometries...")
    print(f"{'geometry':<16} {'field':<10} {'RMSE (s)':>10} {'MAE (s)':>10} "
          f"{'rel RMSE':>10}")
    print("-" * 60)

    rows = []
    for name, mesh in _geometries():
        loader = _loader_for(mesh)
        inputs = _inputs(loader)
        with contextlib.redirect_stdout(io.StringIO()):
            ref = run_pipeline(inputs, stl_loader=loader)
            sur = run_surrogate_pipeline(inputs, model=model, stl_loader=loader)

        ref_end = np.asarray(ref.engine.solidification_times, dtype=np.float64)
        sur_end = np.asarray(sur.engine.solidification_times, dtype=np.float64)
        ref_start = np.asarray(ref.engine.solidification_start_times, dtype=np.float64)
        sur_start = np.asarray(sur.engine.solidification_start_times, dtype=np.float64)
        ref_local = np.asarray(ref.engine.local_solidification_time, dtype=np.float64)
        sur_local = np.asarray(sur.engine.local_solidification_time, dtype=np.float64)

        for field, pred, truth in (
            ("start", sur_start, ref_start),
            ("end", sur_end, ref_end),
            ("local", sur_local, ref_local),
        ):
            m = compute_field_metrics(pred, truth)
            print(f"{name:<16} {field:<10} {m['rmse']:>10.1f} {m['mae']:>10.1f} "
                  f"{m['relative_rmse']:>10.3f}")
            rows.append(m)

    end_row = rows[1]
    print("-" * 60)
    print(f"\nHeadline (end time): RMSE {end_row['rmse']:.1f}s · "
          f"MAE {end_row['mae']:.1f}s · relative RMSE {end_row['relative_rmse']:.3f}")
    if trained_summary:
        h = trained_summary["headline"]
        print(f"Training headline: RMSE {h.get('rmse_s', 0):.1f}s · "
              f"conf {h.get('confidence', 0):.2f}")
    print("\nNOTE: low-data runs underperform until the dataset accumulates; "
          "auto-train refits on every new capture.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
