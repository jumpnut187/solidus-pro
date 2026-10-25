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
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from core.materials import get_all_alloy_names
from core.simulation_pipeline import PipelineInputs, run_pipeline
from core.surrogate import design_rules
from core.surrogate.dataset import DEFAULT_DATASET_DIR, capture_run, load_dataset
from core.surrogate.metrics import (
    aggregate_schedule_metrics,
    apply_train_baseline,
    masked_schedule_metrics,
    schedule_evaluation_mask,
    train_mean_baseline_rmse,
)
from core.surrogate.model import ModelManifest, NumpyWeightsModel, _save_manifest
from core.surrogate.split import geometry_group_key, grouped_train_val_indices

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
    """Return a list of (name, trimesh mesh) pairs covering basic shapes.

    Delegates to ``core.surrogate.geometries.named_catalog`` (one fixed shape
    per family: box, cylinder, plate with boss, T, L, stepped plate, hollow
    and flanged cylinders, framed box, tapered bar, wedge).
    """
    from core.surrogate.geometries import named_catalog

    return [spec.as_pair() for spec in named_catalog().values()]


#: Stack size for the solver thread in ``generate_dataset``.  SciPy's default
#: median-split ``cKDTree`` (``core/solidification.py`` surface-distance query)
#: recurses past the 8 MB macOS main-thread stack on some voxel clouds and
#: kills the process with SIGSEGV; ``core/voxel_mesh.py:_build_kdtree``
#: documents the same crash.  A 256 MB thread stack was verified to survive
#: every failing synthetic geometry.
SOLVER_THREAD_STACK_BYTES = 256 * 1024 * 1024


def _run_on_large_stack(fn: Callable[[], Any]) -> Any:
    """Run ``fn`` on a worker thread with ``SOLVER_THREAD_STACK_BYTES`` stack."""
    import threading

    result: Dict[str, Any] = {}

    def _target() -> None:
        try:
            result["value"] = fn()
        except BaseException as exc:  # re-raised on the caller thread
            result["error"] = exc

    previous = threading.stack_size()
    threading.stack_size(SOLVER_THREAD_STACK_BYTES)
    try:
        worker = threading.Thread(target=_target, name="surrogate-solver", daemon=True)
        worker.start()
    finally:
        threading.stack_size(previous)
    worker.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


def generation_plan_key(
    geometry_name: str,
    alloy: str,
    mold_constant: float,
    voxel_size: float,
    grid: int,
) -> str:
    """Deterministic key for one (geometry, alloy, mold constant) run.

    Plans may also hold keys ending in ``|mesh=tet`` from removed tet runs,
    or keys without ``|mold=sand`` from runs on the sand boundary before the
    explicit sand mold became the default; they never match and are left
    alone.
    """
    return (
        f"{geometry_name}|{alloy}|mc={float(mold_constant):.1f}|vs={float(voxel_size):.3f}"
        f"|grid={int(grid)}|mold=sand"
    )


def _load_plan(plan_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not plan_path or not os.path.exists(plan_path):
        return {}
    import json

    with open(plan_path) as fh:
        return json.load(fh)


def _save_plan(plan_path: Optional[str], plan: Dict[str, Dict[str, Any]]) -> None:
    if not plan_path:
        return
    import json

    tmp = plan_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(plan, fh, indent=1, sort_keys=True)
    os.replace(tmp, plan_path)


def _plan_record_exists(entry: Dict[str, Any], out_dir: str) -> bool:
    """True when the record a plan entry points at is still on disk (the plan
    stores absolute paths; a moved dataset directory is resolved by basename)."""
    path = str(entry.get("path", ""))
    if not path:
        return False
    return os.path.exists(path) or os.path.exists(os.path.join(out_dir, os.path.basename(path)))


def _part_manager_for(spec: Any, work_dir: str):
    """``PartManager`` holding ``spec.parts``, each written to an STL in ``work_dir`` and loaded in place."""
    from core.part_manager import CODE_TO_PART_TYPE, PartManager

    manager = PartManager()
    for index, (mesh, code) in enumerate(spec.parts):
        path = os.path.join(work_dir, f"part_{index}.stl")
        mesh.export(path)
        part_id = manager.load_part(path, name=f"{CODE_TO_PART_TYPE[int(code)].value}_{index}", preserve_origin=True)
        if part_id is None:
            raise RuntimeError(f"could not load assembly part {index} of {spec.name}")
        manager.get_part(part_id).set_part_type(CODE_TO_PART_TYPE[int(code)])
    return manager


def alloys_for_geometry(name: str, alloys: List[str], per_geometry: Optional[int]) -> List[str]:
    """The alloys one geometry runs with: all of them, or a subset of ``per_geometry`` seeded by its name.

    A subset spreads a fixed solver budget over more shapes; seeding by name
    keeps a resumed batch on the same combinations.
    """
    if per_geometry is None or per_geometry >= len(alloys):
        return list(alloys)
    import hashlib

    seed = int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)
    picks = np.random.default_rng(seed).choice(len(alloys), size=max(1, int(per_geometry)), replace=False)
    return [alloys[i] for i in sorted(picks)]


def _as_geometry_specs(geometries) -> List[Any]:
    """Accept ``GeometrySpec`` objects or ``(name, mesh)`` pairs."""
    from core.surrogate.geometries import GeometrySpec

    specs = []
    for item in geometries:
        if isinstance(item, GeometrySpec):
            specs.append(item)
        else:
            name, mesh = item
            specs.append(
                GeometrySpec(
                    name=str(name),
                    family="custom",
                    mesh=mesh,
                    min_thickness=float(np.min(mesh.extents)),
                )
            )
    return specs


def generate_dataset(
    alloys: Optional[List[str]] = None,
    out_dir: str = DEFAULT_DATASET_DIR,
    grid: int = 64,
    progress_cb: ProgressCallback = None,
    detailed_cb: DetailedProgressCallback = None,
    cancel_cb: CancelCallback = None,
    *,
    geometries=None,
    count: Optional[int] = None,
    seed: int = 0,
    voxel_size: Optional[float] = None,
    target_voxels: int = 20_000,
    max_voxels: int = 50_000,
    dense: bool = False,
    mold_constants: Optional[List[float]] = None,
    plan_path: Optional[str] = None,
    retry_failed: bool = True,
    alloys_per_geometry: Optional[int] = None,
) -> int:
    """Run the full solver over geometries x alloys x mold constants and capture runs.

    Args:
        alloys: Alloy names (default: whole catalog).
        out_dir: Record directory.
        grid: Encoding resolution per axis.
        geometries: ``GeometrySpec`` objects or ``(name, mesh)`` pairs.  When
            ``None``: ``count`` seeded geometries from
            ``core.surrogate.geometries.iter_geometries`` if ``count`` is set,
            otherwise the fixed ``synthetic_geometries`` catalog.
        count / seed: Seeded geometry family parameters (see above).
        voxel_size: Fixed voxel size in inches, or ``None`` to apply
            ``core.surrogate.geometries.voxel_size_for`` (about ``target_voxels``
            solver cells, at least three across the thinnest section).
        mold_constants: ``chvorinov_mold_constant`` values swept per run via
            ``PipelineInputs.solidification_params["timing"]`` (default
            ``[280.0]``).  In the
            thermal_coupled solver this only sets the simulation horizon
            (``max_time``) and the conditioning vector; the reference schedule
            changes only when the horizon truncates it.
        plan_path: Optional JSON sidecar keyed by ``generation_plan_key``.
            Runs recorded there as ``done`` (record file still present) or
            ``invalid`` are skipped, which makes a batch resumable even though
            ``capture_run`` names files from a time hash.
        retry_failed: Re-run combinations the plan marks ``failed``.
        alloys_per_geometry: Run each geometry with this many of ``alloys``
            (``alloys_for_geometry``) instead of all of them.

    A ``GeometrySpec`` with ``parts`` (``core.surrogate.assemblies``) runs as
    a multi-part job: casting, riser, chill and core loaded into a
    ``PartManager`` with their part types. Such a run whose casting freezes
    after its last riser (``design_rules.unfed_metal``) is marked ``invalid``
    in the plan and no record is written.

    Every run executes on a 256 MB-stack thread (``_run_on_large_stack``) and
    inside ``try/except`` so one failure does not stop the batch.

    Returns:
        Number of records generated in this call.
    """
    from core.surrogate.dataset import load_record, validate_record
    from core.surrogate.geometries import iter_geometries, voxel_size_for

    if alloys is None or not alloys:
        alloys = get_all_alloy_names()
    os.makedirs(out_dir, exist_ok=True)
    if geometries is None:
        if count is not None:
            geometries = list(iter_geometries(int(count), seed=seed))
        else:
            # Keep the GeometrySpec (family, min_thickness) so the voxel-size
            # rule's thin-section clause works; (name, mesh) pairs lose it.
            from core.surrogate.geometries import named_catalog

            geometries = list(named_catalog().values())
    specs = _as_geometry_specs(geometries)
    mold_constants = [float(m) for m in (mold_constants or [280.0])]

    plan = _load_plan(plan_path)
    total = sum(len(alloys_for_geometry(spec.name, alloys, alloys_per_geometry)) for spec in specs) * len(mold_constants)
    generated = 0
    done = 0
    for spec in specs:
        run_voxel_size = float(voxel_size) if voxel_size is not None else voxel_size_for(
            spec, grid=grid, target_voxels=target_voxels, max_voxels=max_voxels, dense=dense
        )
        loader = _loader_for_trimesh(spec.mesh) if spec.parts is None else None
        for alloy in alloys_for_geometry(spec.name, alloys, alloys_per_geometry):
            for mold_constant in mold_constants:
                _check_cancel(cancel_cb)
                done += 1
                key = generation_plan_key(spec.name, alloy, mold_constant, run_voxel_size, grid)
                entry = plan.get(key)
                if entry is not None:
                    status = entry.get("status")
                    if status == "done" and _plan_record_exists(entry, out_dir):
                        if progress_cb:
                            progress_cb(f"[{done}/{total}] skip (done): {key}")
                        continue
                    if status == "invalid" or (status == "failed" and not retry_failed):
                        if progress_cb:
                            progress_cb(f"[{done}/{total}] skip ({status}): {key}")
                        continue
                if progress_cb:
                    progress_cb(f"[{done}/{total}] {spec.name} / {alloy} / mc={mold_constant:g} / vs={run_voxel_size:.3f} ...")
                inputs = PipelineInputs(
                    alloy_name=alloy,
                    voxel_size=run_voxel_size,
                    refinement_level=1,
                    memory_mode="ram",
                    analysis_options={"shrinkage": False, "hotspot": False, "fraction_liquid": False},
                    show_progress=False,
                    solidification_params={"timing": {"chvorinov_mold_constant": mold_constant}},
                )
                started = time.time()
                entry = {"geometry": spec.name, "family": spec.family, "alloy": alloy,
                         "mold_constant": mold_constant, "voxel_size": run_voxel_size, "grid": int(grid)}
                try:
                    def _solve():
                        with contextlib.redirect_stdout(io.StringIO()):
                            if spec.parts is None:
                                return run_pipeline(inputs, stl_loader=loader)
                            with tempfile.TemporaryDirectory(prefix="surrogate_parts_") as work_dir:
                                return run_pipeline(inputs, part_manager=_part_manager_for(spec, work_dir))

                    result = _run_on_large_stack(_solve)
                    entry["voxels"] = int(len(result.engine.mesh.voxel_centers))
                    # Train only on sound rigging: a design the rules passed
                    # can still leave metal freezing after its risers.
                    unsound = None if spec.parts is None else design_rules.unfed_metal(
                        result.engine.mesh.voxel_part_types, result.engine.solidification_times
                    )
                    if unsound is not None:
                        entry["status"] = "invalid"
                        entry["error"] = unsound
                        entry["wall_s"] = round(time.time() - started, 2)
                        if progress_cb:
                            progress_cb(f"    rejected, {unsound}")
                    else:
                        captured = capture_run(
                            result.engine,
                            voxel_size=run_voxel_size,
                            resolution=grid,
                            out_dir=out_dir,
                        )
                        entry["wall_s"] = round(time.time() - started, 2)
                        if not captured:
                            raise RuntimeError("capture_run returned None")
                        entry["path"] = captured
                        reason = validate_record(load_record(captured))
                        if reason is None:
                            entry["status"] = "done"
                            generated += 1
                            if progress_cb:
                                progress_cb(f"    ok: {entry['voxels']} voxels, {entry['wall_s']:.1f}s -> {os.path.basename(captured)}")
                        else:
                            entry["status"] = "invalid"
                            entry["error"] = reason
                            if progress_cb:
                                progress_cb(f"    captured but fails validation ({reason}): {os.path.basename(captured)}")
                except TrainingCancelledError:
                    raise
                except Exception as exc:
                    entry["status"] = "failed"
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    entry["wall_s"] = round(time.time() - started, 2)
                    if progress_cb:
                        progress_cb(f"    failed: {entry['error']}")
                plan[key] = entry
                _save_plan(plan_path, plan)
                if detailed_cb:
                    detailed_cb(done, total)
    if progress_cb:
        progress_cb(f"Generated {generated} sample records into {out_dir}")
    return generated


def train_numpy(records: List[Dict[str, Any]], grid: int) -> Dict[str, Any]:
    """Fit the linear-log surrogate and return its manifest metadata + weights.

    Records are split by geometry group (``core.surrogate.split``); both heads
    are fit on the train side only and scored on the validation side over
    active, non-chill cells with a positive reference end time.  When every
    record shares one geometry there is no validation side: ``metrics`` is
    empty, every headline number is NaN and ``val_records`` is 0.

    Returns:
        Dict with ``weights``, ``local_weights``, ``train_records``,
        ``val_records`` (also as ``test_records``), ``train_groups``,
        ``val_groups``, ``val_cells``, ``metrics`` and ``headline``.
    """
    train_idx, val_idx = grouped_train_val_indices(records, val_fraction=0.2, seed=0)
    train_groups = len({geometry_group_key(records[i]) for i in train_idx})
    val_groups = len({geometry_group_key(records[i]) for i in val_idx})

    channels = [records[i]["channels"] for i in train_idx]
    conditioning = [records[i]["conditioning"] for i in train_idx]
    weights = NumpyWeightsModel._fit_weights(
        channels, conditioning, [records[i]["end"] for i in train_idx]
    )
    local_weights = NumpyWeightsModel._fit_local_weights(
        channels, conditioning, [records[i]["local"] for i in train_idx]
    )

    model = NumpyWeightsModel(
        ModelManifest(name="eval", version="0", kind="numpy_weights", backend="numpy", grid_resolution=grid),
        weights,
        local_weights,
    )

    if val_idx:
        preds, starts, ends, locals_, masks = [], [], [], [], []
        for i in val_idx:
            r = records[i]
            preds.append(model.predict(r["channels"], r["conditioning"]).reshape(3, -1))
            starts.append(np.asarray(r["start"]).reshape(-1))
            ends.append(np.asarray(r["end"]).reshape(-1))
            locals_.append(np.asarray(r["local"]).reshape(-1))
            masks.append(schedule_evaluation_mask(r["channels"], r["end"]))
        val_mask = np.concatenate(masks)
        val_end = np.concatenate(ends)
        metric_dict = masked_schedule_metrics(
            np.concatenate(preds, axis=1),
            np.concatenate(starts),
            val_end,
            np.concatenate(locals_),
            val_mask,
        )
        # Baseline: the constant fit on TRAIN end times, scored on val cells.
        train_end = np.concatenate([
            np.asarray(records[i]["end"]).reshape(-1)[
                schedule_evaluation_mask(records[i]["channels"], records[i]["end"])
            ]
            for i in train_idx
        ])
        apply_train_baseline(
            metric_dict, train_mean_baseline_rmse(train_end, val_end[val_mask])
        )
        headline = aggregate_schedule_metrics(metric_dict)
        val_cells = int(metric_dict["end_time"].get("n_compared") or 0)
    else:
        metric_dict = {}
        headline = aggregate_schedule_metrics({})
        val_cells = 0

    return {
        "weights": weights,
        "local_weights": local_weights,
        "train_records": len(train_idx),
        "val_records": len(val_idx),
        "test_records": len(val_idx),
        "train_groups": train_groups,
        "val_groups": val_groups,
        "val_cells": val_cells,
        "metrics": metric_dict,
        "headline": headline,
    }


def _manifest_number(value: Any, digits: int) -> Optional[float]:
    """Round for the manifest; NaN becomes ``None`` so the JSON stays standard."""
    if value is None:
        return None
    f = float(value)
    if not np.isfinite(f):
        return None
    return round(f, digits)


def save_trained(
    weights: np.ndarray,
    headline: Dict[str, Any],
    name: str,
    grid: int,
    alloys: List[str],
    output_dir: str,
    notes: str,
    local_weights: Optional[np.ndarray] = None,
    split_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist the trained weights + manifest.

    ``split_info`` carries record / geometry-group counts from ``train_numpy``
    (``train_records``, ``val_records``, ``train_groups``, ``val_groups``,
    ``val_cells``); they are written into ``accuracy`` alongside the metrics
    so the manifest states what the numbers were measured on.
    """
    os.makedirs(output_dir, exist_ok=True)
    split_info = split_info or {}
    accuracy: Dict[str, Any] = {
        "rmse_s": _manifest_number(headline.get("rmse_s"), 2),
        "mae_s": _manifest_number(headline.get("mae_s"), 2),
        "max_error_s": _manifest_number(headline.get("max_error_s"), 2),
        "end_rmse_s": _manifest_number(headline.get("end_rmse_s"), 2),
        "end_mae_s": _manifest_number(headline.get("end_mae_s"), 2),
        "baseline_rmse_s": _manifest_number(headline.get("baseline_rmse_s"), 2),
        "skill": _manifest_number(headline.get("skill"), 3),
        "confidence": _manifest_number(headline.get("confidence"), 3),
        "evaluation": (
            "geometry-grouped holdout; active non-chill cells with reference end > 0"
        ),
    }
    for key in ("train_records", "val_records", "train_groups", "val_groups", "val_cells"):
        if key in split_info:
            accuracy[key] = int(split_info[key])
    if split_info.get("val_records", None) == 0:
        notes = (
            notes
            + " No validation split: all records share one geometry group, "
            "so accuracy is unmeasured (null)."
        ).strip()
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
        accuracy=accuracy,
        trained_on=time.strftime("%Y-%m-%d"),
        notes=notes,
    )
    _save_manifest(manifest, output_dir)
    arrays: Dict[str, np.ndarray] = {"weights": np.asarray(weights)}
    if local_weights is not None:
        arrays["local_weights"] = np.asarray(local_weights)
    np.savez_compressed(os.path.join(output_dir, "weights.npz"), **arrays)
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
        Dict with ``records``, ``train``, ``test``, ``train_groups``,
        ``val_groups``, ``headline``, ``saved_to``, and ``alloys``.  ``test``
        is the geometry-grouped validation record count; when it is 0 the
        headline numbers are NaN.

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
            # Resumable: a second "generate starter data" click must not
            # re-solve and duplicate every catalog x alloy record.
            plan_path=os.path.join(data_dir, "generation_plan.json"),
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

    if progress_cb and result["val_records"] == 0:
        progress_cb(
            "All records share one geometry group; no validation split, "
            "accuracy left unmeasured."
        )

    alloys = sorted({r["alloy_name"] for r in records})
    saved_to = save_trained(
        weights=result["weights"],
        headline=result["headline"],
        name=name,
        grid=grid,
        alloys=alloys,
        output_dir=output_dir,
        notes="Linear-log ridge least-squares surrogate (end and local-time heads) "
              "fit on captured runs, split by geometry group. "
              "Upgrade path: torch 3D U-Net + FiLM for higher fidelity.",
        local_weights=result["local_weights"],
        split_info={
            "train_records": result["train_records"],
            "val_records": result["val_records"],
            "train_groups": result["train_groups"],
            "val_groups": result["val_groups"],
            "val_cells": result["val_cells"],
        },
    )
    if progress_cb:
        progress_cb(f"✓ Model saved to {saved_to}")
    return {
        "records": len(records),
        "train": result["train_records"],
        "test": result["val_records"],
        "train_groups": result["train_groups"],
        "val_groups": result["val_groups"],
        "headline": result["headline"],
        "saved_to": saved_to,
        "alloys": alloys,
    }