"""Training-data capture and loading for surrogate models.

Every traditional simulation can opt-in (``surrogate.auto_capture``) to append
its input channels and schedule fields as a compact ``.npz`` record.  The
trainer aggregates these records to fit a surrogate model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.part_manager import PART_TYPE_CHILL
from core.surrogate.conditioning import Conditioning, build_conditioning_vector
from core.surrogate.fields import PART_TYPE_DIVISOR, GridMeta, encode_channels, encode_field

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "surrogate_dataset")
INDEX_FILENAME = "index.csv"

#: Tolerance around ``PART_TYPE_CHILL / PART_TYPE_DIVISOR`` in channel 1 that
#: marks a chill cell (same window as ``core.deep_model.dataset._occupancy_mask``).
CHILL_CHANNEL_TOL = 0.05
#: Records where more than this fraction of active cells have end < start are
#: rejected by :func:`validate_record`.
MAX_END_BEFORE_START_FRACTION = 0.01
#: Minimum median active end time (seconds).  Five legacy 16^3 WCB records of
#: a 17 in part carry end times of 0.01-0.3 s, i.e. an aborted solve; no real
#: casting in this catalog finishes solidifying in under a second.
MIN_MEDIAN_END_S = 1.0

#: ``(path, reason)`` pairs skipped by the most recent :func:`load_dataset` call.
_LAST_LOAD_SKIPS: List[Tuple[str, str]] = []


def _config_for_engine(engine: Any, voxel_size: float) -> Dict[str, float]:
    """Extract the conditioning config scalars from an engine + mesh."""
    mesh = getattr(engine, "mesh", None)
    centers = getattr(mesh, "voxel_centers", None)
    max_dim = 1.0
    if centers is not None and len(centers) > 0:
        cmin = np.min(centers, axis=0)
        cmax = np.max(centers, axis=0)
        max_dim = float(np.max(cmax - cmin))

    params = getattr(getattr(engine, "analysis_options", None), "solidification_params", None)
    timing = getattr(params, "timing", None)
    feeding = getattr(params, "feeding", None)

    def _g(obj, name, default):
        if obj is None:
            return default
        return float(getattr(obj, name, default) or default)

    return {
        "max_dim": max_dim,
        "voxel_size": float(voxel_size),
        "feeding_zone_vertical_bias": _g(feeding, "feeding_zone_vertical_bias", 0.0),
        "chvorinov_mold_constant": _g(timing, "chvorinov_mold_constant", 280.0),
        "safety_factor_no_risers": _g(timing, "safety_factor_no_risers", 1.5),
        "safety_factor_with_risers": _g(timing, "safety_factor_with_risers", 1.8),
        "min_simulation_time": _g(timing, "min_simulation_time", 120.0),
        "max_simulation_time": _g(timing, "max_simulation_time", 14400.0),
    }


def capture_run(
    engine: Any,
    *,
    voxel_size: float,
    resolution: int = 64,
    out_dir: str = DEFAULT_DATASET_DIR,
) -> Optional[str]:
    """Persist one simulation as a surrogate training record.

    Args:
        engine: A completed ``SolidificationEngine``.
        voxel_size: The voxel size used for the mesh (inches).
        resolution: Grid resolution for the stored channels.
        out_dir: Output directory (default ``data/surrogate_dataset``).

    Returns:
        Path of the written ``.npz`` file, or ``None`` when capture failed.
    """
    try:
        if getattr(engine, "mesh", None) is None:
            return None
        solidification_times = getattr(engine, "solidification_times", None)
        if solidification_times is None or getattr(engine, "solidification_start_times", None) is None:
            return None

        chill_boost = getattr(engine, "_chill_boost", None)
        if chill_boost is None:
            chill_boost = engine._calculate_chill_influence()
        riser_slowdown = engine._calculate_riser_influence()

        channels, meta = encode_channels(engine, chill_boost, riser_slowdown, resolution=resolution)
        conditioning = build_conditioning_vector(
            engine.alloy, _config_for_engine(engine, voxel_size)
        )
        centers = np.asarray(engine.mesh.voxel_centers, dtype=np.float64)
        start_grid = encode_field(centers, engine.solidification_start_times, meta)
        end_grid = encode_field(centers, engine.solidification_times, meta)
        local_grid = encode_field(centers, engine.local_solidification_time, meta)

        os.makedirs(out_dir, exist_ok=True)
        h = hashlib.sha1()
        h.update(f"{time.time()}:{engine.alloy.name}:{end_grid.tobytes()}".encode())
        sample_id = h.hexdigest()[:12]
        path = os.path.join(out_dir, f"run_{sample_id}.npz")

        np.savez_compressed(
            path,
            schema_version=2,
            channels=channels.astype(np.float32),
            solidification_start_times=start_grid,
            solidification_end_times=end_grid,
            local_solidification_time=local_grid,
            conditioning=conditioning.vector,
            conditioning_names=np.asarray(conditioning.names),
            alloy_name=engine.alloy.name,
            grid_resolution=int(resolution),
            grid_meta=json.dumps(meta.to_dict()),
        )
        _append_index(out_dir, {
            "sample_id": sample_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "alloy": engine.alloy.name,
            "voxel_count": len(centers),
            "resolution": int(resolution),
            "path": path,
        })
        return path
    except Exception:
        return None


def _append_index(out_dir: str, row: Dict[str, Any]) -> None:
    index_path = os.path.join(out_dir, INDEX_FILENAME)
    exists = os.path.exists(index_path)
    with open(index_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def dataset_summary(out_dir: str = DEFAULT_DATASET_DIR) -> Dict[str, Any]:
    """Return a summary of the captured dataset."""
    if not os.path.isdir(out_dir):
        return {"count": 0, "alloys": [], "runs": []}
    runs: List[Dict[str, Any]] = []
    index_path = os.path.join(out_dir, INDEX_FILENAME)
    if os.path.exists(index_path):
        with open(index_path, newline="") as fh:
            for row in csv.DictReader(fh):
                runs.append(row)
    alloys = sorted({r.get("alloy", "?") for r in runs})
    return {"count": len(runs), "alloys": alloys, "runs": runs}


def active_cell_mask(channels: np.ndarray) -> np.ndarray:
    """Flat mask of occupied, non-chill grid cells for a record's channels."""
    channels = np.asarray(channels)
    occupied = channels[0].reshape(-1) > 0.5
    chill = np.abs(channels[1].reshape(-1) - PART_TYPE_CHILL / PART_TYPE_DIVISOR) < CHILL_CHANNEL_TOL
    return occupied & ~chill


def validate_record(record: Dict[str, Any]) -> Optional[str]:
    """Return a skip reason for an unusable record, or ``None`` when it is fine.

    Checks run on active (occupied, non-chill) cells:
      * ``no_active_cells`` -- occupancy grid is empty
      * ``nonfinite_end`` / ``nonfinite_start`` / ``nonfinite_local`` -- NaN or inf
      * ``no_positive_end`` -- no active cell has end > 0
      * ``median_end_nonpositive`` -- median active end <= 0
      * ``median_end_below_1s`` -- median active end < ``MIN_MEDIAN_END_S``
      * ``end_before_start_gt_1pct`` -- end < start on > 1% of active cells
    """
    active = active_cell_mask(record["channels"])
    if not np.any(active):
        return "no_active_cells"
    end = np.asarray(record["end"], dtype=np.float32).reshape(-1)[active]
    start = np.asarray(record["start"], dtype=np.float32).reshape(-1)[active]
    local = np.asarray(record["local"], dtype=np.float32).reshape(-1)[active]
    if not np.all(np.isfinite(end)):
        return "nonfinite_end"
    if not np.all(np.isfinite(start)):
        return "nonfinite_start"
    if not np.all(np.isfinite(local)):
        return "nonfinite_local"
    if not np.any(end > 0.0):
        return "no_positive_end"
    # Median over cells the solver actually scheduled.  CORE and gating
    # voxels are encoded as active but carry end == 0, so an assembly with a
    # large core must not be rejected for having a zero median.
    positive_end = end[end > 0.0]
    median_end = float(np.median(positive_end))
    if median_end <= 0.0:
        return "median_end_nonpositive"
    if median_end < MIN_MEDIAN_END_S:
        return "median_end_below_1s"
    if float(np.mean(end < start)) > MAX_END_BEFORE_START_FRACTION:
        return "end_before_start_gt_1pct"
    return None


def last_load_skips() -> List[Tuple[str, str]]:
    """``(path, reason)`` pairs skipped by the most recent :func:`load_dataset`."""
    return list(_LAST_LOAD_SKIPS)


def load_record(path: str) -> Dict[str, Any]:
    """Load one captured ``.npz`` record without validation.

    Raises:
        ValueError: schema older than 2 or field size mismatch.
    """
    with np.load(path, allow_pickle=True) as npz:
        if int(npz.get("schema_version", 1)) < 2:
            raise ValueError("schema_version_lt_2")
        meta = GridMeta.from_dict(json.loads(str(npz["grid_meta"])))
        conditioning = Conditioning(
            vector=np.asarray(npz["conditioning"], dtype=np.float32),
            names=list(npz["conditioning_names"]),
        )
        channels = np.asarray(npz["channels"], dtype=np.float32)
        end = np.asarray(npz["solidification_end_times"], dtype=np.float32)
        if end.size != channels[0].size:
            raise ValueError("grid_size_mismatch")
        return {
            "path": path,
            "alloy_name": str(npz["alloy_name"]),
            "meta": meta,
            "channels": channels,
            "start": np.asarray(npz["solidification_start_times"], dtype=np.float32),
            "end": end,
            "local": np.asarray(npz["local_solidification_time"], dtype=np.float32),
            "conditioning": conditioning,
        }


def load_dataset(
    out_dir: str = DEFAULT_DATASET_DIR,
    limit: Optional[int] = None,
    skipped: Optional[List[Tuple[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """Load captured records for training, dropping unusable ones.

    Every record passes :func:`validate_record`; rejected files are appended
    to ``skipped`` (when given) as ``(path, reason)`` and are also available
    from :func:`last_load_skips`.

    Returns:
        List of records with ``channels``, ``conditioning`` (Conditioning),
        ``end``/``start``/``local`` fields, and ``meta``.
    """
    records: List[Dict[str, Any]] = []
    _LAST_LOAD_SKIPS.clear()
    if not os.path.isdir(out_dir):
        return records
    names = sorted(
        f for f in os.listdir(out_dir) if f.startswith("run_") and f.endswith(".npz")
    )
    if limit is not None:
        names = names[: int(limit)]
    for name in names:
        path = os.path.join(out_dir, name)
        try:
            record = load_record(path)
            reason = validate_record(record)
        except ValueError as exc:
            reason = str(exc)
        except Exception as exc:
            reason = f"load_error: {type(exc).__name__}: {exc}"
        if reason is not None:
            _LAST_LOAD_SKIPS.append((path, reason))
            if skipped is not None:
                skipped.append((path, reason))
            continue
        records.append(record)
    return records