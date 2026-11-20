from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from core.error_logger import logger
except ImportError:
    from core.error_logger_dummy import logger

from core.part_manager import PART_TYPE_CHILL
from core.solidification_params import DEFAULT_MAX_SIMULATION_TIME_S
from core.surrogate.conditioning import Conditioning, build_conditioning_vector
from core.surrogate.fields import (
    CHANNELS,
    PART_TYPE_DIVISOR,
    GridMeta,
    encode_channels,
    encode_field,
    encoding_points,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "surrogate_dataset")
INDEX_FILENAME = "index.csv"
DEFAULT_MESH_TYPE = "voxel"

CHILL_CHANNEL_TOL = 0.05
MAX_END_BEFORE_START_FRACTION = 0.01
MIN_MEDIAN_END_S = 1.0
CHILL_BOOST_CHANNEL = CHANNELS.index("chill_boost")
ZERO_BOOST_TOL = 1e-6

_LAST_LOAD_SKIPS: List[Tuple[str, str]] = []

def _has_cut_cells(mesh: Any) -> bool:
    return bool(getattr(mesh, "cut_cells", False))

def _has_explicit_mold(engine: Any) -> bool:
    return getattr(engine, "_mold_extension", None) is not None

def _config_for_engine(engine: Any, voxel_size: float) -> Dict[str, float]:
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
        "max_simulation_time": _g(timing, "max_simulation_time", DEFAULT_MAX_SIMULATION_TIME_S),
    }

def capture_run(
    engine: Any,
    *,
    voxel_size: float,
    resolution: int = 64,
    out_dir: str = DEFAULT_DATASET_DIR,
) -> Optional[str]:
    try:
        if getattr(engine, "mesh", None) is None:
            return None
        solidification_times = getattr(engine, "solidification_times", None)
        if solidification_times is None or getattr(engine, "solidification_start_times", None) is None:
            return None

        initial_fn = getattr(engine, "compute_initial_chill_influence", None)
        if callable(initial_fn):
            chill_boost = initial_fn()
        else:
            chill_boost = engine._calculate_chill_influence()
        riser_slowdown = engine._calculate_riser_influence()

        mesh_type = DEFAULT_MESH_TYPE
        cut_cells = _has_cut_cells(engine.mesh)
        explicit_mold = _has_explicit_mold(engine)
        points = encoding_points(engine.mesh, resolution)
        channels, meta = encode_channels(
            engine, chill_boost, riser_slowdown, resolution=resolution, points=points
        )
        conditioning = build_conditioning_vector(
            engine.alloy, _config_for_engine(engine, voxel_size)
        )
        centers = np.asarray(engine.mesh.voxel_centers, dtype=np.float64)
        start_grid = encode_field(points.centers, points.values(engine.solidification_start_times), meta)
        end_grid = encode_field(points.centers, points.values(engine.solidification_times), meta)
        local_grid = encode_field(points.centers, points.values(engine.local_solidification_time), meta)

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
            mesh_type=mesh_type,
            cut_cells=cut_cells,
            explicit_mold=explicit_mold,
        )
        _append_index(out_dir, {
            "sample_id": sample_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "alloy": engine.alloy.name,
            "voxel_count": len(centers),
            "resolution": int(resolution),
            "path": path,
            "mesh_type": mesh_type,
            "cut_cells": int(cut_cells),
            "explicit_mold": int(explicit_mold),
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
    if not os.path.isdir(out_dir):
        return {"count": 0, "alloys": [], "runs": []}
    runs: List[Dict[str, Any]] = []
    index_path = os.path.join(out_dir, INDEX_FILENAME)
    if os.path.exists(index_path):
        with open(index_path, newline="") as fh:
            for row in csv.DictReader(fh):
                trailing = list(row.pop(None, None) or [])
                if not row.get("mesh_type"):
                    row["mesh_type"] = trailing.pop(0) if trailing else DEFAULT_MESH_TYPE
                if not row.get("cut_cells"):
                    row["cut_cells"] = trailing.pop(0) if trailing else "0"
                if not row.get("explicit_mold"):
                    row["explicit_mold"] = trailing.pop(0) if trailing else "0"
                runs.append(row)
    alloys = sorted({r.get("alloy", "?") for r in runs})
    return {"count": len(runs), "alloys": alloys, "runs": runs}

def active_cell_mask(channels: np.ndarray) -> np.ndarray:
    channels = np.asarray(channels)
    occupied = channels[0].reshape(-1) > 0.5
    chill = np.abs(channels[1].reshape(-1) - PART_TYPE_CHILL / PART_TYPE_DIVISOR) < CHILL_CHANNEL_TOL
    return occupied & ~chill

def chill_cell_mask(channels: np.ndarray) -> np.ndarray:
    channels = np.asarray(channels)
    occupied = channels[0].reshape(-1) > 0.5
    chill = np.abs(channels[1].reshape(-1) - PART_TYPE_CHILL / PART_TYPE_DIVISOR) < CHILL_CHANNEL_TOL
    return occupied & chill

def validate_record(record: Dict[str, Any]) -> Optional[str]:
    channels = np.asarray(record["channels"])
    active = active_cell_mask(channels)
    if not np.any(active):
        return "no_active_cells"
    if np.any(chill_cell_mask(channels)):
        boost = channels[CHILL_BOOST_CHANNEL].reshape(-1)[active]
        if not np.any(boost > ZERO_BOOST_TOL):
            return "chill_present_but_zero_boost"
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
    return list(_LAST_LOAD_SKIPS)

def load_record(path: str) -> Dict[str, Any]:
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
        mesh_type = str(npz["mesh_type"]) if "mesh_type" in npz.files else DEFAULT_MESH_TYPE
        cut_cells = bool(npz["cut_cells"]) if "cut_cells" in npz.files else False
        explicit_mold = bool(npz["explicit_mold"]) if "explicit_mold" in npz.files else False
        return {
            "path": path,
            "alloy_name": str(npz["alloy_name"]),
            "mesh_type": mesh_type,
            "cut_cells": cut_cells,
            "explicit_mold": explicit_mold,
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
