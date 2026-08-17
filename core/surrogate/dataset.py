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
from typing import Any, Dict, List, Optional

import numpy as np

from core.surrogate.conditioning import Conditioning, build_conditioning_vector
from core.surrogate.fields import GridMeta, encode_channels, encode_field

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "surrogate_dataset")
INDEX_FILENAME = "index.csv"


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


def load_dataset(
    out_dir: str = DEFAULT_DATASET_DIR,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load captured records for training.

    Returns:
        List of records with ``channels``, ``conditioning`` (Conditioning),
        ``end``/``start``/``local`` fields, and ``meta``.
    """
    records: List[Dict[str, Any]] = []
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
            with np.load(path, allow_pickle=True) as npz:
                if int(npz.get("schema_version", 1)) < 2:
                    continue
                meta = GridMeta.from_dict(json.loads(str(npz["grid_meta"])))
                conditioning = Conditioning(
                    vector=np.asarray(npz["conditioning"], dtype=np.float32),
                    names=list(npz["conditioning_names"]),
                )
                channels = np.asarray(npz["channels"], dtype=np.float32)
                grid_size = channels[0].size
                end = np.asarray(npz["solidification_end_times"], dtype=np.float32)
                if end.size != grid_size:
                    continue
                records.append(
                    {
                        "path": path,
                        "alloy_name": str(npz["alloy_name"]),
                        "meta": meta,
                        "channels": np.asarray(npz["channels"], dtype=np.float32),
                        "start": np.asarray(npz["solidification_start_times"], dtype=np.float32),
                        "end": np.asarray(npz["solidification_end_times"], dtype=np.float32),
                        "local": np.asarray(npz["local_solidification_time"], dtype=np.float32),
                        "conditioning": conditioning,
                    }
                )
        except Exception:
            continue
    return records