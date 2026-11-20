from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from core.part_manager import (
    PART_TYPE_CAST,
    PART_TYPE_CORE,
    PART_TYPE_FILTER,
    PART_TYPE_GATE,
    PART_TYPE_MOLD,
    PART_TYPE_POUR_CUP,
    PART_TYPE_RUNNER,
    PART_TYPE_SPRUE,
)
from core.voxel_mesh import AdaptiveVoxelMesh

CHANNELS: List[str] = [
    "occupancy",
    "part_type",
    "surface_distance",
    "local_modulus",
    "normalized_height",
    "chill_boost",
    "riser_slowdown",
]

_LOG6 = float(np.log(6.0))
_MAX_CHILL_BOOST = 5.0

PART_TYPE_DIVISOR = 3.0

SURROGATE_PART_TYPE_ALIASES = {
    PART_TYPE_POUR_CUP: PART_TYPE_CAST,
    PART_TYPE_SPRUE: PART_TYPE_CAST,
    PART_TYPE_RUNNER: PART_TYPE_CAST,
    PART_TYPE_GATE: PART_TYPE_CAST,
    PART_TYPE_FILTER: PART_TYPE_CORE,
    PART_TYPE_MOLD: PART_TYPE_CORE,
}

def apply_part_type_aliases(part_types: np.ndarray) -> np.ndarray:
    aliased = np.asarray(part_types).copy()
    for source, target in SURROGATE_PART_TYPE_ALIASES.items():
        aliased[aliased == source] = target
    return aliased

@dataclass
class GridMeta:

    resolution: int
    bbox: np.ndarray
    cell_size: float
    max_dim: float
    channels: List[str] = field(default_factory=lambda: list(CHANNELS))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resolution": int(self.resolution),
            "bbox": self.bbox.tolist(),
            "cell_size": float(self.cell_size),
            "max_dim": float(self.max_dim),
            "channels": list(self.channels),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GridMeta":
        return cls(
            resolution=int(data["resolution"]),
            bbox=np.asarray(data["bbox"], dtype=np.float64),
            cell_size=float(data["cell_size"]),
            max_dim=float(data["max_dim"]),
            channels=list(data.get("channels", CHANNELS)),
        )

def _grid_cells(points: np.ndarray, bmin: np.ndarray, cell_size: float, resolution: int) -> np.ndarray:
    r = int(resolution)
    idx = np.floor((np.asarray(points, dtype=np.float64) - bmin) / cell_size).astype(np.int64)
    idx = np.clip(idx, 0, r - 1)
    return idx[:, 0] + r * (idx[:, 1] + r * idx[:, 2])

def _scatter(centers: np.ndarray, values: np.ndarray, meta: GridMeta) -> np.ndarray:
    r = meta.resolution
    flat = np.zeros(r * r * r, dtype=np.float64)
    counts = np.zeros(r * r * r, dtype=np.int32)
    flat_idx = _grid_cells(centers, meta.bbox[:, 0], meta.cell_size, r)
    np.add.at(flat, flat_idx, values.astype(np.float64))
    np.add.at(counts, flat_idx, 1)
    filled = counts > 0
    grid = np.zeros(r * r * r, dtype=np.float32)
    grid[filled] = (flat[filled] / counts[filled]).astype(np.float32)
    return grid.reshape(r, r, r)

def _scatter_mode(centers: np.ndarray, codes: np.ndarray, meta: GridMeta) -> np.ndarray:
    r = meta.resolution
    flat_idx = _grid_cells(centers, meta.bbox[:, 0], meta.cell_size, r)
    codes = np.asarray(codes, dtype=np.int64)
    best = np.zeros(r * r * r, dtype=np.int64)
    best_count = np.zeros(r * r * r, dtype=np.int64)
    for code in np.unique(codes):
        count = np.bincount(flat_idx[codes == code], minlength=r * r * r)
        wins = (count > 0) & (count >= best_count)
        best[wins] = code
        best_count[wins] = count[wins]
    return best.reshape(r, r, r)

@dataclass
class EncodingPoints:

    centers: np.ndarray
    part_types: np.ndarray
    bbox: np.ndarray
    resolution: int

    def values(self, mesh_values: np.ndarray) -> np.ndarray:
        return np.asarray(mesh_values)

def encoding_points(mesh: Any, resolution: int = 64) -> EncodingPoints:
    centers = np.asarray(mesh.voxel_centers, dtype=np.float64)
    part_types = getattr(mesh, "voxel_part_types", None)
    if part_types is None:
        part_types = np.zeros(centers.shape[0], dtype=np.int32)
    bbox = (
        np.stack([centers.min(axis=0), centers.max(axis=0)], axis=1)
        if centers.shape[0]
        else np.zeros((3, 2), dtype=np.float64)
    )
    return EncodingPoints(
        centers=centers,
        part_types=np.asarray(part_types, dtype=np.int32),
        bbox=bbox,
        resolution=int(resolution),
    )

def encode_channels(
    engine: Any,
    chill_boost: np.ndarray,
    riser_slowdown: np.ndarray,
    resolution: int = 64,
    *,
    points: Optional[EncodingPoints] = None,
) -> tuple[np.ndarray, GridMeta]:
    mesh: AdaptiveVoxelMesh = engine.mesh
    if points is None:
        points = encoding_points(mesh, resolution)
    elif points.resolution != int(resolution):
        raise ValueError(
            f"encoding points were built for resolution {points.resolution}, not {resolution}"
        )
    centers = points.centers
    if centers.shape[0] == 0:
        raise ValueError("Cannot encode an empty voxel mesh.")

    bbox = np.asarray(points.bbox, dtype=np.float64)
    max_dim = float(np.max(bbox[:, 1] - bbox[:, 0]))
    cell_size = max_dim / resolution

    n_voxels = centers.shape[0]
    part_types = points.part_types

    surface_distances = getattr(engine, "surface_distances", None)
    if surface_distances is None:
        surface_distances = np.zeros(n_voxels, dtype=np.float32)
    else:
        surface_distances = np.asarray(points.values(surface_distances), dtype=np.float32)
    moduli = getattr(mesh, "voxel_moduli", None)
    if moduli is None:
        moduli = np.zeros(n_voxels, dtype=np.float32)
    else:
        moduli = np.asarray(points.values(moduli), dtype=np.float32)
    heights = getattr(engine, "_get_normalized_heights", None)
    if heights is None:
        heights = np.zeros(n_voxels, dtype=np.float32)
    else:
        heights = np.asarray(points.values(heights()), dtype=np.float32)

    chill = np.asarray(points.values(chill_boost), dtype=np.float32)
    riser = np.asarray(points.values(riser_slowdown), dtype=np.float32)

    meta = GridMeta(resolution=int(resolution), bbox=bbox, cell_size=cell_size, max_dim=max_dim)

    channels = np.zeros((len(CHANNELS), resolution, resolution, resolution), dtype=np.float32)

    channels[0] = _scatter(centers, np.ones(n_voxels, dtype=np.float32), meta)
    channels[1] = _scatter_mode(centers, apply_part_type_aliases(part_types), meta).astype(np.float32) / PART_TYPE_DIVISOR
    channels[2] = _scatter(centers, np.clip(surface_distances / max_dim, 0.0, 1.0), meta)
    channels[3] = _scatter(centers, np.clip(moduli / max_dim, 0.0, 1.0), meta)
    channels[4] = _scatter(centers, np.clip(heights, 0.0, 1.0), meta)
    channels[5] = _scatter(centers, np.log1p(np.clip(chill, 1.0, _MAX_CHILL_BOOST) - 1.0) / _LOG6, meta)
    channels[6] = _scatter(centers, np.clip(riser, 0.0, 1.0), meta)

    return channels.astype(np.float32), meta

def encode_field(centers: np.ndarray, values: np.ndarray, meta: GridMeta) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] != centers.shape[0]:
        raise ValueError(
            f"Field length {values.shape[0]} does not match centers {centers.shape[0]}"
        )
    return _scatter(centers, values, meta)

def unproject_grid(
    grid: np.ndarray,
    centers: np.ndarray,
    meta: GridMeta,
) -> np.ndarray:
    flat = _grid_cells(centers, meta.bbox[:, 0], meta.cell_size, meta.resolution)
    if grid.ndim == 3:
        return grid.reshape(-1)[flat]
    c = grid.shape[0]
    return grid.reshape(c, -1)[:, flat]
