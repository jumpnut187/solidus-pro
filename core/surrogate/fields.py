"""Fixed-grid encoding / decoding between the voxel mesh and surrogate models.

Surrogate models operate on a fixed-resolution 3D grid (e.g. 64^3) rather
than the arbitrary voxel count of the adaptive mesh.  This module:

* ``encode_channels`` — scatter per-voxel physics-informed channels onto the
  grid (averaging into cells).
* ``unproject_grid`` — map grid predictions back to the native voxel mesh via
  nearest-neighbour lookup.

All channels are normalized to [0, 1] with deterministic bounds so a trained
model sees comparable inputs across runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from core.part_manager import (
    PART_TYPE_CAST,
    PART_TYPE_CORE,
    PART_TYPE_FILTER,
    PART_TYPE_GATE,
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

#: Divisor used to encode voxel part-type codes into channel 1.
#:
#: This is a **frozen encoding constant**, deliberately not derived from the
#: number of PartType members. Every captured training record in
#: ``data/surrogate_dataset/`` and every trained artifact was written with the
#: value in force at the time, and ``core/deep_model/dataset.py`` recovers chill
#: voxels by testing ``channel1 ~= PART_TYPE_CHILL / PART_TYPE_DIVISOR``. If this
#: tracked the part-type count automatically, adding SPRUE/GATE/RUNNER for a
#: mold-filling stage would silently shift every channel-1 value and quietly
#: invalidate all existing models -- with no error, just worse predictions.
#:
#: To change it: bump this value, bump ModelManifest.part_type_divisor's default,
#: and retrain. ``load_model`` refuses artifacts whose recorded divisor differs
#: from this one, so the break is loud.
#:
#: Note the encoding is not strictly [0, 1]: CORE (code 4) maps to 4/3. That is
#: pre-existing and is preserved deliberately rather than "fixed", because
#: changing it would invalidate the dataset.
PART_TYPE_DIVISOR = 3.0


#: Gating part types remapped to an in-distribution code before encoding.
#:
#: The divisor above is frozen, which keeps existing artifacts *loadable* --
#: ``load_model`` only compares the recorded divisor to this one. It does not
#: make them *applicable*: every model was trained on codes 0-4, i.e. channel-1
#: values in [0, 4/3]. Encoding FILTER (8) and GATE (9) raw would feed 2.667 and
#: 3.0 into a network that has never seen anything above 1.333, with no error
#: and no divisor mismatch to catch it -- just silently worse predictions on
#: exactly the assemblies the mold-filling stage exists to handle.
#:
#: So the gating metals encode as CAST, which is what they are thermally, and
#: the filter encodes as CORE -- the closest in-distribution mold-class code.
#: Deliberately NOT CHILL: ``core/deep_model/dataset.py`` recovers chill voxels
#: with a +/-0.05 window around 2/3 and would pull filter cells into that mask.
#:
#: Delete this table and bump PART_TYPE_DIVISOR once training data containing
#: gating geometry exists; the load gate will then force the retrain loudly.
SURROGATE_PART_TYPE_ALIASES = {
    PART_TYPE_POUR_CUP: PART_TYPE_CAST,
    PART_TYPE_SPRUE: PART_TYPE_CAST,
    PART_TYPE_RUNNER: PART_TYPE_CAST,
    PART_TYPE_GATE: PART_TYPE_CAST,
    PART_TYPE_FILTER: PART_TYPE_CORE,
}


def apply_part_type_aliases(part_types: np.ndarray) -> np.ndarray:
    """Remap gating codes to in-distribution codes for surrogate encoding."""
    aliased = np.asarray(part_types).copy()
    for source, target in SURROGATE_PART_TYPE_ALIASES.items():
        aliased[aliased == source] = target
    return aliased


@dataclass
class GridMeta:
    """Metadata describing a fixed-grid encoding."""

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
    """(M,) int64 flat grid cell of each point: floor binning, clipped to the grid, x-fastest."""
    r = int(resolution)
    idx = np.floor((np.asarray(points, dtype=np.float64) - bmin) / cell_size).astype(np.int64)
    idx = np.clip(idx, 0, r - 1)
    return idx[:, 0] + r * (idx[:, 1] + r * idx[:, 2])


def _scatter(centers: np.ndarray, values: np.ndarray, meta: GridMeta) -> np.ndarray:
    """Average ``values`` into the grid cells occupied by ``centers``."""
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
    """Majority part code of the voxels in each grid cell (ties to the higher code), 0 where empty.

    Part codes are categories, so they are not averaged: half CAST (0) and
    half CORE (4/3) would average to 2/3, the CHILL code, and every metal
    cell on a core face would read as chill.
    """
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
    """The points a mesh is encoded from: the voxel centres, framed by the voxel-centre box."""

    centers: np.ndarray  # (M, 3) float64, inches
    part_types: np.ndarray  # (M,) int32 part codes
    bbox: np.ndarray  # (3, 2) float64, inches: frame of the encoding grid
    resolution: int

    def values(self, mesh_values: np.ndarray) -> np.ndarray:
        """Per-voxel field (``(N,)``) at the encoding points."""
        return np.asarray(mesh_values)


def encoding_points(mesh: Any, resolution: int = 64) -> EncodingPoints:
    """Points :func:`encode_channels` and :func:`encode_field` scatter from for ``mesh``."""
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
    """Encode the voxel mesh + physics-informed channels onto a fixed grid.

    Args:
        engine: A ``SolidificationEngine`` with ``mesh``, ``surface_distances``
            and the ``_get_normalized_heights`` method available.
        chill_boost: Per-voxel chill acceleration factor (>= 1.0).
        riser_slowdown: Per-voxel riser slowdown factor (<= 1.0).
        resolution: Grid resolution per axis (default 64).
        points: Precomputed :func:`encoding_points` for ``engine.mesh``
            (built here when omitted). ``capture_run`` passes them so the
            schedule targets are encoded from the same points.

    Returns:
        Tuple of ``(channels, meta)`` where ``channels`` has shape
        ``(C, resolution, resolution, resolution)`` with channel order
        matching ``CHANNELS``.
    """
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
    """Scatter a single per-voxel field onto the grid (averaging into cells)."""
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
    """Map a grid prediction back to the native voxel mesh.

    Uses the same floor-based cell mapping as :func:`encode_channels`
    (``_scatter``), so each voxel fetches the exact cell its features were
    averaged into — a round-based lookup would read the neighbouring cell
    for voxels in the upper half of a cell and silently drop them.

    Args:
        grid: Shape ``(C, R, R, R)`` or ``(R, R, R)``.
        centers: Voxel centers, shape ``(n_voxels, 3)``.
        meta: The ``GridMeta`` produced by :func:`encode_channels`.

    Returns:
        ``(C, n_voxels)`` or ``(n_voxels,)`` matching ``grid``'s rank.
    """
    flat = _grid_cells(centers, meta.bbox[:, 0], meta.cell_size, meta.resolution)
    if grid.ndim == 3:
        return grid.reshape(-1)[flat]
    c = grid.shape[0]
    return grid.reshape(c, -1)[:, flat]