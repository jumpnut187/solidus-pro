"""
Dense 3D field grids for the web viewport.

The engine hands the web layer a *sparse* result: a list of occupied voxel
centers plus one value per voxel per mode. That shape is fine for drawing
cubes and useless for anything else. Painting a continuous field on the real
casting surface -- the way commercial casting simulators do -- means sampling a
3D texture at each fragment's world position, and a 3D texture needs a dense
lattice.

This module turns the sparse artifact into that dense lattice and persists it
as a compact binary sidecar (``<artifact>.grid.npz``) beside the JSON, so the
50-100 MB JSON is parsed exactly once in the life of a run instead of once per
HTTP request.

Three things matter about the layout and are relied on by the client:

1. The lattice is uniform. Every voxel is ``base_voxel_size`` on a side, so
   ``dims``, ``origin`` and ``spacing`` fully describe it. A tet run has no
   lattice of its own: its nodal fields are sampled barycentrically onto one
   at ``base_voxel_size * tet_lattice_spacing_factor`` (``_build_tet_field_grids``).
2. Cells are indexed x-fastest: ``index = ix + nx * (iy + ny * iz)``. That is
   exactly a WebGL 3D texture of width ``nx``, height ``ny``, depth ``nz``.
3. Scalar grids are *hole-filled*. A cell outside the metal carries the value
   of the nearest cell inside it. Leaving those cells at zero would make
   trilinear sampling fade every surface toward the bottom of the colormap,
   which is the single most visible artifact of naive volume painting: a blue
   halo hugging the whole casting. The mask grid, not the scalar grid, is what
   says where metal actually is.

``origin`` is the world position of the **center** of cell ``(0, 0, 0)``, so a
world point maps to a texture coordinate as
``((p - origin) / spacing + 0.5) / dims``.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import os
import tempfile
import threading

import numpy as np
from scipy import ndimage

from core.part_manager import CODE_TO_PART_TYPE, PART_TYPE_COLORS, PART_TYPE_CAST


# --------------------------------------------------------------------------
# Mode catalogue
# --------------------------------------------------------------------------

#: Public mode id -> the key it is stored under in ``visualization``.
#:
#: The artifact's key names grew organically (``solidification_times`` plural,
#: ``gas_porosity_risk`` suffixed, ``niyama`` and ``niyama_risk`` duplicated);
#: the client's ``RESULT_MODE_IDS`` is the name the UI uses. This table is the
#: only place the two vocabularies meet.
SCALAR_MODE_SOURCES: Dict[str, Tuple[str, ...]] = {
    "shrinkage": ("shrinkage",),
    "porosity": ("porosity",),
    "feed_modulus": ("feed_modulus",),
    "hotspot": ("hotspot",),
    "solidification_time": ("solidification_times",),
    "niyama_raw": ("niyama_raw",),
    "niyama": ("niyama_risk", "niyama"),
    "combined_feed_risk": ("combined_feed_risk",),
    "hot_tear": ("hot_tear",),
    "sand_burn_in": ("sand_burn_in",),
    "gas_generation": ("gas_generation",),
    "gas_entrapment": ("gas_entrapment",),
    "gas_porosity": ("gas_porosity_risk", "gas_porosity"),
    "feeding_quality": ("feeding_quality",),
    "thermal_stress": ("thermal_stress",),
    "distortion": ("distortion",),
    "crack_susceptibility": ("crack_susceptibility",),
    "sdas": ("sdas",),
    "predicted_uts": ("predicted_uts",),
    "fill_time": ("fill_time",),
}

#: Legend unit per mode. Mirrors ``RESULT_MODE_UNITS`` on the client; an absent
#: entry means a dimensionless 0..1 risk field.
MODE_UNITS: Dict[str, str] = {
    "feed_modulus": "in",
    "solidification_time": "s",
    "fill_time": "s",
    "thermal_stress": "MPa",
    "distortion": "in",
    "sdas": "µm",
    "predicted_uts": "MPa",
    "feeding_path_lengths": "in",
}

#: Modes animated by the time scrubber rather than a threshold.
#:
#: Neither carries a unit. Their *frames* are stamped with a time in seconds,
#: but the values in them are fractions: fraction liquid is 0..1 solid to
#: liquid, and a fill-animation frame is the fraction of a cell the metal has
#: reached. Labelling the colour bar "s" put seconds on an axis that never
#: held any.
TIME_MODES = ("fraction_liquid", "fill_animation")

#: Modes whose per-voxel values are only meaningful where the fill reached, or
#: where the value is a physical time rather than a risk. ``fill_time`` marks a
#: misrun with -1 and a non-metal voxel with NaN (see ``core.filling_stage``),
#: so neither may be treated as a sample of the field.
_NON_NEGATIVE_ONLY_MODES = frozenset({"fill_time"})

_QUANT_MAX = 255.0

# The lattice itself, its size guard and the scatter/fill live in
# core.dense_grid so the desktop renderer shares them. Re-exported under the
# names this module always had.
from core.dense_grid import (  # noqa: E402
    MAX_LATTICE_CELLS,
    Lattice,
    LatticeTooLargeError,
    derive_lattice,
    flat_indices as _flat_indices,
    NODE_SAMPLE_FRAME_BATCH as _NODE_SAMPLE_FRAME_BATCH,
    lattice_from_sampler as _lattice_from_sampler,
    nearest_known_map as _nearest_known_map,
    sample_node_values as _sample_node_values,
    scatter_and_fill as _scatter_and_fill,
)

#: ``mesh_type`` of a tet-mesh run in the visualization payload and grid source.
TET_MESH_TYPE = "tet"


class TetGridUnavailableError(RuntimeError):
    """A tet-mesh run has no field grid and none can be built.

    The grid of a tet run is sampled from the tet connectivity, which only the
    in-memory grid source carries; the JSON artifact holds a strided node
    preview. When the sidecar is gone there is nothing correct to rebuild it
    from, and a lattice derived from scattered nodes would be wrong, so the
    run has to be re-run.
    """


@dataclass
class ScalarGrid:
    """One dense scalar field, quantized to bytes plus the range to undo it."""

    values: np.ndarray  # uint8, flat, x-fastest
    minimum: float
    maximum: float
    mean: float
    #: True when the field never varies. A constant field is still a result --
    #: "every voxel solidified at 42 s" is an answer -- so it stays selectable
    #: and the legend degenerates to a single value.
    constant: bool = False
    #: True only when the field is constant *and* zero, which is what "this
    #: analysis produced nothing" looks like. Reporting a constant non-zero
    #: field as empty hid real answers.
    empty: bool = False

    def dequantize(self) -> np.ndarray:
        span = self.maximum - self.minimum
        if span <= 0.0:
            return np.full(self.values.shape, self.minimum, dtype=np.float32)
        return (self.minimum + (self.values.astype(np.float32) / _QUANT_MAX) * span).astype(np.float32)


@dataclass
class FrameStack:
    """A time series of dense scalar grids, quantized to bytes."""

    frames: np.ndarray  # uint8, (count, cell_count)
    times_s: np.ndarray  # float32, (count,)
    minimum: float = 0.0
    maximum: float = 1.0

    @property
    def count(self) -> int:
        return int(self.frames.shape[0])


@dataclass
class FieldGrids:
    """Everything the viewport needs about one run, in dense form."""

    lattice: Lattice
    mask: np.ndarray  # uint8, flat: 0 outside, otherwise part-type code + 1
    scalars: Dict[str, ScalarGrid] = dataclass_field(default_factory=dict)
    stacks: Dict[str, FrameStack] = dataclass_field(default_factory=dict)
    voxel_count: int = 0
    cast_voxel_count: int = 0
    total_volume_in3: float = 0.0
    stats: Dict[str, Any] = dataclass_field(default_factory=dict)
    part_type_counts: Dict[int, int] = dataclass_field(default_factory=dict)
    #: True when this grid was built from a strided visualization payload
    #: rather than from the full mesh, so the mask has holes the casting does
    #: not. The shell warns rather than pretending the sieve is the part.
    downsampled: bool = False
    #: "tet" when the grid was sampled from a tet mesh (``voxel_count`` and
    #: ``part_type_counts`` are then node counts), else "voxel".
    mesh_type: str = "voxel"


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------


def _quantize(dense: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    span = maximum - minimum
    if span <= 0.0:
        return np.zeros(dense.shape, dtype=np.uint8)
    scaled = (dense.astype(np.float64) - minimum) / span
    return np.clip(np.rint(scaled * _QUANT_MAX), 0.0, _QUANT_MAX).astype(np.uint8)


def build_field_grids(
    visualization: Dict[str, Any],
    summary: Optional[Dict[str, Any]] = None,
    *,
    downsampled: Optional[bool] = None,
) -> FieldGrids:
    """
    Build every dense grid for one run from a visualization payload.

    ``downsampled`` says whether the payload is the strided one the JSON
    carries rather than the full mesh. Left as ``None`` it is inferred by
    comparing the number of centers against the run's own voxel count, which
    is the only signal a lazily-rebuilt grid has.
    """
    summary = summary or {}
    if visualization.get("mesh_type") == TET_MESH_TYPE:
        if visualization.get("tets") is None:
            raise TetGridUnavailableError(
                "This tet-mesh run has no saved field grid. Re-run it to regenerate the result fields."
            )
        return _build_tet_field_grids(visualization, summary)
    # `value or []` would raise on a numpy array -- "the truth value of an
    # array with more than one element is ambiguous" -- and the full-resolution
    # grid source hands these in as arrays rather than lists.
    centers = _as_array(visualization.get("centers"))
    if centers.size == 0:
        empty_lattice = Lattice(dims=(1, 1, 1), origin=(0.0, 0.0, 0.0), spacing=1.0)
        return FieldGrids(lattice=empty_lattice, mask=np.zeros(1, dtype=np.uint8))
    centers = centers.reshape((-1, 3))
    n_voxels = int(centers.shape[0])

    sizes = _as_array(visualization.get("sizes"))
    lattice = derive_lattice(centers, sizes if sizes.size == n_voxels else None)
    cell_count = lattice.cell_count
    cell_indices = _flat_indices(centers, lattice)

    raw_part_types = visualization.get("part_types")
    if raw_part_types is not None and len(raw_part_types) == n_voxels:
        part_types = np.asarray(raw_part_types, dtype=np.int32)
    else:
        part_types = np.full(n_voxels, PART_TYPE_CAST, dtype=np.int32)

    # Code + 1 so that 0 stays available for "outside the metal". Nearest-value
    # sampling of this grid on the GPU is what decides whether a fragment gets
    # painted at all.
    mask = np.zeros(cell_count, dtype=np.uint8)
    mask[cell_indices] = np.clip(part_types + 1, 0, 255).astype(np.uint8)

    nearest = _nearest_known_map(mask != 0, lattice)

    reported = summary.get("voxel_count")
    strided = downsampled
    if strided is None:
        # A payload with fewer centers than the run had voxels was strided on
        # the way out; every voxel it dropped is a hole in the mask.
        strided = bool(isinstance(reported, (int, float)) and int(reported) > n_voxels)

    grids = FieldGrids(
        lattice=lattice,
        mask=mask,
        downsampled=bool(strided),
        voxel_count=n_voxels,
        cast_voxel_count=int(np.sum(part_types == PART_TYPE_CAST)),
        total_volume_in3=float(summary.get("total_volume", 0.0) or 0.0),
        stats=_flatten_stats(summary),
    )
    codes, counts = np.unique(part_types, return_counts=True)
    grids.part_type_counts = {int(code): int(count) for code, count in zip(codes, counts)}

    for mode, source_keys in SCALAR_MODE_SOURCES.items():
        values = _first_present(visualization, source_keys, n_voxels)
        if values is None:
            continue
        sample_mask = np.isfinite(values)
        if mode in _NON_NEGATIVE_ONLY_MODES:
            sample_mask &= values >= 0.0
        if not np.any(sample_mask):
            continue
        usable = values[sample_mask]
        minimum = float(np.min(usable))
        maximum = float(np.max(usable))
        mean = float(np.mean(usable))
        dense = _scatter_and_fill(
            np.where(sample_mask, values, 0.0).astype(np.float32),
            cell_indices,
            sample_mask,
            nearest,
            lattice,
        )
        constant = bool(maximum <= minimum)
        grids.scalars[mode] = ScalarGrid(
            values=_quantize(dense, minimum, maximum),
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            constant=constant,
            empty=bool(constant and abs(maximum) < 1e-12),
        )

    fraction_liquid = _build_frame_stack(
        visualization.get("fraction_liquid_history"),
        visualization.get("fraction_liquid_times"),
        cell_indices,
        nearest,
        lattice,
        n_voxels,
    )
    if fraction_liquid is not None:
        grids.stacks["fraction_liquid"] = fraction_liquid

    fill = _build_frame_stack(
        visualization.get("fill_animation_history"),
        visualization.get("fill_animation_times"),
        cell_indices,
        nearest,
        lattice,
        n_voxels,
    )
    if fill is not None:
        grids.stacks["fill_animation"] = fill

    return grids


def _build_tet_field_grids(visualization: Dict[str, Any], summary: Dict[str, Any]) -> FieldGrids:
    """
    Every dense grid for a tet run, from the full-resolution grid source.

    ``centers`` are the node coordinates and every field is per node. The
    lattice is a :class:`core.tet_lattice.TetLatticeSampler` at
    ``lattice_spacing``; scalar fields and frames are barycentric samples at
    the inside cells, and the hole fill outside the mesh is the voxel path's
    nearest-inside-cell fill. The layout handed to the client (dims, origin,
    spacing, uint8, x-fastest, mask = code + 1) is the voxel one.
    """
    from core.tet_lattice import TetLatticeSampler

    nodes = _as_array(visualization.get("centers")).reshape((-1, 3))
    n_nodes = int(nodes.shape[0])
    tets = np.asarray(visualization["tets"], dtype=np.int64).reshape((-1, 4))
    tet_part_types = np.asarray(visualization.get("tet_part_types"), dtype=np.int8).reshape(-1)
    spacing = float(visualization.get("lattice_spacing") or 0.0)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise TetGridUnavailableError("tet grid source has no lattice_spacing")

    sampler = TetLatticeSampler(nodes, tets, tet_part_types, spacing)
    lattice = _lattice_from_sampler(sampler)
    mask = sampler.mask_codes()
    nearest = _nearest_known_map(mask != 0, lattice)

    raw_part_types = visualization.get("part_types")
    if raw_part_types is not None and len(raw_part_types) == n_nodes:
        part_types = np.asarray(raw_part_types, dtype=np.int32)
    else:
        part_types = np.full(n_nodes, PART_TYPE_CAST, dtype=np.int32)

    grids = FieldGrids(
        lattice=lattice,
        mask=mask,
        downsampled=False,
        voxel_count=n_nodes,
        cast_voxel_count=int(np.sum(part_types == PART_TYPE_CAST)),
        total_volume_in3=float(summary.get("total_volume", 0.0) or 0.0),
        stats=_flatten_stats(summary),
        mesh_type=TET_MESH_TYPE,
    )
    # Counts are per node, like voxel_count. A code that only tets carry (a
    # chill one tet thick, whose nodes all take the metal code) still gets a
    # row, so every mask code has a name and a colour.
    codes, counts = np.unique(part_types, return_counts=True)
    grids.part_type_counts = {int(code): int(count) for code, count in zip(codes, counts)}
    for code in np.unique(sampler.cell_part_types):
        grids.part_type_counts.setdefault(int(code), 0)

    cell_flat = sampler.cell_flat
    for mode, source_keys in SCALAR_MODE_SOURCES.items():
        values = _first_present(visualization, source_keys, n_nodes)
        if values is None:
            continue
        sample_mask = np.isfinite(values)
        if mode in _NON_NEGATIVE_ONLY_MODES:
            sample_mask &= values >= 0.0
        if not np.any(sample_mask):
            continue
        usable = values[sample_mask]
        minimum = float(np.min(usable))
        maximum = float(np.max(usable))
        mean = float(np.mean(usable))
        cell_values, cell_known = _sample_node_values(sampler, values, sample_mask)
        dense = _scatter_and_fill(cell_values, cell_flat, cell_known, nearest, lattice)
        constant = bool(maximum <= minimum)
        grids.scalars[mode] = ScalarGrid(
            values=_quantize(dense, minimum, maximum),
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            constant=constant,
            empty=bool(constant and abs(maximum) < 1e-12),
        )

    for stack_key, history_key, times_key in (
        ("fraction_liquid", "fraction_liquid_history", "fraction_liquid_times"),
        ("fill_animation", "fill_animation_history", "fill_animation_times"),
    ):
        stack = _build_tet_frame_stack(
            visualization.get(history_key), visualization.get(times_key), sampler, nearest, lattice, n_nodes
        )
        if stack is not None:
            grids.stacks[stack_key] = stack

    return grids


def _build_tet_frame_stack(
    history: Any,
    times: Any,
    sampler: Any,
    nearest: np.ndarray,
    lattice: Lattice,
    n_nodes: int,
) -> Optional[FrameStack]:
    """Per-node frames sampled onto the lattice, quantized like the voxel stack."""
    if not isinstance(history, list) or not history:
        return None
    if any(np.size(snapshot) != n_nodes for snapshot in history):
        return None
    count = len(history)
    frames = np.empty((count, lattice.cell_count), dtype=np.uint8)
    # Batches bound the float transients to a few frames of inside cells.
    for start in range(0, count, _NODE_SAMPLE_FRAME_BATCH):
        batch = np.stack(
            [np.asarray(s, dtype=np.float64).reshape(-1) for s in history[start : start + _NODE_SAMPLE_FRAME_BATCH]],
            axis=0,
        )
        finite = np.isfinite(batch)
        cell_values, cell_known = _sample_node_values(
            sampler, np.where(finite, np.clip(batch, 0.0, 1.0), 0.0), finite
        )
        for offset in range(batch.shape[0]):
            dense = _scatter_and_fill(
                cell_values[offset], sampler.cell_flat, cell_known[offset], nearest, lattice
            )
            frames[start + offset] = _quantize(dense, 0.0, 1.0)

    time_arr = np.asarray(times if times is not None else [], dtype=np.float32).reshape(-1)
    if time_arr.size != count:
        time_arr = np.linspace(0.0, 1.0, count, dtype=np.float32).reshape(-1)
    return FrameStack(frames=frames, times_s=time_arr)


def _as_array(value: Any) -> np.ndarray:
    """Coerce a payload entry to float64, accepting lists, arrays or nothing."""
    if value is None:
        return np.empty(0, dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _first_present(
    visualization: Dict[str, Any], keys: Iterable[str], expected: int
) -> Optional[np.ndarray]:
    for key in keys:
        raw = visualization.get(key)
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        if arr.size == expected:
            return arr
    return None


def _build_frame_stack(
    history: Any,
    times: Any,
    cell_indices: np.ndarray,
    nearest: np.ndarray,
    lattice: Lattice,
    expected: int,
) -> Optional[FrameStack]:
    if not isinstance(history, list) or not history:
        return None
    frames: List[np.ndarray] = []
    for snapshot in history:
        arr = np.asarray(snapshot, dtype=np.float64).reshape(-1)
        if arr.size != expected:
            return None
        finite = np.isfinite(arr)
        dense = _scatter_and_fill(
            np.where(finite, np.clip(arr, 0.0, 1.0), 0.0).astype(np.float32),
            cell_indices,
            finite,
            nearest,
            lattice,
        )
        frames.append(_quantize(dense, 0.0, 1.0))

    time_arr = np.asarray(times if times is not None else [], dtype=np.float32).reshape(-1)
    if time_arr.size != len(frames):
        # No usable clock: space the frames evenly so the scrubber still maps
        # a percentage onto a frame index.
        time_arr = np.linspace(0.0, 1.0, len(frames), dtype=np.float32).reshape(-1)
    return FrameStack(frames=np.stack(frames, axis=0), times_s=time_arr)


def _flatten_stats(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten the artifact summary into the scalar dictionary the shell shows.

    ``GridMeta.stats`` is a flat ``Record<string, number | string>``; the
    artifact nests its statistics one level deep (``shrinkage_stats.max``).
    Flattening here rather than on the client keeps the statistics panel free
    of per-run special cases.
    """
    flat: Dict[str, Any] = {}
    for key, value in summary.items():
        if key in {"raw_ranges", "available_modes"}:
            continue
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float, str, bool)):
                    flat[f"{key.removesuffix('_stats')}_{sub_key}"] = _scalar(sub_value)
        elif isinstance(value, (int, float, str, bool)):
            flat[key] = _scalar(value)
    return flat


def _scalar(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        number = float(value)
        return number if np.isfinite(number) else 0.0
    return str(value)


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


def build_grid_metadata(grids: FieldGrids) -> Dict[str, Any]:
    """The JSON body of ``GET /api/projects/<id>/grid``."""
    modes: List[str] = []
    empty_modes: List[str] = []
    constant_modes: List[str] = []
    ranges: Dict[str, Dict[str, float]] = {}
    units: Dict[str, str] = {}
    for mode, grid in grids.scalars.items():
        ranges[mode] = {"min": grid.minimum, "max": grid.maximum, "mean": grid.mean}
        if mode in MODE_UNITS:
            units[mode] = MODE_UNITS[mode]
        if grid.constant:
            constant_modes.append(mode)
        if grid.empty:
            # Constant and zero: the analysis ran and found nothing.
            empty_modes.append(mode)
        else:
            # Everything else is selectable, including a field that is the
            # same value everywhere -- that is a result, not an absence.
            modes.append(mode)

    frames: Dict[str, Dict[str, Any]] = {}
    if "fraction_liquid" in grids.stacks:
        stack = grids.stacks["fraction_liquid"]
        frames["fraction_liquid"] = {"count": stack.count, "times_s": [float(t) for t in stack.times_s]}
        modes.append("fraction_liquid")
    if "fill_animation" in grids.stacks:
        stack = grids.stacks["fill_animation"]
        frames["fill"] = {"count": stack.count, "times_s": [float(t) for t in stack.times_s]}
        modes.append("fill_animation")

    nx, ny, nz = grids.lattice.dims
    return {
        "dims": [int(nx), int(ny), int(nz)],
        "origin": [float(v) for v in grids.lattice.origin],
        "spacing": float(grids.lattice.spacing),
        "modes": sorted(set(modes)),
        "emptyModes": sorted(set(empty_modes)),
        "constantModes": sorted(set(constant_modes)),
        "ranges": ranges,
        "units": units,
        "frames": frames,
        "voxelCount": int(grids.voxel_count),
        "castVoxelCount": int(grids.cast_voxel_count),
        "cellCount": int(grids.lattice.cell_count),
        "totalVolumeIn3": float(grids.total_volume_in3),
        "downsampled": bool(grids.downsampled),
        "partTypes": _part_type_table(grids.part_type_counts),
        "stats": grids.stats,
        "quantization": {"dtype": "uint8", "levels": int(_QUANT_MAX) + 1},
    }


def _part_type_table(counts: Dict[int, int]) -> List[Dict[str, Any]]:
    """Mask code -> part type, name and display color, for the whole run."""
    table: List[Dict[str, Any]] = []
    for code in sorted(counts):
        part_type = CODE_TO_PART_TYPE.get(int(code))
        color = PART_TYPE_COLORS.get(part_type) if part_type is not None else None
        table.append(
            {
                "code": int(code) + 1,
                "partTypeCode": int(code),
                "id": part_type.value if part_type is not None else f"type_{int(code)}",
                "label": str(part_type) if part_type is not None else f"Type {int(code)}",
                "color": [float(c) for c in color] if color is not None else [0.7, 0.75, 0.8],
                "voxelCount": int(counts[code]),
            }
        )
    return table


# --------------------------------------------------------------------------
# Sidecar persistence
# --------------------------------------------------------------------------

SIDECAR_VERSION = 1


def sidecar_path(artifact_path: Path | str) -> Path:
    """``.../<id>.json`` -> ``.../<id>.grid.npz``."""
    path = Path(artifact_path)
    return path.with_name(path.stem + ".grid.npz")


def save_sidecar(path: Path, grids: FieldGrids) -> None:
    """Write the dense grids as a single compressed ``.npz``."""
    meta: Dict[str, Any] = {
        "stats": grids.stats,
        "part_type_counts": {str(k): v for k, v in grids.part_type_counts.items()},
    }
    if grids.mesh_type == TET_MESH_TYPE:
        # Voxel sidecars stay byte-identical: the key is only written for tet
        # runs. ``spacing`` above is the lattice pitch the tet run used.
        meta["mesh_type"] = TET_MESH_TYPE
    payload: Dict[str, np.ndarray] = {
        "version": np.asarray([SIDECAR_VERSION], dtype=np.int32),
        "dims": np.asarray(grids.lattice.dims, dtype=np.int32),
        "origin": np.asarray(grids.lattice.origin, dtype=np.float64),
        "spacing": np.asarray([grids.lattice.spacing], dtype=np.float64),
        "mask": grids.mask,
        "counts": np.asarray(
            [grids.voxel_count, grids.cast_voxel_count, 1 if grids.downsampled else 0],
            dtype=np.int64,
        ),
        "total_volume": np.asarray([grids.total_volume_in3], dtype=np.float64),
        "meta_json": np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
    }
    for mode, grid in grids.scalars.items():
        payload[f"scalar__{mode}"] = grid.values
        payload[f"scalar_range__{mode}"] = np.asarray(
            [
                grid.minimum,
                grid.maximum,
                grid.mean,
                1.0 if grid.empty else 0.0,
                1.0 if grid.constant else 0.0,
            ],
            dtype=np.float64,
        )
    for key, stack in grids.stacks.items():
        payload[f"stack__{key}"] = stack.frames
        payload[f"stack_times__{key}"] = stack.times_s

    path.parent.mkdir(parents=True, exist_ok=True)
    # A unique temp name per writer, not a fixed one. Two processes writing
    # the same sidecar at once -- a re-run racing a lazy rebuild, say -- both
    # opened `<id>.grid.npz.tmp`, interleaved their writes and published the
    # splice, which then fails to load forever.
    handle_fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle_fd, "wb") as handle:
            np.savez_compressed(handle, **payload)  # type: ignore[arg-type]
        # Atomic within a directory, so a reader sees the old file or the new
        # one and never a half-written one.
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_sidecar(path: Path) -> Optional[FieldGrids]:
    """
    Read a sidecar back, or ``None`` when it cannot be used.

    Every failure is a `None`, deliberately. The previous version listed the
    exception types it expected, and a truncated file raises
    `zipfile.BadZipFile`, which is not among them: one interrupted write left
    every grid request for that project returning 500 for good, because the
    rebuild path was never reached. A sidecar is a cache, and the only correct
    response to a cache entry that cannot be read is to throw it away and
    build it again -- so a bad file is also unlinked on the way out, or the
    next request would find the same corpse.
    """
    try:
        with np.load(path, allow_pickle=False) as data:
            if int(data["version"][0]) != SIDECAR_VERSION:
                return None
            dims = tuple(int(v) for v in data["dims"])
            lattice = Lattice(
                dims=(dims[0], dims[1], dims[2]),
                origin=tuple(float(v) for v in data["origin"]),  # type: ignore[arg-type]
                spacing=float(data["spacing"][0]),
            )
            counts = data["counts"]
            meta = json.loads(bytes(data["meta_json"]).decode("utf-8"))
            grids = FieldGrids(
                lattice=lattice,
                mask=np.asarray(data["mask"], dtype=np.uint8),
                voxel_count=int(counts[0]),
                cast_voxel_count=int(counts[1]),
                downsampled=bool(counts[2]) if len(counts) > 2 else False,
                total_volume_in3=float(data["total_volume"][0]),
                stats=dict(meta.get("stats", {})),
                part_type_counts={int(k): int(v) for k, v in meta.get("part_type_counts", {}).items()},
                mesh_type=str(meta.get("mesh_type", "voxel")),
            )
            for key in data.files:
                if key.startswith("scalar__"):
                    mode = key.split("__", 1)[1]
                    rng = data[f"scalar_range__{mode}"]
                    grids.scalars[mode] = ScalarGrid(
                        values=np.asarray(data[key], dtype=np.uint8),
                        minimum=float(rng[0]),
                        maximum=float(rng[1]),
                        mean=float(rng[2]),
                        empty=bool(rng[3] > 0.5),
                        # Older sidecars have four entries and no constant
                        # flag; derive it from the range they do carry.
                        constant=bool(rng[4] > 0.5)
                        if len(rng) > 4
                        else bool(float(rng[1]) <= float(rng[0])),
                    )
                elif key.startswith("stack__"):
                    stack_key = key.split("__", 1)[1]
                    grids.stacks[stack_key] = FrameStack(
                        frames=np.asarray(data[key], dtype=np.uint8),
                        times_s=np.asarray(data[f"stack_times__{stack_key}"], dtype=np.float32),
                    )
            return grids
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - read-only artifact directory
            pass
        return None


def write_sidecar_for_artifact(
    artifact_path: Path | str,
    artifact: Dict[str, Any],
    *,
    grid_source: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    Build and persist the sidecar for an artifact that is being written.

    ``grid_source`` is the full-resolution arrays straight off the mesh, when
    the caller has them. Preferring it over ``artifact["visualization"]`` is
    the whole point: the visualization payload is strided once a run exceeds
    ``max_voxels_for_visualization``, and a mask built from every second voxel
    describes a casting with half its metal missing. The JSON stays
    downsampled -- it is a fallback and a debugging aid -- while the grid the
    viewport actually renders is complete.

    Failures are swallowed on purpose: a run whose results are otherwise good
    must not be lost because the render sidecar could not be produced. The
    lazy path in :func:`load_grids` rebuilds it on first request.
    """
    try:
        if grid_source:
            grids = build_field_grids(grid_source, artifact.get("summary", {}), downsampled=False)
        else:
            grids = build_field_grids(artifact.get("visualization", {}), artifact.get("summary", {}))
        if grids.voxel_count == 0:
            return None
        path = sidecar_path(artifact_path)
        save_sidecar(path, grids)
        return path
    except Exception:  # pragma: no cover - defensive; never fail a completed run
        return None


# --------------------------------------------------------------------------
# In-process cache
# --------------------------------------------------------------------------


def grids_nbytes(grids: FieldGrids) -> int:
    """Roughly what a cached run costs in memory."""
    total = int(grids.mask.nbytes)
    for grid in grids.scalars.values():
        total += int(grid.values.nbytes)
    for stack in grids.stacks.values():
        total += int(stack.frames.nbytes) + int(stack.times_s.nbytes)
    return total


class _GridCache:
    """
    LRU over ``FieldGrids``, bounded by bytes rather than by entry count.

    Counting entries was the wrong unit. Three entries is nothing for the
    reference run's 1.6 MB of grids and a third of a gigabyte for three
    million-voxel runs with long frame stacks, which is how a cache meant to
    save work becomes the reason a process runs out of memory. The mtime is
    part of the key, so a re-run of the same project invalidates its entry
    rather than serving the previous run's field.

    A single run larger than the whole budget is still returned to the caller;
    it simply is not kept.
    """

    def __init__(self, max_bytes: int = 256 * 1024 * 1024) -> None:
        self._max_bytes = max_bytes
        self._entries: "OrderedDict[Tuple[str, int, int], Tuple[FieldGrids, int]]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    @property
    def nbytes(self) -> int:
        with self._lock:
            return self._bytes

    def get(self, key: Tuple[str, int, int]) -> Optional[FieldGrids]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry[0]

    def put(self, key: Tuple[str, int, int], grids: FieldGrids) -> None:
        size = grids_nbytes(grids)
        with self._lock:
            existing = self._entries.pop(key, None)
            if existing is not None:
                self._bytes -= existing[1]
            if size > self._max_bytes:
                # Too big to keep; the caller still gets it, it just does not
                # evict everything else on the way past.
                return
            self._entries[key] = (grids, size)
            self._bytes += size
            while self._bytes > self._max_bytes and self._entries:
                _, (_, evicted) = self._entries.popitem(last=False)
                self._bytes -= evicted

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0


_CACHE = _GridCache()


def clear_cache() -> None:
    """Drop every cached grid. Used by tests and by a fresh app instance."""
    _CACHE.clear()


def load_grids(
    project_id: str,
    artifact_path: Path | str,
    *,
    artifact: Optional[Dict[str, Any]] = None,
) -> Optional[FieldGrids]:
    """
    Get the dense grids for a run, building and caching them as needed.

    Order of preference: the in-process LRU, then the on-disk sidecar, then a
    one-time rebuild from the JSON artifact (which is then written to disk so
    the next process does not pay for it either).
    """
    path = Path(artifact_path)
    try:
        stat = path.stat()
    except OSError:
        if artifact is None:
            return None
        stat = None

    key = (project_id, int(stat.st_mtime_ns) if stat else 0, int(stat.st_size) if stat else 0)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    sidecar = sidecar_path(path)
    grids: Optional[FieldGrids] = None
    if stat is not None and sidecar.exists() and sidecar.stat().st_mtime_ns >= stat.st_mtime_ns:
        grids = load_sidecar(sidecar)

    if grids is None:
        source = artifact
        if source is None:
            if not path.exists():
                return None
            try:
                with path.open("r", encoding="utf-8") as handle:
                    source = json.load(handle)
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                # A run being rewritten, or an artifact that never finished
                # being written. Neither is a server fault, and neither is
                # permanent -- "results are not available" is the honest
                # answer, and the next request will find a whole file.
                return None
        grids = build_field_grids(source.get("visualization", {}), source.get("summary", {}))
        if grids.voxel_count == 0:
            return None
        try:
            save_sidecar(sidecar, grids)
        except OSError:  # pragma: no cover - read-only artifact directory
            pass

    _CACHE.put(key, grids)
    return grids


# --------------------------------------------------------------------------
# Binary responses
# --------------------------------------------------------------------------


def mask_bytes(grids: FieldGrids) -> bytes:
    return np.ascontiguousarray(grids.mask, dtype=np.uint8).tobytes()


def scalar_bytes(grid: ScalarGrid, dtype: str) -> bytes:
    """
    Serialize one scalar grid.

    ``u8`` is the wire default: the client uploads it straight into a
    ``RawTexture3D`` and undoes the quantization in the shader with the min/max
    it already has from the metadata. ``f32`` is offered for callers that want
    field units without a second multiply -- little-endian, matching every
    platform the viewport runs on.
    """
    if dtype == "f32":
        return np.ascontiguousarray(grid.dequantize(), dtype="<f4").tobytes()
    return np.ascontiguousarray(grid.values, dtype=np.uint8).tobytes()


def frame_bytes(stack: FrameStack, start: int, count: int) -> Tuple[bytes, int, int]:
    """Consecutive frames as one Uint8 block; returns ``(bytes, start, count)``."""
    total = stack.count
    begin = max(int(start), 0)
    if begin >= total:
        # Past the end is an empty window, not the last frame. A client paging
        # with start += count has to be able to see that it is done.
        return b"", begin, 0
    length = int(np.clip(count, 0, total - begin))
    if length <= 0:
        return b"", begin, 0
    block = stack.frames[begin : begin + length]
    return np.ascontiguousarray(block, dtype=np.uint8).tobytes(), begin, length


# --------------------------------------------------------------------------
# Isosurface envelopes
# --------------------------------------------------------------------------


@dataclass
class Envelope:
    """A threshold isosurface plus the values sampled along it."""

    positions: np.ndarray  # (V, 3) float32
    normals: np.ndarray  # (V, 3) float32
    indices: np.ndarray  # (T * 3,) uint32
    #: Per-vertex field value in field units, or None when not requested.
    scalars: Optional[np.ndarray]
    threshold: float
    minimum: float
    maximum: float

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.indices.size // 3)


def masked_field_volume(
    grids: FieldGrids,
    mode: str,
    *,
    cast_only: bool = False,
    frame_index: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, float, float]]:
    """
    The dense field as a 3D array indexed ``[ix, iy, iz]``, blanked outside the metal.

    Blanking matters. The stored grid is hole-filled so that trilinear
    sampling on the *surface* never fades toward zero, which means every cell
    in the bounding box carries a plausible-looking value. Contouring that
    directly would grow the envelope out through the mold. Here the cells that
    are not metal -- or, with ``cast_only``, not cast metal -- are pushed below
    the field minimum, so the contour closes at the metal boundary instead.
    """
    nx, ny, nz = grids.lattice.dims
    if frame_index is not None:
        stack = grids.stacks.get(mode)
        if stack is None:
            return None
        index = int(np.clip(frame_index, 0, stack.count - 1))
        dense = stack.frames[index].astype(np.float32) / _QUANT_MAX
        minimum, maximum = 0.0, 1.0
    else:
        grid = grids.scalars.get(mode)
        if grid is None:
            return None
        dense = grid.dequantize()
        minimum, maximum = grid.minimum, grid.maximum

    keep = grids.mask != 0
    if cast_only:
        keep &= grids.mask == PART_TYPE_CAST + 1
    if not np.any(keep):
        return None

    span = max(maximum - minimum, 1e-6)
    outside = minimum - span
    blanked = np.where(keep, dense, outside).astype(np.float32)
    return blanked.reshape((nz, ny, nx)).transpose(2, 1, 0), minimum, maximum


def sample_field_at(
    grids: FieldGrids,
    mode: str,
    points: np.ndarray,
    *,
    frame_index: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Trilinearly sample a field at arbitrary world points.

    Used to colour envelope vertices by the value the field actually has
    there. Without it every vertex carries the threshold, and a "both" view
    shows a flat-coloured blob inside a graded surface -- which reads as a
    rendering bug rather than as an isosurface.
    """
    nx, ny, nz = grids.lattice.dims
    if frame_index is not None:
        stack = grids.stacks.get(mode)
        if stack is None:
            return None
        index = int(np.clip(frame_index, 0, stack.count - 1))
        dense = stack.frames[index].astype(np.float32) / _QUANT_MAX
    else:
        grid = grids.scalars.get(mode)
        if grid is None:
            return None
        dense = grid.dequantize()

    volume = dense.reshape((nz, ny, nx))
    origin = grids.lattice.origin
    spacing = grids.lattice.spacing if grids.lattice.spacing > 0 else 1.0
    # map_coordinates indexes in array order, which for this reshape is z, y, x.
    coords = np.vstack(
        [
            (points[:, 2] - origin[2]) / spacing,
            (points[:, 1] - origin[1]) / spacing,
            (points[:, 0] - origin[0]) / spacing,
        ]
    )
    sampled = ndimage.map_coordinates(volume, coords, order=1, mode="nearest")
    return np.asarray(sampled, dtype=np.float32)


def build_envelope(
    grids: FieldGrids,
    mode: str,
    threshold: float,
    *,
    cast_only: bool = False,
    with_scalars: bool = True,
    frame_index: Optional[int] = None,
    smooth_sigma: float = 0.75,
) -> Optional[Envelope]:
    """Contour ``mode`` at ``threshold`` (field units) and sample values along it."""
    from core.surface_extraction import contour_dense_grid

    prepared = masked_field_volume(grids, mode, cast_only=cast_only, frame_index=frame_index)
    if prepared is None:
        return None
    volume, minimum, maximum = prepared

    surface = contour_dense_grid(
        volume,
        grids.lattice.origin,
        grids.lattice.spacing,
        threshold,
        smooth_sigma=smooth_sigma,
    )

    scalars: Optional[np.ndarray] = None
    if with_scalars and surface.vertex_count:
        scalars = sample_field_at(grids, mode, surface.positions, frame_index=frame_index)

    return Envelope(
        positions=surface.positions,
        normals=surface.normals,
        indices=surface.indices,
        scalars=scalars,
        threshold=float(threshold),
        minimum=float(minimum),
        maximum=float(maximum),
    )


def envelope_bytes(envelope: Envelope) -> bytes:
    """
    Serialize an envelope as one little-endian block.

    Layout, in order, with the counts carried in response headers:

        positions  Float32[3 * V]
        normals    Float32[3 * V]
        indices    Uint32[3 * T]
        scalars    Float32[V]      (only when X-Has-Scalar is 1)

    Every section is a whole number of 4-byte words, so each one starts
    aligned and the client can wrap the buffer in typed-array views without
    copying it.
    """
    parts = [
        np.ascontiguousarray(envelope.positions, dtype="<f4").tobytes(),
        np.ascontiguousarray(envelope.normals, dtype="<f4").tobytes(),
        np.ascontiguousarray(envelope.indices, dtype="<u4").tobytes(),
    ]
    if envelope.scalars is not None:
        parts.append(np.ascontiguousarray(envelope.scalars, dtype="<f4").tobytes())
    return b"".join(parts)
