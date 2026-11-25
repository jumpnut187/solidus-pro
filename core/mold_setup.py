from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from scipy import ndimage

try:
    from core.error_logger import logger
except ImportError:
    from core.error_logger_dummy import logger

from core.part_manager import (
    FEEDER_PART_CODES,
    METAL_PART_CODES,
    PART_TYPE_CAST,
    PART_TYPE_CORE,
    PART_TYPE_POUR_CUP,
    PART_TYPE_RISER,
    PART_TYPE_SPRUE,
    SAND_PART_CODES,
)
from core.solidification_params import CORE_RULE_ENCLOSED, MoldParams

SetupPart = Tuple[trimesh.Trimesh, int, bool]

NEIGHBOR_SLOT_OFFSETS = np.array(
    [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
        (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
    ],
    dtype=np.int64,
)
FACE_SLOTS = range(6)

LATTICE_TOLERANCE = 0.05

ALWAYS_OPEN_TOP_CODES = (PART_TYPE_POUR_CUP,)

GRADING_TRANSITION_CELLS = 2

class MoldSetupError(ValueError):
    pass

@dataclass
class Lattice:

    origin: np.ndarray
    pitch: float
    shape: Tuple[int, int, int]

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.shape))

    def centres(self, ijk: np.ndarray) -> np.ndarray:
        return self.origin + np.asarray(ijk, dtype=np.float64) * self.pitch

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lo = self.origin - 0.5 * self.pitch
        return lo, lo + np.asarray(self.shape, dtype=np.float64) * self.pitch

@dataclass
class MoldLayout:

    lattice: Lattice
    mold: np.ndarray
    auto_core: np.ndarray
    gap: np.ndarray
    shaft: np.ndarray
    from_parts: bool = False
    warnings: List[str] = field(default_factory=list)
    cell_index: Optional[np.ndarray] = None
    mesh_ijk: Optional[np.ndarray] = None
    added_core_voxels: int = 0
    wall_in: float = 0.0
    grading: Optional["MoldGrading"] = None

    @property
    def mold_cell_count(self) -> int:
        return int(np.count_nonzero(self.mold))

    @property
    def auto_core_cell_count(self) -> int:
        return int(np.count_nonzero(self.auto_core))

    def summary(self) -> Dict[str, float]:
        lo, hi = self.lattice.bounds()
        out = {
            "pitch_in": float(self.lattice.pitch),
            "mold_cells": self.mold_cell_count,
            "auto_core_cells": self.auto_core_cell_count,
            "shaft_cells": int(np.count_nonzero(self.shaft)),
            "gap_cells": int(np.count_nonzero(self.gap)),
            "wall_in": float(self.wall_in),
            "box_lo_in": [float(v) for v in lo],
            "box_hi_in": [float(v) for v in hi],
        }
        if self.grading is not None:
            out["graded_mold_cells"] = self.grading.n_cells
            out["max_cell_span"] = self.grading.max_span
        return out

AUTO_WALL_SECTIONS = 3.0
AUTO_WALL_MIN_IN = 0.75
AUTO_WALL_MAX_IN = 8.0

def auto_wall_thickness(section_radius_in: float) -> float:
    return float(np.clip(AUTO_WALL_SECTIONS * float(section_radius_in), AUTO_WALL_MIN_IN, AUTO_WALL_MAX_IN))

def heaviest_section_radius(parts: Sequence[SetupPart], max_cells: int = 250_000) -> float:
    metal = [(m, c, o) for m, c, o in parts if c in METAL_PART_CODES]
    if not metal:
        return 0.0
    lo = np.min([m.bounds[0] for m, _, _ in metal], axis=0)
    hi = np.max([m.bounds[1] for m, _, _ in metal], axis=0)
    size = np.maximum(hi - lo, 1e-6)
    pitch = float(np.max(size)) / 60.0
    while np.prod(np.ceil(size / pitch) + 3) > max_cells:
        pitch *= 1.15
    lattice, _ = box_lattice(lo - pitch, hi + pitch, pitch)
    labels, _ = rasterize_parts(metal, lattice)
    inside = labels >= 0
    if not inside.any():
        return 0.0
    distance = ndimage.distance_transform_edt(inside)
    return float(max(np.max(distance) - 0.5, 0.5) * pitch)

def with_wall(params: MoldParams, section_radius_in: Optional[float] = None,
              parts: Sequence[SetupPart] = ()) -> MoldParams:
    if params.wall_thickness_in > 0.0:
        return params
    if section_radius_in is None:
        section_radius_in = heaviest_section_radius(parts)
    return dataclasses.replace(params, wall_thickness_in=auto_wall_thickness(section_radius_in))

def _open_top(code: int, open_flag: bool) -> bool:
    return code in ALWAYS_OPEN_TOP_CODES or (open_flag and code in FEEDER_PART_CODES)

def mold_box(parts: Sequence[SetupPart], params: MoldParams) -> Tuple[np.ndarray, np.ndarray]:
    if not parts:
        raise MoldSetupError("No parts to build a mold around")
    lo = np.min([np.asarray(m.bounds[0], dtype=np.float64) for m, _, _ in parts], axis=0)
    hi = np.max([np.asarray(m.bounds[1], dtype=np.float64) for m, _, _ in parts], axis=0)
    params = with_wall(params, parts=parts)
    wall = float(params.wall_thickness_in)
    cover = float(params.closed_riser_cover_in)

    open_tops: List[float] = []
    covered = [-np.inf]
    walled = [-np.inf]
    for mesh, code, open_flag in parts:
        top = float(mesh.bounds[1][2])
        if _open_top(code, open_flag):
            open_tops.append(top)
        elif code == PART_TYPE_SPRUE:
            covered.append(top - cover)
        elif code in FEEDER_PART_CODES:
            covered.append(top)
        else:
            covered.append(top)
            walled.append(top)
    if open_tops:
        z_top = max(max(open_tops), max(covered) + cover)
    else:
        z_top = max(max(walled) + wall, max(covered) + cover)
    if not np.isfinite(z_top):
        z_top = hi[2] + wall

    box_lo = lo - wall
    box_hi = np.array([hi[0] + wall, hi[1] + wall, z_top], dtype=np.float64)
    return box_lo, box_hi

def box_lattice(lo: np.ndarray, hi: np.ndarray, pitch: float, anchor: Optional[np.ndarray] = None) -> Tuple[Lattice, np.ndarray]:
    pitch = float(pitch)
    if anchor is None:
        anchor = np.asarray(lo, dtype=np.float64) + 0.5 * pitch
    anchor = np.asarray(anchor, dtype=np.float64)
    first = np.ceil((np.asarray(lo) - anchor) / pitch - 1e-9).astype(np.int64)
    last = np.floor((np.asarray(hi) - anchor) / pitch + 1e-9).astype(np.int64)
    shape = tuple(int(n) for n in np.maximum(last - first + 1, 1))
    origin = anchor + first * pitch
    return Lattice(origin=origin, pitch=pitch, shape=shape), first

def rasterize_mesh(mesh: trimesh.Trimesh, lattice: Lattice) -> np.ndarray:
    from core.cut_cells import winding_number_contains

    inside = np.zeros(lattice.shape, dtype=bool)
    if mesh is None or len(mesh.faces) == 0:
        return inside
    i0 = np.maximum(np.floor((mesh.bounds[0] - lattice.origin) / lattice.pitch).astype(int), 0)
    i1 = np.minimum(np.ceil((mesh.bounds[1] - lattice.origin) / lattice.pitch).astype(int) + 1, lattice.shape)
    if np.any(i1 <= i0):
        return inside
    idx = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(i0, i1)], indexing="ij"), -1).reshape(-1, 3)
    chunk = 200_000
    for start in range(0, len(idx), chunk):
        block = idx[start:start + chunk]
        hit = np.asarray(winding_number_contains(mesh, lattice.centres(block)), dtype=bool)
        sel = block[hit]
        inside[sel[:, 0], sel[:, 1], sel[:, 2]] = True
    return inside

def rasterize_parts(parts: Sequence[SetupPart], lattice: Lattice) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.full(lattice.shape, -1, dtype=np.int16)
    open_mask = np.zeros(lattice.shape, dtype=bool)
    for mesh, code, open_flag in sorted(parts, key=lambda p: p[1]):
        inside = rasterize_mesh(mesh, lattice)
        labels[inside] = code
        if _open_top(code, open_flag):
            open_mask |= inside
    open_mask &= labels >= 0
    return labels, open_mask

def _shift(mask: np.ndarray, axis: int, step: int) -> np.ndarray:
    out = np.zeros_like(mask)
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    if step > 0:
        src[axis] = slice(step, None)
        dst[axis] = slice(None, -step)
    else:
        src[axis] = slice(None, step)
        dst[axis] = slice(-step, None)
    out[tuple(dst)] = mask[tuple(src)]
    return out

def _open_shafts(blocked: np.ndarray, open_mask: np.ndarray) -> np.ndarray:
    nz = blocked.shape[2]
    z = np.arange(nz)
    has_open = open_mask.any(axis=2)
    top = np.where(has_open, nz - 1 - np.argmax(open_mask[:, :, ::-1], axis=2), nz)
    above = z[None, None, :] > top[:, :, None]
    stopped = np.logical_or.accumulate(blocked & above, axis=2)
    return above & ~stopped & ~blocked

def _gaps(blocked: np.ndarray) -> np.ndarray:
    gap = np.zeros_like(blocked)
    for axis in range(3):
        gap |= _shift(blocked, axis, 1) & _shift(blocked, axis, -1)
    return gap & ~blocked

def _enclosed(open_space: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(open_space)
    if n == 0:
        return np.zeros_like(open_space)
    border = np.zeros(open_space.shape, dtype=bool)
    border[[0, -1], :, :] = True
    border[:, [0, -1], :] = True
    border[:, :, [0, -1]] = True
    outside = np.unique(labels[border & open_space])
    return open_space & ~np.isin(labels, outside)

def _undrawable(blocked: np.ndarray) -> np.ndarray:
    above = np.logical_or.accumulate(blocked[:, :, ::-1], axis=2)[:, :, ::-1]
    below = np.logical_or.accumulate(blocked, axis=2)
    return _shift(above, 2, 1) & _shift(below, 2, -1)

def plan_layout(
    labels: np.ndarray,
    open_mask: np.ndarray,
    lattice: Lattice,
    params: MoldParams,
    mold_inside: Optional[np.ndarray] = None,
) -> MoldLayout:
    blocked = labels >= 0
    shaft = _open_shafts(blocked, open_mask)
    gap = _gaps(blocked)
    sand = ~blocked & ~shaft & ~gap

    auto_core = np.zeros_like(blocked)
    if params.auto_cores:
        auto_core = _enclosed(~blocked & ~gap)
        if params.core_rule_code != CORE_RULE_ENCLOSED:
            auto_core |= _undrawable(blocked)
        auto_core &= sand
    mold = sand & ~auto_core
    if mold_inside is not None:
        mold &= mold_inside
    return MoldLayout(
        lattice=lattice,
        mold=mold,
        auto_core=auto_core,
        gap=gap,
        shaft=shaft,
        from_parts=mold_inside is not None,
    )

def _check_cell_budget(lattice: Lattice, params: MoldParams) -> None:
    if lattice.n_cells > params.max_mold_cells:
        raise MoldSetupError(
            f"The mold box needs {lattice.n_cells:,} lattice cells at {lattice.pitch:.3g} in, "
            f"over the {params.max_mold_cells:,} limit. Use a larger voxel size or a thinner "
            f"mold wall (now {params.wall_thickness_in:g} in)."
        )

def layout_box(
    parts: Sequence[SetupPart],
    mold_meshes: Sequence[trimesh.Trimesh],
    params: MoldParams,
) -> Tuple[np.ndarray, np.ndarray]:
    if mold_meshes:
        lo = np.min([m.bounds[0] for m in mold_meshes], axis=0)
        hi = np.max([m.bounds[1] for m in mold_meshes], axis=0)
        if parts:
            lo = np.minimum(lo, np.min([m.bounds[0] for m, _, _ in parts], axis=0))
            hi = np.maximum(hi, np.max([m.bounds[1] for m, _, _ in parts], axis=0))
        return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    return mold_box(parts, params)

def auto_cores_apply(params: MoldParams, parts: Sequence[SetupPart]) -> bool:
    return bool(params.auto_cores) and not any(code == PART_TYPE_CORE for _, code, _ in parts)

def _effective(params: MoldParams, parts: Sequence[SetupPart]) -> MoldParams:
    if params.auto_cores and not auto_cores_apply(params, parts):
        return dataclasses.replace(params, auto_cores=0)
    return params

def mold_requested(params: MoldParams, mold_meshes: Sequence[trimesh.Trimesh]) -> bool:
    return bool(mold_meshes) or bool(params.auto_mold)

def preview_layout(
    parts: Sequence[SetupPart],
    mold_meshes: Sequence[trimesh.Trimesh],
    params: MoldParams,
    pitch: float,
) -> MoldLayout:
    params = _effective(params, parts)
    lo, hi = layout_box(parts, mold_meshes, params)
    lattice, _ = box_lattice(lo, hi, pitch)
    _check_cell_budget(lattice, params)
    labels, open_mask = rasterize_parts(parts, lattice)
    mold_inside = None
    if mold_meshes:
        mold_inside = np.zeros(lattice.shape, dtype=bool)
        for mesh in mold_meshes:
            mold_inside |= rasterize_mesh(mesh, lattice)
    elif not params.auto_mold:
        mold_inside = np.zeros(lattice.shape, dtype=bool)
    layout = plan_layout(labels, open_mask, lattice, params, mold_inside)
    if not mold_meshes and not params.auto_mold:
        layout.mold[:] = False
    layout.wall_in = 0.0 if mold_meshes else float(params.wall_thickness_in)
    if params.mold_grading_levels > 0 and layout.mold.any():
        layout.cell_index = np.where((labels >= 0) | layout.auto_core, 0, -1).astype(np.int32)
        grade_mold(layout, params.mold_grading_levels, params.mold_fine_band_cells)
    return layout

def preview_pitch(parts: Sequence[SetupPart], mold_meshes: Sequence[trimesh.Trimesh], params: MoldParams,
                  voxel_size: Optional[float] = None, max_cells: int = 2_000_000) -> float:
    lo, hi = layout_box(parts, mold_meshes, params)
    size = np.maximum(hi - lo, 1e-6)
    pitch = float(voxel_size) if voxel_size and voxel_size > 0 else float(np.max(size)) / 120.0
    while np.prod(np.ceil(size / pitch) + 1) > max_cells:
        pitch *= 1.15
    return pitch

def mesh_lattice_indices(centers: np.ndarray, pitch: float) -> Tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(centers, dtype=np.float64)
    anchor = centers[0].copy()
    rel = (centers - anchor) / float(pitch)
    ijk = np.rint(rel).astype(np.int64)
    off = float(np.max(np.abs(rel - ijk))) if len(rel) else 0.0
    if off > LATTICE_TOLERANCE:
        raise MoldSetupError(
            f"Voxels are not on one lattice (largest offset {off:.3f} pitch); the mold needs a uniform grid."
        )
    return anchor, ijk

def layout_for_voxel_mesh(
    voxel_mesh,
    parts: Sequence[SetupPart],
    mold_meshes: Sequence[trimesh.Trimesh],
    params: MoldParams,
) -> MoldLayout:
    params = _effective(params, parts)
    centers = np.asarray(voxel_mesh.voxel_centers, dtype=np.float64)
    pitch = float(voxel_mesh.base_voxel_size)
    anchor, ijk = mesh_lattice_indices(centers, pitch)
    lo, hi = layout_box(parts, mold_meshes, params)
    lo = np.minimum(lo, centers.min(axis=0) - pitch)
    hi = np.maximum(hi, centers.max(axis=0) + pitch)
    lattice, first = box_lattice(lo, hi, pitch, anchor)
    _check_cell_budget(lattice, params)

    local = ijk - first
    labels = np.full(lattice.shape, -1, dtype=np.int16)
    raw_types = getattr(voxel_mesh, "voxel_part_types", None)
    part_types = (
        np.full(len(centers), PART_TYPE_CAST, dtype=np.int16)
        if raw_types is None
        else np.asarray(raw_types, dtype=np.int16)
    )
    labels[local[:, 0], local[:, 1], local[:, 2]] = part_types

    open_parts = [p for p in parts if _open_top(p[1], p[2])]
    open_mask = np.zeros(lattice.shape, dtype=bool)
    if open_parts:
        _, open_mask = rasterize_parts(open_parts, lattice)
        open_mask &= np.isin(labels, (PART_TYPE_RISER, PART_TYPE_POUR_CUP, *FEEDER_PART_CODES))

    mold_inside = None
    if mold_meshes:
        mold_inside = np.zeros(lattice.shape, dtype=bool)
        for mesh in mold_meshes:
            mold_inside |= rasterize_mesh(mesh, lattice)
    elif not params.auto_mold:
        mold_inside = np.zeros(lattice.shape, dtype=bool)
    layout = plan_layout(labels, open_mask, lattice, params, mold_inside)
    if not mold_meshes and not params.auto_mold:
        layout.mold[:] = False
    layout.wall_in = 0.0 if mold_meshes else float(params.wall_thickness_in)
    layout.cell_index = np.full(lattice.shape, -1, dtype=np.int32)
    layout.cell_index[local[:, 0], local[:, 1], local[:, 2]] = np.arange(len(centers), dtype=np.int32)
    layout.mesh_ijk = local.astype(np.int32)
    return layout

@dataclass
class OccupancyEstimate:

    voxels: int
    mold_cells: int
    lattice_cells: int
    layout: MoldLayout

def estimate_from_occupancy(
    parts: Sequence[SetupPart],
    mold_meshes: Sequence[trimesh.Trimesh],
    params: MoldParams,
    occupied: np.ndarray,
    origin_index: np.ndarray,
    pitch: float,
) -> OccupancyEstimate:
    params = _effective(params, parts)
    pitch = float(pitch)
    occupied = np.asarray(occupied, dtype=bool)
    ijk = np.argwhere(occupied).astype(np.int64)
    if not len(ijk):
        raise MoldSetupError("The assembly has no occupied cells at this voxel size")
    anchor = np.asarray(origin_index, dtype=np.float64) * pitch
    centers = anchor + ijk * pitch
    if params.wall_thickness_in <= 0.0:
        depth = ndimage.distance_transform_edt(np.pad(occupied, 1))
        params = with_wall(params, (float(depth.max()) - 0.5) * pitch)
    lo, hi = layout_box(parts, mold_meshes, params)
    lo = np.minimum(lo, centers.min(axis=0) - pitch)
    hi = np.maximum(hi, centers.max(axis=0) + pitch)
    lattice, first = box_lattice(lo, hi, pitch, anchor)
    _check_cell_budget(lattice, params)
    local = ijk - first
    del ijk, centers

    labels = np.full(lattice.shape, -1, dtype=np.int16)
    labels[local[:, 0], local[:, 1], local[:, 2]] = PART_TYPE_CAST
    open_parts = [p for p in parts if _open_top(p[1], p[2])]
    open_mask = np.zeros(lattice.shape, dtype=bool)
    if open_parts:
        _, open_mask = rasterize_parts(open_parts, lattice)
        open_mask &= labels >= 0
    mold_inside = None
    if mold_meshes:
        mold_inside = np.zeros(lattice.shape, dtype=bool)
        for mesh in mold_meshes:
            mold_inside |= rasterize_mesh(mesh, lattice)
    elif not params.auto_mold:
        mold_inside = np.zeros(lattice.shape, dtype=bool)
    layout = plan_layout(labels, open_mask, lattice, params, mold_inside)
    del labels, open_mask
    if not mold_meshes and not params.auto_mold:
        layout.mold[:] = False
    layout.wall_in = 0.0 if mold_meshes else float(params.wall_thickness_in)
    layout.cell_index = np.full(lattice.shape, -1, dtype=np.int32)
    layout.cell_index[local[:, 0], local[:, 1], local[:, 2]] = 0
    layout.cell_index[layout.auto_core] = 0
    voxels = int(len(local)) + layout.auto_core_cell_count
    lattice_cells = layout.mold_cell_count
    mold_cells = lattice_cells
    if params.mold_grading_levels > 0 and lattice_cells:
        mold_cells = grade_mold(layout, params.mold_grading_levels, params.mold_fine_band_cells).n_cells
    return OccupancyEstimate(voxels=voxels, mold_cells=int(mold_cells), lattice_cells=int(lattice_cells), layout=layout)

def _lookup(grid: np.ndarray, ijk: np.ndarray) -> np.ndarray:
    shape = np.asarray(grid.shape)
    ok = np.all((ijk >= 0) & (ijk < shape), axis=1)
    out = np.full(len(ijk), -1, dtype=np.int64)
    sel = ijk[ok]
    out[ok] = grid[sel[:, 0], sel[:, 1], sel[:, 2]]
    return out

def _lookup_mask(mask: np.ndarray, ijk: np.ndarray, outside: bool = False) -> np.ndarray:
    shape = np.asarray(mask.shape)
    ok = np.all((ijk >= 0) & (ijk < shape), axis=1)
    out = np.full(len(ijk), outside, dtype=bool)
    sel = ijk[ok]
    out[ok] = mask[sel[:, 0], sel[:, 1], sel[:, 2]]
    return out

def _resize_managed(voxel_mesh, name: str, n_new: int, fill) -> None:
    from core.memory_manager import create_managed_array

    old = getattr(voxel_mesh, name, None)
    if old is None:
        return
    old = np.asarray(old)
    shape = (n_new,) + old.shape[1:]
    new = create_managed_array(shape if len(shape) > 1 else n_new, dtype=old.dtype, name=name,
                               manager=voxel_mesh.memory_manager)
    new[: len(old)] = old
    new[len(old):] = fill
    setattr(voxel_mesh, name, new)

def add_auto_cores(voxel_mesh, layout: MoldLayout) -> int:
    if layout.cell_index is None or layout.mesh_ijk is None:
        raise MoldSetupError("Layout is not attached to a voxel mesh")
    new_ijk = np.argwhere(layout.auto_core).astype(np.int64)
    n_new = len(new_ijk)
    if n_new == 0:
        return 0
    n_old = len(voxel_mesh.voxel_centers)
    n_total = n_old + n_new
    pitch = layout.lattice.pitch
    new_index = np.arange(n_old, n_total, dtype=np.int64)
    grid = layout.cell_index.astype(np.int64)
    grid[new_ijk[:, 0], new_ijk[:, 1], new_ijk[:, 2]] = new_index
    old_ijk = layout.mesh_ijk.astype(np.int64)
    all_ijk = np.concatenate([old_ijk, new_ijk])

    old_grid = layout.cell_index
    face_area = pitch * pitch
    exposed_before = np.zeros(n_old, dtype=np.int64)
    covered_now = np.zeros(n_old, dtype=np.int64)
    for slot in FACE_SLOTS:
        probe = old_ijk + NEIGHBOR_SLOT_OFFSETS[slot]
        was_empty = _lookup(old_grid, probe) < 0
        exposed_before += was_empty
        covered_now += was_empty & _lookup_mask(layout.auto_core, probe)

    if getattr(voxel_mesh, "voxel_part_types", None) is None:
        from core.memory_manager import create_managed_array

        voxel_mesh.voxel_part_types = create_managed_array(
            n_old, dtype=np.int8, name="voxel_part_types", manager=voxel_mesh.memory_manager
        )
        voxel_mesh.voxel_part_types[:] = PART_TYPE_CAST
    for name, fill in (
        ("voxel_centers", 0.0),
        ("voxel_sizes", pitch),
        ("voxel_volumes", pitch ** 3),
        ("voxel_surface_areas", 0.0),
        ("voxel_moduli", 0.0),
        ("voxel_states", 0.0),
        ("surface_voxels", False),
        ("voxel_part_types", PART_TYPE_CORE),
        ("voxel_metal_fractions", 1.0),
    ):
        _resize_managed(voxel_mesh, name, n_total, fill)
    voxel_mesh.voxel_centers[n_old:] = layout.lattice.centres(new_ijk).astype(np.float32)

    touched = np.flatnonzero(covered_now > 0)
    if len(touched):
        areas = np.asarray(voxel_mesh.voxel_surface_areas, dtype=np.float64)
        share = covered_now[touched] / np.maximum(exposed_before[touched], 1)
        areas[touched] = np.maximum(areas[touched] * (1.0 - share), 0.0)
        voxel_mesh.voxel_surface_areas[:] = areas.astype(np.float32)

    new_exposed = np.zeros(n_new, dtype=np.int64)
    for slot in FACE_SLOTS:
        new_exposed += _lookup(grid, new_ijk + NEIGHBOR_SLOT_OFFSETS[slot]) < 0
    voxel_mesh.voxel_surface_areas[n_old:] = (new_exposed * face_area).astype(np.float32)
    voxel_mesh.surface_voxels[:] = np.asarray(voxel_mesh.voxel_surface_areas) > 0.0

    old_neighbors = np.asarray(voxel_mesh.voxel_neighbors)
    _resize_managed(voxel_mesh, "voxel_neighbors", n_total, -1)
    neighbors = voxel_mesh.voxel_neighbors
    for slot, offset in enumerate(NEIGHBOR_SLOT_OFFSETS):
        neighbors[n_old:, slot] = _lookup(grid, new_ijk + offset).astype(np.int32)
        target = _lookup(grid, old_ijk + offset)
        hit = np.flatnonzero(target >= n_old)
        if len(hit) == 0:
            continue
        displaced = old_neighbors[hit, slot]
        for row, other in zip(hit[displaced >= 0], displaced[displaced >= 0]):
            back = np.flatnonzero(neighbors[other] == row)
            neighbors[other, back] = -1
        neighbors[hit, slot] = target[hit].astype(np.int32)

    layout.cell_index = grid.astype(np.int32)
    layout.mesh_ijk = all_ijk.astype(np.int32)
    layout.auto_core[:] = False
    layout.grading = None
    layout.added_core_voxels += n_new
    voxel_mesh._material_surface_indices = None
    voxel_mesh._chill_surface_indices = None
    voxel_mesh._spatial_index = None
    return n_new

def _mesh_section_radius(voxel_mesh) -> float:
    moduli = getattr(voxel_mesh, "voxel_moduli", None)
    if moduli is None or len(moduli) == 0:
        return 0.0
    moduli = np.asarray(moduli, dtype=np.float64)
    raw_types = getattr(voxel_mesh, "voxel_part_types", None)
    if raw_types is not None:
        metal = np.isin(np.asarray(raw_types), METAL_PART_CODES)
        if metal.any():
            moduli = moduli[metal]
    return float(np.max(moduli) + 0.5 * float(voxel_mesh.base_voxel_size))

def prepare_mold(
    voxel_mesh,
    parts: Sequence[SetupPart],
    mold_meshes: Sequence[trimesh.Trimesh],
    params: MoldParams,
) -> Optional[MoldLayout]:
    voxel_mesh.mold_layout = None
    if not (mold_requested(params, mold_meshes) or auto_cores_apply(params, parts)):
        return None
    if voxel_mesh.voxel_centers is None or len(voxel_mesh.voxel_centers) == 0:
        return None
    if params.wall_thickness_in <= 0.0:
        params = with_wall(params, _mesh_section_radius(voxel_mesh))
    layout = layout_for_voxel_mesh(voxel_mesh, parts, mold_meshes, params)
    added = add_auto_cores(voxel_mesh, layout)
    if added:
        rule = "enclosed" if params.core_rule_code == CORE_RULE_ENCLOSED else "draw"
        print(f"  Mold stage: {added:,} auto-core voxels added ({rule} rule)")
    if layout.mold_cell_count > 0:
        voxel_mesh.mold_layout = layout
        source = "mold parts" if layout.from_parts else f"auto box, {params.wall_thickness_in:g} in walls"
        if params.mold_grading_levels > 0:
            grading = grade_mold(layout, params.mold_grading_levels, params.mold_fine_band_cells)
            print(f"  Mold stage: {grading.describe(layout.mold_cell_count)} ({source})")
        else:
            print(f"  Mold stage: {layout.mold_cell_count:,} mold sand cells ({source})")
    return layout

@dataclass
class MoldGrading:

    levels: int
    band: int
    ijk: np.ndarray
    span: np.ndarray
    owner: np.ndarray

    @property
    def n_cells(self) -> int:
        return len(self.span)

    @property
    def max_span(self) -> int:
        return int(self.span.max()) if len(self.span) else 1

    def describe(self, lattice_cells: int) -> str:
        if self.max_span <= 1:
            return f"{self.n_cells:,} mold sand cells (none coarsened)"
        return (
            f"{self.n_cells:,} mold sand cells, graded from {lattice_cells:,} lattice cells, "
            f"up to {self.max_span}x"
        )

def grading_thresholds(levels: int, band: int, transition: Optional[int] = None) -> List[int]:
    if transition is None:
        transition = GRADING_TRANSITION_CELLS
    out: List[int] = []
    t = int(band)
    for level in range(1, int(levels) + 1):
        if level > 1:
            t += int(transition) * 2 ** (level - 1)
        out.append(t)
    return out

def _mirror_shift(n: int, block: int) -> int:
    return int((-(int(n) // 2)) % int(block))

def _upsample(blocks: np.ndarray, factor: int) -> np.ndarray:
    out = blocks
    for axis in range(3):
        out = np.repeat(out, factor, axis=axis)
    return out

def grade_mold(layout: MoldLayout, levels: int, band: int) -> MoldGrading:
    levels = int(np.clip(int(levels), 0, 5))
    band = max(int(band), 2)
    cached = layout.grading
    if cached is not None and cached.levels == levels and cached.band == band:
        return cached
    mold = layout.mold
    shape = np.asarray(mold.shape, dtype=np.int64)
    seeds = np.array(layout.gap, dtype=bool, copy=True)
    if layout.cell_index is not None:
        seeds |= layout.cell_index >= 0

    block = 2 ** levels
    shift = np.array([_mirror_shift(n, block) for n in shape], dtype=np.int64)
    padded = ((shape + shift + block - 1) // block) * block
    crop = tuple(slice(int(s), int(s + n)) for s, n in zip(shift, shape))

    min_dist = np.zeros(tuple(int(v) for v in padded), dtype=np.int32)
    if levels > 0 and seeds.any() and mold.any():
        dist = ndimage.distance_transform_cdt(~seeds, metric="chessboard")
        np.copyto(min_dist[crop], dist, where=mold)
        del dist
    elif levels > 0 and mold.any():
        np.copyto(min_dist[crop], np.iinfo(np.int32).max, where=mold)

    thresholds = grading_thresholds(levels, band)
    qualifies: List[np.ndarray] = [np.zeros(0, dtype=bool)]
    d = min_dist
    for level in range(1, levels + 1):
        b = d.shape
        d = d.reshape(b[0] // 2, 2, b[1] // 2, 2, b[2] // 2, 2).min(axis=(1, 3, 5))
        qualifies.append(d > thresholds[level - 1])
    del min_dist, d

    assigned: List[np.ndarray] = [np.zeros(0, dtype=bool)] * (levels + 1)
    for level in range(levels, 0, -1):
        cell = qualifies[level]
        if level < levels:
            cell = cell & ~_upsample(qualifies[level + 1], 2)
        assigned[level] = cell

    if levels > 0:
        covered = _upsample(qualifies[1], 2)[crop]
        fine = mold & ~covered
        del covered
    else:
        fine = mold
    ijk_parts = [np.argwhere(fine).astype(np.int64)]
    span_parts = [np.ones(len(ijk_parts[0]), dtype=np.int64)]
    for level in range(1, levels + 1):
        size = 2 ** level
        corners = np.argwhere(assigned[level]).astype(np.int64) * size - shift
        ijk_parts.append(corners)
        span_parts.append(np.full(len(corners), size, dtype=np.int64))
    ijk = np.concatenate(ijk_parts)
    span = np.concatenate(span_parts)

    owner_pad = np.full(tuple(int(v) for v in padded), -1, dtype=np.int32)
    n_fine = len(ijk_parts[0])
    fine_pad = ijk_parts[0] + shift
    owner_pad[fine_pad[:, 0], fine_pad[:, 1], fine_pad[:, 2]] = np.arange(n_fine, dtype=np.int32)
    start = n_fine
    for level in range(1, levels + 1):
        count = len(ijk_parts[level])
        if count == 0:
            continue
        size = 2 ** level
        ids = np.full(assigned[level].shape, -1, dtype=np.int32)
        ids[assigned[level]] = np.arange(start, start + count, dtype=np.int32)
        p = owner_pad.shape
        view = owner_pad.reshape(p[0] // size, size, p[1] // size, size, p[2] // size, size)
        expand = (slice(None), None, slice(None), None, slice(None), None)
        np.copyto(view, ids[expand], where=assigned[level][expand])
        start += count
    owner = np.ascontiguousarray(owner_pad[crop])
    grading = MoldGrading(levels=levels, band=band, ijk=ijk, span=span, owner=owner)
    layout.grading = grading
    return grading

def _cross_size_links(owner: np.ndarray, span: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(span)
    level = np.full(owner.shape, -1, dtype=np.int8)
    inside = owner >= 0
    level[inside] = np.log2(span[owner[inside]]).astype(np.int8)
    keys: List[np.ndarray] = []
    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(None, -1)
        hi[axis] = slice(1, None)
        la, lb = level[tuple(lo)], level[tuple(hi)]
        cross = (la >= 0) & (lb >= 0) & (la != lb)
        if not cross.any():
            continue
        a = owner[tuple(lo)][cross].astype(np.int64)
        b = owner[tuple(hi)][cross].astype(np.int64)
        finer = np.where(span[a] < span[b], a, b)
        coarser = np.where(span[a] < span[b], b, a)
        keys.append(finer * n + coarser)
    if not keys:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    unique, counts = np.unique(np.concatenate(keys), return_counts=True)
    return unique // n, unique % n, counts.astype(np.int64)

def _graded_air_faces(grid: np.ndarray, gap: np.ndarray, owner: np.ndarray, n_cells: int) -> np.ndarray:
    mold = owner >= 0
    solid = (grid >= 0) | gap
    faces = np.zeros(owner.shape, dtype=np.uint8)
    for axis in range(3):
        for step in (1, -1):
            faces += mold & ~_shift(solid, axis, step)
    return np.bincount(owner[mold], weights=faces[mold], minlength=n_cells).astype(np.int64)

@dataclass
class ThermalExtension:

    n_mesh: int
    mold_ijk: np.ndarray
    mesh_neighbors: np.ndarray
    mold_neighbors: np.ndarray
    pair_first: np.ndarray
    pair_second: np.ndarray
    pair_area: np.ndarray
    stencil_first: np.ndarray
    stencil_second: np.ndarray
    stencil_slot: np.ndarray
    air_mesh_cells: np.ndarray
    air_mesh_area: np.ndarray
    air_mold_cells: np.ndarray
    air_mold_area: np.ndarray
    unassigned_area: float = 0.0
    mold_span: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    link_first: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    link_second: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    link_area: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    lattice_mold_cells: int = 0

    @property
    def n_mold(self) -> int:
        return len(self.mold_ijk)

    @property
    def graded(self) -> bool:
        return bool(len(self.mold_span)) and int(self.mold_span.max()) > 1

    @property
    def max_span(self) -> int:
        return int(self.mold_span.max()) if len(self.mold_span) else 1

def _symmetric(neighbors: np.ndarray) -> None:
    rows, slots = np.nonzero(neighbors >= 0)
    cols = neighbors[rows, slots]
    has_back = np.any(neighbors[cols] == rows[:, None], axis=1)
    neighbors[rows[~has_back], slots[~has_back]] = -1

def build_thermal_extension(
    voxel_mesh,
    layout: MoldLayout,
    grading_levels: int = 0,
    fine_band_cells: int = 4,
) -> ThermalExtension:
    if layout.cell_index is None or layout.mesh_ijk is None:
        raise MoldSetupError("Layout is not attached to a voxel mesh")
    n_mesh = len(voxel_mesh.voxel_centers)
    mesh_ijk = layout.mesh_ijk.astype(np.int64)
    pitch = layout.lattice.pitch
    face_area = pitch * pitch
    lattice_mold_cells = layout.mold_cell_count

    grid = layout.cell_index.astype(np.int64)
    grading = grade_mold(layout, grading_levels, fine_band_cells) if int(grading_levels) > 0 else None
    if grading is None:
        mold_ijk = np.argwhere(layout.mold).astype(np.int64)
        n_mold = len(mold_ijk)
        span = np.ones(n_mold, dtype=np.int64)
        grid[mold_ijk[:, 0], mold_ijk[:, 1], mold_ijk[:, 2]] = n_mesh + np.arange(n_mold)
    else:
        mold_ijk = grading.ijk
        n_mold = grading.n_cells
        span = grading.span
        in_mold = grading.owner >= 0
        grid[in_mold] = n_mesh + grading.owner[in_mold]
        del in_mold
    raw_types = getattr(voxel_mesh, "voxel_part_types", None)
    part_types = np.full(n_mesh, PART_TYPE_CAST) if raw_types is None else np.asarray(raw_types)
    sand_mesh = np.isin(part_types, SAND_PART_CODES)
    solid_mesh = ~sand_mesh

    mesh_neighbors = np.array(voxel_mesh.voxel_neighbors, dtype=np.int32, copy=True)
    rows, slots = np.nonzero(mesh_neighbors >= 0)
    cols = mesh_neighbors[rows, slots].astype(np.int64)
    delta = mesh_ijk[cols] - mesh_ijk[rows]
    long_link = np.max(np.abs(delta), axis=1) > 1
    if np.any(long_link):
        r, sl, d = rows[long_link], slots[long_link], delta[long_link]
        through = np.zeros(len(r), dtype=bool)
        for t in (1.0 / 3.0, 0.5, 2.0 / 3.0):
            probe = mesh_ijk[r] + np.rint(d * t).astype(np.int64)
            through |= _lookup(grid, probe) >= n_mesh
        mesh_neighbors[r[through], sl[through]] = -1
    _symmetric(mesh_neighbors)

    mold_neighbors = np.full((n_mold, 26), -1, dtype=np.int32)
    stencil_first: List[np.ndarray] = []
    stencil_second: List[np.ndarray] = []
    stencil_slot: List[np.ndarray] = []
    core_rows = np.flatnonzero(sand_mesh)
    for slot, offset in enumerate(NEIGHBOR_SLOT_OFFSETS):
        if grading is None:
            target = _lookup(grid, mold_ijk + offset)
            mold_neighbors[:, slot] = np.where(target >= n_mesh, target - n_mesh, -1).astype(np.int32)
        else:
            local = _lookup(grid, mold_ijk + span[:, None] * offset) - n_mesh
            same = local >= 0
            same[same] = span[local[same]] == span[same]
            mold_neighbors[:, slot] = np.where(same, local, -1).astype(np.int32)
        if len(core_rows):
            target = _lookup(grid, mesh_ijk[core_rows] + offset)
            hit = target >= n_mesh
            stencil_first.append(core_rows[hit])
            stencil_second.append(target[hit] - n_mesh)
            stencil_slot.append(np.full(int(hit.sum()), slot, dtype=np.int64))

    exterior = (
        np.asarray(voxel_mesh.voxel_surface_areas, dtype=np.float64)
        if getattr(voxel_mesh, "cut_cells", False)
        else None
    )
    face_kind = np.zeros((n_mesh, 6), dtype=np.int8)
    face_target = np.full((n_mesh, 6), -1, dtype=np.int64)
    for slot in FACE_SLOTS:
        probe = mesh_ijk + NEIGHBOR_SLOT_OFFSETS[slot]
        target = _lookup(grid, probe)
        face_target[:, slot] = target
        gap = _lookup_mask(layout.gap, probe)
        face_kind[:, slot] = np.where(
            target >= n_mesh, 1, np.where(target >= 0, 0, np.where(gap, 3, 2))
        )
    exposed = np.count_nonzero(face_kind > 0, axis=1)
    if exterior is None:
        per_face = np.where(exposed > 0, face_area, 0.0)
    else:
        per_face = np.divide(exterior, exposed, out=np.zeros(n_mesh), where=exposed > 0)

    pair_first: List[np.ndarray] = []
    pair_second: List[np.ndarray] = []
    pair_area: List[np.ndarray] = []
    for slot in FACE_SLOTS:
        mold_face = (face_kind[:, slot] == 1) & solid_mesh & (per_face > 0.0)
        pair_first.append(np.flatnonzero(mold_face))
        pair_second.append(face_target[mold_face, slot] - n_mesh)
        pair_area.append(per_face[mold_face])
    air_faces_mesh = np.count_nonzero(face_kind == 2, axis=1)
    air_mesh = air_faces_mesh * (face_area if exterior is None else per_face)
    air_mesh = np.where(sand_mesh, air_faces_mesh * face_area, air_mesh)
    air_mesh_cells = np.flatnonzero(air_mesh > 0.0)

    unassigned = 0.0
    if exterior is not None:
        hidden = np.flatnonzero(solid_mesh & (exposed == 0) & (exterior > 0.0))
        if len(hidden):
            diag = np.zeros((len(hidden), 20), dtype=np.int64)
            for n, slot in enumerate(range(6, 26)):
                diag[:, n] = _lookup(grid, mesh_ijk[hidden] + NEIGHBOR_SLOT_OFFSETS[slot])
            diag_mold = diag >= n_mesh
            count = np.count_nonzero(diag_mold, axis=1)
            has = count > 0
            unassigned = float(np.sum(exterior[hidden[~has]]))
            rr, cc = np.nonzero(diag_mold & has[:, None])
            pair_first.append(hidden[rr])
            pair_second.append(diag[rr, cc] - n_mesh)
            pair_area.append(exterior[hidden[rr]] / count[rr])

    if grading is None:
        mold_air = np.zeros(n_mold, dtype=np.int64)
        for slot in FACE_SLOTS:
            probe = mold_ijk + NEIGHBOR_SLOT_OFFSETS[slot]
            mold_air += (_lookup(grid, probe) < 0) & ~_lookup_mask(layout.gap, probe)
        link_first = link_second = np.zeros(0, dtype=np.int64)
        link_faces = np.zeros(0, dtype=np.int64)
    else:
        mold_air = _graded_air_faces(grid, layout.gap, grading.owner, n_mold)
        link_first, link_second, link_faces = _cross_size_links(grading.owner, span)
    air_mold_cells = np.flatnonzero(mold_air > 0)

    def cat(parts: List[np.ndarray], dtype) -> np.ndarray:
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)

    if grading is not None:
        targets = np.concatenate([cat(pair_second, np.int64), cat(stencil_second, np.int64)])
        if len(targets) and int(span[targets].max()) != 1:
            raise AssertionError("graded mold: a mesh-mold pair reaches a coarsened mold cell")

    return ThermalExtension(
        n_mesh=n_mesh,
        mold_ijk=mold_ijk,
        mesh_neighbors=mesh_neighbors,
        mold_neighbors=mold_neighbors,
        pair_first=cat(pair_first, np.int64),
        pair_second=cat(pair_second, np.int64),
        pair_area=cat(pair_area, np.float64),
        stencil_first=cat(stencil_first, np.int64),
        stencil_second=cat(stencil_second, np.int64),
        stencil_slot=cat(stencil_slot, np.int64),
        air_mesh_cells=air_mesh_cells.astype(np.int64),
        air_mesh_area=air_mesh[air_mesh_cells].astype(np.float64),
        air_mold_cells=air_mold_cells.astype(np.int64),
        air_mold_area=(mold_air[air_mold_cells] * face_area).astype(np.float64),
        unassigned_area=unassigned,
        mold_span=span,
        link_first=link_first,
        link_second=link_second,
        link_area=(link_faces * face_area).astype(np.float64),
        lattice_mold_cells=lattice_mold_cells,
    )

def mask_surface(mask: np.ndarray, lattice: Lattice, smooth: int = 8) -> Optional[trimesh.Trimesh]:
    if not np.any(mask):
        return None
    padded = np.pad(mask, 1)
    quads = []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        e_a, e_u, e_v = np.eye(3, dtype=np.int64)[[axis, u, v]]
        for side in (1, -1):
            face = padded & ~_shift(padded, axis, side)
            cells = np.argwhere(face) - 1
            if len(cells) == 0:
                continue
            base = cells + (e_a if side > 0 else 0)
            corners = [base, base + e_u, base + e_u + e_v, base + e_v]
            if side < 0:
                corners = [corners[0], corners[3], corners[2], corners[1]]
            quads.append(np.stack(corners, axis=1))
    quad = np.concatenate(quads)
    corner_ids, inverse = np.unique(quad.reshape(-1, 3), axis=0, return_inverse=True)
    q = inverse.reshape(-1, 4)
    faces = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    vertices = lattice.origin - 0.5 * lattice.pitch + corner_ids * lattice.pitch
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if smooth > 0:
        trimesh.smoothing.filter_taubin(mesh, iterations=smooth)
    return mesh

def preview_meshes(layout: MoldLayout) -> Dict[str, List[trimesh.Trimesh]]:
    out: Dict[str, List[trimesh.Trimesh]] = {"mold": [], "cores": []}
    if layout.mold.any() and not layout.from_parts:
        lo, hi = layout.lattice.bounds()
        box = trimesh.creation.box(extents=hi - lo)
        box.apply_translation(0.5 * (lo + hi))
        out["mold"].append(box)
    if layout.auto_core.any():
        labels, n = ndimage.label(layout.auto_core)
        for body in range(1, n + 1):
            mesh = mask_surface(labels == body, layout.lattice)
            if mesh is not None:
                out["cores"].append(mesh)
    return out

def mesh_to_payload(mesh: trimesh.Trimesh) -> Dict[str, list]:
    return {
        "vertices": np.asarray(mesh.vertices, dtype=np.float32).reshape(-1).tolist(),
        "faces": np.asarray(mesh.faces, dtype=np.int32).reshape(-1).tolist(),
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
    }

def preview(
    parts: Sequence[SetupPart],
    mold_meshes: Sequence[trimesh.Trimesh],
    params: MoldParams,
    voxel_size: Optional[float] = None,
) -> Tuple[MoldLayout, Dict[str, List[trimesh.Trimesh]]]:
    if not mold_meshes:
        params = with_wall(params, parts=parts)
    pitch = preview_pitch(parts, mold_meshes, params, voxel_size)
    layout = preview_layout(parts, mold_meshes, params, pitch)
    return layout, preview_meshes(layout)
