"""Sand mold and core setup on the solver lattice.

The mold stage runs after voxelisation (``core/simulation_pipeline.py``) and
for the setup preview in both front ends. It works on the voxel lattice the
solver uses, so what it decides is exactly what the thermal solve sees:

1. **Mold box.** With mold parts loaded (flask, cope, drag) the mold is their
   interior. Otherwise, with ``MoldParams.auto_mold``, it is a box around the
   assembly, ``wall_thickness_in`` of sand on the sides and bottom (0 picks it
   from the heaviest section, :func:`auto_wall_thickness`). The top
   follows the riser types:

   - a blind (closed) riser is covered by ``closed_riser_cover_in`` of sand;
   - an open riser or a pour cup is open to the air at the top of the mold.
     When the cope must stand higher than an open top (a closed riser or the
     casting needs the cover), the column above the open body is left open
     as a shaft to the cope surface.

   MAGMASOFT builds the mold as a sand cuboid the same way; it has no flask,
   and its outer faces radiate and convect to the air.

2. **Cores.** With ``MoldParams.auto_cores``, sand the mold cannot form is
   core:

   - ``enclosed`` (MAGMASOFT's automatic core): sand the outside of the mold
     cannot reach. A flood fill from the mold's outer faces through sand and
     air leaves cavities sealed by the assembly.
   - ``draw`` (foundry rule, default): a pattern is drawn straight up out of
     the drag and straight down out of the cope, so sand with the assembly
     both above and below it in its column cannot be formed by the mold --
     enclosed cavities, horizontal holes, and undercuts.

   Auto cores become ordinary CORE voxels in the mesh, so every core analysis
   applies to them. Supplied geometry overrides the automatic cores: with
   any core part loaded, no cores are added (:func:`auto_cores_apply`).

3. **Gaps.** A cell with assembly on both sides along an axis is a one-cell
   gap between bodies that the lattice cannot resolve (usually two bodies
   that touch in CAD). It is neither mold nor core: the mesh bridges such
   gaps as contact, and filling them with sand would insulate a chill from
   the casting it is clamped to.

The mold sand never enters the voxel mesh. The thermal solve carries it as
its own model on a slower clock, coupled to the mesh's through contact pairs
(:func:`build_thermal_extension`, ``physics/mold_coupling.py``), so every
mesh-sized result array, analysis and display stays as it was, and the mold
is not drawn in results.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from scipy import ndimage

try:
    from core.error_logger import logger
except ImportError:  # pragma: no cover - headless/dummy fallback
    from core.error_logger_dummy import logger  # type: ignore

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

#: (mesh, part code, is_open_top) per non-mold part, as
#: ``PartManager.get_mold_setup_parts`` returns them.
SetupPart = Tuple[trimesh.Trimesh, int, bool]

#: Lattice offset of each canonical neighbour slot, matching
#: ``AdaptiveVoxelMesh._canonicalize_neighbor_order``: faces +X -X +Y -Y +Z -Z,
#: then the XY, XZ and YZ edges, then the corners.
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

#: A mesh voxel must sit within this fraction of a pitch of a lattice point.
LATTICE_TOLERANCE = 0.05

#: Codes whose top is open to the air regardless of the part's flag.
ALWAYS_OPEN_TOP_CODES = (PART_TYPE_POUR_CUP,)

#: Graded mold (``MoldParams.mold_grading_levels``): cells of each level kept
#: around the next finer level, in cells of the finer level. Level l starts
#: GRADING_TRANSITION_CELLS * 2^(l-1) lattice cells beyond level l-1.
GRADING_TRANSITION_CELLS = 2


class MoldSetupError(ValueError):
    """The mold cannot be built as asked (too many cells, off-lattice voxels)."""


@dataclass
class Lattice:
    """Cells of size ``pitch`` (inches); cell (i, j, k) is centred at ``origin + (i, j, k) * pitch``."""

    origin: np.ndarray
    pitch: float
    shape: Tuple[int, int, int]

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.shape))

    def centres(self, ijk: np.ndarray) -> np.ndarray:
        return self.origin + np.asarray(ijk, dtype=np.float64) * self.pitch

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Outer faces of the lattice, inches."""
        lo = self.origin - 0.5 * self.pitch
        return lo, lo + np.asarray(self.shape, dtype=np.float64) * self.pitch


@dataclass
class MoldLayout:
    """Where the sand goes, on one box-shaped piece of the lattice."""

    lattice: Lattice
    #: Mold sand cells.
    mold: np.ndarray
    #: Core sand the mold stage adds (cells not already a core part).
    auto_core: np.ndarray
    #: One-cell gaps between bodies: neither sand nor air (see module doc).
    gap: np.ndarray
    #: Open shafts over open risers and pour cups.
    shaft: np.ndarray
    #: True when the mold came from mold parts, False for the automatic box.
    from_parts: bool = False
    warnings: List[str] = field(default_factory=list)
    #: Mesh voxel index per lattice cell (-1: none); set by layout_for_voxel_mesh.
    cell_index: Optional[np.ndarray] = None
    #: (n_mesh, 3) lattice index of every mesh voxel; set by layout_for_voxel_mesh.
    mesh_ijk: Optional[np.ndarray] = None
    #: Core voxels add_auto_cores appended to the mesh.
    added_core_voxels: int = 0
    #: Wall of the automatic box, inches (resolved when automatic); 0 for mold parts.
    wall_in: float = 0.0
    #: The graded partition of the mold sand (:func:`grade_mold`), cached by
    #: prepare_mold for the thermal extension; None when not graded yet.
    grading: Optional["MoldGrading"] = None

    @property
    def mold_cell_count(self) -> int:
        """Mold sand cells on the lattice (the uniform mold's cell count)."""
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


# ---------------------------------------------------------------------------
# Mold box
# ---------------------------------------------------------------------------


#: Automatic wall: AUTO_WALL_SECTIONS x the half-thickness of the heaviest
#: metal section, within [AUTO_WALL_MIN_IN, AUTO_WALL_MAX_IN].
#:
#: Sand heats to a depth of about sqrt(alpha t). By Chvorinov, t = B M^2,
#: and for steel in silica sand sqrt(alpha B) is about 1, so the penetration
#: depth at the freeze time is about the modulus M, which is the
#: half-thickness for a plate and less for compact shapes. Three depths
#: (outer face warming < 5 %) is 3 x the half-thickness. A 2 in WCB cube
#: froze at 248 s with 2 in and with 4 in walls.
AUTO_WALL_SECTIONS = 3.0
AUTO_WALL_MIN_IN = 0.75
AUTO_WALL_MAX_IN = 8.0


def auto_wall_thickness(section_radius_in: float) -> float:
    """Automatic mold wall (inches) for a heaviest section of half-thickness ``section_radius_in``."""
    return float(np.clip(AUTO_WALL_SECTIONS * float(section_radius_in), AUTO_WALL_MIN_IN, AUTO_WALL_MAX_IN))


def heaviest_section_radius(parts: Sequence[SetupPart], max_cells: int = 250_000) -> float:
    """Half-thickness (inches) of the heaviest metal section: the largest inscribed radius.

    Rasterises the metal parts coarsely and takes the largest distance from a
    metal cell to the nearest non-metal cell.
    """
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
    """``params`` with an automatic wall (0) replaced by the wall for the heaviest section.

    ``section_radius_in`` when known (the voxel mesh), else measured from ``parts``.
    """
    if params.wall_thickness_in > 0.0:
        return params
    if section_radius_in is None:
        section_radius_in = heaviest_section_radius(parts)
    return dataclasses.replace(params, wall_thickness_in=auto_wall_thickness(section_radius_in))


def _open_top(code: int, open_flag: bool) -> bool:
    return code in ALWAYS_OPEN_TOP_CODES or (open_flag and code in FEEDER_PART_CODES)


def mold_box(parts: Sequence[SetupPart], params: MoldParams) -> Tuple[np.ndarray, np.ndarray]:
    """Outer faces (lo, hi), inches, of the automatic mold box around ``parts``.

    Sides and bottom: ``wall_thickness_in`` beyond the assembly. Top:

    - with open risers or pour cups, the cope surface is flush with the
      highest open top, raised where needed so blind risers and the casting
      keep ``closed_riser_cover_in`` of sand over them (an open body below
      the surface then gets a shaft, see :func:`plan_layout`);
    - without, the highest of the body tops (casting, chills, cores,
      runners, gates) plus the wall and the blind-riser tops plus the cover.

    Sprue tops always reach the cope surface (or their pour cup).
    """
    if not parts:
        raise MoldSetupError("No parts to build a mold around")
    lo = np.min([np.asarray(m.bounds[0], dtype=np.float64) for m, _, _ in parts], axis=0)
    hi = np.max([np.asarray(m.bounds[1], dtype=np.float64) for m, _, _ in parts], axis=0)
    params = with_wall(params, parts=parts)
    wall = float(params.wall_thickness_in)
    cover = float(params.closed_riser_cover_in)

    open_tops: List[float] = []
    covered = [-np.inf]  # tops that need the cover over them
    walled = [-np.inf]   # tops that need the full wall when nothing is open
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
    """The lattice cells whose centres lie in [lo, hi].

    ``anchor`` is any cell centre of the lattice to align with (the solver's
    voxels); without one the box's lo corner is a cell corner. Returns the
    box lattice and its first cell's index relative to ``anchor``.
    """
    pitch = float(pitch)
    if anchor is None:
        anchor = np.asarray(lo, dtype=np.float64) + 0.5 * pitch
    anchor = np.asarray(anchor, dtype=np.float64)
    first = np.ceil((np.asarray(lo) - anchor) / pitch - 1e-9).astype(np.int64)
    last = np.floor((np.asarray(hi) - anchor) / pitch + 1e-9).astype(np.int64)
    shape = tuple(int(n) for n in np.maximum(last - first + 1, 1))
    origin = anchor + first * pitch
    return Lattice(origin=origin, pitch=pitch, shape=shape), first


# ---------------------------------------------------------------------------
# Rasterising parts
# ---------------------------------------------------------------------------


def rasterize_mesh(mesh: trimesh.Trimesh, lattice: Lattice) -> np.ndarray:
    """(shape) bool: lattice cells whose centre is inside ``mesh`` (winding number)."""
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
    """(labels, open_mask) of ``parts`` on ``lattice``.

    ``labels`` holds the part code per cell (-1 empty), overlap resolved
    toward the higher code like the voxeliser; ``open_mask`` marks cells of
    open-top bodies (open risers, pour cups).
    """
    labels = np.full(lattice.shape, -1, dtype=np.int16)
    open_mask = np.zeros(lattice.shape, dtype=bool)
    for mesh, code, open_flag in sorted(parts, key=lambda p: p[1]):
        inside = rasterize_mesh(mesh, lattice)
        labels[inside] = code
        if _open_top(code, open_flag):
            open_mask |= inside
    open_mask &= labels >= 0
    return labels, open_mask


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _shift(mask: np.ndarray, axis: int, step: int) -> np.ndarray:
    """mask shifted so out[c] = mask[c + step along axis]; False off the edge."""
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
    """Columns above each open body's top, up to the box top or the next body above."""
    nz = blocked.shape[2]
    z = np.arange(nz)
    has_open = open_mask.any(axis=2)
    top = np.where(has_open, nz - 1 - np.argmax(open_mask[:, :, ::-1], axis=2), nz)
    above = z[None, None, :] > top[:, :, None]
    stopped = np.logical_or.accumulate(blocked & above, axis=2)
    return above & ~stopped & ~blocked


def _gaps(blocked: np.ndarray) -> np.ndarray:
    """Empty cells with assembly on both sides along some axis."""
    gap = np.zeros_like(blocked)
    for axis in range(3):
        gap |= _shift(blocked, axis, 1) & _shift(blocked, axis, -1)
    return gap & ~blocked


def _enclosed(open_space: np.ndarray) -> np.ndarray:
    """Cells of ``open_space`` that no 6-connected path links to the box faces."""
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
    """Cells with assembly both above and below them in their column."""
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
    """Mold, auto core, gap and shaft masks on ``lattice`` from the assembly ``labels``.

    ``mold_inside`` (bool, lattice shape) is the interior of loaded mold
    parts; None means the whole lattice is the automatic mold box. Cores are
    found the same way either way: they belong to the assembly, not to the
    mold.
    """
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
    """Outer faces of the mold: the loaded mold parts' bounds, else the automatic box."""
    if mold_meshes:
        lo = np.min([m.bounds[0] for m in mold_meshes], axis=0)
        hi = np.max([m.bounds[1] for m in mold_meshes], axis=0)
        if parts:
            lo = np.minimum(lo, np.min([m.bounds[0] for m, _, _ in parts], axis=0))
            hi = np.maximum(hi, np.max([m.bounds[1] for m, _, _ in parts], axis=0))
        return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    return mold_box(parts, params)


def auto_cores_apply(params: MoldParams, parts: Sequence[SetupPart]) -> bool:
    """True when automatic cores are on and no core part was supplied (supplied cores override)."""
    return bool(params.auto_cores) and not any(code == PART_TYPE_CORE for _, code, _ in parts)


def _effective(params: MoldParams, parts: Sequence[SetupPart]) -> MoldParams:
    """``params`` with auto_cores off when core parts were supplied."""
    if params.auto_cores and not auto_cores_apply(params, parts):
        return dataclasses.replace(params, auto_cores=0)
    return params


def mold_requested(params: MoldParams, mold_meshes: Sequence[trimesh.Trimesh]) -> bool:
    """True when the run carries the mold as sand cells (mold parts or the automatic box)."""
    return bool(mold_meshes) or bool(params.auto_mold)


def preview_layout(
    parts: Sequence[SetupPart],
    mold_meshes: Sequence[trimesh.Trimesh],
    params: MoldParams,
    pitch: float,
) -> MoldLayout:
    """Layout for the setup preview: parts rasterised on their own lattice at ``pitch``."""
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
        # Cores only: no mold sand is drawn or solved.
        mold_inside = np.zeros(lattice.shape, dtype=bool)
    layout = plan_layout(labels, open_mask, lattice, params, mold_inside)
    if not mold_meshes and not params.auto_mold:
        layout.mold[:] = False
    layout.wall_in = 0.0 if mold_meshes else float(params.wall_thickness_in)
    if params.mold_grading_levels > 0 and layout.mold.any():
        # Graded like the run's mold (grade_mold), on the preview's lattice: its seeds are the rasterised parts, the
        # auto cores and the gaps. The summary then carries the graded count (graded_mold_cells) beside mold_cells.
        layout.cell_index = np.where((labels >= 0) | layout.auto_core, 0, -1).astype(np.int32)
        grade_mold(layout, params.mold_grading_levels, params.mold_fine_band_cells)
    return layout


def preview_pitch(parts: Sequence[SetupPart], mold_meshes: Sequence[trimesh.Trimesh], params: MoldParams,
                  voxel_size: Optional[float] = None, max_cells: int = 2_000_000) -> float:
    """Lattice pitch for the preview: the run's voxel size, coarsened to ``max_cells`` cells."""
    lo, hi = layout_box(parts, mold_meshes, params)
    size = np.maximum(hi - lo, 1e-6)
    pitch = float(voxel_size) if voxel_size and voxel_size > 0 else float(np.max(size)) / 120.0
    while np.prod(np.ceil(size / pitch) + 1) > max_cells:
        pitch *= 1.15
    return pitch


# ---------------------------------------------------------------------------
# Voxel mesh
# ---------------------------------------------------------------------------


def mesh_lattice_indices(centers: np.ndarray, pitch: float) -> Tuple[np.ndarray, np.ndarray]:
    """(anchor, ijk): the mesh voxels' integer lattice indices relative to voxel 0."""
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
    """Layout on the solver lattice, from the voxel mesh's own part types."""
    params = _effective(params, parts)
    centers = np.asarray(voxel_mesh.voxel_centers, dtype=np.float64)
    pitch = float(voxel_mesh.base_voxel_size)
    anchor, ijk = mesh_lattice_indices(centers, pitch)
    lo, hi = layout_box(parts, mold_meshes, params)
    # The box always holds every voxel, and one more cell of air around them,
    # so a voxel on the box face still has a lattice neighbour to look up.
    lo = np.minimum(lo, centers.min(axis=0) - pitch)
    hi = np.maximum(hi, centers.max(axis=0) + pitch)
    lattice, first = box_lattice(lo, hi, pitch, anchor)
    _check_cell_budget(lattice, params)

    local = ijk - first
    labels = np.full(lattice.shape, -1, dtype=np.int16)
    # A bare STL run never assigns part types: every voxel is casting.
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
    """The run's mold and metal sizes, estimated from the assembly's own occupancy (:func:`estimate_from_occupancy`)."""

    #: Metal and core voxels the run will have: the filled occupancy plus the auto cores.
    voxels: int
    #: Mold sand cells the solver steps (graded when ``mold_grading_levels`` > 0, else the lattice's).
    mold_cells: int
    #: Mold sand cells of the lattice.
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
    """The run's voxels and mold sand cells from the assembly's occupancy, before any voxel mesh exists.

    ``occupied`` is the filled occupancy the mesher starts from (``voxel_mesh.voxelize_surface(mesh, pitch).fill()``),
    its cell (i, j, k) centred at ``(origin_index + (i, j, k)) * pitch``. The mold lattice is then built as
    :func:`layout_for_voxel_mesh` builds it on the run's mesh: the mold box (:func:`layout_box`) grown to hold every
    occupied cell plus one of air, anchored on the occupancy's cells, the open shafts, the gaps, the auto cores and a
    loaded mold part's interior (:func:`plan_layout`), and the grading (:func:`grade_mold`) with the occupied cells, the
    auto cores and the gaps as its seeds. Part types are not known here (every occupied cell counts as casting); an
    open-topped part's shaft is found from its own mesh. The mesher's removal of pockets sealed between parts but inside
    none (``voxel_mesh._unfill_sealed_pockets``) is not done, so a sealed pocket counts as metal.
    """
    params = _effective(params, parts)
    pitch = float(pitch)
    occupied = np.asarray(occupied, dtype=bool)
    ijk = np.argwhere(occupied).astype(np.int64)
    if not len(ijk):
        raise MoldSetupError("The assembly has no occupied cells at this voxel size")
    anchor = np.asarray(origin_index, dtype=np.float64) * pitch
    centers = anchor + ijk * pitch
    if params.wall_thickness_in <= 0.0:
        # The run takes the wall from its mesh (prepare_mold, _mesh_section_radius): the largest distance from a voxel
        # to the nearest surface voxel, plus half a pitch. The occupancy's distance transform measures to the nearest
        # empty cell, one cell beyond the surface voxel: (largest - 1 + 0.5) pitches is the same radius.
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
    # The grading's seeds are the run's mesh voxels: the occupied cells and the auto cores the run appends.
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
    """grid[ijk] per row, -1 outside the grid."""
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
    """Grow the mesh array ``name`` along axis 0 to ``n_new`` rows, new rows ``fill``."""
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
    """Append the layout's auto-core cells to the voxel mesh as CORE voxels.

    New voxels are full cells (fraction 1). Their exterior area is the
    faces they show to cells that are not mesh voxels; metal voxels lose
    the exterior area they showed to the new cores, which now conduct to
    them through the core contact instead. Neighbour slots are filled from
    the lattice both ways. Returns the number of voxels added.
    """
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

    # Exterior area each existing voxel showed to the new cores, faces only.
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
        # A bare STL run: every existing voxel is casting.
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

    # Existing voxels: remove the area now in contact with a core.
    touched = np.flatnonzero(covered_now > 0)
    if len(touched):
        areas = np.asarray(voxel_mesh.voxel_surface_areas, dtype=np.float64)
        share = covered_now[touched] / np.maximum(exposed_before[touched], 1)
        areas[touched] = np.maximum(areas[touched] * (1.0 - share), 0.0)
        voxel_mesh.voxel_surface_areas[:] = areas.astype(np.float32)

    # New voxels: area on faces to non-mesh cells (prints into the mold).
    new_exposed = np.zeros(n_new, dtype=np.int64)
    for slot in FACE_SLOTS:
        new_exposed += _lookup(grid, new_ijk + NEIGHBOR_SLOT_OFFSETS[slot]) < 0
    voxel_mesh.voxel_surface_areas[n_old:] = (new_exposed * face_area).astype(np.float32)
    voxel_mesh.surface_voxels[:] = np.asarray(voxel_mesh.voxel_surface_areas) > 0.0

    # Neighbours: new rows from the lattice, existing rows gain the new cells.
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
            # A bridge or two-cell link through the new core: drop both ways.
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
    """Heaviest-section half-thickness (inches) from the mesh's distance-to-surface moduli."""
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
    """Run the mold stage on a voxel mesh with part types assigned.

    Adds auto cores to the mesh and attaches the layout as
    ``voxel_mesh.mold_layout`` when it carries mold sand (None otherwise).
    Returns the layout, or None when neither a mold nor auto cores apply.
    """
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


# ---------------------------------------------------------------------------
# Graded mold
# ---------------------------------------------------------------------------


@dataclass
class MoldGrading:
    """The layout's mold sand as cubic cells of 1, 2, 4, ... lattice cells per side.

    Cell c covers the ``span[c]``^3 lattice cells from its low corner
    ``ijk[c]``. Voxel-size cells come first, in lattice order (so with no
    block coarsened the cells are the uniform mold's, in its order), then
    each coarser level. ``owner`` maps every lattice cell to the cell that
    holds it (-1: not mold sand).
    """

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
        """'41,200 mold sand cells, graded from 1,310,000 lattice cells, up to 8x'."""
        if self.max_span <= 1:
            return f"{self.n_cells:,} mold sand cells (none coarsened)"
        return (
            f"{self.n_cells:,} mold sand cells, graded from {lattice_cells:,} lattice cells, "
            f"up to {self.max_span}x"
        )


def grading_thresholds(levels: int, band: int, transition: Optional[int] = None) -> List[int]:
    """[t_1 .. t_levels]: a level-l cell holds only lattice cells more than t_l cells from the assembly.

    t_1 = band, t_l = t_(l-1) + transition * 2^(l-1): ``band`` voxel-size
    layers around the assembly, then ``transition`` cells of each level
    before the next.
    """
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
    """Shift s of the block grid (block edges at lattice index i with (i + s) % block == 0).

    2 s = -n (mod block) makes the edges mirror-symmetric about the lattice
    centre; solvable when ``n`` is even. Odd ``n`` takes the nearest shift.
    """
    return int((-(int(n) // 2)) % int(block))


def _upsample(blocks: np.ndarray, factor: int) -> np.ndarray:
    """Each element of ``blocks`` repeated ``factor`` times along every axis."""
    out = blocks
    for axis in range(3):
        out = np.repeat(out, factor, axis=axis)
    return out


def grade_mold(layout: MoldLayout, levels: int, band: int) -> MoldGrading:
    """Partition the layout's mold sand into cells up to 2^levels lattice cells per side.

    ``dist`` is the Chebyshev distance (lattice cells) from each cell to the
    nearest seed: a mesh voxel (``layout.cell_index >= 0``) or a gap. A
    2^l block, aligned to the block grid, is one cell when every lattice cell
    in it is mold sand with dist > t_l (:func:`grading_thresholds`); its 8
    children then qualify at level l - 1 too. Blocks cut by the lattice
    edge, a shaft, a core or the boundary of a loaded mold part stay finer,
    so the interior steps by 2:1 and irregular boundaries keep fine cells.
    The block grid is anchored per axis so the partition is mirror-symmetric
    where the lattice allows (:func:`_mirror_shift`). Every cell within
    ``band`` (at least 2) of a seed is voxel size, so every contact and
    stencil pair of the mesh lands on a voxel-size cell.

    The result is cached on the layout for the same (levels, band).
    """
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

    # Distance of each mold cell to the nearest seed; 0 off the mold and in
    # the padding, so a block touching either never qualifies.
    min_dist = np.zeros(tuple(int(v) for v in padded), dtype=np.int32)
    if levels > 0 and seeds.any() and mold.any():
        dist = ndimage.distance_transform_cdt(~seeds, metric="chessboard")
        np.copyto(min_dist[crop], dist, where=mold)
        del dist
    elif levels > 0 and mold.any():
        np.copyto(min_dist[crop], np.iinfo(np.int32).max, where=mold)

    thresholds = grading_thresholds(levels, band)
    qualifies: List[np.ndarray] = [np.zeros(0, dtype=bool)]  # index = level
    d = min_dist
    for level in range(1, levels + 1):
        b = d.shape
        d = d.reshape(b[0] // 2, 2, b[1] // 2, 2, b[2] // 2, 2).min(axis=(1, 3, 5))
        qualifies.append(d > thresholds[level - 1])
    del min_dist, d

    # A level-l block is a cell when it qualifies and its parent does not.
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
    """(finer, coarser, shared lattice faces) for every pair of face-touching cells of different size."""
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
    """(n_cells,) lattice faces each graded cell shows the air: off the lattice, or onto a cell that is
    neither mesh, mold nor gap (the uniform mold's rule, summed over each cell's lattice cells)."""
    mold = owner >= 0
    solid = (grid >= 0) | gap
    faces = np.zeros(owner.shape, dtype=np.uint8)
    for axis in range(3):
        for step in (1, -1):
            faces += mold & ~_shift(solid, axis, step)
    return np.bincount(owner[mold], weights=faces[mold], minlength=n_cells).astype(np.int64)


# ---------------------------------------------------------------------------
# Thermal extension
# ---------------------------------------------------------------------------


@dataclass
class ThermalExtension:
    """The mold sand as its own thermal domain, and how it meets the mesh's.

    Mold cell m (local index, 0 .. n_mold - 1) sits at ``mold_ijk[m]``.
    ``mesh_neighbors`` is the mesh's stencil with nothing reaching into the
    mold (bridges across what is now sand dropped); ``mold_neighbors`` is the
    mold's own stencil, local indices. The two meet only through pairs:

    - contact pairs (metal and chills against sand): ``pair_first`` (mesh
      cell), ``pair_second`` (mold cell), ``pair_area`` (in^2);
    - stencil pairs (a sand core against mold sand): ``stencil_first``,
      ``stencil_second`` and ``stencil_slot``, the neighbour slot of the
      27-point link each one replaces.

    Air faces are split the same way (``air_mesh_*``, ``air_mold_*``, areas
    in in^2).

    A graded mold (:func:`grade_mold`) has cells of ``mold_span`` lattice
    cells per side, ``mold_ijk`` their low corner. ``mold_neighbors`` then
    holds same-size neighbours only (``span`` x the slot offset); cells of
    different size that share faces meet through cross-size links
    (``link_first`` the finer cell, ``link_second`` the coarser, ``link_area``
    in^2). Every pair target is a voxel-size cell. On the uniform mold
    ``mold_span`` is all 1 and there are no links.
    """

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
    #: Exterior area of metal/chill cells that faced neither mold nor air.
    unassigned_area: float = 0.0
    #: Lattice cells per side of each mold cell (1 on the uniform mold).
    mold_span: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    link_first: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    link_second: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    link_area: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    #: Mold sand cells of the lattice (the uniform mold's count).
    lattice_mold_cells: int = 0

    @property
    def n_mold(self) -> int:
        return len(self.mold_ijk)

    @property
    def graded(self) -> bool:
        """True when some mold cell is larger than a voxel."""
        return bool(len(self.mold_span)) and int(self.mold_span.max()) > 1

    @property
    def max_span(self) -> int:
        return int(self.mold_span.max()) if len(self.mold_span) else 1


def _symmetric(neighbors: np.ndarray) -> None:
    """Drop links whose reverse is missing, so a stencil stays conservative (in place)."""
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
    """Mold stencil, mesh-mold pairs and air faces for the layout's mold sand.

    The exterior area of a metal or chill cell (its cut-cell CAD area, or
    one face per exposed face without cut cells) is shared equally among its
    exposed faces: faces onto mold sand become contact pairs, faces onto
    nothing become air, and faces onto a gap are dropped (bridged contact).
    A cut cell whose exterior area shows on no face (an oblique skin seen
    only diagonally) shares it among its diagonal mold neighbours. A core
    cell meets the mold through a stencil pair for every link it would have
    had to a mold cell.

    ``grading_levels`` > 0 grades the mold (:func:`grade_mold`, cells up to
    2^levels lattice cells per side, ``fine_band_cells`` voxel-size layers
    around the assembly); 0 is the uniform lattice mold.
    """
    if layout.cell_index is None or layout.mesh_ijk is None:
        raise MoldSetupError("Layout is not attached to a voxel mesh")
    n_mesh = len(voxel_mesh.voxel_centers)
    mesh_ijk = layout.mesh_ijk.astype(np.int64)
    pitch = layout.lattice.pitch
    face_area = pitch * pitch
    lattice_mold_cells = layout.mold_cell_count

    # Mesh cells keep their index; mold cells are n_mesh + local index here.
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
    solid_mesh = ~sand_mesh  # metal, chills, filters

    mesh_neighbors = np.array(voxel_mesh.voxel_neighbors, dtype=np.int32, copy=True)
    # Mesh links longer than one cell that pass through the mold are bridges
    # across what is now sand: drop them both ways.
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
            # Same-size neighbours only; the block grid is aligned, so a cell
            # of the same span there starts exactly at that corner.
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

    # Exterior area of metal/chill cells over their exposed faces.
    exterior = (
        np.asarray(voxel_mesh.voxel_surface_areas, dtype=np.float64)
        if getattr(voxel_mesh, "cut_cells", False)
        else None
    )
    face_kind = np.zeros((n_mesh, 6), dtype=np.int8)  # 0 mesh, 1 mold, 2 air, 3 gap
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
    # Cores showing faces to the air (prints through a loaded mold's edge).
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

    # Mold faces to the air: the box's outer faces, shaft walls, outside a
    # loaded mold, and faces onto mesh cells that are not sand (none, by
    # construction: metal meets the mold through pairs, not open faces).
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
        # The band keeps every pair target voxel size, so the mesh side sees
        # the same mold cells as on the uniform lattice.
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


# ---------------------------------------------------------------------------
# Preview meshes
# ---------------------------------------------------------------------------


def mask_surface(mask: np.ndarray, lattice: Lattice, smooth: int = 8) -> Optional[trimesh.Trimesh]:
    """Closed surface of the cells in ``mask``, Taubin-smoothed ``smooth`` iterations."""
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
    """Display meshes: ``mold`` (the automatic box) and ``cores`` (one per auto-core body)."""
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
    """Flat vertex and face lists, the shape the front ends draw parts from."""
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
    """Layout and display meshes for the setup view."""
    if not mold_meshes:
        params = with_wall(params, parts=parts)
    pitch = preview_pitch(parts, mold_meshes, params, voxel_size)
    layout = preview_layout(parts, mold_meshes, params, pitch)
    return layout, preview_meshes(layout)
