"""
Mesh-to-lattice seam for the mold-filling stage.

:mod:`physics.filling` solves on a dense ``(nx, ny, nz)`` lattice and knows
nothing about voxel meshes; :class:`~core.voxel_mesh.AdaptiveVoxelMesh` stores a
flat list of occupied voxel centres. This module owns the translation, plus the
filter geometry and drag-coefficient construction that needs both the mesh and
the material data.

Layering: ``core/`` may import ``physics/``; the reverse is forbidden and
enforced by ``tests/test_architecture_boundaries.py``. Keeping the mapping here
rather than in ``physics/`` is what lets the solver be tested on synthetic
lattices with no mesh at all.

Two verified facts drive the design
-----------------------------------
**The dense shape must be derived from the voxel centres, not from
``mesh.grid_shape``.** ``grid_shape`` is a *pre-voxelization* bounding-box
estimate, ``ceil((max - min) / dx)``, assigned before any voxel exists. Measured
disagreements: ``thick_thin_test.stl`` at dx=0.15 gives centres spanning
(21, 21, 34) against a ``grid_shape`` of (21, 21, 35), and the
``Filling_Test`` gating assembly at dx=0.3 spans (57, 33, 39) against (58, 34,
40). Sizing a lattice from ``grid_shape`` would leave a phantom empty layer on
two faces.

**The mapping ``ijk = round((centre - min) / dx)`` is bijective.** Verified on
``l_bracket``, ``block_2inch``, ``cylinder`` and ``thick_thin_test``, on both the
default and chunked voxelization paths, with ``unique(ijk) == n_voxels`` in every
case and a maximum lattice residual of 3.8e-6. That is what makes a dense array
representation viable at all.

Coarsening
----------
``fill_grid_coarsening`` is implemented by **box-downsampling the dense
part-type grid**, never by regenerating the mesh at a larger ``base_voxel_size``.
Re-meshing is not merely slower: ``AdaptiveVoxelMesh.generate`` segfaults
reproducibly at certain voxel sizes (see :data:`COARSENING_NOTE`), so a stage
that re-meshes would crash on geometry that the fine mesh handled.

Coarse cells resolve by **priority, never by majority vote**. A filter is
typically one or two cells thick, so a majority rule would vote it out of
existence and silently remove the restriction the user placed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.part_manager import (
    PART_TYPE_CAST,
    PART_TYPE_CHILL,
    PART_TYPE_CORE,
    PART_TYPE_FILTER,
    PART_TYPE_GATE,
    PART_TYPE_POUR_CUP,
    PART_TYPE_RISER,
    PART_TYPE_RISER_POST,
    PART_TYPE_RUNNER,
    PART_TYPE_SPRUE,
    is_metal,
)
from core.filter_database import (
    POUR_MODE_BOTTOM,
    POUR_MODE_LIP,
    flow_rate_kg_s_for_area,
    steel_class_for_alloy,
)
from physics.filling import PorousZone
from physics.fluid_flow import (
    SMAGORINSKY_CS,
    ceramic_foam_permeability,
    forchheimer_inverse_length,
    porous_drag_coefficients,
    capillary_entry_density_offset,
    capillary_entry_pressure,
)
from physics.units import inches_to_meters

#: Why coarsening downsamples rather than re-meshes. Kept as a named constant
#: so the reason travels with the code rather than living only in a commit.
COARSENING_NOTE = (
    "AdaptiveVoxelMesh.generate() segfaults at certain voxel sizes, so the "
    "fill grid is produced by downsampling the fine dense grid rather than by "
    "regenerating the mesh at a larger base_voxel_size."
)

#: Sentinel for a dense cell that lies outside the meshed geometry.
EMPTY_CELL = np.int8(-1)

#: Resolution order when several fine part types fall into one coarse cell.
#:
#: Highest priority first. The filter leads deliberately: it is often one or
#: two cells thick, and any majority or averaging rule would erase it. After
#: that, mold materials outrank metal (a chill that vanishes changes the
#: thermal answer), and within metal the flow path outranks the casting so the
#: gating system stays connected at coarse resolution.
COARSENING_PRIORITY: Tuple[int, ...] = (
    PART_TYPE_FILTER,
    PART_TYPE_CHILL,
    PART_TYPE_CORE,
    PART_TYPE_GATE,
    PART_TYPE_RUNNER,
    PART_TYPE_SPRUE,
    PART_TYPE_POUR_CUP,
    PART_TYPE_RISER_POST,
    PART_TYPE_RISER,
    PART_TYPE_CAST,
)

#: Frames captured across a fill for playback. The solver overwrites its whole
#: state every step, so without this the only surviving record of the pour is
#: one write-once arrival time per cell -- enough to answer "when did this fill?"
#: but not to show metal moving. 60 frames is smooth playback at a size that
#: stays trivial next to the mesh itself (two float32 per voxel per frame:
#: ~16 MB at 33k voxels).
FILL_ANIMATION_FRAMES = 60

#: How a filling run ended. See ``FillingResult.status``.
STATUS_COMPLETE = "complete"
STATUS_TIME_CAPPED = "time_capped"
STATUS_STALLED = "stalled"
STATUS_DIVERGED = "diverged"

#: Cell thickness required at the narrowest gating section before the fill can
#: be believed. Halfway bounce-back puts the walls at +-0.5 cells, so a channel
#: one cell thick has no interior at all and seals shut; two is the bare
#: minimum that admits any flow, and the field notes ask for three to four.
#:
#: This is a *thickness*, not a cross-sectional area. The first version of this
#: check compared an area against 4 and so passed a wide, thin gate -- exactly
#: the geometry that seals -- while looking like it had checked something.
MIN_CHOKE_THICKNESS_CELLS = 2.0

#: Thickness at which a channel is resolved well enough to stop apologising for.
#:
#: Three is the observed floor, not a round number: ``Test_Model/Filling_Test``
#: on a 0.20 in fill grid puts three cells across its 0.5 in gate, and that run
#: time-caps at 57 % of cells with two of 154 gate cells ever going FLUID (see
#: ``docs/filling_field_notes.md``). Above the hard minimum, below this, the
#: fill is marginal rather than wrong -- worth saying, in a softer voice.
MARGINAL_CHOKE_THICKNESS_CELLS = 3.0

#: Parts the metal has to flow *through* on its way to the casting.
#:
#: The filter is included, unlike in :func:`_min_flow_slice_cells`. Thickness is
#: measured on the connected channel, and a filter sitting between the runner
#: and the gate is part of that channel -- leaving it out would put a wall
#: across the flow path and report the runner and gate as pinched at a junction
#: that is actually open. It is also a body worth measuring in its own right: a
#: filter thinner than the cells that carry the metal across it is not a
#: restriction the solver can represent.
_METAL_PATH_CODES: Tuple[int, ...] = (
    PART_TYPE_SPRUE,
    PART_TYPE_RUNNER,
    PART_TYPE_FILTER,
    PART_TYPE_GATE,
)

#: How the metal-path parts are named in warnings, in flow order.
_METAL_PATH_LABELS: Dict[int, str] = {
    PART_TYPE_SPRUE: "sprue",
    PART_TYPE_RUNNER: "runner",
    PART_TYPE_FILTER: "filter",
    PART_TYPE_GATE: "gate",
}

#: Largest Smagorinsky constant the stage will accept from settings.
#:
#: The eddy viscosity goes as Cs^2, so the tolerated band is not a matter of
#: taste: at 0.5 the subgrid viscosity is already ten times what the literature
#: values (0.1-0.2, 0.16 for homogeneous isotropic turbulence) produce, and
#: past that the closure stops behaving like a subgrid model and starts
#: smearing the fill front into a diffusion problem -- while still returning a
#: perfectly stable, perfectly wrong pour. Zero is legal and means exactly what
#: it says: tau_eff collapses to tau and the collision is plain TRT.
MAX_SMAGORINSKY_CS = 0.5

#: Total wet mass, as a multiple of the wet-cell count, above which the run is
#: declared diverged. Every fluid cell holds mass ~= rho * porosity ~= 1 and
#: every interface cell less, so the physical ceiling is the wet-cell count
#: itself; 1.5x is margin, not a threshold to tune. Crossing it does not mean
#: "slightly inaccurate", it means the free-surface bookkeeping has broken.
DIVERGENCE_MASS_FACTOR = 1.5

#: Steps between progress-bar updates. Frequent enough that the bar moves
#: smoothly on a run that takes minutes -- the text reports are twenty times
#: sparser, which is right for a log and far too slow for a progress bar. A
#: fill used to drive no bar at all, so the whole stage looked like a hang.
PROGRESS_UPDATE_INTERVAL = 200

#: Steps between mass-conservation checks. Cheap next to a lattice step.
DIVERGENCE_CHECK_INTERVAL = 200

#: Consecutive progress reports with no advance -- in either measure -- before
#: the run is called stalled. The front genuinely pauses while a cavity backs
#: up, so a single flat report proves nothing.
STALL_REPORTS = 5

#: Blend weight for the inlet's measured velocity, ``u = (1-b) u_prev + b u_raw``.
#:
#: A numerical filter on a boundary measurement, not a physical term. The
#: pressure inlet reads its velocity out of the populations every step, so
#: population noise in the inlet layer feeds straight back into what the
#: boundary writes. Measured on a 2x2x40 column, unfiltered: peak |u| in the
#: two cells below the inlet reached 0.183 against a 0.05 Mach guard, swinging
#: 0.05 <-> 0.18 report to report and still rising at the end of the run. At
#: b = 0.2 the same run peaks at 0.077 and fills in 2,060 steps; at b = 0.05 it
#: peaks at 0.053 but takes 3,296. It damps the boundary's response, it does
#: not bias it: the filter has unit gain, so a steady flow settles on exactly
#: the velocity it was measuring.
INLET_VELOCITY_BETA = 0.2

#: Extra fraction of the metal's full mass that must arrive between reports for
#: the fill to count as advancing.
#:
#: The filled-cell high-water mark alone is not enough. A cell only flips to
#: FLUID at the top of the eps hysteresis band, so a front creeping through
#: interface cells takes on mass every report while flipping no flags and reads
#: as stopped. Measured on Test_Model/Filling_Test: 428 casting cells held at
#: eps ~ 0.5, advancing, with the cell count flat. A stall now requires both
#: measures to be flat.
STALL_MASS_FRACTION_EPSILON = 0.001

#: Steps that must elapse before a stall can be declared at all. Reports are
#: spaced at ``max_steps / 20``, so on a short step budget five of them can
#: pass within a handful of steps -- before the front has left the inlet. That
#: is a run too young to judge, not a stalled one.
STALL_MIN_STEPS = 2000


@dataclass(frozen=True)
class GridMapping:
    """
    Bijection between the mesh's voxel list and a dense lattice.

    Attributes:
        shape: Dense grid shape, derived from the voxel centres.
        dx_in: Voxel pitch in inches.
        origin_in: Minimum voxel centre, inches, shape ``(3,)``.
        voxel_to_grid: ``(n_voxels, 3)`` int32 lattice index per voxel.
        grid_to_voxel: ``(nx, ny, nz)`` int32 voxel index per cell, ``-1``
            where the lattice cell is outside the meshed geometry.
    """

    shape: Tuple[int, int, int]
    dx_in: float
    origin_in: np.ndarray
    voxel_to_grid: np.ndarray
    grid_to_voxel: np.ndarray

    @property
    def n_voxels(self) -> int:
        return int(self.voxel_to_grid.shape[0])

    @property
    def occupancy(self) -> float:
        """
        Fraction of the dense box that is real geometry.

        Worth watching: a gating assembly is mostly empty bounding box. The
        ``Filling_Test`` rigging measures 0.147, well below the 0.3-0.8 the
        design doc anticipated, so the dense lattice is ~7x the occupied voxel
        count and the memory estimate must be made per dense cell.
        """
        return self.n_voxels / float(np.prod(self.shape))

    @property
    def dx_m(self) -> float:
        return float(inches_to_meters(self.dx_in))


def build_grid_mapping(
    voxel_centers: np.ndarray, voxel_size_in: float
) -> GridMapping:
    """
    Map occupied voxel centres onto a dense lattice.

    Args:
        voxel_centers: ``(n, 3)`` voxel centres in inches.
        voxel_size_in: Uniform voxel pitch in inches.

    Returns:
        A :class:`GridMapping`.

    Raises:
        ValueError: If the centres are empty, malformed, or do not sit on a
            uniform lattice of the given pitch. The last case is checked rather
            than assumed: the whole dense representation rests on it, and a
            silent failure would produce a mapping that loses voxels.
    """
    centers = np.asarray(voxel_centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(f"voxel_centers must be (n, 3), got {centers.shape}")
    if centers.shape[0] == 0:
        raise ValueError("voxel_centers is empty")
    if voxel_size_in <= 0:
        raise ValueError(f"voxel_size_in must be positive, got {voxel_size_in}")

    origin = centers.min(axis=0)
    scaled = (centers - origin) / float(voxel_size_in)
    ijk = np.round(scaled).astype(np.int64)

    residual = float(np.abs(scaled - ijk).max())
    if residual > 1e-3:
        raise ValueError(
            f"voxel centres do not lie on a uniform lattice of pitch "
            f"{voxel_size_in}: maximum residual {residual:.3e} cells. The dense "
            f"grid representation requires a uniform mesh."
        )

    shape = tuple(int(v) for v in (ijk.max(axis=0) + 1))
    grid_to_voxel = np.full(shape, -1, dtype=np.int32)
    grid_to_voxel[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = np.arange(
        centers.shape[0], dtype=np.int32
    )

    mapped = int(np.count_nonzero(grid_to_voxel >= 0))
    if mapped != centers.shape[0]:
        raise ValueError(
            f"voxel-to-grid mapping is not bijective: {centers.shape[0]} voxels "
            f"collapsed onto {mapped} lattice cells. Two voxels share a cell, so "
            f"the mesh is not uniform."
        )

    return GridMapping(
        shape=(shape[0], shape[1], shape[2]),
        dx_in=float(voxel_size_in),
        origin_in=origin,
        voxel_to_grid=ijk.astype(np.int32),
        grid_to_voxel=grid_to_voxel,
    )


def dense_part_types(
    mapping: GridMapping, voxel_part_types: np.ndarray
) -> np.ndarray:
    """
    Scatter the mesh's per-voxel part codes onto the dense lattice.

    Cells outside the geometry take :data:`EMPTY_CELL`, which is distinct from
    ``PART_TYPE_CAST`` (0). Conflating "no geometry here" with "casting metal
    here" would fill the entire bounding box with metal.
    """
    codes = np.asarray(voxel_part_types)
    if codes.shape[0] != mapping.n_voxels:
        raise ValueError(
            f"voxel_part_types has {codes.shape[0]} entries but the mapping "
            f"covers {mapping.n_voxels} voxels"
        )
    dense = np.full(mapping.shape, EMPTY_CELL, dtype=np.int8)
    ijk = mapping.voxel_to_grid
    dense[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = codes.astype(np.int8)
    return dense


def scatter_to_grid(
    mapping: GridMapping, values: np.ndarray, fill: float = 0.0
) -> np.ndarray:
    """Scatter a per-voxel field onto the dense lattice."""
    values = np.asarray(values)
    if values.shape[0] != mapping.n_voxels:
        raise ValueError(
            f"values has {values.shape[0]} entries but the mapping covers "
            f"{mapping.n_voxels} voxels"
        )
    dense = np.full(mapping.shape, fill, dtype=np.float64)
    ijk = mapping.voxel_to_grid
    dense[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = values.astype(np.float64)
    return dense


def gather_to_voxels(mapping: GridMapping, dense: np.ndarray) -> np.ndarray:
    """
    Gather a dense field back onto the voxel list.

    The inverse of :func:`scatter_to_grid`. The result has shape
    ``(n_voxels,)`` -- which is exactly what
    ``SolidificationEngine.set_initial_temperature_field`` requires, and is the
    occupied-voxel count, not the dense cell count.
    """
    dense = np.asarray(dense)
    if tuple(dense.shape) != tuple(mapping.shape):
        raise ValueError(
            f"dense field has shape {dense.shape}, expected {mapping.shape}"
        )
    ijk = mapping.voxel_to_grid
    gathered: np.ndarray = dense[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    return gathered


def coarsen_part_types(
    dense: np.ndarray, factor: int, priority: Sequence[int] = COARSENING_PRIORITY
) -> np.ndarray:
    """
    Box-downsample a dense part-type grid.

    Each coarse cell takes the highest-priority code present among the fine
    cells it covers, so thin features survive. A coarse cell covering only
    empty fine cells stays empty.

    Args:
        dense: ``(nx, ny, nz)`` int8 part-type grid, :data:`EMPTY_CELL` outside.
        factor: Downsampling factor. 1 returns a copy.
        priority: Codes highest-priority first.

    Returns:
        The coarsened grid, shape ``ceil(shape / factor)``.

    Raises:
        ValueError: If ``factor`` is not a positive integer.
    """
    dense = np.asarray(dense, dtype=np.int8)
    if factor < 1 or int(factor) != factor:
        raise ValueError(f"factor must be a positive integer, got {factor}")
    factor = int(factor)
    if factor == 1:
        return dense.copy()

    nx, ny, nz = dense.shape
    pad = (
        (0, (-nx) % factor),
        (0, (-ny) % factor),
        (0, (-nz) % factor),
    )
    padded = np.pad(dense, pad, mode="constant", constant_values=EMPTY_CELL)
    cx, cy, cz = (s // factor for s in padded.shape)
    blocks = padded.reshape(cx, factor, cy, factor, cz, factor)
    blocks = blocks.transpose(0, 2, 4, 1, 3, 5).reshape(cx, cy, cz, -1)

    out = np.full((cx, cy, cz), EMPTY_CELL, dtype=np.int8)
    # Fill from lowest priority upward so higher priority overwrites.
    for code in reversed(list(priority)):
        present = np.any(blocks == code, axis=-1)
        out[present] = np.int8(code)
    return out


@dataclass(frozen=True)
class FilterZoneSpec:
    """
    One ceramic filter body, resolved to lattice cells and drag coefficients.

    Attributes:
        zone_id: Index of this filter among those found.
        part_name: Name of the originating part, for reporting.
        cell_count: Number of lattice cells the filter occupies.
        front_face_area_m2: Face area presented to the flow.
        thickness_m: Thickness along the flow.
        porosity: Void fraction used for the drag closure.
        pore_diameter_m: Mean cell diameter.
        k1_m2: Darcy permeability.
        inv_k2_per_m: Forchheimer inertial coefficient.
        cells_along_flow: Effective cell count through the thickness.
        thickness_correction: Factor keeping the integrated pressure drop
            invariant to how many cells resolve the thickness.
        entry_pressure_pa: Capillary pressure required to prime.
        priming_head_m: The same, expressed as a metallostatic head.
    """

    zone_id: int
    part_name: str
    cell_count: int
    front_face_area_m2: float
    thickness_m: float
    porosity: float
    pore_diameter_m: float
    k1_m2: float
    inv_k2_per_m: float
    cells_along_flow: float
    thickness_correction: float
    entry_pressure_pa: float
    priming_head_m: float


def filter_geometry_from_mesh(part_mesh) -> Tuple[float, float]:
    """
    Front-face area and thickness of a filter body, in metres.

    Thickness is the **smallest oriented-bounding-box extent**, which is
    correct for every catalog size: 20 < 50, 25 < 55, 25 < 75, 25 < 100,
    30 < 125 and 30 < 150 mm.

    Area is ``volume / thickness``, which is exact for a prism of *any*
    cross-section. Multiplying the two larger bbox extents instead
    over-estimates a round filter by 4/pi, i.e. 27 %, and the catalog has seven
    round sizes.

    Args:
        part_mesh: A ``trimesh.Trimesh`` in inches.

    Returns:
        ``(front_face_area_m2, thickness_m)``.

    Raises:
        ValueError: If the body has no volume.
    """
    extents = np.asarray(part_mesh.bounding_box_oriented.extents, dtype=np.float64)
    thickness_in = float(extents.min())
    volume_in3 = float(abs(part_mesh.volume))
    if thickness_in <= 0.0 or volume_in3 <= 0.0:
        raise ValueError(
            "filter body has no measurable volume; it may be an open or "
            "degenerate mesh"
        )
    area_in2 = volume_in3 / thickness_in
    thickness_m = float(inches_to_meters(thickness_in))
    area_m2 = area_in2 * (0.0254 ** 2)
    return area_m2, thickness_m


def thickness_correction(
    thickness_m: float,
    cells_along_flow: float,
    dx_m: float,
    *,
    minimum: float = 0.25,
    maximum: float = 4.0,
) -> float:
    """
    Keep the integrated pressure drop invariant to cell count.

    The solver applies drag per cell, so a filter resolved by 2 cells and the
    same filter resolved by 8 produce different total drops unless the
    coefficients are scaled. This returns the factor by which the resolved
    thickness over- or under-states the true one; divide ``k1`` by it and
    multiply ``1/k2`` by it.

    Clamped, because a filter resolved by a fraction of a cell would otherwise
    demand an unbounded correction.
    """
    resolved = max(cells_along_flow * dx_m, dx_m)
    return float(np.clip(thickness_m / resolved, minimum, maximum))


def cells_along_flow(cell_count: int, front_face_area_m2: float, dx_m: float) -> float:
    """
    Effective number of cells through a filter's thickness.

    Derived as ``volume / (area * dx)`` rather than by casting a single ray,
    so ragged voxelization averages out instead of depending on where the ray
    happens to land.
    """
    if front_face_area_m2 <= 0 or dx_m <= 0:
        raise ValueError("front_face_area_m2 and dx_m must be positive")
    volume = cell_count * dx_m ** 3
    return float(volume / (front_face_area_m2 * dx_m))


def build_filter_zone_specs(
    dense_types: np.ndarray,
    *,
    dx_m: float,
    part_meshes: Optional[Dict[int, object]] = None,
    porosity: float = 0.85,
    pore_diameter_m: float = 2.54e-3,
    ergun_prefactor: float = 2.2,
    forchheimer_c_f: float = 0.55,
    surface_tension_n_m: float = 1.8,
    contact_angle_deg: float = 120.0,
    window_fraction: float = 0.45,
    density_kg_m3: float = 7200.0,
) -> List[FilterZoneSpec]:
    """
    Find connected filter regions and derive their drag coefficients.

    Each connected component of FILTER cells becomes one zone, so an assembly
    with two filters gets two independently reported zones.

    Geometry is taken from the lattice by default: face area is the component's
    largest axis-aligned cross-section and thickness follows from the volume.
    When the originating ``trimesh`` bodies are available they give a better
    answer via :func:`filter_geometry_from_mesh`.

    Args:
        dense_types: Dense part-type grid.
        dx_m: Cell size in metres.
        part_meshes: Optional ``{zone_id: trimesh}`` for exact geometry.
        porosity: Foam void fraction.
        pore_diameter_m: Mean cell diameter.
        ergun_prefactor: Fitted Ergun constant.
        forchheimer_c_f: Ward inertial coefficient.
        surface_tension_n_m: Melt surface tension.
        contact_angle_deg: Melt-ceramic contact angle.
        window_fraction: Window-to-cell diameter ratio.
        density_kg_m3: Melt density.

    Returns:
        One spec per filter zone, empty when the assembly has no filter.
    """
    from scipy import ndimage

    mask = np.asarray(dense_types) == PART_TYPE_FILTER
    if not np.any(mask):
        return []

    labels, count = ndimage.label(mask)
    specs: List[FilterZoneSpec] = []

    k1 = ceramic_foam_permeability(
        porosity, pore_diameter_m, ergun_prefactor=ergun_prefactor
    )
    inv_k2 = forchheimer_inverse_length(k1, c_f=forchheimer_c_f)
    entry_pressure = capillary_entry_pressure(
        surface_tension_n_m, contact_angle_deg, pore_diameter_m, window_fraction
    )
    head_m = entry_pressure / (density_kg_m3 * 9.81)

    for zone_id in range(1, count + 1):
        component = labels == zone_id
        cell_count = int(np.count_nonzero(component))

        supplied = (part_meshes or {}).get(zone_id)
        if supplied is not None:
            area_m2, thickness_m = filter_geometry_from_mesh(supplied)
        else:
            area_m2, thickness_m = _lattice_filter_geometry(component, dx_m)

        along = cells_along_flow(cell_count, area_m2, dx_m)
        correction = thickness_correction(thickness_m, along, dx_m)

        specs.append(
            FilterZoneSpec(
                zone_id=zone_id,
                part_name=f"filter_{zone_id}",
                cell_count=cell_count,
                front_face_area_m2=area_m2,
                thickness_m=thickness_m,
                porosity=porosity,
                pore_diameter_m=pore_diameter_m,
                k1_m2=k1 / correction,
                inv_k2_per_m=inv_k2 * correction,
                cells_along_flow=along,
                thickness_correction=correction,
                entry_pressure_pa=entry_pressure,
                priming_head_m=head_m,
            )
        )
    return specs


def _lattice_filter_geometry(
    component: np.ndarray, dx_m: float
) -> Tuple[float, float]:
    """
    Face area and thickness of a filter region, measured from the lattice.

    The flow axis is taken as the one with the smallest extent -- a filter is a
    thin slab -- and the face area is the volume divided by that thickness,
    matching :func:`filter_geometry_from_mesh`'s convention so the two routes
    agree.
    """
    occupied = np.argwhere(component)
    extents = occupied.max(axis=0) - occupied.min(axis=0) + 1
    thickness_cells = int(extents.min())
    thickness_m = thickness_cells * dx_m
    volume_m3 = int(component.sum()) * dx_m ** 3
    return volume_m3 / thickness_m, thickness_m


def build_porous_zone(
    dense_types: np.ndarray,
    specs: Sequence[FilterZoneSpec],
    *,
    dx_m: float,
    dt_s: float,
    nu_lattice: float,
    density_kg_m3: float = 7200.0,
) -> PorousZone:
    """
    Assemble the per-cell drag arrays the solver consumes.

    Cells outside every filter keep the inert values ``(porosity 1, c0 0.5,
    c1 0)``, which make the Guo-Zhao solve reduce to ``u = v`` exactly.

    Args:
        dense_types: Dense part-type grid.
        specs: Filter zones from :func:`build_filter_zone_specs`.
        dx_m: Cell size in metres.
        dt_s: Timestep in seconds.
        nu_lattice: Kinematic viscosity in lattice units.
        density_kg_m3: Melt density, for the capillary offset.

    Returns:
        A populated :class:`~physics.filling.PorousZone`.
    """
    from scipy import ndimage

    dense_types = np.asarray(dense_types)
    shape = (
        int(dense_types.shape[0]),
        int(dense_types.shape[1]),
        int(dense_types.shape[2]),
    )
    zone = PorousZone.inert(shape)
    if not specs:
        return zone

    mask = dense_types == PART_TYPE_FILTER
    labels, _ = ndimage.label(mask)

    for spec in specs:
        cells = labels == spec.zone_id
        if not np.any(cells):
            continue
        c0, c1 = porous_drag_coefficients(
            spec.porosity,
            spec.k1_m2,
            spec.inv_k2_per_m,
            nu_lattice=nu_lattice,
            dx_m=dx_m,
        )
        offset = capillary_entry_density_offset(
            spec.entry_pressure_pa,
            density_kg_m3=density_kg_m3,
            dt_s=dt_s,
            dx_m=dx_m,
        )
        zone.porosity[cells] = np.float32(spec.porosity)
        zone.drag_c0[cells] = np.float32(c0)
        zone.drag_c1[cells] = np.float32(c1)
        zone.entry_offset[cells] = np.float32(offset)

    return zone


def _static_vent_mask(
    dense_types: np.ndarray, inlet: np.ndarray, vent_risers_enabled: bool
) -> np.ndarray:
    """
    Vent-eligible cells whose eligibility does not depend on the fill's
    progress: open riser tops (when enabled) and the inlet.

    "Open riser top" is a static geometric fact -- the cell one step above a
    RISER cell is either off the top of the domain or has no geometry at all
    (:data:`EMPTY_CELL`), which is exactly section 6.3's "vents/risers open
    to atmosphere". Whether that riser cell currently *holds* gas is not
    this function's concern -- :func:`~physics.filling.find_entrapped_gas`
    intersects the mask with the live GAS set itself, so a riser that has
    already filled simply contributes nothing further, harmlessly.

    Inlet cells are included per the design doc even though they hold FLUID
    for essentially the whole run and so rarely intersect a gas set -- the
    one moment they might (before the very first fill step) is exactly the
    moment a well-formed vent mask should already cover them.
    """
    from core.part_manager import PART_TYPE_RISER

    shape = dense_types.shape
    vent = np.zeros(shape, dtype=bool)

    if vent_risers_enabled:
        riser = dense_types == PART_TYPE_RISER
        above = np.full(shape, EMPTY_CELL, dtype=dense_types.dtype)
        above[:, :, :-1] = dense_types[:, :, 1:]
        vent |= riser & (above == EMPTY_CELL)

    if inlet.shape[0]:
        vent[inlet[:, 0], inlet[:, 1], inlet[:, 2]] = True

    return vent


def _domain_boundary_mask(shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Cells sitting on the outer envelope of the dense grid.

    ``build_grid_mapping`` sizes the dense grid exactly to the voxel
    centres' bounding box (module docstring, "the dense shape must be
    derived from the voxel centres"), so a cell at index 0 or ``shape - 1``
    along any axis is at the true edge of the meshed geometry -- nothing is
    modelled beyond it, and no wall was ever asserted there either. A GAS
    cell found there is therefore open to atmosphere by omission, not by
    any wall the user drew.
    """
    boundary = np.zeros(shape, dtype=bool)
    boundary[[0, -1], :, :] = True
    boundary[:, [0, -1], :] = True
    boundary[:, :, [0, -1]] = True
    return boundary


def _entrapment_warning_if_no_vent(vent_mask: np.ndarray) -> Optional[str]:
    """
    Message to warn with when the assembled vent mask has no open cell.

    Section 6.3 of the design doc: without at least one vent the cavity
    cannot fill at all and air compresses unphysically; every gas pocket
    then reads as entrapped by construction, which is worth telling the user
    rather than letting them discover a 100%-entrapped result and assume the
    detector is broken.
    """
    if np.any(vent_mask):
        return None
    return (
        "no vent path found for air-entrapment checking (vent_risers is "
        "off, no riser is present, and no gas cell currently touches the "
        "inlet or the domain boundary): every gas pocket will read as "
        "entrapped until a vent opens"
    )


def measure_choke_area_cm2(dense_types: np.ndarray, dx_m: float) -> float:
    """
    Smallest cross-section anywhere in the gating system, in cm^2.

    The choke is measured rather than asked for. Foseco's sizing rules are
    expressed as a ratio of filter face area to choke area, and the choke is a
    property of the geometry the user already drew -- asking them to type it in
    invites a number that disagrees with their own model.

    Scanned as the minimum per-slice count across all three axes, over the
    sprue, runner and gate. The pour cup is excluded: it is a reservoir, not a
    metering section, and its cross-section is irrelevant to the choke.

    **Measure this on the grid that is actually solved.** A cell count times a
    cell area is a property of a particular lattice, not of the CAD, and the
    two disagree by more than rounding: on the ``Test 2`` assembly the fine
    mesh gives 2.98 cm^2 and the coarsened grid the solver runs on gives
    7.46 cm^2. The stage used to report the fine-grid figure next to results
    produced on the coarse one, which invited a filter-ratio check against a
    choke the fill never had. ``FillingResult`` now carries both, labelled.

    Returns:
        Choke area in cm^2, or 0.0 when no gating metal is present.
    """
    cells = _min_flow_slice_cells(dense_types)
    if cells <= 0:
        return 0.0
    return float(cells) * (dx_m * 100.0) ** 2


def pour_weight_lb(voxel_mesh, alloy) -> float:
    """
    Total weight of metal to be poured, in pounds.

    Everything the pour has to fill: the casting, the risers and the whole
    gating system. Mold, cores and chills are excluded -- no metal goes into
    them. Measured from the voxel volumes rather than the CAD bounding boxes so
    it matches the geometry the solver actually sees.

    Returns:
        Pounds of metal, or 0.0 when the mesh carries no part types.
    """
    part_types = getattr(voxel_mesh, "voxel_part_types", None)
    volumes = getattr(voxel_mesh, "voxel_volumes", None)
    if part_types is None:
        return 0.0

    metal = is_metal(np.asarray(part_types, dtype=np.int32))
    if not np.any(metal):
        return 0.0

    if volumes is not None and len(volumes) == len(metal):
        volume_in3 = float(np.sum(np.asarray(volumes, dtype=np.float64)[metal]))
    else:
        sizes = np.asarray(voxel_mesh.voxel_sizes, dtype=np.float64)[metal]
        volume_in3 = float(np.sum(sizes ** 3))

    # density_liquid is kg/m^3; 1 in^3 = 1.6387064e-5 m^3, 1 kg = 2.20462262 lb.
    kg = float(alloy.density_liquid) * volume_in3 * 1.6387064e-5
    return kg * 2.20462262


def filter_flow_limit_kg_s(
    filter_zones: Sequence["FilterZoneSpec"],
    alloy,
    pour_mode_code: int = 0,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Mass flow the gating system's filters are rated to pass.

    A ceramic foam filter is a rated restriction, not just a drag term: run
    metal through it faster than its rating and it does not politely throttle,
    it blinds off or breaks up and sends ceramic into the casting. The pour
    rate therefore has a hard ceiling whenever a filter is in the flow path,
    and that ceiling is a property of the filter the user already placed.

    Filters are summed, not minimised: a gating system with several of them
    runs one per ingate, in parallel, so the capacities add. Filters stacked
    in series would share one rating instead -- that is not a layout this
    reads from the geometry, and it is the rarer arrangement by far.

    Args:
        filter_zones: Resolved filter bodies; empty means no restriction.
        alloy: Provides ``name``, to pick the Foseco steel table.
        pour_mode_code: 0 = lip, 1 = bottom. Bottom pouring is rated higher.

    Returns:
        ``(limit_kg_s, note)``. ``limit_kg_s`` is None when no filter is
        present or none can be rated; ``note`` explains the latter case and is
        None otherwise.
    """
    if not filter_zones:
        return None, None

    steel_class = steel_class_for_alloy(getattr(alloy, "name", "") or "")
    if steel_class is None:
        # Refusing to guess, for the same reason flow_rate_kg_s_for_area
        # refuses: substituting a steel figure for a non-ferrous alloy gives a
        # plausible-looking rating that is wrong by tens of percent.
        return None, (
            f"{getattr(alloy, 'name', 'this alloy')} is not covered by the "
            f"Foseco steel flow tables, so the filter's rated flow could not "
            f"be checked -- verify the pour rate against the filter's own "
            f"data sheet."
        )

    pour_mode = POUR_MODE_BOTTOM if int(pour_mode_code) == 1 else POUR_MODE_LIP
    total = 0.0
    for zone in filter_zones:
        area_cm2 = float(zone.front_face_area_m2) * 1.0e4
        if area_cm2 <= 0.0:
            continue
        total += flow_rate_kg_s_for_area(area_cm2, steel_class, pour_mode)
    return (total if total > 0.0 else None), None


def recommended_pour_rate_lb_s(weight_lb: float) -> float:
    """
    Pour rate from the foundry square-root rule.

    ``t = sqrt(W)`` gives the pour time in seconds for a casting of ``W``
    pounds, so the rate is ``W / t = W / sqrt(W) = sqrt(W)``. Pour time in
    seconds and pour rate in lb/s are therefore the same number, which is the
    whole convenience of the rule.

    It replaces a fixed 5 lb/s default that was independent of the casting --
    a 20 lb bracket and a 2,000 lb housing are not poured at the same rate,
    and the mesh already knows which one it is looking at.

    Returns:
        Pounds per second, or 0.0 for a non-positive weight.
    """
    if weight_lb <= 0.0:
        return 0.0
    return float(np.sqrt(weight_lb))


def _min_flow_slice_cells(dense_types: np.ndarray) -> int:
    """
    Cells in the smallest gating cross-section, on the fill grid.

    A cross-sectional *area* in cell units, which is what
    :func:`measure_choke_area_cm2` needs. It is emphatically NOT "cells
    across" -- see :func:`measure_choke_thickness_cells`.
    """
    dense_types = np.asarray(dense_types)
    flow = np.isin(dense_types, (PART_TYPE_SPRUE, PART_TYPE_RUNNER, PART_TYPE_GATE))
    if not np.any(flow):
        return 0

    best = np.inf
    for axis in (0, 1, 2):
        counts = flow.sum(axis=tuple(a for a in (0, 1, 2) if a != axis))
        occupied = counts[counts > 0]
        if occupied.size:
            best = min(best, float(occupied.min()))
    if not np.isfinite(best):
        return 0
    return int(best)


def _local_thickness_cells(region: np.ndarray) -> np.ndarray:
    """
    Channel thickness at every cell of a boolean region, in cells.

    A cell's thickness is the diameter of the **largest inscribed sphere that
    covers it**, not the distance from it to the nearest wall. The difference
    is the whole point. A distance transform reports 1.0 for every surface
    cell of every body, thick or thin, so any statistic taken over those
    numbers -- median, percentile, anything but the maximum -- says 1.0 about
    a solid block. Only the maximum carries thickness information, and the
    maximum is exactly what a fat junction hijacks.

    Computed the standard way (Hildebrand & Ruegsegger): take the distance
    transform ``d``, then for each radius ``r``, every cell within ``r`` of a
    cell whose ``d >= r`` is covered by a sphere of that radius. Sweeping
    ``r`` downward in half-cell steps and keeping the largest that reaches
    each cell gives the map. Rounding the radii down to a half cell is
    deliberate: it can only under-report thickness, and under-reporting errs
    toward warning.

    Thickness is ``2*r - 1``, not ``2*r``, for the reason the scalar measure
    has always used: ``distance_transform_edt`` measures to the nearest
    *background* cell, so a cell in a one-thick section reports 1.0 -- the gap
    to the wall, not the metal. Exact for odd thicknesses, one cell
    conservative for even ones.

    Args:
        region: Boolean mask of the connected material to measure.

    Returns:
        Float array shaped like ``region``: thickness in cells inside the
        region, 0.0 outside it.
    """
    from scipy import ndimage

    region = np.asarray(region, dtype=bool)
    out = np.zeros(region.shape, dtype=np.float64)
    if not np.any(region):
        return out

    # Cropped to the region's own bounding box, with a cell of margin. The
    # sweep below runs one distance transform per radius, so the cost is worth
    # bounding; nothing outside the box can change a result inside it.
    occupied = np.nonzero(region)
    box = tuple(
        slice(int(axis.min()), int(axis.max()) + 1) for axis in occupied
    )
    # Padded with one cell of background. Without it a channel that reaches
    # the edge of the grid has nothing to measure to on that side, and a
    # domain entirely full of metal returns the array diagonal rather than a
    # thickness. The pad is physically right as well as defensive: the lattice
    # ends in obstacle, not in more metal.
    sub = np.pad(region[box], 1, mode="constant", constant_values=False)

    distance = ndimage.distance_transform_edt(sub)
    largest = np.zeros_like(distance)
    radius = np.floor(float(distance.max()) * 2.0) / 2.0
    while radius >= 0.5:
        seeds = distance >= radius
        if np.any(seeds):
            # Distance from every cell to the nearest sphere centre of this
            # radius; a cell within `radius` of one is inside that sphere.
            reach = ndimage.distance_transform_edt(~seeds)
            np.maximum(largest, np.where(reach <= radius, radius, 0.0), out=largest)
        radius -= 0.5

    inner = largest[(slice(1, -1),) * largest.ndim]
    out[box] = np.where(region[box], 2.0 * inner - 1.0, 0.0)
    return out


def measure_choke_thickness_by_part(dense_types: np.ndarray) -> Dict[int, float]:
    """
    Robust channel thickness of each metal-path part, in fill-grid cells.

    This is the number that decides whether metal can get through: halfway
    bounce-back puts the walls at +-0.5 cells, so a channel one cell thick has
    no interior at all and seals shut regardless of how wide it is.

    Cross-sectional *area* cannot answer that question, and the first version
    of this check used area and was wrong for the case that matters most. A
    gate is characteristically wide and thin -- the one on the reference
    assembly is 4.0 x 0.5 in -- so on a coarse grid it presents a perfectly
    healthy-looking 11-cell cross-section while being one cell thick. Area
    passes; the metal does not.

    The version after that measured thickness, but as **one maximum per
    part**, and a maximum is the wrong end of the distribution. Measured on
    the ``Test 2`` assembly at its solved 8.6 mm grid: the gates voxelize to
    11 x 13 x 2 plates -- two cells thick, sealed -- yet the check reported
    3.00 cells and stayed silent, because a single fat cell where each gate
    meets the runner set the maximum for the entire plate. The thin plate is
    the part the metal has to cross; the fillet at its root is not.

    So the statistic is the **median** thickness over the part's own cells,
    taken from :func:`_local_thickness_cells`. Justified against four measured
    grids rather than chosen for elegance:

    ======================================  =====  ======  ======  ====
    grid                                    sprue  runner  filter  gate
    ======================================  =====  ======  ======  ====
    ``Test 2``, 8.6 mm (the failing case)     5      4       7      1
    ``Test 2``, 4.3 mm                       11      7      15      3
    ``Full_Gating_Test``, 0.20 in             5      7       -      5
    ``Filling_Test``, 0.20 in                 7      7       -      3
    ======================================  =====  ======  ======  ====

    The failing gate reads 1 and warns; the same gate on the finer grid reads
    3 and does not; ``Full_Gating_Test`` is silent at 5 and up;
    ``Filling_Test`` reads the 3 its field notes record for that gate.

    The *minimum* would fail two of the four. ``Full_Gating_Test``'s sprue has
    one 1-cell corner on an otherwise 5-cell channel, and ``Test 2`` at 4.3 mm
    has the same in its runner and filter -- three false alarms on grids that
    resolve their gating perfectly well. The 25th percentile agrees with the
    median on all four; it is simply nearer that noise floor for no gain.

    What the median gives up, honestly: a short pinch in an otherwise
    generous channel is voted down by the rest of the part. Persistent
    thinness is what it detects, which is the failure mode that stops a fill;
    a one-cell nick in a runner is not.

    Returns:
        ``{part code: thickness in cells}`` for every metal-path part present.
        Empty when the grid holds no gating metal.
    """
    dense_types = np.asarray(dense_types)
    # One map over the whole connected path, so a gate cell touching the
    # runner is interior rather than boundary -- measuring each part in
    # isolation would report every junction as a constriction.
    thickness = _local_thickness_cells(np.isin(dense_types, _METAL_PATH_CODES))

    by_part: Dict[int, float] = {}
    for code in _METAL_PATH_CODES:
        part = dense_types == code
        if not np.any(part):
            continue
        by_part[code] = max(0.0, float(np.median(thickness[part])))
    return by_part


def narrowest_choke_part_code(by_part: Dict[int, float]) -> Optional[int]:
    """
    The metal-path part code with the smallest robust thickness.

    Ties break in flow order (sprue first), so the message names the part
    furthest upstream -- the one the metal meets first.
    """
    if not by_part:
        return None
    return min(
        (code for code in _METAL_PATH_CODES if code in by_part),
        key=lambda code: by_part[code],
    )


def measure_choke_thickness_cells(dense_types: np.ndarray) -> float:
    """
    Thickness of the thinnest gating section, in fill-grid cells.

    The scalar summary of :func:`measure_choke_thickness_by_part` -- the
    smallest of the per-part figures. See that function for what is measured
    and why the median rather than the maximum.

    Returns:
        Thickness in cells at the narrowest section, or 0.0 when there is no
        gating metal.
    """
    by_part = measure_choke_thickness_by_part(dense_types)
    if not by_part:
        return 0.0
    return max(0.0, float(min(by_part.values())))


# ---------------------------------------------------------------------------
# Stage orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FillFrame:
    """
    One instant of the pour, gathered onto voxels.

    Attributes:
        time_s: Physical time since the pour started.
        fill_fraction: Per-voxel metal content, 0 (empty) to 1 (full). The
            partial values are the free surface -- the interface layer is
            exactly where 0 < eps < 1, so a threshold on this field draws the
            advancing front rather than a staircase of whole cells.
        speed_m_s: Per-voxel velocity magnitude, in metres per second.
    """

    time_s: float
    fill_fraction: np.ndarray
    speed_m_s: np.ndarray


@dataclass(frozen=True)
class FillingResult:
    """
    What the filling stage hands back to the pipeline.

    ``fill_time_s`` and ``temperature_c`` are ``(n_voxels,)`` -- the *occupied*
    voxel count, which is what
    ``SolidificationEngine.set_initial_temperature_field`` requires. They are
    not dense-grid shaped.

    Attributes:
        fill_time_s: Arrival time per voxel; ``-1`` where metal never arrived.
        temperature_c: Initial temperature per voxel, degrees Celsius. F2:
            the metal's actual thermal history through the fill -- the metal
            that entered first and travelled furthest starts cooler than
            metal at the gate -- not the two-constant (pour temp / mold temp)
            placeholder F1 used.
        inlet_voxel_count: Voxels the pour location resolved to.
        filled_voxel_count: Metal voxels the fill reached.
        misrun_voxel_count: Metal voxels it did not.
        choke_area_cm2: Narrowest gating cross-section, measured **on the grid
            the solver ran** -- the coarsened one, when
            ``fill_grid_coarsening`` is above 1. That is the choke the fill
            actually had.
        filter_zones: One entry per ceramic filter found.
        steps: Lattice steps taken.
        fill_seconds: Physical time simulated.
        coarsening: Fill-grid coarsening factor used.
        min_fill_temperature_c: Coolest filled metal at handover. Below the
            alloy's liquidus this is incipient cold-shut territory.
        mean_fill_temperature_c: Average filled-metal temperature at handover.
        entrapped_voxel_count: Voxels of trapped gas at the end of the fill
            (F3). Zero when every gas pocket found a vent.
        peak_entrapped_voxel_count: The largest entrapped-gas volume seen at
            any periodic check during the fill, not just at the end -- a
            pocket can be entrapped mid-fill and vent later, so the final
            snapshot alone understates the risk. Always ``>=
            entrapped_voxel_count``.
        cold_shut_voxel_count: Voxels flagged as a cold-shut seam (F3): a
            wet cell where fronts converged from two or more directions and
            the metal had already cooled to within
            ``FillingParams.cold_shut_temp_margin_c`` of the liquidus.
        entrapped_gas: Per-voxel boolean, ``(n_voxels,)``, the final
            entrapped-gas set gathered the same way ``fill_time_s`` is.
        cold_shut: Per-voxel boolean, ``(n_voxels,)``, the cold-shut set.
        frames: Playback snapshots of the pour, oldest first. Empty when the
            run diverged, because a frame taken after the blow-up shows the
            blow-up. See :class:`FillFrame`.
        warnings: Human-readable conditions worth surfacing.
        status: How the run ended. One of ``STATUS_COMPLETE`` (every metal
            cell filled), ``STATUS_TIME_CAPPED`` (still advancing when
            ``max_fill_time_s`` ran out), ``STATUS_STALLED`` (the front stopped
            advancing while metal remained unfilled) or ``STATUS_DIVERGED``
            (the solver lost mass conservation -- the fields are numerically
            meaningless and must not be trusted or handed downstream).

            This distinction is the difference between "the metal could not
            get there", "the run needed longer" and "the answer is garbage".
            All three used to surface identically as a misrun count.
        diverged_at_step: Lattice step at which divergence was detected, or 0.
        choke_thickness_cells: Thickness of the thinnest gating section, in
            fill-grid cells. At or below ``MIN_CHOKE_THICKNESS_CELLS`` the
            channel has no interior and halfway bounce-back seals it shut.
        choke_thickness_by_part: ``{"sprue"|"runner"|"filter"|"gate":
            thickness in cells}`` for every metal-path part present, so a
            reader can see *which* body is under-resolved rather than only
            that something is.
        narrowest_choke_part: Name of the thinnest of them, or ``""`` when the
            grid holds no gating metal.
        choke_area_fine_cm2: The same cross-section measured on the *mesh*
            grid, before coarsening. Reported alongside ``choke_area_cm2``
            rather than instead of it, because the two answer different
            questions -- what the geometry has, and what the solver saw. They
            are not close: 2.98 against 7.46 cm^2 on the ``Test 2`` assembly.
    """

    fill_time_s: np.ndarray
    temperature_c: np.ndarray
    inlet_voxel_count: int
    filled_voxel_count: int
    misrun_voxel_count: int
    choke_area_cm2: float
    filter_zones: List[FilterZoneSpec]
    steps: int
    fill_seconds: float
    coarsening: int
    min_fill_temperature_c: float
    mean_fill_temperature_c: float
    entrapped_voxel_count: int = 0
    peak_entrapped_voxel_count: int = 0
    cold_shut_voxel_count: int = 0
    entrapped_gas: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    cold_shut: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    warnings: Tuple[str, ...] = ()
    status: str = STATUS_COMPLETE
    diverged_at_step: int = 0
    choke_thickness_cells: float = 0.0
    choke_thickness_by_part: Dict[str, float] = field(default_factory=dict)
    narrowest_choke_part: str = ""
    choke_area_fine_cm2: float = 0.0
    frames: Tuple[FillFrame, ...] = ()

    @property
    def filled_fraction(self) -> float:
        total = self.filled_voxel_count + self.misrun_voxel_count
        return self.filled_voxel_count / total if total else 0.0

    @property
    def trustworthy(self) -> bool:
        """
        Whether the fields may be used downstream.

        A diverged run still carries a ``fill_time_s`` and a ``temperature_c``
        of the right shape and dtype -- they are simply meaningless, because
        the blow-up flags cells as filled that the metal never reached. Seeding
        solidification from them produces a plausible-looking casting result
        built on nothing, which is worse than not running the stage at all.
        """
        return self.status != STATUS_DIVERGED


def _obstacle_mask(dense_types: np.ndarray) -> np.ndarray:
    """
    Cells the metal cannot enter.

    Empty cells (outside the meshed geometry) and mold materials are solid --
    **except the filter**, which metal flows *through*. ``is_mold()`` alone
    would make a ceramic filter an impermeable wall and stop the fill dead at
    the gate, which is why the flow-restrictor axis exists separately.
    """
    from core.part_manager import is_flow_restrictor, is_mold

    types = np.asarray(dense_types)
    empty = types == EMPTY_CELL
    safe = np.where(empty, PART_TYPE_CAST, types)
    solid: np.ndarray = empty | (is_mold(safe) & ~is_flow_restrictor(safe))
    return solid


def _metal_mask(dense_types: np.ndarray) -> np.ndarray:
    """Cells that should end up holding metal."""
    from core.part_manager import is_metal

    types = np.asarray(dense_types)
    empty = types == EMPTY_CELL
    safe = np.where(empty, PART_TYPE_CHILL, types)
    metal: np.ndarray = (~empty) & is_metal(safe)
    return metal


def _resolve_inlet_cells(
    pour_location,
    mapping: GridMapping,
    dense_types: np.ndarray,
    coarsening: int,
) -> np.ndarray:
    """
    Turn the user's clicked point into a set of lattice cells.

    Works in lattice index space rather than voxel indices, because the fill
    grid may be coarser than the mesh and voxel indices do not survive
    downsampling.

    Only metal cells are eligible: a click that lands on a chill, a core or
    empty space is a user error, and injecting metal there would produce a
    fill that starts inside the mold.
    """
    from core.pour_location import resolve_inlet_voxels

    voxel_idx = resolve_inlet_voxels(
        pour_location,
        _voxel_centers_from_mapping(mapping),
        mapping.dx_in,
    )
    if voxel_idx.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    ijk = mapping.voxel_to_grid[voxel_idx].astype(np.int64)
    if coarsening > 1:
        ijk = ijk // coarsening
        ijk = np.unique(ijk, axis=0)

    keep = _metal_mask(dense_types)[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    selected: np.ndarray = ijk[keep]
    return selected


def _expand_inlet_to_mouth(
    inlet: np.ndarray,
    dense_types: np.ndarray,
) -> np.ndarray:
    """
    Widen the resolved inlet from a point to the whole mouth of its feature.

    The pour location's job is to pick *which* feature is poured into. The
    physical inlet is that feature's entire top cross-section, because a
    pouring basin sits on the full mouth of the sprue, not on a spot in the
    middle of it.

    Measured on ``Test_Model/Full_Gating_Test``: :func:`_resolve_inlet_cells`'
    1.5-voxel radius picked 8 cells of the sprue top's cross-section -- 44
    cells when that was first measured, 48 on the current mesh. The 8-cell jet
    atomised on the way down the sprue and the run diverged at 24 % by mass.
    Feeding the same pour through the whole mouth -- changing nothing else --
    cleared the divergence and ran the sprue full.

    The flood is confined to the seeds' own top layer and to the part types the
    seeds are standing on, so a pour cup widens over the cup, a sprue over the
    sprue and a bare casting over its own top face. None of them leak into a
    neighbouring part that happens to share the layer, and none of them leak
    downwards into the feature's bore.

    It widens the layer the click landed in, not "the topmost layer of the
    feature", so a pour location placed on a *side* face resolves to the whole
    horizontal cross-section at that height rather than to a patch of the
    flank. That is the intended reading -- the click chooses the feature and
    the metal enters across it -- but it is why the UI asks for a point on the
    top face of the sprue.

    Args:
        inlet: ``(n, 3)`` seed cells, as :func:`_resolve_inlet_cells` returns.
        dense_types: The part-type grid those cells index into.

    Returns:
        ``(m, 3)`` int64 cells: the seeds plus the flooded mouth, ``m >= n``.
    """
    from scipy import ndimage

    seeds = np.asarray(inlet, dtype=np.int64).reshape(-1, 3)
    if seeds.shape[0] == 0:
        return seeds

    types = np.asarray(dense_types)
    z_top = int(seeds[:, 2].max())
    top = seeds[seeds[:, 2] == z_top]
    layer = types[:, :, z_top]

    # Only the types the seeds actually stand on, and only metal ones: the
    # flood must not walk out of the feature being poured into.
    codes = np.unique(layer[top[:, 0], top[:, 1]])
    member = np.isin(layer, codes) & _metal_mask(layer)
    if not member.any():
        return seeds

    # 8-connectivity in the plane. A sprue mouth is a disc drawn on a lattice;
    # its corner-touching cells are one body, not several.
    labels, _ = ndimage.label(member, structure=np.ones((3, 3), dtype=bool))
    seed_labels = np.unique(labels[top[:, 0], top[:, 1]])
    seed_labels = seed_labels[seed_labels > 0]
    if seed_labels.size == 0:
        return seeds

    mx, my = np.nonzero(np.isin(labels, seed_labels))
    mouth = np.column_stack(
        (mx, my, np.full(mx.shape[0], z_top))
    ).astype(np.int64)
    widened: np.ndarray = np.unique(np.vstack((seeds, mouth)), axis=0)
    return widened


def _voxel_centers_from_mapping(mapping: GridMapping) -> np.ndarray:
    """Reconstruct voxel centres in inches from the mapping."""
    return mapping.origin_in[None, :] + mapping.voxel_to_grid.astype(np.float64) * mapping.dx_in


def _upsample(coarse: np.ndarray, fine_shape: Tuple[int, int, int], factor: int) -> np.ndarray:
    """Nearest-neighbour expansion of a coarse field back to the fine grid."""
    if factor == 1:
        return coarse
    idx = [np.minimum(np.arange(n) // factor, coarse.shape[a] - 1)
           for a, n in enumerate(fine_shape)]
    return coarse[np.ix_(idx[0], idx[1], idx[2])]


def run_filling_stage(
    voxel_mesh,
    alloy,
    analysis_options,
    pour_location,
    *,
    cancel_callback=None,
    progress_message_callback=None,
    detailed_progress_callback=None,
) -> Optional[FillingResult]:
    """
    Simulate the pour and return the state it leaves behind.

    Args:
        voxel_mesh: A generated :class:`~core.voxel_mesh.AdaptiveVoxelMesh`
            with part types assigned.
        alloy: An :class:`~core.materials.AlloyProperties`.
        analysis_options: Carries ``solidification_params.filling``.
        pour_location: A :class:`~core.pour_location.PourLocation`.
        cancel_callback: Raises to abort.
        progress_message_callback: Receives human-readable progress.

    Returns:
        A :class:`FillingResult`, or ``None`` if the stage could not run. The
        reason is emitted through ``progress_message_callback`` rather than
        raised: an unfillable pour point is a modelling problem the user should
        see and fix, not a crash.
    """
    from physics.filling import (
        FLAG_FLUID,
        FLAG_GAS,
        FLAG_INTERFACE,
        FLAG_OBSTACLE,
        REFERENCE_DENSITY,
        advect_temperature,
        apply_wall_heat_loss,
        create_state,
        ensure_interface_skin,
        find_cold_shuts,
        find_entrapped_gas,
        fluid_fraction,
        step,
    )
    from physics.fluid_flow import LATTICE_VELOCITIES, equilibrium, lattice_units
    from core.part_manager import CODE_TO_PART_TYPE, is_gating_metal

    def emit(message: str) -> None:
        if progress_message_callback is not None:
            progress_message_callback(message)

    params = getattr(analysis_options, "solidification_params", None)
    filling = getattr(params, "filling", None)
    if filling is None:
        emit("Filling stage skipped: no filling parameters configured.")
        return None
    thermal_boundary = getattr(params, "thermal_boundary", None)

    # 0 means "use the alloy's own pouring temperature" -- see FillingParams'
    # docstring for pour_temperature_c.
    pour_temp = float(getattr(filling, "pour_temperature_c", 0.0)) or float(alloy.pouring_temp)
    mold_temp = float(getattr(thermal_boundary, "mold_temperature", 21.0))

    centers = np.asarray(voxel_mesh.voxel_centers, dtype=np.float64)
    dx_in = float(np.asarray(voxel_mesh.voxel_sizes).ravel()[0])
    try:
        mapping = build_grid_mapping(centers, dx_in)
    except ValueError as exc:
        emit(f"Filling stage skipped: {exc}")
        return None

    fine_types = dense_part_types(mapping, np.asarray(voxel_mesh.voxel_part_types))
    coarsening = max(1, int(getattr(filling, "fill_grid_coarsening", 2)))
    solve_types = coarsen_part_types(fine_types, coarsening)

    inlet_seeds = _resolve_inlet_cells(pour_location, mapping, solve_types, coarsening)
    if inlet_seeds.shape[0] == 0:
        emit(
            "Filling stage skipped: the pour location does not resolve to any "
            "metal cell. Move it onto the pour cup, sprue or casting."
        )
        return None

    # The click picks the feature; the mouth of that feature is the inlet. This
    # is geometry resolution rather than boundary mechanics, so it applies to
    # both inlet modes. In velocity mode the prescribed mass rate is divided by
    # the now-larger inlet area, which lowers the imposed velocity for the same
    # kg/s -- correct, and exactly what a basin sitting on the whole sprue
    # does. See _expand_inlet_to_mouth for the measurement that made this
    # necessary: 8 cells of a sprue mouth is a jet, not a pour.
    inlet = _expand_inlet_to_mouth(inlet_seeds, solve_types)
    inlet_seed_count = int(inlet_seeds.shape[0])
    inlet_widening = ""
    if inlet.shape[0] > inlet_seed_count:
        mouth_top = inlet[:, 2] == inlet[:, 2].max()
        mouth_names = [
            CODE_TO_PART_TYPE[int(code)].value.replace("_", " ")
            for code in np.unique(
                solve_types[inlet[mouth_top, 0], inlet[mouth_top, 1],
                            inlet[mouth_top, 2]]
            )
            if int(code) in CODE_TO_PART_TYPE
        ]
        inlet_widening = (
            f" ({inlet_seed_count:,} at the pour point, widened to the "
            f"{'/'.join(mouth_names) or 'poured feature'} mouth)"
        )

    dx_m = mapping.dx_m * coarsening
    nu = float(alloy.dynamic_viscosity) / float(alloy.density_liquid)

    # The timestep follows from the fastest velocity that must be
    # representable. Free fall from the top of the assembly, sqrt(2 g H), is a
    # defensible upper bound: nothing in a gravity pour moves faster than metal
    # dropped from the highest point of the rigging. Under-estimating it breaks
    # the Mach guard; over-estimating it only costs steps.
    height_m = float(inches_to_meters(
        (mapping.shape[2] - 1) * mapping.dx_in
        + float(getattr(filling, "initial_head_in", 4.0))
    ))
    u_max_physical = max(0.5, float(np.sqrt(2.0 * 9.81 * max(height_m, 1e-6))))
    units = lattice_units(
        dx_m,
        nu,
        u_max_physical_m_s=u_max_physical,
        u_max_lattice=float(getattr(filling, "lattice_velocity_max", 0.05)),
    )

    obstacle = _obstacle_mask(solve_types)
    metal = _metal_mask(solve_types)
    if not np.any(metal):
        emit("Filling stage skipped: no metal cells in the assembly.")
        return None

    filter_zones = build_filter_zone_specs(
        solve_types,
        dx_m=dx_m,
        porosity=float(getattr(filling.filter, "porosity", 0.85)),
        pore_diameter_m=float(getattr(filling.filter, "pore_diameter_mm", 2.54)) / 1000.0,
        ergun_prefactor=float(getattr(filling.filter, "ergun_prefactor", 2.2)),
        forchheimer_c_f=float(getattr(filling.filter, "forchheimer_c_f", 0.55)),
        surface_tension_n_m=float(getattr(alloy, "surface_tension", 1.8)),
        contact_angle_deg=float(getattr(filling.filter, "contact_angle_deg", 120.0)),
        window_fraction=float(getattr(filling.filter, "window_fraction", 0.45)),
        density_kg_m3=float(alloy.density_liquid),
    )
    porous = build_porous_zone(
        solve_types,
        filter_zones,
        dx_m=dx_m,
        dt_s=units.dt_s,
        nu_lattice=units.nu_lattice,
        density_kg_m3=float(alloy.density_liquid),
    )

    seed = np.zeros(solve_types.shape, dtype=bool)
    seed[inlet[:, 0], inlet[:, 1], inlet[:, 2]] = True
    state = create_state(
        obstacle,
        dx_m=dx_m,
        dt_s=units.dt_s,
        initial_fluid=seed,
        porous=porous,
        initial_temperature_c=pour_temp,
    )

    # F2 thermal coupling. alpha_lattice is rescaled exactly the way
    # lattice_units() rescales viscosity into nu_lattice, so it drops
    # straight into physics.filling's explicit stencil as a dimensionless
    # number -- see tests/test_filling_thermal.py for the stability margin
    # this buys at realistic settings.
    alpha_si = float(alloy.thermal_conductivity) / (
        float(alloy.density_liquid) * float(alloy.specific_heat)
    )
    alpha_lattice = alpha_si * units.dt_s / (dx_m * dx_m)

    # Gating-system surfaces (pour cup, sprue, runner, gate) and the casting
    # itself sit in different sand contexts and lose heat at different rates
    # -- see ThermalBoundaryParams.htc_gating's docstring for why an
    # unconfigured gating surface would otherwise integrate adiabatically.
    wall_htc_multiplier = float(getattr(filling, "wall_htc_multiplier", 1.0))
    htc_gating = float(getattr(thermal_boundary, "htc_gating", 400.0)) * wall_htc_multiplier
    htc_cast = float(getattr(thermal_boundary, "htc_cast_interface", 500.0)) * wall_htc_multiplier
    htc_dense = np.where(is_gating_metal(solve_types), htc_gating, htc_cast)

    max_time = float(getattr(filling, "max_fill_time_s", 60.0))
    max_steps = int(max_time / units.dt_s)
    emit(
        f"Filling: {int(metal.sum()):,} cells at {dx_m * 1000:.1f} mm "
        f"(coarsening x{coarsening}), up to {max_steps:,} steps"
    )

    warnings: List[str] = []

    # The subgrid constant the collision operator runs at. Out-of-range values
    # are refused rather than clamped: silently pulling 1.2 back to 0.5 would
    # still hand the user a viscosity 40x what they think they asked for, and
    # the whole reason this knob is visible is that a control which quietly
    # does something else is worse than no control. NaN fails the comparison
    # too, which is the intent.
    smagorinsky_cs = float(getattr(filling, "smagorinsky_cs", SMAGORINSKY_CS))
    if not 0.0 <= smagorinsky_cs <= MAX_SMAGORINSKY_CS:
        warnings.append(
            f"smagorinsky_cs of {smagorinsky_cs:.3f} is outside the usable "
            f"range 0-{MAX_SMAGORINSKY_CS:.1f} (literature values are "
            f"0.1-0.2); running at the default {SMAGORINSKY_CS:.2f}."
        )
        smagorinsky_cs = float(SMAGORINSKY_CS)

    # Inlet density from the metallostatic head above the pour point:
    # rho = rho_0 (1 + g h / c_s^2) in lattice units. Imposing a *pressure*
    # rather than a velocity is what lets the sprue choke meter the flow, which
    # is the behaviour Foseco's own gating guidance asks for.
    head_m = float(inches_to_meters(float(getattr(filling, "initial_head_in", 4.0))))
    head_lattice = head_m / dx_m
    inlet_density = REFERENCE_DENSITY * (
        1.0 + units.gravity_lattice * head_lattice / (1.0 / 3.0)
    )

    # What the inlet imposes -- and what it must not.
    #
    # The original source rebuilt the inlet populations from ``state.u``, the
    # *Guo force-corrected* velocity, which carries half of that step's
    # gravitational impulse by construction (see guo_velocity's docstring).
    # Feeding it back into equilibrium() re-imposed that half-impulse every
    # step with nothing to remove it, so the inlet gained 0.5*g_lattice of
    # velocity per step forever and took the whole solve with it.
    #
    # Its replacement -- a constant *prescribed* velocity, Torricelli's
    # sqrt(2 g h) in head mode -- could not accumulate, and could not work
    # either. A fixed-velocity equilibrium pin cannot feel back-pressure: it
    # overwrites whatever the domain streams back into it. Instrumented on
    # Test_Model/Filling_Test at 0.20 in voxels, coarsening 2, that produced
    # three measurements, and they are the reason this design is gone:
    #
    #   * Delivery collapsed. Net mass added by the inlet fell from 0.110
    #     lattice-mass/step -- the full prescribed rate -- to 0.006 by step
    #     28,000, about 6 %, and then went negative: the pin was deleting
    #     metal that had streamed back into it.
    #   * The column starved. With delivery choked, the whole sprue sat in
    #     suction: mean rho 0.99995 at the sprue bottom against a hydrostatic
    #     requirement of ~1.0028. No head developed, so nothing pushed the
    #     metal onward, and 428 casting cells crept at eps ~ 0.5, about 50x
    #     slower than the geometry allows.
    #   * The inlet seeded a staggered oscillation. Per-cell mass in the top
    #     sprue layers reached +-218 against a physical range of [0, ~1], u_z
    #     alternating sign per z-layer, total wet mass swinging 170 <-> 2,815
    #     between 200-step samples, until the divergence guard fired at step
    #     126,400.
    #
    # So head mode imposes PRESSURE and lets the velocity be whatever the flow
    # field itself produces. The pressure is the metallostatic head of a
    # pouring basin held topped up, which is exactly what ``initial_head_in``
    # has always described. Run freely, gravity accelerates the column and the
    # inlet feeds it; back the downstream up and the measured velocity falls,
    # so delivery falls with it -- what a real basin does, and what a
    # prescribed velocity cannot do. Momentum cannot accumulate, because the
    # velocity is re-measured from the populations every step and the rebuild
    # adds no force term. The measurement is the RAW first moment, deliberately
    # without the Guo half-force correction: that correction is precisely the
    # term that made the original a pump.
    #
    # Velocity mode still prescribes a constant velocity from the pour rate.
    # There the user is asserting a delivery rate, and a boundary that meters
    # itself would quietly ignore them.
    inlet_area_m2 = float(inlet.shape[0]) * dx_m * dx_m
    inlet_mode = int(getattr(filling, "inlet_mode_code", 0))
    ix, iy, iz = inlet[:, 0], inlet[:, 1], inlet[:, 2]
    u_inlet_physical = 0.0
    inlet_mode_label = ""
    if inlet_mode == 1:
        # 0 means "derive", the same convention pour_time_s and
        # pour_temperature_c already use. The square-root rule ties the rate to
        # this casting's own weight; a fixed number cannot, and 5 lb/s is
        # wrong for almost every casting that is not about 25 lb.
        weight_lb = pour_weight_lb(voxel_mesh, alloy)
        pour_rate_lb_s = float(getattr(filling, "pour_rate_lb_s", 0.0) or 0.0)
        if pour_rate_lb_s <= 0.0:
            pour_rate_lb_s = recommended_pour_rate_lb_s(weight_lb)
            rate_source = f"sqrt of {weight_lb:.0f} lb"
        else:
            rate_source = "set"
        if pour_rate_lb_s <= 0.0:
            # No part types, so no weight to take a root of. Fall back to the
            # head rather than dividing by zero and reporting a still pour.
            warnings.append(
                "velocity inlet requested but the pour weight could not be "
                "measured; falling back to the metallostatic head."
            )
            inlet_mode = 0
        else:
            pour_rate_kg_s = pour_rate_lb_s * 0.45359237
            u_inlet_physical = pour_rate_kg_s / max(
                float(alloy.density_liquid) * inlet_area_m2, 1e-9
            )
            inlet_mode_label = (
                f"velocity ({pour_rate_lb_s:.1f} lb/s, {rate_source}; "
                f"{pour_rate_lb_s:.1f} s pour)"
            )
    if inlet_mode != 1:
        inlet_mode_label = f"head ({head_m * 39.3701:.1f} in)"

    # The filter ceiling, applied to BOTH inlet modes. A filter restricts the
    # flow path itself, so it binds whatever is driving the metal -- a deep
    # head does not entitle the pour to exceed the filter's rating, it just
    # means the sprue backs up. In velocity mode it lowers the prescribed
    # velocity. In head mode nothing is prescribed to lower, so it lowers the
    # *cap* the measured velocity is clamped against instead: the restriction
    # is still modelled, it just binds a measurement rather than a setting.
    flow_limit_kg_s, flow_limit_note = filter_flow_limit_kg_s(
        filter_zones, alloy, int(getattr(filling, "pour_mode_code", 0))
    )
    if flow_limit_note:
        warnings.append(flow_limit_note)
    u_filter_max: Optional[float] = None
    if flow_limit_kg_s is not None:
        u_filter_max = flow_limit_kg_s / max(
            float(alloy.density_liquid) * inlet_area_m2, 1e-9
        )
        if inlet_mode == 1 and u_inlet_physical > u_filter_max:
            requested_lb_s = (
                u_inlet_physical * float(alloy.density_liquid) * inlet_area_m2
                / 0.45359237
            )
            warnings.append(
                f"pour rate limited by the filter: {requested_lb_s:.1f} lb/s "
                f"requested, {flow_limit_kg_s / 0.45359237:.1f} lb/s rated "
                f"across {len(filter_zones)} filter(s). Running at the rating. "
                f"Above it a ceramic foam filter blinds off or breaks up "
                f"rather than throttling."
            )
            u_inlet_physical = u_filter_max
            inlet_mode_label += ", filter-limited"

    # The Mach guard the timestep was sized for, lowered by the filter rating
    # where one applies. Nothing the inlet writes may exceed it: a boundary
    # condition that violates the solver's own stability limit is not a
    # boundary condition.
    u_cap_lattice = float(units.u_max_lattice)
    if u_filter_max is not None:
        u_cap_lattice = min(
            u_cap_lattice,
            float(units.to_lattice_velocity(np.asarray(u_filter_max, dtype=np.float64))),
        )
        if inlet_mode != 1 and u_cap_lattice < float(units.u_max_lattice):
            inlet_mode_label += ", filter-limited"

    inlet_rho = np.full(inlet.shape[0], inlet_density, dtype=np.float64)
    inlet_u = np.zeros((3, inlet.shape[0]), dtype=np.float64)
    if inlet_mode == 1:
        u_inlet_lattice = float(units.to_lattice_velocity(u_inlet_physical))
        if u_inlet_lattice > units.u_max_lattice:
            warnings.append(
                f"inlet velocity {u_inlet_physical:.2f} m/s exceeds the Mach guard "
                f"the timestep was sized for; clamped to "
                f"{units.u_max_lattice * units.velocity_scale:.2f} m/s. Raise "
                f"lattice_velocity_max or lower the pour rate."
            )
            u_inlet_lattice = float(units.u_max_lattice)
        # Gravity points -Z (physics.filling.step's default), and metal enters
        # falling, so the inlet drives along it.
        inlet_u[2, :] = -u_inlet_lattice
        # Velocity mode's mass pin is unchanged: a whole cell's worth, every
        # step, matching the constant delivery it is asserting.
        inlet_full_mass = np.full(
            inlet.shape[0], float(REFERENCE_DENSITY), dtype=np.float64
        )
        emit(
            f"Filling: inlet {inlet.shape[0]:,} cells{inlet_widening}, "
            f"{inlet_mode_label}, "
            f"{u_inlet_physical:.2f} m/s ({u_inlet_lattice:.4f} lattice)"
        )
    else:
        # A full cell holds rho * porosity -- apply_transitions' own definition
        # of full. Pinning REFERENCE_DENSITY instead would read as under-filled
        # the moment the free-surface pass looked at an inlet inside a filter.
        inlet_full_mass = inlet_rho * np.asarray(
            state.porous.porosity[ix, iy, iz], dtype=np.float64
        )
        emit(
            f"Filling: inlet {inlet.shape[0]:,} cells{inlet_widening}, "
            f"{inlet_mode_label} "
            f"pressure boundary, rho {inlet_density:.5f} "
            f"(+{(inlet_density / REFERENCE_DENSITY - 1.0) * 100.0:.3f}% over "
            f"reference); velocity follows the flow, capped at "
            f"{u_cap_lattice * units.velocity_scale:.2f} m/s "
            f"({u_cap_lattice:.4f} lattice)"
        )

    # Whether the head can reach the top of the metal at all.
    #
    # Metal does not climb above the free surface of the basin feeding it, so
    # a pour point plus head sitting below the highest metal cell cannot
    # finish the casting however long the solve runs. Without this the run
    # spends its whole budget proving that and then reports a time cap, which
    # reads as "needed longer" when it means "cannot get there from here".
    # Head mode only: velocity mode is a pump, and a pump has no free surface.
    if inlet_mode != 1:
        solve_dx_in = mapping.dx_in * coarsening
        z_centres_in = float(mapping.origin_in[2]) + (
            np.arange(solve_types.shape[2], dtype=np.float64) * coarsening
            + (coarsening - 1) / 2.0
        ) * mapping.dx_in
        metal_top_z_in = float(
            z_centres_in[np.nonzero(np.any(metal, axis=(0, 1)))[0].max()]
        )
        inlet_z_in = float(z_centres_in[iz].max())
        head_reach_in = inlet_z_in + head_m * 39.3701
        if head_reach_in < metal_top_z_in + solve_dx_in:
            warnings.append(
                f"the metallostatic head cannot reach the top of the metal, so "
                f"the fill physically cannot complete: the pour point sits at "
                f"z = {inlet_z_in:.2f} in and {head_m * 39.3701:.1f} in of head "
                f"reaches {head_reach_in:.2f} in, while metal extends to "
                f"z = {metal_top_z_in:.2f} in. Raise the pour point or "
                f"increase initial_head_in."
            )

    # An under-resolved choke is not a small inaccuracy -- it is the
    # difference between a fill and a sealed channel -- and it is measured
    # here rather than left for the user to infer from a misrun count.
    #
    # Measured on solve_types: the grid the solver runs on, not the mesh it
    # was coarsened from. A gate that is four cells across the mesh is two
    # across a coarsening of 2, and it is the two that seals.
    #
    # Reported per part, and warned about per part. One number for "the gating
    # system" cannot say which body the metal will not cross, and the answer
    # is usually the gate while the sprue and runner are fine.
    choke_by_part = measure_choke_thickness_by_part(solve_types)
    choke_thickness_cells = (
        min(choke_by_part.values()) if choke_by_part else 0.0
    )
    narrowest_code = narrowest_choke_part_code(choke_by_part)
    if choke_by_part:
        emit(
            "Filling: channel thickness on the "
            f"{dx_m * 1000:.1f} mm fill grid -- "
            + ", ".join(
                f"{_METAL_PATH_LABELS[code]} {choke_by_part[code]:.1f} cells"
                for code in _METAL_PATH_CODES
                if code in choke_by_part
            )
            + (
                f" (narrowest: {_METAL_PATH_LABELS[narrowest_code]})"
                if narrowest_code is not None
                else ""
            )
        )
    for code in _METAL_PATH_CODES:
        thickness = choke_by_part.get(code)
        if thickness is None or thickness <= 0.0:
            continue
        label = _METAL_PATH_LABELS[code]
        inches = thickness * dx_m * 39.3701
        if thickness <= MIN_CHOKE_THICKNESS_CELLS:
            warnings.append(
                f"the {label} is {thickness:.1f} cells thick on a "
                f"{dx_m * 1000:.1f} mm fill grid ({inches:.2f} in) -- at or "
                f"below the {MIN_CHOKE_THICKNESS_CELLS:.0f} cells a channel "
                f"needs before it has any interior. Halfway bounce-back seals "
                f"it: this grid cannot resolve the {label}, and the fill "
                f"through it cannot be trusted whatever this run reports. "
                f"Reduce voxel size or fill_grid_coarsening."
            )
        elif thickness <= MARGINAL_CHOKE_THICKNESS_CELLS:
            warnings.append(
                f"the {label} resolves to {thickness:.1f} cells across on a "
                f"{dx_m * 1000:.1f} mm fill grid ({inches:.2f} in), which is "
                f"the floor at which this solver has ever produced a usable "
                f"fill -- Test_Model/Filling_Test, whose gate is also three "
                f"cells, time-caps at 57% of cells. Treat the flow through "
                f"the {label} as marginal, not as a result."
            )

    # F3: air entrapment. The vent mask is reassembled at every periodic
    # check (see below) because its "domain boundary gas" component is a
    # live snapshot -- built once here, "static_vent" only covers the parts
    # that do not change as the fill progresses (open riser tops, the
    # inlet).
    static_vent = _static_vent_mask(
        solve_types, inlet, bool(getattr(filling, "vent_risers", 1))
    )
    boundary_mask = _domain_boundary_mask(solve_types.shape)
    entrapment_check_interval = max(
        1, int(getattr(filling, "entrapment_check_interval", 50))
    )
    # Peak rather than final: a pocket can be entrapped at one moment and
    # vent later as the geometry around it changes, so checking only at the
    # end understates the risk (design doc section 6.4).
    peak_entrapped_voxels = 0
    vent_warning: Optional[str] = None

    report_every = max(1, max_steps // 20)
    # Captured on an adaptive stride, not on max_steps / FILL_ANIMATION_FRAMES.
    # A run almost never uses its whole budget -- a good pour finishes in a
    # few seconds of a 60 s cap, and a stalled one gives up early -- so a
    # stride sized for the budget yields a handful of frames exactly when the
    # animation matters most. Instead sample finely and halve the stride
    # whenever the buffer doubles, which keeps the frames evenly spaced across
    # however long the run actually turns out to be, for a bounded cost.
    frame_every = max(1, max_steps // (FILL_ANIMATION_FRAMES * 4))
    frames: List[FillFrame] = []

    def capture_frame(step_index: int) -> None:
        """Gather the live free surface and velocity onto voxels."""
        eps = np.clip(fluid_fraction(state), 0.0, 1.0)
        eps[state.flags == FLAG_OBSTACLE] = 0.0
        speed = np.sqrt(np.sum(np.square(state.u.astype(np.float64)), axis=0))
        speed *= units.velocity_scale
        frames.append(
            FillFrame(
                time_s=step_index * units.dt_s,
                fill_fraction=gather_to_voxels(
                    mapping, _upsample(eps, mapping.shape, coarsening)
                ).astype(np.float32),
                speed_m_s=gather_to_voxels(
                    mapping, _upsample(speed, mapping.shape, coarsening)
                ).astype(np.float32),
            )
        )

    taken = 0
    metal_total = int(metal.sum())
    # Mass the metal region holds when full. Full is rho * porosity -- the same
    # definition apply_transitions converts on -- so inside a filter this sits
    # below the cell count, which is correct: a filter cell full of metal holds
    # only its pore volume.
    metal_full_mass = float(
        np.sum(
            float(REFERENCE_DENSITY)
            * np.asarray(state.porous.porosity, dtype=np.float64)[metal]
        )
    )
    # Net mass the inlet has injected, in lattice mass units, accumulated at
    # each pin: what the pin writes, minus what was there to overwrite. The
    # conversion to kilograms is total_mass_kg's: one lattice mass unit at
    # REFERENCE_DENSITY is one cell volume of liquid metal.
    poured = 0.0
    poured_lb_per_lattice_mass = (
        float(alloy.density_liquid) * (dx_m ** 3)
        / float(REFERENCE_DENSITY) / 0.45359237
    )
    # Hoisted out of the loop: the inlet's raw first moment is measured every
    # step from a (19, n_inlet) slice, and n_inlet is a handful of cells.
    lattice_c_t = LATTICE_VELOCITIES.T.astype(np.float64)
    inlet_u_measured = np.zeros((3, inlet.shape[0]), dtype=np.float64)

    # The cells the inlet streams into, so the pin can keep them wet.
    #
    # Pinning FLAG_FLUID by hand puts the inlet outside apply_transitions'
    # bookkeeping, and with it outside the scheme's central invariant: no FLUID
    # cell is ever directly adjacent to a GAS cell. Measured on a 2x2x12 column
    # -- the cell above the inlet drained to gas, streaming zeroed it, and the
    # inlet stopped receiving every population arriving with c_z = -1, one
    # sixth of the weight. The measured z-momentum jumped from -0.009 (falling)
    # to +0.147 (rising) in one step and the whole column reversed at the Mach
    # cap. Rolled with the same periodic wrap the streaming kernel uses, so the
    # halo is exactly the set of cells the inlet can stream into.
    inlet_mask = np.zeros(solve_types.shape, dtype=bool)
    inlet_mask[ix, iy, iz] = True
    inlet_halo = np.zeros(solve_types.shape, dtype=bool)
    for _c in LATTICE_VELOCITIES[1:]:
        inlet_halo |= np.roll(
            inlet_mask, shift=(int(_c[0]), int(_c[1]), int(_c[2])), axis=(0, 1, 2)
        )
    inlet_halo &= ~inlet_mask & ~obstacle
    hx, hy, hz = np.nonzero(inlet_halo)
    status = STATUS_TIME_CAPPED
    diverged_at_step = 0
    flat_reports = 0
    last_reached = -1
    best_mass_fraction = -1.0
    for taken in range(1, max_steps + 1):
        # Hold the inlet full. What that means differs by mode, and the
        # difference is the whole redesign -- see the block above inlet_area_m2
        # for the three measurements that forced it.
        state.flags[ix, iy, iz] = FLAG_FLUID
        # Restore the interface skin the pin needs and cannot get from
        # apply_transitions -- see the inlet_halo derivation above. Checked on
        # the handful of halo cells first; the full-domain mask is only built
        # on the steps where something has actually gone dry.
        if np.any(state.flags[hx, hy, hz] == FLAG_GAS):
            ensure_interface_skin(state, inlet_halo)
        if inlet_mode == 1:
            # Prescribed velocity: constant, so it cannot accumulate.
            state.f[:, ix, iy, iz] = equilibrium(inlet_rho, inlet_u).astype(np.float32)
        else:
            # Pressure boundary. Measure the inlet cells' own velocity from the
            # populations they are holding at pin time -- post free-surface
            # pass, so this is what the flow field actually delivered last step
            # -- and re-impose only the head pressure on top of it.
            #
            # The RAW first moment, sum_i c_i f_i / rho. ``state.u`` carries the
            # Guo half-force correction; feeding that back is what made the
            # original inlet a momentum pump, and it is not re-introduced here
            # by the back door. The rebuild adds no force term of its own, so
            # nothing can accumulate: whatever the flow slows to, the boundary
            # follows.
            f_slice = state.f[:, ix, iy, iz].astype(np.float64)
            rho_now = f_slice.sum(axis=0)
            u_raw = lattice_c_t @ f_slice / np.clip(rho_now, 1e-9, None)
            # A numerical filter on a boundary *measurement*, not physics --
            # see INLET_VELOCITY_BETA for the measured amplitudes it removes.
            inlet_u_measured *= 1.0 - INLET_VELOCITY_BETA
            inlet_u_measured += INLET_VELOCITY_BETA * u_raw
            u_used = inlet_u_measured.copy()
            # Clamped by norm rather than per component, so clamping cannot
            # rotate the flow direction the measurement reported.
            speed = np.sqrt(np.sum(u_used * u_used, axis=0))
            u_used *= np.minimum(1.0, u_cap_lattice / np.maximum(speed, 1e-12))
            state.f[:, ix, iy, iz] = equilibrium(inlet_rho, u_used).astype(np.float32)
        # Net delivery, measured before the pin overwrites what it is replacing.
        # A boundary that meters itself has to be *reported* honestly too: on a
        # backed-up fill this goes to zero, and a poured figure that kept
        # climbing would be the same lie the old prescribed rate told.
        poured += float(np.sum(inlet_full_mass)) - float(
            np.sum(state.mass[ix, iy, iz])
        )
        state.mass[ix, iy, iz] = inlet_full_mass
        # Metal entering at the inlet is at the pour temperature -- pinned
        # every step, same as mass and momentum above, so the source stays a
        # true Dirichlet boundary rather than drifting as wall loss and
        # diffusion nibble at it.
        state.temperature[ix, iy, iz] = pour_temp

        step(state, units, track_free_surface=True, smagorinsky_cs=smagorinsky_cs)
        advect_temperature(state, alpha_lattice)
        apply_wall_heat_loss(
            state,
            htc_w_m2k=htc_dense,
            mold_temperature_c=mold_temp,
            density_kg_m3=float(alloy.density_liquid),
            specific_heat_j_kgk=float(alloy.specific_heat),
        )

        if detailed_progress_callback is not None and (
            taken % PROGRESS_UPDATE_INTERVAL == 0
        ):
            # Against the step budget, not the fill percentage. The bar has to
            # move monotonically, and filled-cell counts do not: metal sloshes
            # back and forth across the same cells while a front is stalled.
            # A run that finishes early simply finishes early.
            detailed_progress_callback(taken, max_steps, "Filling")

        if taken % frame_every == 0:
            capture_frame(taken)
            if len(frames) >= 2 * FILL_ANIMATION_FRAMES:
                # Drop every other frame and sample half as often from here.
                # The survivors stay evenly spaced, so playback never shows a
                # dense early sequence followed by a sparse late one.
                del frames[1::2]
                frame_every *= 2

        if taken % DIVERGENCE_CHECK_INTERVAL == 0:
            # Mass conservation is the honest health check. The free surface
            # can only hold as much mass as it has wet cells, so a total above
            # that ceiling is not drift -- it is the collision operator having
            # left the range where it means anything. Checking flags alone
            # cannot see this: a blow-up promotes every cell to FLUID, which
            # reads as a perfect 100% fill.
            wet = (state.flags == FLAG_FLUID) | (state.flags == FLAG_INTERFACE)
            wet_cells = int(np.count_nonzero(wet))
            wet_mass = float(np.sum(state.mass[wet])) if wet_cells else 0.0
            ceiling = DIVERGENCE_MASS_FACTOR * max(wet_cells, 1)
            if (not np.isfinite(wet_mass)) or wet_mass > ceiling or wet_mass < 0.0:
                status = STATUS_DIVERGED
                diverged_at_step = taken
                emit(
                    f"Filling ABORTED: solver diverged at step {taken:,} "
                    f"({taken * units.dt_s:.2f} s). Wet mass {wet_mass:.4g} "
                    f"against a physical ceiling of {ceiling:.4g} over "
                    f"{wet_cells:,} wet cells. Results discarded."
                )
                break

        if taken % entrapment_check_interval == 0:
            # Per output frame, not per step (design doc 6.4): a
            # connected-component labelling over the whole coarse grid every
            # single lattice step would be the single most expensive thing
            # in this loop.
            vent_mask = static_vent | (boundary_mask & (state.flags == FLAG_GAS))
            if vent_warning is None:
                vent_warning = _entrapment_warning_if_no_vent(vent_mask)
            entrapped_now = find_entrapped_gas(state.flags, vent_mask)
            entrapped_now_fine = _upsample(entrapped_now, mapping.shape, coarsening)
            entrapped_now_count = int(
                np.count_nonzero(gather_to_voxels(mapping, entrapped_now_fine))
            )
            peak_entrapped_voxels = max(peak_entrapped_voxels, entrapped_now_count)

        if taken % report_every == 0:
            if cancel_callback is not None:
                cancel_callback()
            reached = int(np.count_nonzero((state.flags == FLAG_FLUID) & metal))
            # Two measures of the same fill, because neither alone is honest.
            # The cell count is what "filled" means, but it moves in steps of a
            # whole cell: an interface cell fills continuously from eps 0 to
            # eps 1 and only flips to FLUID at the top of that band. Mass moves
            # the whole time. Reporting both makes a slow fill legible as slow
            # rather than as stopped.
            mass_fraction = float(
                np.clip(float(np.sum(state.mass[metal])), 0.0, metal_full_mass)
            ) / max(metal_full_mass, 1e-12)
            emit(
                f"Filling: {reached:,}/{metal_total:,} cells "
                f"({reached / max(1, metal_total):.0%}), "
                f"{mass_fraction:.0%} by mass, "
                f"{poured * poured_lb_per_lattice_mass:.1f} lb poured, at "
                f"{taken * units.dt_s:.2f} s"
            )
            if reached >= metal_total:
                status = STATUS_COMPLETE
                break
            # A front that has stopped advancing will not restart: the metal
            # either cannot get through the section ahead or has frozen the
            # channel behind it. Running out the clock on that produces the
            # same answer far more slowly, and reports it as a time cap --
            # which reads as "needed longer" when it means "cannot get there".
            #
            # Measured against the high-water mark, not the previous report.
            # A stalled front does not sit still: metal sloshes back and forth
            # across the same cells, so the count ticks up and down by a few
            # percent forever. Comparing consecutive reports resets on every
            # one of those wobbles and never fires.
            #
            # BOTH measures must be flat. The cell count alone declared a stall
            # on a front that was genuinely advancing: measured on
            # Test_Model/Filling_Test, 428 casting cells sat at eps ~ 0.5,
            # taking on mass every report while flipping no flags. Requiring
            # the mass high-water mark to be flat as well is the difference
            # between "cannot get there" and "is getting there slowly".
            advanced = False
            if reached > last_reached:
                last_reached = reached
                advanced = True
            if mass_fraction > best_mass_fraction + STALL_MASS_FRACTION_EPSILON:
                best_mass_fraction = mass_fraction
                advanced = True
            if advanced:
                flat_reports = 0
            else:
                flat_reports += 1
            if flat_reports >= STALL_REPORTS and taken >= STALL_MIN_STEPS:
                status = STATUS_STALLED
                emit(
                    f"Filling stopped: the front has not advanced for "
                    f"{flat_reports} consecutive reports, holding at "
                    f"{reached:,}/{metal_total:,} cells "
                    f"({reached / max(1, metal_total):.0%}, "
                    f"{mass_fraction:.0%} by mass)."
                )
                break

    if status == STATUS_DIVERGED:
        # Every frame after the blow-up shows the blow-up, and the frames
        # before it end mid-pour with no indication that what follows is
        # rubbish. Playing that back would be the most convincing possible
        # presentation of a failed solve, so there is nothing to play.
        frames.clear()
    else:
        # The loop exits on a report boundary, which is up to report_every
        # steps after the state the run actually ended in. Without this the
        # last frame is not the final state -- most visibly on a complete
        # fill, where playback would stop just short of full.
        capture_frame(taken)

    # Final entrapment snapshot -- evaluated unconditionally rather than
    # relying on the last step happening to land on a check interval, so the
    # reported "final" set always reflects the state actually handed off.
    final_vent_mask = static_vent | (boundary_mask & (state.flags == FLAG_GAS))
    if vent_warning is None:
        vent_warning = _entrapment_warning_if_no_vent(final_vent_mask)
    entrapped_coarse = find_entrapped_gas(state.flags, final_vent_mask)

    entrapped_fine = _upsample(entrapped_coarse, mapping.shape, coarsening)
    entrapped_voxels = gather_to_voxels(mapping, entrapped_fine).astype(bool)
    entrapped_count = int(np.count_nonzero(entrapped_voxels))
    # Peak, not just final: a pocket can be entrapped mid-fill and vent later
    # as the geometry around it changes, so the final count alone
    # understates the risk (design doc section 6.4). Tracked in voxel units
    # throughout, at every periodic check during the loop above, so it is
    # never below the final count.
    peak_entrapped_count = max(entrapped_count, peak_entrapped_voxels)

    # F3: cold shuts. Evaluated once on the completed fill/thermal fields --
    # unlike entrapment there is no "peak" concept here: a seam does not heal
    # once the fronts have met and cooled, so the end-of-fill field is the
    # answer.
    liquidus_c = float(getattr(alloy, "liquidus_temp", -np.inf))
    cold_shut_coarse = find_cold_shuts(
        state.fill_time_s,
        state.temperature,
        state.flags,
        liquidus_c=liquidus_c,
        arrival_gap_s=float(getattr(filling, "cold_shut_arrival_gap_s", 0.5)),
        temperature_margin_c=float(getattr(filling, "cold_shut_temp_margin_c", 25.0)),
    )
    cold_shut_fine = _upsample(cold_shut_coarse, mapping.shape, coarsening)
    cold_shut_voxels = gather_to_voxels(mapping, cold_shut_fine).astype(bool)
    cold_shut_count = int(np.count_nonzero(cold_shut_voxels))

    filled_coarse = state.flags == FLAG_FLUID
    fill_time_coarse = np.asarray(state.fill_time_s, dtype=np.float64)

    fill_time_fine = _upsample(fill_time_coarse, mapping.shape, coarsening)
    filled_fine = _upsample(filled_coarse, mapping.shape, coarsening)

    fill_time_voxels = gather_to_voxels(mapping, fill_time_fine)
    filled_voxels = gather_to_voxels(mapping, filled_fine).astype(bool)

    voxel_metal = gather_to_voxels(mapping, _upsample(metal, mapping.shape, coarsening)).astype(bool)
    filled_count = int(np.count_nonzero(filled_voxels & voxel_metal))
    misrun_count = int(np.count_nonzero(voxel_metal & ~filled_voxels))

    # Mold, core and chill voxels can never fill, so they must not be able to
    # read as misruns. Every consumer of this field -- the desktop renderer,
    # the slider label, the web payload -- defines "misrun" as a negative
    # arrival time, and gather_to_voxels hands back the -1 sentinel for all of
    # them alike. Left as -1 they show up in the Fill Time view as a solid grey
    # shell of "metal that never arrived": on Test_Model/Filling_Test that is
    # 1,485 chill voxels reported as misruns against a true misrun count of 0.
    #
    # NaN is the sentinel for "this voxel is not metal, the question does not
    # apply". It is not a second flavour of -1: NaN is false against `< 0`,
    # `>= 0` and `<= t` alike, so it drops out of the misrun mask, the arrived
    # mask and the time range without any consumer special-casing it, whereas
    # any negative sentinel has to be excluded by hand everywhere.
    fill_time_voxels = np.where(voxel_metal, fill_time_voxels, np.nan)

    # F2: gather the temperature the fill actually left behind, rather than
    # the two-constant (pour temp / mold temp) placeholder F1 used. Metal
    # that entered first and travelled furthest has been advecting and
    # losing heat to the mold the whole run, so it comes back cooler than
    # metal still at the gate.
    temperature_dense_fine = _upsample(
        np.asarray(state.temperature, dtype=np.float64), mapping.shape, coarsening
    )
    temperature_fill = gather_to_voxels(mapping, temperature_dense_fine)

    # A cell can hold an arrival time and no temperature. fill_time_s is written
    # once, when metal first reaches a cell, and is never cleared; the
    # temperature field is NaN-reset whenever a cell drains back to gas. So any
    # cell that filled and later drained -- routine wherever the front sloshes
    # or a channel empties behind it -- carries a valid fill time alongside a
    # NaN temperature.
    #
    # set_initial_temperature_field rejects non-finite values outright, so
    # trusting fill_time alone here crashes the whole run at handover. It never
    # showed up on a synthetic column, where nothing drains, only on real
    # gating geometry. Require BOTH signals before treating a voxel as carrying
    # metal temperature.
    filled_mask = filled_voxels & voxel_metal & np.isfinite(temperature_fill)

    # Every voxel must carry a finite temperature -- the seam validates that
    # -- so unfilled metal cannot simply be left blank. Seeding it at mold
    # temperature says "no metal arrived here", which is what a misrun is. The
    # alternative, seeding it at pouring temperature, would hide the defect
    # behind a plausible-looking result.
    temperature = np.full(mapping.n_voxels, mold_temp, dtype=np.float64)
    temperature[filled_mask] = temperature_fill[filled_mask]

    if not np.all(np.isfinite(temperature)):
        raise AssertionError(
            "filling produced a non-finite handover temperature; "
            "set_initial_temperature_field would reject it"
        )

    if np.any(filled_mask):
        min_fill_temperature_c = float(temperature[filled_mask].min())
        mean_fill_temperature_c = float(temperature[filled_mask].mean())
    else:
        min_fill_temperature_c = mold_temp
        mean_fill_temperature_c = mold_temp

    if status == STATUS_DIVERGED:
        # Said first and said plainly. Everything below this describes fields
        # the solver produced after it stopped conserving mass, so a misrun
        # count or a cold-shut count from here is an artefact of the blow-up,
        # not a casting defect.
        warnings.append(
            f"SOLVER DIVERGED at step {diverged_at_step:,} -- this fill result is "
            f"not physical and must not be used. Any misrun, entrapment or "
            f"cold-shut figure below is an artefact of the divergence."
        )
    elif status == STATUS_STALLED:
        warnings.append(
            f"the fill stalled at {filled_count / max(1, filled_count + misrun_count):.0%} "
            f"-- the front stopped advancing with metal still unfilled. This is "
            f"a genuine misrun or an under-resolved section, not a time limit."
        )
    elif status == STATUS_TIME_CAPPED:
        warnings.append(
            f"the fill was still advancing when max_fill_time_s "
            f"({max_time:.0f} s) ran out; it did not finish. Raise the cap "
            f"before reading the misrun count as a defect."
        )

    if misrun_count:
        warnings.append(
            f"{misrun_count:,} metal voxels never filled (misrun): "
            f"{misrun_count / max(1, filled_count + misrun_count):.1%} of the metal"
        )
    liquidus_temp = float(getattr(alloy, "liquidus_temp", -np.inf))
    if np.any(filled_mask) and min_fill_temperature_c < liquidus_temp:
        warnings.append(
            f"filled metal cooled below liquidus ({liquidus_temp:.0f} C) before "
            f"handover: minimum fill temperature {min_fill_temperature_c:.0f} C -- "
            f"incipient cold-shut risk"
        )
    solver_mode = str(getattr(analysis_options, "solver_mode", "") or "")
    if solver_mode and solver_mode != "thermal_coupled":
        warnings.append(
            f"solver_mode is {solver_mode!r}; the seeded temperature field is "
            f"only consumed by 'thermal_coupled', so the fill's thermal result "
            f"will be ignored by the solidification solve."
        )
    # F3: air entrapment and cold shuts.
    if vent_warning is not None:
        warnings.append(vent_warning)
    if entrapped_count:
        warnings.append(
            f"{entrapped_count:,} voxels of trapped air detected (no vent path "
            f"reached, peak {peak_entrapped_count:,}): risk of gas porosity"
        )
    if cold_shut_count:
        warnings.append(
            f"{cold_shut_count:,} voxels show a cold-shut signature: fill "
            f"fronts met after the metal had cooled to within "
            f"{float(getattr(filling, 'cold_shut_temp_margin_c', 25.0)):.0f} C "
            f"of the liquidus ({liquidus_c:.0f} C)"
        )
    for message in warnings:
        emit(f"Filling warning: {message}")

    return FillingResult(
        fill_time_s=fill_time_voxels.astype(np.float32),
        temperature_c=temperature,
        inlet_voxel_count=int(inlet.shape[0]),
        filled_voxel_count=filled_count,
        misrun_voxel_count=misrun_count,
        choke_area_cm2=measure_choke_area_cm2(solve_types, dx_m),
        filter_zones=filter_zones,
        steps=taken,
        fill_seconds=taken * units.dt_s,
        coarsening=coarsening,
        min_fill_temperature_c=min_fill_temperature_c,
        mean_fill_temperature_c=mean_fill_temperature_c,
        entrapped_voxel_count=entrapped_count,
        peak_entrapped_voxel_count=peak_entrapped_count,
        cold_shut_voxel_count=cold_shut_count,
        entrapped_gas=entrapped_voxels,
        cold_shut=cold_shut_voxels,
        warnings=tuple(warnings),
        status=status,
        diverged_at_step=diverged_at_step,
        choke_thickness_cells=choke_thickness_cells,
        choke_thickness_by_part={
            _METAL_PATH_LABELS[code]: value
            for code, value in choke_by_part.items()
        },
        narrowest_choke_part=(
            _METAL_PATH_LABELS[narrowest_code]
            if narrowest_code is not None
            else ""
        ),
        choke_area_fine_cm2=measure_choke_area_cm2(fine_types, mapping.dx_m),
        frames=tuple(frames),
    )


def maybe_run_filling(
    voxel_mesh,
    alloy,
    analysis_options,
    inputs,
    *,
    cancel_callback=None,
    progress_message_callback=None,
    detailed_progress_callback=None,
) -> Optional[FillingResult]:
    """
    Run the filling stage if, and only if, it is both enabled and possible.

    The single gate both pipelines call. ``core/surrogate/pipeline.py``
    re-implements ``run_pipeline``'s sequence rather than calling it, so a
    stage wired into only one of them silently no-ops in the other -- which is
    why this is a shared helper and not inline code.

    Requirement: filling runs only when the user has picked a pour location.
    A missing location with the toggle on is reported rather than ignored,
    because silently skipping the stage looks identical to running it and
    finding nothing wrong.
    """

    def emit(message: str) -> None:
        if progress_message_callback is not None:
            progress_message_callback(message)

    if not getattr(analysis_options, "filling", False):
        return None

    raw = getattr(inputs, "pour_location", None)
    if not raw:
        emit(
            "Filling is enabled but no pour location has been selected; "
            "skipping the fill. Click a point on the assembly to set one."
        )
        return None

    from core.pour_location import PourLocation

    try:
        location = raw if isinstance(raw, PourLocation) else PourLocation.from_dict(raw)
    except (ValueError, KeyError, TypeError) as exc:
        emit(f"Filling skipped: pour location is invalid ({exc}).")
        return None

    return run_filling_stage(
        voxel_mesh,
        alloy,
        analysis_options,
        location,
        cancel_callback=cancel_callback,
        detailed_progress_callback=detailed_progress_callback,
        progress_message_callback=progress_message_callback,
    )
