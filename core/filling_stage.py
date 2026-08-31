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
)
from physics.filling import PorousZone
from physics.fluid_flow import (
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

    Returns:
        Choke area in cm^2, or 0.0 when no gating metal is present.
    """
    dense_types = np.asarray(dense_types)
    flow = np.isin(dense_types, (PART_TYPE_SPRUE, PART_TYPE_RUNNER, PART_TYPE_GATE))
    if not np.any(flow):
        return 0.0

    cell_area_cm2 = (dx_m * 100.0) ** 2
    best = np.inf
    for axis in (0, 1, 2):
        counts = flow.sum(axis=tuple(a for a in (0, 1, 2) if a != axis))
        occupied = counts[counts > 0]
        if occupied.size:
            best = min(best, float(occupied.min()))
    if not np.isfinite(best):
        return 0.0
    return best * cell_area_cm2


# ---------------------------------------------------------------------------
# Stage orchestration
# ---------------------------------------------------------------------------


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
        choke_area_cm2: Narrowest gating cross-section, measured.
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
        warnings: Human-readable conditions worth surfacing.
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

    @property
    def filled_fraction(self) -> float:
        total = self.filled_voxel_count + self.misrun_voxel_count
        return self.filled_voxel_count / total if total else 0.0


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
        FLAG_OBSTACLE,
        REFERENCE_DENSITY,
        advect_temperature,
        apply_wall_heat_loss,
        create_state,
        find_cold_shuts,
        find_entrapped_gas,
        step,
    )
    from physics.fluid_flow import equilibrium, lattice_units
    from core.part_manager import is_gating_metal

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

    inlet = _resolve_inlet_cells(pour_location, mapping, solve_types, coarsening)
    if inlet.shape[0] == 0:
        emit(
            "Filling stage skipped: the pour location does not resolve to any "
            "metal cell. Move it onto the pour cup, sprue or casting."
        )
        return None

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

    # Inlet density from the metallostatic head above the pour point:
    # rho = rho_0 (1 + g h / c_s^2) in lattice units. Imposing a *pressure*
    # rather than a velocity is what lets the sprue choke meter the flow, which
    # is the behaviour Foseco's own gating guidance asks for.
    head_lattice = float(inches_to_meters(
        float(getattr(filling, "initial_head_in", 4.0))
    )) / dx_m
    inlet_density = REFERENCE_DENSITY * (
        1.0 + units.gravity_lattice * head_lattice / (1.0 / 3.0)
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
    taken = 0
    for taken in range(1, max_steps + 1):
        # Hold the inlet full. A Dirichlet source is the stable choice for F1:
        # it lets gravity and the geometry meter the flow rather than the
        # boundary condition dictating it, which is what Foseco's own guidance
        # asks for (the sprue choke should control, not the inlet).
        ix, iy, iz = inlet[:, 0], inlet[:, 1], inlet[:, 2]
        state.flags[ix, iy, iz] = FLAG_FLUID
        state.mass[ix, iy, iz] = REFERENCE_DENSITY
        # Re-impose the head pressure while KEEPING the cell's current
        # velocity. Resetting to a zero-velocity equilibrium instead pins the
        # source at rest every step, destroying the very momentum that drives
        # the fill -- the metal then dribbles out of the inlet and the front
        # never establishes.
        state.f[:, ix, iy, iz] = equilibrium(
            np.full(inlet.shape[0], inlet_density, dtype=np.float64),
            state.u[:, ix, iy, iz].astype(np.float64),
        ).astype(np.float32)
        # Metal entering at the inlet is at the pour temperature -- pinned
        # every step, same as mass and momentum above, so the source stays a
        # true Dirichlet boundary rather than drifting as wall loss and
        # diffusion nibble at it.
        state.temperature[ix, iy, iz] = pour_temp

        step(state, units, track_free_surface=True)
        advect_temperature(state, alpha_lattice)
        apply_wall_heat_loss(
            state,
            htc_w_m2k=htc_dense,
            mold_temperature_c=mold_temp,
            density_kg_m3=float(alloy.density_liquid),
            specific_heat_j_kgk=float(alloy.specific_heat),
        )

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
            emit(
                f"Filling: {reached:,}/{int(metal.sum()):,} cells "
                f"({reached / max(1, int(metal.sum())):.0%}) at "
                f"{taken * units.dt_s:.2f} s"
            )
            if reached >= int(metal.sum()):
                break

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
        choke_area_cm2=measure_choke_area_cm2(fine_types, mapping.dx_m),
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
    )


def maybe_run_filling(
    voxel_mesh,
    alloy,
    analysis_options,
    inputs,
    *,
    cancel_callback=None,
    progress_message_callback=None,
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
        progress_message_callback=progress_message_callback,
    )
