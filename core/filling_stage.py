from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, cast

import numpy as np

from core.part_manager import (
    PART_TYPE_CAST,
    PART_TYPE_CHILL,
    PART_TYPE_CORE,
    PART_TYPE_FILTER,
    PART_TYPE_GATE,
    PART_TYPE_MOLD,
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
    CS2 as LATTICE_CS2,
    SMAGORINSKY_CS,
    ceramic_foam_permeability,
    forchheimer_inverse_length,
    porous_drag_coefficients,
    capillary_entry_density_offset,
    capillary_entry_pressure,
)
from physics.units import inches_to_meters

COARSENING_NOTE = (
    "AdaptiveVoxelMesh.generate() segfaults at certain voxel sizes, so the "
    "fill grid is produced by downsampling the fine dense grid rather than by "
    "regenerating the mesh at a larger base_voxel_size."
)

EMPTY_CELL = np.int8(-1)

COARSENING_PRIORITY: Tuple[int, ...] = (
    PART_TYPE_FILTER,
    PART_TYPE_CHILL,
    PART_TYPE_CORE,
    PART_TYPE_MOLD,
    PART_TYPE_GATE,
    PART_TYPE_RUNNER,
    PART_TYPE_SPRUE,
    PART_TYPE_POUR_CUP,
    PART_TYPE_RISER_POST,
    PART_TYPE_RISER,
    PART_TYPE_CAST,
)

FILL_ANIMATION_FRAMES = 60

STATUS_COMPLETE = "complete"
STATUS_TIME_CAPPED = "time_capped"
STATUS_STALLED = "stalled"
STATUS_DIVERGED = "diverged"

MIN_CHOKE_THICKNESS_CELLS = 2.0

MARGINAL_CHOKE_THICKNESS_CELLS = 3.0

_METAL_PATH_CODES: Tuple[int, ...] = (
    PART_TYPE_SPRUE,
    PART_TYPE_RUNNER,
    PART_TYPE_FILTER,
    PART_TYPE_GATE,
)

_METAL_PATH_LABELS: Dict[int, str] = {
    PART_TYPE_SPRUE: "sprue",
    PART_TYPE_RUNNER: "runner",
    PART_TYPE_FILTER: "filter",
    PART_TYPE_GATE: "gate",
}

MAX_SMAGORINSKY_CS = 0.5

DIVERGENCE_MASS_FACTOR = 1.5

PROGRESS_UPDATE_INTERVAL = 200

DIVERGENCE_CHECK_INTERVAL = 200

STALL_REPORTS = 5

INLET_VELOCITY_BETA = 0.2

STALL_MASS_FRACTION_EPSILON = 0.001

INLET_DISCHARGE_COEFFICIENT = 0.8

INLET_DELIVERY_DEPTH_CELLS = 3

INLET_RATE_GAIN = 0.02

NEGATIVE_POPULATION_WINDOW = 40
NEGATIVE_POPULATION_PREVALENCE = 0.5

NEGATIVE_POPULATION_CASCADE_FRACTION = 0.02

NEGATIVE_POPULATION_CASCADE_MIN_CELLS = 8

MIN_CHOKE_BODY_CELLS = 8

_SIX_CONNECTED: np.ndarray = np.zeros((3, 3, 3), dtype=bool)
_SIX_CONNECTED[1, 1, :] = True
_SIX_CONNECTED[1, :, 1] = True
_SIX_CONNECTED[:, 1, 1] = True

NEGATIVE_POPULATION_CASCADE_CHECKS = 2

FLUID_FILL_FRACTION_MIN = -0.1
FLUID_FILL_FRACTION_MAX = 1.1

FLUID_FILL_FRACTION_CHECKS = 2

MOLD_PAD_CELLS = 1

HEADSPACE_CELLS = 2

STALL_MIN_STEPS = 2000

@dataclass(frozen=True)
class GridMapping:

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
        return self.n_voxels / float(np.prod(self.shape))

    @property
    def dx_m(self) -> float:
        return float(inches_to_meters(self.dx_in))

def build_grid_mapping(
    voxel_centers: np.ndarray, voxel_size_in: float
) -> GridMapping:
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
    dense = np.asarray(dense, dtype=np.int8)
    if factor < 1 or int(factor) != factor:
        raise ValueError(f"factor must be a positive integer, got {factor}")
    factor = int(factor)
    if factor == 1:
        return cast("np.ndarray", dense.copy())

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
    for code in reversed(list(priority)):
        present = np.any(blocks == code, axis=-1)
        out[present] = np.int8(code)
    return out

@dataclass(frozen=True)
class FilterZoneSpec:

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
    resolved = max(cells_along_flow * dx_m, dx_m)
    return float(np.clip(thickness_m / resolved, minimum, maximum))

def cells_along_flow(cell_count: int, front_face_area_m2: float, dx_m: float) -> float:
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
    occupied = np.argwhere(component)
    extents = occupied.max(axis=0) - occupied.min(axis=0) + 1
    thickness_cells = int(np.min(extents))
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

def _open_air_mask(shape: Tuple[int, int, int]) -> np.ndarray:
    open_air = np.zeros(shape, dtype=bool)
    open_air[:, :, -1] = True
    return open_air

def _entrapment_warning_if_no_vent(vent_mask: np.ndarray) -> Optional[str]:
    if np.any(vent_mask):
        return None
    return (
        "no vent path found for air-entrapment checking (vent_risers is "
        "off, no riser is present, and no gas cell currently touches the "
        "inlet or the open top of the mold): every gas pocket will read as "
        "entrapped until a vent opens"
    )

def measure_choke_area_cm2(dense_types: np.ndarray, dx_m: float) -> float:
    cells = _min_flow_slice_cells(dense_types)
    if cells <= 0:
        return 0.0
    return float(cells) * (dx_m * 100.0) ** 2

def pour_weight_lb(voxel_mesh, alloy) -> float:
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

    kg = float(alloy.density_liquid) * volume_in3 * 1.6387064e-5
    return kg * 2.20462262

def filter_flow_limit_kg_s(
    filter_zones: Sequence["FilterZoneSpec"],
    alloy,
    pour_mode_code: int = 0,
) -> Tuple[Optional[float], Optional[str]]:
    if not filter_zones:
        return None, None

    steel_class = steel_class_for_alloy(getattr(alloy, "name", "") or "")
    if steel_class is None:
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
    if weight_lb <= 0.0:
        return 0.0
    return float(np.sqrt(weight_lb))

def _min_flow_slice_cells(dense_types: np.ndarray) -> int:
    dense_types = np.asarray(dense_types)
    flow = np.isin(dense_types, (PART_TYPE_SPRUE, PART_TYPE_RUNNER, PART_TYPE_GATE))
    if not np.any(flow):
        return 0

    best = np.inf
    for axis in (0, 1, 2):
        counts = flow.sum(axis=tuple(a for a in (0, 1, 2) if a != axis))
        occupied = counts[counts > 0]
        if occupied.size:
            best = min(best, float(np.min(occupied)))
    if not np.isfinite(best):
        return 0
    return int(best)

def _local_thickness_cells(region: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    region = np.asarray(region, dtype=bool)
    out = np.zeros(region.shape, dtype=np.float64)
    if not np.any(region):
        return out

    occupied = np.nonzero(region)
    box = tuple(
        slice(int(axis.min()), int(axis.max()) + 1) for axis in occupied
    )
    sub = np.pad(region[box], 1, mode="constant", constant_values=False)

    distance = ndimage.distance_transform_edt(sub)
    largest = np.zeros_like(distance)
    radius = np.floor(float(distance.max()) * 2.0) / 2.0
    while radius >= 0.5:
        seeds = distance >= radius
        if np.any(seeds):
            reach = ndimage.distance_transform_edt(~seeds)
            np.maximum(largest, np.where(reach <= radius, radius, 0.0), out=largest)
        radius -= 0.5

    inner = largest[(slice(1, -1),) * largest.ndim]
    out[box] = np.where(region[box], 2.0 * inner - 1.0, 0.0)
    return out

def measure_choke_thickness_by_part(dense_types: np.ndarray) -> Dict[int, float]:
    by_part: Dict[int, float] = {}
    for code, bodies in _choke_bodies(dense_types).items():
        resolvable = [t for t, cells in bodies if cells >= MIN_CHOKE_BODY_CELLS]
        by_part[code] = min(resolvable) if resolvable else min(
            t for t, _ in bodies
        )
    return by_part

def measure_choke_body_fragments(
    dense_types: np.ndarray,
) -> Dict[int, Tuple[int, ...]]:
    fragments: Dict[int, Tuple[int, ...]] = {}
    for code, bodies in _choke_bodies(dense_types).items():
        small = sorted(
            cells for _, cells in bodies if cells < MIN_CHOKE_BODY_CELLS
        )
        if small and len(small) < len(bodies):
            fragments[code] = tuple(small)
    return fragments

def _choke_bodies(
    dense_types: np.ndarray,
) -> Dict[int, Tuple[Tuple[float, int], ...]]:
    from scipy import ndimage

    dense_types = np.asarray(dense_types)
    metal_path = np.isin(dense_types, _METAL_PATH_CODES)
    if not np.any(metal_path):
        return {}
    thickness = _local_thickness_cells(metal_path)

    by_part: Dict[int, Tuple[Tuple[float, int], ...]] = {}
    for code in _METAL_PATH_CODES:
        part = dense_types == code
        if not np.any(part):
            continue
        labels, count = ndimage.label(part, structure=_SIX_CONNECTED)
        bodies = []
        for index in range(1, count + 1):
            body = labels == index
            cells = int(np.count_nonzero(body))
            if not cells:
                continue
            bodies.append(
                (max(0.0, float(np.median(thickness[body]))), cells)
            )
        if bodies:
            by_part[code] = tuple(sorted(bodies))
    return by_part

def measure_choke_thickness_components(
    dense_types: np.ndarray,
) -> Dict[int, Tuple[float, ...]]:
    return {
        code: tuple(t for t, _ in bodies)
        for code, bodies in _choke_bodies(dense_types).items()
    }

@dataclass(frozen=True)
class MetalPathSevering:

    severed: bool
    fine_components: int
    solve_components: int
    erased: Dict[int, int]
    split: Dict[int, int]

    @property
    def parts(self) -> Tuple[int, ...]:
        codes = set(self.erased) | set(self.split)
        return tuple(code for code in _METAL_PATH_CODES if code in codes)

    def describe(self) -> str:
        pieces = []
        for code in _METAL_PATH_CODES:
            if code in self.erased:
                count = self.erased[code]
                pieces.append(
                    f"{count} {_METAL_PATH_LABELS.get(code, 'gating')} "
                    f"{'body' if count == 1 else 'bodies'} erased outright"
                )
        broken = tuple(code for code in _METAL_PATH_CODES if code in self.split)
        if broken:
            names = "/".join(
                _METAL_PATH_LABELS.get(code, "gating") for code in broken
            )
            pieces.append(f"the path through the {names} broken in two")
        if not pieces:
            return "the metal path is intact"
        return " and ".join(pieces)

def metal_path_severing(
    fine_types: np.ndarray, solve_types: np.ndarray, factor: int
) -> MetalPathSevering:
    from scipy import ndimage

    fine_types = np.asarray(fine_types)
    solve_types = np.asarray(solve_types)
    factor = max(1, int(factor))

    fine_mask = np.isin(fine_types, _METAL_PATH_CODES)
    solve_mask = np.isin(solve_types, _METAL_PATH_CODES)
    fine_labels, fine_count = ndimage.label(fine_mask, structure=_SIX_CONNECTED)
    _solve_labels, solve_count = ndimage.label(
        solve_mask, structure=_SIX_CONNECTED
    )
    intact = MetalPathSevering(
        severed=False,
        fine_components=int(fine_count),
        solve_components=int(solve_count),
        erased={},
        split={},
    )
    if factor == 1 or not fine_count:
        return intact

    shape = solve_types.shape

    def blocks_of(mask_labels: np.ndarray, index: int):
        ix, iy, iz = np.nonzero(mask_labels == index)
        if ix.size == 0:
            return None
        return (
            np.minimum(ix // factor, shape[0] - 1),
            np.minimum(iy // factor, shape[1] - 1),
            np.minimum(iz // factor, shape[2] - 1),
            (ix, iy, iz),
        )

    split: Dict[int, int] = {}
    for index in range(1, fine_count + 1):
        mapped = blocks_of(fine_labels, index)
        if mapped is None:
            continue
        cx, cy, cz, (ix, iy, iz) = mapped
        landed = _solve_labels[cx, cy, cz]
        if np.unique(landed[landed > 0]).size < 2:
            continue
        lost = landed == 0
        codes = np.unique(
            fine_types[ix[lost], iy[lost], iz[lost]]
            if np.any(lost)
            else fine_types[ix, iy, iz]
        )
        for code in codes:
            if int(code) in _METAL_PATH_CODES:
                split[int(code)] = split.get(int(code), 0) + 1

    erased: Dict[int, int] = {}
    for code in _METAL_PATH_CODES:
        part = fine_types == code
        if not np.any(part):
            continue
        part_labels, part_count = ndimage.label(part, structure=_SIX_CONNECTED)
        gone = 0
        for index in range(1, part_count + 1):
            mapped = blocks_of(part_labels, index)
            if mapped is None:
                continue
            cx, cy, cz, _ = mapped
            if not np.any(np.isin(solve_types[cx, cy, cz], _METAL_PATH_CODES)):
                gone += 1
        if gone:
            erased[code] = gone

    if not (split or erased):
        return intact
    return MetalPathSevering(
        severed=True,
        fine_components=int(fine_count),
        solve_components=int(solve_count),
        erased=erased,
        split=split,
    )

def narrowest_choke_part_code(by_part: Dict[int, float]) -> Optional[int]:
    if not by_part:
        return None
    return min(
        (code for code in _METAL_PATH_CODES if code in by_part),
        key=lambda code: by_part[code],
    )

@dataclass(frozen=True)
class CoarseningDecision:

    coarsening: int
    configured: int
    solve_types: np.ndarray
    by_part: Dict[int, float]
    components_by_part: Dict[int, Tuple[float, ...]]
    fragments_by_part: Dict[int, Tuple[int, ...]]
    severing: MetalPathSevering
    forced_by: Optional[int]
    forced_by_cause: str
    forced_by_thickness: float
    forced_by_severing: Optional[MetalPathSevering]
    cells_configured: int
    cells_chosen: int
    resolved: bool
    enabled: bool

    @property
    def refined(self) -> bool:
        return self.coarsening != self.configured

def choose_fill_coarsening(
    fine_types: np.ndarray,
    configured: int,
    *,
    enabled: bool = True,
) -> CoarseningDecision:
    configured = max(1, int(configured))

    class _Measured(NamedTuple):
        types: np.ndarray
        by_part: Dict[int, float]
        components: Dict[int, Tuple[float, ...]]
        fragments: Dict[int, Tuple[int, ...]]
        severing: MetalPathSevering

    class _Blocked(NamedTuple):
        code: Optional[int]
        cause: str

    def measure(factor: int) -> _Measured:
        types = coarsen_part_types(fine_types, factor)
        bodies = _choke_bodies(types)
        by_part: Dict[int, float] = {}
        fragments: Dict[int, Tuple[int, ...]] = {}
        for code, group in bodies.items():
            resolvable = [
                t for t, cells in group if cells >= MIN_CHOKE_BODY_CELLS
            ]
            by_part[code] = (
                min(resolvable) if resolvable else min(t for t, _ in group)
            )
            small = sorted(
                cells for _, cells in group if cells < MIN_CHOKE_BODY_CELLS
            )
            if small and len(small) < len(group):
                fragments[code] = tuple(small)
        return _Measured(
            types=types,
            by_part=by_part,
            components={
                code: tuple(t for t, _ in group)
                for code, group in bodies.items()
            },
            fragments=fragments,
            severing=metal_path_severing(fine_types, types, factor),
        )

    def blocked(m: _Measured) -> _Blocked:
        if m.severing.severed:
            parts = m.severing.parts
            return _Blocked(parts[0] if parts else None, "severed")
        under = {
            code: value
            for code, value in m.by_part.items()
            if 0.0 < value <= MIN_CHOKE_THICKNESS_CELLS
        }
        code = narrowest_choke_part_code(under)
        return _Blocked(code, "thickness" if code is not None else "")

    def decide(m: _Measured, factor: int, **kw) -> CoarseningDecision:
        return CoarseningDecision(
            coarsening=factor,
            configured=configured,
            solve_types=m.types,
            by_part=m.by_part,
            components_by_part=m.components,
            fragments_by_part=m.fragments,
            severing=m.severing,
            cells_configured=cells_configured,
            cells_chosen=int(np.prod(m.types.shape)),
            enabled=enabled,
            **kw,
        )

    base = measure(configured)
    cells_configured = int(np.prod(base.types.shape))
    first = blocked(base)
    forced_kw = dict(
        forced_by=first.code,
        forced_by_cause=first.cause,
        forced_by_thickness=(
            base.by_part.get(first.code, 0.0)
            if first.cause == "thickness" and first.code is not None
            else 0.0
        ),
        forced_by_severing=base.severing if first.cause == "severed" else None,
    )

    if not enabled or first.code is None or configured == 1:
        return decide(base, configured, resolved=first.code is None, **forced_kw)

    for factor in range(configured - 1, 0, -1):
        m = measure(factor)
        if blocked(m).code is None:
            return decide(m, factor, resolved=True, **forced_kw)
        if factor == 1:
            return decide(m, 1, resolved=False, **forced_kw)

    raise AssertionError("unreachable: the loop above always returns")

def measure_choke_thickness_cells(dense_types: np.ndarray) -> float:
    by_part = measure_choke_thickness_by_part(dense_types)
    if not by_part:
        return 0.0
    return max(0.0, float(min(by_part.values())))

@dataclass(frozen=True)
class FillFrame:

    time_s: float
    fill_fraction: np.ndarray
    speed_m_s: np.ndarray

@dataclass(frozen=True)
class FillingResult:

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
    choke_thickness_bodies_by_part: Dict[str, List[float]] = field(
        default_factory=dict
    )
    choke_body_fragments_by_part: Dict[str, List[int]] = field(
        default_factory=dict
    )
    narrowest_choke_part: str = ""
    choke_area_fine_cm2: float = 0.0
    delivered_lb: float = 0.0
    target_delivery_lb_s: float = 0.0
    configured_coarsening: int = 0
    frames: Tuple[FillFrame, ...] = ()

    @property
    def filled_fraction(self) -> float:
        total = self.filled_voxel_count + self.misrun_voxel_count
        return self.filled_voxel_count / total if total else 0.0

    @property
    def trustworthy(self) -> bool:
        return self.status != STATUS_DIVERGED

def _obstacle_mask(dense_types: np.ndarray) -> np.ndarray:
    from core.part_manager import is_flow_restrictor, is_mold

    types = np.asarray(dense_types)
    empty = types == EMPTY_CELL
    safe = np.where(empty, PART_TYPE_CAST, types)
    solid: np.ndarray = empty | (is_mold(safe) & ~is_flow_restrictor(safe))
    return solid

def _metal_mask(dense_types: np.ndarray) -> np.ndarray:
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
    from scipy import ndimage

    seeds = np.asarray(inlet, dtype=np.int64).reshape(-1, 3)
    if seeds.shape[0] == 0:
        return seeds

    types = np.asarray(dense_types)
    z_top = int(seeds[:, 2].max())
    top = seeds[seeds[:, 2] == z_top]
    layer = types[:, :, z_top]

    codes = np.unique(layer[top[:, 0], top[:, 1]])
    member = np.isin(layer, codes) & _metal_mask(layer)
    if not member.any():
        return seeds

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
    return mapping.origin_in[None, :] + mapping.voxel_to_grid.astype(np.float64) * mapping.dx_in

def _upsample(coarse: np.ndarray, fine_shape: Tuple[int, int, int], factor: int) -> np.ndarray:
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

    coarsening_decision = choose_fill_coarsening(
        fine_types,
        max(1, int(getattr(filling, "fill_grid_coarsening", 2))),
        enabled=int(getattr(filling, "choke_auto_resolution", 1)) != 0,
    )
    coarsening = coarsening_decision.coarsening
    solve_types = coarsening_decision.solve_types
    if coarsening_decision.refined:
        forced = coarsening_decision.forced_by
        cost = (
            coarsening_decision.cells_chosen
            / max(1, coarsening_decision.cells_configured)
        )
        label = _METAL_PATH_LABELS.get(forced, "gating") if forced is not None else "gating"
        severing = coarsening_decision.forced_by_severing
        if severing is not None:
            cause = (
                f"coarsening x{coarsening_decision.configured} cut the metal "
                f"path -- {severing.describe()}, "
                f"{severing.fine_components} connected "
                f"{'run' if severing.fine_components == 1 else 'runs'} of gating "
                f"metal on the mesh against "
                f"{severing.solve_components} on that grid. The downsample "
                f"gives each coarse block its highest-priority code and mold "
                f"outranks the flow path by design, so a channel one mesh cell "
                f"thick against a core is replaced by the core"
            )
        else:
            cause = (
                f"the {label} measured "
                f"{coarsening_decision.forced_by_thickness:.1f} cells across at "
                f"coarsening x{coarsening_decision.configured}, at or below the "
                f"{MIN_CHOKE_THICKNESS_CELLS:.0f} cells a channel needs before "
                f"it has an interior"
            )
        emit(
            f"Filling: refining the fill grid -- {cause}. "
            f"Coarsening x{coarsening_decision.configured} -> "
            f"x{coarsening}, {coarsening_decision.cells_configured:,} -> "
            f"{coarsening_decision.cells_chosen:,} cells "
            f"({cost:.0f}x the cells, and a shorter timestep with them, so "
            f"expect a substantially longer solve)."
            + (
                ""
                if coarsening_decision.resolved
                else " Even at coarsening x1 the channel is still sealed -- "
                "the mesh itself is too coarse. Running anyway, under the "
                "warning below."
            )
        )

    if coarsening_decision.severing.severed:
        severing = coarsening_decision.severing
        emit(
            f"Filling stage skipped: coarsening x{coarsening} cuts the metal "
            f"path -- {severing.describe()}. The gating metal is "
            f"{severing.fine_components} connected "
            f"{'run' if severing.fine_components == 1 else 'runs'} on the mesh "
            f"and {severing.solve_components} on the fill grid. The downsample "
            f"gives each coarse block its highest-priority part code, and mold "
            f"outranks the flow path by design (a chill that vanishes changes "
            f"the thermal answer), so a channel one mesh cell thick against a "
            f"core is replaced by the core. Nothing downstream can see it: the "
            f"thickness guards are keyed on parts that are present, so the fill "
            f"would run with the path severed and report a misrun. Set "
            f"choke_auto_resolution to 1 and the stage will refine the grid "
            f"itself, or reduce fill_grid_coarsening."
        )
        return None

    pad_lo = int(MOLD_PAD_CELLS)
    core_shape: Tuple[int, int, int] = (
        int(solve_types.shape[0]),
        int(solve_types.shape[1]),
        int(solve_types.shape[2]),
    )
    core = tuple(slice(pad_lo, pad_lo + n) for n in core_shape)
    solve_types = np.pad(
        solve_types,
        (
            (pad_lo, pad_lo),
            (pad_lo, pad_lo),
            (pad_lo, pad_lo + int(HEADSPACE_CELLS)),
        ),
        mode="constant",
        constant_values=EMPTY_CELL,
    )
    solve_core = solve_types[core]

    def to_voxels(field: np.ndarray) -> np.ndarray:
        return gather_to_voxels(
            mapping, _upsample(np.asarray(field)[core], mapping.shape, coarsening)
        )

    inlet_seeds = _resolve_inlet_cells(pour_location, mapping, solve_core, coarsening)
    if inlet_seeds.shape[0] == 0:
        emit(
            "Filling stage skipped: the pour location does not resolve to any "
            "metal cell. Move it onto the pour cup, sprue or casting."
        )
        return None

    inlet = _expand_inlet_to_mouth(inlet_seeds, solve_core) + pad_lo
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

    headspace = np.zeros(solve_types.shape, dtype=bool)
    if HEADSPACE_CELLS:
        from scipy import ndimage

        z_top = core[2].stop - 1
        open_above = metal[:, :, z_top].copy()
        mouth = np.zeros(solve_types.shape[:2], dtype=bool)
        on_mouth = inlet[:, 2] == z_top
        mouth[inlet[on_mouth, 0], inlet[on_mouth, 1]] = True
        open_above &= ~ndimage.binary_dilation(
            mouth, structure=np.ones((3, 3), dtype=bool)
        )
        headspace[:, :, core[2].stop : core[2].stop + int(HEADSPACE_CELLS)] = (
            open_above[:, :, None]
        )
        obstacle &= ~headspace

    open_faces = [
        name
        for name, plane in (
            ("x = 0", obstacle[0]),
            ("x = max", obstacle[-1]),
            ("y = 0", obstacle[:, 0]),
            ("y = max", obstacle[:, -1]),
            ("z = 0", obstacle[:, :, 0]),
            ("z = max", obstacle[:, :, -1]),
        )
        if not plane.all()
    ]
    if open_faces:
        emit(
            f"Filling stage skipped: the solve lattice is not closed. "
            f"Non-obstacle cells sit on the {', '.join(open_faces)} "
            f"face{'' if len(open_faces) == 1 else 's'} of the array, and both "
            f"step paths index neighbours periodically, so opposite faces of "
            f"the domain are neighbours of each other -- metal on one face "
            f"would stream and exchange mass straight into whatever sits on "
            f"the other, however far apart they are. The "
            f"{pad_lo}-cell mold pad is what makes that unrepresentable and it "
            f"has been defeated; this is a bug in the stage, not in the model."
        )
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

    alpha_si = float(alloy.thermal_conductivity) / (
        float(alloy.density_liquid) * float(alloy.specific_heat)
    )
    alpha_lattice = alpha_si * units.dt_s / (dx_m * dx_m)

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

    smagorinsky_cs = float(getattr(filling, "smagorinsky_cs", SMAGORINSKY_CS))
    if not 0.0 <= smagorinsky_cs <= MAX_SMAGORINSKY_CS:
        warnings.append(
            f"smagorinsky_cs of {smagorinsky_cs:.3f} is outside the usable "
            f"range 0-{MAX_SMAGORINSKY_CS:.1f} (literature values are "
            f"0.1-0.2); running at the default {SMAGORINSKY_CS:.2f}."
        )
        smagorinsky_cs = float(SMAGORINSKY_CS)

    head_m = float(inches_to_meters(float(getattr(filling, "initial_head_in", 4.0))))
    head_lattice = head_m / dx_m
    inlet_density = REFERENCE_DENSITY * (
        1.0 + units.gravity_lattice * head_lattice / LATTICE_CS2
    )

    inlet_area_m2 = float(inlet.shape[0]) * dx_m * dx_m
    inlet_mode = int(getattr(filling, "inlet_mode_code", 0))
    ix, iy, iz = inlet[:, 0], inlet[:, 1], inlet[:, 2]
    mouth_cells = int(np.count_nonzero(inlet[:, 2] == int(inlet[:, 2].max())))
    mouth_area_m2 = float(max(mouth_cells, 1)) * dx_m * dx_m
    weight_lb = pour_weight_lb(voxel_mesh, alloy)
    recommended_lb_s = recommended_pour_rate_lb_s(weight_lb)
    u_inlet_physical = 0.0
    inlet_mode_label = ""
    if inlet_mode == 1:
        pour_rate_lb_s = float(getattr(filling, "pour_rate_lb_s", 0.0) or 0.0)
        if pour_rate_lb_s <= 0.0:
            pour_rate_lb_s = recommended_lb_s
            rate_source = f"sqrt of {weight_lb:.0f} lb"
        else:
            rate_source = "set"
        if pour_rate_lb_s <= 0.0:
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

    u_cap_lattice = float(units.u_max_lattice)
    if u_filter_max is not None:
        u_cap_lattice = min(
            u_cap_lattice,
            float(units.to_lattice_velocity(np.asarray(u_filter_max, dtype=np.float64))),
        )
        if inlet_mode != 1 and u_cap_lattice < float(units.u_max_lattice):
            inlet_mode_label += ", filter-limited"

    lb_per_lattice_mass = (
        float(alloy.density_liquid) * (dx_m ** 3)
        / float(REFERENCE_DENSITY) / 0.45359237
    )
    torricelli_m_s = float(np.sqrt(2.0 * 9.81 * max(head_m, 0.0)))
    if inlet_mode == 1:
        target_kg_s = (
            u_inlet_physical * float(alloy.density_liquid) * inlet_area_m2
        )
        target_source = "prescribed"
    else:
        pour_rate_override_lb_s = float(
            getattr(filling, "pour_rate_lb_s", 0.0) or 0.0
        )
        if pour_rate_override_lb_s > 0.0:
            target_kg_s = pour_rate_override_lb_s * 0.45359237
            target_source = "set"
        else:
            mouth_ceiling_lb_s = (
                INLET_DISCHARGE_COEFFICIENT
                * float(alloy.density_liquid)
                * mouth_area_m2
                * torricelli_m_s
                / 0.45359237
            )
            if recommended_lb_s > 0.0 and recommended_lb_s < mouth_ceiling_lb_s:
                target_kg_s = recommended_lb_s * 0.45359237
                target_source = (
                    f"sqrt of {weight_lb:.0f} lb; the {mouth_cells:,}-cell "
                    f"mouth could pass {mouth_ceiling_lb_s:.1f}"
                )
            else:
                target_kg_s = mouth_ceiling_lb_s * 0.45359237
                target_source = (
                    f"Cd {INLET_DISCHARGE_COEFFICIENT:.2f} x "
                    f"{mouth_cells:,}-cell mouth x sqrt(2gh)"
                )
                if recommended_lb_s > 0.0:
                    warnings.append(
                        f"the mouth cannot pass the rate this casting wants: "
                        f"{mouth_ceiling_lb_s:.1f} lb/s through the "
                        f"{mouth_cells:,}-cell mouth at "
                        f"{head_m * 39.3701:.1f} in of head, against "
                        f"{recommended_lb_s:.1f} lb/s from the square-root "
                        f"rule for {weight_lb:.0f} lb. Pouring at what the "
                        f"rigging passes -- the fill will be slower than the "
                        f"casting wants, which is where cold shuts come from. "
                        f"Widen the sprue mouth or raise the head."
                    )

    if flow_limit_kg_s is not None and target_kg_s > flow_limit_kg_s:
        if inlet_mode != 1:
            warnings.append(
                f"delivery limited by the filter: the head would drive "
                f"{target_kg_s / 0.45359237:.1f} lb/s through the mouth, "
                f"{flow_limit_kg_s / 0.45359237:.1f} lb/s is rated across "
                f"{len(filter_zones)} filter(s). Metering at the rating."
            )
            inlet_mode_label += ", filter-limited"
        target_kg_s = float(flow_limit_kg_s)
        target_source += ", filter-rated"

    u_target_lattice = float(
        units.to_lattice_velocity(
            np.asarray(
                target_kg_s / max(float(alloy.density_liquid) * mouth_area_m2, 1e-9),
                dtype=np.float64,
            )
        )
    )
    if u_target_lattice > u_cap_lattice:
        capped_kg_s = target_kg_s * (u_cap_lattice / max(u_target_lattice, 1e-12))
        warnings.append(
            f"pour rate unachievable at this resolution: delivering "
            f"{target_kg_s / 0.45359237:.1f} lb/s through the "
            f"{mouth_cells:,}-cell mouth ({mouth_area_m2 * 1550.0:.2f} in^2) "
            f"implies {u_target_lattice:.4f} lattice velocity against a guard "
            f"of {u_cap_lattice:.4f}. Metering at "
            f"{capped_kg_s / 0.45359237:.1f} lb/s instead -- the fill will run "
            f"slower than the real pour. Reduce fill_grid_coarsening or voxel "
            f"size to widen the mouth in cells, or raise "
            f"lattice_velocity_max."
        )
        target_kg_s = capped_kg_s
        u_target_lattice = u_cap_lattice
        target_source += ", Mach-capped"

    u_mouth_ceiling_lattice = min(
        u_cap_lattice,
        float(units.to_lattice_velocity(
            np.asarray(
                INLET_DISCHARGE_COEFFICIENT * torricelli_m_s, dtype=np.float64
            )
        )),
    )

    target_lb_s = target_kg_s / 0.45359237
    target_mass_per_step = (
        target_lb_s * float(units.dt_s) / max(lb_per_lattice_mass, 1e-30)
    )

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
        inlet_u[2, :] = -u_inlet_lattice
        inlet_full_mass = np.full(
            inlet.shape[0], float(REFERENCE_DENSITY), dtype=np.float64
        )
        emit(
            f"Filling: inlet {inlet.shape[0]:,} cells{inlet_widening}, "
            f"{inlet_mode_label}, "
            f"{u_inlet_physical:.2f} m/s ({u_inlet_lattice:.4f} lattice); "
            f"metering {target_lb_s:.1f} lb/s ({target_source}), "
            f"sqrt-rule cross-check {recommended_lb_s:.1f} lb/s "
            f"for {weight_lb:.0f} lb"
        )
    else:
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
            f"({u_cap_lattice:.4f} lattice); metering {target_lb_s:.1f} lb/s "
            f"({target_source}, {torricelli_m_s:.2f} m/s efflux, "
            f"{u_target_lattice:.4f} lattice), sqrt-rule cross-check "
            f"{recommended_lb_s:.1f} lb/s for {weight_lb:.0f} lb"
        )

    if inlet_mode != 1:
        solve_dx_in = mapping.dx_in * coarsening
        z_centres_in = float(mapping.origin_in[2]) + (
            (np.arange(solve_types.shape[2], dtype=np.float64) - pad_lo)
            * coarsening
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

    choke_by_part = coarsening_decision.by_part
    choke_thickness_cells = (
        min(choke_by_part.values()) if choke_by_part else 0.0
    )
    narrowest_code = narrowest_choke_part_code(choke_by_part)
    if choke_by_part:
        emit(
            "Filling: channel thickness on the "
            f"{dx_m * 1000:.1f} mm fill grid -- "
            + ", ".join(
                f"{_METAL_PATH_LABELS[code]} "
                + (
                    "/".join(
                        f"{value:.1f}"
                        for value in coarsening_decision.components_by_part[code]
                    )
                    if len(coarsening_decision.components_by_part.get(code, ()))
                    > 1
                    else f"{choke_by_part[code]:.1f}"
                )
                + " cells"
                for code in _METAL_PATH_CODES
                if code in choke_by_part
            )
            + (
                f" (narrowest: {_METAL_PATH_LABELS[narrowest_code]})"
                if narrowest_code is not None
                else ""
            )
        )
    choke_components = coarsening_decision.components_by_part
    for code in _METAL_PATH_CODES:
        thickness = choke_by_part.get(code)
        if thickness is None or thickness <= 0.0:
            continue
        bodies = choke_components.get(code, ())
        label = _METAL_PATH_LABELS[code]
        if len(bodies) > 1:
            label = f"thinnest of the {len(bodies)} {label} bodies"
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
                f"the thinnest channel this solver has ever carried a complete "
                f"fill through -- and it carries it at the edge of the velocity "
                f"range the timestep was sized for. Treat the flow through the "
                f"{label} as marginal, not as a result; if it matters to the "
                f"answer, resolve it and rerun."
            )

    for code, sizes in coarsening_decision.fragments_by_part.items():
        label = _METAL_PATH_LABELS.get(code, "gating")
        listed = ", ".join(f"{size:,}" for size in sizes)
        warnings.append(
            f"{len(sizes)} disconnected "
            f"{'fragment' if len(sizes) == 1 else 'fragments'} of {label} "
            f"({listed} cells) "
            f"{'was' if len(sizes) == 1 else 'were'} ignored by the choke "
            f"guard: below {MIN_CHOKE_BODY_CELLS} cells a body has no "
            f"cross-section in any orientation, and it would read as sealed at "
            f"every resolution and pin the grid at coarsening 1 for good. The "
            f"{label}'s reported thickness is its main body. Check the part "
            f"assignment if you did not expect a detached "
            f"{'piece' if len(sizes) == 1 else 'pieces'} of {label}."
        )

    static_vent = _static_vent_mask(
        solve_types, inlet, bool(getattr(filling, "vent_risers", 1))
    )
    open_air_mask = np.zeros(solve_types.shape, dtype=bool)
    open_air_mask[core] = _open_air_mask(core_shape)
    open_air_mask |= headspace
    open_air_mask &= ~obstacle
    entrapment_check_interval = max(
        1, int(getattr(filling, "entrapment_check_interval", 50))
    )
    peak_entrapped_voxels = 0
    vent_warning: Optional[str] = None

    report_every = max(1, max_steps // 20)
    frame_every = max(1, max_steps // (FILL_ANIMATION_FRAMES * 4))
    frames: List[FillFrame] = []

    def capture_frame(step_index: int) -> None:
        eps = np.clip(fluid_fraction(state), 0.0, 1.0)
        eps[state.flags == FLAG_OBSTACLE] = 0.0
        speed = np.sqrt(np.sum(np.square(state.u.astype(np.float64)), axis=0))
        speed *= units.velocity_scale
        frames.append(
            FillFrame(
                time_s=step_index * units.dt_s,
                fill_fraction=to_voxels(eps).astype(np.float32),
                speed_m_s=to_voxels(speed).astype(np.float32),
            )
        )

    taken = 0
    metal_total = int(metal.sum())
    metal_full_mass = float(
        np.sum(
            float(REFERENCE_DENSITY)
            * np.asarray(state.porous.porosity, dtype=np.float64)[metal]
        )
    )
    poured = 0.0
    poured_lb_per_lattice_mass = lb_per_lattice_mass
    lattice_c_t = LATTICE_VELOCITIES.T.astype(np.float64)
    inlet_u_measured = np.zeros((3, inlet.shape[0]), dtype=np.float64)

    inlet_mask = np.zeros(solve_types.shape, dtype=bool)
    inlet_mask[ix, iy, iz] = True
    inlet_halo = np.zeros(solve_types.shape, dtype=bool)
    for _c in LATTICE_VELOCITIES[1:]:
        inlet_halo |= np.roll(
            inlet_mask, shift=(int(_c[0]), int(_c[1]), int(_c[2])), axis=(0, 1, 2)
        )
    inlet_halo &= ~inlet_mask & ~obstacle
    hx, hy, hz = np.nonzero(inlet_halo)

    delivery_layers: List[Tuple[Tuple[np.ndarray, ...], np.ndarray]] = []
    _frontier = inlet_mask
    _claimed = inlet_mask.copy()
    for _ in range(INLET_DELIVERY_DEPTH_CELLS):
        _below = np.zeros_like(_frontier)
        _below[:, :, :-1] = _frontier[:, :, 1:]
        _below &= metal & ~obstacle & ~_claimed
        if not _below.any():
            break
        _claimed |= _below
        _idx = np.nonzero(_below)
        delivery_layers.append(
            (_idx, np.asarray(state.porous.porosity[_idx], dtype=np.float64))
        )
        _frontier = _below

    status = STATUS_TIME_CAPPED
    diverged_at_step = 0
    flat_reports = 0
    last_reached = -1
    best_mass_fraction = -1.0
    negative_history: List[int] = []
    cascade_streak = 0
    fill_fraction_breaches = 0
    delivered_previous = 0.0
    inlet_drive_head = inlet_density - float(REFERENCE_DENSITY)
    u_drive = 0.0
    thx, thy, thz = (
        delivery_layers[0][0] if delivery_layers else (hx, hy, hz)
    )
    poured_at_last_report = 0.0
    step_at_last_report = 0
    for taken in range(1, max_steps + 1):
        state.flags[ix, iy, iz] = FLAG_FLUID
        if np.any(state.flags[hx, hy, hz] == FLAG_GAS):
            ensure_interface_skin(state, inlet_halo)
        if inlet_mode == 1:
            state.f[:, ix, iy, iz] = equilibrium(inlet_rho, inlet_u).astype(np.float32)
        else:
            throat_wet = state.flags[thx, thy, thz] >= FLAG_INTERFACE
            rho_throat = (
                float(
                    np.asarray(
                        state.rho[thx, thy, thz], dtype=np.float64
                    )[throat_wet].mean()
                )
                if throat_wet.any()
                else float(REFERENCE_DENSITY)
            )
            inlet_drive_head += INLET_VELOCITY_BETA * (
                (inlet_density - rho_throat) - inlet_drive_head
            )
            u_ceiling = min(
                u_cap_lattice,
                INLET_DISCHARGE_COEFFICIENT * float(np.sqrt(
                    2.0 * LATTICE_CS2
                    * max(0.0, inlet_drive_head)
                    / float(REFERENCE_DENSITY)
                )),
            )
            delivery_error = (
                target_mass_per_step - delivered_previous
            ) / max(target_mass_per_step, 1e-30)
            u_drive = float(np.clip(
                u_drive + INLET_RATE_GAIN * delivery_error * u_target_lattice,
                0.0,
                u_ceiling,
            ))
            f_slice = state.f[:, ix, iy, iz].astype(np.float64)
            rho_now = f_slice.sum(axis=0)
            u_raw = lattice_c_t @ f_slice / np.clip(rho_now, 1e-9, None)
            inlet_u_measured *= 1.0 - INLET_VELOCITY_BETA
            inlet_u_measured += INLET_VELOCITY_BETA * u_raw
            u_used = inlet_u_measured.copy()
            np.minimum(u_used[2], -u_drive, out=u_used[2])
            np.maximum(u_used[2], -u_mouth_ceiling_lattice, out=u_used[2])
            speed = np.sqrt(np.sum(u_used * u_used, axis=0))
            u_used *= np.minimum(1.0, u_cap_lattice / np.maximum(speed, 1e-12))
            state.f[:, ix, iy, iz] = equilibrium(inlet_rho, u_used).astype(np.float32)
        inlet_refill = float(np.sum(inlet_full_mass)) - float(
            np.sum(state.mass[ix, iy, iz])
        )
        state.mass[ix, iy, iz] = inlet_full_mass
        delivered = inlet_refill
        budget = target_mass_per_step - delivered
        if budget > 0.0:
            for _idx, _porosity in delivery_layers:
                wet_here = state.flags[_idx] == FLAG_INTERFACE
                if not wet_here.any():
                    continue
                room = np.where(
                    wet_here,
                    np.asarray(state.rho[_idx], dtype=np.float64) * _porosity
                    - np.asarray(state.mass[_idx], dtype=np.float64),
                    0.0,
                )
                np.maximum(room, 0.0, out=room)
                want = float(room.sum())
                if want <= 0.0:
                    continue
                if want > budget:
                    room *= budget / want
                    want = budget
                state.mass[_idx] += room
                delivered += want
                budget -= want
                if budget <= 0.0:
                    break
        poured += delivered
        delivered_previous = inlet_refill
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
            detailed_progress_callback(taken, max_steps, "Filling")

        if taken % frame_every == 0:
            capture_frame(taken)
            if len(frames) >= 2 * FILL_ANIMATION_FRAMES:
                del frames[1::2]
                frame_every *= 2

        if taken % DIVERGENCE_CHECK_INTERVAL == 0:
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

            negative_cells = (
                int(np.count_nonzero(np.any(state.f < 0.0, axis=0) & wet))
                if wet_cells
                else 0
            )
            negative_history.append(1 if negative_cells else 0)
            if len(negative_history) > NEGATIVE_POPULATION_WINDOW:
                del negative_history[0]
            prevalence = (
                sum(negative_history) / float(NEGATIVE_POPULATION_WINDOW)
                if len(negative_history) >= NEGATIVE_POPULATION_WINDOW
                else 0.0
            )
            cascading_now = (
                negative_cells >= NEGATIVE_POPULATION_CASCADE_MIN_CELLS
                and negative_cells
                >= NEGATIVE_POPULATION_CASCADE_FRACTION * max(wet_cells, 1)
            )
            cascade_streak = cascade_streak + 1 if cascading_now else 0
            cascading = cascade_streak >= NEGATIVE_POPULATION_CASCADE_CHECKS
            if cascading or prevalence >= NEGATIVE_POPULATION_PREVALENCE:
                status = STATUS_DIVERGED
                diverged_at_step = taken
                if cascading:
                    precursor = (
                        f"{negative_cells:,} of {wet_cells:,} wet cells "
                        f"({negative_cells / max(wet_cells, 1):.1%}) are "
                        f"holding a negative population, at "
                        f"{NEGATIVE_POPULATION_CASCADE_CHECKS} consecutive "
                        f"checks "
                        f"({NEGATIVE_POPULATION_CASCADE_CHECKS * DIVERGENCE_CHECK_INTERVAL} "
                        f"steps) -- a reconstruction cascade in progress, not a "
                        f"transient"
                    )
                else:
                    precursor = (
                        f"negative populations have been present at "
                        f"{prevalence:.0%} of the last "
                        f"{NEGATIVE_POPULATION_WINDOW} checks "
                        f"({NEGATIVE_POPULATION_WINDOW * DIVERGENCE_CHECK_INTERVAL:,} "
                        f"steps), {negative_cells:,} cells at this one. The "
                        f"free-surface reconstruction is failing persistently "
                        f"and this run will go non-finite"
                    )
                emit(
                    f"Filling ABORTED: solver diverging at step {taken:,} "
                    f"({taken * units.dt_s:.2f} s). {precursor}. Stopped "
                    f"before the field lost finiteness, so the state below is "
                    f"still readable -- but it is not a fill. Results "
                    f"discarded."
                )
                break

            fluid_cells = state.flags == FLAG_FLUID
            eps_fluid = fluid_fraction(state)
            out_of_band = fluid_cells & (
                (eps_fluid < FLUID_FILL_FRACTION_MIN)
                | (eps_fluid > FLUID_FILL_FRACTION_MAX)
            )
            offenders = int(np.count_nonzero(out_of_band))
            fill_fraction_breaches = fill_fraction_breaches + 1 if offenders else 0
            if fill_fraction_breaches >= FLUID_FILL_FRACTION_CHECKS:
                worst = float(eps_fluid[out_of_band].max())
                lowest = float(eps_fluid[out_of_band].min())
                where_z = np.nonzero(np.any(out_of_band, axis=(0, 1)))[0]
                parts = "/".join(
                    sorted(
                        CODE_TO_PART_TYPE[int(code)].value.replace("_", " ")
                        for code in np.unique(solve_types[out_of_band])
                        if int(code) in CODE_TO_PART_TYPE
                    )
                ) or "the mold pad"
                status = STATUS_DIVERGED
                diverged_at_step = taken
                emit(
                    f"Filling ABORTED: solver diverging at step {taken:,} "
                    f"({taken * units.dt_s:.2f} s). {offenders:,} FLUID "
                    f"cell{'' if offenders == 1 else 's'} "
                    f"hold{'s' if offenders == 1 else ''} a fill fraction "
                    f"outside "
                    f"[{FLUID_FILL_FRACTION_MIN:.1f}, "
                    f"{FLUID_FILL_FRACTION_MAX:.1f}] -- as low as "
                    f"{lowest:.3f} and as high as {worst:.3f} -- in the "
                    f"{parts}, at solve z "
                    f"{int(where_z.min()) - pad_lo}-"
                    f"{int(where_z.max()) - pad_lo}, and did so at "
                    f"{FLUID_FILL_FRACTION_CHECKS} consecutive checks "
                    f"({FLUID_FILL_FRACTION_CHECKS * DIVERGENCE_CHECK_INTERVAL:,} "
                    f"steps), so this is not a transient. A full cell holds "
                    f"exactly its pore volume, so the mass bookkeeping has "
                    f"come apart rather than the metal having backed up: "
                    f"nothing in the solver can take mass back out of a FLUID "
                    f"cell, and because the exchange conserves mass exactly "
                    f"the surplus was stolen from real cells. Results "
                    f"discarded."
                )
                break

        if taken % entrapment_check_interval == 0:
            vent_mask = static_vent | (open_air_mask & (state.flags == FLAG_GAS))
            if vent_warning is None:
                vent_warning = _entrapment_warning_if_no_vent(vent_mask)
            entrapped_now = find_entrapped_gas(state.flags, vent_mask)
            entrapped_now_count = int(np.count_nonzero(to_voxels(entrapped_now)))
            peak_entrapped_voxels = max(peak_entrapped_voxels, entrapped_now_count)

        if taken % report_every == 0:
            if cancel_callback is not None:
                cancel_callback()
            reached = int(np.count_nonzero((state.flags == FLAG_FLUID) & metal))
            mass_fraction = float(
                np.clip(float(np.sum(state.mass[metal])), 0.0, metal_full_mass)
            ) / max(metal_full_mass, 1e-12)
            interval_steps = max(1, taken - step_at_last_report)
            delivery_lb_s = (
                (poured - poured_at_last_report) * poured_lb_per_lattice_mass
                / (interval_steps * units.dt_s)
            )
            poured_at_last_report = poured
            step_at_last_report = taken
            emit(
                f"Filling: {reached:,}/{metal_total:,} cells "
                f"({reached / max(1, metal_total):.0%}), "
                f"{mass_fraction:.0%} by mass, "
                f"{poured * poured_lb_per_lattice_mass:.1f} lb poured, "
                f"delivering {delivery_lb_s:.2f} of "
                f"{target_lb_s:.2f} lb/s target "
                f"({delivery_lb_s / max(target_lb_s, 1e-9):.0%}), at "
                f"{taken * units.dt_s:.2f} s"
            )
            if reached >= metal_total:
                status = STATUS_COMPLETE
                break
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
        frames.clear()
    else:
        capture_frame(taken)

    final_vent_mask = static_vent | (open_air_mask & (state.flags == FLAG_GAS))
    if vent_warning is None:
        vent_warning = _entrapment_warning_if_no_vent(final_vent_mask)
    entrapped_coarse = find_entrapped_gas(state.flags, final_vent_mask)

    entrapped_voxels = to_voxels(entrapped_coarse).astype(bool)
    entrapped_count = int(np.count_nonzero(entrapped_voxels))
    peak_entrapped_count = max(entrapped_count, peak_entrapped_voxels)

    liquidus_c = float(getattr(alloy, "liquidus_temp", -np.inf))
    cold_shut_coarse = find_cold_shuts(
        state.fill_time_s,
        state.temperature,
        state.flags,
        liquidus_c=liquidus_c,
        arrival_gap_s=float(getattr(filling, "cold_shut_arrival_gap_s", 0.5)),
        temperature_margin_c=float(getattr(filling, "cold_shut_temp_margin_c", 25.0)),
    )
    cold_shut_voxels = to_voxels(cold_shut_coarse).astype(bool)
    cold_shut_count = int(np.count_nonzero(cold_shut_voxels))

    filled_coarse = state.flags == FLAG_FLUID
    fill_time_coarse = np.asarray(state.fill_time_s, dtype=np.float64)

    fill_time_voxels = to_voxels(fill_time_coarse)
    filled_voxels = to_voxels(filled_coarse).astype(bool)

    voxel_metal = to_voxels(metal).astype(bool)
    filled_count = int(np.count_nonzero(filled_voxels & voxel_metal))
    misrun_count = int(np.count_nonzero(voxel_metal & ~filled_voxels))

    fill_time_voxels = np.where(voxel_metal, fill_time_voxels, np.nan)

    temperature_fill = to_voxels(
        np.asarray(state.temperature, dtype=np.float64)
    )

    filled_mask = filled_voxels & voxel_metal & np.isfinite(temperature_fill)

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
        choke_area_cm2=measure_choke_area_cm2(solve_core, dx_m),
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
        choke_thickness_bodies_by_part={
            _METAL_PATH_LABELS[code]: list(values)
            for code, values in choke_components.items()
        },
        choke_body_fragments_by_part={
            _METAL_PATH_LABELS[code]: list(sizes)
            for code, sizes in coarsening_decision.fragments_by_part.items()
        },
        narrowest_choke_part=(
            _METAL_PATH_LABELS[narrowest_code]
            if narrowest_code is not None
            else ""
        ),
        choke_area_fine_cm2=measure_choke_area_cm2(fine_types, mapping.dx_m),
        delivered_lb=poured * poured_lb_per_lattice_mass,
        target_delivery_lb_s=target_lb_s,
        configured_coarsening=coarsening_decision.configured,
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
