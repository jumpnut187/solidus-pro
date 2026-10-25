"""Foundry design rules for synthetic training castings: rig risers and chills, then check the design.

Surrogate models learn whatever the training castings show, so a training
casting has to be one a foundry engineer would sign off. This module rigs a
casting (risers, chills) from steel-foundry rules of thumb and checks a
finished assembly against them; ``core.surrogate.assemblies`` only keeps
designs that pass.

Everything runs on a raster of the assembly (cells at ``pitch`` inches,
part overlap resolved toward the higher part code like the solver's
voxeliser: CAST < RISER < CHILL < CORE).

Rules (steel; ``core/riser_design.py`` uses the same constants):

- Riser modulus (Caine / Wlodawer): ``MEF * M_riser >= RISER_MODULUS_FACTOR *
  M_zone`` (MEF: insulating sleeve, ``RISER_SLEEVE_MEF``), where the zone is the metal a riser feeds (nearest riser) and
  ``M_zone = V / A`` with A the zone's cooling surface: faces to sand and
  chill count fully; faces to a core, a riser or another zone not at all. A
  core saturates (heats through) well before the metal around it freezes,
  so it stops extracting heat; counting core faces at half cooling left the
  metal around cores freezing after the risers in most solved housings and
  hubs. M_zone is at least the modulus of the zone's heaviest section
  (Wlodawer): the metal within ``HEAVY_SECTION_FRACTION`` of the zone's
  largest inscribed radius of its thickest cells, whose faces to the rest of
  the casting do not cool. Averaged with a thin flange, a heavy hub's own
  modulus was diluted and its riser froze before it. A cylinder riser with
  H = 1.5 D standing on the casting has M = 0.214 D.
- Riser placement: each riser stands on a flat, clear patch of the top
  surface over the heaviest unfed section (largest inscribed radius). The
  print of a cored bore inside the top surface counts as part of the patch
  (the core runs up into the riser), as long as metal is under at least
  ``MIN_SEATED_FRACTION`` of the riser's base.
- Feeding distance (SFSA plate/bar rules): metal is fed when it lies within
  ``min(4.5 T, 6 sqrt(T))`` of a riser's edge, T the zone's section
  thickness (twice its largest inscribed radius), or within
  ``CHILL_EXTENSION_T`` T (at most ``CHILL_EXTENSION_MAX_IN``) of a chill.
- Chills go where feeding falls short, flush on the bottom or a side face
  nearest the unfed metal (or, as an end chill, the metal farthest from the
  risers), at least ``max(0.75 in, T / 2)`` thick, never under a riser.
- Cores need a print (a face to the sand outside the casting) so they can be
  located, vented and knocked out; a fully enclosed core cannot be made.
  Cored holes are at least ``MIN_CORE_DIAMETER_IN`` across.
- One connected casting; every riser fully seated on metal; every chill in
  contact with the casting.

Rules of thumb can still miss a hot spot, so the solved run is checked too
(``unfed_metal``): the last metal to freeze must be in a riser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from scipy import ndimage

from core.cut_cells import winding_number_contains
from core.part_manager import PART_TYPE_CAST, PART_TYPE_CHILL, PART_TYPE_CORE, PART_TYPE_RISER

SAND, METAL, CORE, CHILL, RISER = 0, 1, 2, 3, 4
#: Column label only (``_columns``): the print of a cored bore a riser may stand over.
BORE = 5
_LABEL_OF_CODE = {PART_TYPE_CAST: METAL, PART_TYPE_RISER: RISER, PART_TYPE_CHILL: CHILL, PART_TYPE_CORE: CORE}
#: Rasterisation order: later labels overwrite earlier ones (the higher part code wins).
_RASTER_ORDER = (PART_TYPE_CAST, PART_TYPE_RISER, PART_TYPE_CHILL, PART_TYPE_CORE)

RISER_MODULUS_FACTOR = 1.2
#: Modulus extension of an insulating riser sleeve. The solver gives every
#: riser the insulated-sleeve boundary (``ThermalBoundaryParams.htc_riser``,
#: 220 vs 500 W/m^2K on casting faces), so designs are sized as sleeved
#: risers; 1.3 is the usual insulating-sleeve figure.
RISER_SLEEVE_MEF = 1.3
RISER_HEIGHT_RATIO = 1.5
RISER_MODULUS_PER_DIAMETER = 0.375 / 1.75  # H = 1.5 D, base on the casting
RISER_SINK_IN = 0.1
MIN_RISER_DIAMETER_IN = 0.75
MAX_RISERS = 8
MAX_CHILLS = 4
FED_FRACTION_REQUIRED = 0.99
CHILL_EXTENSION_T = 2.0
CHILL_EXTENSION_MAX_IN = 2.0
MIN_CORE_DIAMETER_IN = 0.5
CORE_FACE_WEIGHT = 0.0
MAX_RASTER_CELLS = 1_200_000
HEAVY_SECTION_FRACTION = 0.5
MIN_SEATED_FRACTION = 0.8

Part = Tuple[trimesh.Trimesh, int]


@dataclass
class Raster:
    labels: np.ndarray  # (nx, ny, nz) int8
    origin: np.ndarray  # (3,) centre of cell (0, 0, 0), inches
    pitch: float

    def centres(self, mask: np.ndarray) -> np.ndarray:
        return self.origin + np.argwhere(mask) * self.pitch


@dataclass
class DesignReport:
    violations: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.violations


def unfed_metal(part_types: np.ndarray, end_times: np.ndarray) -> Optional[str]:
    """Why a solved assembly is unsound, or ``None``: casting metal that freezes after every riser.

    ``part_types`` and ``end_times`` are per solver voxel. Metal still liquid
    when the last riser freezes cannot be fed, so it shrinks.
    """
    part_types = np.asarray(part_types)
    end_times = np.asarray(end_times, dtype=np.float64)
    cast = part_types == PART_TYPE_CAST
    riser = part_types == PART_TYPE_RISER
    if not riser.any() or not cast.any():
        return None
    riser_end = float(end_times[riser].max())
    late = cast & (end_times > riser_end)
    if not late.any():
        return None
    return (
        f"unfed metal: {late.sum() / cast.sum():.2%} of the casting freezes after the last riser "
        f"({float(end_times[cast].max()):.0f} s vs {riser_end:.0f} s)"
    )


def feeding_distance(thickness_in: float) -> float:
    """Steel feeding distance from a riser's edge for section thickness T (inches)."""
    t = max(float(thickness_in), 1e-6)
    return min(4.5 * t, 6.0 * np.sqrt(t))


def riser_diameter_for(zone_modulus: float, factor: float = RISER_MODULUS_FACTOR) -> float:
    """Sleeved riser diameter whose effective modulus is ``factor`` times ``zone_modulus``."""
    return float(factor * zone_modulus / (RISER_SLEEVE_MEF * RISER_MODULUS_PER_DIAMETER))


def rasterize(parts: Sequence[Part], frame: Optional[Raster] = None) -> Raster:
    """Label raster of ``parts`` on ``frame``'s cells (parts outside it are clipped).

    Without a frame the box is the parts' bounds plus one cell, at max
    dimension / 80 coarsened to ``MAX_RASTER_CELLS``.
    """
    if frame is None:
        lo = np.min([m.bounds[0] for m, _ in parts], axis=0)
        hi = np.max([m.bounds[1] for m, _ in parts], axis=0)
        pitch = float(np.max(hi - lo)) / 80.0
        while np.prod(np.ceil((hi - lo) / pitch) + 3) > MAX_RASTER_CELLS:
            pitch *= 1.1
        origin = lo - pitch
        shape = tuple(int(n) for n in np.ceil((hi - lo) / pitch) + 3)
    else:
        pitch, origin, shape = frame.pitch, frame.origin, frame.labels.shape
    labels = np.zeros(shape, dtype=np.int8)
    for code in _RASTER_ORDER:
        for mesh, part_code in parts:
            if part_code != code:
                continue
            for shell in mesh.split(only_watertight=False) if len(mesh.faces) else []:
                i0 = np.maximum(np.floor((shell.bounds[0] - origin) / pitch).astype(int), 0)
                i1 = np.minimum(np.ceil((shell.bounds[1] - origin) / pitch).astype(int) + 1, shape)
                idx = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(i0, i1)], indexing="ij"), -1).reshape(-1, 3)
                if len(idx) == 0:
                    continue
                # Winding number, not shell.contains: contains builds an embree
                # scene per new shell object, rigging rasterises hundreds of
                # times per design, and the scenes pile up until embree
                # segfaults while building one.
                inside = np.asarray(winding_number_contains(shell, origin + idx * pitch), dtype=bool)
                hit = idx[inside]
                labels[hit[:, 0], hit[:, 1], hit[:, 2]] = _LABEL_OF_CODE[code]
    return Raster(labels=labels, origin=origin, pitch=float(pitch))


def _neighbour(labels: np.ndarray, axis: int, step: int) -> np.ndarray:
    """Label of the face neighbour along ``axis``; outside the raster reads as sand."""
    shifted = np.roll(labels, -step, axis=axis)
    edge = [slice(None)] * 3
    edge[axis] = slice(-1, None) if step > 0 else slice(0, 1)
    shifted[tuple(edge)] = SAND
    return shifted


def _zone_moduli(raster: Raster, zones: np.ndarray, n_zones: int) -> np.ndarray:
    """Modulus per zone (inches): the larger of its V / A and its heaviest section's; see the module docstring."""
    labels = raster.labels
    pitch = raster.pitch
    metal = labels == METAL
    cooling = np.zeros(labels.shape)
    for axis in range(3):
        for step in (-1, 1):
            nb = _neighbour(labels, axis, step)
            cooling += np.where(nb == SAND, 1.0, 0.0) + np.where(nb == CHILL, 1.0, 0.0) + np.where(nb == CORE, CORE_FACE_WEIGHT, 0.0)
    volume = np.bincount(zones[metal], minlength=n_zones).astype(float) * pitch ** 3
    area = np.bincount(zones[metal], weights=cooling[metal], minlength=n_zones) * pitch ** 2
    moduli = volume / np.maximum(area, 1e-9)
    radius = ndimage.distance_transform_edt(metal) * pitch
    for k in range(n_zones):
        zone = metal & (zones == k)
        if not zone.any():
            continue
        # The heaviest section: the thickest cells grown back out to the
        # faces they sit behind. Its faces to the rest of the zone count
        # nothing in ``cooling`` (metal never does).
        depth = HEAVY_SECTION_FRACTION * float(radius[zone].max())
        heavy = zone & (radius >= depth)
        box = tuple(slice(max(int(a) - 2, 0), int(b) + 3) for a, b in zip(np.argwhere(zone).min(axis=0), np.argwhere(zone).max(axis=0)))
        section = np.zeros(labels.shape, dtype=bool)
        section[box] = zone[box] & (ndimage.distance_transform_edt(~heavy[box]) * pitch <= depth + 0.5 * pitch)
        section_area = float(cooling[section].sum())
        if section_area > 0.0:
            moduli[k] = max(moduli[k], float(section.sum()) * pitch / section_area)
    return moduli


@dataclass
class _Riser:
    x: float
    y: float
    z_base: float  # top of the metal it stands on
    diameter: float
    limit: float  # largest diameter the site takes


def _columns(raster: Raster) -> Tuple[np.ndarray, np.ndarray]:
    """Per (x, y) column: index of the topmost non-sand cell (-1 if empty) and its label.

    The print of a cored bore (core-topped columns enclosed by metal-topped
    ones) reads as ``BORE`` at the level of the metal around it, so a riser
    can stand over the bore with the core running up into it.
    """
    occupied = raster.labels != SAND
    nz = raster.labels.shape[2]
    top = np.where(occupied.any(axis=2), nz - 1 - np.argmax(occupied[:, :, ::-1], axis=2), -1)
    top_label = np.where(top >= 0, np.take_along_axis(raster.labels, np.maximum(top, 0)[..., None], axis=2)[..., 0], SAND)
    metal_top = top_label == METAL
    bores, n_bores = ndimage.label((top_label == CORE) & ndimage.binary_fill_holes(metal_top))
    for b in range(1, n_bores + 1):
        bore = bores == b
        rim = ndimage.binary_dilation(bore) & metal_top
        if rim.any():
            top[bore] = int(np.median(top[rim]))
            top_label[bore] = BORE
    return top, top_label


def _site_limit(raster: Raster, top: np.ndarray, top_label: np.ndarray, ix: int, iy: int) -> float:
    """Largest riser diameter centred on column (ix, iy): the patch must be flat, clear metal top.

    A bore in the patch is bridged only while metal stays under
    ``MIN_SEATED_FRACTION`` of the base (sampled out to 0.45 D, as ``check`` does).
    """
    if top_label[ix, iy] not in (METAL, BORE):
        return 0.0
    tol = max(1, int(round(0.2 / raster.pitch)))
    uneven = np.abs(top - top[ix, iy]) > tol
    rho = np.hypot(*np.indices(top.shape) - np.array([ix, iy])[:, None, None]) * raster.pitch

    def clear(bad: np.ndarray) -> float:
        if not bad.any():
            return float(min(raster.labels.shape[:2]) * raster.pitch)
        return float(2.0 * max(float(rho[bad].min()) - 0.5 * raster.pitch, 0.0))

    bore = top_label == BORE
    limit = clear(((top_label != METAL) & ~bore) | uneven)
    if bore.any():
        disk = rho <= 0.45 * limit
        if bore[disk].mean() > 1.0 - MIN_SEATED_FRACTION:
            limit = 0.0 if bore[ix, iy] else clear((top_label != METAL) | uneven)
    return limit


def _distance_to_risers(raster: Raster, centres: np.ndarray, risers: List[_Riser]) -> np.ndarray:
    """(N, R) distance from each point to each riser's base disk edge."""
    out = np.empty((len(centres), len(risers)))
    for k, r in enumerate(risers):
        rho = np.hypot(centres[:, 0] - r.x, centres[:, 1] - r.y)
        out[:, k] = np.hypot(np.maximum(rho - r.diameter / 2, 0.0), centres[:, 2] - r.z_base)
    return out


def _feeding(raster: Raster, risers: List[_Riser], chill_extension: Optional[np.ndarray]):
    """Zone of every metal cell, zone moduli and thicknesses, and the fed mask."""
    labels = raster.labels
    metal = labels == METAL
    centres = raster.centres(metal)
    radius = ndimage.distance_transform_edt(metal) * raster.pitch
    r_metal = radius[metal]
    zones = np.full(labels.shape, 0, dtype=np.int64)
    if not risers:
        return zones, np.zeros(0), np.zeros(0), np.zeros(metal.sum(), dtype=bool)
    dist = _distance_to_risers(raster, centres, risers)
    nearest = np.argmin(dist, axis=1)
    zones[metal] = nearest
    moduli = _zone_moduli(raster, zones, len(risers))
    thickness = np.array([2.0 * float(r_metal[nearest == k].max()) if np.any(nearest == k) else 0.0 for k in range(len(risers))])
    reach = np.array([feeding_distance(t) for t in thickness])
    fed = dist[np.arange(len(centres)), nearest] <= reach[nearest]
    if chill_extension is not None:
        fed |= chill_extension[metal]
    return zones, moduli, thickness, fed


def _chill_extension(raster: Raster, thickness: float) -> np.ndarray:
    chill = raster.labels == CHILL
    if not chill.any():
        return np.zeros(raster.labels.shape, dtype=bool)
    reach = min(CHILL_EXTENSION_T * thickness, CHILL_EXTENSION_MAX_IN)
    return ndimage.distance_transform_edt(~chill) * raster.pitch <= reach


def _place_riser(raster: Raster, target: np.ndarray, top, top_label, risers: List[_Riser]) -> Optional[_Riser]:
    """Riser site nearest ``target`` (a hot-spot centre) that takes at least the minimum diameter."""
    pitch = raster.pitch
    ti, tj = np.round((target[:2] - raster.origin[:2]) / pitch).astype(int)
    span = int(np.ceil(4.0 / pitch))
    best = None
    for di in range(-span, span + 1, max(1, span // 12)):
        for dj in range(-span, span + 1, max(1, span // 12)):
            i, j = ti + di, tj + dj
            if not (0 <= i < top.shape[0] and 0 <= j < top.shape[1]):
                continue
            limit = _site_limit(raster, top, top_label, i, j)
            x, y = raster.origin[0] + i * pitch, raster.origin[1] + j * pitch
            clear = all(np.hypot(x - r.x, y - r.y) > 0.5 * (limit + r.diameter) + pitch for r in risers)
            if limit < MIN_RISER_DIAMETER_IN or not clear:
                continue
            score = np.hypot(di, dj) * pitch - 0.25 * limit
            if best is None or score < best[0]:
                z = raster.origin[2] + (top[i, j] + 0.5) * pitch
                best = (score, _Riser(x=x, y=y, z_base=z, diameter=MIN_RISER_DIAMETER_IN, limit=limit))
    return None if best is None else best[1]


def _axis_hits(mesh: trimesh.Trimesh, point: Sequence[float], axis: int) -> np.ndarray:
    """Coordinates along ``axis`` where the axis-parallel line through ``point`` crosses ``mesh``.

    A NumPy point-in-triangle test on the projection, not ``mesh.ray``:
    the embree intersector builds a native scene per mesh object, rigging
    draws many meshes, and the scenes pile up until embree segfaults.
    """
    other = [a for a in range(3) if a != axis]
    tri = np.asarray(mesh.triangles, dtype=np.float64)
    p = np.asarray(point, dtype=np.float64)[other]
    a, b, c = tri[:, 0, other], tri[:, 1, other], tri[:, 2, other]
    v0, v1, v2 = b - a, c - a, p - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    ok = np.abs(den) > 1e-12
    den = np.where(ok, den, 1.0)
    u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
    v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
    inside = ok & (u >= 0) & (v >= 0) & (u + v <= 1)
    t = tri[inside]
    return t[:, 0, axis] + u[inside] * (t[:, 1, axis] - t[:, 0, axis]) + v[inside] * (t[:, 2, axis] - t[:, 0, axis])


def _riser_mesh(r: _Riser, cast: trimesh.Trimesh) -> trimesh.Trimesh:
    hits = _axis_hits(cast, (r.x, r.y, 0.0), 2)
    top = float(np.max(hits)) if len(hits) else r.z_base
    height = RISER_HEIGHT_RATIO * r.diameter
    mesh = trimesh.creation.cylinder(radius=r.diameter / 2, height=height, sections=48)
    mesh.apply_translation((r.x, r.y, top - RISER_SINK_IN + height / 2))
    return mesh


#: Faces a chill may sit on, as (axis, side): side -1 is the low face. Not the
#: top face, which belongs to the risers.
_CHILL_FACES = ((2, -1), (0, -1), (0, 1), (1, -1), (1, 1))


def _face_chill(
    raster: Raster,
    region: np.ndarray,
    thickness: float,
    risers: List[_Riser],
    chills: List[trimesh.Trimesh],
    cast: Optional[trimesh.Trimesh] = None,
) -> Optional[trimesh.Trimesh]:
    """Chill block flush on the flat casting face nearest ``region`` (a metal-cell mask).

    Tries the bottom and the four side faces, preferring the one with the
    most of ``region`` behind it; the block covers the region's footprint on
    that face, is ``max(0.75, T / 2)`` thick (at most 2.5 in), and must not
    sit under a riser or overlap another chill.
    """
    pitch = raster.pitch
    labels = raster.labels
    chill_t = float(np.clip(max(0.75, 0.5 * thickness), 0.75, 2.5))
    options = []
    for axis, side in _CHILL_FACES:
        lab = np.moveaxis(labels, axis, -1)
        reg = np.moveaxis(region, axis, -1)
        if side > 0:
            lab, reg = lab[..., ::-1], reg[..., ::-1]
        occupied = lab != SAND
        first = np.where(occupied.any(axis=-1), np.argmax(occupied, axis=-1), -1)
        first_label = np.where(first >= 0, np.take_along_axis(lab, np.maximum(first, 0)[..., None], axis=-1)[..., 0], SAND)
        cols = np.argwhere(reg.any(axis=-1) & (first_label == METAL))
        if len(cols) == 0:
            continue
        face = first[cols[:, 0], cols[:, 1]].min()
        cols = cols[first[cols[:, 0], cols[:, 1]] <= face + 1]
        options.append((len(cols), axis, side, cols, face, lab.shape[-1]))
    for _count, axis, side, cols, face, depth in sorted(options, key=lambda o: -o[0]):
        other = [a for a in range(3) if a != axis]
        span = np.clip(cols.max(axis=0) - cols.min(axis=0) + 1, 2, None) * pitch
        size_other = np.minimum(span, 0.8 * span.max())
        centre_other = raster.origin[other] + cols.mean(axis=0) * pitch
        face_index = face if side < 0 else depth - 1 - face
        face_coord = raster.origin[axis] + (face_index + 0.5 * side) * pitch
        if cast is not None:
            # Keep the block inside the casting outline, then snap it flush to
            # the real face (the raster only knows it to a cell), sampling the
            # whole footprint: its centre can sit over a thinner step.
            lo_o = np.maximum(centre_other - size_other / 2, cast.bounds[0][other])
            hi_o = np.minimum(centre_other + size_other / 2, cast.bounds[1][other])
            centre_other, size_other = (lo_o + hi_o) / 2, hi_o - lo_o
            hits = []
            for u in np.linspace(lo_o[0], hi_o[0], 7)[1:-1]:
                for v in np.linspace(lo_o[1], hi_o[1], 7)[1:-1]:
                    point = np.zeros(3)
                    point[other] = (u, v)
                    hits.extend(_axis_hits(cast, point, axis))
            if hits:
                face_coord = float(max(hits) if side > 0 else min(hits))
        centre = np.zeros(3)
        size = np.zeros(3)
        centre[other], size[other] = centre_other, size_other
        centre[axis] = face_coord + side * chill_t / 2
        size[axis] = chill_t
        if axis == 2 and any(np.hypot(centre[0] - r.x, centre[1] - r.y) < r.diameter for r in risers):
            continue
        lo, hi = centre - size / 2, centre + size / 2
        if any(np.all(lo < c.bounds[1]) and np.all(hi > c.bounds[0]) for c in chills):
            continue
        return trimesh.creation.box(extents=size).apply_translation(centre)
    return None


def rig(
    cast: trimesh.Trimesh,
    cores: Sequence[trimesh.Trimesh],
    rng: np.random.Generator,
    *,
    extra_chill_probability: float = 0.3,
    reasons: Optional[List[str]] = None,
) -> Optional[List[Part]]:
    """Risers (and chills where feeding falls short) for ``cast`` with ``cores`` by the module rules.

    Returns the full part list, or None when no design within ``MAX_RISERS``
    risers feeds the casting (``reasons``, when given, collects why). Riser moduli use a factor drawn from
    [1.2, 1.5]: always sound, with the spread a real methods sheet shows.
    """
    base: List[Part] = [(cast, PART_TYPE_CAST)] + [(c, PART_TYPE_CORE) for c in cores]
    raster = rasterize(base)
    metal = raster.labels == METAL
    def fail(why: str) -> None:
        if reasons is not None:
            reasons.append(why)
        return None

    if not metal.any():
        return fail("no metal in the raster")
    radius = ndimage.distance_transform_edt(metal) * raster.pitch
    centres = raster.centres(metal)
    r_metal = radius[metal]
    top, top_label = _columns(raster)
    risers: List[_Riser] = []
    chills: List[trimesh.Trimesh] = []
    factors: List[float] = []
    fed = np.zeros(len(centres), dtype=bool)
    while (~fed).mean() > 1.0 - FED_FRACTION_REQUIRED:
        unfed = ~fed
        hot = centres[unfed][np.argmax(r_metal[unfed])]
        riser = _place_riser(raster, hot, top, top_label, risers) if len(risers) < MAX_RISERS else None
        if riser is None:
            # Feeding falls short where no riser can stand: chill that region.
            if len(chills) >= MAX_CHILLS:
                return fail(f"{(~fed).mean():.1%} unfed after {len(risers)} risers and {len(chills)} chills")
            region = np.zeros(raster.labels.shape, dtype=bool)
            idx = np.argwhere(metal)[unfed]
            region[idx[:, 0], idx[:, 1], idx[:, 2]] = True
            chill = _face_chill(raster, region, 2.0 * float(r_metal[unfed].max()), risers, chills, cast)
            if chill is None:
                return fail(f"{(~fed).mean():.1%} unfed, no riser site and no chill face for it")
            chills.append(chill)
        else:
            risers.append(riser)
            factors.append(float(rng.uniform(RISER_MODULUS_FACTOR, 1.5)))
        if not risers:
            return fail("no riser site on the top surface")
        fed = _size_and_feed(raster, base, cast, risers, chills, factors, float(np.max(2 * r_metal)))
    if risers and rng.uniform() < extra_chill_probability:
        # An end chill at the metal farthest from every riser: sound practice
        # that steepens the gradient toward the risers.
        dist = _distance_to_risers(raster, centres, risers).min(axis=1)
        far = dist >= np.quantile(dist, 0.9)
        region = np.zeros(raster.labels.shape, dtype=bool)
        idx = np.argwhere(metal)[far]
        region[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        chill = _face_chill(raster, region, 2.0 * float(r_metal[far].max()), risers, chills, cast)
        if chill is not None:
            chills.append(chill)
    parts = base + [(_riser_mesh(r, cast), PART_TYPE_RISER) for r in risers] + [(c, PART_TYPE_CHILL) for c in chills]
    report = check(parts)
    return parts if report.ok else fail("check: " + "; ".join(report.violations))


def _resite(raster: Raster, top, top_label, riser: _Riser, need: float, others: List[_Riser]) -> bool:
    """Move ``riser`` to the nearest top-surface site that takes diameter ``need``.

    Within 6 in, or half of ``need``: a riser that size still covers the spot
    it was put on.
    """
    pitch = raster.pitch
    ci, cj = np.round((np.array([riser.x, riser.y]) - raster.origin[:2]) / pitch).astype(int)
    span = int(np.ceil(max(6.0, 0.5 * need) / pitch))
    stride = max(1, span // 20)
    best = None
    for di in range(-span, span + 1, stride):
        for dj in range(-span, span + 1, stride):
            i, j = ci + di, cj + dj
            if not (0 <= i < top.shape[0] and 0 <= j < top.shape[1]):
                continue
            dist = np.hypot(di, dj) * pitch
            if best is not None and dist >= best[0]:
                continue
            x, y = raster.origin[0] + i * pitch, raster.origin[1] + j * pitch
            if any(np.hypot(x - o.x, y - o.y) < 0.5 * (need + o.diameter) + pitch for o in others):
                continue
            limit = _site_limit(raster, top, top_label, i, j)
            if limit >= need:
                best = (dist, x, y, limit, raster.origin[2] + (top[i, j] + 0.5) * pitch)
    if best is None:
        return False
    _, riser.x, riser.y, riser.limit, riser.z_base = best
    return True


def _size_and_feed(raster, base, cast, risers, chills, factors, max_thickness) -> np.ndarray:
    """Size every riser for its zone and return the fed flags of ``raster``'s metal cells.

    Passes alternate between computing zone moduli (with risers and chills in
    the raster, so riser contact stops counting as cooling surface) and
    sizing; a riser whose site cannot take the diameter its zone needs moves
    to the nearest site that can.
    """
    top, top_label = _columns(raster)
    for _pass in range(3):
        parts = base + [(_riser_mesh(r, cast), PART_TYPE_RISER) for r in risers] + [(c, PART_TYPE_CHILL) for c in chills]
        full = rasterize(parts, frame=raster)
        _zones, moduli, _thickness, fed_full = _feeding(full, risers, _chill_extension(full, max_thickness))
        for k, r in enumerate(risers):
            need = riser_diameter_for(moduli[k], factors[k])
            others = [o for o in risers if o is not r]
            if need > r.limit:
                _resite(raster, top, top_label, r, need, others)
            # Neighbouring risers must not touch: cap by the gap to each.
            gap = min((np.hypot(r.x - o.x, r.y - o.y) - o.diameter / 2 - raster.pitch for o in others), default=np.inf)
            limit = min(r.limit, 2.0 * gap)
            r.diameter = float(np.clip(need, MIN_RISER_DIAMETER_IN, max(limit, MIN_RISER_DIAMETER_IN)))
    parts = base + [(_riser_mesh(r, cast), PART_TYPE_RISER) for r in risers] + [(c, PART_TYPE_CHILL) for c in chills]
    full = rasterize(parts, frame=raster)
    _zones, _moduli, _thickness, fed_full = _feeding(full, risers, _chill_extension(full, max_thickness))
    return _remap(fed_full, full, raster)


def _remap(fed_full: np.ndarray, full: Raster, reference: Raster) -> np.ndarray:
    """Fed flags of ``full``'s metal cells, read back on ``reference``'s metal cells (sand-claimed cells are fed)."""
    grid = np.ones(reference.labels.shape, dtype=bool)
    grid[full.labels == METAL] = fed_full
    return grid[reference.labels == METAL]


def check(parts: Sequence[Part]) -> DesignReport:
    """Check a finished assembly against the module rules."""
    report = DesignReport()
    raster = rasterize(parts)
    labels = raster.labels
    pitch = raster.pitch
    metal = labels == METAL
    if not metal.any():
        report.violations.append("no metal")
        return report
    comp, n = ndimage.label(metal, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(comp.ravel())[1:]
    if n > 1 and sizes.max() < 0.995 * sizes.sum():
        report.violations.append(f"casting is {n} separate bodies")

    risers = [m for m, c in parts if c == PART_TYPE_RISER]
    if not risers:
        report.violations.append("no riser")
    riser_specs: List[_Riser] = []
    for k, mesh in enumerate(risers):
        lo, hi = mesh.bounds
        r = _Riser(x=float(0.5 * (lo[0] + hi[0])), y=float(0.5 * (lo[1] + hi[1])), z_base=float(lo[2] + RISER_SINK_IN),
                   diameter=float(hi[0] - lo[0]), limit=np.inf)
        riser_specs.append(r)
        # Sampled over the whole base: a cored bore under the riser is not metal.
        g = np.linspace(-0.45, 0.45, 11) * r.diameter
        gx, gy = np.meshgrid(g, g)
        disk = np.hypot(gx, gy) <= 0.45 * r.diameter + 1e-9
        pts = np.column_stack([r.x + gx[disk], r.y + gy[disk], np.full(int(disk.sum()), r.z_base - RISER_SINK_IN - 0.5 * pitch)])
        idx = np.clip(np.round((pts - raster.origin) / pitch).astype(int), 0, np.array(labels.shape) - 1)
        seated = np.mean(labels[idx[:, 0], idx[:, 1], idx[:, 2]] == METAL)
        if seated < MIN_SEATED_FRACTION:
            report.violations.append(f"riser {k} only {seated:.0%} seated on metal")
    for k, mesh in enumerate(m for m, c in parts if c == PART_TYPE_CHILL):
        lo, hi = mesh.bounds
        pad = np.array([pitch, pitch, pitch]) * 1.5
        i0 = np.clip(np.floor((lo - pad - raster.origin) / pitch).astype(int), 0, None)
        i1 = np.ceil((hi + pad - raster.origin) / pitch).astype(int) + 1
        if not np.any(metal[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]]):
            report.violations.append(f"chill {k} does not touch the casting")
        xy = 0.5 * (lo[:2] + hi[:2])
        if any(np.hypot(xy[0] - r.x, xy[1] - r.y) < 0.5 * r.diameter for r in riser_specs):
            report.violations.append(f"chill {k} sits under a riser")
    core = labels == CORE
    if core.any():
        ccomp, n_cores = ndimage.label(core, structure=np.ones((3, 3, 3)))
        sand_adjacent = np.zeros(labels.shape, dtype=bool)
        for axis in range(3):
            for step in (-1, 1):
                sand_adjacent |= _neighbour(labels, axis, step) == SAND
        core_radius = ndimage.distance_transform_edt(core) * pitch
        for c in range(1, n_cores + 1):
            this = ccomp == c
            if not np.any(this & sand_adjacent):
                report.violations.append(f"core {c} is fully enclosed (no print)")
            if 2.0 * core_radius[this].max() < MIN_CORE_DIAMETER_IN:
                report.violations.append(f"core {c} thinner than {MIN_CORE_DIAMETER_IN} in")
        report.metrics["core_count"] = float(n_cores)

    if riser_specs:
        r_metal = (ndimage.distance_transform_edt(metal) * pitch)[metal]
        zones, moduli, thickness, fed = _feeding(raster, riser_specs, _chill_extension(raster, 2.0 * float(r_metal.max())))
        for k, r in enumerate(riser_specs):
            ratio = RISER_SLEEVE_MEF * RISER_MODULUS_PER_DIAMETER * r.diameter / max(moduli[k], 1e-9)
            report.metrics[f"riser_{k}_modulus_ratio"] = float(ratio)
            if ratio < RISER_MODULUS_FACTOR - 0.05:
                report.violations.append(f"riser {k} modulus {ratio:.2f}x its zone (< {RISER_MODULUS_FACTOR})")
        fed_fraction = float(fed.mean())
        report.metrics["fed_fraction"] = fed_fraction
        if fed_fraction < FED_FRACTION_REQUIRED - 0.01:
            report.violations.append(f"only {fed_fraction:.1%} of the metal within feeding distance")
    report.metrics["riser_count"] = float(len(risers))
    return report
