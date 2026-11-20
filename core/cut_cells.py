from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from numba import njit, prange
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

DEFAULT_SUBSAMPLES = 64
DEFAULT_AREA_SAMPLE_SPACING = 0.25
INTERFACE_PROBE_OFFSET = 0.5
DEFAULT_PHI_MIN = 0.25
WINDING_NUMBER_MAX_PAIRS = 10_000_000_000

_POINT_CHUNK = 2_000_000
_KEY_OFFSET = 1 << 20
_KEY_BASE = 1 << 21

_FACE_SLOTS = 6

@dataclass
class CutCellGeometry:

    centers: np.ndarray
    metal_fractions: np.ndarray
    surface_areas: np.ndarray
    sampled: np.ndarray
    part_counts: np.ndarray
    pitch: float
    n_input: int = 0
    n_duplicates: int = 0
    n_added: int = 0
    n_dropped: int = 0
    timings: Dict[str, float] = field(default_factory=dict)

@dataclass
class CutCellMerge:

    capacity_weights: np.ndarray
    boundary_areas: np.ndarray
    slaves: np.ndarray
    masters: np.ndarray
    n_unmerged_small: int = 0

def r3_offsets(n: int) -> np.ndarray:
    g = 1.22074408460575947536
    alpha = np.array([1.0 / g, 1.0 / g**2, 1.0 / g**3])
    j = np.arange(1, n + 1, dtype=np.float64)[:, None]
    return np.mod(0.5 + j * alpha[None, :], 1.0)

def _r2_offsets(j: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    g = 1.32471795724474602596
    u = np.mod(0.5 + j / g, 1.0)
    v = np.mod(0.5 + j / (g * g), 1.0)
    return u, v

def contains(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    if mesh.is_watertight:
        return np.asarray(mesh.contains(points), dtype=bool)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    chunk = max(1, WINDING_NUMBER_MAX_PAIRS // max(1, len(mesh.faces)))
    inside = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), chunk):
        stop = min(len(points), start + chunk)
        inside[start:stop] = np.asarray(winding_number_contains(mesh, points[start:stop]), dtype=bool)
    return inside

def winding_number_contains(
    mesh: trimesh.Trimesh, points: np.ndarray, *, near_surface_tol: float = 0.0
) -> np.ndarray:
    from core.voxel_mesh import _near_surface_mask, _robust_contains_check_impl

    query = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
    if len(query) == 0 or len(mesh.faces) == 0:
        return np.zeros(len(query), dtype=bool)
    pairs = len(query) * len(mesh.faces)
    if pairs > WINDING_NUMBER_MAX_PAIRS:
        logger.warning(
            "winding-number containment skipped: %d points x %d faces exceeds %d pairs; "
            "using the ray-test containment check, which is unreliable on open surfaces",
            len(query),
            len(mesh.faces),
            WINDING_NUMBER_MAX_PAIRS,
        )
        return np.asarray(_robust_contains_check_impl(mesh, query, near_surface_tol), dtype=bool)
    triangles = np.ascontiguousarray(mesh.triangles, dtype=np.float64)
    inside = _winding_numbers(triangles, query) > 0.5
    if near_surface_tol > 0.0:
        inside |= np.asarray(_near_surface_mask(mesh, query, near_surface_tol), dtype=bool)
    return inside

@njit(parallel=True, cache=True)
def _winding_numbers(triangles: np.ndarray, points: np.ndarray) -> np.ndarray:
    n_points = points.shape[0]
    n_faces = triangles.shape[0]
    result = np.empty(n_points, dtype=np.float64)
    for i in prange(n_points):
        px = points[i, 0]
        py = points[i, 1]
        pz = points[i, 2]
        total = 0.0
        for f in range(n_faces):
            ax = triangles[f, 0, 0] - px
            ay = triangles[f, 0, 1] - py
            az = triangles[f, 0, 2] - pz
            bx = triangles[f, 1, 0] - px
            by = triangles[f, 1, 1] - py
            bz = triangles[f, 1, 2] - pz
            cx = triangles[f, 2, 0] - px
            cy = triangles[f, 2, 1] - py
            cz = triangles[f, 2, 2] - pz
            la = math.sqrt(ax * ax + ay * ay + az * az)
            lb = math.sqrt(bx * bx + by * by + bz * bz)
            lc = math.sqrt(cx * cx + cy * cy + cz * cz)
            det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
            den = (
                la * lb * lc
                + (ax * bx + ay * by + az * bz) * lc
                + (bx * cx + by * cy + bz * cz) * la
                + (cx * ax + cy * ay + cz * az) * lb
            )
            total += 2.0 * math.atan2(det, den)
        result[i] = total / (4.0 * math.pi)
    return result

def _keys(ijk: np.ndarray) -> np.ndarray:
    shifted = ijk.astype(np.int64) + _KEY_OFFSET
    return (shifted[:, 0] * _KEY_BASE + shifted[:, 1]) * _KEY_BASE + shifted[:, 2]

def _ijk_from_keys(keys: np.ndarray) -> np.ndarray:
    k = keys % _KEY_BASE
    rest = keys // _KEY_BASE
    j = rest % _KEY_BASE
    i = rest // _KEY_BASE
    return np.column_stack([i, j, k]) - _KEY_OFFSET

def _lookup(sorted_keys: np.ndarray, order: np.ndarray, query: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(sorted_keys, query)
    pos = np.minimum(pos, len(sorted_keys) - 1)
    found = sorted_keys[pos] == query
    out = np.full(len(query), -1, dtype=np.int64)
    out[found] = order[pos[found]]
    return out

def surface_samples(
    mesh: trimesh.Trimesh, spacing: float, face_chunk: int = 200_000
):
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    counts = np.maximum(1, np.ceil(areas / (spacing * spacing))).astype(np.int64)
    n_faces = len(triangles)
    start = 0
    while start < n_faces:
        stop = min(n_faces, start + face_chunk)
        csum = np.cumsum(counts[start:stop])
        if csum[-1] > _POINT_CHUNK:
            stop = start + max(1, int(np.searchsorted(csum, _POINT_CHUNK)))
        c = counts[start:stop]
        face = np.repeat(np.arange(start, stop), c)
        first = np.repeat(np.cumsum(c) - c, c)
        j = (np.arange(len(face)) - first + 1).astype(np.float64)
        single = np.repeat(c == 1, c)
        u, v = _r2_offsets(j)
        fold = (u + v) > 1.0
        u[fold] = 1.0 - u[fold]
        v[fold] = 1.0 - v[fold]
        u[single] = 1.0 / 3.0
        v[single] = 1.0 / 3.0
        tri = triangles[face]
        points = tri[:, 0] + u[:, None] * (tri[:, 1] - tri[:, 0]) + v[:, None] * (tri[:, 2] - tri[:, 0])
        weights = areas[face] / c[face - start]
        yield points, normals[face], weights
        start = stop

def _in_box(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    return np.all((points >= box[0]) & (points <= box[1]), axis=1)

def split_into_bodies(
    part_meshes: Sequence[trimesh.Trimesh],
) -> Tuple[List[trimesh.Trimesh], np.ndarray]:
    units: List[trimesh.Trimesh] = []
    owner: List[int] = []
    for p, mesh in enumerate(part_meshes):
        bodies = [mesh]
        split = mesh.split(only_watertight=False)
        if (len(split) > 1 or not mesh.is_watertight) and len(split) > 0 and all(
            b.is_watertight and b.volume > 0.0 for b in split
        ):
            bodies = list(split)
        units.extend(bodies)
        owner.extend([p] * len(bodies))
    return units, np.asarray(owner, dtype=np.int64)

def exterior_surface_area_by_cell(
    part_meshes: Sequence[trimesh.Trimesh],
    origin: np.ndarray,
    pitch: float,
    *,
    spacing_fraction: float = DEFAULT_AREA_SAMPLE_SPACING,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spacing = spacing_fraction * pitch
    probe = INTERFACE_PROBE_OFFSET * pitch
    boxes = [np.asarray(m.bounds, dtype=np.float64) for m in part_meshes]
    key_parts: List[np.ndarray] = []
    area_parts: List[np.ndarray] = []
    xyz_parts: List[np.ndarray] = []
    interface_parts: List[np.ndarray] = []
    for p, mesh in enumerate(part_meshes):
        for points, normals, weights in surface_samples(mesh, spacing):
            exterior = np.ones(len(points), dtype=bool)
            probes = points + probe * normals
            for q, other in enumerate(part_meshes):
                if q == p:
                    continue
                test = exterior & _in_box(probes, boxes[q])
                if np.any(test):
                    idx = np.flatnonzero(test)
                    exterior[idx[contains(other, probes[idx])]] = False
            if not np.all(exterior):
                interface_parts.append(
                    np.unique(_keys(np.floor((points[~exterior] - origin) / pitch + 0.5).astype(np.int64)))
                )
            points = points[exterior]
            weights = weights[exterior]
            if len(points) == 0:
                continue
            ijk = np.floor((points - origin) / pitch + 0.5).astype(np.int64)
            keys = _keys(ijk)
            uniq, inverse = np.unique(keys, return_inverse=True)
            key_parts.append(uniq)
            area_parts.append(np.bincount(inverse, weights=weights, minlength=len(uniq)))
            xyz_parts.append(
                np.column_stack(
                    [np.bincount(inverse, weights=weights * points[:, d], minlength=len(uniq)) for d in range(3)]
                )
            )
    interface_keys = (
        np.unique(np.concatenate(interface_parts)) if interface_parts else np.zeros(0, dtype=np.int64)
    )
    if not key_parts:
        return np.zeros(0, np.int64), np.zeros(0), np.zeros((0, 3)), interface_keys
    keys = np.concatenate(key_parts)
    uniq, inverse = np.unique(keys, return_inverse=True)
    area = np.bincount(inverse, weights=np.concatenate(area_parts), minlength=len(uniq))
    wxyz = np.concatenate(xyz_parts)
    centroid = np.column_stack(
        [np.bincount(inverse, weights=wxyz[:, d], minlength=len(uniq)) for d in range(3)]
    ) / np.maximum(area, 1e-300)[:, None]
    return uniq, area, centroid, interface_keys

def _face_neighbour_keys(ijk: np.ndarray) -> List[np.ndarray]:
    out = []
    for axis in range(3):
        for sign in (1, -1):
            nb = ijk.copy()
            nb[:, axis] += sign
            out.append(_keys(nb))
    return out

def _union_contains(units: Sequence[trimesh.Trimesh], points: np.ndarray) -> np.ndarray:
    inside = np.zeros(len(points), dtype=bool)
    for mesh in units:
        cand = np.flatnonzero(~inside & _in_box(points, np.asarray(mesh.bounds, dtype=np.float64)))
        if len(cand):
            inside[cand[contains(mesh, points[cand])]] = True
    return inside

def _flood_missing_interior(
    keys: np.ndarray, origin: np.ndarray, pitch: float, units: Sequence[trimesh.Trimesh]
) -> np.ndarray:
    present = np.unique(keys)
    tested = present
    added: List[np.ndarray] = []
    frontier_from = _ijk_from_keys(present)
    while len(frontier_from):
        cand = np.unique(np.concatenate(_face_neighbour_keys(frontier_from)))
        cand = cand[~np.isin(cand, tested, assume_unique=True)]
        if len(cand) == 0:
            break
        tested = np.union1d(tested, cand)
        inside = _union_contains(units, origin + _ijk_from_keys(cand) * pitch)
        new = cand[inside]
        if len(new) == 0:
            break
        added.append(new)
        frontier_from = _ijk_from_keys(new)
    return np.concatenate(added) if added else np.zeros(0, dtype=np.int64)

def sample_metal_fractions(
    centers: np.ndarray,
    pitch: float,
    part_meshes: Sequence[trimesh.Trimesh],
    *,
    subsamples: int = DEFAULT_SUBSAMPLES,
    cell_chunk: Optional[int] = None,
    unit_owner: Optional[np.ndarray] = None,
    n_owners: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    n_cells = len(centers)
    if unit_owner is None:
        unit_owner = np.arange(len(part_meshes), dtype=np.int64)
    if n_owners is None:
        n_owners = int(unit_owner.max()) + 1 if len(unit_owner) else 0
    offsets = (r3_offsets(subsamples) - 0.5) * pitch
    phi = np.zeros(n_cells, dtype=np.float64)
    counts = np.zeros((n_cells, n_owners), dtype=np.uint16)
    if n_cells == 0:
        return phi, counts
    boxes = [np.asarray(m.bounds, dtype=np.float64) for m in part_meshes]
    chunk = cell_chunk or max(1, _POINT_CHUNK // subsamples)
    for start in range(0, n_cells, chunk):
        stop = min(n_cells, start + chunk)
        pts = (centers[start:stop, None, :] + offsets[None, :, :]).reshape(-1, 3)
        union = np.zeros(len(pts), dtype=bool)
        owner_inside = np.zeros((n_owners, len(pts)), dtype=bool)
        for u, mesh in enumerate(part_meshes):
            cand = np.flatnonzero(_in_box(pts, boxes[u]))
            if len(cand) == 0:
                continue
            owner_inside[unit_owner[u], cand[contains(mesh, pts[cand])]] = True
        for p in range(n_owners):
            union |= owner_inside[p]
            counts[start:stop, p] = owner_inside[p].reshape(stop - start, subsamples).sum(axis=1)
        phi[start:stop] = union.reshape(stop - start, subsamples).mean(axis=1)
    return phi, counts

def compute_cut_cell_geometry(
    centers: np.ndarray,
    pitch: float,
    part_meshes: Sequence[trimesh.Trimesh],
    *,
    subsamples: int = DEFAULT_SUBSAMPLES,
    area_spacing_fraction: float = DEFAULT_AREA_SAMPLE_SPACING,
) -> CutCellGeometry:
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    n_input = len(centers)
    present = [i for i, m in enumerate(part_meshes) if m is not None and len(m.faces) > 0]
    units, unit_owner = split_into_bodies([part_meshes[i] for i in present])
    origin = centers[0].copy() if n_input else np.zeros(3)

    surf_keys, surf_area, surf_centroid, interface_keys = exterior_surface_area_by_cell(
        units, origin, pitch, spacing_fraction=area_spacing_fraction
    )
    timings["area_s"] = time.perf_counter() - t0

    ijk = np.rint((centers - origin) / pitch).astype(np.int64)
    in_keys = _keys(ijk)
    _, first = np.unique(in_keys, return_index=True)
    n_duplicates = n_input - len(first)
    if n_duplicates:
        first = np.sort(first)
        ijk = ijk[first]
        in_keys = in_keys[first]
    n_unique = len(in_keys)
    added_keys = np.setdiff1d(surf_keys, in_keys, assume_unique=False)
    added_keys = np.concatenate(
        [added_keys, _flood_missing_interior(np.concatenate([in_keys, added_keys]), origin, pitch, units)]
    )
    all_keys = np.concatenate([in_keys, added_keys])
    all_ijk = np.concatenate([ijk, _ijk_from_keys(added_keys)])
    order = np.argsort(all_keys, kind="stable")
    sorted_keys = all_keys[order]
    timings["add_s"] = time.perf_counter() - t0 - timings["area_s"]

    boundary = np.zeros(len(all_keys), dtype=bool)
    boundary[_lookup(sorted_keys, order, surf_keys)] = True
    interface_hit = _lookup(sorted_keys, order, interface_keys)
    boundary[interface_hit[interface_hit >= 0]] = True
    for nb in _face_neighbour_keys(all_ijk):
        boundary |= _lookup(sorted_keys, order, nb) < 0
    all_centers = origin + all_ijk * pitch
    t1 = time.perf_counter()

    sampled_idx = np.flatnonzero(boundary)
    phi_sampled, counts_sampled = sample_metal_fractions(
        all_centers[sampled_idx],
        pitch,
        units,
        subsamples=subsamples,
        unit_owner=unit_owner,
        n_owners=len(present),
    )
    timings["phi_s"] = time.perf_counter() - t1

    phi = np.ones(len(all_keys), dtype=np.float64)
    phi[sampled_idx] = phi_sampled
    counts = np.zeros((len(all_keys), len(part_meshes)), dtype=np.uint16)
    counts[np.ix_(sampled_idx, np.asarray(present, dtype=np.int64))] = counts_sampled
    keep = phi > 0.0
    kept_idx = np.flatnonzero(keep)

    kept_keys = all_keys[kept_idx]
    kept_order = np.argsort(kept_keys, kind="stable")
    area = np.zeros(len(kept_idx), dtype=np.float64)
    target = _lookup(kept_keys[kept_order], kept_order, surf_keys)
    hit = target >= 0
    np.add.at(area, target[hit], surf_area[hit])
    if np.any(~hit) and len(kept_idx):
        tree = cKDTree(all_centers[kept_idx])
        _, nearest = tree.query(surf_centroid[~hit])
        np.add.at(area, nearest, surf_area[~hit])

    geometry = CutCellGeometry(
        centers=all_centers[kept_idx].astype(np.float32),
        metal_fractions=phi[kept_idx].astype(np.float32),
        surface_areas=area.astype(np.float32),
        sampled=boundary[kept_idx],
        part_counts=counts[kept_idx],
        pitch=float(pitch),
        n_input=n_input,
        n_duplicates=n_duplicates,
        n_added=int(np.count_nonzero(keep[n_unique:])),
        n_dropped=int(np.count_nonzero(~keep[:n_unique])),
        timings=timings,
    )
    timings["total_s"] = time.perf_counter() - t0
    logger.info(
        "Cut cells: %d input, %d duplicates, %d added, %d dropped, %d sampled, %.1f s",
        n_input, n_duplicates, geometry.n_added, geometry.n_dropped, len(sampled_idx), timings["total_s"],
    )
    return geometry

def dominant_part_codes(part_counts: np.ndarray, part_codes: Sequence[int]) -> np.ndarray:
    codes = np.asarray(part_codes, dtype=np.int64)
    n = part_counts.shape[0]
    if n == 0 or len(codes) == 0:
        return np.full(n, -1, dtype=np.int64)
    by_code = np.argsort(codes, kind="stable")[::-1]
    counts = part_counts[:, by_code].astype(np.int64)
    best = np.argmax(counts, axis=1)
    out = codes[by_code][best]
    out[counts.max(axis=1) == 0] = -1
    return out

def merge_small_cut_cells(
    metal_fractions: np.ndarray,
    surface_areas: np.ndarray,
    neighbors: np.ndarray,
    part_types: Optional[np.ndarray],
    phi_min: float = DEFAULT_PHI_MIN,
) -> CutCellMerge:
    phi = np.asarray(metal_fractions, dtype=np.float64).reshape(-1)
    area = np.asarray(surface_areas, dtype=np.float64).reshape(-1)
    nbrs = np.asarray(neighbors)
    n = len(phi)
    codes = (
        np.zeros(n, dtype=np.int64) if part_types is None else np.asarray(part_types, dtype=np.int64)
    )
    small = np.flatnonzero(phi < phi_min)
    masters = np.full(len(small), -1, dtype=np.int64)
    for slots in (slice(0, _FACE_SLOTS), slice(_FACE_SLOTS, nbrs.shape[1])):
        todo = masters < 0
        if not np.any(todo):
            break
        rows = nbrs[small[todo], slots].astype(np.int64)
        valid = rows >= 0
        safe = np.maximum(rows, 0)
        cand_phi = np.where(
            valid & (phi[safe] >= phi_min) & (codes[safe] == codes[small[todo]][:, None]),
            phi[safe],
            -1.0,
        )
        if cand_phi.shape[1] == 0:
            continue
        best = np.argmax(cand_phi, axis=1)
        ok = cand_phi[np.arange(len(best)), best] > 0.0
        chosen = np.where(ok, safe[np.arange(len(best)), best], -1)
        masters[np.flatnonzero(todo)] = chosen
    merged = masters >= 0
    slaves = small[merged]
    masters = masters[merged]
    weights = phi.copy()
    areas = area.copy()
    np.add.at(weights, masters, phi[slaves])
    np.add.at(areas, masters, area[slaves])
    weights[slaves] = 0.0
    areas[slaves] = 0.0
    return CutCellMerge(
        capacity_weights=weights,
        boundary_areas=areas,
        slaves=slaves,
        masters=masters,
        n_unmerged_small=int(np.count_nonzero(~merged)),
    )
