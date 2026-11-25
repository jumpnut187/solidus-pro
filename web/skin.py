from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TypedDict
import hashlib
import json
import logging
import os
import tempfile
import threading

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from web.fields import Lattice

logger = logging.getLogger(__name__)

SKIN_RANGE_VOXELS = 2.0

_BAND_CELLS = 2

_AMBIGUOUS_VOXELS = float(np.sqrt(3.0) / 2.0)

MAX_SKIN_BAND_CELLS = 2_000_000

SKIN_VERSION = 1

@dataclass
class SkinGrid:

    values: np.ndarray
    range_voxels: float
    signature: str

    def tobytes(self) -> bytes:
        return np.ascontiguousarray(self.values, dtype=np.uint8).tobytes()

def quantize(distance_voxels: np.ndarray, range_voxels: float = SKIN_RANGE_VOXELS) -> np.ndarray:
    scaled = np.clip(distance_voxels / range_voxels, -1.0, 1.0)
    return np.rint((scaled + 1.0) * 0.5 * 255.0).astype(np.uint8)

def dequantize(values: np.ndarray, range_voxels: float = SKIN_RANGE_VOXELS) -> np.ndarray:
    return (values.astype(np.float32) / 255.0 * 2.0 - 1.0) * range_voxels

def build_skin_grid(lattice: Lattice, mask: np.ndarray, meshes: Sequence[Any]) -> Optional[np.ndarray]:
    meshes = [mesh for mesh in meshes if mesh is not None and len(getattr(mesh, "faces", ())) > 0]
    if not meshes:
        return None

    nx, ny, nz = lattice.dims
    spacing = float(lattice.spacing)
    metal = np.asarray(mask, dtype=np.uint8).reshape((nz, ny, nx)) != 0

    distance = np.where(metal, -SKIN_RANGE_VOXELS, SKIN_RANGE_VOXELS).astype(np.float32)

    structure = ndimage.generate_binary_structure(3, 3)
    grown = ndimage.binary_dilation(metal, structure=structure, iterations=_BAND_CELLS)
    shrunk = ndimage.binary_erosion(metal, structure=structure, iterations=_BAND_CELLS)
    band = grown & ~shrunk
    band_count = int(np.count_nonzero(band))
    if band_count == 0:
        return quantize(distance.reshape(-1))
    if band_count > MAX_SKIN_BAND_CELLS:
        logger.info("Skin grid skipped: %d band cells exceeds %d", band_count, MAX_SKIN_BAND_CELLS)
        return None

    bz, by, bx = np.nonzero(band)
    origin = np.asarray(lattice.origin, dtype=np.float64)
    points = origin + np.stack([bx, by, bz], axis=1).astype(np.float64) * spacing

    inside = metal[bz, by, bx].copy()

    unsigned = np.full(band_count, np.inf, dtype=np.float64)
    facing_inside = inside.copy()
    radius = SKIN_RANGE_VOXELS * spacing
    for mesh in meshes:
        closest, part_distance, triangle = _closest_point_within(mesh, points, radius)
        nearer = part_distance < unsigned
        if not np.any(nearer):
            continue
        unsigned[nearer] = part_distance[nearer]
        normals = np.asarray(mesh.face_normals)[triangle[nearer]]
        facing_inside[nearer] = np.einsum("ij,ij->i", points[nearer] - closest[nearer], normals) < 0.0
    unsigned_voxels = unsigned / spacing

    doubtful = (unsigned_voxels <= _AMBIGUOUS_VOXELS) | (facing_inside != inside)
    if np.any(doubtful):
        resolved = _contains_any(meshes, points[doubtful])
        if resolved is not None:
            inside[doubtful] = resolved

    signed = np.where(inside, -unsigned_voxels, unsigned_voxels)
    distance[bz, by, bx] = np.clip(signed, -SKIN_RANGE_VOXELS, SKIN_RANGE_VOXELS)
    return quantize(distance.reshape(-1))

_PAIR_BUDGET = 3_000_000

_CONTAINS_CHUNK = 20_000

def _closest_point_within(
    mesh: Any, points: np.ndarray, radius: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import trimesh

    n_points = len(points)
    best_distance = np.full(n_points, np.inf, dtype=np.float64)
    best_triangle = np.full(n_points, -1, dtype=np.int64)
    best_closest = np.zeros((n_points, 3), dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if n_points == 0 or len(faces) == 0:
        return best_closest, best_distance, best_triangle

    lo = vertices.min(axis=0) - radius
    hi = vertices.max(axis=0) + radius
    selected = np.flatnonzero(np.all((points >= lo) & (points <= hi), axis=1))
    if selected.size == 0:
        return best_closest, best_distance, best_triangle
    local_points = points[selected]

    triangles = vertices[faces]
    face_index = np.arange(len(faces), dtype=np.int64)
    centroids = triangles.mean(axis=1)
    reach = np.linalg.norm(triangles - centroids[:, None, :], axis=2).max(axis=1)
    big = reach > radius
    if np.any(big):
        sub_vertices, sub_faces, sub_index = trimesh.remesh.subdivide_to_size(
            vertices, faces[big], max_edge=radius * 1.5, max_iter=12, return_index=True
        )
        sub_triangles = np.asarray(sub_vertices, dtype=np.float64)[np.asarray(sub_faces, dtype=np.int64)]
        triangles = np.vstack([triangles[~big], sub_triangles])
        face_index = np.concatenate([face_index[~big], np.flatnonzero(big)[np.asarray(sub_index, dtype=np.int64)]])
        centroids = triangles.mean(axis=1)
        reach = np.linalg.norm(triangles - centroids[:, None, :], axis=2).max(axis=1)
    query_radius = reach + radius

    tree = cKDTree(local_points)
    counts = np.asarray(
        tree.query_ball_point(centroids, query_radius, workers=-1, return_length=True), dtype=np.int64
    )
    total_pairs = int(counts.sum())
    if total_pairs == 0:
        return best_closest, best_distance, best_triangle

    cumulative = np.cumsum(counts)
    boundaries = [0]
    while boundaries[-1] < len(counts):
        start = boundaries[-1]
        base = cumulative[start - 1] if start > 0 else 0
        stop = int(np.searchsorted(cumulative, base + _PAIR_BUDGET, side="right"))
        boundaries.append(max(stop, start + 1))

    local_best_distance = np.full(len(local_points), np.inf, dtype=np.float64)
    local_best_triangle = np.full(len(local_points), -1, dtype=np.int64)
    local_best_closest = np.zeros((len(local_points), 3), dtype=np.float64)

    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        chunk_counts = counts[start:stop]
        if chunk_counts.sum() == 0:
            continue
        hits = tree.query_ball_point(
            centroids[start:stop], query_radius[start:stop], workers=-1, return_sorted=False
        )
        point_idx = np.concatenate([np.asarray(h, dtype=np.int64) for h in hits if len(h)])
        tri_idx = np.repeat(np.arange(start, stop, dtype=np.int64), chunk_counts)

        closest = trimesh.triangles.closest_point(triangles[tri_idx], local_points[point_idx])
        distance = np.linalg.norm(closest - local_points[point_idx], axis=1)

        within = distance <= radius
        if not np.any(within):
            continue
        point_idx, tri_idx, closest, distance = (
            point_idx[within], tri_idx[within], closest[within], distance[within]
        )
        order = np.lexsort((distance, point_idx))
        point_sorted = point_idx[order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = point_sorted[1:] != point_sorted[:-1]
        winners = order[first]
        cand_points = point_idx[winners]
        better = distance[winners] < local_best_distance[cand_points]
        if not np.any(better):
            continue
        winners = winners[better]
        cand_points = cand_points[better]
        local_best_distance[cand_points] = distance[winners]
        local_best_triangle[cand_points] = face_index[tri_idx[winners]]
        local_best_closest[cand_points] = closest[winners]

    best_distance[selected] = local_best_distance
    best_triangle[selected] = local_best_triangle
    best_closest[selected] = local_best_closest
    return best_closest, best_distance, best_triangle

def _contains_any(meshes: Sequence[Any], points: np.ndarray) -> Optional[np.ndarray]:
    inside = np.zeros(len(points), dtype=bool)
    for mesh in meshes:
        for start in range(0, len(points), _CONTAINS_CHUNK):
            stop = start + _CONTAINS_CHUNK
            try:
                result = np.asarray(mesh.contains(points[start:stop]), dtype=bool)
            except Exception as exc:
                logger.info("Skin grid: inside test failed (%s); keeping the mask's sign", exc)
                return None
            if result.shape != inside[start:stop].shape:
                return None
            inside[start:stop] |= result
    return inside

def parts_signature(lattice: Lattice, parts: Iterable[Dict[str, Any]]) -> Optional[str]:
    entries: List[Any] = []
    for part in parts:
        filepath = str(part.get("filepath") or "")
        if not filepath:
            continue
        try:
            stat = os.stat(filepath)
        except OSError:
            continue
        entries.append(
            [
                filepath,
                int(stat.st_size),
                int(stat.st_mtime_ns),
                [float(part.get(f"position_{axis}", 0.0) or 0.0) for axis in "xyz"],
                [float(part.get(f"rotation_{axis}", 0.0) or 0.0) for axis in "xyz"],
                float(part.get("scale", 1.0) or 1.0),
            ]
        )
    if not entries:
        return None
    entries.sort(key=lambda entry: entry[0])
    payload = {
        "version": SKIN_VERSION,
        "range": SKIN_RANGE_VOXELS,
        "dims": [int(v) for v in lattice.dims],
        "origin": [float(v) for v in lattice.origin],
        "spacing": float(lattice.spacing),
        "parts": entries,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

def part_meshes(parts: Sequence[Dict[str, Any]]) -> List[Any]:
    from web.payloads import build_part_manager

    present = [part for part in parts if part.get("filepath") and os.path.exists(str(part["filepath"]))]
    if not present:
        return []
    try:
        manager = build_part_manager(list(present))
    except Exception as exc:
        logger.info("Skin grid: parts could not be loaded (%s)", exc)
        return []
    meshes: List[Any] = []
    for part in manager.get_all_parts():
        try:
            meshes.append(part.get_transformed_mesh())
        except Exception as exc:
            logger.info("Skin grid: part %s could not be transformed (%s)", part.name, exc)
    return meshes

def skin_path(artifact_path: Path | str) -> Path:
    path = Path(artifact_path)
    return path.with_name(path.stem + ".skin.npz")

class _SkinPayload(TypedDict):
    version: np.ndarray
    range: np.ndarray
    signature: np.ndarray
    values: np.ndarray

def _save(path: Path, grid: SkinGrid) -> None:
    payload: _SkinPayload = {
        "version": np.asarray([SKIN_VERSION], dtype=np.int32),
        "range": np.asarray([grid.range_voxels], dtype=np.float64),
        "signature": np.frombuffer(grid.signature.encode("utf-8"), dtype=np.uint8),
        "values": np.asarray(grid.values, dtype=np.uint8),
    }
    handle, temp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise

def _load(path: Path, signature: str) -> Optional[SkinGrid]:
    try:
        with np.load(path, allow_pickle=False) as data:
            if int(data["version"][0]) != SKIN_VERSION:
                return None
            stored = bytes(data["signature"]).decode("utf-8")
            if stored != signature:
                return None
            return SkinGrid(
                values=np.asarray(data["values"], dtype=np.uint8),
                range_voxels=float(data["range"][0]),
                signature=stored,
            )
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return None

_MEMORY_MAX = 8
_memory: "OrderedDict[Tuple[str, str], SkinGrid]" = OrderedDict()
_lock = threading.Lock()

def clear_cache() -> None:
    with _lock:
        _memory.clear()

def load_skin_grid(
    project_id: str,
    lattice: Lattice,
    mask: np.ndarray,
    parts: Sequence[Dict[str, Any]],
    artifact_path: Optional[Path | str],
) -> Optional[SkinGrid]:
    signature = parts_signature(lattice, parts)
    if signature is None:
        return None

    key = (project_id, signature)
    with _lock:
        cached = _memory.get(key)
        if cached is not None:
            _memory.move_to_end(key)
            return cached

    disk = skin_path(artifact_path) if artifact_path is not None else None
    grid: Optional[SkinGrid] = None
    if disk is not None and disk.exists():
        grid = _load(disk, signature)

    if grid is None:
        values = build_skin_grid(lattice, mask, part_meshes(parts))
        if values is None:
            return None
        grid = SkinGrid(values=values, range_voxels=SKIN_RANGE_VOXELS, signature=signature)
        if disk is not None:
            try:
                _save(disk, grid)
            except OSError:
                pass

    with _lock:
        _memory[key] = grid
        _memory.move_to_end(key)
        while len(_memory) > _MEMORY_MAX:
            _memory.popitem(last=False)
    return grid
