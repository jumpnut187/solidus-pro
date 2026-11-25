from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import trimesh

Vec3 = Tuple[float, float, float]

_AXES = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}

def _clean_angle(value: float) -> float:
    angle = (float(value) + 180.0) % 360.0 - 180.0
    if angle <= -180.0 + 1e-9:
        angle = 180.0
    rounded = round(angle, 6)
    return 0.0 if rounded == 0.0 else rounded

def euler_to_matrix(rotation_deg: Sequence[float]) -> np.ndarray:
    rx, ry, rz = (np.radians(float(a)) for a in rotation_deg)
    return trimesh.transformations.euler_matrix(rx, ry, rz, "sxyz")[:3, :3]

def matrix_to_euler(matrix: np.ndarray) -> Vec3:
    m = np.eye(4)
    m[:3, :3] = np.asarray(matrix, dtype=float)[:3, :3]
    rx, ry, rz = trimesh.transformations.euler_from_matrix(m, "sxyz")
    return (_clean_angle(np.degrees(rx)), _clean_angle(np.degrees(ry)), _clean_angle(np.degrees(rz)))

def axis_rotation_matrix(axis: str | Sequence[float], angle_deg: float) -> np.ndarray:
    vec = _AXES[axis.lower()] if isinstance(axis, str) else np.asarray(axis, dtype=float)
    return trimesh.transformations.rotation_matrix(np.radians(float(angle_deg)), vec)[:3, :3]

def pivot_position(position: Sequence[float], delta: np.ndarray, pivot: Sequence[float]) -> Vec3:
    p = np.asarray(position, dtype=float)
    c = np.asarray(pivot, dtype=float)
    new = np.asarray(delta, dtype=float) @ (p - c) + c
    return (float(new[0]), float(new[1]), float(new[2]))

def rotate_about_world_axis(
    rotation_deg: Sequence[float],
    position: Sequence[float],
    axis: str | Sequence[float],
    angle_deg: float,
    pivot: Sequence[float],
) -> Tuple[Vec3, Vec3]:
    delta = axis_rotation_matrix(axis, angle_deg)
    new_r = delta @ euler_to_matrix(rotation_deg)
    return matrix_to_euler(new_r), pivot_position(position, delta, pivot)

def align_vector_matrix(src: Sequence[float], dst: Sequence[float]) -> np.ndarray:
    a = np.asarray(src, dtype=float)
    b = np.asarray(dst, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    cross = np.cross(a, b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-9:
        if dot > 0:
            return np.eye(3)
        perp = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0.0, 1.0, 0.0])
        return axis_rotation_matrix(perp / np.linalg.norm(perp), 180.0)
    angle = np.degrees(np.arctan2(np.linalg.norm(cross), dot))
    return axis_rotation_matrix(cross / np.linalg.norm(cross), angle)

def face_to_direction(
    rotation_deg: Sequence[float],
    position: Sequence[float],
    world_normal: Sequence[float],
    pivot: Sequence[float],
    target: Sequence[float] = (0.0, 0.0, -1.0),
) -> Tuple[Vec3, Vec3]:
    delta = align_vector_matrix(world_normal, target)
    new_r = delta @ euler_to_matrix(rotation_deg)
    return matrix_to_euler(new_r), pivot_position(position, delta, pivot)

def world_bounds(
    mesh: trimesh.Trimesh,
    rotation_deg: Sequence[float],
    position: Sequence[float],
    scale: float = 1.0,
) -> np.ndarray:
    v = np.asarray(mesh.vertices, dtype=float) * float(scale)
    v = _apply_rotation(v, euler_to_matrix(rotation_deg)) + np.asarray(position, dtype=float)
    return np.vstack([v.min(axis=0), v.max(axis=0)])

def _apply_rotation(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return points @ np.asarray(matrix, dtype=float).T

def drop_to_floor_dz(bounds: np.ndarray, floor_z: float = 0.0) -> float:
    return float(floor_z - np.asarray(bounds)[0, 2])

def snap_to_parting_dz(bounds: np.ndarray, parting_z: float, side: str = "above") -> float:
    b = np.asarray(bounds, dtype=float)
    if side == "below":
        return float(parting_z - b[1, 2])
    return float(parting_z - b[0, 2])

def section_profile(
    mesh: trimesh.Trimesh,
    rotation_deg: Sequence[float],
    position: Sequence[float],
    scale: float = 1.0,
    samples: int = 64,
) -> Dict[str, List[float]]:
    placed = mesh.copy()
    m = np.eye(4)
    m[:3, :3] = euler_to_matrix(rotation_deg) * float(scale)
    m[:3, 3] = np.asarray(position, dtype=float)
    placed.apply_transform(m)
    zmin, zmax = float(placed.bounds[0, 2]), float(placed.bounds[1, 2])
    height = zmax - zmin
    if height <= 0:
        return {"z": [zmin], "area": [0.0]}
    n = max(int(samples), 3)
    levels = zmin + height * (np.arange(n) + 0.5) / n
    areas = [_section_area(placed, float(z)) for z in levels]
    return {"z": [float(z) for z in levels], "area": areas}

def _section_area(mesh: trimesh.Trimesh, z: float) -> float:
    try:
        lines, faces = trimesh.intersections.mesh_plane(
            mesh, plane_normal=[0.0, 0.0, 1.0], plane_origin=[0.0, 0.0, z], return_faces=True
        )
    except Exception:
        return 0.0
    if len(lines) == 0:
        return 0.0
    a = lines[:, 0, :2]
    b = lines[:, 1, :2]
    normals = mesh.face_normals[faces][:, :2]
    tangent = np.column_stack([-normals[:, 1], normals[:, 0]])
    flip = np.einsum("ij,ij->i", b - a, tangent) < 0
    a2 = np.where(flip[:, None], b, a)
    b2 = np.where(flip[:, None], a, b)
    area = 0.5 * float(np.sum(a2[:, 0] * b2[:, 1] - b2[:, 0] * a2[:, 1]))
    return abs(area)

def suggest_parting_z(
    mesh: trimesh.Trimesh,
    rotation_deg: Sequence[float],
    position: Sequence[float],
    scale: float = 1.0,
    samples: int = 64,
) -> Dict[str, float]:
    profile = section_profile(mesh, rotation_deg, position, scale, samples)
    bounds = world_bounds(mesh, rotation_deg, position, scale)
    areas = np.asarray(profile["area"])
    zs = np.asarray(profile["z"])
    if areas.size and float(areas.max()) > 0:
        widest = int(np.argmax(areas))
        step = float(zs[1] - zs[0]) if zs.size > 1 else 0.0
        hi = widest
        while hi + 1 < areas.size and areas[hi + 1] >= areas[widest] * 0.98:
            hi += 1
        max_z = float(zs[hi] + step / 2.0)
        vz = _apply_rotation(
            np.asarray(mesh.vertices, dtype=float) * float(scale), euler_to_matrix(rotation_deg)
        )[:, 2] + float(np.asarray(position, dtype=float)[2])
        near = vz[np.abs(vz - max_z) <= step]
        if near.size:
            max_z = float(near[np.argmin(np.abs(near - max_z))])
    else:
        max_z = float((bounds[0, 2] + bounds[1, 2]) / 2.0)
    return {
        "max_section": round(max_z, 4),
        "bottom": round(float(bounds[0, 2]), 4),
        "top": round(float(bounds[1, 2]), 4),
        "center": round(float((bounds[0, 2] + bounds[1, 2]) / 2.0), 4),
    }

def cope_drag_split(
    mesh: trimesh.Trimesh,
    rotation_deg: Sequence[float],
    position: Sequence[float],
    parting_z: float,
    scale: float = 1.0,
) -> Dict[str, float]:
    bounds = world_bounds(mesh, rotation_deg, position, scale)
    cope_depth = max(0.0, float(bounds[1, 2] - parting_z))
    drag_depth = max(0.0, float(parting_z - bounds[0, 2]))
    profile = section_profile(mesh, rotation_deg, position, scale, 96)
    zs = np.asarray(profile["z"])
    areas = np.asarray(profile["area"])
    total = float(areas.sum())
    cope_frac = float(areas[zs >= parting_z].sum() / total) if total > 0 else 0.0
    return {
        "cope_depth": round(cope_depth, 4),
        "drag_depth": round(drag_depth, 4),
        "cope_fraction": round(cope_frac, 4),
        "drag_fraction": round(1.0 - cope_frac, 4) if total > 0 else 0.0,
    }

def snap_angle(angle_deg: float, step: float = 15.0) -> float:
    if step <= 0:
        return float(angle_deg)
    return _clean_angle(round(float(angle_deg) / step) * step)

def as_vec3(values: Iterable[float]) -> Vec3:
    a, b, c = (float(v) for v in values)
    return (a, b, c)
