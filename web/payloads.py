"""Babylon-oriented payload helpers for the web application."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, cast
import json
import os

import numpy as np
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from core.part_manager import PartManager, PartType, build_part_manager as _core_build_part_manager, infer_part_type_from_filename
from core.simulation_pipeline import PipelineResult
from core.stl_loader import STLLoader
from core.surface_extraction import build_surface_payload, normalize_surface_request


BASIC_SURFACE_MODES = {
    "shrinkage",
    "hotspot",
    "feed_modulus",
    "niyama",
    "niyama_raw",
    "combined_feed_risk",
    "fraction_liquid",
}
ADVANCED_SURFACE_MODES = {
    "hot_tear",
    "sand_burn_in",
    "feeding_quality",
    "thermal_stress",
    "distortion",
    "crack_susceptibility",
    "sdas",
    "predicted_uts",
}
SURFACE_MODES = BASIC_SURFACE_MODES | ADVANCED_SURFACE_MODES


def upload_stl_file(file: FileStorage, upload_folder: Path) -> Dict[str, Any]:
    if not file.filename:
        raise ValueError("No file selected")
    if not file.filename.lower().endswith(".stl"):
        raise ValueError("Only STL files are allowed")

    upload_folder.mkdir(parents=True, exist_ok=True)
    file_id = os.urandom(4).hex()
    filename = secure_filename(file.filename)
    filepath = upload_folder / f"{file_id}_{filename}"
    file.save(str(filepath))

    loader = STLLoader()
    if not loader.load_stl(str(filepath)):
        try:
            filepath.unlink()
        except FileNotFoundError:
            pass
        raise ValueError("Failed to load STL file")

    bounds = loader.get_bounds_array()
    vertices = np.asarray(loader.get_vertices(), dtype=np.float32)
    faces = np.asarray(loader.get_faces(), dtype=np.int32)
    part_type = infer_part_type_from_filename(filename).value

    return {
        "file_id": file_id,
        "filename": filename,
        "filepath": str(filepath),
        "part_type": part_type,
        "dimensions": loader.get_dimensions(),
        "mesh_data": {
            "vertices": vertices.reshape(-1).tolist(),
            "faces": faces.reshape(-1).tolist(),
            "bounds": _bounds_payload(bounds),
            "volume": float(loader.volume) if loader.volume else 0.0,
            "surface_area": float(loader.surface_area) if loader.surface_area else 0.0,
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
        },
    }


def build_part_manager(parts_data: list[Dict[str, Any]]) -> PartManager:
    """Build a :class:`PartManager` from serialized web payload data.

    Re-export of :func:`core.part_manager.build_part_manager` so existing
    web callers keep their import path. The canonical helper honours the
    ``visible`` and ``is_open_top`` per-part overrides that web payloads
    send.
    """
    return build_part_manager_from_core(parts_data, preserve_origin=True)


def build_part_manager_from_core(
    parts_data: list[Dict[str, Any]],
    *,
    preserve_origin: bool = True,
) -> PartManager:
    return _core_build_part_manager(
        parts_data,
        preserve_origin=preserve_origin,
        default_visible=True,
        default_is_open_top=False,
    )


def build_fork_artifact(result: PipelineResult) -> Dict[str, Any]:
    visualization = dict(result.visualization_data)
    engine = result.engine
    mesh = result.voxel_mesh

    centers = np.asarray(visualization.get("centers", []), dtype=np.float32)
    source_centers = np.asarray(mesh.voxel_centers, dtype=np.float32) if mesh.voxel_centers is not None else centers
    indices = _matching_indices(source_centers, centers)

    if mesh.voxel_part_types is not None:
        visualization["part_types"] = np.asarray(mesh.voxel_part_types, dtype=np.int32)[indices].tolist()

    field_specs = {
        "feed_modulus": getattr(engine, "local_modulus", None),
        "hot_tear": getattr(engine, "hot_tear_risk", None),
        "sand_burn_in": getattr(engine, "sand_burn_in_risk", None),
        "feeding_quality": _invert(getattr(engine, "feeding_quality", None)),
        "niyama_raw": getattr(engine, "niyama_values", None),
        "niyama_risk": getattr(engine, "niyama_risk", None),
        "combined_feed_risk": getattr(engine, "combined_feed_risk", None),
        "thermal_stress": _normalize(getattr(engine, "effective_stress", None)),
        "distortion": _normalize(getattr(engine, "distortion_magnitude", None)),
        "crack_susceptibility": getattr(engine, "crack_susceptibility", None),
        "sdas": _normalize(getattr(engine, "sdas", None)),
        "predicted_uts": _invert(_normalize(getattr(engine, "predicted_uts", None))),
    }
    raw_ranges: Dict[str, Dict[str, float]] = {}
    part_types = getattr(mesh, "voxel_part_types", None)
    for key, values in field_specs.items():
        if values is None:
            continue
        arr = np.asarray(values, dtype=np.float32)
        if len(arr) != len(source_centers):
            continue
        payload_arr = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
        # Feed modulus is only meaningful for cast metal; exclude chills.
        if key == "feed_modulus" and part_types is not None and len(part_types) == len(source_centers):
            chill_mask = np.asarray(part_types, dtype=np.int32) == 2
            if np.any(chill_mask):
                payload_arr = payload_arr.copy()
                payload_arr[chill_mask] = 0.0
        visualization[key] = payload_arr[indices].tolist()
        raw_ranges[key] = _range_payload(arr)

    summary = dict(result.result_data)
    niyama_metadata = getattr(engine, "niyama_metadata", None)
    if isinstance(niyama_metadata, dict):
        summary["niyama_metadata"] = dict(niyama_metadata)
    summary["available_modes"] = sorted(
        mode for mode in SURFACE_MODES if mode in visualization or mode in {"shrinkage", "hotspot", "niyama", "fraction_liquid"}
    )
    summary["raw_ranges"] = raw_ranges

    return {
        "summary": summary,
        "visualization": visualization,
    }


def read_artifact(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return cast(Dict[str, Any], json.load(handle))


def write_artifact(path: Path, artifact: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle)


def build_result_fields_payload(
    visualization: Dict[str, Any],
    *,
    max_points: int = 120_000,
) -> Dict[str, Any]:
    centers = np.asarray(visualization.get("centers", []), dtype=np.float32)
    sizes = np.asarray(visualization.get("sizes", []), dtype=np.float32)
    if centers.size == 0:
        return {
            "version": 1,
            "empty": True,
            "centers": [],
            "sizes": [],
            "modes": [],
            "fields": {},
            "solidification": {"normalized_times": [], "min_time": 0.0, "max_time": 0.0, "front_width": 0.075},
            "source_count": 0,
            "sample_count": 0,
        }

    step = max(1, len(centers) // max(1, max_points))
    idx = np.arange(0, len(centers), step, dtype=np.int32)
    fields: Dict[str, Dict[str, Any]] = {}
    for mode in sorted(SURFACE_MODES - {"fraction_liquid"}):
        values = visualization.get(mode)
        if values is None:
            continue
        arr = np.asarray(values, dtype=np.float32)
        if len(arr) != len(centers):
            continue
        sampled = arr[idx]
        fields[mode] = {
            "values": sampled.tolist(),
            "empty": bool(not np.any(np.isfinite(sampled) & (sampled > 0.0))),
            **_range_payload(sampled),
        }

    solidification = _solidification_payload(visualization, idx, len(centers))
    fraction_liquid = _fraction_liquid_frames_payload(visualization, idx, len(centers))
    has_fraction_liquid = bool(solidification["normalized_times"] or fraction_liquid["frames"])
    modes = sorted(list(fields.keys()) + (["fraction_liquid"] if has_fraction_liquid else []))
    return {
        "version": 1,
        "empty": False,
        "centers": centers[idx].reshape(-1).tolist(),
        "sizes": sizes[idx].tolist() if len(sizes) == len(centers) else [],
        "modes": modes,
        "fields": fields,
        "solidification": solidification,
        "fraction_liquid": fraction_liquid,
        "source_count": int(len(centers)),
        "sample_count": int(len(idx)),
    }


def build_fork_surface_payload(
    visualization: Dict[str, Any],
    mode: str,
    threshold: float,
    time_percent: float,
) -> Dict[str, Any]:
    normalized_mode = mode.strip().lower()
    if normalized_mode not in SURFACE_MODES:
        raise ValueError(f"Unsupported surface mode: {mode}")
    if normalized_mode == "fraction_liquid":
        return build_voxel_glyph_payload(visualization, normalized_mode, max(threshold, 0.01), time_percent, max_voxels=12_000)
    if normalized_mode in {"shrinkage", "hotspot", "feed_modulus", "niyama", "niyama_raw", "combined_feed_risk"}:
        return build_voxel_glyph_payload(visualization, normalized_mode, threshold, time_percent, max_voxels=8_000)
    if normalized_mode in BASIC_SURFACE_MODES:
        return cast(Dict[str, Any], build_surface_payload(
            visualization,
            normalize_surface_request(normalized_mode, threshold, time_percent),
        ))
    return build_voxel_glyph_payload(visualization, normalized_mode, threshold, time_percent)


def build_volume_payload(
    visualization: Dict[str, Any],
    mode: str,
    time_percent: float,
    *,
    max_points: int = 120_000,
) -> Dict[str, Any]:
    normalized_mode = mode.strip().lower()
    values = _values_for_mode(visualization, normalized_mode, time_percent)
    centers = np.asarray(visualization.get("centers", []), dtype=np.float32)
    sizes = np.asarray(visualization.get("sizes", []), dtype=np.float32)
    if values is None or centers.size == 0 or len(values) != len(centers):
        return {"mode": normalized_mode, "empty": True, "centers": [], "sizes": [], "values": []}
    step = max(1, len(centers) // max_points)
    idx = np.arange(0, len(centers), step, dtype=np.int32)
    return {
        "mode": normalized_mode,
        "empty": False,
        "centers": centers[idx].reshape(-1).tolist(),
        "sizes": sizes[idx].tolist() if len(sizes) == len(centers) else [],
        "values": np.asarray(values, dtype=np.float32)[idx].tolist(),
        "source_count": int(len(centers)),
        "sample_count": int(len(idx)),
    }


def build_voxel_glyph_payload(
    visualization: Dict[str, Any],
    mode: str,
    threshold: float,
    time_percent: float,
    *,
    max_voxels: int = 6_000,
) -> Dict[str, Any]:
    values = _values_for_mode(visualization, mode, time_percent)
    centers = np.asarray(visualization.get("centers", []), dtype=np.float32)
    sizes = np.asarray(visualization.get("sizes", []), dtype=np.float32)
    threshold = float(np.clip(threshold, 0.0, 1.0))
    if values is None or centers.size == 0 or len(values) != len(centers):
        return _empty_surface(mode, threshold, time_percent, "no values")
    values = np.asarray(values, dtype=np.float32)
    active = np.where(np.isfinite(values) & (values >= threshold))[0]
    if active.size == 0:
        return _empty_surface(mode, threshold, time_percent, "below threshold")

    if active.size > max_voxels:
        order = np.argsort(values[active])[-max_voxels:]
        active = active[order]

    positions: list[float] = []
    normals: list[float] = []
    indices: list[int] = []
    scalar_values: list[float] = []
    base_size = float(np.median(sizes)) if sizes.size else 0.1
    for center, value_index in zip(centers[active], active):
        size = float(sizes[value_index]) if len(sizes) == len(centers) else base_size
        _append_cube(positions, normals, indices, center, size * 0.86)
        scalar_values.append(float(values[value_index]))

    return {
        "mode": mode,
        "threshold": threshold,
        "time_percent": float(np.clip(time_percent, 0.0, 100.0)),
        "empty": False,
        "positions": positions,
        "normals": normals,
        "indices": indices,
        "values": scalar_values,
        "active_count": int(active.size),
        "source_count": int(len(values)),
        "vertex_count": len(positions) // 3,
        "triangle_count": len(indices) // 3,
    }


def _values_for_mode(visualization: Dict[str, Any], mode: str, time_percent: float) -> Optional[np.ndarray]:
    if mode == "fraction_liquid":
        solidification_path = _solidification_path_values(visualization, time_percent)
        if solidification_path is not None:
            return solidification_path
        history = visualization.get("fraction_liquid_history")
        if isinstance(history, list) and history:
            times = np.asarray(visualization.get("fraction_liquid_times") or [], dtype=np.float32)
            frames = [np.asarray(frame, dtype=np.float32) for frame in history]
            if len(times) != len(frames) or len(frames) == 1:
                index = min(int((time_percent / 100.0) * len(frames)), len(frames) - 1)
                return frames[index]
            target = float(times[0] + (time_percent / 100.0) * (times[-1] - times[0]))
            after = int(np.searchsorted(times, target, side="left"))
            if after <= 0:
                return frames[0]
            if after >= len(frames):
                return frames[-1]
            before = after - 1
            span = float(times[after] - times[before])
            alpha = 0.0 if span <= 0.0 else float((target - times[before]) / span)
            return (frames[before] * (1.0 - alpha) + frames[after] * alpha).astype(np.float32)
    values = visualization.get(mode)
    return np.asarray(values, dtype=np.float32) if values is not None else None


def _solidification_path_values(visualization: Dict[str, Any], time_percent: float) -> Optional[np.ndarray]:
    solidification_times = visualization.get("solidification_times")
    if solidification_times is None:
        return None
    times = np.asarray(solidification_times, dtype=np.float32)
    finite = times[np.isfinite(times) & (times >= 0.0)]
    if finite.size == 0:
        return None
    min_time = float(np.min(finite))
    max_time = float(np.max(finite))
    if max_time <= min_time:
        return None

    progress = float(np.clip(time_percent / 100.0, 0.0, 1.0))
    if progress >= 0.995:
        return np.zeros_like(times, dtype=np.float32)

    normalized = np.clip((times - min_time) / (max_time - min_time), 0.0, 1.0)
    front_width = 0.075
    remaining_liquid = np.clip((normalized - progress) / front_width, 0.0, 1.0)
    return remaining_liquid.astype(np.float32)


def _solidification_payload(visualization: Dict[str, Any], idx: np.ndarray, expected_length: int) -> Dict[str, Any]:
    solidification_times = visualization.get("solidification_times")
    if solidification_times is None:
        return {"normalized_times": [], "min_time": 0.0, "max_time": 0.0, "front_width": 0.075}
    times = np.asarray(solidification_times, dtype=np.float32)
    if len(times) != expected_length:
        return {"normalized_times": [], "min_time": 0.0, "max_time": 0.0, "front_width": 0.075}
    finite = times[np.isfinite(times) & (times >= 0.0)]
    if finite.size == 0:
        return {"normalized_times": [], "min_time": 0.0, "max_time": 0.0, "front_width": 0.075}
    min_time = float(np.min(finite))
    max_time = float(np.max(finite))
    if max_time <= min_time:
        normalized = np.zeros_like(times, dtype=np.float32)
    else:
        normalized = np.clip((times - min_time) / (max_time - min_time), 0.0, 1.0).astype(np.float32)
    return {
        "normalized_times": normalized[idx].tolist(),
        "min_time": min_time,
        "max_time": max_time,
        "front_width": 0.075,
    }


def _fraction_liquid_frames_payload(
    visualization: Dict[str, Any],
    idx: np.ndarray,
    expected_length: int,
    *,
    max_frames: int = 48,
) -> Dict[str, Any]:
    history = visualization.get("fraction_liquid_history")
    if not isinstance(history, list) or not history:
        return {"frames": [], "frame_percents": []}

    frame_indices = np.linspace(0, len(history) - 1, min(max_frames, len(history)), dtype=np.int32)
    frames: list[list[float]] = []
    for frame_index in frame_indices:
        frame = np.asarray(history[int(frame_index)], dtype=np.float32)
        if len(frame) != expected_length:
            return {"frames": [], "frame_percents": []}
        frames.append(np.clip(frame[idx], 0.0, 1.0).tolist())

    if len(frame_indices) == 1:
        frame_percents = [0.0]
    else:
        denominator = max(len(history) - 1, 1)
        frame_percents = [float((int(frame_index) / denominator) * 100.0) for frame_index in frame_indices]
    return {"frames": frames, "frame_percents": frame_percents}


def _append_cube(
    positions: list[float],
    normals: list[float],
    indices: list[int],
    center: np.ndarray,
    size: float,
) -> None:
    cx, cy, cz = [float(v) for v in center]
    h = max(float(size) * 0.5, 1e-4)
    vertices = [
        (cx - h, cy - h, cz - h),
        (cx + h, cy - h, cz - h),
        (cx + h, cy + h, cz - h),
        (cx - h, cy + h, cz - h),
        (cx - h, cy - h, cz + h),
        (cx + h, cy - h, cz + h),
        (cx + h, cy + h, cz + h),
        (cx - h, cy + h, cz + h),
    ]
    faces = [
        (0, 1, 2, 3, (0, 0, -1)),
        (4, 7, 6, 5, (0, 0, 1)),
        (0, 4, 5, 1, (0, -1, 0)),
        (1, 5, 6, 2, (1, 0, 0)),
        (2, 6, 7, 3, (0, 1, 0)),
        (3, 7, 4, 0, (-1, 0, 0)),
    ]
    for a, b, c, d, normal in faces:
        base = len(positions) // 3
        for idx in (a, b, c, d):
            positions.extend(vertices[idx])
            normals.extend(normal)
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])


def _matching_indices(source_centers: np.ndarray, sampled_centers: np.ndarray) -> np.ndarray:
    if len(sampled_centers) == 0 or len(source_centers) == len(sampled_centers):
        return np.arange(len(source_centers), dtype=np.int32)
    if len(source_centers) == 0:
        return np.array([], dtype=np.int32)
    step = max(1, len(source_centers) // len(sampled_centers))
    indices = np.arange(0, len(source_centers), step, dtype=np.int32)
    return indices[: len(sampled_centers)]


def _bounds_payload(bounds: np.ndarray) -> Dict[str, list[float]]:
    return {
        "x": [float(bounds[0][0]), float(bounds[0][1])],
        "y": [float(bounds[1][0]), float(bounds[1][1])],
        "z": [float(bounds[2][0]), float(bounds[2][1])],
    }


def _normalize(values: Any) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    min_v = float(np.min(finite))
    max_v = float(np.max(finite))
    if max_v <= min_v:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - min_v) / (max_v - min_v), 0.0, 1.0).astype(np.float32)


def _invert(values: Any) -> Optional[np.ndarray]:
    if values is None:
        return None
    return (1.0 - np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)).astype(np.float32)


def _range_payload(values: np.ndarray) -> Dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def _empty_surface(mode: str, threshold: float, time_percent: float, reason: str) -> Dict[str, Any]:
    return {
        "mode": mode,
        "threshold": threshold,
        "time_percent": time_percent,
        "empty": True,
        "reason": reason,
        "positions": [],
        "normals": [],
        "indices": [],
        "active_count": 0,
        "source_count": 0,
        "vertex_count": 0,
        "triangle_count": 0,
    }
