from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast
import json
import math
import os
import re
import tempfile

import numpy as np
import trimesh
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from core.part_manager import (
    PartManager,
    PartType,
    STEP_SUPPORTED,
    _scale_step_to_inches,
    build_part_manager as _core_build_part_manager,
    infer_part_type_from_filename,
    infer_part_type_from_name,
    riser_sleeve_from_value,
    strip_cad_extension,
    tessellation_tolerance_mm,
)
from core.simulation_pipeline import PipelineResult
from web.repository import DEFAULT_VOXEL_SIZE_IN, strip_upload_prefix

ALLOWED_CAD_EXTENSIONS = {".stl", ".step", ".stp"}
_STEP_EXTENSIONS = (".step", ".stp")

def _format_dimensions(bounds: np.ndarray) -> str:
    if bounds is None:
        return "N/A"
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (2, 3):
        raise ValueError(
            f"Expected trimesh bounds of shape (2, 3), got {bounds.shape}"
        )
    x_size = float(bounds[1][0] - bounds[0][0])
    y_size = float(bounds[1][1] - bounds[0][1])
    z_size = float(bounds[1][2] - bounds[0][2])
    return f"{x_size:.2f}\" x {y_size:.2f}\" x {z_size:.2f}\""

def _build_mesh_data(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    bounds_native = np.asarray(mesh.bounds, dtype=np.float64)
    bounds_per_axis = bounds_native.T
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    return {
        "vertices": vertices.reshape(-1).tolist(),
        "faces": faces.reshape(-1).tolist(),
        "bounds": _bounds_payload(bounds_per_axis),
        "volume": float(mesh.volume) if mesh.volume else 0.0,
        "surface_area": float(mesh.area) if mesh.area else 0.0,
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
    }

SIMULATED_PART_FIELDS = (
    "id",
    "filepath",
    "part_type",
    "visible",
    "is_open_top",
    "riser_sleeve",
    "position_x",
    "position_y",
    "position_z",
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "scale",
)

def simulated_parts_snapshot(parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    snapshot: List[Dict[str, Any]] = []
    for part in parts:
        entry: Dict[str, Any] = {}
        for field in SIMULATED_PART_FIELDS:
            value = part.get(field)
            if field in ("visible", "is_open_top"):
                value = bool(value if value is not None else field == "visible")
            elif field == "riser_sleeve":
                value = riser_sleeve_from_value(value)
            elif field.startswith(("position_", "rotation_")) or field == "scale":
                value = float(value if value is not None else (1.0 if field == "scale" else 0.0))
            else:
                value = str(value or "")
            entry[field] = value
        snapshot.append(entry)
    return snapshot

def _sanitize_body_name(name: str) -> str:
    name = strip_cad_extension(name)
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return sanitized or "body"

def upload_voxel_size_in(value: Union[str, float, None]) -> float:
    if value is None:
        return DEFAULT_VOXEL_SIZE_IN
    try:
        voxel_size_in = float(value)
    except (TypeError, ValueError):
        return DEFAULT_VOXEL_SIZE_IN
    if not math.isfinite(voxel_size_in) or voxel_size_in <= 0.0:
        return DEFAULT_VOXEL_SIZE_IN
    return voxel_size_in

def upload_cad_file(
    file: FileStorage,
    upload_folder: Path,
    voxel_size_in: Union[str, float, None] = None,
) -> List[Dict[str, Any]]:
    if not file.filename:
        raise ValueError("No file selected")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_CAD_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Allowed: {sorted(ALLOWED_CAD_EXTENSIONS)}"
        )

    upload_folder.mkdir(parents=True, exist_ok=True)
    file_id = os.urandom(4).hex()
    filename = secure_filename(file.filename)
    filepath = upload_folder / f"{file_id}_{filename}"
    file.save(str(filepath))

    if ext in _STEP_EXTENSIONS and not STEP_SUPPORTED:
        try:
            filepath.unlink()
        except FileNotFoundError:
            pass
        raise ValueError(
            "STEP support requires the 'cascadio' package. "
            "Install with: pip install cascadio"
        )

    is_step = ext in _STEP_EXTENSIONS
    try:
        if is_step:
            tol_linear = tessellation_tolerance_mm(upload_voxel_size_in(voxel_size_in))
            loaded = trimesh.load(str(filepath), force="scene", tol_linear=tol_linear)
        else:
            loaded = trimesh.load(str(filepath), force="scene")
    except Exception as exc:
        try:
            filepath.unlink()
        except FileNotFoundError:
            pass
        raise ValueError(f"Failed to load CAD file: {exc}") from exc

    if is_step:
        _scale_step_to_inches(loaded, str(filepath))

    if isinstance(loaded, trimesh.Scene):
        body_items = list(loaded.geometry.items())
        if not body_items:
            try:
                filepath.unlink()
            except FileNotFoundError:
                pass
            raise ValueError("CAD assembly contains no bodies")
    else:
        body_items = [("Body_1", loaded)]

    nonempty_indices = [
        index
        for index, (_, body_mesh) in enumerate(body_items)
        if body_mesh is not None and len(body_mesh.faces) > 0
    ]
    if len(nonempty_indices) == 1:
        fallback_name = strip_upload_prefix(os.path.splitext(filename)[0])
        only_index = nonempty_indices[0]
        body_items[only_index] = (fallback_name, body_items[only_index][1])

    entries: List[Dict[str, Any]] = []
    for body_index, (body_name, body_mesh) in enumerate(body_items):
        if body_mesh is None or len(body_mesh.faces) == 0:
            continue

        display_name = strip_upload_prefix(_sanitize_body_name(body_name))
        body_filename = f"{file_id}_b{body_index}_{display_name}.stl"
        body_path = upload_folder / body_filename
        body_mesh.export(str(body_path))

        part_type = infer_part_type_from_name(body_name).value

        entries.append(
            {
                "file_id": f"{file_id}_b{body_index}",
                "filename": body_filename,
                "display_name": display_name,
                "filepath": str(body_path),
                "source_file": filename,
                "body_name": body_name,
                "part_type": part_type,
                "dimensions": _format_dimensions(body_mesh.bounds),
                "mesh_data": _build_mesh_data(body_mesh),
            }
        )

    if not entries:
        try:
            filepath.unlink()
        except FileNotFoundError:
            pass
        raise ValueError("CAD file contains no loadable bodies")

    if is_step or len(entries) > 1:
        try:
            filepath.unlink()
        except FileNotFoundError:
            pass

    return entries

def load_part_mesh_data(filepath: str) -> Optional[Dict[str, Any]]:
    if not filepath:
        return None
    path = Path(filepath)
    if not path.exists():
        return None
    try:
        loaded = trimesh.load(str(path), force="mesh")
    except Exception:
        return None
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        return None
    return _build_mesh_data(loaded)

def build_part_manager(parts_data: list[Dict[str, Any]]) -> PartManager:
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
        "porosity": _display_field(engine, "get_microporosity_display_intensity", "porosity_probability"),
        "feeding_blocked": getattr(engine, "feeding_blocked", None),
        "feeding_path_lengths": getattr(engine, "feeding_path_lengths", None),
        "feeding_open_fraction": getattr(engine, "feeding_open_fraction", None),
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
        if key == "feed_modulus" and part_types is not None and len(part_types) == len(source_centers):
            chill_mask = np.isin(np.asarray(part_types, dtype=np.int32), (2, 4))
            if np.any(chill_mask):
                payload_arr = payload_arr.copy()
                payload_arr[chill_mask] = 0.0
        visualization[key] = payload_arr[indices].tolist()
        raw_ranges[key] = _range_payload(arr)

    summary = dict(result.result_data)
    niyama_metadata = getattr(engine, "niyama_metadata", None)
    if isinstance(niyama_metadata, dict):
        summary["niyama_metadata"] = dict(niyama_metadata)
    summary["raw_ranges"] = raw_ranges

    return {
        "summary": summary,
        "visualization": visualization,
        GRID_SOURCE_KEY: _full_grid_source(engine, mesh, source_centers),
    }

GRID_SOURCE_KEY = "_grid_source"

def _full_grid_source(engine: Any, mesh: Any, source_centers: np.ndarray) -> Dict[str, Any]:
    count = int(len(source_centers))
    source: Dict[str, Any] = {
        "centers": source_centers,
        "sizes": _full_array(getattr(mesh, "voxel_sizes", None), count),
        "part_types": _full_array(getattr(mesh, "voxel_part_types", None), count),
    }
    for key, method_name, attribute in (
        ("shrinkage", "get_shrink_display_intensity", "shrinkage_probability"),
        ("porosity", "get_microporosity_display_intensity", "porosity_probability"),
    ):
        values = _full_array(_display_field(engine, method_name, attribute), count)
        if values is not None:
            source[key] = values

    for key, attribute in (
        ("feed_modulus", "local_modulus"),
        ("hotspot", "hotspot_intensity"),
        ("solidification_times", "solidification_times"),
        ("niyama_raw", "niyama_values"),
        ("niyama_risk", "niyama_risk"),
        ("combined_feed_risk", "combined_feed_risk"),
        ("hot_tear", "hot_tear_risk"),
        ("sand_burn_in", "sand_burn_in_risk"),
        ("gas_generation", "gas_generation"),
        ("gas_entrapment", "gas_entrapment"),
        ("gas_porosity_risk", "gas_porosity_risk"),
        ("crack_susceptibility", "crack_susceptibility"),
        ("feeding_blocked", "feeding_blocked"),
        ("feeding_path_lengths", "feeding_path_lengths"),
        ("feeding_open_fraction", "feeding_open_fraction"),
        ("fill_time", "_fill_arrival_times"),
    ):
        values = _full_array(getattr(engine, attribute, None), count)
        if values is not None:
            source[key] = values

    feeding_quality = _invert(getattr(engine, "feeding_quality", None))
    if _full_array(feeding_quality, count) is not None:
        source["feeding_quality"] = np.asarray(feeding_quality, dtype=np.float32)
    for key, attribute in (("thermal_stress", "effective_stress"), ("distortion", "distortion_magnitude")):
        normalized = _normalize(getattr(engine, attribute, None))
        if _full_array(normalized, count) is not None:
            source[key] = np.asarray(normalized, dtype=np.float32)
    sdas = _normalize(getattr(engine, "sdas", None))
    if _full_array(sdas, count) is not None:
        source["sdas"] = np.asarray(sdas, dtype=np.float32)
    predicted_uts = _invert(_normalize(getattr(engine, "predicted_uts", None)))
    if _full_array(predicted_uts, count) is not None:
        source["predicted_uts"] = np.asarray(predicted_uts, dtype=np.float32)

    history = getattr(engine, "fraction_liquid_history", None)
    if history:
        frames = [np.asarray(1.0 - np.asarray(snapshot), dtype=np.float32) for snapshot in history]
        if all(frame.size == count for frame in frames):
            mask_liquid = getattr(engine, "_mask_non_cast_fraction_liquid", None)
            if callable(mask_liquid):
                frames = [np.asarray(mask_liquid(frame), dtype=np.float32) for frame in frames]
            source["fraction_liquid_history"] = frames
            source["fraction_liquid_times"] = list(getattr(engine, "fraction_liquid_times", None) or [])

    fill_frames = getattr(engine, "_fill_frames", None)
    if fill_frames:
        fractions = []
        times = []
        for frame in fill_frames:
            values = _full_array(getattr(frame, "fill_fraction", None), count)
            if values is None:
                continue
            fractions.append(np.clip(values, 0.0, 1.0).astype(np.float32))
            times.append(float(getattr(frame, "time_s", 0.0)))
        if fractions:
            source["fill_animation_history"] = fractions
            source["fill_animation_times"] = times

    return source

def _full_array(values: Any, count: int) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return arr if arr.size == count else None

def read_artifact(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return cast(Dict[str, Any], json.load(handle))

def write_artifact(path: Path, artifact: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid_source = artifact.pop(GRID_SOURCE_KEY, None)

    from web.fields import write_sidecar_for_artifact

    handle_fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    write_sidecar_for_artifact(path, artifact, grid_source=grid_source)

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

def _display_field(engine: Any, method_name: str, attribute: str) -> Any:
    if getattr(engine, attribute, None) is None:
        return None
    method = getattr(engine, method_name, None)
    if callable(method):
        return method()
    return getattr(engine, attribute)

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
