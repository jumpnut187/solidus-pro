from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import joblib
except ImportError:
    joblib = None

from config.hardware_config import hardware
from core.memory_limits import (
    get_max_voxels_for_system,
    BYTES_PER_VOXEL,
    calculate_safe_voxel_size,
    get_system_memory_info,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")
_DATASET_PATH = os.path.join(_DATA_DIR, "mesh_runs.csv")
_MODEL_PATH = os.path.join(_MODEL_DIR, "mesh_recommender.joblib")
_SCALER_PATH = os.path.join(_MODEL_DIR, "mesh_scaler.joblib")

_BYTES_PER_VOXEL = BYTES_PER_VOXEL

@dataclass
class MeshRecommendation:

    base_voxel_size: float
    refinement_level: int
    confidence: float
    reason: str
    used_model: bool
    warning: Optional[str] = None

def _ensure_directories() -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    os.makedirs(_MODEL_DIR, exist_ok=True)

def _extract_hardware_features() -> Dict[str, Any]:
    available_memory = getattr(hardware, "available_memory", 0) or 0
    physical_cores = getattr(hardware, "physical_cores", None) or getattr(hardware, "cpu_count", 1)
    performance_cores = getattr(hardware, "performance_cores", None) or 0
    efficiency_cores = getattr(hardware, "efficiency_cores", None) or 0

    return {
        "available_memory_bytes": int(available_memory),
        "available_memory_gb": available_memory / (1024 ** 3) if available_memory else 0.0,
        "physical_cores": int(physical_cores),
        "performance_cores": int(performance_cores),
        "efficiency_cores": int(efficiency_cores),
        "cpu_brand": getattr(hardware, "cpu_brand", "unknown"),
    }

def extract_mesh_features(stl_loader: Any) -> Dict[str, Any]:
    if not getattr(stl_loader, "is_valid", lambda: False)():
        raise ValueError("STL loader is not initialized; load a model first.")

    trimesh_obj = getattr(stl_loader, "trimesh_obj", None)
    bounds = getattr(stl_loader, "bounds", None)
    volume = float(getattr(stl_loader, "volume", 0.0) or 0.0)
    surface_area = float(getattr(stl_loader, "surface_area", 0.0) or 0.0)

    if trimesh_obj is None or bounds is None:
        raise ValueError("STL loader does not have geometry information available.")

    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    z_min, z_max = bounds[2]

    dim_x = float(x_max - x_min)
    dim_y = float(y_max - y_min)
    dim_z = float(z_max - z_min)
    max_dim = max(dim_x, dim_y, dim_z)
    min_dim = min(d for d in (dim_x, dim_y, dim_z) if d > 0) if max_dim > 0 else 0.0
    bbox_volume = dim_x * dim_y * dim_z if dim_x and dim_y and dim_z else 0.0

    triangle_count = int(len(getattr(trimesh_obj, "faces", [])))
    vertex_count = int(len(getattr(trimesh_obj, "vertices", [])))

    surface_to_volume = (surface_area / volume) if volume > 0 else 0.0
    fill_ratio = (volume / bbox_volume) if bbox_volume > 0 else 0.0
    aspect_ratio = (max_dim / min_dim) if min_dim > 0 else 1.0

    complexity = min(1.0, math.log10(max(triangle_count, 1)) / 6.0 + surface_to_volume / 50.0)

    return {
        "dim_x": dim_x,
        "dim_y": dim_y,
        "dim_z": dim_z,
        "max_dim": max_dim,
        "min_dim": min_dim,
        "bbox_volume": bbox_volume,
        "volume": volume,
        "surface_area": surface_area,
        "surface_to_volume": surface_to_volume,
        "fill_ratio": fill_ratio,
        "aspect_ratio": aspect_ratio,
        "triangle_count": triangle_count,
        "vertex_count": vertex_count,
        "complexity": complexity,
    }

def _load_model():
    if joblib is None:
        return None, None
    if not os.path.exists(_MODEL_PATH):
        return None, None
    try:
        model = joblib.load(_MODEL_PATH)
        scaler = None
        if os.path.exists(_SCALER_PATH):
            scaler = joblib.load(_SCALER_PATH)
        return model, scaler
    except Exception as exc:
        print(f"[MeshML] Failed to load model: {exc}")
        return None, None

def _estimate_simulation_time(voxel_count: int, refinement_level: int, physical_cores: int) -> float:
    voxel_count = max(0, int(voxel_count) if voxel_count else 0)
    refinement_level = max(1, min(4, int(refinement_level) if refinement_level else 1))
    physical_cores = max(1, int(physical_cores) if physical_cores else 1)

    core_factor = max(1.0, 8.0 / physical_cores)
    refinement_factor = 1.0 + (refinement_level - 1) * 0.3

    mesh_time = (voxel_count * 0.00006) * core_factor
    sim_time = (voxel_count * 0.00004) * core_factor * refinement_factor

    return max(0.0, mesh_time + sim_time)

def _heuristic_recommendation(mesh_features: Dict[str, Any], hardware_features: Dict[str, Any]) -> MeshRecommendation:
    physical_cores = max(hardware_features.get("physical_cores", 1), 1)
    available_memory_gb = hardware_features.get("available_memory_gb", 8.0) or 8.0
    max_dim = mesh_features.get("max_dim", 1.0) or 1.0
    min_dim = mesh_features.get("min_dim", max_dim) or max_dim
    volume = mesh_features.get("volume", 0.0) or 0.0
    complexity = mesh_features.get("complexity", 0.5) or 0.0
    surface_to_volume = mesh_features.get("surface_to_volume", 0.0) or 0.0
    aspect_ratio = mesh_features.get("aspect_ratio", 1.0) or 1.0
    fill_ratio = mesh_features.get("fill_ratio", 0.5)
    fill_ratio = max(fill_ratio, 0.1)

    max_voxels = get_max_voxels_for_system()
    mem_info = get_system_memory_info()
    total_ram_gb = mem_info.get('total_ram_gb', 8.0)

    ram_factor = min(total_ram_gb / 16.0, 2.0)
    core_factor = min(physical_cores / 8.0, 1.5)

    compute_factor = (ram_factor * 0.7 + core_factor * 0.3)

    if volume < 20.0:
        base_target = 150_000
    elif volume < 100.0:
        base_target = 250_000
    elif volume < 500.0:
        base_target = 400_000
    elif volume < 2000.0:
        base_target = 500_000
    else:
        base_target = 600_000

    target_voxels = int(base_target * compute_factor)

    if aspect_ratio > 5.0:
        target_voxels = int(target_voxels * 1.4)
    elif aspect_ratio > 3.0:
        target_voxels = int(target_voxels * 1.2)

    if surface_to_volume > 2.0:
        target_voxels = int(target_voxels * 1.2)
    elif surface_to_volume > 1.5:
        target_voxels = int(target_voxels * 1.1)

    target_voxels = min(target_voxels, max_voxels)

    target_grid_1d = (target_voxels / fill_ratio) ** (1.0 / 3.0)
    target_grid_1d = max(target_grid_1d, 25)
    target_grid_1d = min(target_grid_1d, 300)

    base_voxel_size = max_dim / target_grid_1d

    min_voxels_across_thin = 10
    max_size_for_thin_section = min_dim / min_voxels_across_thin
    base_voxel_size = min(base_voxel_size, max_size_for_thin_section)

    if min_dim < 0.5:
        min_voxel = 0.04
        max_voxel = 0.10
    elif min_dim < 1.0:
        min_voxel = 0.05
        max_voxel = 0.15
    elif max_dim > 40.0:
        min_voxel = 0.15
        max_voxel = 0.40
    elif max_dim > 25.0:
        min_voxel = 0.10
        max_voxel = 0.30
    else:
        min_voxel = 0.06
        max_voxel = 0.22

    base_voxel_size = max(min_voxel, min(base_voxel_size, max_voxel))

    if surface_to_volume > 2.5:
        base_voxel_size = max(min_voxel, base_voxel_size * 0.8)
    elif surface_to_volume > 2.0:
        base_voxel_size = max(min_voxel, base_voxel_size * 0.85)
    elif surface_to_volume > 1.5:
        base_voxel_size = max(min_voxel, base_voxel_size * 0.9)
    elif surface_to_volume < 0.4:
        base_voxel_size = min(max_voxel, base_voxel_size * 1.05)

    estimated_grid = max_dim / base_voxel_size
    estimated_voxels = int((estimated_grid ** 3) * fill_ratio)

    refinement_level = 4

    if estimated_voxels > max_voxels * 0.8:
        refinement_level = 3
    elif estimated_voxels > 600_000 and total_ram_gb < 24:
        refinement_level = 3
    elif estimated_voxels > 800_000:
        refinement_level = 3

    if complexity < 0.2 and estimated_voxels < 50_000 and surface_to_volume < 0.8:
        refinement_level = 3

    est_time = _estimate_simulation_time(estimated_voxels, refinement_level, physical_cores)

    if est_time > 300 and refinement_level > 3:
        refinement_level = 3
        est_time = _estimate_simulation_time(estimated_voxels, refinement_level, physical_cores)

    size_category = "small" if max_dim < 8 else ("medium" if max_dim < 20 else "large")
    geometry_note = ""
    if aspect_ratio > 4.0:
        geometry_note = ", high aspect ratio"
    elif surface_to_volume > 2.0:
        geometry_note = ", thin sections"
    elif fill_ratio < 0.2:
        geometry_note = ", sparse geometry"

    time_str = f"{est_time:.0f}s" if est_time < 60 else f"{est_time/60:.1f}min"
    reason = (
        f"Optimized for {size_category} {max_dim:.1f}\" part{geometry_note}. "
        f"~{estimated_voxels/1000:.0f}k voxels @ refinement {refinement_level}, "
        f"est. {time_str} ({physical_cores} cores, {total_ram_gb:.0f}GB RAM)"
    )

    warning = None
    warnings = []

    if estimated_voxels > max_voxels * 0.85:
        warnings.append("Near memory limit")
    if est_time > 120:
        warnings.append(f"Long simulation (~{est_time/60:.1f} min)")
    if fill_ratio < 0.12:
        warnings.append("Very low fill ratio - verify model is solid")
    if aspect_ratio > 7.0:
        warnings.append("Extreme aspect ratio may affect accuracy")
    if min_dim < 0.4 and base_voxel_size > min_dim / 6:
        warnings.append("Very thin section - verify mesh captures detail")

    if warnings:
        warning = "; ".join(warnings)

    confidence = 0.70

    if 0.2 < fill_ratio < 0.6 and 1.0 < aspect_ratio < 4.0:
        confidence = 0.85
    elif 0.15 < fill_ratio < 0.7 and aspect_ratio < 6.0:
        confidence = 0.78

    if estimated_voxels < max_voxels * 0.5 and est_time < 60:
        confidence += 0.05
    elif estimated_voxels > max_voxels * 0.8:
        confidence -= 0.08

    if complexity > 0.8:
        confidence -= 0.05

    confidence = max(0.55, min(confidence, 0.90))

    return MeshRecommendation(
        base_voxel_size=float(round(base_voxel_size, 3)),
        refinement_level=int(refinement_level),
        confidence=confidence,
        reason=reason,
        used_model=False,
        warning=warning,
    )

def recommend_mesh_settings(stl_loader: Any, target_accuracy: Optional[float] = None) -> MeshRecommendation:
    try:
        mesh_features = extract_mesh_features(stl_loader)
    except ValueError as e:
        raise ValueError(f"Cannot recommend mesh settings: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to extract mesh features: {e}") from e

    try:
        hardware_features = _extract_hardware_features()
    except Exception:
        hardware_features = {
            "available_memory_bytes": 8 * 1024**3,
            "available_memory_gb": 8.0,
            "physical_cores": 4,
            "performance_cores": 0,
            "efficiency_cores": 0,
            "cpu_brand": "unknown",
        }

    model, scaler = _load_model()
    if model is None:
        return _heuristic_recommendation(mesh_features, hardware_features)

    try:
        feature_vector = [
            mesh_features["max_dim"],
            mesh_features["surface_to_volume"],
            mesh_features["complexity"],
            mesh_features["fill_ratio"],
            mesh_features["triangle_count"],
            hardware_features["physical_cores"],
            hardware_features["available_memory_gb"],
        ]

        if scaler is not None:
            feature_vector_scaled = scaler.transform([feature_vector])
        else:
            feature_vector_scaled = [feature_vector]

        prediction = model.predict(feature_vector_scaled)[0]
        base_voxel_size = float(prediction[0])
        refinement_level = int(round(prediction[1]))
        confidence = 0.85

        reason = "Predicted by trained ML model"
        warning = None
        if target_accuracy is not None and target_accuracy > 0.9 and base_voxel_size > 0.05:
            warning = "High accuracy target detected; manual verification recommended."

        return MeshRecommendation(
            base_voxel_size=float(round(max(0.02, min(base_voxel_size, 0.4)), 3)),
            refinement_level=min(max(refinement_level, 1), 4),
            confidence=confidence,
            reason=reason,
            used_model=True,
            warning=warning,
        )
    except Exception as exc:
        print(f"[MeshML] Model prediction failed: {exc}")
        return _heuristic_recommendation(mesh_features, hardware_features)

def log_mesh_training_example(
    stl_loader: Any,
    mesh_parameters: Dict[str, Any],
    metrics: Dict[str, Any],
    origin: str = "manual",
) -> None:
    try:
        mesh_features = extract_mesh_features(stl_loader)
        hardware_features = _extract_hardware_features()
    except Exception as exc:
        print(f"[MeshML] Skipping logging; feature extraction failed: {exc}")
        return

    _ensure_directories()

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    row: Dict[str, Any] = {
        "timestamp": timestamp,
        "origin": origin,
        "base_voxel_size": mesh_parameters.get("base_voxel_size"),
        "refinement_level": mesh_parameters.get("refinement_level"),
        "voxel_count": metrics.get("voxel_count"),
        "mesh_generation_time_s": metrics.get("mesh_generation_time_s"),
        "simulation_time_s": metrics.get("simulation_time_s"),
        "success": metrics.get("success"),
        "memory_warning": metrics.get("memory_warning"),
    }

    for key, value in mesh_features.items():
        row[f"mesh_{key}"] = value
    for key, value in hardware_features.items():
        row[f"hw_{key}"] = value

    extra_fields = sorted(k for k in metrics.keys() if k not in row)
    for key in extra_fields:
        row[f"metric_{key}"] = metrics[key]

    fieldnames = list(row.keys())
    file_exists = os.path.exists(_DATASET_PATH)

    try:
        with open(_DATASET_PATH, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as exc:
        print(f"[MeshML] Failed to log mesh run: {exc}")
