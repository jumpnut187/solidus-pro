"""Helpers for saving and loading desktop project state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from typing import Any, Dict, List

from core.part_manager import PartManager


PROJECT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ProjectState:
    """Normalized project state for the desktop application."""

    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return self.payload


def build_project_state(
    part_manager: PartManager,
    *,
    alloy_name: str,
    voxel_size: float,
    refinement_level: int,
    memory_mode: str,
    analysis_options: Dict[str, Any],
    performance_options: Dict[str, Any],
    mesh_setting_origin: str,
    current_vis_mode: str,
    ui_state: Dict[str, Any],
) -> ProjectState:
    """Create a serializable project state payload."""
    parts: List[Dict[str, Any]] = []
    for part in part_manager.get_all_parts():
        parts.append(
            {
                "name": part.name,
                "filepath": part.filepath,
                "part_type": part.part_type.value,
                "position": [float(value) for value in part.position],
                "rotation": [float(value) for value in part.rotation],
                "scale": float(part.scale),
                "color": [float(value) for value in part.color],
                "visible": bool(part.visible),
                "is_open_top": bool(part.is_open_top),
            }
        )

    payload = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "simulation": {
            "alloy_name": alloy_name,
            "voxel_size": float(voxel_size),
            "refinement_level": int(refinement_level),
            "memory_mode": memory_mode,
            "analysis_options": dict(analysis_options),
            "performance_options": dict(performance_options),
            "mesh_setting_origin": mesh_setting_origin,
            "current_vis_mode": current_vis_mode,
        },
        "ui_state": dict(ui_state),
        "parts": parts,
    }
    return ProjectState(payload)


def load_project_state(payload: Dict[str, Any]) -> ProjectState:
    """Validate and normalize a project payload."""
    if not isinstance(payload, dict):
        raise ValueError("Project file must contain a JSON object")

    version = int(payload.get("schema_version", 0))
    if version != PROJECT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported project schema version: {version}")

    simulation = payload.get("simulation")
    if not isinstance(simulation, dict):
        raise ValueError("Project file is missing simulation settings")

    parts = payload.get("parts")
    if not isinstance(parts, list):
        raise ValueError("Project file is missing parts data")

    normalized_parts: List[Dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            raise ValueError("Part entries must be objects")
        filepath = str(part.get("filepath", ""))
        if not filepath:
            raise ValueError("Each part must include a filepath")
        normalized_parts.append(
            {
                "name": str(part.get("name", os.path.splitext(os.path.basename(filepath))[0])),
                "filepath": filepath,
                "part_type": str(part.get("part_type", "cast")),
                "position": _normalize_vector(part.get("position"), 3, default=0.0),
                "rotation": _normalize_vector(part.get("rotation"), 3, default=0.0),
                "scale": float(part.get("scale", 1.0)),
                "color": _normalize_vector(part.get("color"), 3, default=0.7),
                "visible": bool(part.get("visible", True)),
                "is_open_top": bool(part.get("is_open_top", False)),
            }
        )

    normalized = {
        "schema_version": version,
        "saved_at": str(payload.get("saved_at", "")),
        "simulation": {
            "alloy_name": str(simulation.get("alloy_name", "WCB")),
            "voxel_size": float(simulation.get("voxel_size", 0.1)),
            "refinement_level": int(simulation.get("refinement_level", 3)),
            "memory_mode": str(simulation.get("memory_mode", "auto")),
            "analysis_options": dict(simulation.get("analysis_options", {})),
            "performance_options": dict(simulation.get("performance_options", {})),
            "mesh_setting_origin": str(simulation.get("mesh_setting_origin", "manual")),
            "current_vis_mode": str(simulation.get("current_vis_mode", "shrinkage")),
        },
        "ui_state": dict(payload.get("ui_state", {})),
        "parts": normalized_parts,
    }
    return ProjectState(normalized)


def _normalize_vector(value: Any, expected_len: int, *, default: float) -> List[float]:
    """Normalize a vector-like value into a list of floats."""
    if not isinstance(value, (list, tuple)):
        return [default] * expected_len

    normalized = [float(item) for item in value[:expected_len]]
    while len(normalized) < expected_len:
        normalized.append(default)
    return normalized
