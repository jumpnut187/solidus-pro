"""Helpers for saving and loading desktop project state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
from typing import Any, Dict, List, Optional

from config.settings import MESH_TYPES
from core.part_manager import PartManager
from core.solidification_params import migrate_legacy_pour_rate
from ui.result_types import canonical_result_type


#: Version stamped on every project this build saves.
#:
#: 2 marks the pour-rate migration below. The bump is what makes that
#: migration a one-time repair of old files rather than a rule that reapplies
#: on every load: a value the user re-sets and saves comes back under version
#: 2 and is left alone.
PROJECT_SCHEMA_VERSION = 2

#: Versions this build can still open. An older project is migrated forward on
#: load (see load_project_state); an unrecognized one must fail loudly rather
#: than be misread as the current shape.
SUPPORTED_SCHEMA_VERSIONS = (1, 2)


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
    solidification_params: Optional[Dict[str, Any]] = None,
    pour_location: Optional[Dict[str, Any]] = None,
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
                # Which CAD body of a multi-body STEP file this part is.
                # None for STL parts and for STEP parts loaded via load_part.
                "body_name": part.body_name,
                "body_index": part.body_index,
            }
        )

    if solidification_params is None:
        solidification_params = {}
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
            "solidification_params": dict(solidification_params),
            # Serialized PourLocation (see core.pour_location.PourLocation.to_dict),
            # or None when the user has not picked one yet.
            "pour_location": dict(pour_location) if pour_location else None,
            "mesh_setting_origin": mesh_setting_origin,
            "current_vis_mode": current_vis_mode,
        },
        "ui_state": dict(ui_state),
        "parts": parts,
    }
    return ProjectState(payload)


def load_project_state(payload: Dict[str, Any]) -> ProjectState:
    """Validate and normalize a project payload, migrating older versions.

    What comes back is always at PROJECT_SCHEMA_VERSION: an older file is
    brought forward here, so nothing downstream has to know which version it
    was saved under.
    """
    if not isinstance(payload, dict):
        raise ValueError("Project file must contain a JSON object")

    version = int(payload.get("schema_version", 0))
    if version not in SUPPORTED_SCHEMA_VERSIONS:
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
                # Absent on projects saved before bodies were recorded; the
                # loader then matches STEP bodies by saved order.
                "body_name": _normalize_optional_str(part.get("body_name")),
                "body_index": _normalize_optional_int(part.get("body_index")),
            }
        )

    solidification_params = dict(simulation.get("solidification_params", {}))
    if version < 2:
        # A project written before the square-root pour rate existed carries
        # the fixed 5 lb/s default it inherited. See
        # core.solidification_params.migrate_legacy_pour_rate for why that is
        # rewritten rather than honoured, and why the version gate is what
        # keeps a deliberate 5 lb/s settable.
        solidification_params = migrate_legacy_pour_rate(solidification_params)

    normalized = {
        # The payload returned is at the current schema, whatever the file
        # said: every migration above has already been applied to it.
        "schema_version": PROJECT_SCHEMA_VERSION,
        "saved_at": str(payload.get("saved_at", "")),
        "simulation": {
            "alloy_name": str(simulation.get("alloy_name", "WCB")),
            "voxel_size": float(simulation.get("voxel_size", 0.1)),
            "refinement_level": int(simulation.get("refinement_level", 3)),
            "memory_mode": str(simulation.get("memory_mode", "auto")),
            "analysis_options": dict(simulation.get("analysis_options", {})),
            "performance_options": _normalize_performance_options(simulation.get("performance_options")),
            "solidification_params": solidification_params,
            # Absent on any project saved before pour locations existed --
            # None is the correct "not set" value for those older files.
            "pour_location": _normalize_pour_location(simulation.get("pour_location")),
            "mesh_setting_origin": str(simulation.get("mesh_setting_origin", "manual")),
            "current_vis_mode": str(simulation.get("current_vis_mode", "shrinkage")),
        },
        "ui_state": _normalize_ui_state(payload.get("ui_state")),
        "parts": normalized_parts,
    }
    return ProjectState(normalized)


def _normalize_performance_options(value: Any) -> Dict[str, Any]:
    """Copy ``performance_options``, keeping ``mesh_type`` only when it names a known mesh.

    ``mesh_type`` ("voxel" or "tet") lives here so it round-trips without a
    schema bump. Absent (projects saved before the toggle) or unknown means
    "use ``settings.mesh.mesh_type``", so an unknown value is dropped.
    ``voxel_cut_cells`` works the same way: kept when it is a bool,
    dropped otherwise (absent means ``settings.mesh.voxel_cut_cells``).
    """
    options = dict(value) if isinstance(value, dict) else {}
    if "mesh_type" in options:
        mesh_type = str(options["mesh_type"] or "").strip().lower()
        if mesh_type in MESH_TYPES:
            options["mesh_type"] = mesh_type
        else:
            del options["mesh_type"]
    if "voxel_cut_cells" in options and not isinstance(options["voxel_cut_cells"], bool):
        del options["voxel_cut_cells"]
    return options


def _normalize_ui_state(value: Any) -> Dict[str, Any]:
    """Copy ``ui_state``, keeping ``parting_line`` only when it is well formed.

    A missing ``parting_line`` (projects saved before the Orient & Parting
    Line tool) means "not set yet", so a malformed one is dropped rather than
    guessed at. A legacy ``result_type`` (``"Shrinkage/Porosity"``) is renamed
    to its current label (``"Shrink"``).
    """
    ui_state = dict(value) if isinstance(value, dict) else {}
    if isinstance(ui_state.get("result_type"), str):
        ui_state["result_type"] = canonical_result_type(ui_state["result_type"])
    if "parting_line" in ui_state:
        parting_line = normalize_parting_line(ui_state["parting_line"])
        if parting_line is None:
            del ui_state["parting_line"]
        else:
            ui_state["parting_line"] = parting_line
    return ui_state


def normalize_parting_line(value: Any) -> Optional[Dict[str, Any]]:
    """Normalize ``ui_state["parting_line"]`` = ``{"z", "visible", "locked_to_casting"}``.

    Returns ``None`` when absent or when ``z`` is not a finite number.
    """
    if not isinstance(value, dict):
        return None
    try:
        z = float(value.get("z"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(z):
        return None
    return {
        "z": z,
        "visible": bool(value.get("visible", True)),
        "locked_to_casting": bool(value.get("locked_to_casting", True)),
    }


def _normalize_pour_location(value: Any) -> Optional[Dict[str, Any]]:
    """Normalize a persisted pour location payload.

    Returns ``None`` for anything that is not a dict -- absent (an older
    project file predating pour locations), ``null``, or malformed -- so a
    corrupt value cannot crash project load. Deeper validation (schema
    version, finite coordinates, ...) happens where the dict is turned into
    an actual ``core.pour_location.PourLocation``.
    """
    if not isinstance(value, dict):
        return None
    return dict(value)


def _normalize_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _normalize_optional_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_vector(value: Any, expected_len: int, *, default: float) -> List[float]:
    """Normalize a vector-like value into a list of floats."""
    if not isinstance(value, (list, tuple)):
        return [default] * expected_len

    normalized = [float(item) for item in value[:expected_len]]
    while len(normalized) < expected_len:
        normalized.append(default)
    return normalized
