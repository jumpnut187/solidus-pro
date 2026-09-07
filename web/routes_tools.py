"""Application-shell routes: project management extras, riser design, reports,
hardware info, job logs, project file import/export.

Owned by the shell/parity workstream. Registered by
``web.app_factory.create_app``. Everything here lives under ``/api``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import os
import shutil

from flask import Blueprint, current_app, jsonify, request

from core.materials import get_all_alloy_names, get_alloy
from web import repository as repo

tools_bp = Blueprint("web_tools", __name__, url_prefix="/api")


def _database_path() -> Path:
    return Path(current_app.config["DATABASE"])


def _upload_folder() -> Path:
    return Path(current_app.config["UPLOAD_FOLDER"])


@tools_bp.route("/tools/health")
def tools_health():
    return jsonify({"ok": True, "blueprint": "web_tools"})


# ---------------------------------------------------------------------------
# Project management
# ---------------------------------------------------------------------------


@tools_bp.route("/projects/<project_id>/rename", methods=["POST"])
def rename_project(project_id: str):
    name = str((request.json or {}).get("name", "")).strip()
    if not name:
        return jsonify({"error": "A project name is required"}), 400
    if not repo.rename_project(_database_path(), project_id, name):
        return jsonify({"error": "Project not found"}), 404
    return jsonify({"project": repo.get_project(_database_path(), project_id)})


@tools_bp.route("/projects/<project_id>/duplicate", methods=["POST"])
def duplicate_project(project_id: str):
    """Copy a project's setup (parts, settings, pour location) but not results.

    Each part's CAD file is copied on disk as well, so deleting either project
    later cannot unlink geometry the other one still points at.
    """
    database_path = _database_path()
    source = repo.get_project(database_path, project_id)
    if source is None:
        return jsonify({"error": "Project not found"}), 404
    requested = str((request.json or {}).get("name", "")).strip()
    name = requested or f"{source.get('name', 'Project')} (copy)"

    upload_folder = _upload_folder()
    upload_folder.mkdir(parents=True, exist_ok=True)
    filepath_map: Dict[str, str] = {}
    for part in repo.get_parts(database_path, project_id):
        original = Path(str(part.get("filepath") or ""))
        if not original.name or not original.exists():
            continue
        copy_path = upload_folder / f"{os.urandom(4).hex()}_{original.name}"
        try:
            shutil.copy2(original, copy_path)
        except OSError:
            # A missing or unreadable source file should not abort the whole
            # duplicate -- the copied row keeps pointing at the original.
            continue
        filepath_map[str(part["id"])] = str(copy_path)

    new_id = repo.duplicate_project(database_path, project_id, name, filepath_map)
    if new_id is None:
        return jsonify({"error": "Project not found"}), 404
    return jsonify({"id": new_id, "project": repo.get_project(database_path, new_id)}), 201


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------


@tools_bp.route("/hardware")
def hardware():
    """Detected hardware, mirroring the desktop Help > Hardware Info dialog.

    Every field is optional from the client's point of view: detection runs
    through ``psutil`` and platform-specific probes that can fail on an
    unfamiliar host, and a missing core count should degrade the panel, not
    500 the request.
    """
    payload: Dict[str, Any] = {
        "cpu_brand": "Unknown",
        "cpu_vendor": "Unknown",
        "physical_cores": None,
        "logical_cores": None,
        "performance_cores": None,
        "efficiency_cores": None,
        "total_memory_gb": None,
        "available_memory_gb": None,
        "numba_threads": None,
        "gpu_backend": "none",
        "gpu_available": False,
        "metal_available": False,
        "neural_engine_available": False,
        "detection_error": None,
    }
    try:
        from config.hardware_config import HardwareConfig

        config = HardwareConfig()
        payload.update(
            {
                "cpu_brand": str(config.cpu_brand),
                "cpu_vendor": str(config.cpu_vendor),
                "physical_cores": config.physical_cores,
                "logical_cores": config.logical_cores,
                "performance_cores": config.performance_cores,
                "efficiency_cores": config.efficiency_cores,
                "total_memory_gb": round(config.total_memory / (1024 ** 3), 1),
                "available_memory_gb": round(config.available_memory / (1024 ** 3), 1),
                "numba_threads": config.get_optimal_thread_count("compute"),
                "gpu_available": bool(config.should_use_gpu()),
                "metal_available": bool(config.metal_available),
                "neural_engine_available": bool(config.neural_engine_available),
            }
        )
    except Exception as exc:  # pragma: no cover - host-specific detection
        payload["detection_error"] = str(exc)

    try:
        from core.gpu_compute import get_gpu_compute

        backend = get_gpu_compute().backend
        payload["gpu_backend"] = str(getattr(backend, "value", backend))
    except Exception:  # pragma: no cover - optional accelerator stack
        payload["gpu_backend"] = "none"

    return jsonify(payload)


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------


@tools_bp.route("/help/alloys")
def help_alloys():
    """The alloy reference table behind Help > Help Guide."""
    rows = []
    for name in get_all_alloy_names():
        alloy = get_alloy(name)
        rows.append(
            {
                "name": name,
                "full_name": alloy.name,
                "liquidus_temp": float(alloy.liquidus_temp),
                "solidus_temp": float(alloy.solidus_temp),
                "freezing_range": float(alloy.freezing_range),
                "pouring_temp": float(alloy.pouring_temp),
                "volumetric_shrinkage": round(float(alloy.volumetric_shrinkage), 2),
                "feeding_multiplier": float(alloy.feeding_distance_multiplier),
                "niyama_threshold": float(alloy.niyama_threshold),
            }
        )
    return jsonify({"alloys": rows})
