from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
import math
import os
import shutil

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from core.materials import get_all_alloy_names, get_alloy
from web import repository as repo
from web.payloads import read_artifact
from web.services.simulation_queue import SimulationQueueManager

MAX_PROJECT_FILE_BYTES = 4 * 1024 * 1024

MAX_SCREENSHOT_BYTES = 10 * 1024 * 1024

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

def _triple(value: Any) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return [0.0, 0.0, 0.0]
    out: List[float] = []
    for item in value:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return [0.0, 0.0, 0.0]
        out.append(number if math.isfinite(number) else 0.0)
    return out

tools_bp = Blueprint("web_tools", __name__, url_prefix="/api")

def _database_path() -> Path:
    return Path(current_app.config["DATABASE"])

def _upload_folder() -> Path:
    return Path(current_app.config["UPLOAD_FOLDER"])

def _queue_manager() -> Optional[SimulationQueueManager]:
    manager = current_app.extensions.get("simulation_queue_manager")
    return cast(Optional[SimulationQueueManager], manager)

_ARTIFACT_CACHE: "OrderedDict[Tuple[str, int, int], Dict[str, Any]]" = OrderedDict()
_ARTIFACT_CACHE_MAX = 2

def _read_artifact_cached(artifact_path: str) -> Optional[Dict[str, Any]]:
    path = Path(artifact_path)
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    cached = _ARTIFACT_CACHE.get(key)
    if cached is not None:
        _ARTIFACT_CACHE.move_to_end(key)
        return cached
    artifact = read_artifact(artifact_path)
    _ARTIFACT_CACHE[key] = artifact
    _ARTIFACT_CACHE.move_to_end(key)
    while len(_ARTIFACT_CACHE) > _ARTIFACT_CACHE_MAX:
        _ARTIFACT_CACHE.popitem(last=False)
    return artifact

def _load_artifact(project_id: str) -> Optional[Dict[str, Any]]:
    manager = _queue_manager()
    if manager is not None:
        cached = manager.get_result(project_id)
        if cached is not None:
            return cached
    artifact_path = repo.get_artifact_path(_database_path(), project_id)
    if artifact_path is None or not Path(artifact_path).exists():
        return None
    return _read_artifact_cached(artifact_path)

@tools_bp.route("/tools/health")
def tools_health():
    return jsonify({"ok": True, "blueprint": "web_tools"})

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
    database_path = _database_path()
    source = repo.get_project(database_path, project_id)
    if source is None:
        return jsonify({"error": "Project not found"}), 404
    requested = str((request.json or {}).get("name", "")).strip()
    name = requested or f"{source.get('name', 'Project')} (copy)"

    upload_folder = _upload_folder()
    upload_folder.mkdir(parents=True, exist_ok=True)

    def copy_into_uploads(stored: str) -> Optional[str]:
        if not stored:
            return None
        try:
            original = Path(repo.resolve_upload_path(stored, upload_folder))
        except repo.UnsafePartFileError:
            return None
        if not original.name or not original.is_file():
            return None
        copy_path = upload_folder / f"{os.urandom(4).hex()}_{secure_filename(original.name)}"
        try:
            shutil.copy2(original, copy_path)
        except OSError:
            return None
        return str(copy_path)

    filepath_map: Dict[str, str] = {}
    base_map: Dict[str, str] = {}
    for part in repo.get_parts(database_path, project_id):
        copied = copy_into_uploads(str(part.get("filepath") or ""))
        if copied:
            filepath_map[str(part["id"])] = copied
        copied_base = copy_into_uploads(str(part.get("base_filepath") or ""))
        if copied_base:
            base_map[str(part["id"])] = copied_base

    new_id = repo.duplicate_project(database_path, project_id, name, filepath_map, base_map)
    if new_id is None:
        return jsonify({"error": "Project not found"}), 404
    return jsonify({"id": new_id, "project": repo.get_project(database_path, new_id)}), 201

@tools_bp.route("/projects/<project_id>/log")
def project_log(project_id: str):
    try:
        since = max(0, int(request.args.get("since", 0)))
    except (TypeError, ValueError):
        since = 0

    manager = _queue_manager()
    payload = manager.get_log(project_id, since) if manager is not None else None
    if payload is None:
        return jsonify({"lines": [], "next": 0, "dropped": 0, "available": False})
    return jsonify({**payload, "available": True})

@tools_bp.route("/hardware")
def hardware():
    payload: Dict[str, Any] = {
        "cpu_brand": "Unknown",
        "cpu_vendor": "Unknown",
        "physical_cores": None,
        "logical_cores": None,
        "performance_cores": None,
        "efficiency_cores": None,
        "core_tiers": [],
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
                "core_tiers": [
                    {"name": str(tier.get("name", "")), "cores": int(tier.get("cores", 0))}
                    for tier in getattr(config, "core_tiers", []) or []
                ],
                "total_memory_gb": round(config.total_memory / (1024 ** 3), 1),
                "available_memory_gb": round(config.available_memory / (1024 ** 3), 1),
                "numba_threads": config.get_optimal_thread_count("compute"),
                "gpu_available": bool(config.should_use_gpu()),
                "metal_available": bool(config.metal_available),
                "neural_engine_available": bool(config.neural_engine_available),
            }
        )
    except Exception as exc:
        payload["detection_error"] = str(exc)

    try:
        from core.gpu_compute import get_gpu_compute

        backend = get_gpu_compute().backend
        payload["gpu_backend"] = str(getattr(backend, "value", backend))
    except Exception:
        payload["gpu_backend"] = "none"

    return jsonify(payload)

@tools_bp.route("/projects/<project_id>/export")
def export_project_file(project_id: str):
    from flask import send_file
    import io
    import json

    from ui.project_state import PROJECT_SCHEMA_VERSION

    database_path = _database_path()
    project = repo.get_project(database_path, project_id)
    if project is None:
        return jsonify({"error": "Project not found"}), 404
    parts = repo.get_parts(database_path, project_id)

    from datetime import datetime, timezone

    payload = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "simulation": {
            "alloy_name": project.get("alloy", "WCB"),
            "voxel_size": float(project.get("voxel_size", 0.1)),
            "refinement_level": int(project.get("refinement_level", 3)),
            "memory_mode": project.get("memory_mode", "auto"),
            "analysis_options": project.get("analysis_options", {}),
            "performance_options": project.get("performance_options", {}),
            "solidification_params": project.get("solidification_params", {}),
            "pour_location": project.get("pour_location"),
            "mesh_setting_origin": "manual",
            "current_vis_mode": "shrinkage",
        },
        "ui_state": {
            key: value for key, value in (project.get("ui_state") or {}).items() if key != "pour_location"
        },
        "parts": [
            {
                "id": part["id"],
                "name": part["name"],
                "filepath": repo.strip_upload_prefix(os.path.basename(str(part["filepath"]))),
                "part_type": part["part_type"],
                "position": [part["position_x"], part["position_y"], part["position_z"]],
                "rotation": [part["rotation_x"], part["rotation_y"], part["rotation_z"]],
                "scale": part["scale"],
                "color": [0.7, 0.7, 0.7],
                "visible": bool(part["visible"]),
                "is_open_top": bool(part["is_open_top"]),
                "riser_sleeve": part.get("riser_sleeve", "insulating"),
                **({"feature": part["feature"]} if part.get("feature") else {}),
            }
            for part in parts
        ],
    }

    safe_name = "".join(
        char if char.isalnum() or char in "-_" else "_" for char in str(project.get("name", "project"))
    ).strip("_") or "project"
    data = json.dumps(payload, indent=2).encode("utf-8")
    return send_file(
        io.BytesIO(data),
        mimetype="application/json",
        as_attachment=True,
        download_name=f"{safe_name}.solidus-project.json",
    )

@tools_bp.route("/projects/import", methods=["POST"])
def import_project_file():
    import json

    from ui.project_state import load_project_state
    from web.payloads import upload_cad_file

    upload = request.files.get("project")
    if upload is None:
        return jsonify({"error": "Attach a .solidus-project.json file."}), 400

    head = upload.read(MAX_PROJECT_FILE_BYTES + 1)
    if len(head) > MAX_PROJECT_FILE_BYTES:
        return jsonify(
            {"error": f"The project file is larger than {MAX_PROJECT_FILE_BYTES // (1024 * 1024)} MB."}
        ), 413
    try:
        payload = json.loads(head.decode("utf-8"))
    except Exception:
        return jsonify({"error": "That file is not valid JSON."}), 400
    if not isinstance(payload, dict):
        return jsonify({"error": "A project file must contain a JSON object."}), 400

    try:
        state = load_project_state(payload).to_dict()
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"That project file could not be read: {exc}"}), 400

    simulation = state.get("simulation", {})
    if not isinstance(simulation, dict):
        return jsonify({"error": "Project file simulation settings must be an object."}), 400
    ui_state = state.get("ui_state", {})
    ui_state = dict(ui_state) if isinstance(ui_state, dict) else {}

    stored: Dict[str, Dict[str, Any]] = {}
    collisions: List[str] = []
    for file in request.files.getlist("files"):
        if not file.filename:
            continue
        try:
            for entry in upload_cad_file(
                file, _upload_folder(), simulation.get("voxel_size")
            ):
                key = str(entry.get("display_name", "")).lower()
                if not key:
                    continue
                if key in stored:
                    collisions.append(entry.get("display_name", key))
                    continue
                stored[key] = entry
        except ValueError as exc:
            return jsonify({"error": f"{file.filename}: {exc}"}), 400

    project_id = repo.create_project(
        _database_path(),
        {
            "name": str(payload.get("name") or upload.filename or "Imported project").replace(
                ".solidus-project.json", ""
            ),
            "alloy": simulation.get("alloy_name", "WCB"),
            "voxel_size": simulation.get("voxel_size", 0.1),
            "refinement_level": simulation.get("refinement_level", 3),
            "memory_mode": simulation.get("memory_mode", "auto"),
            "analysis_options": simulation.get("analysis_options", {}),
            "performance_options": simulation.get("performance_options", {}),
            "solidification_params": simulation.get("solidification_params", {}),
            "ui_state": ui_state,
            "pour_location": simulation.get("pour_location"),
        },
    )

    missing: List[str] = []
    id_map: Dict[str, str] = {}
    raw_parts = payload.get("parts") if isinstance(payload.get("parts"), list) else []
    for index, part in enumerate(state.get("parts", [])):
        if not isinstance(part, dict):
            continue
        raw_part = raw_parts[index] if index < len(raw_parts) and isinstance(raw_parts[index], dict) else {}
        raw_feature = raw_part.get("feature")
        feature = raw_feature if isinstance(raw_feature, dict) else None
        stem = os.path.basename(str(part.get("filepath", "")))
        candidates = [
            os.path.splitext(stem)[0].lower(),
            stem.lower(),
            repo.strip_upload_prefix(os.path.splitext(stem)[0]).lower(),
            repo.strip_upload_prefix(stem).lower(),
        ]
        entry = next((stored[key] for key in candidates if key in stored), None)
        if entry is None:
            missing.append(part.get("name") or stem)
            continue
        position = _triple(part.get("position"))
        rotation = _triple(part.get("rotation"))
        new_part_id = repo.add_part(
            _database_path(),
            project_id,
            {
                "name": part.get("name") or entry.get("display_name") or stem,
                "part_type": part.get("part_type", "cast"),
                "filepath": entry["filepath"],
                "position_x": position[0],
                "position_y": position[1],
                "position_z": position[2],
                "rotation_x": rotation[0],
                "rotation_y": rotation[1],
                "rotation_z": rotation[2],
                "scale": part.get("scale", 1.0),
                "visible": part.get("visible", True),
                "is_open_top": part.get("is_open_top", False),
                "riser_sleeve": part.get("riser_sleeve", "insulating"),
            },
            upload_root=_upload_folder(),
            feature=feature,
        )
        if isinstance(raw_part.get("id"), str):
            id_map[raw_part["id"]] = new_part_id

    if id_map:
        for imported in repo.get_parts(_database_path(), project_id):
            feature = imported.get("feature")
            remapped = repo.remap_feature_part_ids(feature, id_map)
            if feature and remapped != feature:
                repo.set_part_feature(_database_path(), project_id, str(imported["id"]), remapped or {})
    if "cut_features" in ui_state:
        cuts = repo.remap_cut_part_ids(ui_state, id_map).get("cut_features")
        repo.set_ui_state_key(_database_path(), project_id, "cut_features", cuts)

    return jsonify(
        {
            "id": project_id,
            "project": repo.get_project(_database_path(), project_id),
            "missing_parts": missing,
            "name_collisions": collisions,
            "schema_version": state.get("schema_version"),
        }
    ), 201

def _pdf_backend() -> Optional[str]:
    for module in ("weasyprint", "pdfkit"):
        try:
            __import__(module)
            return module
        except Exception:
            continue
    return None

REPORT_FORMATS = ("html", "markdown", "pdf")

@tools_bp.route("/report/formats")
def report_formats():
    backend = _pdf_backend()
    return jsonify(
        {
            "formats": [
                {"id": "html", "label": "HTML", "available": True, "reason": None},
                {"id": "markdown", "label": "Markdown", "available": True, "reason": None},
                {
                    "id": "pdf",
                    "label": "PDF",
                    "available": backend is not None,
                    "reason": None
                    if backend is not None
                    else "PDF export needs weasyprint or pdfkit installed on the server.",
                },
            ]
        }
    )

def _report_settings_text(project: Dict[str, Any]) -> str:
    analysis = project.get("analysis_options", {}) or {}
    enabled = sorted(key for key, value in analysis.items() if value is True)
    performance = project.get("performance_options", {}) or {}
    lines = [
        f"Alloy: {project.get('alloy', '')}",
        f"Base voxel size: {project.get('voxel_size', 0)} in",
        f"Refinement level: {project.get('refinement_level', 0)}",
        f"Memory mode: {project.get('memory_mode', 'auto')}",
        f"GPU acceleration: {'on' if performance.get('use_gpu') else 'off'}",
        f"Analyses run: {', '.join(enabled) if enabled else 'none'}",
        f"Sand restraint index: {analysis.get('sand_restraint_index', '--')}",
    ]
    pour = project.get("pour_location")
    if pour and pour.get("point_in"):
        point = ", ".join(f"{float(value):.2f}" for value in pour["point_in"])
        lines.append(f"Pour location: ({point}) in")
    return "\n".join(lines)

@tools_bp.route("/projects/<project_id>/report", methods=["POST"])
def project_report(project_id: str):
    from flask import send_file

    database_path = _database_path()
    project = repo.get_project(database_path, project_id)
    if project is None:
        return jsonify({"error": "Project not found"}), 404

    payload = request.json if isinstance(request.json, dict) else None
    if payload is None:
        return jsonify({"error": "The request body must be a JSON object."}), 400
    fmt = str(payload.get("format", "html")).lower()
    if fmt not in REPORT_FORMATS:
        return jsonify({"error": f"Unknown report format '{fmt}'."}), 400
    if fmt == "pdf" and _pdf_backend() is None:
        return jsonify(
            {"error": "PDF export needs weasyprint or pdfkit installed on the server."}
        ), 503

    sections = payload.get("sections")
    if sections is None:
        sections = {}
    if not isinstance(sections, dict):
        return jsonify({"error": "`sections` must be an object of booleans."}), 400

    def want(key: str) -> bool:
        return bool(sections.get(key, True))

    from core.report_generator import (
        MetricsData,
        OutputFormat,
        ReportConfig,
        ReportGenerator,
        TableData,
    )

    output_format = {
        "html": OutputFormat.HTML,
        "markdown": OutputFormat.MARKDOWN,
        "pdf": OutputFormat.PDF,
    }[fmt]
    generator = ReportGenerator(
        ReportConfig(
            output_format=output_format,
            title=f"{project.get('name', 'Casting')} — Simulation Report",
            company="Solidus Pro",
        )
    )

    if want("settings"):
        generator.add_text_section("Simulation Setup", _report_settings_text(project))

    if want("parts"):
        parts = repo.get_parts(database_path, project_id)
        generator.add_table_section(
            "Parts",
            TableData(
                headers=["Name", "Type", "Position (in)", "Rotation (deg)", "Scale"],
                rows=[
                    [
                        part["name"],
                        part["part_type"],
                        f"{part['position_x']:.2f}, {part['position_y']:.2f}, {part['position_z']:.2f}",
                        f"{part['rotation_x']:.0f}, {part['rotation_y']:.0f}, {part['rotation_z']:.0f}",
                        f"{part['scale']:.2f}",
                    ]
                    for part in parts
                ],
                caption=f"{len(parts)} part(s) in the assembly",
            ),
        )

    if want("statistics"):
        summary = project.get("summary") or {}
        if not summary:
            summary = (_load_artifact(project_id) or {}).get("summary", {}) or {}
        metrics = {
            key: float(value)
            for key, value in _flat_summary(summary).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if metrics:
            generator.add_metrics_section("Summary Statistics", MetricsData(metrics=metrics))
        else:
            generator.add_text_section(
                "Summary Statistics", "This project has no completed run to report on yet."
            )

    screenshot = payload.get("screenshot")
    if want("screenshot") and isinstance(screenshot, str) and screenshot:
        image = _decode_data_url(screenshot)
        if image is None:
            return jsonify({"error": "The screenshot could not be decoded."}), 400
        if len(image) > MAX_SCREENSHOT_BYTES:
            return jsonify(
                {"error": f"The screenshot exceeds {MAX_SCREENSHOT_BYTES // (1024 * 1024)} MB."}
            ), 413
        if not image.startswith(PNG_MAGIC):
            return jsonify({"error": "The screenshot must be a PNG."}), 400
        generator.add_image_section("Result View", image, caption="Viewport screenshot")

    import tempfile

    suffix = {"html": ".html", "markdown": ".md", "pdf": ".pdf"}[fmt]
    with tempfile.TemporaryDirectory() as temp_dir:
        target = Path(temp_dir) / f"report{suffix}"
        try:
            generator.generate(target)
        except Exception as exc:
            return jsonify({"error": f"Report generation failed: {exc}"}), 500
        if not target.exists():
            return jsonify(
                {"error": "The server could not render that format. Try HTML or Markdown."}
            ), 503
        data = target.read_bytes()

    safe_name = "".join(
        char if char.isalnum() or char in "-_" else "_" for char in str(project.get("name", "report"))
    ).strip("_") or "report"
    mimetype = {
        "html": "text/html",
        "markdown": "text/markdown",
        "pdf": "application/pdf",
    }[fmt]
    import io

    return send_file(
        io.BytesIO(data),
        mimetype=mimetype,
        as_attachment=True,
        download_name=f"{safe_name}_report{suffix}",
    )

def _flat_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in summary.items():
        if key in {"raw_ranges", "available_modes", "bounds"}:
            continue
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float, str, bool)):
                    flat[f"{key.removesuffix('_stats')}_{sub_key}"] = sub_value
        elif isinstance(value, (int, float, str, bool)):
            flat[key] = value
    return flat

def _decode_data_url(value: str) -> Optional[bytes]:
    import base64

    if len(value) > MAX_SCREENSHOT_BYTES * 2:
        return None
    encoded = value.split(",", 1)[1] if value.startswith("data:") else value
    try:
        return base64.b64decode(encoded, validate=True)
    except Exception:
        return None

@tools_bp.route("/help/alloys")
def help_alloys():
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
