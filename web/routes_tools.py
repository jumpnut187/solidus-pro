"""Application-shell routes: project management extras, reports, hardware info,
job logs, project file import/export.

Owned by the shell/parity workstream. Registered by
``web.app_factory.create_app``. Everything here lives under ``/api``.
"""

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

#: A project file is settings, not geometry. The app-wide 500 MB limit exists
#: for STEP assemblies and is far too generous for a JSON blob read into memory.
MAX_PROJECT_FILE_BYTES = 4 * 1024 * 1024

#: Cap on a client-supplied report screenshot, decoded.
MAX_SCREENSHOT_BYTES = 10 * 1024 * 1024

#: PNG signature. The report renderer embeds images under a hardcoded
#: ``data:image/png`` URL, so anything else would be mislabelled.
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _triple(value: Any) -> List[float]:
    """Three finite floats from untrusted JSON, defaulting to the origin."""
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


#: Parsed artifacts, keyed by (path, mtime_ns, size). Artifacts run to 50-180
#: MB of JSON and the report re-reads one per request; two
#: entries is enough to keep an interactive session off the disk while bounding
#: the memory a web worker holds.
_ARTIFACT_CACHE: "OrderedDict[Tuple[str, int, int], Dict[str, Any]]" = OrderedDict()
_ARTIFACT_CACHE_MAX = 2


def _read_artifact_cached(artifact_path: str) -> Optional[Dict[str, Any]]:
    """Parse an artifact, reusing the last few parses.

    The key includes mtime and size so a re-run's artifact is never served from
    a stale entry -- the path alone is reused for every run of a project.
    """
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
    """The completed run's artifact, from the queue cache or from disk.

    Mirrors ``web.routes._load_artifact``; duplicated rather than imported so a
    change to the legacy blueprint's private helper cannot silently alter these
    endpoints.
    """
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

    def copy_into_uploads(stored: str) -> Optional[str]:
        if not stored:
            return None
        try:
            # Rows written before the path check landed may still name a file
            # anywhere on the filesystem. Copying one into the upload folder
            # would publish it through /api/uploads, so refuse instead: this
            # endpoint must not be a way to launder a path past the check.
            original = Path(repo.resolve_upload_path(stored, upload_folder))
        except repo.UnsafePartFileError:
            return None
        if not original.name or not original.is_file():
            return None
        copy_path = upload_folder / f"{os.urandom(4).hex()}_{secure_filename(original.name)}"
        try:
            shutil.copy2(original, copy_path)
        except OSError:
            # A missing or unreadable source file should not abort the whole
            # duplicate -- the copied row keeps pointing at the original.
            return None
        return str(copy_path)

    filepath_map: Dict[str, str] = {}
    # A body an extrude cut has cut also keeps its pre-cut STL, which the copy
    # needs for its cuts to stay editable.
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


# ---------------------------------------------------------------------------
# Job log
# ---------------------------------------------------------------------------


@tools_bp.route("/projects/<project_id>/log")
def project_log(project_id: str):
    """Progress lines for a run, live while it works and after it finishes.

    ``since`` is a monotonic sequence cursor, not an index into the lines
    returned: the buffer is a ring, so once it wraps an index would point at
    the wrong line (or at nothing, forever). Pass back the ``next`` value from
    the previous response.
    """
    try:
        since = max(0, int(request.args.get("since", 0)))
    except (TypeError, ValueError):
        since = 0

    manager = _queue_manager()
    payload = manager.get_log(project_id, since) if manager is not None else None
    if payload is None:
        # Not an error: a project that has never run simply has no log.
        return jsonify({"lines": [], "next": 0, "dropped": 0, "available": False})
    return jsonify({**payload, "available": True})


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
                # `performance_cores`/`efficiency_cores` assume perflevel0 is
                # the fast tier, which is wrong on chips whose tiers are named
                # otherwise; `core_tiers` carries the OS's own names and is
                # what the UI shows. See config.hardware_config for why the
                # older pair is not simply corrected.
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
# Project file import and export (desktop-compatible)
# ---------------------------------------------------------------------------
#
# The desktop format (``ui/project_state.py``, schema v2) and the web project
# row disagree in three ways, and every one of them is a silent data-loss trap
# if translated carelessly:
#
#   * the desktop nests settings under ``simulation`` and calls the alloy
#     ``alloy_name``; the web row is flat and calls it ``alloy``. Feeding
#     ``alloy_name`` straight to the pipeline yields WCB whatever the file said.
#   * the desktop keeps ``pour_location`` in ``simulation``; the web keeps it
#     inside the ``ui_state`` blob and mirrors it to a top-level key on read.
#   * part paths in the file are relative to the file's own directory, which
#     means nothing on a server. Imports match them to uploaded blobs by name.


@tools_bp.route("/projects/<project_id>/export")
def export_project_file(project_id: str):
    """Download a project as a desktop-compatible ``.solidus-project.json``."""
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
                # Lets an import point features and cuts that name a body at
                # that body's new row. The desktop loader ignores it.
                "id": part["id"],
                "name": part["name"],
                # The original file name, without the server's storage prefix
                # and without a path: an absolute server path means nothing on
                # another machine, and the prefix is not what the user's copy of
                # the file is called.
                "filepath": repo.strip_upload_prefix(os.path.basename(str(part["filepath"]))),
                "part_type": part["part_type"],
                "position": [part["position_x"], part["position_y"], part["position_z"]],
                "rotation": [part["rotation_x"], part["rotation_y"], part["rotation_z"]],
                "scale": part["scale"],
                # The web parts table has no colour column; the desktop schema
                # requires the key, so emit its own load-time default.
                "color": [0.7, 0.7, 0.7],
                "visible": bool(part["visible"]),
                "is_open_top": bool(part["is_open_top"]),
                # A body modeled in the browser carries the sketch and extrude
                # that made it, so it stays editable after a round trip. The
                # desktop loader ignores keys it does not know.
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
    """Create a project from a ``.solidus-project.json`` plus its CAD files.

    Multipart: ``project`` is the JSON file, ``files`` are the STL/STEP parts it
    references. Parts whose files were not supplied are reported back rather
    than silently dropped, so the user knows exactly what to re-attach.
    """
    import json

    from ui.project_state import load_project_state
    from web.payloads import upload_cad_file

    upload = request.files.get("project")
    if upload is None:
        return jsonify({"error": "Attach a .solidus-project.json file."}), 400

    # The app's MAX_CONTENT_LENGTH is 500 MB to admit large STEP assemblies; a
    # project file is a few kilobytes of settings, so it gets its own cap and
    # is parsed off the stream rather than read whole into memory first.
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
        # Schema v1 files are migrated forward here -- including the legacy
        # pour-rate repair, which nothing on the web side reproduces.
        # TypeError as well as ValueError: `int(payload["schema_version"])` on a
        # non-scalar raises the former, which used to escape as a 500.
        state = load_project_state(payload).to_dict()
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"That project file could not be read: {exc}"}), 400

    simulation = state.get("simulation", {})
    if not isinstance(simulation, dict):
        return jsonify({"error": "Project file simulation settings must be an object."}), 400
    ui_state = state.get("ui_state", {})
    ui_state = dict(ui_state) if isinstance(ui_state, dict) else {}

    # Store the uploaded CAD files, indexed by the display name each body
    # reports. Indexing by the *upload's* filename collapsed a multi-body STEP
    # onto whichever body came last, and let bracket.stl and bracket.step
    # overwrite one another; the display name is per body and is what the
    # project file's paths are matched against.
    stored: Dict[str, Dict[str, Any]] = {}
    collisions: List[str] = []
    for file in request.files.getlist("files"):
        if not file.filename:
            continue
        try:
            # Tessellate STEP bodies for the imported project's own voxel
            # size; upload_voxel_size_in falls back to the default when the
            # saved value is missing or invalid.
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
            # create_project folds this into ui_state for us.
            "pour_location": simulation.get("pour_location"),
        },
    )

    missing: List[str] = []
    # Exported part id -> imported part id, for features and cuts that name a body.
    id_map: Dict[str, str] = {}
    # `load_project_state` keeps the desktop schema only, so a modeled body's
    # feature definition is read from the raw entry at the same index (the
    # loader keeps every entry, in order, or raises).
    raw_parts = payload.get("parts") if isinstance(payload.get("parts"), list) else []
    for index, part in enumerate(state.get("parts", [])):
        if not isinstance(part, dict):
            continue
        raw_part = raw_parts[index] if index < len(raw_parts) and isinstance(raw_parts[index], dict) else {}
        raw_feature = raw_part.get("feature")
        feature = raw_feature if isinstance(raw_feature, dict) else None
        # A project file may name a part by its original file name, by the
        # server's prefixed storage name (an export from an older build), with
        # or without an extension. Try all four rather than making the user
        # guess which one the importer wants.
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
            },
            upload_root=_upload_folder(),
            feature=feature,
        )
        if isinstance(raw_part.get("id"), str):
            id_map[raw_part["id"]] = new_part_id

    # Point what names a body at the imported rows. An imported body has no
    # pre-cut base: the file it was imported from is taken as its base, and
    # Rebuild in the Model panel applies the cuts to it again.
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
            # Bodies that shared a display name; only the first was kept, and
            # the user needs to know the rest were not silently applied.
            "name_collisions": collisions,
            "schema_version": state.get("schema_version"),
        }
    ), 201


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _pdf_backend() -> Optional[str]:
    """Which PDF renderer ``ReportGenerator`` would find, if any.

    ``ReportGenerator._html_to_pdf`` degrades *silently* when neither backend
    is installed: it writes a ``.html`` and a ``.txt`` beside the requested
    path, returns the ``.pdf`` path anyway, and raises nothing. Probing up
    front is what lets the UI grey the option out with a reason instead of
    handing the user a download that does not exist.
    """
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
    """Build a report for a completed run and return it as a download."""
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

    # `sections` used to be read with `.get` on whatever arrived, so a list or a
    # string turned into an AttributeError and a 500.
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
        # The project row carries the same summary the artifact does, recorded
        # by mark_project_completed. Prefer it: reading it avoids parsing a
        # 50-180 MB artifact whose visualization arrays this section never
        # touches. The artifact is only opened when the row has no summary.
        summary = project.get("summary") or {}
        if not summary:
            summary = (_load_artifact(project_id) or {}).get("summary", {}) or {}
        # add_metrics_section formats every value with `f"{value:.2f}"`, so a
        # string statistic (the Niyama metadata is full of them) would raise.
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
        # The renderer hardcodes `data:image/png`, so anything that is not a
        # PNG would be embedded under a lying media type.
        if not image.startswith(PNG_MAGIC):
            return jsonify({"error": "The screenshot must be a PNG."}), 400
        # `_process_image` treats a str as a FILE PATH, so the PNG has to be
        # handed over as bytes.
        generator.add_image_section("Result View", image, caption="Viewport screenshot")

    import tempfile

    suffix = {"html": ".html", "markdown": ".md", "pdf": ".pdf"}[fmt]
    with tempfile.TemporaryDirectory() as temp_dir:
        target = Path(temp_dir) / f"report{suffix}"
        try:
            generator.generate(target)
        except Exception as exc:  # pragma: no cover - depends on optional backends
            return jsonify({"error": f"Report generation failed: {exc}"}), 500
        if not target.exists():
            # The silent-fallback branch above; treat it as the failure it is.
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
    """One level of flattening, matching what the results panel shows."""
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
    """Decode a ``data:image/png;base64,...`` URL, or a bare base64 payload.

    Refuses anything longer than the screenshot cap before decoding, so a
    gigabyte of base64 is not expanded in memory just to be rejected after.
    """
    import base64

    if len(value) > MAX_SCREENSHOT_BYTES * 2:
        return None
    encoded = value.split(",", 1)[1] if value.startswith("data:") else value
    try:
        return base64.b64decode(encoded, validate=True)
    except Exception:
        return None


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
