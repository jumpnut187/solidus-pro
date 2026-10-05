"""Application-shell routes: project management extras, riser design, reports,
hardware info, job logs, project file import/export.

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

import numpy as np
from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from core.materials import get_all_alloy_names, get_alloy
from web import repository as repo
from web.payloads import _sanitize_body_name, read_artifact
from web.services.simulation_queue import SimulationQueueManager

#: Upper bound on a single riser-apply request, so one call cannot fill the
#: upload folder with cylinders.
MAX_RISERS_PER_APPLY = 20

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
#: MB of JSON and both riser design and the report re-read one per request; two
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
# Riser design
# ---------------------------------------------------------------------------

#: The desktop dialog's hidden constants (``ui/main_window.py`` ``_run_analysis``).
#: They are not user inputs there and are not exposed here either, but the
#: engine's own defaults for the two diameters are metric while this whole
#: application works in inches -- passing them through unchanged would design
#: risers a fiftieth of the intended size.
RISER_SAFETY_FACTOR = 1.15
RISER_MIN_DIAMETER_IN = 0.5
RISER_MAX_DIAMETER_IN = 12.0

#: Above this voxel count the analysis is refused rather than run inline.
#: ``HeuversCircleMethod`` is O(interior x surface), so a 40k-voxel run takes
#: about three seconds and the cost climbs quadratically from there; a request
#: that would tie up a worker for minutes should fail fast and say why.
RISER_MAX_VOXELS = 150_000


def _riser_inputs(artifact: Dict[str, Any]) -> Tuple[Any, float, Any]:
    """Pull the three arrays ``RiserDesigner`` actually reads out of an artifact.

    The engine's constructor also takes ``neighbors`` and ``alloy``, but it
    stores both without ever reading them, so this does not fabricate values
    for them.

    Raises:
        ValueError: when the artifact predates the surface mask, or when the
            visualization payload was strided down (which punches holes in the
            surface set and inflates every modulus).
    """
    visualization = artifact.get("visualization", {})
    centers = np.asarray(visualization.get("centers", []), dtype=np.float64)
    if centers.ndim != 2 or centers.shape[0] == 0:
        raise ValueError("This run has no voxel data to analyze.")

    surface = visualization.get("surface_voxels")
    if surface is None:
        raise ValueError(
            "This run was saved without a surface mask, which riser design needs. Re-run the simulation."
        )
    surface_mask = np.asarray(surface, dtype=bool)
    if surface_mask.shape[0] != centers.shape[0]:
        raise ValueError("The stored surface mask does not match the voxel data.")

    stored_count = artifact.get("summary", {}).get("voxel_count")
    if isinstance(stored_count, (int, float)) and int(stored_count) > centers.shape[0]:
        # Refusing beats answering wrongly: a strided result punches holes in
        # the surface set, so distance-to-surface -- which *is* the modulus --
        # comes out too large and every riser is over-sized. Name both ways out,
        # because "re-run" alone does not say what to change first.
        raise ValueError(
            f"This run stored {int(stored_count):,} voxels as {centers.shape[0]:,}, so riser design "
            "would over-estimate every modulus and propose over-sized risers. Either raise "
            "'Voxels kept in the stored result' in the Performance panel and run again, or run again "
            "at a coarser voxel size so the whole result fits under the cap."
        )

    sizes = np.asarray(visualization.get("sizes", []), dtype=np.float64)
    voxel_size = float(np.min(sizes)) if sizes.size else 0.0
    if not voxel_size > 0:
        raise ValueError("This run has no usable voxel size.")

    return centers, voxel_size, surface_mask


def _dedupe_recommendations(recommendations: List[Any], voxel_size: float) -> Tuple[List[Any], int]:
    """Collapse risers the engine placed on the same attachment point.

    ``RiserDesigner`` snaps every riser to the topmost few voxel layers and then
    picks the nearest point in x-y only, so on a geometry whose hot spots all
    sit under one flat top face -- a plate with a single feeder, say -- it
    happily returns five identical risers. Offering the user five copies of one
    riser to tick is not a recommendation, it is noise, so near-coincident
    entries are merged and the strongest of each group kept.

    Returns the surviving recommendations and how many were merged away.
    """
    tolerance = max(voxel_size, 1e-6)
    kept: List[Any] = []
    for candidate in sorted(recommendations, key=lambda item: -float(item.confidence)):
        duplicate = any(
            all(
                abs(float(candidate.position[axis]) - float(existing.position[axis])) <= tolerance
                for axis in range(3)
            )
            and abs(float(candidate.diameter) - float(existing.diameter)) <= tolerance
            for existing in kept
        )
        if not duplicate:
            kept.append(candidate)
    return kept, len(recommendations) - len(kept)


def _serialize_recommendation(index: int, recommendation: Any) -> Dict[str, Any]:
    """One recommendation as JSON, in the units the desktop dialog displays."""
    position = [float(value) for value in recommendation.position]
    return {
        "index": index,
        "riser_type": recommendation.riser_type.name,
        "shape": recommendation.shape.name,
        "position": position,
        "diameter_in": float(recommendation.diameter),
        "height_in": float(recommendation.height),
        "volume_in3": float(recommendation.volume),
        "surface_area_in2": float(recommendation.surface_area),
        "modulus_in": float(recommendation.modulus),
        "voxels_fed": int(len(recommendation.feeding_zone)),
        "confidence": float(recommendation.confidence),
        "rationale": str(recommendation.rationale),
    }


@tools_bp.route("/projects/<project_id>/riser-design", methods=["POST"])
def riser_design(project_id: str):
    """Analyze a completed run's geometry and propose risers.

    Runs against the stored artifact rather than re-voxelizing: the designer
    needs voxel centers, a voxel size and a surface mask, all of which the
    artifact already carries. Note this analyzes the whole assembly, including
    any risers already present -- the desktop dialog voxelizes the casting body
    alone, and the two will not agree on a rigged model.
    """
    artifact = _load_artifact(project_id)
    if artifact is None:
        return jsonify({"error": "This project has no results yet. Run a simulation first."}), 404

    payload = request.json or {}
    try:
        max_risers = max(1, min(10, int(payload.get("max_risers", 5))))
        modulus_factor = max(1.0, min(2.0, float(payload.get("modulus_factor", 1.2))))
        feeding_constant = max(2.0, min(6.0, float(payload.get("feeding_constant", 4.5))))
    except (TypeError, ValueError):
        return jsonify({"error": "Riser design parameters must be numbers."}), 400

    try:
        centers, voxel_size, surface_mask = _riser_inputs(artifact)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if centers.shape[0] > RISER_MAX_VOXELS:
        return jsonify(
            {
                "error": (
                    f"This run has {centers.shape[0]:,} voxels, above the "
                    f"{RISER_MAX_VOXELS:,} riser-design limit. Re-run at a coarser voxel size to "
                    "design risers."
                )
            }
        ), 400

    from core.riser_design import RiserDesigner, RiserDesignParameters

    designer = RiserDesigner(
        voxel_positions=centers,
        voxel_size=voxel_size,
        neighbors=None,
        surface_mask=surface_mask,
        alloy=None,
        params=RiserDesignParameters(
            modulus_factor=modulus_factor,
            feeding_constant=feeding_constant,
            safety_factor=RISER_SAFETY_FACTOR,
            min_diameter=RISER_MIN_DIAMETER_IN,
            max_diameter=RISER_MAX_DIAMETER_IN,
        ),
    )
    raw_recommendations = designer.design_risers(max_risers=max_risers)
    # Coverage is read before deduplication: it is the union of feeding zones,
    # and duplicates contribute nothing to a union, so the figure is unchanged
    # by the merge and stays comparable with the desktop's.
    coverage = designer.analyze_coverage()
    recommendations, merged = _dedupe_recommendations(raw_recommendations, voxel_size)

    return jsonify(
        {
            "recommendations": [
                _serialize_recommendation(index, item) for index, item in enumerate(recommendations)
            ],
            "merged_duplicates": merged,
            # analyze_coverage returns only two keys when nothing was designed,
            # so every optional key is defaulted rather than indexed.
            "coverage": {
                "covered_fraction": float(coverage.get("covered_fraction", 0.0)),
                "covered_voxels": int(coverage.get("covered_voxels", 0)),
                "uncovered_voxels": int(coverage.get("uncovered_voxels", 0)),
                "n_risers": int(coverage.get("n_risers", 0)),
                "avg_riser_diameter_in": float(coverage.get("avg_riser_diameter", 0.0)),
                "total_riser_volume_in3": float(coverage.get("total_riser_volume", 0.0)),
            },
            "voxel_count": int(centers.shape[0]),
            "voxel_size_in": voxel_size,
        }
    )


@tools_bp.route("/projects/<project_id>/riser-design/apply", methods=["POST"])
def riser_design_apply(project_id: str):
    """Materialise chosen recommendations as riser parts.

    Each becomes a cylinder STL in the upload folder and a ``riser`` row on the
    project, matching the desktop: 32 sections, and the base sunk 40 percent of
    a diameter into the casting so the riser and the cast body actually
    intersect instead of touching at a plane.
    """
    if repo.get_project(_database_path(), project_id) is None:
        return jsonify({"error": "Project not found"}), 404

    payload = request.json or {}
    raw = payload.get("risers")
    if not isinstance(raw, list) or not raw:
        return jsonify({"error": "Select at least one riser to add."}), 400

    if len(raw) > MAX_RISERS_PER_APPLY:
        return jsonify({"error": f"At most {MAX_RISERS_PER_APPLY} risers can be added at once."}), 400

    # Validate every entry before writing anything. The previous version
    # checked inside the write loop, so a bad third entry left the first two on
    # disk and in the database behind a 400 the client read as "nothing
    # happened".
    validated: List[Dict[str, Any]] = []
    for offset, item in enumerate(raw):
        if not isinstance(item, dict):
            return jsonify({"error": "Each riser must be an object."}), 400
        try:
            diameter = float(item["diameter_in"])
            height = float(item["height_in"])
            position = [float(value) for value in item["position"]]
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "Each riser needs a diameter, a height and a position."}), 400
        if len(position) != 3 or not all(math.isfinite(value) for value in position):
            return jsonify({"error": "A riser position needs three finite coordinates."}), 400
        if not (math.isfinite(diameter) and math.isfinite(height)):
            return jsonify({"error": "Riser dimensions must be finite."}), 400
        if not (0 < diameter <= RISER_MAX_DIAMETER_IN * 4 and 0 < height <= RISER_MAX_DIAMETER_IN * 12):
            return jsonify({"error": "Riser dimensions are outside the buildable range."}), 400

        type_name = str(item.get("riser_type", "BLIND"))[:32]
        # The display name is user text and is stored as such; it never reaches
        # the filesystem. The on-disk name is derived below, so a name like
        # "../../../etc/cron.d/x" cannot steer the export out of the folder.
        #
        # Note `_sanitize_body_name("")` returns "body", so the default has to
        # be chosen before sanitizing rather than as an `or` fallback after it.
        requested = str(item.get("name") or "").strip()
        display_name = (
            _sanitize_body_name(requested)
            if requested
            else f"AutoRiser_{offset + 1}_{_sanitize_body_name(type_name)}"
        )
        validated.append(
            {
                "diameter": diameter,
                "height": height,
                "position": position,
                "type_name": type_name,
                "name": display_name[:120],
            }
        )

    import trimesh

    upload_folder = _upload_folder()
    upload_folder.mkdir(parents=True, exist_ok=True)
    added: List[Dict[str, Any]] = []
    written: List[Path] = []

    try:
        for offset, spec in enumerate(validated):
            mesh = trimesh.creation.cylinder(
                radius=spec["diameter"] / 2.0, height=spec["height"], sections=32
            )
            # trimesh centres a cylinder on the origin, so lift it by half its
            # height to sit on the attachment point, then sink the penetration.
            penetration = spec["diameter"] * 0.40
            position = spec["position"]
            mesh.apply_translation(
                [position[0], position[1], position[2] - penetration + spec["height"] / 2.0]
            )

            filepath = upload_folder / f"{os.urandom(4).hex()}_b0_riser_{offset + 1}.stl"
            # Belt and braces: the name above contains nothing but hex digits
            # and an index, and this asserts that stays true.
            repo.resolve_upload_path(str(filepath), upload_folder)
            mesh.export(str(filepath))
            written.append(filepath)

            part_id = repo.add_part(
                _database_path(),
                project_id,
                {
                    "name": spec["name"],
                    "part_type": "riser",
                    "filepath": str(filepath),
                    "visible": True,
                    "is_open_top": spec["type_name"].upper() == "OPEN",
                },
                upload_root=upload_folder,
            )
            part = repo.get_part(_database_path(), project_id, part_id)
            if part is not None:
                added.append(part)
    except Exception as exc:
        # Roll back so a half-applied set does not leave orphan parts and STLs.
        for part in added:
            repo.delete_part(_database_path(), project_id, str(part["id"]))
        for path in written:
            path.unlink(missing_ok=True)
        current_app.logger.exception("Riser apply failed")
        return jsonify({"error": f"Could not add the risers: {exc}"}), 500

    return jsonify({"parts": added, "added": len(added)}), 201


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
            for entry in upload_cad_file(file, _upload_folder()):
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
