"""HTTP routes for the React/Babylon web application."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, cast

from flask import Blueprint, current_app, jsonify, request, send_from_directory

from core.materials import get_all_alloy_names, get_alloy
from core.memory_limits import estimate_voxel_count, validate_mesh_parameters
from core.mesh_ml import MeshRecommendation, recommend_mesh_settings
from core.stl_loader import STLLoader
from web.services.simulation_queue import SimulationQueueManager
from web import repository as repo
from web.payloads import (
    build_fork_surface_payload,
    build_part_manager,
    build_result_fields_payload,
    build_volume_payload,
    load_part_mesh_data,
    read_artifact,
    upload_cad_file,
)


api_bp = Blueprint("web_api", __name__, url_prefix="/api")


def database_path() -> Path:
    return Path(current_app.config["DATABASE"])


def upload_folder() -> Path:
    return Path(current_app.config["UPLOAD_FOLDER"])


def queue_manager() -> SimulationQueueManager:
    manager = current_app.extensions.get("simulation_queue_manager")
    if manager is None:
        raise RuntimeError("Simulation queue manager is not initialized")
    return cast(SimulationQueueManager, manager)


def user_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return str(forwarded.split(",")[0].strip())
    return request.remote_addr or "unknown"


@api_bp.route("/health")
def health():
    return jsonify({"ok": True, "app": "web"})


@api_bp.route("/materials")
def materials():
    return jsonify([material_payload(name) for name in get_all_alloy_names()])


def material_payload(name: str) -> Dict[str, Any]:
    """Serialize one alloy for the Material panel.

    Carries every property the desktop Material dialog lists read-only, not
    just the handful the first web client needed: freezing range, pour
    temperature, both shrinkage figures, the thermal group (conductivity,
    specific heat, latent heat, diffusivity) and both densities. The web panel
    is meant to reach desktop parity, and a foundry engineer picking an alloy
    reads these numbers before committing to a four-minute run.
    """
    alloy = get_alloy(name)
    return {
        "name": name,
        "full_name": alloy.name,
        "liquidus_temp": float(alloy.liquidus_temp),
        "solidus_temp": float(alloy.solidus_temp),
        "freezing_range": float(alloy.freezing_range),
        "pouring_temp": float(alloy.pouring_temp),
        "volumetric_shrinkage": round(float(alloy.volumetric_shrinkage), 2),
        "linear_shrinkage": float(alloy.linear_shrinkage),
        "feeding_multiplier": float(alloy.feeding_distance_multiplier),
        "niyama_threshold": float(alloy.niyama_threshold),
        "hot_tear_susceptibility": float(getattr(alloy, "hot_tear_susceptibility", 0.0)),
        "sand_burn_in_susceptibility": float(getattr(alloy, "sand_burn_in_susceptibility", 0.0)),
        "gas_sensitivity": float(getattr(alloy, "gas_sensitivity", 0.0)),
        "thermal_conductivity": float(alloy.thermal_conductivity),
        "specific_heat": float(alloy.specific_heat),
        "latent_heat": float(alloy.latent_heat),
        "thermal_diffusivity": float(alloy.thermal_diffusivity),
        "density_liquid": float(alloy.density_liquid),
        "density_solid": float(alloy.density_solid),
        "critical_fraction_solid": float(alloy.critical_fraction_solid),
        "dendrite_arm_spacing": float(alloy.dendrite_arm_spacing),
        "surface_tension": float(getattr(alloy, "surface_tension", 0.0)),
    }


@api_bp.route("/uploads", methods=["POST"])
def upload_files():
    files = request.files.getlist("files") or request.files.getlist("file")
    if not files:
        return jsonify({"error": "No files provided"}), 400
    uploaded = []
    try:
        for file in files:
            # upload_cad_file returns one entry per body in the file.
            # Single-body files return a one-element list; multi-body
            # STEP assemblies return one entry per body. Flatten so the
            # frontend sees a uniform list of file entries.
            uploaded.extend(upload_cad_file(file, upload_folder()))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"files": uploaded})


@api_bp.route("/projects", methods=["GET"])
def list_projects():
    return jsonify(repo.list_projects(database_path()))


@api_bp.route("/projects", methods=["POST"])
def create_project():
    project_id = repo.create_project(database_path(), request.json or {})
    project = repo.get_project(database_path(), project_id)
    return jsonify({"id": project_id, "project": project}), 201


@api_bp.route("/projects/<project_id>", methods=["GET"])
def get_project(project_id: str):
    project = repo.get_project(database_path(), project_id)
    if project is None:
        return jsonify({"error": "Project not found"}), 404
    parts = repo.get_parts(database_path(), project_id)
    # ``?include_mesh=1`` re-reads each part's stored CAD file so a reloaded
    # project renders its geometry. It is opt-in because a large assembly can
    # push the response into the tens of megabytes; clients that only need the
    # settings (the project list, a rename) should not pay for it. The
    # per-part route below is the incremental alternative.
    if request.args.get("include_mesh") in {"1", "true", "True"}:
        for part in parts:
            mesh_data = load_part_mesh_data(str(part.get("filepath") or ""))
            if mesh_data is not None:
                part["mesh_data"] = mesh_data
    return jsonify({"project": project, "parts": parts})


@api_bp.route("/projects/<project_id>/parts/<part_id>/mesh")
def part_mesh(project_id: str, part_id: str):
    """Triangle geometry for one stored part, for restoring a project view.

    The shell fetches these in parallel after a refresh so parts appear one by
    one instead of the whole assembly blocking on the largest STL.
    """
    part = repo.get_part(database_path(), project_id, part_id)
    if part is None:
        return jsonify({"error": "Part not found"}), 404
    mesh_data = load_part_mesh_data(str(part.get("filepath") or ""))
    if mesh_data is None:
        return jsonify({"error": "Part geometry is unavailable"}), 404
    return jsonify({"mesh_data": mesh_data})


@api_bp.route("/projects/<project_id>", methods=["PUT"])
def update_project(project_id: str):
    if repo.get_project(database_path(), project_id) is None:
        return jsonify({"error": "Project not found"}), 404
    repo.update_project(database_path(), project_id, request.json or {})
    return jsonify({"project": repo.get_project(database_path(), project_id)})


@api_bp.route("/projects/<project_id>", methods=["DELETE"])
def delete_project(project_id: str):
    for filepath in repo.delete_project(database_path(), project_id):
        try:
            Path(filepath).unlink()
        except FileNotFoundError:
            pass
    return jsonify({"status": "deleted"})


@api_bp.route("/projects/<project_id>/parts", methods=["POST"])
def add_part(project_id: str):
    if repo.get_project(database_path(), project_id) is None:
        return jsonify({"error": "Project not found"}), 404
    try:
        # `filepath` must name a file this server stored; see
        # repo.resolve_upload_path for why an arbitrary path is a file-read
        # primitive rather than a convenience.
        part_id = repo.add_part(
            database_path(), project_id, request.json or {}, upload_root=upload_folder()
        )
    except repo.UnsafePartFileError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"id": part_id, "part": repo.get_part(database_path(), project_id, part_id)}), 201


@api_bp.route("/projects/<project_id>/parts/<part_id>", methods=["PUT"])
def update_part(project_id: str, part_id: str):
    try:
        repo.update_part(
            database_path(), project_id, part_id, request.json or {}, upload_root=upload_folder()
        )
    except repo.UnsafePartFileError as exc:
        return jsonify({"error": str(exc)}), 400
    part = repo.get_part(database_path(), project_id, part_id)
    if part is None:
        return jsonify({"error": "Part not found"}), 404
    return jsonify({"part": part})


@api_bp.route("/projects/<project_id>/parts/<part_id>", methods=["DELETE"])
def delete_part(project_id: str, part_id: str):
    filepath = repo.delete_part(database_path(), project_id, part_id)
    if filepath:
        try:
            Path(filepath).unlink()
        except FileNotFoundError:
            pass
    return jsonify({"status": "deleted"})


@api_bp.route("/projects/<project_id>/mesh-preview", methods=["POST"])
def mesh_preview(project_id: str):
    parts = repo.get_parts(database_path(), project_id)
    if not parts:
        return jsonify({"error": "No parts in project"}), 400
    voxel_size = float((request.json or {}).get("voxel_size", 0.1))
    manager = build_part_manager(parts)
    bounds = manager.get_combined_bounds()
    if bounds is None:
        return jsonify({"error": "Unable to calculate bounds"}), 400
    bounds_tuple = (
        (float(bounds[0][0]), float(bounds[0][1])),
        (float(bounds[1][0]), float(bounds[1][1])),
        (float(bounds[2][0]), float(bounds[2][1])),
    )
    bbox_volume = max(
        (bounds[0][1] - bounds[0][0]) * (bounds[1][1] - bounds[1][0]) * (bounds[2][1] - bounds[2][0]),
        1e-6,
    )
    fill_ratio = max(0.1, min(manager.get_total_volume() / bbox_volume, 0.9))
    validation = validate_mesh_parameters(bounds_tuple, voxel_size, fill_ratio)
    return jsonify(
        {
            "estimated_voxels": estimate_voxel_count(bounds_tuple, voxel_size, fill_ratio),
            "estimated_memory_mb": validation.estimated_memory_mb,
            "can_proceed": validation.can_proceed,
            "message": validation.message,
            "recommended_voxel_size": validation.recommended_voxel_size,
        }
    )


@api_bp.route("/projects/<project_id>/ml-mesh", methods=["POST"])
def ml_mesh(project_id: str):
    if repo.get_project(database_path(), project_id) is None:
        return jsonify({"error": "Project not found"}), 404
    parts = repo.get_parts(database_path(), project_id)
    if not parts:
        return jsonify({"error": "No parts in project"}), 400
    stl_filepath = str(parts[0].get("filepath") or "")
    if not stl_filepath or not Path(stl_filepath).exists():
        return jsonify({"error": "No CAD file uploaded for this project"}), 400
    loader = STLLoader()
    if not loader.load_stl(stl_filepath):
        return jsonify({"error": "Failed to load CAD file"}), 400
    target_accuracy = (request.json or {}).get("target_accuracy")
    try:
        recommendation: MeshRecommendation = recommend_mesh_settings(
            loader, target_accuracy=target_accuracy
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(
        {
            "base_voxel_size": recommendation.base_voxel_size,
            "refinement_level": recommendation.refinement_level,
            "confidence": recommendation.confidence,
            "reason": recommendation.reason,
            "used_model": recommendation.used_model,
            "warning": recommendation.warning,
        }
    )


@api_bp.route("/projects/<project_id>/run", methods=["POST"])
def run_project(project_id: str):
    project, parts = repo.get_project_and_parts(database_path(), project_id)
    if project is None:
        return jsonify({"error": "Project not found"}), 404
    if not parts:
        return jsonify({"error": "No parts in project"}), 400
    repo.mark_project_status(database_path(), project_id, "queued")
    response = queue_manager().submit(project_id, project, parts, user_ip())
    if response.get("status") in {"rate_limited", "queue_full"}:
        repo.mark_project_status(database_path(), project_id, "draft")
        return jsonify(response), 429 if response.get("status") == "rate_limited" else 503
    return jsonify(response)


@api_bp.route("/jobs/<project_id>/status")
def job_status(project_id: str):
    status = queue_manager().get_status(project_id)
    if status is not None:
        return jsonify(status)
    project = repo.get_project(database_path(), project_id)
    if project is None:
        return jsonify({"status": "not_found"}), 404
    return jsonify({"status": project.get("status", "draft"), "progress": 100 if project.get("status") == "completed" else 0})


@api_bp.route("/projects/<project_id>/cancel", methods=["POST"])
def cancel_project(project_id: str):
    response = queue_manager().cancel(project_id)
    if response.get("status") == "not_found":
        return jsonify({"error": "Project is not running"}), 404
    repo.mark_project_status(database_path(), project_id, "cancelled")
    return jsonify(response)


@api_bp.route("/projects/<project_id>/results")
def results(project_id: str):
    artifact = _load_artifact(project_id)
    if artifact is None:
        return jsonify({"error": "Results not found"}), 404
    if request.args.get("summary_only") in {"1", "true", "True"}:
        return jsonify({"data": artifact.get("summary", {})})
    return jsonify({"data": artifact.get("summary", {}), "visualization": artifact.get("visualization", {})})


@api_bp.route("/projects/<project_id>/result-fields")
def result_fields(project_id: str):
    artifact = _load_artifact(project_id)
    if artifact is None:
        return jsonify({"error": "Results not found"}), 404
    max_points = int(request.args.get("max_points", "120000"))
    return jsonify(build_result_fields_payload(artifact.get("visualization", {}), max_points=max_points))


@api_bp.route("/projects/<project_id>/surface")
def surface(project_id: str):
    artifact = _load_artifact(project_id)
    if artifact is None:
        return jsonify({"error": "Results not found"}), 404
    try:
        payload = build_fork_surface_payload(
            artifact.get("visualization", {}),
            str(request.args.get("mode", "shrinkage")),
            float(request.args.get("threshold", "0.3")),
            float(request.args.get("time_percent", "0")),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(payload)


@api_bp.route("/projects/<project_id>/volume")
def volume(project_id: str):
    artifact = _load_artifact(project_id)
    if artifact is None:
        return jsonify({"error": "Results not found"}), 404
    payload = build_volume_payload(
        artifact.get("visualization", {}),
        str(request.args.get("mode", "fraction_liquid")),
        float(request.args.get("time_percent", "0")),
    )
    return jsonify(payload)


@api_bp.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    """Serve a stored CAD file.

    ``send_from_directory`` already refuses to escape the folder, so the guard
    that matters is upstream: nothing outside the folder may be *placed* in it.
    See ``repo.resolve_upload_path`` and the duplicate endpoint.
    """
    return send_from_directory(str(upload_folder()), filename)


def _load_artifact(project_id: str) -> Dict[str, Any] | None:
    result = queue_manager().get_result(project_id)
    if result is not None:
        return result
    artifact_path = repo.get_artifact_path(database_path(), project_id)
    if artifact_path is None or not Path(artifact_path).exists():
        return None
    return read_artifact(artifact_path)
