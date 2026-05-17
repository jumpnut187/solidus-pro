"""HTTP routes for the React/Babylon web application."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, cast
import traceback

from flask import Blueprint, current_app, jsonify, request, send_from_directory

from core.materials import get_all_alloy_names, get_alloy
from core.memory_limits import estimate_voxel_count, validate_mesh_parameters
from core.simulation_pipeline import PipelineInputs, SimulationCancelledError, run_pipeline
from web.services.simulation_queue import SimulationQueueManager
from web import repository as repo
from web.payloads import (
    build_fork_artifact,
    build_fork_surface_payload,
    build_part_manager,
    build_result_fields_payload,
    build_volume_payload,
    read_artifact,
    upload_stl_file,
    write_artifact,
)


api_bp = Blueprint("web_api", __name__, url_prefix="/api")


def database_path() -> Path:
    return Path(current_app.config["DATABASE"])


def upload_folder() -> Path:
    return Path(current_app.config["UPLOAD_FOLDER"])


def artifact_folder() -> Path:
    folder = Path(current_app.config["TEMP_FOLDER"]) / "artifacts"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


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


def execute_project_task(
    project_id: str,
    project: Dict[str, Any],
    parts: list[Dict[str, Any]],
    manager: SimulationQueueManager,
    db_path: Path,
) -> None:
    def check_cancelled() -> bool:
        return manager.should_cancel(project_id)

    try:
        manager.set_progress(project_id, 4, "Loading assembly...")
        part_manager = build_part_manager(parts)
        if check_cancelled():
            raise SimulationCancelledError("Cancelled")

        performance = project.get("performance_options") or {}
        analysis = project.get("analysis_options") or {}
        inputs = PipelineInputs(
            alloy_name=str(project.get("alloy", "WCB")),
            voxel_size=float(project.get("voxel_size", 0.1)),
            refinement_level=int(project.get("refinement_level", 3)),
            memory_mode=str(project.get("memory_mode", "auto")),
            analysis_options=analysis,
            use_gpu=bool(performance.get("use_gpu", False)),
            use_incremental=bool(performance.get("use_incremental", False)),
            use_chunked_voxelization=bool(performance.get("use_chunked_voxelization", False)),
            part_assignment_mode=str(performance.get("part_assignment_mode", "auto")),
            part_assignment_workers=int(performance.get("part_assignment_workers", 0) or 0),
            part_assignment_batch_size=int(performance.get("part_assignment_batch_size", 2500) or 2500),
            max_voxels_for_visualization=int(performance.get("max_voxels_for_visualization", 80_000) or 80_000),
            solidification_params=project.get("solidification_params") or {},
        )

        def progress_message(message: str) -> None:
            if message.startswith("Generating adaptive voxel mesh"):
                manager.set_progress(project_id, 15, "Generating adaptive voxel mesh...")
            elif message.startswith("Assigning part types"):
                manager.set_progress(project_id, 42, "Assigning cast, riser, chill, and riser-post voxels...")
            elif message.startswith("Running solidification simulation"):
                manager.set_progress(project_id, 55, "Running solidification physics...")
            else:
                manager.set_progress(project_id, 82, message)

        def detailed_progress(current: int, total: int, stage: str) -> None:
            stage_l = stage.lower()
            if stage_l.startswith("solid"):
                progress = 55 + int((current / max(total, 1)) * 32)
                manager.set_progress(project_id, progress, f"Solidifying: {current}%")
            else:
                progress = 15 + int((current / max(total, 1)) * 25)
                manager.set_progress(project_id, progress, f"Voxelizing: {stage} ({current}/{total})")

        result = run_pipeline(
            inputs,
            part_manager=part_manager,
            progress_message_callback=progress_message,
            detailed_progress_callback=detailed_progress,
            cancel_callback=check_cancelled,
        )
        artifact = build_fork_artifact(result)
        artifact_path = artifact_folder() / f"{project_id}.json"
        write_artifact(artifact_path, artifact)
        repo.mark_project_completed(db_path, project_id, artifact["summary"], str(artifact_path))
        manager.save_result(project_id, artifact)
        manager.set_status(project_id, "completed", "Simulation complete", 100)
    except SimulationCancelledError:
        manager.set_status(project_id, "cancelled", "Simulation cancelled", 0)
        repo.mark_project_status(db_path, project_id, "cancelled")
    except Exception as exc:
        traceback.print_exc()
        manager.set_status(project_id, "error", str(exc), 0)
        repo.mark_project_status(db_path, project_id, "error")


@api_bp.route("/health")
def health():
    return jsonify({"ok": True, "app": "web"})


@api_bp.route("/materials")
def materials():
    payload = []
    for name in get_all_alloy_names():
        alloy = get_alloy(name)
        payload.append(
            {
                "name": name,
                "full_name": alloy.name,
                "liquidus_temp": alloy.liquidus_temp,
                "solidus_temp": alloy.solidus_temp,
                "volumetric_shrinkage": round(alloy.volumetric_shrinkage, 2),
                "feeding_multiplier": alloy.feeding_distance_multiplier,
                "niyama_threshold": alloy.niyama_threshold,
                "hot_tear_susceptibility": getattr(alloy, "hot_tear_susceptibility", 0.0),
                "sand_burn_in_susceptibility": getattr(alloy, "sand_burn_in_susceptibility", 0.0),
            }
        )
    return jsonify(payload)


@api_bp.route("/uploads", methods=["POST"])
def upload_files():
    files = request.files.getlist("files") or request.files.getlist("file")
    if not files:
        return jsonify({"error": "No files provided"}), 400
    uploaded = []
    try:
        for file in files:
            uploaded.append(upload_stl_file(file, upload_folder()))
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
    return jsonify({"project": project, "parts": repo.get_parts(database_path(), project_id)})


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
    part_id = repo.add_part(database_path(), project_id, request.json or {})
    return jsonify({"id": part_id, "part": repo.get_part(database_path(), project_id, part_id)}), 201


@api_bp.route("/projects/<project_id>/parts/<part_id>", methods=["PUT"])
def update_part(project_id: str, part_id: str):
    repo.update_part(database_path(), project_id, part_id, request.json or {})
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
    return send_from_directory(str(upload_folder()), filename)


def _load_artifact(project_id: str) -> Dict[str, Any] | None:
    result = queue_manager().get_result(project_id)
    if result is not None:
        return result
    artifact_path = repo.get_artifact_path(database_path(), project_id)
    if artifact_path is None or not Path(artifact_path).exists():
        return None
    return read_artifact(artifact_path)
