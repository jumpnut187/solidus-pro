"""Primary web routes and background simulation task execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, cast
import os
import traceback
import uuid

import numpy as np
from flask import Blueprint, current_app, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from core.materials import get_all_alloy_names, get_alloy
from core.memory_limits import estimate_voxel_count, validate_mesh_parameters
from core.simulation_pipeline import (
    PipelineInputs,
    SimulationCancelledError,
    build_part_manager_from_parts,
    run_pipeline,
)
from core.stl_loader import STLLoader
from web.repositories import simulation_repository as repo
from web.services.simulation_queue import SimulationQueueManager


api_bp = Blueprint("api", __name__)


def _database_path() -> Path:
    return Path(current_app.config["DATABASE"])


def _upload_folder() -> Path:
    return Path(current_app.config["UPLOAD_FOLDER"])


def _queue_manager() -> SimulationQueueManager:
    manager = cast(Any, current_app.extensions.get("simulation_queue_manager"))
    if manager is None:
        raise RuntimeError("Simulation queue manager is not initialized")
    return cast(SimulationQueueManager, manager)


def get_user_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return str(forwarded.split(",")[0].strip())
    return request.remote_addr or "unknown"


def execute_simulation_task(
    sim_id: str,
    sim_data: Dict[str, Any],
    parts_data: list[Dict[str, Any]],
    queue_manager: SimulationQueueManager,
    database_path: Path,
) -> None:
    """
    Background simulation worker called by the queue manager.
    """

    def check_cancelled() -> bool:
        return queue_manager.should_cancel(sim_id)

    try:
        if check_cancelled():
            raise SimulationCancelledError("Cancelled")

        queue_manager.set_progress(sim_id, 5, "Loading parts...")
        part_manager = build_part_manager_from_parts(parts_data)

        if check_cancelled():
            raise SimulationCancelledError("Cancelled")

        queue_manager.set_progress(sim_id, 15, "Generating voxel mesh...")
        raw_analysis_options = sim_data.get("analysis_options") or {}
        if isinstance(raw_analysis_options, str):
            try:
                import json

                raw_analysis_options = json.loads(raw_analysis_options)
            except Exception:
                raw_analysis_options = {}

        inputs = PipelineInputs(
            alloy_name=str(sim_data.get("alloy", "WCB")),
            voxel_size=float(sim_data.get("voxel_size", 0.1)),
            refinement_level=int(sim_data.get("refinement_level", 4)),
            analysis_options=raw_analysis_options,
            part_assignment_mode=str(sim_data.get("part_assignment_mode", "auto")),
            part_assignment_workers=int(sim_data.get("part_assignment_workers", 0) or 0),
            part_assignment_batch_size=int(sim_data.get("part_assignment_batch_size", 2500) or 2500),
            show_progress=False,
        )

        def progress_message(message: str) -> None:
            if message.startswith("Generating adaptive voxel mesh"):
                queue_manager.set_progress(sim_id, 15, "Generating voxel mesh...")
            elif message.startswith("Running solidification simulation"):
                queue_manager.set_progress(sim_id, 55, "Running solidification simulation...")
            else:
                queue_manager.set_progress(sim_id, 85, message)

        def detailed_progress(current: int, total: int, stage: str) -> None:
            if stage.lower().startswith("solid"):
                progress = 65 + int((current / max(total, 1)) * 20)
                queue_manager.set_progress(sim_id, progress, f"Solidifying: {current}%")
            else:
                progress = 15 + int((current / max(total, 1)) * 35)
                queue_manager.set_progress(sim_id, progress, f"Voxelizing: {stage} ({current}/{total})")

        result = run_pipeline(
            inputs,
            part_manager=part_manager,
            progress_message_callback=progress_message,
            detailed_progress_callback=detailed_progress,
            cancel_callback=check_cancelled,
        )

        queue_manager.save_result(
            sim_id,
            {
                "data": result.result_data,
                "visualization": result.visualization_data,
            },
        )

        repo.mark_simulation_completed(database_path, sim_id, result.result_data)
        queue_manager.set_status(sim_id, "completed", "Simulation complete!", 100)
    except SimulationCancelledError:
        queue_manager.set_status(sim_id, "cancelled", "Simulation cancelled", 0)
        repo.mark_simulation_status(database_path, sim_id, "cancelled")
    except Exception as exc:
        traceback.print_exc()
        queue_manager.set_status(sim_id, "error", str(exc), 0)
        repo.mark_simulation_status(database_path, sim_id, "error")


@api_bp.route("/")
def index():
    return render_template("index.html")


@api_bp.route("/api/materials")
def get_materials():
    materials = []
    for name in get_all_alloy_names():
        alloy = get_alloy(name)
        materials.append(
            {
                "name": name,
                "full_name": alloy.name,
                "liquidus_temp": alloy.liquidus_temp,
                "solidus_temp": alloy.solidus_temp,
                "volumetric_shrinkage": round(alloy.volumetric_shrinkage, 2),
                "feeding_multiplier": alloy.feeding_distance_multiplier,
                "niyama_threshold": alloy.niyama_threshold,
            }
        )
    return jsonify(materials)


@api_bp.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith(".stl"):
        return jsonify({"error": "Only STL files are allowed"}), 400

    file_id = str(uuid.uuid4())[:8]
    filename = secure_filename(file.filename)
    unique_filename = f"{file_id}_{filename}"
    filepath = _upload_folder() / unique_filename
    file.save(str(filepath))

    loader = STLLoader()
    if not loader.load_stl(str(filepath)):
        os.remove(filepath)
        return jsonify({"error": "Failed to load STL file"}), 400

    bounds = loader.get_bounds_array()
    vertices = loader.get_vertices()
    faces = loader.get_faces()
    mesh_data = {
        "vertices": vertices.flatten().tolist(),
        "faces": faces.flatten().tolist(),
        "bounds": {
            "x": [float(bounds[0][0]), float(bounds[0][1])],
            "y": [float(bounds[1][0]), float(bounds[1][1])],
            "z": [float(bounds[2][0]), float(bounds[2][1])],
        },
        "volume": float(loader.volume) if loader.volume else 0,
        "surface_area": float(loader.surface_area) if loader.surface_area else 0,
        "vertex_count": len(vertices),
        "face_count": len(faces),
    }

    return jsonify(
        {
            "file_id": file_id,
            "filename": filename,
            "filepath": str(filepath),
            "dimensions": loader.get_dimensions(),
            "mesh_data": mesh_data,
        }
    )


@api_bp.route("/api/simulations", methods=["GET"])
def list_simulations():
    return jsonify(repo.list_simulations(_database_path()))


@api_bp.route("/api/simulations", methods=["POST"])
def create_simulation():
    sim_id = repo.create_simulation(_database_path(), request.json or {})
    return jsonify({"id": sim_id, "status": "created"})


@api_bp.route("/api/simulations/<sim_id>", methods=["GET"])
def get_simulation(sim_id: str):
    simulation = repo.get_simulation(_database_path(), sim_id)
    if simulation is None:
        return jsonify({"error": "Simulation not found"}), 404

    return jsonify(
        {
            "simulation": simulation,
            "parts": repo.get_simulation_parts(_database_path(), sim_id),
            "results": repo.get_simulation_results(_database_path(), sim_id),
        }
    )


@api_bp.route("/api/simulations/<sim_id>", methods=["PUT"])
def update_simulation(sim_id: str):
    repo.update_simulation(_database_path(), sim_id, request.json or {})
    return jsonify({"status": "updated"})


@api_bp.route("/api/simulations/<sim_id>", methods=["DELETE"])
def delete_simulation(sim_id: str):
    filepaths = repo.delete_simulation(_database_path(), sim_id)
    for filepath in filepaths:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass
    return jsonify({"status": "deleted"})


@api_bp.route("/api/simulations/<sim_id>/parts", methods=["POST"])
def add_part(sim_id: str):
    part_id = repo.add_part(_database_path(), sim_id, request.json or {})
    return jsonify({"id": part_id, "status": "added"})


@api_bp.route("/api/simulations/<sim_id>/parts/<part_id>", methods=["PUT"])
def update_part(sim_id: str, part_id: str):
    repo.update_part(_database_path(), sim_id, part_id, request.json or {})
    return jsonify({"status": "updated"})


@api_bp.route("/api/simulations/<sim_id>/parts/<part_id>", methods=["DELETE"])
def delete_part(sim_id: str, part_id: str):
    filepath = repo.delete_part(_database_path(), sim_id, part_id)
    if filepath:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass
    return jsonify({"status": "deleted"})


@api_bp.route("/api/mesh/preview", methods=["POST"])
def preview_mesh():
    data = request.json or {}
    filepath = data.get("filepath")
    voxel_size = float(data.get("voxel_size", 0.1))

    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 400

    loader = STLLoader()
    if not loader.load_stl(filepath):
        return jsonify({"error": "Failed to load STL"}), 400

    bounds = loader.get_bounds_array()
    bounds_tuple = (
        (float(bounds[0][0]), float(bounds[0][1])),
        (float(bounds[1][0]), float(bounds[1][1])),
        (float(bounds[2][0]), float(bounds[2][1])),
    )

    bbox_vol = (bounds[0][1] - bounds[0][0]) * (bounds[1][1] - bounds[1][0]) * (bounds[2][1] - bounds[2][0])
    fill_ratio = max(0.1, min(loader.volume / bbox_vol, 0.9)) if bbox_vol > 0 and loader.volume else 0.5
    estimated_voxels = estimate_voxel_count(bounds_tuple, voxel_size, fill_ratio)
    validation = validate_mesh_parameters(bounds_tuple, voxel_size, fill_ratio)

    return jsonify(
        {
            "estimated_voxels": estimated_voxels,
            "estimated_memory_mb": validation.estimated_memory_mb,
            "can_proceed": validation.can_proceed,
            "message": validation.message,
            "recommended_voxel_size": validation.recommended_voxel_size,
        }
    )


@api_bp.route("/api/mesh/recommend", methods=["POST"])
def recommend_mesh():
    data = request.json or {}
    bounds = data.get("bounds", {})
    vertex_count = data.get("vertex_count", 0)
    face_count = data.get("face_count", 0)
    volume = data.get("volume", 0)

    x_range = bounds.get("x", [0, 1])
    y_range = bounds.get("y", [0, 1])
    z_range = bounds.get("z", [0, 1])
    dx = x_range[1] - x_range[0]
    dy = y_range[1] - y_range[0]
    dz = z_range[1] - z_range[0]

    max_dim = max(dx, dy, dz)
    min_dim = min(dx, dy, dz)
    bbox_volume = dx * dy * dz
    complexity = face_count / max(bbox_volume, 0.001) if bbox_volume > 0 else 1000

    try:
        from core.mesh_ml import recommend_mesh_settings

        class MockLoader:
            def __init__(self) -> None:
                self.volume = volume
                self.face_count = face_count
                self.vertex_count = vertex_count

            def get_bounds_array(self):
                return np.array(
                    [
                        [x_range[0], x_range[1]],
                        [y_range[0], y_range[1]],
                        [z_range[0], z_range[1]],
                    ]
                )

        recommendation = recommend_mesh_settings(MockLoader())
        return jsonify(
            {
                "voxel_size": float(recommendation.voxel_size),
                "refinement_level": int(recommendation.refinement_level),
                "confidence": float(recommendation.confidence),
                "source": recommendation.source,
            }
        )
    except Exception:
        pass

    base_voxel_size = max_dim / 50
    if complexity > 5000:
        voxel_size = base_voxel_size * 0.7
        refinement_level = 4
    elif complexity > 1000:
        voxel_size = base_voxel_size
        refinement_level = 4
    else:
        voxel_size = base_voxel_size * 1.3
        refinement_level = 3

    voxel_size = max(0.02, min(0.5, voxel_size))
    est_voxels = (dx / voxel_size) * (dy / voxel_size) * (dz / voxel_size) * 0.5
    if est_voxels > 500000:
        voxel_size *= 1.5
        refinement_level = max(2, refinement_level - 1)
    elif est_voxels < 10000:
        voxel_size *= 0.7
        refinement_level = min(4, refinement_level + 1)
    voxel_size = max(0.02, min(0.5, voxel_size))

    return jsonify(
        {
            "voxel_size": round(voxel_size, 3),
            "refinement_level": refinement_level,
            "confidence": 0.7,
            "source": "heuristic",
        }
    )


@api_bp.route("/api/simulations/<sim_id>/run", methods=["POST"])
def run_simulation(sim_id: str):
    manager = _queue_manager()
    simulation, parts = repo.get_simulation_and_parts(_database_path(), sim_id)

    if simulation is None:
        return jsonify({"error": "Simulation not found"}), 404
    if not parts:
        return jsonify({"error": "No parts in simulation"}), 400

    response = manager.submit(sim_id, simulation, parts, get_user_ip())
    status = response.get("status")
    if status == "rate_limited":
        return jsonify(response), 429
    if status == "queue_full":
        return jsonify(response), 503
    return jsonify(response)


@api_bp.route("/api/queue/status", methods=["GET"])
def queue_status():
    return jsonify(_queue_manager().get_queue_status(get_user_ip()))


@api_bp.route("/api/simulations/<sim_id>/cancel", methods=["POST"])
def cancel_simulation(sim_id: str):
    response = _queue_manager().cancel(sim_id)
    if response.get("status") == "not_found":
        return jsonify({"error": "Simulation not found or not running"}), 404
    repo.mark_simulation_status(_database_path(), sim_id, "cancelled")
    return jsonify(response)


@api_bp.route("/api/simulations/cleanup", methods=["POST"])
def cleanup_simulations():
    cancelled = _queue_manager().cleanup_user(get_user_ip())
    for sim_id in cancelled:
        repo.mark_simulation_status(_database_path(), sim_id, "cancelled")
    return jsonify({"cancelled": cancelled})


@api_bp.route("/api/simulations/<sim_id>/status")
def get_simulation_status(sim_id: str):
    manager = _queue_manager()
    status = manager.get_status(sim_id)
    if status is not None:
        if status.get("status") == "queued":
            position = manager.get_queue_position(sim_id)
            status["queue_position"] = position
            status["message"] = f"Queued (position {position})"
        return jsonify(status)

    db_status = repo.get_simulation_status(_database_path(), sim_id)
    if db_status is not None:
        return jsonify({"status": db_status, "progress": 100 if db_status == "completed" else 0})
    return jsonify({"status": "not_found"}), 404


@api_bp.route("/api/simulations/<sim_id>/results")
def get_simulation_results(sim_id: str):
    result = _queue_manager().get_result(sim_id)
    if result is not None:
        return jsonify(result)

    db_result = repo.get_summary_result(_database_path(), sim_id)
    if db_result is not None:
        return jsonify({"data": db_result})
    return jsonify({"error": "Results not found"}), 404


@api_bp.route("/api/measure/distance", methods=["POST"])
def measure_distance():
    data = request.json or {}
    p1 = np.asarray(data["point1"])
    p2 = np.asarray(data["point2"])
    return jsonify({"distance": float(np.linalg.norm(p2 - p1)), "unit": "inches"})


@api_bp.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(str(_upload_folder()), filename)
