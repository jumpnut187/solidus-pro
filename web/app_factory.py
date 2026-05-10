"""Flask app factory for the web simulator."""

from __future__ import annotations

from pathlib import Path
import os

from flask import Flask, jsonify
from flask_cors import CORS

from web.repositories import simulation_repository as repo
from web.routes import api_bp, execute_simulation_task
from web.services.simulation_queue import QueueConfig, SimulationQueueManager


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    CORS(app)

    base_dir = Path(__file__).parent
    app.config["SECRET_KEY"] = os.environ.get("CAST_SIM_SECRET_KEY", "cast-sim-dev-key")
    app.config["UPLOAD_FOLDER"] = base_dir / "static" / "uploads"
    app.config["TEMP_FOLDER"] = base_dir / "static" / "temp"
    app.config["DATABASE"] = base_dir / "simulations.db"
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

    app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)
    app.config["TEMP_FOLDER"].mkdir(parents=True, exist_ok=True)

    repo.init_db(Path(app.config["DATABASE"]))

    queue_manager = SimulationQueueManager(
        QueueConfig(
            max_concurrent_simulations=2,
            max_queue_size=10,
            max_simulations_per_user=2,
        )
    )
    app.extensions["simulation_queue_manager"] = queue_manager

    app_object = app
    database_path = Path(app.config["DATABASE"])

    def worker(sim_id: str, sim_data, parts_data) -> None:
        with app_object.app_context():
            execute_simulation_task(
                sim_id=sim_id,
                sim_data=sim_data,
                parts_data=parts_data,
                queue_manager=queue_manager,
                database_path=database_path,
            )

    queue_manager.set_worker(worker)

    @app.after_request
    def add_no_cache_headers(response):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({"error": "File too large. Maximum size is 100MB"}), 413

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": "Internal server error"}), 500

    app.register_blueprint(api_bp)
    return app
