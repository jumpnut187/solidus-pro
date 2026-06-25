"""Flask factory for the React/Babylon web application."""

from __future__ import annotations

from pathlib import Path
import os

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

from web.services.simulation_queue import QueueConfig, SimulationQueueManager
from web import repository as repo
from web.routes import api_bp


def create_app() -> Flask:
    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parent
    frontend_dist = repo_root / "web_frontend" / "dist"

    app = Flask(
        __name__,
        static_folder=str(frontend_dist) if frontend_dist.exists() else None,
        static_url_path="",
    )
    CORS(app)

    app.config["SECRET_KEY"] = os.environ.get("CASTSIM_SECRET_KEY", "cast-sim-dev-key")
    app.config["UPLOAD_FOLDER"] = base_dir / "static" / "uploads"
    app.config["TEMP_FOLDER"] = base_dir / "static" / "temp"
    app.config["DATABASE"] = base_dir / "simulations.db"
    app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024

    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["TEMP_FOLDER"]).mkdir(parents=True, exist_ok=True)
    repo.init_db(Path(app.config["DATABASE"]))

    queue_manager = SimulationQueueManager(
        QueueConfig(
            max_concurrent_simulations=1,
            max_queue_size=6,
            max_simulations_per_user=1,
        )
    )
    app.extensions["simulation_queue_manager"] = queue_manager

    database_path = Path(app.config["DATABASE"])
    queue_manager.set_paths(str(database_path), str(repo_root))

    def worker(project_id: str, project, parts) -> None:
        pass

    queue_manager.set_worker(worker)

    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({"error": "File too large. Maximum size is 250MB"}), 413

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": "Internal server error"}), 500

    app.register_blueprint(api_bp)

    @app.route("/")
    def index():
        index_path = frontend_dist / "index.html"
        if index_path.exists():
            return send_from_directory(str(frontend_dist), "index.html")
        return jsonify(
            {
                "app": "Solidus React/Babylon Web Application",
                "frontend": "Run `npm install` and `npm run build` in web_frontend to serve the React app.",
            }
        )

    @app.route("/<path:path>")
    def spa_fallback(path: str):
        asset_path = frontend_dist / path
        if asset_path.exists() and asset_path.is_file():
            return send_from_directory(str(frontend_dist), path)
        index_path = frontend_dist / "index.html"
        if index_path.exists():
            return send_from_directory(str(frontend_dist), "index.html")
        return jsonify({"error": "Frontend build not found"}), 404

    return app

