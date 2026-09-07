"""Result-field routes for the web viewport: dense 3D grids, frames, isosurfaces.

Owned by the rendering/data-pipeline workstream. Registered by
``web.app_factory.create_app``. Everything here lives under ``/api``.
"""

from __future__ import annotations

from flask import Blueprint, jsonify

fields_bp = Blueprint("web_fields", __name__, url_prefix="/api")


@fields_bp.route("/fields/health")
def fields_health():
    return jsonify({"ok": True, "blueprint": "web_fields"})
