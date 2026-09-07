"""Application-shell routes: project management extras, riser design, reports,
hardware info, job logs, project file import/export.

Owned by the shell/parity workstream. Registered by
``web.app_factory.create_app``. Everything here lives under ``/api``.
"""

from __future__ import annotations

from flask import Blueprint, jsonify

tools_bp = Blueprint("web_tools", __name__, url_prefix="/api")


@tools_bp.route("/tools/health")
def tools_health():
    return jsonify({"ok": True, "blueprint": "web_tools"})
