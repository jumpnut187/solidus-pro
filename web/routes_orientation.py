"""Routes for the "Orient & Parting Line" tool.

The browser turns and lifts parts itself (``web_frontend/src/lib/orientation.ts``
mirrors :mod:`core.orientation`), and stores each placement through the
ordinary part PUT. Two things need the server:

* the parting-line suggestions and the cope/drag volume split, which slice the
  stored mesh (:func:`core.orientation.suggest_parting_z`,
  :func:`core.orientation.cope_drag_split`);
* the parting line itself, a horizontal plane kept in the project's
  ``ui_state["parting_line"]`` as ``{"z", "visible", "locked_to_casting"}``. A
  project without the key has no parting line yet.
"""

from __future__ import annotations

import math
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import trimesh
from flask import Blueprint, current_app, jsonify, request

from core import orientation
from web import repository as repo

orientation_bp = Blueprint("web_orientation", __name__, url_prefix="/api")

#: ``ui_state`` key the parting line lives under; route-owned, see
#: :data:`web.repository.ROUTE_OWNED_UI_STATE_KEYS`.
PARTING_LINE_KEY = "parting_line"

#: Parsed meshes kept between requests. Every nudge of the plane asks for a
#: fresh split, and re-reading a large STL for each would dominate the reply.
_MESH_CACHE_SIZE = 4
_mesh_cache: "OrderedDict[Tuple[str, float, int], trimesh.Trimesh]" = OrderedDict()
_mesh_cache_lock = threading.Lock()

#: A coordinate further out than this is a typo, not a foundry.
MAX_ABS_COORDINATE_IN = 1.0e5


class OrientationRequestError(ValueError):
    """A request the client must fix; reported as a 400 with this message."""


def _database_path() -> Path:
    return Path(current_app.config["DATABASE"])


def _upload_folder() -> Path:
    return Path(current_app.config["UPLOAD_FOLDER"])


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise OrientationRequestError(f"{name} must be a number.") from error
    if not math.isfinite(number) or abs(number) > MAX_ABS_COORDINATE_IN:
        raise OrientationRequestError(f"{name} must be a finite number of inches.")
    return number


def _triple_param(raw: Optional[str], name: str) -> Optional[Tuple[float, float, float]]:
    """``"a,b,c"`` from a query string, or None when absent."""
    if raw is None or raw == "":
        return None
    pieces = raw.split(",")
    if len(pieces) != 3:
        raise OrientationRequestError(f"{name} takes three comma-separated numbers.")
    a, b, c = (_finite(piece, name) for piece in pieces)
    return (a, b, c)


def checked_parting_line(raw: Any) -> Dict[str, Any]:
    """Validate a parting-line payload into the stored shape."""
    if not isinstance(raw, dict):
        raise OrientationRequestError("Send the parting line as an object with a z value.")
    if "z" not in raw:
        raise OrientationRequestError("The parting line needs a z value in inches.")
    return {
        "z": _finite(raw.get("z"), "z"),
        "visible": bool(raw.get("visible", True)),
        "locked_to_casting": bool(raw.get("locked_to_casting", True)),
    }


def stored_parting_line(project: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The project's parting line, or None when none was set (or it is malformed)."""
    raw = (project.get("ui_state") or {}).get(PARTING_LINE_KEY)
    try:
        return checked_parting_line(raw)
    except OrientationRequestError:
        return None


def _load_mesh(filepath: str) -> Optional[trimesh.Trimesh]:
    """The part's STL as a mesh, cached by path and modification time."""
    try:
        stat = os.stat(filepath)
    except OSError:
        return None
    key = (filepath, stat.st_mtime, stat.st_size)
    with _mesh_cache_lock:
        cached = _mesh_cache.get(key)
        if cached is not None:
            _mesh_cache.move_to_end(key)
            return cached
    try:
        loaded = trimesh.load(filepath, force="mesh")
    except Exception:
        return None
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        return None
    with _mesh_cache_lock:
        _mesh_cache[key] = loaded
        while len(_mesh_cache) > _MESH_CACHE_SIZE:
            _mesh_cache.popitem(last=False)
    return loaded


def _casting_part(parts: List[Dict[str, Any]], part_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """The named part, else the first visible casting, else the first casting."""
    if part_id:
        return next((part for part in parts if str(part.get("id")) == part_id), None)
    castings = [part for part in parts if part.get("part_type") == "cast"]
    visible = [part for part in castings if part.get("visible", True)]
    pool = visible or castings
    return pool[0] if pool else None


@orientation_bp.route("/projects/<project_id>/parting-line", methods=["GET"])
def get_parting_line(project_id: str):
    """Stored parting line plus suggestions and the cope/drag split.

    Query parameters, all optional:

    * ``part_id``: the body to measure; defaults to the project's casting.
    * ``rotation``, ``position`` (``"x,y,z"``) and ``scale``: the placement to
      measure at. The client sends the placement it is showing, so the reply
      never lags a part write still in flight; omitted, the stored row is used.
    * ``z``: the plane to split at; defaults to the stored parting line, then
      to the widest-section suggestion.
    """
    database_path = _database_path()
    project, parts = repo.get_project_and_parts(database_path, project_id)
    if project is None:
        return jsonify({"error": "Project not found"}), 404
    parting_line = stored_parting_line(project)
    part = _casting_part(parts, request.args.get("part_id") or None)
    if part is None:
        return jsonify(
            {
                "parting_line": parting_line,
                "part_id": None,
                "error": "The project has no casting to measure.",
            }
        ), 404

    try:
        rotation = _triple_param(request.args.get("rotation"), "rotation") or (
            float(part["rotation_x"]),
            float(part["rotation_y"]),
            float(part["rotation_z"]),
        )
        position = _triple_param(request.args.get("position"), "position") or (
            float(part["position_x"]),
            float(part["position_y"]),
            float(part["position_z"]),
        )
        scale_raw = request.args.get("scale")
        scale = _finite(scale_raw, "scale") if scale_raw not in (None, "") else float(part.get("scale") or 1.0)
        if scale <= 0:
            raise OrientationRequestError("scale must be positive.")
        z_raw = request.args.get("z")
        requested_z = _finite(z_raw, "z") if z_raw not in (None, "") else None
    except OrientationRequestError as error:
        return jsonify({"error": str(error)}), 400

    try:
        filepath = repo.resolve_upload_path(str(part.get("filepath") or ""), _upload_folder())
    except repo.UnsafePartFileError as error:
        return jsonify({"error": str(error)}), 400
    mesh = _load_mesh(filepath)
    if mesh is None:
        return jsonify({"error": "Part geometry is unavailable"}), 404

    suggestions = orientation.suggest_parting_z(mesh, rotation, position, scale)
    if requested_z is not None:
        split_z = requested_z
    elif parting_line is not None:
        split_z = float(parting_line["z"])
    else:
        split_z = float(suggestions["max_section"])
    split = orientation.cope_drag_split(mesh, rotation, position, split_z, scale)
    bounds = orientation.world_bounds(mesh, rotation, position, scale)
    return jsonify(
        {
            "parting_line": parting_line,
            "part_id": str(part["id"]),
            "suggestions": suggestions,
            "split_z": round(split_z, 4),
            "split": split,
            "bounds": [[round(float(v), 4) for v in row] for row in bounds],
        }
    )


@orientation_bp.route("/projects/<project_id>/parting-line", methods=["PUT"])
def put_parting_line(project_id: str):
    """Store the parting line: ``{"z": float, "visible": bool, "locked_to_casting": bool}``.

    ``{"parting_line": null}`` removes it. A dedicated route rather than the
    project PUT, for the same reason as the reference planes: the shell's
    settings autosave does not carry it and must not clear it.
    """
    database_path = _database_path()
    if repo.get_project(database_path, project_id) is None:
        return jsonify({"error": "Project not found"}), 404
    payload = request.get_json(silent=True)
    if isinstance(payload, dict) and "parting_line" in payload and payload["parting_line"] is None:
        repo.delete_ui_state_key(database_path, project_id, PARTING_LINE_KEY)
        return jsonify({"parting_line": None})
    raw = payload.get("parting_line", payload) if isinstance(payload, dict) else payload
    try:
        parting_line = checked_parting_line(raw)
    except OrientationRequestError as error:
        return jsonify({"error": str(error)}), 400
    repo.set_ui_state_key(database_path, project_id, PARTING_LINE_KEY, parting_line)
    return jsonify({"parting_line": parting_line})
