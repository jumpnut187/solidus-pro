"""Routes for bodies modeled in the browser (sketch + extrude features).

The geometry kernel runs in the browser (Manifold, in a web worker), because a
sketch has to preview at interactive speed against the uploaded castings the
client already holds. The server's part is storage: it receives the finished,
watertight triangle mesh with the parametric definition that produced it,
writes the mesh as an STL in the upload folder -- which is all the simulation
engine ever reads -- and keeps the definition on the part row so the body can
be edited and rebuilt later.

An extrude cut is stored in two halves. Its definition is one entry of the
project's ``ui_state["cut_features"]``; what it did is the STL of every body
it cut. A cut body's row keeps the STL it had before its first cut as
``base_filepath``, so editing or deleting a cut rebuilds the body from that
base and gives back the material. The simulation still reads ``filepath``
only.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import trimesh
from flask import Blueprint, current_app, jsonify, request

from core.part_manager import PartType
from web import repository as repo
from web.payloads import _build_mesh_data, _sanitize_body_name, load_part_mesh_data

modeling_bp = Blueprint("web_modeling", __name__, url_prefix="/api")

#: Feature definitions are a few kilobytes of numbers; anything near this is
#: not a sketch.
MAX_FEATURE_BYTES = 256 * 1024
#: A conforming riser or chill on a fine casting skin can carry a few hundred
#: thousand triangles; ten times that is not a modeled body.
MAX_MODELED_VERTICES = 2_000_000
MAX_MODELED_FACES = 4_000_000

FEATURE_KINDS = frozenset({"extrude"})
PART_TYPE_VALUES = frozenset(part_type.value for part_type in PartType)


class ModelingRequestError(ValueError):
    """A request the client must fix; reported as a 400 with this message."""


def _database_path() -> Path:
    return Path(current_app.config["DATABASE"])


def _upload_folder() -> Path:
    return Path(current_app.config["UPLOAD_FOLDER"])


def _checked_feature(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ModelingRequestError("A modeled body needs its feature definition.")
    if raw.get("kind") not in FEATURE_KINDS:
        raise ModelingRequestError(f"Unknown feature kind {raw.get('kind')!r}.")
    if len(json.dumps(raw)) > MAX_FEATURE_BYTES:
        raise ModelingRequestError("The feature definition is too large.")
    return raw


def _checked_mesh(raw: Any) -> trimesh.Trimesh:
    """Validate the client's triangle mesh and build it.

    The arrays are untrusted input: every coordinate must be finite and every
    index must name a vertex, or the STL written from them would carry NaNs or
    crash the voxelizer much later and far from the cause.
    """
    if not isinstance(raw, dict):
        raise ModelingRequestError("A modeled body needs its mesh.")
    vertices_raw = raw.get("vertices")
    faces_raw = raw.get("faces")
    if not isinstance(vertices_raw, list) or not isinstance(faces_raw, list):
        raise ModelingRequestError("The mesh needs vertex and face arrays.")
    if len(vertices_raw) % 3 or len(faces_raw) % 3:
        raise ModelingRequestError("Mesh arrays must hold whole points and whole triangles.")
    if not vertices_raw or not faces_raw:
        raise ModelingRequestError("The modeled body is empty.")
    if len(vertices_raw) > MAX_MODELED_VERTICES * 3 or len(faces_raw) > MAX_MODELED_FACES * 3:
        raise ModelingRequestError("The modeled body has too many triangles.")
    try:
        vertices = np.asarray(vertices_raw, dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(faces_raw, dtype=np.int64).reshape(-1, 3)
    except (TypeError, ValueError) as exc:
        raise ModelingRequestError("Mesh arrays must be numbers.") from exc
    if not np.all(np.isfinite(vertices)):
        raise ModelingRequestError("The mesh has a non-finite coordinate.")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ModelingRequestError("A mesh face names a vertex that does not exist.")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    if len(mesh.faces) == 0:
        raise ModelingRequestError("The modeled body has no volume.")
    # The STL is welded by position wherever it is read -- here, by the
    # voxelizer and by the browser's next build -- so this weld is the one that
    # counts. A body that is not closed after it cannot be voxelized: its
    # containment test has no inside, and the part silently loses voxels.
    if not (mesh.is_watertight and mesh.is_winding_consistent):
        raise ModelingRequestError(
            "The modeled body is not a closed solid, so the simulation cannot voxelize it."
        )
    if not math.isfinite(float(mesh.volume)) or float(mesh.volume) <= 0:
        raise ModelingRequestError("The modeled body has no volume.")
    return mesh


def _checked_body_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    part_type = str(payload.get("part_type") or "cast")
    if part_type not in PART_TYPE_VALUES:
        raise ModelingRequestError(f"Unknown body type {part_type!r}.")
    name = str(payload.get("name") or "").strip()[:120] or "Extrude"
    return {
        "name": name,
        "part_type": part_type,
        # Open-top is a riser property; see lib/partTypes.ts supportsOpenTop.
        "is_open_top": bool(payload.get("is_open_top")) and part_type == "riser",
        "visible": bool(payload.get("visible", True)),
    }


def _write_body(mesh: trimesh.Trimesh, name: str) -> Path:
    upload_folder = _upload_folder()
    upload_folder.mkdir(parents=True, exist_ok=True)
    # The display name is user text; only its sanitized stem reaches the disk,
    # behind a random prefix, so two bodies never share a file.
    stem = _sanitize_body_name(name)[:60]
    filepath = upload_folder / f"{os.urandom(4).hex()}_b0_{stem}.stl"
    repo.resolve_upload_path(str(filepath), upload_folder)
    mesh.export(str(filepath))
    return filepath


def _parse(
    payload: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any], trimesh.Trimesh, Optional[trimesh.Trimesh]]:
    """The feature, body fields, mesh and optional ``cut_mesh`` of a body request.

    ``cut_mesh`` is the body after the extrude cuts that apply to it; when it
    is sent, ``mesh`` is stored as the body's pre-cut base.
    """
    if not isinstance(payload, dict):
        raise ModelingRequestError("Send the body as a JSON object.")
    feature = _checked_feature(payload.get("feature"))
    fields = _checked_body_fields(payload)
    mesh = _checked_mesh(payload.get("mesh"))
    cut_mesh = _checked_mesh(payload["cut_mesh"]) if payload.get("cut_mesh") is not None else None
    return feature, fields, mesh, cut_mesh


def _write_geometry(
    mesh: trimesh.Trimesh, cut_mesh: Optional[trimesh.Trimesh], name: str
) -> Tuple[Path, Optional[Path]]:
    """Write a body's STL and, when it is cut, its base. Returns ``(filepath, base)``."""
    base = _write_body(mesh, name)
    if cut_mesh is None:
        return base, None
    try:
        return _write_body(cut_mesh, name), base
    except Exception:
        base.unlink(missing_ok=True)
        raise


def _part_response(project_id: str, part_id: str, mesh: trimesh.Trimesh) -> Optional[Dict[str, Any]]:
    part = repo.get_part(_database_path(), project_id, part_id)
    if part is None:
        return None
    part["mesh_data"] = _build_mesh_data(mesh)
    part["watertight"] = bool(mesh.is_watertight)
    return part


@modeling_bp.route("/projects/<project_id>/features", methods=["POST"])
def create_feature_body(project_id: str):
    """Store a body modeled in the browser as a new part."""
    if repo.get_project(_database_path(), project_id) is None:
        return jsonify({"error": "Project not found"}), 404
    try:
        feature, fields, mesh, cut_mesh = _parse(request.get_json(silent=True))
    except ModelingRequestError as exc:
        return jsonify({"error": str(exc)}), 400

    filepath, base = _write_geometry(mesh, cut_mesh, fields["name"])
    try:
        part_id = repo.add_part(
            _database_path(),
            project_id,
            {**fields, "filepath": str(filepath)},
            upload_root=_upload_folder(),
            feature=feature,
        )
        if base is not None:
            repo.set_part_geometry(_database_path(), project_id, part_id, str(filepath), str(base))
    except Exception:
        filepath.unlink(missing_ok=True)
        if base is not None:
            base.unlink(missing_ok=True)
        current_app.logger.exception("Storing a modeled body failed")
        return jsonify({"error": "Could not store the modeled body."}), 500
    shown = cut_mesh if cut_mesh is not None else mesh
    return jsonify({"id": part_id, "part": _part_response(project_id, part_id, shown)}), 201


@modeling_bp.route("/projects/<project_id>/features/<part_id>", methods=["PUT"])
def update_feature_body(project_id: str, part_id: str):
    """Replace a modeled body's geometry, definition and body properties.

    The new mesh goes to a new file rather than over the old one: the viewport
    keys a part's triangles on its file path, so an edit that kept the path
    and the vertex count would never be redrawn.
    """
    database_path = _database_path()
    current = repo.get_part(database_path, project_id, part_id)
    if current is None:
        return jsonify({"error": "Part not found"}), 404
    try:
        feature, fields, mesh, cut_mesh = _parse(request.get_json(silent=True))
    except ModelingRequestError as exc:
        return jsonify({"error": str(exc)}), 400

    filepath, base = _write_geometry(mesh, cut_mesh, fields["name"])
    old_paths = [str(current.get("filepath") or ""), str(current.get("base_filepath") or "")]
    try:
        repo.update_part(
            database_path,
            project_id,
            part_id,
            {**fields, "filepath": str(filepath)},
            upload_root=_upload_folder(),
        )
        repo.set_part_feature(database_path, project_id, part_id, feature)
        repo.set_part_geometry(
            database_path, project_id, part_id, str(filepath), str(base) if base is not None else None
        )
    except Exception:
        filepath.unlink(missing_ok=True)
        if base is not None:
            base.unlink(missing_ok=True)
        current_app.logger.exception("Updating a modeled body failed")
        return jsonify({"error": "Could not update the modeled body."}), 500

    for old_path in old_paths:
        _discard_file(old_path)
    shown = cut_mesh if cut_mesh is not None else mesh
    return jsonify({"part": _part_response(project_id, part_id, shown)})


def _discard_file(filepath: str) -> None:
    """Delete a superseded body file unless another part still uses it."""
    if not filepath:
        return
    try:
        resolved = Path(repo.resolve_upload_path(filepath, _upload_folder()))
    except repo.UnsafePartFileError:
        return
    if repo.count_parts_using_file(_database_path(), str(resolved)) > 0:
        return
    resolved.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Extrude cuts
# ---------------------------------------------------------------------------

#: A project carries a handful of cuts; each is a sketch and a few numbers.
MAX_CUT_FEATURES = 200
MAX_CUT_LIST_BYTES = 4 * MAX_FEATURE_BYTES


@modeling_bp.route("/projects/<project_id>/parts/<part_id>/cut", methods=["PUT"])
def put_part_cut(project_id: str, part_id: str):
    """Store a body as the extrude cuts left it, or restore it uncut.

    ``{"mesh": {...}}`` is the body after every cut that applies to it, in the
    part's own frame, computed in the browser from the body's base. The STL
    the body had before its first cut becomes its base and is kept; a later
    cut replaces only the cut file. ``{"mesh": null}`` points the body back at
    its base, for a body no cut reaches any more.
    """
    database_path = _database_path()
    current = repo.get_part(database_path, project_id, part_id)
    if current is None:
        return jsonify({"error": "Part not found"}), 404
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or "mesh" not in payload:
        return jsonify({"error": "Send the cut body as {\"mesh\": ...}, or null to restore it."}), 400
    filepath = str(current.get("filepath") or "")
    base = str(current.get("base_filepath") or "")

    if payload["mesh"] is None:
        if base:
            repo.set_part_geometry(database_path, project_id, part_id, base, None)
            _discard_file(filepath)
        part = repo.get_part(database_path, project_id, part_id)
        if part is not None:
            mesh_data = load_part_mesh_data(str(part.get("filepath") or ""))
            if mesh_data is not None:
                part["mesh_data"] = mesh_data
        return jsonify({"part": part})

    try:
        mesh = _checked_mesh(payload["mesh"])
    except ModelingRequestError as exc:
        return jsonify({"error": str(exc)}), 400
    written = _write_body(mesh, str(current.get("name") or "Body"))
    try:
        repo.set_part_geometry(database_path, project_id, part_id, str(written), base or filepath)
    except Exception:
        written.unlink(missing_ok=True)
        current_app.logger.exception("Storing a cut body failed")
        return jsonify({"error": "Could not store the cut body."}), 500
    # The previous cut file; the uncut file is the base and stays.
    if base:
        _discard_file(filepath)
    return jsonify({"part": _part_response(project_id, part_id, mesh)})


@modeling_bp.route("/projects/<project_id>/parts/<part_id>/base-mesh")
def part_base_mesh(project_id: str, part_id: str):
    """The body's geometry before any extrude cut: its base, or its STL if uncut."""
    part = repo.get_part(_database_path(), project_id, part_id)
    if part is None:
        return jsonify({"error": "Part not found"}), 404
    mesh_data = load_part_mesh_data(str(part.get("base_filepath") or part.get("filepath") or ""))
    if mesh_data is None:
        return jsonify({"error": "Part geometry is unavailable"}), 404
    return jsonify({"mesh_data": mesh_data})


@modeling_bp.route("/projects/<project_id>/cuts", methods=["PUT"])
def put_cut_features(project_id: str):
    """Replace the project's extrude cut definitions (``ui_state["cut_features"]``).

    Only the definitions: the bodies each cut changes are stored through
    :func:`put_part_cut`, one by one, by the browser that computed them.
    """
    database_path = _database_path()
    if repo.get_project(database_path, project_id) is None:
        return jsonify({"error": "Project not found"}), 404
    payload = request.get_json(silent=True)
    cuts = payload.get("cuts") if isinstance(payload, dict) else None
    if not isinstance(cuts, list):
        return jsonify({"error": "Send the cuts as a list."}), 400
    if len(cuts) > MAX_CUT_FEATURES:
        return jsonify({"error": f"At most {MAX_CUT_FEATURES} cuts."}), 400
    for cut in cuts:
        if not (
            isinstance(cut, dict)
            and cut.get("kind") == "cut"
            and isinstance(cut.get("id"), str)
            and isinstance(cut.get("name"), str)
            and isinstance(cut.get("seq"), (int, float))
            and isinstance(cut.get("sketch"), dict)
            and isinstance(cut.get("partIds"), list)
            and all(isinstance(pid, str) for pid in cut["partIds"])
        ):
            return jsonify({"error": "Each cut needs a kind, an id, a name, a seq, a sketch and its bodies."}), 400
    if len(json.dumps(cuts)) > MAX_CUT_LIST_BYTES:
        return jsonify({"error": "The cut definitions are too large."}), 400
    repo.set_ui_state_key(database_path, project_id, "cut_features", cuts)
    return jsonify({"cuts": cuts})


#: A project carries a few reference planes, not thousands.
MAX_REFERENCE_PLANES = 200


@modeling_bp.route("/projects/<project_id>/reference-planes", methods=["PUT"])
def put_reference_planes(project_id: str):
    """Replace the project's reference planes (SOLIDWORKS-style offset planes).

    They live in the project's ``ui_state`` JSON under ``reference_planes``,
    next to the pour location, so duplicate, export and import carry them
    without a schema change. A dedicated route rather than the project PUT:
    the shell's settings autosave does not hold them and must not clear them.
    """
    database_path = _database_path()
    project = repo.get_project(database_path, project_id)
    if project is None:
        return jsonify({"error": "Project not found"}), 404
    payload = request.get_json(silent=True)
    planes = payload.get("planes") if isinstance(payload, dict) else None
    if not isinstance(planes, list):
        return jsonify({"error": "Send the planes as a list."}), 400
    if len(planes) > MAX_REFERENCE_PLANES:
        return jsonify({"error": f"At most {MAX_REFERENCE_PLANES} reference planes."}), 400
    for plane in planes:
        if not (
            isinstance(plane, dict)
            and isinstance(plane.get("id"), str)
            and isinstance(plane.get("name"), str)
            and isinstance(plane.get("source"), dict)
        ):
            return jsonify({"error": "Each plane needs an id, a name and a source."}), 400
    if len(json.dumps(planes)) > MAX_FEATURE_BYTES:
        return jsonify({"error": "The reference planes are too large."}), 400
    repo.set_ui_state_key(database_path, project_id, "reference_planes", planes)
    return jsonify({"planes": planes})
