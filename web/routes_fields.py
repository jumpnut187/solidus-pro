"""Result-field routes for the web viewport: dense 3D grids, frames, isosurfaces.

Owned by the rendering/data-pipeline workstream. Registered by
``web.app_factory.create_app``. Everything here lives under ``/api``.

The grid endpoints serve the dense lattice built by :mod:`web.fields`. Metadata
is JSON; every grid and frame body is a raw little-endian typed array with
``Content-Type: application/octet-stream``, because the client's destination is
a GPU texture and any JSON step between here and there costs more than the
transfer itself. Layout is x-fastest -- ``index = ix + nx * (iy + ny * iz)`` --
which is a WebGL 3D texture of width ``nx`` with no shuffling on arrival.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast
import gzip

from flask import Blueprint, Response, current_app, jsonify, request

from web import fields
from web import repository as repo
from web.services.simulation_queue import SimulationQueueManager

fields_bp = Blueprint("web_fields", __name__, url_prefix="/api")


#: Compression level for binary bodies. These are byte grids of highly
#: repetitive data (a mask is a handful of distinct values; a quantized field
#: is smooth), so level 1 already gets most of the ratio and keeps the
#: per-request CPU cost off the critical path for a 10 MB frame stack.
_GZIP_LEVEL = 1

#: Below this a response is not worth compressing at all.
_GZIP_MIN_BYTES = 4096


@fields_bp.route("/fields/health")
def fields_health():
    return jsonify({"ok": True, "blueprint": "web_fields"})


# --------------------------------------------------------------------------
# Grid endpoints
# --------------------------------------------------------------------------


@fields_bp.route("/projects/<project_id>/grid")
def grid_metadata(project_id: str):
    """
    Everything the viewport needs before it asks for a single byte of field.

    Dimensions and placement of the lattice, the modes this run actually has
    with their ranges and units, frame counts and times in seconds, the
    part-type code table behind the mask, voxel counts, total volume, and the
    run's summary statistics flattened to scalars.
    """
    grids = _load_grids(project_id)
    if grids is None:
        return jsonify({"error": "Results not found"}), 404
    return jsonify(fields.build_grid_metadata(grids))


@fields_bp.route("/projects/<project_id>/grid/mask")
def grid_mask(project_id: str):
    """
    Uint8 occupancy grid: 0 outside the metal, else the part-type code plus 1.

    ``+ 1`` because 0 has to mean "no metal here"; the code table in the
    metadata maps the stored byte back to a part type.
    """
    grids = _load_grids(project_id)
    if grids is None:
        return jsonify({"error": "Results not found"}), 404
    return _binary_response(
        fields.mask_bytes(grids),
        {
            "X-Grid-Dims": _dims_header(grids),
            "X-Grid-Dtype": "uint8",
        },
    )


@fields_bp.route("/projects/<project_id>/grid/<mode>")
def grid_field(project_id: str, mode: str):
    """
    One dense scalar grid.

    Uint8 by default -- the client normalizes with the ``X-Field-Min`` and
    ``X-Field-Max`` headers (also present in the metadata) inside the shader.
    ``?dtype=f32`` returns little-endian float32 in field units instead.
    """
    grids = _load_grids(project_id)
    if grids is None:
        return jsonify({"error": "Results not found"}), 404
    grid = grids.scalars.get(mode)
    if grid is None:
        return jsonify({"error": f"Mode '{mode}' has no data for this run"}), 404

    dtype = "f32" if request.args.get("dtype", "u8").lower() in {"f32", "float32"} else "u8"
    return _binary_response(
        fields.scalar_bytes(grid, dtype),
        {
            "X-Grid-Dims": _dims_header(grids),
            "X-Grid-Dtype": "float32" if dtype == "f32" else "uint8",
            "X-Field-Mode": mode,
            "X-Field-Min": repr(float(grid.minimum)),
            "X-Field-Max": repr(float(grid.maximum)),
            "X-Field-Mean": repr(float(grid.mean)),
            "X-Field-Unit": fields.MODE_UNITS.get(mode, ""),
        },
    )


@fields_bp.route("/projects/<project_id>/grid/<mode>/frames")
def grid_frames(project_id: str, mode: str):
    """
    Consecutive time frames for an animated mode, as one Uint8 block.

    ``start`` and ``count`` slice the stack; the block is
    ``count * nx * ny * nz`` bytes, frame-major, each frame in the same
    x-fastest layout as a static grid. Frames are always 0..1 (fraction liquid,
    fill fraction), so 8 bits is the field's own resolution rather than a
    lossy choice, and the client can upload two adjacent frames and lerp
    between them in the shader.
    """
    grids = _load_grids(project_id)
    if grids is None:
        return jsonify({"error": "Results not found"}), 404
    stack = grids.stacks.get(mode)
    if stack is None:
        return jsonify({"error": f"Mode '{mode}' has no frames for this run"}), 404

    try:
        start = int(request.args.get("start", "0"))
        count = int(request.args.get("count", str(stack.count)))
    except ValueError:
        return jsonify({"error": "start and count must be integers"}), 400

    payload, begin, length = fields.frame_bytes(stack, start, count)
    return _binary_response(
        payload,
        {
            "X-Grid-Dims": _dims_header(grids),
            "X-Grid-Dtype": "uint8",
            "X-Field-Mode": mode,
            "X-Field-Min": repr(float(stack.minimum)),
            "X-Field-Max": repr(float(stack.maximum)),
            "X-Frame-Start": str(begin),
            "X-Frame-Count": str(length),
            "X-Frame-Total": str(stack.count),
        },
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _dims_header(grids: fields.FieldGrids) -> str:
    nx, ny, nz = grids.lattice.dims
    return f"{int(nx)},{int(ny)},{int(nz)}"


def _binary_response(payload: bytes, headers: Dict[str, str]) -> Response:
    """
    Send a typed array, gzipped when the client asked for it.

    ``Vary: Accept-Encoding`` is not optional here: these URLs are cacheable
    and a proxy that handed a gzipped body to a client that did not ask for
    one would deliver a texture full of noise.
    """
    body = payload
    all_headers = dict(headers)
    all_headers["Vary"] = "Accept-Encoding"
    if len(payload) >= _GZIP_MIN_BYTES and "gzip" in request.headers.get("Accept-Encoding", "").lower():
        body = gzip.compress(payload, _GZIP_LEVEL)
        all_headers["Content-Encoding"] = "gzip"
    all_headers["Content-Length"] = str(len(body))
    all_headers["X-Uncompressed-Length"] = str(len(payload))
    return Response(body, mimetype="application/octet-stream", headers=all_headers)


def _database_path() -> Path:
    return Path(current_app.config["DATABASE"])


def _queue_manager() -> Optional[SimulationQueueManager]:
    manager = current_app.extensions.get("simulation_queue_manager")
    if manager is None:
        return None
    return cast(SimulationQueueManager, manager)


def _resolve_artifact(project_id: str) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """
    Find a run's results as a path, an in-memory artifact, or both.

    A run that just finished lives in the queue manager before it is on disk;
    an older run is only a path. The path is preferred when it exists because
    it is what makes the sidecar and the mtime-keyed cache work -- an
    in-memory artifact has no mtime to key on.
    """
    artifact_path = repo.get_artifact_path(_database_path(), project_id)
    path = Path(artifact_path) if artifact_path else None
    if path is not None and not path.exists():
        path = None

    manager = _queue_manager()
    artifact = manager.get_result(project_id) if manager is not None else None
    if path is None and artifact is None:
        return None, None
    return path, artifact


def _load_grids(project_id: str) -> Optional[fields.FieldGrids]:
    path, artifact = _resolve_artifact(project_id)
    if path is None and artifact is None:
        return None
    if path is None:
        # No file yet: build straight from the in-memory result. Nothing to
        # key a cache on, so this rebuilds per request until the run's
        # artifact lands on disk.
        assert artifact is not None
        grids = fields.build_field_grids(artifact.get("visualization", {}), artifact.get("summary", {}))
        return grids if grids.voxel_count else None
    return fields.load_grids(project_id, path, artifact=artifact)
