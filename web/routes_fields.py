from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast
import gzip
import threading

from flask import Blueprint, Response, current_app, jsonify, request

from web import fields
from web import repository as repo
from web import skin
from web.services.simulation_queue import SimulationQueueManager

fields_bp = Blueprint("web_fields", __name__, url_prefix="/api")

_GZIP_LEVEL = 1

_GZIP_MIN_BYTES = 4096

@fields_bp.route("/fields/health")
def fields_health():
    return jsonify({"ok": True, "blueprint": "web_fields"})

@fields_bp.route("/projects/<project_id>/grid")
def grid_metadata(project_id: str):
    grids, failure = _grids_or_error(project_id)
    if failure is not None:
        return failure
    assert grids is not None
    _warm_contouring()
    return jsonify(fields.build_grid_metadata(grids))

@fields_bp.route("/projects/<project_id>/grid/mask")
def grid_mask(project_id: str):
    grids, failure = _grids_or_error(project_id)
    if failure is not None:
        return failure
    assert grids is not None
    return _binary_response(
        fields.mask_bytes(grids),
        {
            "X-Grid-Dims": _dims_header(grids),
            "X-Grid-Dtype": "uint8",
        },
    )

@fields_bp.route("/projects/<project_id>/grid/skin")
def grid_skin(project_id: str):
    grids, failure = _grids_or_error(project_id)
    if failure is not None:
        return failure
    assert grids is not None
    parts = repo.get_parts(_database_path(), project_id)
    grid = skin.load_skin_grid(project_id, grids.lattice, grids.mask, parts, _artifact_path(project_id))
    if grid is None:
        return jsonify({"error": "No part geometry is available to measure this run against"}), 404
    return _binary_response(
        grid.tobytes(),
        {
            "X-Grid-Dims": _dims_header(grids),
            "X-Grid-Dtype": "uint8",
            "X-Skin-Range-Voxels": repr(float(grid.range_voxels)),
        },
    )

@fields_bp.route("/projects/<project_id>/grid/<mode>")
def grid_field(project_id: str, mode: str):
    grids, failure = _grids_or_error(project_id)
    if failure is not None:
        return failure
    assert grids is not None
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
    grids, failure = _grids_or_error(project_id)
    if failure is not None:
        return failure
    assert grids is not None
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

@fields_bp.route("/projects/<project_id>/grid/<mode>/isosurface")
def grid_isosurface(project_id: str, mode: str):
    grids, failure = _grids_or_error(project_id)
    if failure is not None:
        return failure
    assert grids is not None

    known = mode in grids.scalars or mode in grids.stacks
    if not known:
        return jsonify({"error": f"Mode '{mode}' has no data for this run"}), 404

    try:
        threshold = float(request.args.get("threshold", "0.3"))
    except ValueError:
        return jsonify({"error": "threshold must be a number"}), 400
    frame_arg = request.args.get("frame")
    try:
        frame_index = int(frame_arg) if frame_arg is not None else None
    except ValueError:
        return jsonify({"error": "frame must be an integer"}), 400

    cast_only = request.args.get("cast_only") in {"1", "true", "True"}
    with_scalars = request.args.get("scalars", "1") not in {"0", "false", "False"}

    envelope = _cached_envelope(
        project_id,
        grids,
        mode,
        threshold,
        cast_only=cast_only,
        with_scalars=with_scalars,
        frame_index=frame_index,
    )
    if envelope is None:
        return jsonify({"error": f"Mode '{mode}' cannot be contoured for this run"}), 404

    return _binary_response(
        fields.envelope_bytes(envelope),
        {
            "X-Vertex-Count": str(envelope.vertex_count),
            "X-Index-Count": str(int(envelope.indices.size)),
            "X-Has-Scalar": "1" if envelope.scalars is not None else "0",
            "X-Field-Mode": mode,
            "X-Field-Min": repr(float(envelope.minimum)),
            "X-Field-Max": repr(float(envelope.maximum)),
            "X-Field-Unit": fields.MODE_UNITS.get(mode, ""),
            "X-Threshold": repr(float(envelope.threshold)),
        },
    )

_ENVELOPE_CACHE: "OrderedDict[Tuple[Any, ...], fields.Envelope]" = OrderedDict()
_ENVELOPE_CACHE_MAX = 12

def _cached_envelope(
    project_id: str,
    grids: fields.FieldGrids,
    mode: str,
    threshold: float,
    *,
    cast_only: bool,
    with_scalars: bool,
    frame_index: Optional[int],
) -> Optional[fields.Envelope]:
    key = (
        project_id,
        id(grids),
        mode,
        round(float(threshold), 6),
        cast_only,
        with_scalars,
        frame_index,
    )
    cached = _ENVELOPE_CACHE.get(key)
    if cached is not None:
        _ENVELOPE_CACHE.move_to_end(key)
        return cached

    envelope = fields.build_envelope(
        grids,
        mode,
        threshold,
        cast_only=cast_only,
        with_scalars=with_scalars,
        frame_index=frame_index,
    )
    if envelope is None:
        return None

    _ENVELOPE_CACHE[key] = envelope
    while len(_ENVELOPE_CACHE) > _ENVELOPE_CACHE_MAX:
        _ENVELOPE_CACHE.popitem(last=False)
    return envelope

def clear_envelope_cache() -> None:
    _ENVELOPE_CACHE.clear()

_WARMED = threading.Event()

def _warm_contouring() -> None:
    if _WARMED.is_set():
        return
    _WARMED.set()

    def load() -> None:
        try:
            import vtk
        except Exception:
            pass

    threading.Thread(target=load, name="castsim-vtk-warmup", daemon=True).start()

def _dims_header(grids: fields.FieldGrids) -> str:
    nx, ny, nz = grids.lattice.dims
    return f"{int(nx)},{int(ny)},{int(nz)}"

def _binary_response(payload: bytes, headers: Dict[str, str]) -> Response:
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

def _grids_or_error(project_id: str) -> Tuple[Optional[fields.FieldGrids], Optional[Response]]:
    try:
        grids = _load_grids(project_id)
    except fields.LatticeTooLargeError as exc:
        response = jsonify(
            {
                "error": (
                    "This run's result lattice is too large to display "
                    f"({exc.cell_count:,} cells at {exc.spacing:g} in). "
                    "Re-run it at a coarser voxel size."
                ),
                "code": "lattice_too_large",
                "dims": list(exc.dims),
                "cellCount": exc.cell_count,
                "spacing": exc.spacing,
                "maxCellCount": fields.MAX_LATTICE_CELLS,
            }
        )
        response.status_code = 413
        return None, response
    except fields.RemovedTetResultError as exc:
        response = jsonify({"error": str(exc), "code": "removed_mesh_type"})
        response.status_code = 410
        return None, response
    if grids is None:
        response = jsonify({"error": "Results not found"})
        response.status_code = 404
        return None, response
    return grids, None

def _artifact_path(project_id: str) -> Optional[Path]:
    artifact_path = repo.get_artifact_path(_database_path(), project_id)
    if not artifact_path:
        return None
    path = Path(artifact_path)
    return path if path.exists() else None

def _in_memory_artifact(project_id: str) -> Optional[Dict[str, Any]]:
    manager = _queue_manager()
    if manager is None:
        return None
    return manager.get_result(project_id)

def _load_grids(project_id: str) -> Optional[fields.FieldGrids]:
    path = _artifact_path(project_id)
    if path is not None:
        return fields.load_grids(project_id, path)

    artifact = _in_memory_artifact(project_id)
    if artifact is None:
        return None
    grids = fields.build_field_grids(artifact.get("visualization", {}), artifact.get("summary", {}))
    return grids if grids.voxel_count else None
