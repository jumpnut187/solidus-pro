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
    grids, failure = _grids_or_error(project_id)
    if failure is not None:
        return failure
    assert grids is not None
    # Someone looking at a run's metadata is a few seconds away from asking
    # for an isosurface; get the contouring library on its feet now.
    _warm_contouring()
    return jsonify(fields.build_grid_metadata(grids))


@fields_bp.route("/projects/<project_id>/grid/mask")
def grid_mask(project_id: str):
    """
    Uint8 occupancy grid: 0 outside the metal, else the part-type code plus 1.

    ``+ 1`` because 0 has to mean "no metal here"; the code table in the
    metadata maps the stored byte back to a part type.
    """
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
    """
    Uint8 signed distance from each cell center to the parts' real surface.

    The mask says which voxels the mesher counted as metal; this says where
    the STL actually is, which on a coarse lattice is up to half a voxel
    inside the mask's edge. Distances are in voxels, clamped to
    ``+-X-Skin-Range-Voxels`` and mapped onto ``0..255`` with the surface at
    127.5, so sampled linearly the grid crosses one half exactly on the skin.

    404 when it cannot be built -- the part files are gone, or the lattice is
    too large to measure -- and the client clips against the mask instead.
    """
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
    """
    One dense scalar grid.

    Uint8 by default -- the client normalizes with the ``X-Field-Min`` and
    ``X-Field-Max`` headers (also present in the metadata) inside the shader.
    ``?dtype=f32`` returns little-endian float32 in field units instead.
    """
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
    """
    Consecutive time frames for an animated mode, as one Uint8 block.

    ``start`` and ``count`` slice the stack; the block is
    ``count * nx * ny * nz`` bytes, frame-major, each frame in the same
    x-fastest layout as a static grid. Frames are always 0..1 (fraction liquid,
    fill fraction), so 8 bits is the field's own resolution rather than a
    lossy choice, and the client can upload two adjacent frames and lerp
    between them in the shader.
    """
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
    """
    The threshold isosurface for a mode, as one binary block.

    ``threshold`` is in **field units**, not percent: the client already knows
    the range from the metadata and converting there keeps one definition of
    "30 percent" rather than two that can drift apart.

    The body is little-endian and laid out as positions, normals, indices and
    optionally per-vertex scalars, with counts in the headers. A hundred
    thousand triangle envelope is about 2 MB this way against 7 MB of JSON,
    and it arrives as typed arrays the renderer can hand straight to a vertex
    buffer.

    Query parameters:
      ``threshold``  field units (required in practice; defaults to 0.3)
      ``cast_only``  1 to contour only cast bodies
      ``scalars``    0 to skip the per-vertex sampling pass
      ``frame``      frame index, for the animated modes
    """
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


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


#: Contoured envelopes, keyed by everything that changes the geometry.
#:
#: Dragging a threshold slider asks for a new isosurface several times a
#: second; the client debounces, but a user sweeping back and forth still
#: revisits the same thresholds. Contouring the reference run takes tens of
#: milliseconds, so a handful of cached results turns a repeat into a lookup.
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
    """Drop every cached envelope. Used by tests."""
    _ENVELOPE_CACHE.clear()


#: Guards the one-shot contouring warm-up.
_WARMED = threading.Event()


def _warm_contouring() -> None:
    """
    Import VTK on a background thread, once.

    Contouring a million-cell grid takes about 12 ms; the *first* call takes
    about 950 ms, because that is when `import vtk` happens. The import is
    deliberately lazy in :mod:`core.surface_extraction` so the module can be
    used where VTK is absent, which leaves the cost falling on whoever asks
    for the first envelope -- and they asked by dragging a threshold slider,
    which is the worst possible moment to spend a second.

    Doing it here, off the request thread, means the cost lands while the
    user is still looking at the painted surface. Failures are ignored: this
    is a warm-up, and the real call reports properly if VTK is genuinely
    unavailable.
    """
    if _WARMED.is_set():
        return
    _WARMED.set()

    def load() -> None:
        try:
            import vtk  # noqa: F401
        except Exception:  # pragma: no cover - environments without VTK
            pass

    threading.Thread(target=load, name="castsim-vtk-warmup", daemon=True).start()


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


def _grids_or_error(project_id: str) -> Tuple[Optional[fields.FieldGrids], Optional[Response]]:
    """
    Load a run's grids, or the response explaining why not.

    The one failure worth spelling out is a lattice too large to build: it is
    not a bug and not a missing file, it is a run meshed too finely across too
    wide an envelope, and the only thing that fixes it is a coarser voxel
    size. Letting that escape as an unhandled exception gave a 500 with a
    stack trace, on every retry, forever.
    """
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
    if grids is None:
        response = jsonify({"error": "Results not found"})
        response.status_code = 404
        return None, response
    return grids, None


def _artifact_path(project_id: str) -> Optional[Path]:
    """The recorded artifact file for a run, if it exists on disk."""
    artifact_path = repo.get_artifact_path(_database_path(), project_id)
    if not artifact_path:
        return None
    path = Path(artifact_path)
    return path if path.exists() else None


def _in_memory_artifact(project_id: str) -> Optional[Dict[str, Any]]:
    """
    The just-finished run still held by the simulation queue.

    Reaching for this is expensive and is therefore the *last* resort. The
    queue manager lives in another process, so every call pickles the whole
    artifact -- 50 to 180 MB of result arrays -- across a socket, allocates it
    again on this side, and holds the queue's lock while it does, which is the
    same lock the worker needs to report progress. Doing that on every grid
    request cost 0.12 to 0.15 s and a 50 MB allocation per request and stalled
    the running simulation's progress updates.

    On disk, the artifact and its sidecar answer the same question for free.
    """
    manager = _queue_manager()
    if manager is None:
        return None
    return manager.get_result(project_id)


def _load_grids(project_id: str) -> Optional[fields.FieldGrids]:
    """
    The dense grids for a run.

    Ordered by cost: the mtime-keyed in-process cache and the on-disk sidecar
    first, and only if the run has no artifact on disk at all does this go
    asking the simulation queue for the in-memory copy.
    """
    path = _artifact_path(project_id)
    if path is not None:
        return fields.load_grids(project_id, path)

    artifact = _in_memory_artifact(project_id)
    if artifact is None:
        return None
    # No file yet: build straight from the in-memory result. There is no mtime
    # to key a cache on, so this rebuilds per request until the artifact lands.
    grids = fields.build_field_grids(artifact.get("visualization", {}), artifact.get("summary", {}))
    return grids if grids.voxel_count else None
