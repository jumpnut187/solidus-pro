"""Babylon-oriented payload helpers for the web application."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast
import json
import math
import os
import re
import tempfile

import numpy as np
import trimesh
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from core.part_manager import (
    PartManager,
    PartType,
    STEP_SUPPORTED,
    _scale_step_to_inches,
    build_part_manager as _core_build_part_manager,
    infer_part_type_from_filename,
    infer_part_type_from_name,
    tessellation_tolerance_mm,
)
from core.simulation_pipeline import PipelineResult
from web.repository import DEFAULT_VOXEL_SIZE_IN, strip_upload_prefix


# Extensions accepted by upload_cad_file. STL is handled natively by
# trimesh; STEP requires the optional ``cascadio`` package.
ALLOWED_CAD_EXTENSIONS = {".stl", ".step", ".stp"}
_STEP_EXTENSIONS = (".step", ".stp")


def _format_dimensions(bounds: np.ndarray) -> str:
    """Format bounds as a 'X" x Y" x Z"' human-readable string.

    Accepts trimesh's native ``(2, 3)`` bounds format
    (``[[xmin, ymin, zmin], [xmax, ymax, zmax]]``) and computes the
    per-axis extents.
    """
    if bounds is None:
        return "N/A"
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (2, 3):
        raise ValueError(
            f"Expected trimesh bounds of shape (2, 3), got {bounds.shape}"
        )
    x_size = float(bounds[1][0] - bounds[0][0])
    y_size = float(bounds[1][1] - bounds[0][1])
    z_size = float(bounds[1][2] - bounds[0][2])
    return f"{x_size:.2f}\" x {y_size:.2f}\" x {z_size:.2f}\""


def _build_mesh_data(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    """Build the mesh_data payload section from a trimesh object.

    ``mesh.bounds`` is shape ``(2, 3)`` natively; ``_bounds_payload``
    expects the legacy ``(3, 2)`` per-axis shape (matching what the
    frontend already consumes), so we transpose here.
    """
    bounds_native = np.asarray(mesh.bounds, dtype=np.float64)  # (2, 3)
    bounds_per_axis = bounds_native.T  # (3, 2): [xmin,xmax], [ymin,ymax], [zmin,zmax]
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    return {
        "vertices": vertices.reshape(-1).tolist(),
        "faces": faces.reshape(-1).tolist(),
        "bounds": _bounds_payload(bounds_per_axis),
        "volume": float(mesh.volume) if mesh.volume else 0.0,
        "surface_area": float(mesh.area) if mesh.area else 0.0,
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
    }


#: The part fields a simulation depends on. A change to any of them after a
#: run means the run no longer describes the model. Mirrored in the client's
#: ``simulatedPartsKey`` (web_frontend/src/lib/resultsFreshness.ts).
SIMULATED_PART_FIELDS = (
    "id",
    "filepath",
    "part_type",
    "visible",
    "is_open_top",
    "position_x",
    "position_y",
    "position_z",
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "scale",
)


def simulated_parts_snapshot(parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The parts a run was given, reduced to the fields the run depends on.

    Stored in the run's summary. Hidden parts are listed with ``visible``
    false rather than dropped, so un-hiding one also marks the run stale.
    """
    snapshot: List[Dict[str, Any]] = []
    for part in parts:
        entry: Dict[str, Any] = {}
        for field in SIMULATED_PART_FIELDS:
            value = part.get(field)
            if field in ("visible", "is_open_top"):
                value = bool(value if value is not None else field == "visible")
            elif field.startswith(("position_", "rotation_")) or field == "scale":
                value = float(value if value is not None else (1.0 if field == "scale" else 0.0))
            else:
                value = str(value or "")
            entry[field] = value
        snapshot.append(entry)
    return snapshot


def _sanitize_body_name(name: str) -> str:
    """Sanitize a CAD body name for safe use in a filename.

    Strips any file extension, replaces runs of unsafe characters with
    a single underscore, and falls back to ``"body"`` for empty input.
    """
    name = os.path.splitext(name)[0]
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return sanitized or "body"


def upload_voxel_size_in(value: Union[str, float, None]) -> float:
    """The voxel size (inches) a STEP upload is tessellated for.

    ``value`` is the optional ``voxel_size_in`` upload form field. Anything
    that is not a positive finite number is ignored in favour of
    :data:`DEFAULT_VOXEL_SIZE_IN`.
    """
    if value is None:
        return DEFAULT_VOXEL_SIZE_IN
    try:
        voxel_size_in = float(value)
    except (TypeError, ValueError):
        return DEFAULT_VOXEL_SIZE_IN
    if not math.isfinite(voxel_size_in) or voxel_size_in <= 0.0:
        return DEFAULT_VOXEL_SIZE_IN
    return voxel_size_in


def upload_cad_file(
    file: FileStorage,
    upload_folder: Path,
    voxel_size_in: Union[str, float, None] = None,
) -> List[Dict[str, Any]]:
    """Load a CAD file (STL or STEP) and return one file entry per body.

    The function unifies STL and STEP handling on top of ``trimesh``.
    A single STL produces a one-element list; a multi-body STEP
    assembly produces one entry per body, with each body exported to
    its own STL file on disk before the response is built. Body names
    from the STEP B-rep are preserved in the response under
    ``body_name`` and used for ``part_type`` inference.

    Every entry's ``filepath`` points at its exported STL. The original
    upload is removed after fanning out for every STEP file and for any
    multi-body file; a single-body STL keeps its original on disk.

    ``voxel_size_in`` sets the STEP tessellation tolerance (see
    :func:`upload_voxel_size_in`); it is ignored for STL.

    Raises:
        ValueError: on missing file, unsupported extension, missing
            ``cascadio`` (when a STEP file is uploaded), or load
            failure.
    """
    if not file.filename:
        raise ValueError("No file selected")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_CAD_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Allowed: {sorted(ALLOWED_CAD_EXTENSIONS)}"
        )

    upload_folder.mkdir(parents=True, exist_ok=True)
    file_id = os.urandom(4).hex()
    filename = secure_filename(file.filename)
    filepath = upload_folder / f"{file_id}_{filename}"
    file.save(str(filepath))

    # STEP-specific guard: fail fast with a clear install hint rather
    # than letting trimesh raise an opaque ImportError downstream.
    if ext in _STEP_EXTENSIONS and not STEP_SUPPORTED:
        try:
            filepath.unlink()
        except FileNotFoundError:
            pass
        raise ValueError(
            "STEP support requires the 'cascadio' package. "
            "Install with: pip install cascadio"
        )

    is_step = ext in _STEP_EXTENSIONS
    try:
        # force="scene" ensures trimesh returns a Scene even for STL
        # files, so we can iterate uniformly without a type branch.
        if is_step:
            # Tessellate for the voxel size rather than at cascadio's 0.01 mm
            # default (see core.part_manager.tessellation_tolerance_mm). The
            # bodies are stored as STL below, so they are not re-tessellated
            # if the project's voxel size later becomes finer: re-upload the
            # STEP to get the finer tessellation.
            tol_linear = tessellation_tolerance_mm(upload_voxel_size_in(voxel_size_in))
            loaded = trimesh.load(str(filepath), force="scene", tol_linear=tol_linear)
        else:
            loaded = trimesh.load(str(filepath), force="scene")
    except Exception as exc:
        try:
            filepath.unlink()
        except FileNotFoundError:
            pass
        raise ValueError(f"Failed to load CAD file: {exc}") from exc

    # STEP files arrive in meters (cascadio's GLB intermediate). The
    # rest of the web pipeline — the persisted STL exports, the
    # ``mesh_data`` payload, the dimensions string, the volume/area
    # numbers — assumes inches (matching the existing STL path). Apply
    # the meters→inches rescale at the single chokepoint so every
    # downstream value is already in inches. Passing the path lets the
    # rescale check the result against the file's declared length unit.
    if is_step:
        _scale_step_to_inches(loaded, str(filepath))

    if isinstance(loaded, trimesh.Scene):
        body_items = list(loaded.geometry.items())
        if not body_items:
            try:
                filepath.unlink()
            except FileNotFoundError:
                pass
            raise ValueError("CAD assembly contains no bodies")
    else:
        # Defensive: trimesh.load(force="scene") should always return a
        # Scene, but if a backend bypasses that (e.g. raises) we wrap.
        body_items = [("Body_1", loaded)]

    # For a single-body file, use the file's basename (without
    # extension) as the body name so the part_type inference and the
    # fan-out filename read cleanly. For multi-body files, trimesh
    # already provides real CAD body names that we trust. As in
    # core.part_manager, only bodies with faces count, so empty
    # construction bodies do not turn a one-part file into an assembly.
    nonempty_indices = [
        index
        for index, (_, body_mesh) in enumerate(body_items)
        if body_mesh is not None and len(body_mesh.faces) > 0
    ]
    if len(nonempty_indices) == 1:
        fallback_name = strip_upload_prefix(os.path.splitext(filename)[0])
        only_index = nonempty_indices[0]
        body_items[only_index] = (fallback_name, body_items[only_index][1])

    entries: List[Dict[str, Any]] = []
    for body_index, (body_name, body_mesh) in enumerate(body_items):
        if body_mesh is None or len(body_mesh.faces) == 0:
            # Skip degenerate bodies but keep going — a STEP assembly
            # may contain construction geometry that isn't a real part.
            continue

        # Fan out each body as its own STL file. Downstream code
        # expects STL on disk regardless of the original upload format,
        # so this preserves the existing pipeline without changes.
        # ``body_name`` for a single-body upload has already had any stored
        # prefix stripped; a STEP body name never carries one. Strip again so
        # the on-disk name gains exactly one prefix layer no matter what.
        display_name = strip_upload_prefix(_sanitize_body_name(body_name))
        body_filename = f"{file_id}_b{body_index}_{display_name}.stl"
        body_path = upload_folder / body_filename
        body_mesh.export(str(body_path))

        # Single-body files already have a clean name (the file's
        # basename); multi-body files use the CAD body name.
        part_type = infer_part_type_from_name(body_name).value

        entries.append(
            {
                "file_id": f"{file_id}_b{body_index}",
                "filename": body_filename,
                # The name the UI should show: the original file (or CAD body)
                # stem, free of the storage prefix.
                "display_name": display_name,
                "filepath": str(body_path),
                "source_file": filename,
                "body_name": body_name,
                "part_type": part_type,
                "dimensions": _format_dimensions(body_mesh.bounds),
                "mesh_data": _build_mesh_data(body_mesh),
            }
        )

    if not entries:
        try:
            filepath.unlink()
        except FileNotFoundError:
            pass
        raise ValueError("CAD file contains no loadable bodies")

    # Once each body is on disk as an STL the original STEP is redundant:
    # no entry points at it. Remove it to avoid keeping a 100MB+ STEP
    # around just for traceability. A single-body STL original is kept.
    if is_step or len(entries) > 1:
        try:
            filepath.unlink()
        except FileNotFoundError:
            pass

    return entries


def load_part_mesh_data(filepath: str) -> Optional[Dict[str, Any]]:
    """Re-read a stored part file and rebuild its ``mesh_data`` payload.

    Upload responses carry the triangle soup the viewport needs, but nothing
    persisted it: a browser refresh dropped every part's geometry because the
    project row only stores the path on disk. This reads that path back so a
    reloaded project renders identically to a freshly uploaded one.

    Returns ``None`` when the file is missing or unreadable rather than
    raising, so one broken part cannot fail a whole project load.
    """
    if not filepath:
        return None
    path = Path(filepath)
    if not path.exists():
        return None
    try:
        loaded = trimesh.load(str(path), force="mesh")
    except Exception:
        return None
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        return None
    return _build_mesh_data(loaded)


def build_part_manager(parts_data: list[Dict[str, Any]]) -> PartManager:
    """Build a :class:`PartManager` from serialized web payload data.

    Re-export of :func:`core.part_manager.build_part_manager` so existing
    web callers keep their import path. The canonical helper honours the
    ``visible`` and ``is_open_top`` per-part overrides that web payloads
    send.
    """
    return build_part_manager_from_core(parts_data, preserve_origin=True)


def build_part_manager_from_core(
    parts_data: list[Dict[str, Any]],
    *,
    preserve_origin: bool = True,
) -> PartManager:
    return _core_build_part_manager(
        parts_data,
        preserve_origin=preserve_origin,
        default_visible=True,
        default_is_open_top=False,
    )


def build_fork_artifact(result: PipelineResult) -> Dict[str, Any]:
    visualization = dict(result.visualization_data)
    engine = result.engine
    mesh = result.voxel_mesh

    centers = np.asarray(visualization.get("centers", []), dtype=np.float32)
    source_centers = np.asarray(mesh.voxel_centers, dtype=np.float32) if mesh.voxel_centers is not None else centers
    indices = _matching_indices(source_centers, centers)

    if mesh.voxel_part_types is not None:
        visualization["part_types"] = np.asarray(mesh.voxel_part_types, dtype=np.int32)[indices].tolist()

    field_specs = {
        "feed_modulus": getattr(engine, "local_modulus", None),
        "hot_tear": getattr(engine, "hot_tear_risk", None),
        "sand_burn_in": getattr(engine, "sand_burn_in_risk", None),
        "feeding_quality": _invert(getattr(engine, "feeding_quality", None)),
        "niyama_raw": getattr(engine, "niyama_values", None),
        "niyama_risk": getattr(engine, "niyama_risk", None),
        "combined_feed_risk": getattr(engine, "combined_feed_risk", None),
        "thermal_stress": _normalize(getattr(engine, "effective_stress", None)),
        "distortion": _normalize(getattr(engine, "distortion_magnitude", None)),
        "crack_susceptibility": getattr(engine, "crack_susceptibility", None),
        "sdas": _normalize(getattr(engine, "sdas", None)),
        "predicted_uts": _invert(_normalize(getattr(engine, "predicted_uts", None))),
        "porosity": _display_field(engine, "get_microporosity_display_intensity", "porosity_probability"),
        "feeding_blocked": getattr(engine, "feeding_blocked", None),
        "feeding_path_lengths": getattr(engine, "feeding_path_lengths", None),
        "feeding_open_fraction": getattr(engine, "feeding_open_fraction", None),
    }
    raw_ranges: Dict[str, Dict[str, float]] = {}
    part_types = getattr(mesh, "voxel_part_types", None)
    for key, values in field_specs.items():
        if values is None:
            continue
        arr = np.asarray(values, dtype=np.float32)
        if len(arr) != len(source_centers):
            continue
        payload_arr = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
        # Feed modulus is only meaningful for cast metal; exclude chills.
        if key == "feed_modulus" and part_types is not None and len(part_types) == len(source_centers):
            chill_mask = np.asarray(part_types, dtype=np.int32) == 2
            if np.any(chill_mask):
                payload_arr = payload_arr.copy()
                payload_arr[chill_mask] = 0.0
        visualization[key] = payload_arr[indices].tolist()
        raw_ranges[key] = _range_payload(arr)

    summary = dict(result.result_data)
    niyama_metadata = getattr(engine, "niyama_metadata", None)
    if isinstance(niyama_metadata, dict):
        summary["niyama_metadata"] = dict(niyama_metadata)
    summary["raw_ranges"] = raw_ranges

    return {
        "summary": summary,
        "visualization": visualization,
        # Full-resolution arrays for the dense grid sidecar, keyed privately
        # so nothing mistakes them for part of the artifact. `visualization`
        # above is *strided* when a run has more voxels than
        # ``max_voxels_for_visualization``: it keeps every step-th voxel, which
        # is fine for a downsampled JSON but ruinous for an occupancy grid,
        # because the cells in between are then marked as "not metal" and the
        # casting renders as a sieve. These are references to the arrays the
        # engine and mesh already hold, not copies, so carrying them costs
        # nothing until the sidecar is built from them.
        GRID_SOURCE_KEY: _full_grid_source(engine, mesh, source_centers),
    }


#: Private key on the in-memory artifact carrying full-resolution grid data.
#: Stripped before the JSON is written; never present in a file on disk.
GRID_SOURCE_KEY = "_grid_source"

def _full_grid_source(engine: Any, mesh: Any, source_centers: np.ndarray) -> Dict[str, Any]:
    """
    The unstrided arrays the dense grid needs, straight off the engine.

    Mirrors the keys :mod:`web.fields` reads, at full mesh resolution. Values
    stay as numpy arrays: this dictionary never reaches JSON, and converting a
    third of a million floats to Python lists for something that is about to
    be quantized to bytes would be the most expensive thing in the pipeline.
    """
    count = int(len(source_centers))
    source: Dict[str, Any] = {
        "centers": source_centers,
        "sizes": _full_array(getattr(mesh, "voxel_sizes", None), count),
        "part_types": _full_array(getattr(mesh, "voxel_part_types", None), count),
    }
    # Shrink and Microporosity: the same display arrays the desktop paints.
    for key, method_name, attribute in (
        ("shrinkage", "get_shrink_display_intensity", "shrinkage_probability"),
        ("porosity", "get_microporosity_display_intensity", "porosity_probability"),
    ):
        values = _full_array(_display_field(engine, method_name, attribute), count)
        if values is not None:
            source[key] = values

    for key, attribute in (
        ("feed_modulus", "local_modulus"),
        ("hotspot", "hotspot_intensity"),
        ("solidification_times", "solidification_times"),
        ("niyama_raw", "niyama_values"),
        ("niyama_risk", "niyama_risk"),
        ("combined_feed_risk", "combined_feed_risk"),
        ("hot_tear", "hot_tear_risk"),
        ("sand_burn_in", "sand_burn_in_risk"),
        ("gas_generation", "gas_generation"),
        ("gas_entrapment", "gas_entrapment"),
        ("gas_porosity_risk", "gas_porosity_risk"),
        ("crack_susceptibility", "crack_susceptibility"),
        ("feeding_blocked", "feeding_blocked"),
        ("feeding_path_lengths", "feeding_path_lengths"),
        ("feeding_open_fraction", "feeding_open_fraction"),
        ("fill_time", "_fill_arrival_times"),
    ):
        values = _full_array(getattr(engine, attribute, None), count)
        if values is not None:
            source[key] = values

    # These three are derived rather than read straight off an attribute.
    feeding_quality = _invert(getattr(engine, "feeding_quality", None))
    if _full_array(feeding_quality, count) is not None:
        source["feeding_quality"] = np.asarray(feeding_quality, dtype=np.float32)
    for key, attribute in (("thermal_stress", "effective_stress"), ("distortion", "distortion_magnitude")):
        normalized = _normalize(getattr(engine, attribute, None))
        if _full_array(normalized, count) is not None:
            source[key] = np.asarray(normalized, dtype=np.float32)
    sdas = _normalize(getattr(engine, "sdas", None))
    if _full_array(sdas, count) is not None:
        source["sdas"] = np.asarray(sdas, dtype=np.float32)
    predicted_uts = _invert(_normalize(getattr(engine, "predicted_uts", None)))
    if _full_array(predicted_uts, count) is not None:
        source["predicted_uts"] = np.asarray(predicted_uts, dtype=np.float32)

    history = getattr(engine, "fraction_liquid_history", None)
    if history:
        frames = [np.asarray(1.0 - np.asarray(snapshot), dtype=np.float32) for snapshot in history]
        if all(frame.size == count for frame in frames):
            mask_liquid = getattr(engine, "_mask_non_cast_fraction_liquid", None)
            if callable(mask_liquid):
                frames = [np.asarray(mask_liquid(frame), dtype=np.float32) for frame in frames]
            source["fraction_liquid_history"] = frames
            source["fraction_liquid_times"] = list(getattr(engine, "fraction_liquid_times", None) or [])

    fill_frames = getattr(engine, "_fill_frames", None)
    if fill_frames:
        fractions = []
        times = []
        for frame in fill_frames:
            values = _full_array(getattr(frame, "fill_fraction", None), count)
            if values is None:
                continue
            fractions.append(np.clip(values, 0.0, 1.0).astype(np.float32))
            times.append(float(getattr(frame, "time_s", 0.0)))
        if fractions:
            source["fill_animation_history"] = fractions
            source["fill_animation_times"] = times

    return source


def _full_array(values: Any, count: int) -> Optional[np.ndarray]:
    """The array, as float32, only if it really covers every voxel."""
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return arr if arr.size == count else None


def read_artifact(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return cast(Dict[str, Any], json.load(handle))


def write_artifact(path: Path, artifact: Dict[str, Any]) -> None:
    """
    Persist a completed run, plus the dense grid sidecar the viewport reads.

    The sidecar is written here, once, at the only moment the full artifact is
    already in memory. Building it lazily on first request works (see
    ``web.fields.load_grids``) but means the first person to open a finished
    run waits out a 50-100 MB JSON parse.

    The JSON goes to a temporary file and is then renamed over the target.
    Truncating the real path and streaming a hundred megabytes into it leaves
    a window -- seconds, on a large run -- in which any reader gets a partial
    document; re-running a project while its results were open did exactly
    that. A rename within a directory is atomic, so a reader sees the previous
    run's artifact or this one, never half of either.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # The full-resolution grid source never goes to JSON; it exists only to
    # build the sidecar from the whole mesh rather than from the strided
    # payload the JSON carries.
    grid_source = artifact.pop(GRID_SOURCE_KEY, None)

    # Imported here rather than at module scope: web.fields pulls in scipy,
    # and this module is imported by paths that only need the CAD helpers.
    from web.fields import write_sidecar_for_artifact

    handle_fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    # The sidecar is saved after the JSON so its mtime is not older.
    write_sidecar_for_artifact(path, artifact, grid_source=grid_source)


def _matching_indices(source_centers: np.ndarray, sampled_centers: np.ndarray) -> np.ndarray:
    if len(sampled_centers) == 0 or len(source_centers) == len(sampled_centers):
        return np.arange(len(source_centers), dtype=np.int32)
    if len(source_centers) == 0:
        return np.array([], dtype=np.int32)
    step = max(1, len(source_centers) // len(sampled_centers))
    indices = np.arange(0, len(source_centers), step, dtype=np.int32)
    return indices[: len(sampled_centers)]


def _bounds_payload(bounds: np.ndarray) -> Dict[str, list[float]]:
    return {
        "x": [float(bounds[0][0]), float(bounds[0][1])],
        "y": [float(bounds[1][0]), float(bounds[1][1])],
        "z": [float(bounds[2][0]), float(bounds[2][1])],
    }


def _normalize(values: Any) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    min_v = float(np.min(finite))
    max_v = float(np.max(finite))
    if max_v <= min_v:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - min_v) / (max_v - min_v), 0.0, 1.0).astype(np.float32)


def _display_field(engine: Any, method_name: str, attribute: str) -> Any:
    """The engine's display array for a result (``method_name()``), else the raw attribute.

    Shrink and Microporosity go through ``get_shrink_display_intensity`` and
    ``get_microporosity_display_intensity`` so the web shows the same arrays
    as the desktop (chill points at 0). An engine without the method, or
    without the attribute, falls back to the attribute (or None).
    """
    if getattr(engine, attribute, None) is None:
        return None
    method = getattr(engine, method_name, None)
    if callable(method):
        return method()
    return getattr(engine, attribute)


def _invert(values: Any) -> Optional[np.ndarray]:
    if values is None:
        return None
    return (1.0 - np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)).astype(np.float32)


def _range_payload(values: np.ndarray) -> Dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }
