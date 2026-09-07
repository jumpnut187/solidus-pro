"""Smooth isosurface extraction for web result visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


#: Public mode id -> the visualization key(s) it reads, most specific first.
#:
#: Isosurface extraction used to serve four modes, all of them 0..1 risk
#: fields. Every scalar the engine produces is now contourable, which means
#: two things this table encodes: modes whose artifact key does not match
#: their public name (``solidification_time`` is stored as
#: ``solidification_times``), and modes with more than one historical spelling.
#: It is deliberately the same vocabulary as ``web.fields.SCALAR_MODE_SOURCES``
#: -- the isosurface and the painted grid must never disagree about what
#: "niyama" means.
SURFACE_MODE_SOURCES: Dict[str, tuple[str, ...]] = {
    "shrinkage": ("shrinkage",),
    "porosity": ("porosity",),
    "feed_modulus": ("feed_modulus",),
    "hotspot": ("hotspot",),
    "solidification_time": ("solidification_times",),
    "niyama_raw": ("niyama_raw",),
    "niyama": ("niyama_risk", "niyama"),
    "combined_feed_risk": ("combined_feed_risk",),
    "hot_tear": ("hot_tear",),
    "sand_burn_in": ("sand_burn_in",),
    "gas_generation": ("gas_generation",),
    "gas_entrapment": ("gas_entrapment",),
    "gas_porosity": ("gas_porosity_risk", "gas_porosity"),
    "feeding_quality": ("feeding_quality",),
    "feeding_blocked": ("feeding_blocked",),
    "feeding_path_lengths": ("feeding_path_lengths",),
    "feeding_open_fraction": ("feeding_open_fraction",),
    "thermal_stress": ("thermal_stress",),
    "distortion": ("distortion",),
    "crack_susceptibility": ("crack_susceptibility",),
    "sdas": ("sdas",),
    "predicted_uts": ("predicted_uts",),
    "fill_time": ("fill_time",),
}

#: Modes whose values are a fraction by construction, so a threshold outside
#: 0..1 is a caller error rather than a field-unit value.
_UNIT_INTERVAL_MODES = frozenset({"fraction_liquid"})

VALID_SURFACE_MODES = frozenset(SURFACE_MODE_SOURCES) | _UNIT_INTERVAL_MODES


@dataclass(frozen=True)
class SurfaceRequest:
    """Normalized request parameters for a result surface."""

    mode: str
    threshold: float
    time_percent: float = 0.0


def normalize_surface_request(mode: str, threshold: float, time_percent: float = 0.0) -> SurfaceRequest:
    """
    Validate user-facing surface request parameters.

    ``threshold`` is in **field units**: seconds for ``solidification_time``,
    inches for ``feed_modulus``, MPa for ``thermal_stress``, and a 0..1
    fraction for the risk fields. It used to be clamped to 0..1 unconditionally,
    which was invisible while only risk fields were contourable and would
    silently collapse every feed-modulus request onto the 1-inch isosurface
    now that they are not. Only genuinely fractional modes are still clamped.
    """
    normalized_mode = mode.strip().lower()
    if normalized_mode not in VALID_SURFACE_MODES:
        raise ValueError(f"Unsupported surface mode: {mode}")

    value = float(threshold)
    if not np.isfinite(value):
        value = 0.0
    if normalized_mode in _UNIT_INTERVAL_MODES:
        value = float(np.clip(value, 0.0, 1.0))

    return SurfaceRequest(
        mode=normalized_mode,
        threshold=value,
        time_percent=float(np.clip(time_percent, 0.0, 100.0)),
    )


def build_surface_payload(
    visualization_data: Dict[str, Any],
    request: SurfaceRequest,
    *,
    smooth_sigma: float = 0.75,
) -> Dict[str, Any]:
    """
    Convert voxel-center scalar data into a Babylon-ready isosurface payload.

    The input payload is the same downsampled visualization payload used by the
    web client. The scalar field is reconstructed on the implicit voxel-center
    lattice, padded with zero values, smoothed, then contoured.
    """
    centers = _as_float_array(visualization_data.get("centers"), expected_columns=3)
    if centers.size == 0:
        return _empty_payload(request, reason="no centers")

    values = _values_for_request(visualization_data, request)
    if values is None or values.size == 0:
        return _empty_payload(request, reason="no values")
    if len(values) != len(centers):
        return _empty_payload(request, reason="value count mismatch")

    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return _empty_payload(request, reason="no finite values")

    centers = centers[finite_mask]
    values = values[finite_mask].astype(np.float32)
    if request.mode in _UNIT_INTERVAL_MODES:
        values = np.clip(values, 0.0, 1.0)
    if request.mode == "fill_time":
        # -1 marks metal the fill never reached and is not a sample of the
        # arrival-time field; contouring it would drag the isosurface through
        # a value the physics never produced.
        arrived = values >= 0.0
        if not np.any(arrived):
            return _empty_payload(request, reason="no filled voxels")
        centers = centers[arrived]
        values = values[arrived]
    if request.mode == "fraction_liquid":
        interior_mask = _interior_voxel_mask(
            centers,
            visualization_data.get("surface_voxels"),
            finite_mask,
            threshold=request.threshold,
            values=values,
        )
        if not np.any(interior_mask):
            return _empty_payload(request, reason="no interior liquid voxels")
        centers = centers[interior_mask]
        values = values[interior_mask]

    active_count = int(np.sum(values >= request.threshold))
    if active_count == 0:
        return _empty_payload(request, active_count=0, reason="below threshold")

    volume, origin, spacing = _build_scalar_volume(centers, values)
    effective_smooth_sigma = 0.35 if request.mode == "fraction_liquid" else smooth_sigma
    if effective_smooth_sigma > 0.0 and min(volume.shape) >= 4:
        smoothed = ndimage.gaussian_filter(volume, sigma=effective_smooth_sigma)
        # Smoothing is cosmetic: it rounds the staircase off the lattice. It
        # must never be the reason a real feature disappears. A defect pocket a
        # couple of voxels across -- exactly the size that matters most -- has
        # its peak pulled below the threshold by the blur, so when that
        # happens the raw volume is contoured instead of reporting nothing.
        if float(np.max(smoothed)) >= request.threshold:
            volume = smoothed

    if float(np.max(volume)) < request.threshold:
        return _empty_payload(request, active_count=active_count, reason="below threshold after gridding")

    mesh = _extract_with_vtk(volume, origin, spacing, request.threshold)
    mesh.update(
        {
            "mode": request.mode,
            "threshold": request.threshold,
            "time_percent": request.time_percent,
            "active_count": active_count,
            "source_count": int(len(values)),
        }
    )
    return mesh


def _values_for_request(visualization_data: Dict[str, Any], request: SurfaceRequest) -> Optional[np.ndarray]:
    if request.mode == "fraction_liquid":
        history = visualization_data.get("fraction_liquid_history")
        times = visualization_data.get("fraction_liquid_times")
        if isinstance(history, list) and history:
            return _fraction_liquid_snapshot(history, times, request.time_percent)
        values = visualization_data.get("fraction_liquid")
        if values is not None:
            return np.asarray(values, dtype=np.float32)
        solidification_times = visualization_data.get("solidification_times")
        if solidification_times is None:
            return None
        times_arr = np.asarray(solidification_times, dtype=np.float32)
        max_time = float(np.max(times_arr)) if times_arr.size else 0.0
        if max_time <= 0.0:
            return None
        time_factor = request.time_percent / 100.0
        return np.clip((1.0 - (times_arr / max_time)) - time_factor, 0.0, 1.0).astype(np.float32)

    for key in SURFACE_MODE_SOURCES.get(request.mode, (request.mode,)):
        values = visualization_data.get(key)
        if values is not None:
            return np.asarray(values, dtype=np.float32)
    return None


def _interior_voxel_mask(
    centers: np.ndarray,
    surface_voxels: Any,
    finite_mask: np.ndarray,
    *,
    threshold: float,
    values: np.ndarray,
) -> np.ndarray:
    if surface_voxels is not None:
        surface_arr = np.asarray(surface_voxels, dtype=bool).reshape(-1)
        if len(surface_arr) == len(finite_mask):
            surface_arr = surface_arr[finite_mask]
        if len(surface_arr) == len(centers):
            buffered = _surface_buffered_interior_mask(centers, surface_arr)
            if np.any(buffered & (values >= threshold)):
                return buffered
            return ~surface_arr

    buffered = _infer_lattice_interior_mask(centers, buffer_layers=2)
    if np.any(buffered & (values >= threshold)):
        return buffered
    return _infer_lattice_interior_mask(centers, buffer_layers=1)


def _surface_buffered_interior_mask(centers: np.ndarray, surface_mask: np.ndarray) -> np.ndarray:
    if centers.size == 0:
        return np.zeros(0, dtype=bool)
    if len(surface_mask) != len(centers):
        return _infer_lattice_interior_mask(centers, buffer_layers=2)

    non_surface = ~surface_mask
    if not np.any(non_surface):
        return np.zeros(len(centers), dtype=bool)

    surface_centers = centers[surface_mask]
    if surface_centers.size == 0:
        return non_surface

    spacing = _representative_spacing(centers)
    distances, _ = cKDTree(surface_centers).query(centers, k=1)
    return non_surface & (distances > spacing * 1.25)


def _infer_lattice_interior_mask(centers: np.ndarray, *, buffer_layers: int) -> np.ndarray:
    if centers.size == 0:
        return np.zeros(0, dtype=bool)

    x_values = _unique_axis_values(centers[:, 0])
    y_values = _unique_axis_values(centers[:, 1])
    z_values = _unique_axis_values(centers[:, 2])
    ix = _nearest_axis_indices(centers[:, 0], x_values)
    iy = _nearest_axis_indices(centers[:, 1], y_values)
    iz = _nearest_axis_indices(centers[:, 2], z_values)

    lower = max(0, int(buffer_layers) - 1)
    upper_offset = max(1, int(buffer_layers))

    return (
        (ix > lower)
        & (ix < len(x_values) - upper_offset)
        & (iy > lower)
        & (iy < len(y_values) - upper_offset)
        & (iz > lower)
        & (iz < len(z_values) - upper_offset)
    )


def _representative_spacing(centers: np.ndarray) -> float:
    if centers.size == 0:
        return 1.0
    spacings = [
        _axis_spacing(_unique_axis_values(centers[:, 0])),
        _axis_spacing(_unique_axis_values(centers[:, 1])),
        _axis_spacing(_unique_axis_values(centers[:, 2])),
    ]
    positive = [spacing for spacing in spacings if spacing > 1e-6]
    return float(max(positive)) if positive else 1.0


def _fraction_liquid_snapshot(history: list[Any], times: Any, time_percent: float) -> np.ndarray:
    history_arrays = [np.asarray(frame, dtype=np.float32) for frame in history]
    if not history_arrays:
        return np.array([], dtype=np.float32)

    times_arr = np.asarray(times if times is not None else [], dtype=np.float32)
    if times_arr.size != len(history_arrays) or times_arr.size == 0:
        index = min(int((time_percent / 100.0) * len(history_arrays)), len(history_arrays) - 1)
        return history_arrays[index]

    min_time = float(times_arr[0])
    max_time = float(times_arr[-1])
    if max_time <= min_time:
        return history_arrays[-1]

    target_time = min_time + (time_percent / 100.0) * (max_time - min_time)
    if target_time <= min_time:
        return history_arrays[0]
    if target_time >= max_time:
        return history_arrays[-1]

    after_index = int(np.searchsorted(times_arr, target_time, side="left"))
    before_index = max(0, after_index - 1)
    if after_index >= len(history_arrays):
        return history_arrays[-1]

    before_time = float(times_arr[before_index])
    after_time = float(times_arr[after_index])
    if after_time <= before_time:
        return history_arrays[after_index]

    alpha = (target_time - before_time) / (after_time - before_time)
    return (history_arrays[before_index] * (1.0 - alpha) + history_arrays[after_index] * alpha).astype(np.float32)


def _build_scalar_volume(centers: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = _unique_axis_values(centers[:, 0])
    y_values = _unique_axis_values(centers[:, 1])
    z_values = _unique_axis_values(centers[:, 2])
    spacing = np.asarray(
        [
            _axis_spacing(x_values),
            _axis_spacing(y_values),
            _axis_spacing(z_values),
        ],
        dtype=np.float32,
    )
    origin = np.asarray(
        [
            float(x_values[0] - spacing[0]),
            float(y_values[0] - spacing[1]),
            float(z_values[0] - spacing[2]),
        ],
        dtype=np.float32,
    )

    # The padding ring has to sit at or below every sample so the contour
    # closes around the casting instead of clipping against the box. Zero was
    # right while every field was a 0..1 risk; a field in physical units can
    # legitimately go negative, and seeding the whole volume at zero would then
    # bury part of it under the padding.
    background = min(0.0, float(np.min(values))) if values.size else 0.0
    volume = np.full(
        (len(x_values) + 2, len(y_values) + 2, len(z_values) + 2),
        background,
        dtype=np.float32,
    )
    ix = _nearest_axis_indices(centers[:, 0], x_values) + 1
    iy = _nearest_axis_indices(centers[:, 1], y_values) + 1
    iz = _nearest_axis_indices(centers[:, 2], z_values) + 1
    np.maximum.at(volume, (ix, iy, iz), values)
    return volume, origin, spacing


def _extract_with_vtk(volume: np.ndarray, origin: np.ndarray, spacing: np.ndarray, threshold: float) -> Dict[str, Any]:
    try:
        import vtk
        from vtk.util import numpy_support
    except Exception as exc:  # pragma: no cover - exercised only when VTK is unavailable
        raise RuntimeError("VTK is required for smooth web result surfaces") from exc

    image = vtk.vtkImageData()
    image.SetDimensions(int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2]))
    image.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
    image.SetSpacing(float(spacing[0]), float(spacing[1]), float(spacing[2]))

    vtk_array = numpy_support.numpy_to_vtk(
        num_array=np.ascontiguousarray(volume.ravel(order="F")),
        deep=True,
        array_type=vtk.VTK_FLOAT,
    )
    vtk_array.SetName("result")
    image.GetPointData().SetScalars(vtk_array)

    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputData(image)
    contour.SetValue(0, float(threshold))
    contour.ComputeNormalsOn()
    contour.Update()

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(contour.GetOutputPort())
    smoother.SetNumberOfIterations(12)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetPassBand(0.08)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(smoother.GetOutputPort())
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()

    polydata = normals.GetOutput()
    return _polydata_to_payload(polydata)


def _polydata_to_payload(polydata: Any) -> Dict[str, Any]:
    from vtk.util import numpy_support

    points = polydata.GetPoints()
    if points is None or points.GetNumberOfPoints() == 0:
        return {
            "positions": [],
            "indices": [],
            "normals": [],
            "vertex_count": 0,
            "triangle_count": 0,
            "empty": True,
        }

    positions = numpy_support.vtk_to_numpy(points.GetData()).astype(np.float32)
    normal_data = polydata.GetPointData().GetNormals()
    normals = (
        numpy_support.vtk_to_numpy(normal_data).astype(np.float32)
        if normal_data is not None
        else np.zeros_like(positions, dtype=np.float32)
    )

    polys = polydata.GetPolys()
    triangles: list[int] = []
    if polys is not None and polys.GetNumberOfCells() > 0:
        offsets_data = polys.GetOffsetsArray()
        connectivity_data = polys.GetConnectivityArray()
        offsets = (
            numpy_support.vtk_to_numpy(offsets_data).astype(np.int64)
            if offsets_data is not None
            else np.array([], dtype=np.int64)
        )
        connectivity = (
            numpy_support.vtk_to_numpy(connectivity_data).astype(np.int64)
            if connectivity_data is not None
            else np.array([], dtype=np.int64)
        )
    else:
        offsets = np.array([], dtype=np.int64)
        connectivity = np.array([], dtype=np.int64)

    for cell_index in range(max(0, len(offsets) - 1)):
        start = int(offsets[cell_index])
        end = int(offsets[cell_index + 1])
        cell = connectivity[start:end]
        count = len(cell)
        if count == 3:
            triangles.extend(int(v) for v in cell)
        elif count > 3:
            first = int(cell[0])
            for idx in range(1, count - 1):
                triangles.extend([first, int(cell[idx]), int(cell[idx + 1])])

    return {
        "positions": positions.reshape(-1).round(6).tolist(),
        "indices": triangles,
        "normals": normals.reshape(-1).round(6).tolist(),
        "vertex_count": int(len(positions)),
        "triangle_count": int(len(triangles) // 3),
        "empty": len(triangles) == 0,
    }


def _as_float_array(value: Any, *, expected_columns: int) -> np.ndarray:
    if value is None:
        return np.empty((0, expected_columns), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, expected_columns), dtype=np.float32)
    return arr.reshape((-1, expected_columns))


def _unique_axis_values(values: Iterable[float]) -> np.ndarray:
    unique = np.unique(np.round(np.asarray(values, dtype=np.float32), 6))
    if unique.size == 0:
        return np.asarray([0.0], dtype=np.float32)
    return unique.astype(np.float32)


def _axis_spacing(values: np.ndarray) -> float:
    if values.size < 2:
        return 1.0
    diffs = np.diff(np.sort(values))
    positive = diffs[diffs > 1e-6]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive))


def _nearest_axis_indices(values: np.ndarray, axis_values: np.ndarray) -> np.ndarray:
    sorted_axis = np.asarray(axis_values, dtype=np.float32)
    indices = np.searchsorted(sorted_axis, values)
    indices = np.clip(indices, 0, len(sorted_axis) - 1)
    previous = np.clip(indices - 1, 0, len(sorted_axis) - 1)
    use_previous = np.abs(values - sorted_axis[previous]) < np.abs(values - sorted_axis[indices])
    return np.where(use_previous, previous, indices).astype(np.int32)


def _empty_payload(
    request: SurfaceRequest,
    *,
    active_count: int = 0,
    reason: str,
) -> Dict[str, Any]:
    return {
        "positions": [],
        "indices": [],
        "normals": [],
        "vertex_count": 0,
        "triangle_count": 0,
        "active_count": active_count,
        "source_count": 0,
        "mode": request.mode,
        "threshold": request.threshold,
        "time_percent": request.time_percent,
        "empty": True,
        "reason": reason,
    }
