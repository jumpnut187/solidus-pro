"""Smooth isosurface extraction for web result visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class SurfaceArrays:
    """
    An isosurface as raw arrays, before any serialization decision.

    The binary body the viewport fetches builds straight from these, so it
    never pays for a round trip through Python lists -- a 100,000 triangle
    envelope is 7 MB of JSON and about 2 MB of Float32.
    """

    positions: np.ndarray  # (V, 3) float32, world inches
    normals: np.ndarray  # (V, 3) float32, unit
    indices: np.ndarray  # (T * 3,) uint32

    @staticmethod
    def empty_surface() -> "SurfaceArrays":
        return SurfaceArrays(
            positions=np.empty((0, 3), dtype=np.float32),
            normals=np.empty((0, 3), dtype=np.float32),
            indices=np.empty((0,), dtype=np.uint32),
        )

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.indices.size // 3)

    @property
    def empty(self) -> bool:
        return self.indices.size == 0


def contour_dense_grid(
    volume: np.ndarray,
    origin: Iterable[float],
    spacing: float,
    threshold: float,
    *,
    smooth_sigma: float = 0.75,
) -> SurfaceArrays:
    """
    Contour an already-dense scalar grid.

    This is the only contouring path left. It is handed the dense grid the
    sidecar already holds rather than rebuilding a lattice out of sparse voxel
    centers per request, so there is no JSON parse, and -- crucially -- the
    caller has already masked the grid to the metal, so the contour cannot
    leak into the hole-filled values outside the casting.

    ``volume`` is indexed ``[ix, iy, iz]``; ``origin`` is the world position of
    the center of cell ``(0, 0, 0)``. A one-cell ring of background is added on
    every side so the surface closes where the casting reaches the edge of its
    own bounding box instead of being sliced open by it.
    """
    if volume.ndim != 3 or volume.size == 0:
        return SurfaceArrays.empty_surface()

    background = float(np.min(volume))
    if background >= threshold:
        # Every cell is above the threshold, so a closed surface needs a
        # background strictly below it to contour against.
        background = threshold - max(abs(threshold), 1.0) * 1e-3

    padded = np.full(tuple(size + 2 for size in volume.shape), background, dtype=np.float32)
    padded[1:-1, 1:-1, 1:-1] = volume.astype(np.float32, copy=False)

    if smooth_sigma > 0.0 and min(padded.shape) >= 4:
        smoothed = ndimage.gaussian_filter(padded, sigma=smooth_sigma)
        if float(np.max(smoothed)) >= threshold:
            padded = smoothed

    if float(np.max(padded)) < threshold:
        return SurfaceArrays.empty_surface()

    origin_arr = np.asarray(list(origin), dtype=np.float32) - float(spacing)
    spacing_arr = np.full(3, float(spacing), dtype=np.float32)
    return _extract_with_vtk(padded, origin_arr, spacing_arr, float(threshold))


def _extract_with_vtk(volume: np.ndarray, origin: np.ndarray, spacing: np.ndarray, threshold: float) -> SurfaceArrays:
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
    return _polydata_to_arrays(polydata)


def _polydata_to_arrays(polydata: Any) -> SurfaceArrays:
    """
    Pull positions, normals and triangles out of a vtkPolyData as numpy arrays.

    Vectorized rather than looped. The previous version walked the cell array
    in Python and appended to a list, which is fine for the few thousand
    triangles a risk-field envelope produces and quadratically unpleasant for
    the hundred thousand a fine solidification-time isosurface does.
    """
    from vtk.util import numpy_support

    points = polydata.GetPoints()
    if points is None or points.GetNumberOfPoints() == 0:
        return SurfaceArrays.empty_surface()

    positions = numpy_support.vtk_to_numpy(points.GetData()).astype(np.float32, copy=False)
    normal_data = polydata.GetPointData().GetNormals()
    normals = (
        numpy_support.vtk_to_numpy(normal_data).astype(np.float32, copy=False)
        if normal_data is not None
        else np.zeros_like(positions, dtype=np.float32)
    )

    polys = polydata.GetPolys()
    if polys is None or polys.GetNumberOfCells() == 0:
        return SurfaceArrays.empty_surface()

    offsets_data = polys.GetOffsetsArray()
    connectivity_data = polys.GetConnectivityArray()
    if offsets_data is None or connectivity_data is None:
        return SurfaceArrays.empty_surface()
    offsets = numpy_support.vtk_to_numpy(offsets_data).astype(np.int64, copy=False)
    connectivity = numpy_support.vtk_to_numpy(connectivity_data).astype(np.int64, copy=False)
    if offsets.size < 2:
        return SurfaceArrays.empty_surface()

    sizes = np.diff(offsets)
    if np.all(sizes == 3):
        # The fast path, and the only one a contour filter ever takes.
        triangles = connectivity.reshape(-1, 3)
    else:
        triangles = _fan_triangulate(offsets, connectivity, sizes)

    if triangles.size == 0:
        return SurfaceArrays.empty_surface()

    return SurfaceArrays(
        positions=np.ascontiguousarray(positions, dtype=np.float32).reshape(-1, 3),
        normals=np.ascontiguousarray(normals, dtype=np.float32).reshape(-1, 3),
        indices=np.ascontiguousarray(triangles.reshape(-1), dtype=np.uint32),
    )


def _fan_triangulate(offsets: np.ndarray, connectivity: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Fan-triangulate any polygon with more than three corners."""
    fans: list[np.ndarray] = []
    triples = sizes == 3
    if np.any(triples):
        starts = offsets[:-1][triples]
        fans.append(connectivity[(starts[:, None] + np.arange(3)[None, :]).reshape(-1)].reshape(-1, 3))
    for index in np.nonzero(sizes > 3)[0]:
        cell = connectivity[int(offsets[index]) : int(offsets[index + 1])]
        first = cell[0]
        fans.append(
            np.stack(
                [np.full(len(cell) - 2, first, dtype=np.int64), cell[1:-1], cell[2:]],
                axis=1,
            )
        )
    return np.concatenate(fans, axis=0) if fans else np.empty((0, 3), dtype=np.int64)
