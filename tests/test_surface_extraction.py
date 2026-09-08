"""Unit tests for the one contouring path that survives: dense-grid isosurfaces.

The sparse ``/surface`` JSON endpoint and its lattice-rebuilding payload
builder are gone; everything now contours the dense grid the sidecar already
holds. The HTTP surface of that path lives in ``tests/test_web_fields.py``
(``/api/projects/<id>/grid/<mode>/isosurface``); this file covers the geometry
kernel underneath it.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.surface_extraction import SurfaceArrays, contour_dense_grid


def _sphere_grid(size: int = 24, radius: float = 7.0) -> np.ndarray:
    """A smooth blob centred in the grid, 1.0 at the middle falling to 0.0."""
    axis = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    distance = np.sqrt(gx**2 + gy**2 + gz**2)
    return np.clip(1.0 - distance / radius, 0.0, 1.0).astype(np.float32)


class SurfaceArraysTests(unittest.TestCase):
    def test_empty_surface_reports_itself_as_empty(self) -> None:
        surface = SurfaceArrays.empty_surface()
        self.assertTrue(surface.empty)
        self.assertEqual(surface.vertex_count, 0)
        self.assertEqual(surface.triangle_count, 0)

    def test_counts_derive_from_the_arrays(self) -> None:
        surface = SurfaceArrays(
            positions=np.zeros((4, 3), dtype=np.float32),
            normals=np.zeros((4, 3), dtype=np.float32),
            indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32),
        )
        self.assertFalse(surface.empty)
        self.assertEqual(surface.vertex_count, 4)
        self.assertEqual(surface.triangle_count, 2)


class ContourDenseGridTests(unittest.TestCase):
    def test_contours_a_blob(self) -> None:
        surface = contour_dense_grid(_sphere_grid(), origin=(0.0, 0.0, 0.0), spacing=0.5, threshold=0.5)
        self.assertFalse(surface.empty)
        self.assertGreater(surface.triangle_count, 100)
        self.assertEqual(surface.positions.shape[1], 3)
        self.assertEqual(surface.normals.shape, surface.positions.shape)
        self.assertEqual(surface.indices.size % 3, 0)
        self.assertLess(int(surface.indices.max()), surface.vertex_count)

    def test_normals_are_unit_length(self) -> None:
        surface = contour_dense_grid(_sphere_grid(), origin=(0.0, 0.0, 0.0), spacing=0.5, threshold=0.5)
        lengths = np.linalg.norm(surface.normals, axis=1)
        self.assertTrue(np.allclose(lengths, 1.0, atol=1e-3))

    def test_origin_and_spacing_place_the_surface_in_world_space(self) -> None:
        volume = _sphere_grid()
        origin = (2.0, -3.0, 10.0)
        spacing = 0.25
        surface = contour_dense_grid(volume, origin=origin, spacing=spacing, threshold=0.5)

        # The blob is centred in the grid, so its contour must straddle the
        # world position of the grid centre on every axis.
        centre = np.asarray(origin, dtype=np.float64) + (np.asarray(volume.shape) - 1) / 2.0 * spacing
        self.assertTrue(np.all(surface.positions.min(axis=0) < centre))
        self.assertTrue(np.all(surface.positions.max(axis=0) > centre))

    def test_a_higher_threshold_encloses_less(self) -> None:
        volume = _sphere_grid()
        low = contour_dense_grid(volume, origin=(0.0, 0.0, 0.0), spacing=1.0, threshold=0.2)
        high = contour_dense_grid(volume, origin=(0.0, 0.0, 0.0), spacing=1.0, threshold=0.8)
        low_extent = float(np.ptp(low.positions, axis=0).max())
        high_extent = float(np.ptp(high.positions, axis=0).max())
        self.assertGreater(low_extent, high_extent)

    def test_threshold_above_the_field_is_empty(self) -> None:
        surface = contour_dense_grid(_sphere_grid(), origin=(0.0, 0.0, 0.0), spacing=1.0, threshold=5.0)
        self.assertTrue(surface.empty)

    def test_degenerate_input_is_empty_rather_than_an_error(self) -> None:
        self.assertTrue(
            contour_dense_grid(
                np.empty((0, 0, 0), dtype=np.float32), origin=(0.0, 0.0, 0.0), spacing=1.0, threshold=0.5
            ).empty
        )
        self.assertTrue(
            contour_dense_grid(
                np.zeros((4, 4), dtype=np.float32), origin=(0.0, 0.0, 0.0), spacing=1.0, threshold=0.5
            ).empty
        )

    def test_a_saturated_grid_still_closes(self) -> None:
        """Every cell above the threshold must still yield a closed surface.

        The padding ring is what makes this work: the grid is lowered to a
        background strictly below the threshold so there is something to
        contour against. Smoothing is off here because a Gaussian over a
        one-cell ring that sits a thousandth below the threshold pulls it back
        above, which would leave nothing to cross.
        """
        volume = np.ones((10, 10, 10), dtype=np.float32)
        surface = contour_dense_grid(
            volume, origin=(0.0, 0.0, 0.0), spacing=1.0, threshold=0.5, smooth_sigma=0.0
        )
        self.assertFalse(surface.empty)
        self.assertGreater(surface.triangle_count, 0)
        # The surface wraps the whole grid, so it spans every cell.
        extent = np.ptp(surface.positions, axis=0)
        self.assertTrue(np.all(extent >= 9.0))

    def test_smoothing_can_be_disabled(self) -> None:
        volume = _sphere_grid()
        rough = contour_dense_grid(
            volume, origin=(0.0, 0.0, 0.0), spacing=1.0, threshold=0.5, smooth_sigma=0.0
        )
        smooth = contour_dense_grid(volume, origin=(0.0, 0.0, 0.0), spacing=1.0, threshold=0.5)
        self.assertFalse(rough.empty)
        self.assertFalse(smooth.empty)


if __name__ == "__main__":
    unittest.main()
