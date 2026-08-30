"""Tests for ``core.pour_location``.

Covers the pure data model (validation, normalization, JSON round-trip,
unit conversion) and ``resolve_inlet_voxels``, which maps a pour point to
nearby voxel indices on a small synthetic lattice built with numpy -- no
mesh object involved, per the module's mesh-agnostic contract.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from core.pour_location import PourLocation, resolve_inlet_voxels


class PourLocationValidationTests(unittest.TestCase):
    def test_round_trip_to_dict_from_dict(self) -> None:
        location = PourLocation(
            point_in=(1.0, 2.0, 3.0),
            normal=(0.0, 0.0, 1.0),
            part_id="abc123",
        )
        restored = PourLocation.from_dict(location.to_dict())
        self.assertEqual(restored, location)

    def test_round_trip_without_optional_fields(self) -> None:
        location = PourLocation(point_in=(1.0, 2.0, 3.0))
        restored = PourLocation.from_dict(location.to_dict())
        self.assertEqual(restored, location)
        self.assertIsNone(restored.normal)
        self.assertIsNone(restored.part_id)

    def test_point_m_converts_inches_to_meters_exactly(self) -> None:
        location = PourLocation(point_in=(1.0, 2.0, -3.0))
        x_m, y_m, z_m = location.point_m
        self.assertAlmostEqual(x_m, 0.0254, places=12)
        self.assertAlmostEqual(y_m, 0.0508, places=12)
        self.assertAlmostEqual(z_m, -0.0762, places=12)

    def test_normal_is_normalized(self) -> None:
        location = PourLocation(point_in=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 5.0))
        self.assertAlmostEqual(location.normal[2], 1.0, places=12)
        length = math.sqrt(sum(c * c for c in location.normal))
        self.assertAlmostEqual(length, 1.0, places=12)

    def test_zero_normal_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PourLocation(point_in=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 0.0))

    def test_nan_coordinate_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PourLocation(point_in=(0.0, float("nan"), 0.0))

    def test_inf_coordinate_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PourLocation(point_in=(0.0, 0.0, float("inf")))

    def test_nan_normal_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PourLocation(point_in=(0.0, 0.0, 0.0), normal=(float("nan"), 0.0, 1.0))

    def test_unknown_schema_version_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PourLocation(point_in=(0.0, 0.0, 0.0), schema_version=999)

    def test_wrong_length_point_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PourLocation(point_in=(0.0, 0.0))  # type: ignore[arg-type]


class ResolveInletVoxelsTests(unittest.TestCase):
    def _lattice(self, voxel_size: float, n_per_axis: int = 5) -> np.ndarray:
        """Build an (n^3, 3) grid of voxel centers spaced by voxel_size."""
        coords = np.arange(n_per_axis) * voxel_size
        xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
        return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    def test_finds_expected_voxels_near_point(self) -> None:
        voxel_size = 1.0
        centers = self._lattice(voxel_size)
        # Pick a point exactly on a lattice node well inside the grid.
        target = np.array([2.0, 2.0, 2.0])
        location = PourLocation(point_in=tuple(target))

        indices = resolve_inlet_voxels(
            location, centers, voxel_size, radius_voxels=1.5
        )

        self.assertGreater(len(indices), 0)
        found_points = centers[indices]
        distances = np.linalg.norm(found_points - target, axis=1)
        self.assertTrue(np.all(distances <= 1.5 * voxel_size))
        # The target point itself must be included.
        self.assertTrue(np.any(np.all(np.isclose(found_points, target), axis=1)))
        # Nothing outside the radius should sneak in.
        all_distances = np.linalg.norm(centers - target, axis=1)
        expected_count = int(np.count_nonzero(all_distances <= 1.5 * voxel_size))
        self.assertEqual(len(indices), expected_count)

    def test_returns_empty_when_point_is_far_away(self) -> None:
        voxel_size = 1.0
        centers = self._lattice(voxel_size)
        far_point = np.array([1000.0, 1000.0, 1000.0])
        location = PourLocation(point_in=tuple(far_point))

        indices = resolve_inlet_voxels(
            location, centers, voxel_size, radius_voxels=1.5
        )

        self.assertEqual(len(indices), 0)
        self.assertEqual(indices.dtype, np.intp)

    def test_respects_allowed_codes(self) -> None:
        voxel_size = 1.0
        centers = self._lattice(voxel_size)
        target = np.array([2.0, 2.0, 2.0])
        location = PourLocation(point_in=tuple(target))

        # Assign part-type code 7 (RUNNER-like) only to the exact target
        # voxel; everything else is code 0 (CAST-like).
        part_types = np.zeros(len(centers), dtype=np.int32)
        target_idx = np.flatnonzero(np.all(np.isclose(centers, target), axis=1))
        part_types[target_idx] = 7

        indices_unrestricted = resolve_inlet_voxels(
            location, centers, voxel_size, radius_voxels=1.5
        )
        indices_restricted = resolve_inlet_voxels(
            location,
            centers,
            voxel_size,
            part_types=part_types,
            allowed_codes={7},
            radius_voxels=1.5,
        )

        self.assertGreater(len(indices_unrestricted), len(indices_restricted))
        self.assertEqual(set(indices_restricted.tolist()), set(target_idx.tolist()))

    def test_allowed_codes_without_part_types_raises(self) -> None:
        centers = self._lattice(1.0)
        location = PourLocation(point_in=(2.0, 2.0, 2.0))
        with self.assertRaises(ValueError):
            resolve_inlet_voxels(
                location, centers, 1.0, allowed_codes={0}
            )

    def test_radius_scales_with_voxel_size(self) -> None:
        target = np.array([0.0, 0.0, 0.0])
        location = PourLocation(point_in=tuple(target))

        small_voxel = 1.0
        large_voxel = 2.0
        small_centers = self._lattice(small_voxel, n_per_axis=7) - 3.0 * small_voxel
        large_centers = self._lattice(large_voxel, n_per_axis=7) - 3.0 * large_voxel

        small_indices = resolve_inlet_voxels(
            location, small_centers, small_voxel, radius_voxels=1.5
        )
        large_indices = resolve_inlet_voxels(
            location, large_centers, large_voxel, radius_voxels=1.5
        )

        # Same lattice shape both times, but the physical search radius is
        # 1.5x bigger when the voxel size doubles, so more of the (equally
        # spaced-out) large lattice's neighbors fall within range.
        small_radius = 1.5 * small_voxel
        large_radius = 1.5 * large_voxel
        self.assertLess(small_radius, large_radius)

        small_distances = np.linalg.norm(small_centers - target, axis=1)
        large_distances = np.linalg.norm(large_centers - target, axis=1)
        expected_small = int(np.count_nonzero(small_distances <= small_radius))
        expected_large = int(np.count_nonzero(large_distances <= large_radius))

        self.assertEqual(len(small_indices), expected_small)
        self.assertEqual(len(large_indices), expected_large)

    def test_rejects_bad_voxel_centers_shape(self) -> None:
        location = PourLocation(point_in=(0.0, 0.0, 0.0))
        bad_centers = np.zeros((5, 2))
        with self.assertRaises(ValueError):
            resolve_inlet_voxels(location, bad_centers, 1.0)

    def test_rejects_non_positive_voxel_size(self) -> None:
        location = PourLocation(point_in=(0.0, 0.0, 0.0))
        centers = self._lattice(1.0)
        with self.assertRaises(ValueError):
            resolve_inlet_voxels(location, centers, 0.0)


if __name__ == "__main__":
    unittest.main()
