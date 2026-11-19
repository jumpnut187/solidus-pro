from __future__ import annotations

import unittest

import numpy as np

from core.defect_regions import (
    DefectPocket,
    compute_merged_shrink_display_intensity,
    extract_pocket_at,
    format_pocket_volume_label,
)


def _build_line_neighbors(n: int) -> np.ndarray:
    """Return a (N, 26) int32 neighbor table for a 1D line of N voxels.

    Each voxel sees at most two face neighbors (+X / -X) in the first two
    slots. The remaining slots are filled with -1.
    """
    neighbors = np.full((n, 26), -1, dtype=np.int32)
    for i in range(n):
        if i > 0:
            neighbors[i, 1] = i - 1  # -X
        if i < n - 1:
            neighbors[i, 0] = i + 1  # +X
    return neighbors


def _build_3d_grid_neighbors(shape: tuple[int, int, int]) -> np.ndarray:
    """Return a (N, 26) int32 neighbor table for a regular 3D grid of voxels.

    Slot 0 = +X, slot 1 = -X, slot 2 = +Y, slot 3 = -Y, slot 4 = +Z, slot 5 = -Z.
    Corner-touching voxels share slot 18 (lowest corner) and higher for other
    corner combinations. This matches ``_canonicalize_neighbor_order`` in
    ``core/voxel_mesh.py``.
    """
    nx, ny, nz = shape
    n = nx * ny * nz
    neighbors = np.full((n, 26), -1, dtype=np.int32)

    def index(x: int, y: int, z: int) -> int:
        return (z * ny + y) * nx + x

    corner_slot = [
        # (sx, sy, sz) -> slot
        ((1, 1, 1), 18), ((1, 1, -1), 19),
        ((1, -1, 1), 20), ((1, -1, -1), 21),
        ((-1, 1, 1), 22), ((-1, 1, -1), 23),
        ((-1, -1, 1), 24), ((-1, -1, -1), 25),
    ]

    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                i = index(x, y, z)
                # Face neighbors
                if x + 1 < nx:
                    neighbors[i, 0] = index(x + 1, y, z)
                if x - 1 >= 0:
                    neighbors[i, 1] = index(x - 1, y, z)
                if y + 1 < ny:
                    neighbors[i, 2] = index(x, y + 1, z)
                if y - 1 >= 0:
                    neighbors[i, 3] = index(x, y - 1, z)
                if z + 1 < nz:
                    neighbors[i, 4] = index(x, y, z + 1)
                if z - 1 >= 0:
                    neighbors[i, 5] = index(x, y, z - 1)
                # Corner neighbors
                for sx, sy, sz in ((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                                   (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)):
                    nx2, ny2, nz2 = x + sx, y + sy, z + sz
                    if not (0 <= nx2 < nx and 0 <= ny2 < ny and 0 <= nz2 < nz):
                        continue
                    slot = next(slot for (sxx, syy, szz), slot in corner_slot
                                if sxx == sx and syy == sy and szz == sz)
                    neighbors[i, slot] = index(nx2, ny2, nz2)

    return neighbors


class MergedDisplayIntensityTests(unittest.TestCase):
    """Verify the desktop merged shrink/porosity display mask."""

    def test_max_of_shrink_and_porosity(self) -> None:
        shrink = np.array([0.9, 0.2, 0.3, 0.7], dtype=np.float32)
        poro = np.array([0.4, 0.8, 0.1, 0.6], dtype=np.float32)
        display = compute_merged_shrink_display_intensity(
            shrinkage_probability=shrink,
            porosity_probability=poro,
            voxel_part_types=None,
        )
        np.testing.assert_allclose(display, [0.9, 0.8, 0.3, 0.7], atol=1e-6)

    def test_chill_voxels_zeroed(self) -> None:
        # CHILL = 2
        part_types = np.array([0, 1, 3, 2], dtype=np.int32)
        shrink = np.array([0.9, 0.2, 0.3, 0.7], dtype=np.float32)
        poro = np.array([0.4, 0.8, 0.1, 0.6], dtype=np.float32)
        display = compute_merged_shrink_display_intensity(
            shrinkage_probability=shrink,
            porosity_probability=poro,
            voxel_part_types=part_types,
        )
        self.assertAlmostEqual(float(display[3]), 0.0, places=6)

    def test_none_arrays_return_zero(self) -> None:
        display = compute_merged_shrink_display_intensity(
            shrinkage_probability=None,
            porosity_probability=None,
            voxel_part_types=None,
        )
        self.assertEqual(display.shape, (0,))
        self.assertEqual(display.dtype, np.float32)


class FaceConnectivityPocketTests(unittest.TestCase):
    """Verify pocket extraction with 6-connectivity (face neighbors only)."""

    def setUp(self) -> None:
        # 5-voxel line: indices 0..4 with centers spaced 1" apart.
        self.centers = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
             [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        self.sizes = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.volumes = (self.sizes ** 3).astype(np.float32)
        self.neighbors = _build_line_neighbors(5)

    def test_single_voxel_pocket(self) -> None:
        intensity = np.array([0.0, 0.0, 0.8, 0.0, 0.0], dtype=np.float32)
        pocket = extract_pocket_at(
            voxel_index=2,
            display_intensity=intensity,
            voxel_neighbors=self.neighbors,
            voxel_centers=self.centers,
            voxel_volumes=self.volumes,
            shrinkage_risk=intensity,
            porosity_risk=None,
            threshold=0.5,
        )
        self.assertIsNotNone(pocket)
        self.assertEqual(pocket.voxel_count, 1)
        self.assertAlmostEqual(pocket.geometric_volume_in3, 1.0, places=6)
        self.assertEqual(pocket.anchor_voxel_index, 2)
        self.assertEqual(pocket.dominant_defect_type, "shrinkage")

    def test_face_connected_chain(self) -> None:
        # All 5 voxels above threshold and connected by face neighbors.
        intensity = np.array([0.7, 0.6, 0.9, 0.8, 0.75], dtype=np.float32)
        pocket = extract_pocket_at(
            voxel_index=0,
            display_intensity=intensity,
            voxel_neighbors=self.neighbors,
            voxel_centers=self.centers,
            voxel_volumes=self.volumes,
            shrinkage_risk=None,
            porosity_risk=intensity,
            threshold=0.5,
        )
        self.assertIsNotNone(pocket)
        self.assertEqual(pocket.voxel_count, 5)
        self.assertAlmostEqual(pocket.geometric_volume_in3, 5.0, places=6)
        # Centroid of the chain
        self.assertAlmostEqual(pocket.centroid[0], 2.0, places=6)
        self.assertEqual(pocket.dominant_defect_type, "porosity")

    def test_threshold_split(self) -> None:
        # Voxels 2 and 4 above threshold; voxel 3 (between them) below.
        intensity = np.array([0.0, 0.0, 0.7, 0.1, 0.7], dtype=np.float32)
        pocket = extract_pocket_at(
            voxel_index=2,
            display_intensity=intensity,
            voxel_neighbors=self.neighbors,
            voxel_centers=self.centers,
            voxel_volumes=self.volumes,
            shrinkage_risk=intensity,
            porosity_risk=None,
            threshold=0.5,
        )
        self.assertIsNotNone(pocket)
        self.assertEqual(pocket.voxel_count, 1)
        np.testing.assert_array_equal(pocket.voxel_indices, [2])

    def test_anchor_below_threshold_returns_none(self) -> None:
        intensity = np.array([0.0, 0.0, 0.1, 0.0, 0.0], dtype=np.float32)
        pocket = extract_pocket_at(
            voxel_index=2,
            display_intensity=intensity,
            voxel_neighbors=self.neighbors,
            voxel_centers=self.centers,
            voxel_volumes=self.volumes,
            shrinkage_risk=None,
            porosity_risk=None,
            threshold=0.5,
        )
        self.assertIsNone(pocket)

    def test_invalid_voxel_index_returns_none(self) -> None:
        intensity = np.array([0.7, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.assertIsNone(extract_pocket_at(
            voxel_index=-1,
            display_intensity=intensity,
            voxel_neighbors=self.neighbors,
            voxel_centers=self.centers,
            voxel_volumes=self.volumes,
            shrinkage_risk=None,
            porosity_risk=None,
            threshold=0.5,
        ))
        self.assertIsNone(extract_pocket_at(
            voxel_index=5,
            display_intensity=intensity,
            voxel_neighbors=self.neighbors,
            voxel_centers=self.centers,
            voxel_volumes=self.volumes,
            shrinkage_risk=None,
            porosity_risk=None,
            threshold=0.5,
        ))


class CornerConnectivityTests(unittest.TestCase):
    """Verify 6- vs 26-connectivity on a 3D grid."""

    def setUp(self) -> None:
        # 2x2x2 grid; voxels are 1" cubes; voxel 0 and voxel 7 share only a corner.
        self.shape = (2, 2, 2)
        nx, ny, nz = self.shape
        coords = np.array(
            [
                [(x + 0.5), (y + 0.5), (z + 0.5)]
                for z in range(nz) for y in range(ny) for x in range(nx)
            ],
            dtype=np.float32,
        )
        self.centers = coords
        self.sizes = np.full(coords.shape[0], 1.0, dtype=np.float32)
        self.volumes = np.full(coords.shape[0], 1.0, dtype=np.float32)
        self.neighbors = _build_3d_grid_neighbors(self.shape)

    def test_corner_touching_separated_by_6(self) -> None:
        # Voxels 0 and 7 (diagonal corners of the cube) are above threshold.
        intensity = np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8], dtype=np.float32)
        pocket = extract_pocket_at(
            voxel_index=0,
            display_intensity=intensity,
            voxel_neighbors=self.neighbors,
            voxel_centers=self.centers,
            voxel_volumes=self.volumes,
            shrinkage_risk=None,
            porosity_risk=None,
            threshold=0.5,
        )
        self.assertIsNotNone(pocket)
        self.assertEqual(pocket.voxel_count, 1)

    def test_corner_touching_merged_by_26(self) -> None:
        # Same setup, but with 26-connectivity voxels 0 and 7 join into one pocket.
        intensity = np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8], dtype=np.float32)
        pocket = extract_pocket_at(
            voxel_index=0,
            display_intensity=intensity,
            voxel_neighbors=self.neighbors,
            voxel_centers=self.centers,
            voxel_volumes=self.volumes,
            shrinkage_risk=None,
            porosity_risk=None,
            threshold=0.5,
            connectivity=26,
        )
        self.assertIsNotNone(pocket)
        self.assertEqual(pocket.voxel_count, 2)
        self.assertAlmostEqual(pocket.geometric_volume_in3, 2.0, places=6)

    def test_face_connected_chain_3d(self) -> None:
        # All 8 voxels above threshold (a single connected blob in 6-connectivity).
        intensity = np.full(8, 0.7, dtype=np.float32)
        pocket = extract_pocket_at(
            voxel_index=0,
            display_intensity=intensity,
            voxel_neighbors=self.neighbors,
            voxel_centers=self.centers,
            voxel_volumes=self.volumes,
            shrinkage_risk=None,
            porosity_risk=None,
            threshold=0.5,
        )
        self.assertIsNotNone(pocket)
        self.assertEqual(pocket.voxel_count, 8)
        self.assertAlmostEqual(pocket.geometric_volume_in3, 8.0, places=6)
        # Centroid of a 2x2x2 cube is at (1.0, 1.0, 1.0)
        for coord, expected in zip(pocket.centroid, (1.0, 1.0, 1.0)):
            self.assertAlmostEqual(coord, expected, places=6)


class PocketMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.centers = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        self.sizes = np.array([1.0, 1.0, 2.0], dtype=np.float32)
        self.volumes = (self.sizes ** 3).astype(np.float32)
        self.neighbors = _build_line_neighbors(3)

    def test_dominant_defect_type_shrinkage(self) -> None:
        intensity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        shrink = np.array([0.8, 0.0, 0.0], dtype=np.float32)
        poro = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        intensity[0] = max(shrink[0], poro[0])
        pocket = extract_pocket_at(
            voxel_index=0,
            display_intensity=intensity,
            voxel_neighbors=self.neighbors,
            voxel_centers=self.centers,
            voxel_volumes=self.volumes,
            shrinkage_risk=shrink,
            porosity_risk=poro,
            threshold=0.5,
        )
        self.assertEqual(pocket.dominant_defect_type, "shrinkage")

    def test_dominant_defect_type_mixed(self) -> None:
        intensity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        shrink = np.array([0.50, 0.0, 0.0], dtype=np.float32)
        poro = np.array([0.52, 0.0, 0.0], dtype=np.float32)
        intensity[0] = max(shrink[0], poro[0])
        pocket = extract_pocket_at(
            voxel_index=0,
            display_intensity=intensity,
            voxel_neighbors=self.neighbors,
            voxel_centers=self.centers,
            voxel_volumes=self.volumes,
            shrinkage_risk=shrink,
            porosity_risk=poro,
            threshold=0.5,
        )
        self.assertEqual(pocket.dominant_defect_type, "mixed")

    def test_adaptive_volume_uses_per_voxel(self) -> None:
        intensity = np.array([0.0, 0.0, 0.9], dtype=np.float32)
        pocket = extract_pocket_at(
            voxel_index=2,
            display_intensity=intensity,
            voxel_neighbors=self.neighbors,
            voxel_centers=self.centers,
            voxel_volumes=self.volumes,
            shrinkage_risk=None,
            porosity_risk=None,
            threshold=0.5,
        )
        # 2.0**3 = 8.0 in^3 (only voxel 2 above threshold)
        self.assertAlmostEqual(pocket.geometric_volume_in3, 8.0, places=6)

    def test_bounds_and_centroid(self) -> None:
        intensity = np.array([0.0, 0.0, 0.9], dtype=np.float32)
        pocket = extract_pocket_at(
            voxel_index=2,
            display_intensity=intensity,
            voxel_neighbors=self.neighbors,
            voxel_centers=self.centers,
            voxel_volumes=self.volumes,
            shrinkage_risk=None,
            porosity_risk=None,
            threshold=0.5,
        )
        self.assertAlmostEqual(pocket.bounds[0][0], 2.0, places=6)
        self.assertAlmostEqual(pocket.bounds[0][1], 2.0, places=6)
        self.assertAlmostEqual(pocket.centroid[0], 2.0, places=6)
        self.assertEqual(pocket.anchor_position, (2.0, 0.0, 0.0))

    def test_label_format(self) -> None:
        intensity = np.array([0.0, 0.0, 0.9], dtype=np.float32)
        pocket = extract_pocket_at(
            voxel_index=2,
            display_intensity=intensity,
            voxel_neighbors=self.neighbors,
            voxel_centers=self.centers,
            voxel_volumes=self.volumes,
            shrinkage_risk=None,
            porosity_risk=None,
            threshold=0.5,
        )
        label = format_pocket_volume_label(pocket)
        self.assertIn("in^3", label)
        self.assertIn("8.000", label)


class MeshShapeValidationTests(unittest.TestCase):
    def test_shape_mismatch_returns_none(self) -> None:
        centers = np.zeros((4, 3), dtype=np.float32)
        sizes = np.zeros(3, dtype=np.float32)  # wrong length
        volumes = np.zeros(3, dtype=np.float32)
        neighbors = np.full((4, 26), -1, dtype=np.int32)
        intensity = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        pocket = extract_pocket_at(
            voxel_index=0,
            display_intensity=intensity,
            voxel_neighbors=neighbors,
            voxel_centers=centers,
            voxel_volumes=volumes,
            shrinkage_risk=None,
            porosity_risk=None,
            threshold=0.0,
        )
        self.assertIsNone(pocket)

    def test_invalid_connectivity_raises(self) -> None:
        centers = np.zeros((2, 3), dtype=np.float32)
        sizes = np.ones(2, dtype=np.float32)
        volumes = np.ones(2, dtype=np.float32)
        neighbors = _build_line_neighbors(2)
        intensity = np.ones(2, dtype=np.float32)
        with self.assertRaises(ValueError):
            extract_pocket_at(
                voxel_index=0,
                display_intensity=intensity,
                voxel_neighbors=neighbors,
                voxel_centers=centers,
                voxel_volumes=volumes,
                shrinkage_risk=None,
                porosity_risk=None,
                threshold=0.0,
                connectivity=12,  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
