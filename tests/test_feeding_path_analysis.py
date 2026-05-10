from __future__ import annotations

import unittest

import numpy as np

from core.feeding_path import FeedingPathAnalyzer


class _StubMesh:
    def __init__(self, voxel_centers: np.ndarray, voxel_neighbors: np.ndarray) -> None:
        self.voxel_centers = voxel_centers
        self.voxel_neighbors = voxel_neighbors

    def get_voxel_count(self) -> int:
        return int(self.voxel_centers.shape[0])


class FeedingPathAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.voxel_centers = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.voxel_neighbors = np.array(
            [
                [1, -1, -1, -1, -1, -1],
                [0, 2, -1, -1, -1, -1],
                [1, -1, -1, -1, -1, -1],
            ],
            dtype=np.int32,
        )
        self.mesh = _StubMesh(self.voxel_centers, self.voxel_neighbors)
        self.analyzer = FeedingPathAnalyzer(self.mesh, None)

    def test_linear_mesh_feeds_from_riser(self) -> None:
        riser_indices = np.array([0], dtype=np.int64)
        fraction_solid = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        result = self.analyzer.analyze(
            riser_indices,
            fraction_solid,
            show_progress=False,
        )

        self.assertAlmostEqual(result.path_length[0], 0.0, places=6)
        self.assertAlmostEqual(result.path_length[1], 1.0, places=6)
        self.assertAlmostEqual(result.path_length[2], 2.0, places=6)
        self.assertEqual(result.feeding_source[1], 0)
        self.assertEqual(result.feeding_source[2], 0)
        self.assertFalse(bool(result.blocked_voxels[1]))
        self.assertFalse(bool(result.blocked_voxels[2]))

    def test_solid_voxels_are_blocked_from_feeding(self) -> None:
        riser_indices = np.array([0], dtype=np.int64)
        fraction_solid = np.array([0.5, 0.995, 0.995], dtype=np.float32)

        result = self.analyzer.analyze(
            riser_indices,
            fraction_solid,
            show_progress=False,
        )

        self.assertAlmostEqual(result.path_length[0], 0.0, places=6)
        self.assertAlmostEqual(result.path_length[1], -1.0, places=6)
        self.assertAlmostEqual(result.path_length[2], -1.0, places=6)
        self.assertEqual(result.feeding_source[1], -1)
        self.assertEqual(result.feeding_source[2], -1)
        self.assertAlmostEqual(result.fed_fraction, 1.0 / 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
