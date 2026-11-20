from __future__ import annotations

import unittest

import numpy as np

from core.acceleration import compute_isolation_score_parallel

def _all_cast(solid_times: np.ndarray) -> np.ndarray:
    return np.zeros(len(solid_times), dtype=np.int32)

class ComputeIsolationScoreParallelTests(unittest.TestCase):

    def test_returns_array_of_correct_shape(self) -> None:
        n_voxels = 1000
        solid_times = np.zeros(n_voxels, dtype=np.float32)
        neighbors = np.full((n_voxels, 26), -1, dtype=np.int32)

        result = compute_isolation_score_parallel(solid_times, neighbors, _all_cast(solid_times))

        self.assertEqual(result.shape, (n_voxels,))
        self.assertEqual(result.dtype, np.float32)

    def test_zero_solidification_times_yield_zero_scores(self) -> None:
        n_voxels = 500
        solid_times = np.zeros(n_voxels, dtype=np.float32)
        rng = np.random.default_rng(0)
        neighbors = rng.integers(-1, n_voxels, (n_voxels, 26), dtype=np.int32)

        result = compute_isolation_score_parallel(solid_times, neighbors, _all_cast(solid_times))

        self.assertTrue(np.all(result == 0.0))

    def test_simple_three_voxel_chain(self) -> None:
        solid_times = np.array([10.0, 5.0, 0.0], dtype=np.float32)
        neighbors = np.array(
            [
                [1, 2, -1, -1, -1, -1],
                [0, 2, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
            ],
            dtype=np.int32,
        )

        result = compute_isolation_score_parallel(solid_times, neighbors, _all_cast(solid_times))

        self.assertAlmostEqual(result[0], 0.5, places=5)
        self.assertAlmostEqual(result[1], 0.0, places=5)
        self.assertAlmostEqual(result[2], 0.0, places=5)

    def test_full_isolation_when_all_neighbors_solidified_first(self) -> None:
        n_voxels = 10
        solid_times = np.full(n_voxels, 1.0, dtype=np.float32)
        solid_times[9] = 100.0
        neighbors = np.full((n_voxels, 26), -1, dtype=np.int32)
        for j in range(9):
            neighbors[9, j] = j

        result = compute_isolation_score_parallel(solid_times, neighbors, _all_cast(solid_times))

        self.assertAlmostEqual(result[9], 1.0, places=5)

    def test_large_random_runs_fast(self) -> None:
        rng = np.random.default_rng(42)
        n_voxels = 50_000
        solid_times = rng.uniform(0, 100, n_voxels).astype(np.float32)
        solid_times[rng.choice(n_voxels, 5_000, replace=False)] = 0.0
        neighbors = rng.integers(-1, n_voxels, (n_voxels, 26), dtype=np.int32)

        compute_isolation_score_parallel(solid_times, neighbors, _all_cast(solid_times))

        import time

        t0 = time.perf_counter()
        for _ in range(3):
            result = compute_isolation_score_parallel(solid_times, neighbors, _all_cast(solid_times))
        elapsed = (time.perf_counter() - t0) / 3.0

        self.assertEqual(result.shape, (n_voxels,))
        self.assertLess(elapsed, 2.0)

    def test_chill_neighbors_are_excluded_from_both_counts(self) -> None:
        solid_times = np.array([10.0, 5.0, 0.0], dtype=np.float32)
        neighbors = np.array(
            [
                [1, 2, -1, -1, -1, -1],
                [0, 2, -1, -1, -1, -1],
                [0, 1, -1, -1, -1, -1],
            ],
            dtype=np.int32,
        )
        part_types = np.array([0, 0, 2], dtype=np.int32)

        result = compute_isolation_score_parallel(solid_times, neighbors, part_types)

        self.assertAlmostEqual(result[0], 1.0, places=5)

    def test_core_neighbors_are_excluded_like_chills(self) -> None:
        solid_times = np.array([10.0, 5.0, 0.0], dtype=np.float32)
        neighbors = np.array(
            [
                [1, 2, -1, -1, -1, -1],
                [0, 2, -1, -1, -1, -1],
                [0, 1, -1, -1, -1, -1],
            ],
            dtype=np.int32,
        )
        part_types = np.array([0, 0, 4], dtype=np.int32)

        result = compute_isolation_score_parallel(solid_times, neighbors, part_types)

        self.assertAlmostEqual(result[0], 1.0, places=5)

    def test_riser_neighbors_still_count(self) -> None:
        solid_times = np.array([10.0, 5.0, 20.0], dtype=np.float32)
        neighbors = np.array(
            [
                [1, 2, -1, -1, -1, -1],
                [0, 2, -1, -1, -1, -1],
                [0, 1, -1, -1, -1, -1],
            ],
            dtype=np.int32,
        )
        part_types = np.array([0, 0, 1], dtype=np.int32)

        result = compute_isolation_score_parallel(solid_times, neighbors, part_types)

        self.assertAlmostEqual(result[0], 0.5, places=5)

    def test_negative_neighbor_indices_are_skipped(self) -> None:
        solid_times = np.array([5.0, 0.0, 0.0], dtype=np.float32)
        neighbors = np.array(
            [
                [-1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
            ],
            dtype=np.int32,
        )

        result = compute_isolation_score_parallel(solid_times, neighbors, _all_cast(solid_times))

        self.assertAlmostEqual(result[0], 0.0, places=5)

if __name__ == "__main__":
    unittest.main()
