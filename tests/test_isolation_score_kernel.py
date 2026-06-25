"""Tests for the JIT isolation-score kernel."""

from __future__ import annotations

import unittest

import numpy as np

from core.acceleration import compute_isolation_score_parallel


class ComputeIsolationScoreParallelTests(unittest.TestCase):
    """Validate the Numba `prange` replacement for the pure-Python loop."""

    def test_returns_array_of_correct_shape(self) -> None:
        n_voxels = 1000
        solid_times = np.zeros(n_voxels, dtype=np.float32)
        neighbors = np.full((n_voxels, 26), -1, dtype=np.int32)

        result = compute_isolation_score_parallel(solid_times, neighbors)

        self.assertEqual(result.shape, (n_voxels,))
        self.assertEqual(result.dtype, np.float32)

    def test_zero_solidification_times_yield_zero_scores(self) -> None:
        n_voxels = 500
        solid_times = np.zeros(n_voxels, dtype=np.float32)
        rng = np.random.default_rng(0)
        neighbors = rng.integers(-1, n_voxels, (n_voxels, 26), dtype=np.int32)

        result = compute_isolation_score_parallel(solid_times, neighbors)

        self.assertTrue(np.all(result == 0.0))

    def test_simple_three_voxel_chain(self) -> None:
        # Voxel 0 solidifies last, voxel 1 first, voxel 2 never.
        # Voxel 0 has neighbors [1, 2]. neighbor 1 solidified before, neighbor 2
        # never solidified (excluded from count), so ratio = 1/2.
        solid_times = np.array([10.0, 5.0, 0.0], dtype=np.float32)
        neighbors = np.array(
            [
                [1, 2, -1, -1, -1, -1],
                [0, 2, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
            ],
            dtype=np.int32,
        )

        result = compute_isolation_score_parallel(solid_times, neighbors)

        # Voxel 0: neighbors_before=1, total=2 → 0.5
        self.assertAlmostEqual(result[0], 0.5, places=5)
        # Voxel 1: solidifies first, no earlier neighbor → 0.0
        self.assertAlmostEqual(result[1], 0.0, places=5)
        # Voxel 2: never solidified → 0.0
        self.assertAlmostEqual(result[2], 0.0, places=5)

    def test_full_isolation_when_all_neighbors_solidified_first(self) -> None:
        n_voxels = 10
        # Voxel 9 solidifies last; all other voxels solidified earlier.
        solid_times = np.full(n_voxels, 1.0, dtype=np.float32)
        solid_times[9] = 100.0
        # Every other voxel is a neighbor of voxel 9.
        neighbors = np.full((n_voxels, 26), -1, dtype=np.int32)
        for j in range(9):
            neighbors[9, j] = j

        result = compute_isolation_score_parallel(solid_times, neighbors)

        # Voxel 9 has 9 neighbors, all solidified before → ratio = 1.0
        self.assertAlmostEqual(result[9], 1.0, places=5)

    def test_large_random_runs_fast(self) -> None:
        rng = np.random.default_rng(42)
        n_voxels = 50_000
        solid_times = rng.uniform(0, 100, n_voxels).astype(np.float32)
        solid_times[rng.choice(n_voxels, 5_000, replace=False)] = 0.0
        neighbors = rng.integers(-1, n_voxels, (n_voxels, 26), dtype=np.int32)

        # First call warms the JIT cache; subsequent calls are fast.
        compute_isolation_score_parallel(solid_times, neighbors)

        import time

        t0 = time.perf_counter()
        for _ in range(3):
            result = compute_isolation_score_parallel(solid_times, neighbors)
        elapsed = (time.perf_counter() - t0) / 3.0

        self.assertEqual(result.shape, (n_voxels,))
        # Should be well under a second on any modern CPU; allow generous slack.
        self.assertLess(elapsed, 2.0)

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

        result = compute_isolation_score_parallel(solid_times, neighbors)

        # No real neighbors → score stays 0.0
        self.assertAlmostEqual(result[0], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()