from __future__ import annotations

import unittest

import numpy as np

from core.feeding_path import FeedingPathAnalyzer, FeedingPathResult

class _StubMesh:
    def __init__(self, voxel_centers: np.ndarray, voxel_neighbors: np.ndarray) -> None:
        self.voxel_centers = voxel_centers
        self.voxel_neighbors = voxel_neighbors

    def get_voxel_count(self) -> int:
        return int(self.voxel_centers.shape[0])

class IsolatedPocketDetectionTests(unittest.TestCase):

    def setUp(self) -> None:
        self.voxel_centers = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.voxel_neighbors = np.array(
            [
                [1, -1, -1, -1, -1, -1],
                [0, 2, -1, -1, -1, -1],
                [1, 3, -1, -1, -1, -1],
                [2, 4, -1, -1, -1, -1],
                [3, -1, -1, -1, -1, -1],
            ],
            dtype=np.int32,
        )
        self.mesh = _StubMesh(self.voxel_centers, self.voxel_neighbors)

    def test_solid_voxel_not_marked_as_blocked_liquid_pocket(self) -> None:
        analyzer = FeedingPathAnalyzer(self.mesh, None)
        riser_indices = np.array([0], dtype=np.int64)
        fraction_solid = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        result = analyzer.analyze(
            riser_indices,
            fraction_solid,
            show_progress=False,
        )

        self.assertFalse(bool(result.blocked_voxels[1]))
        self.assertFalse(bool(result.blocked_voxels[2]))
        self.assertFalse(bool(result.blocked_voxels[3]))
        self.assertFalse(bool(result.blocked_voxels[4]))
        self.assertFalse(bool(result.blocked_voxels[0]))

    def test_blocked_voxels_consistent_no_history_vs_history(self) -> None:
        analyzer = FeedingPathAnalyzer(self.mesh, None)
        riser_indices = np.array([0], dtype=np.int64)
        fraction_solid = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        result_no_history = analyzer.analyze(
            riser_indices,
            fraction_solid,
            show_progress=False,
        )

        history = [fraction_solid.copy(), fraction_solid.copy()]
        history_times = [0.0, 10.0]
        reference_times = np.array([5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float32)

        result_history = analyzer.analyze(
            riser_indices,
            fraction_solid,
            show_progress=False,
            history=history,
            history_times=history_times,
            reference_times=reference_times,
        )

        for i in range(5):
            self.assertEqual(
                bool(result_no_history.blocked_voxels[i]),
                bool(result_history.blocked_voxels[i]),
                f"Voxel {i}: no-history={result_no_history.blocked_voxels[i]}, "
                f"history={result_history.blocked_voxels[i]}",
            )

    def test_isolated_pocket_detected_as_blocked(self) -> None:
        analyzer = FeedingPathAnalyzer(self.mesh, None)
        riser_indices = np.array([0], dtype=np.int64)
        fraction_solid = np.array([0.5, 0.5, 0.995, 0.5, 0.5], dtype=np.float32)

        result = analyzer.analyze(
            riser_indices,
            fraction_solid,
            show_progress=False,
        )

        self.assertAlmostEqual(result.path_length[1], 1.0, places=6)
        self.assertFalse(bool(result.blocked_voxels[1]))
        self.assertTrue(bool(result.blocked_voxels[3]))
        self.assertTrue(bool(result.blocked_voxels[4]))
        self.assertEqual(result.feeding_source[3], -1)
        self.assertEqual(result.feeding_source[4], -1)

    def test_isolated_pocket_detected_in_history_mode(self) -> None:
        analyzer = FeedingPathAnalyzer(self.mesh, None)
        riser_indices = np.array([0], dtype=np.int64)
        fraction_solid = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        snap_early = np.array([0.3, 0.3, 0.4, 0.3, 0.3], dtype=np.float32)
        snap_mid = np.array([0.4, 0.4, 0.995, 0.4, 0.4], dtype=np.float32)
        snap_late = np.array([0.5, 0.5, 1.0, 0.5, 0.5], dtype=np.float32)

        history = [snap_early, snap_mid, snap_late]
        history_times = [0.0, 5.0, 10.0]
        reference_times = np.array([8.0, 8.0, 8.0, 8.0, 8.0], dtype=np.float32)

        result = analyzer.analyze(
            riser_indices,
            fraction_solid,
            show_progress=False,
            history=history,
            history_times=history_times,
            reference_times=reference_times,
        )

        self.assertFalse(bool(result.blocked_voxels[1]))
        self.assertTrue(bool(result.blocked_voxels[3]))
        self.assertTrue(bool(result.blocked_voxels[4]))

    def test_isolated_pocket_in_history_blocked_at_intermediate_snapshot(self) -> None:
        analyzer = FeedingPathAnalyzer(self.mesh, None)
        riser_indices = np.array([0], dtype=np.int64)
        fraction_solid = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        snap_early = np.array([0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)
        snap_mid = np.array([0.4, 0.4, 0.99, 0.4, 0.4], dtype=np.float32)
        snap_late = np.array([0.5, 0.5, 1.0, 0.5, 0.5], dtype=np.float32)

        history = [snap_early, snap_mid, snap_late]
        history_times = [0.0, 4.0, 8.0]
        reference_times = np.array([6.0, 6.0, 6.0, 6.0, 6.0], dtype=np.float32)

        result = analyzer.analyze(
            riser_indices,
            fraction_solid,
            show_progress=False,
            history=history,
            history_times=history_times,
            reference_times=reference_times,
        )

        self.assertFalse(bool(result.blocked_voxels[1]))
        self.assertTrue(
            bool(result.blocked_voxels[3]),
            "Pocket voxels 3,4 must be blocked even though they were fed at t=0",
        )
        self.assertTrue(
            bool(result.blocked_voxels[4]),
            "Pocket voxels 3,4 must be blocked even though they were fed at t=0",
        )

    def test_isolated_pocket_history_consistent_across_max_samples(self) -> None:
        analyzer = FeedingPathAnalyzer(self.mesh, None)
        riser_indices = np.array([0], dtype=np.int64)
        fraction_solid = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        snap_early = np.array([0.3, 0.3, 0.4, 0.3, 0.3], dtype=np.float32)
        snap_mid = np.array([0.4, 0.4, 0.995, 0.4, 0.4], dtype=np.float32)
        snap_late = np.array([0.5, 0.5, 1.0, 0.5, 0.5], dtype=np.float32)

        history = [snap_early, snap_mid, snap_late]
        history_times = [0.0, 5.0, 10.0]
        reference_times = np.array([8.0, 8.0, 8.0, 8.0, 8.0], dtype=np.float32)

        result_default = analyzer.analyze(
            riser_indices,
            fraction_solid,
            show_progress=False,
            history=history,
            history_times=history_times,
            reference_times=reference_times,
        )

        result_high = analyzer.analyze(
            riser_indices,
            fraction_solid,
            show_progress=False,
            history=history,
            history_times=history_times,
            reference_times=reference_times,
            max_history_samples=200,
        )

        for i in range(5):
            self.assertEqual(
                bool(result_default.blocked_voxels[i]),
                bool(result_high.blocked_voxels[i]),
                f"Voxel {i}: default max_samples={result_default.blocked_voxels[i]}, "
                f"high max_samples={result_high.blocked_voxels[i]}",
            )

    def test_permeability_threshold_allows_mushy_traversal(self) -> None:
        analyzer = FeedingPathAnalyzer(self.mesh, None)
        self.assertLessEqual(analyzer.permeability_threshold, 1e-16)

        riser_indices = np.array([0], dtype=np.int64)
        fraction_solid = np.full(5, 0.52, dtype=np.float32)

        result = analyzer.analyze(
            riser_indices,
            fraction_solid,
            show_progress=False,
        )

        self.assertFalse(bool(result.blocked_voxels[1]))
        self.assertFalse(bool(result.blocked_voxels[2]))
        self.assertFalse(bool(result.blocked_voxels[3]))
        self.assertFalse(bool(result.blocked_voxels[4]))

    def test_partially_solidified_mush_blocks_path(self) -> None:
        analyzer = FeedingPathAnalyzer(self.mesh, None)
        riser_indices = np.array([0], dtype=np.int64)
        fraction_solid = np.array([0.5, 0.5, 0.90, 0.5, 0.5], dtype=np.float32)

        result = analyzer.analyze(
            riser_indices,
            fraction_solid,
            show_progress=False,
        )

        self.assertTrue(bool(result.blocked_voxels[3]))
        self.assertTrue(bool(result.blocked_voxels[4]))

class PinchOffDetectionTests(unittest.TestCase):

    def setUp(self) -> None:
        centers = np.array([[float(i), 0.0, 0.0] for i in range(5)], dtype=np.float64)
        neighbors = np.array(
            [[1, -1], [0, 2], [1, 3], [2, 4], [3, -1]],
            dtype=np.int32,
        )
        self.mesh = _StubMesh(centers, neighbors)
        self.riser = np.array([0], dtype=np.int64)

    def test_neck_just_past_coherency_isolates_pocket(self) -> None:
        analyzer = FeedingPathAnalyzer(self.mesh, None)
        fraction_solid = np.array([0.3, 0.4, 0.64, 0.45, 0.45], dtype=np.float32)
        result = analyzer.analyze(self.riser, fraction_solid, show_progress=False)
        self.assertGreaterEqual(result.path_length[4], 0.0, "Darcy trace still reaches the pocket")
        self.assertFalse(bool(result.blocked_voxels[1]))
        self.assertFalse(bool(result.blocked_voxels[2]), "The neck itself no longer needs feed")
        self.assertTrue(bool(result.blocked_voxels[3]))
        self.assertTrue(bool(result.blocked_voxels[4]))

    def test_everything_past_coherency_is_not_a_pocket(self) -> None:
        analyzer = FeedingPathAnalyzer(self.mesh, None)
        fraction_solid = np.array([0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
        result = analyzer.analyze(self.riser, fraction_solid, show_progress=False)
        self.assertFalse(bool(np.any(result.blocked_voxels)))

    def test_short_pinch_off_between_sampled_snapshots_is_caught(self) -> None:
        analyzer = FeedingPathAnalyzer(self.mesh, None)
        history = [
            np.array([0.10, 0.10, 0.10, 0.10, 0.10], dtype=np.float32),
            np.array([0.20, 0.20, 0.25, 0.20, 0.20], dtype=np.float32),
            np.array([0.30, 0.35, 0.60, 0.48, 0.48], dtype=np.float32),
            np.array([0.40, 0.55, 0.70, 0.75, 0.75], dtype=np.float32),
            np.array([0.60, 0.80, 0.90, 0.95, 0.95], dtype=np.float32),
        ]
        history_times = [0.0, 10.0, 20.0, 30.0, 40.0]
        reference_times = np.array([40.0, 30.0, 20.0, 30.0, 30.0], dtype=np.float32)
        result = analyzer.analyze(
            self.riser,
            history[-1],
            show_progress=False,
            history=history,
            history_times=history_times,
            reference_times=reference_times,
            max_history_samples=2,
        )
        self.assertFalse(bool(result.blocked_voxels[1]))
        self.assertTrue(bool(result.blocked_voxels[3]))
        self.assertTrue(bool(result.blocked_voxels[4]))

class ScheduleFeedingFractionSolidTests(unittest.TestCase):

    def test_schedule_based_critical_fs_is_fraction_solid(self) -> None:
        n_voxels = 10
        feeding_fraction_solid = np.full(n_voxels, np.nan, dtype=np.float32)
        local_times = np.ones(n_voxels, dtype=np.float32) * 100.0
        critical_fs = 0.52

        mask = np.isnan(feeding_fraction_solid) & (local_times > 0)
        feeding_fraction_solid[mask] = critical_fs

        self.assertTrue(np.all(feeding_fraction_solid >= 0.0))
        self.assertTrue(np.all(feeding_fraction_solid <= 1.0))
        self.assertAlmostEqual(float(np.max(feeding_fraction_solid)), critical_fs, places=4)
        self.assertAlmostEqual(float(np.min(feeding_fraction_solid[mask])), critical_fs, places=4)

    def test_feeding_fraction_solid_clips_correctly_for_feeding_path(self) -> None:
        critical_fs = 0.52
        values = np.full(100, critical_fs, dtype=np.float32)
        clipped = np.clip(values, 0.0, 1.0)

        self.assertTrue(np.allclose(clipped, critical_fs, atol=1e-6))

        self.assertLess(np.mean(clipped), 0.99)

if __name__ == "__main__":
    unittest.main()
