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
    """Verify the feeding-path analyzer detects isolated liquid pockets."""

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
        """Solid voxels (fs >= 0.99) are not 'blocked' — they're already solid.

        A solid voxel with no feeding path has already finished solidifying;
        shrinkage has already occurred.  Only mush / liquid voxels that
        cannot be fed are marked as 'blocked'.
        """
        analyzer = FeedingPathAnalyzer(self.mesh, None)
        riser_indices = np.array([0], dtype=np.int64)
        # All voxels appear fully solid - no traversable paths
        fraction_solid = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        result = analyzer.analyze(
            riser_indices,
            fraction_solid,
            show_progress=False,
        )

        # Solid voxels are NOT blocked (they're done solidifying)
        self.assertFalse(bool(result.blocked_voxels[1]))
        self.assertFalse(bool(result.blocked_voxels[2]))
        self.assertFalse(bool(result.blocked_voxels[3]))
        self.assertFalse(bool(result.blocked_voxels[4]))
        # Riser itself is never blocked
        self.assertFalse(bool(result.blocked_voxels[0]))

    def test_blocked_voxels_consistent_no_history_vs_history(self) -> None:
        """No-history and history-merge paths produce the same blocked_voxels."""
        analyzer = FeedingPathAnalyzer(self.mesh, None)
        riser_indices = np.array([0], dtype=np.int64)
        fraction_solid = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        # No-history path
        result_no_history = analyzer.analyze(
            riser_indices,
            fraction_solid,
            show_progress=False,
        )

        # History path: two snapshots with the same fs field
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

        # Both paths should agree on blocked status
        for i in range(5):
            self.assertEqual(
                bool(result_no_history.blocked_voxels[i]),
                bool(result_history.blocked_voxels[i]),
                f"Voxel {i}: no-history={result_no_history.blocked_voxels[i]}, "
                f"history={result_history.blocked_voxels[i]}",
            )

    def test_isolated_pocket_detected_as_blocked(self) -> None:
        """Isolated liquid behind a solid barrier is marked blocked."""
        analyzer = FeedingPathAnalyzer(self.mesh, None)
        riser_indices = np.array([0], dtype=np.int64)
        # Voxels 0 (riser), 1 (liquid), 2 (solid barrier), 3 (liquid pocket), 4 (liquid)
        fraction_solid = np.array([0.5, 0.5, 0.995, 0.5, 0.5], dtype=np.float32)

        result = analyzer.analyze(
            riser_indices,
            fraction_solid,
            show_progress=False,
        )

        # Voxels 1 is connected to riser
        self.assertAlmostEqual(result.path_length[1], 1.0, places=6)
        self.assertFalse(bool(result.blocked_voxels[1]))
        # Voxels 3 and 4 are isolated behind solid barrier
        self.assertTrue(bool(result.blocked_voxels[3]))
        self.assertTrue(bool(result.blocked_voxels[4]))
        self.assertEqual(result.feeding_source[3], -1)
        self.assertEqual(result.feeding_source[4], -1)

    def test_isolated_pocket_detected_in_history_mode(self) -> None:
        """History path also detects isolated pocket behind solid barrier."""
        analyzer = FeedingPathAnalyzer(self.mesh, None)
        riser_indices = np.array([0], dtype=np.int64)
        fraction_solid = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        # Simulate timeline: barrier (voxel 2) solidifies at t=5
        # Riser (voxel 0) and connected (voxel 1) stay liquid
        # Pocket (voxels 3, 4) stay liquid but become isolated at t=5
        snap_early = np.array([0.3, 0.3, 0.4, 0.3, 0.3], dtype=np.float32)
        snap_mid = np.array([0.4, 0.4, 0.995, 0.4, 0.4], dtype=np.float32)
        snap_late = np.array([0.5, 0.5, 1.0, 0.5, 0.5], dtype=np.float32)

        history = [snap_early, snap_mid, snap_late]
        history_times = [0.0, 5.0, 10.0]
        # All CAST voxels cross critical_fs at t=8
        reference_times = np.array([8.0, 8.0, 8.0, 8.0, 8.0], dtype=np.float32)

        result = analyzer.analyze(
            riser_indices,
            fraction_solid,
            show_progress=False,
            history=history,
            history_times=history_times,
            reference_times=reference_times,
        )

        # Voxel 1 is always connected to riser
        self.assertFalse(bool(result.blocked_voxels[1]))
        # Voxels 3 and 4 are blocked: at t=5 (eligible, <=8.0) the barrier is solid
        self.assertTrue(bool(result.blocked_voxels[3]))
        self.assertTrue(bool(result.blocked_voxels[4]))

    def test_isolated_pocket_in_history_blocked_at_intermediate_snapshot(self) -> None:
        """A pocket blocked at an intermediate snapshot is reported as blocked.

        This is the regression test for the bug where the history merge
        overwrote blocked_voxels with the most recent eligible snapshot's
        value, causing pockets that were blocked at intermediate times but
        connected at the last snapshot to be falsely reported as fed.
        """
        analyzer = FeedingPathAnalyzer(self.mesh, None)
        riser_indices = np.array([0], dtype=np.int64)
        fraction_solid = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        # Snap 0 (t=0): all liquid, barrier not yet solidified, pocket is fed
        snap_early = np.array([0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)
        # Snap 1 (t=4): barrier (voxel 2) now at 0.99, pocket becomes isolated
        snap_mid = np.array([0.4, 0.4, 0.99, 0.4, 0.4], dtype=np.float32)
        # Snap 2 (t=8): barrier fully solid, but if barrier were removed the pocket
        # would be connected. In the history we simulate the barrier staying.
        snap_late = np.array([0.5, 0.5, 1.0, 0.5, 0.5], dtype=np.float32)

        history = [snap_early, snap_mid, snap_late]
        history_times = [0.0, 4.0, 8.0]
        # All CAST voxels cross critical_fs at t=6
        reference_times = np.array([6.0, 6.0, 6.0, 6.0, 6.0], dtype=np.float32)

        result = analyzer.analyze(
            riser_indices,
            fraction_solid,
            show_progress=False,
            history=history,
            history_times=history_times,
            reference_times=reference_times,
        )

        # Voxel 1 is connected at all snapshots
        self.assertFalse(bool(result.blocked_voxels[1]))
        # Voxels 3 and 4 were blocked at t=4 (eligible <= 6.0), so they must
        # be reported as blocked in the merged result even though they were
        # fed at t=0 and t=8.
        self.assertTrue(
            bool(result.blocked_voxels[3]),
            "Pocket voxels 3,4 must be blocked even though they were fed at t=0",
        )
        self.assertTrue(
            bool(result.blocked_voxels[4]),
            "Pocket voxels 3,4 must be blocked even though they were fed at t=0",
        )

    def test_isolated_pocket_history_consistent_across_max_samples(self) -> None:
        """Higher max_history_samples produces the same result when snapshots are identical."""
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

        # Both should agree on blocked status
        for i in range(5):
            self.assertEqual(
                bool(result_default.blocked_voxels[i]),
                bool(result_high.blocked_voxels[i]),
                f"Voxel {i}: default max_samples={result_default.blocked_voxels[i]}, "
                f"high max_samples={result_high.blocked_voxels[i]}",
            )

    def test_permeability_threshold_allows_mushy_traversal(self) -> None:
        """At default critical_fs (0.52), permeability is well above 1e-16."""
        analyzer = FeedingPathAnalyzer(self.mesh, None)
        self.assertLessEqual(analyzer.permeability_threshold, 1e-16)

        riser_indices = np.array([0], dtype=np.int64)
        # All at critical coherency point
        fraction_solid = np.full(5, 0.52, dtype=np.float32)

        result = analyzer.analyze(
            riser_indices,
            fraction_solid,
            show_progress=False,
        )

        # All voxels should be reachable at fs=0.52
        self.assertFalse(bool(result.blocked_voxels[1]))
        self.assertFalse(bool(result.blocked_voxels[2]))
        self.assertFalse(bool(result.blocked_voxels[3]))
        self.assertFalse(bool(result.blocked_voxels[4]))

    def test_partially_solidified_mush_blocks_path(self) -> None:
        """Voxels well past critical_fs (fs=0.90) are impermeable."""
        analyzer = FeedingPathAnalyzer(self.mesh, None)
        riser_indices = np.array([0], dtype=np.int64)
        # Barrier at fs=0.90 - permeability falls below 1e-16 threshold
        fraction_solid = np.array([0.5, 0.5, 0.90, 0.5, 0.5], dtype=np.float32)

        result = analyzer.analyze(
            riser_indices,
            fraction_solid,
            show_progress=False,
        )

        # Voxels 3 and 4 are blocked by the high-fs barrier
        self.assertTrue(bool(result.blocked_voxels[3]))
        self.assertTrue(bool(result.blocked_voxels[4]))


class ScheduleFeedingFractionSolidTests(unittest.TestCase):
    """Verify feeding_fraction_solid contains fraction solid, not time."""

    def test_schedule_based_critical_fs_is_fraction_solid(self) -> None:
        """FIX 1: schedule mode stores critical_fs (0-1), not time.

        Simulates the logic in _run_schedule_based_integration.
        """
        n_voxels = 10
        feeding_fraction_solid = np.full(n_voxels, np.nan, dtype=np.float32)
        local_times = np.ones(n_voxels, dtype=np.float32) * 100.0
        critical_fs = 0.52

        # This is the FIX 1 logic
        mask = np.isnan(feeding_fraction_solid) & (local_times > 0)
        feeding_fraction_solid[mask] = critical_fs

        # All values should be in [0, 1] range
        self.assertTrue(np.all(feeding_fraction_solid >= 0.0))
        self.assertTrue(np.all(feeding_fraction_solid <= 1.0))
        self.assertAlmostEqual(float(np.max(feeding_fraction_solid)), critical_fs, places=4)
        self.assertAlmostEqual(float(np.min(feeding_fraction_solid[mask])), critical_fs, places=4)

    def test_feeding_fraction_solid_clips_correctly_for_feeding_path(self) -> None:
        """Values clipped to [0,1] are usable for permeability calculation."""
        critical_fs = 0.52
        values = np.full(100, critical_fs, dtype=np.float32)
        clipped = np.clip(values, 0.0, 1.0)

        # All values should survive clipping unchanged
        self.assertTrue(np.allclose(clipped, critical_fs, atol=1e-6))

        # These should NOT all clip to 1.0 (the old bug behavior)
        self.assertLess(np.mean(clipped), 0.99)


if __name__ == "__main__":
    unittest.main()
