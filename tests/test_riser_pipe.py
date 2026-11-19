"""Regression tests for the gravity-drained riser pipe (physics/riser_pipe.py)."""

from __future__ import annotations

import unittest

import numpy as np

from physics.riser_pipe import gravity_drain_pipe, label_unfed_pools


def _stack(times):
    """Ten unit voxels in a vertical column. z >= 3 is the riser."""
    z = np.arange(10.0)
    return (
        np.asarray(times, dtype=np.float64),
        z,
        np.ones(10),
        np.zeros(10, dtype=np.int64),
        z >= 3,
    )


class GravityDrainPipeTests(unittest.TestCase):
    def test_volume_is_conserved(self) -> None:
        t, z, v, pools, feeder = _stack([10, 12, 14, 30, 40, 50, 60, 70, 80, 5])
        fraction, (pool,) = gravity_drain_pipe(t, z, v, pools, feeder, 0.05)
        self.assertAlmostEqual(pool.pipe_volume + pool.unfed_volume, pool.demand_volume, places=9)
        self.assertAlmostEqual(pool.demand_volume, 0.05 * (10.0 - pool.pipe_volume), places=9)
        self.assertAlmostEqual(float(fraction.sum()), pool.pipe_volume, places=5)

    def test_sound_riser_keeps_pipe_under_crust_and_out_of_casting(self) -> None:
        t, z, v, pools, feeder = _stack([10, 12, 14, 30, 40, 50, 60, 70, 80, 5])
        fraction, (pool,) = gravity_drain_pipe(t, z, v, pools, feeder, 0.05)
        self.assertEqual(pool.section_pipe_volume, 0.0)
        self.assertEqual(int(np.argmax(fraction)), 8)  # highest liquid under the frozen top
        self.assertEqual(float(fraction[9]), 0.0)

    def test_riser_freezing_first_sends_pipe_into_casting(self) -> None:
        t, z, v, pools, feeder = _stack([100, 90, 80, 20, 15, 12, 10, 8, 6, 5])
        _fraction, (pool,) = gravity_drain_pipe(t, z, v, pools, feeder, 0.05)
        self.assertGreater(pool.section_pipe_volume, 0.0)

    def test_slower_riser_cooling_gives_shallower_pipe(self) -> None:
        # Same column, riser frozen later (sleeve): liquid stays higher longer.
        cold = [10, 12, 14, 30, 40, 50, 60, 70, 80, 5]
        hot = [10, 12, 14, 300, 400, 500, 600, 700, 800, 5]
        f_cold, _ = gravity_drain_pipe(*_stack(cold), 0.2)
        f_hot, _ = gravity_drain_pipe(*_stack(hot), 0.2)
        depth = lambda f: float(np.arange(10.0)[f > 0].min())
        self.assertGreaterEqual(depth(f_hot), depth(f_cold))

    def test_riser_still_liquid_at_the_end_feeds_itself_from_the_top(self) -> None:
        # The casting froze; the riser (z >= 3) has no times: the run stopped with it liquid.
        nan = float("nan")
        t, z, v, pools, feeder = _stack([10, 12, 14, nan, nan, nan, nan, nan, nan, nan])
        fraction, (pool,) = gravity_drain_pipe(t, z, v, pools, feeder, 0.05)
        # Every voxel's shrinkage is drained but the last one's, which has no liquid left above it.
        self.assertAlmostEqual(pool.unfed_volume, 0.05 * (1.0 - float(fraction[9])), places=6)
        self.assertAlmostEqual(pool.pipe_volume + pool.unfed_volume, pool.demand_volume, places=9)
        self.assertEqual(pool.section_pipe_volume, 0.0)
        self.assertEqual(int(np.argmax(fraction)), 9)  # the level falls from the top
        np.testing.assert_array_equal(fraction[:9], 0.0)

    def test_voxels_outside_pools_are_untouched(self) -> None:
        t, z, v, pools, feeder = _stack([10, 12, 14, 30, 40, 50, 60, 70, 80, 5])
        pools = np.full(10, -1, dtype=np.int64)
        fraction, report = gravity_drain_pipe(t, z, v, pools, feeder, 0.05)
        self.assertEqual(report, [])
        self.assertEqual(float(fraction.sum()), 0.0)


def _chain(n: int) -> np.ndarray:
    """Neighbor array of n voxels in a line (two slots: previous, next)."""
    nbr = np.full((n, 2), -1, dtype=np.int32)
    nbr[1:, 0] = np.arange(n - 1)
    nbr[:-1, 1] = np.arange(1, n)
    return nbr


class LabelUnfedPoolsTests(unittest.TestCase):
    def test_each_connected_region_is_one_pool(self) -> None:
        # Two runs of metal separated by a gap voxel: 0-2 and 4-6.
        metal = np.array([1, 1, 1, 0, 1, 1, 1], dtype=bool)
        labels, n_pools = label_unfed_pools(_chain(7), metal, np.zeros(7, dtype=bool))
        self.assertEqual(n_pools, 2)
        self.assertEqual(labels[3], -1)
        self.assertEqual(len(set(labels[:3])), 1)
        self.assertEqual(len(set(labels[4:])), 1)
        self.assertNotEqual(labels[0], labels[4])
        self.assertEqual(sorted(set(labels[metal])), [0, 1])

    def test_a_region_holding_a_feeder_voxel_is_left_out(self) -> None:
        metal = np.array([1, 1, 1, 0, 1, 1, 1], dtype=bool)
        feeder = np.zeros(7, dtype=bool)
        feeder[5] = True
        labels, n_pools = label_unfed_pools(_chain(7), metal, feeder)
        self.assertEqual(n_pools, 1)
        self.assertTrue(np.all(labels[:3] == 0))
        self.assertTrue(np.all(labels[3:] == -1))

    def test_no_metal_or_all_feeder_gives_no_pools(self) -> None:
        none = np.zeros(4, dtype=bool)
        self.assertEqual(label_unfed_pools(_chain(4), none, none)[1], 0)
        everything = np.ones(4, dtype=bool)
        labels, n_pools = label_unfed_pools(_chain(4), everything, everything)
        self.assertEqual(n_pools, 0)
        self.assertTrue(np.all(labels == -1))

    def test_unrisered_column_opens_a_pipe_under_the_top_skin(self) -> None:
        # No feeder at all: a column that freezes from the bottom up sinks at the top.
        n = 10
        times = np.array([100.0, 90, 80, 70, 60, 50, 40, 30, 20, 10])  # top freezes first
        z = np.arange(float(n))
        labels, n_pools = label_unfed_pools(_chain(n), np.ones(n, dtype=bool), np.zeros(n, dtype=bool))
        self.assertEqual(n_pools, 1)
        fraction, (pool,) = gravity_drain_pipe(times, z, np.ones(n), labels, np.zeros(n, dtype=bool), 0.05)
        self.assertGreater(pool.pipe_volume, 0.0)
        self.assertEqual(pool.feeder_pipe_volume, 0.0)
        self.assertAlmostEqual(pool.section_pipe_volume, pool.pipe_volume, places=9)
        self.assertAlmostEqual(pool.pipe_volume + pool.unfed_volume, pool.demand_volume, places=9)
        # The top skin freezes first and is never drained; the pipe opens just
        # under it, where the first demand finds the highest liquid voxel.
        self.assertEqual(float(fraction[n - 1]), 0.0)
        self.assertEqual(int(np.argmax(fraction)), n - 2)

    def test_pool_order_does_not_change_a_pools_result(self) -> None:
        # Interleave the voxels of two pools; each must drain as if alone.
        t_a = np.array([10.0, 12, 14, 30, 40, 50, 60, 70])
        t_b = np.array([5.0, 9, 20, 22, 28])
        z_a, z_b = np.arange(8.0), np.arange(5.0)
        _, (alone_a,) = gravity_drain_pipe(t_a, z_a, np.ones(8), np.zeros(8, dtype=np.int64), np.zeros(8, dtype=bool), 0.08)
        _, (alone_b,) = gravity_drain_pipe(t_b, z_b, np.ones(5), np.zeros(5, dtype=np.int64), np.zeros(5, dtype=bool), 0.08)
        times = np.empty(13)
        heights = np.empty(13)
        pools = np.empty(13, dtype=np.int64)
        times[0::2][:5], heights[0::2][:5], pools[0::2][:5] = t_b, z_b, 1
        times[1::2][:5], heights[1::2][:5], pools[1::2][:5] = t_a[:5], z_a[:5], 0
        times[10:], heights[10:], pools[10:] = t_a[5:], z_a[5:], 0
        _, both = gravity_drain_pipe(times, heights, np.ones(13), pools, np.zeros(13, dtype=bool), 0.08)
        self.assertAlmostEqual(both[0].pipe_volume, alone_a.pipe_volume, places=9)
        self.assertAlmostEqual(both[1].pipe_volume, alone_b.pipe_volume, places=9)


if __name__ == "__main__":
    unittest.main()
