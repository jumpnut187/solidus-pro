"""Regression tests for the gravity-drained riser pipe (physics/riser_pipe.py)."""

from __future__ import annotations

import unittest

import numpy as np

from physics.riser_pipe import gravity_drain_pipe


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

    def test_voxels_outside_pools_are_untouched(self) -> None:
        t, z, v, pools, feeder = _stack([10, 12, 14, 30, 40, 50, 60, 70, 80, 5])
        pools = np.full(10, -1, dtype=np.int64)
        fraction, report = gravity_drain_pipe(t, z, v, pools, feeder, 0.05)
        self.assertEqual(report, [])
        self.assertEqual(float(fraction.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
