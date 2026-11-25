from __future__ import annotations

import unittest

import numpy as np

def _shrinkage_feed_velocity(
    cooling_rates: np.ndarray | None,
    gradients: np.ndarray | None,
    beta: float,
    n_voxels: int,
    fallback: float = 5.0e-4,
) -> np.ndarray:
    if (
        cooling_rates is not None
        and gradients is not None
        and len(cooling_rates) == n_voxels
        and len(gradients) == n_voxels
    ):
        grad_c_per_m = np.asarray(gradients, dtype=np.float64) * 1000.0
        front_speed = np.asarray(cooling_rates, dtype=np.float64) / np.maximum(
            grad_c_per_m, 1e-6
        )
        return np.clip(beta * front_speed, 1e-6, 1e-2)
    return np.full(n_voxels, fallback, dtype=np.float64)

def _inline_pressure_drop(
    vis_arr: np.ndarray,
    dist_m_arr: np.ndarray,
    velocity_arr: np.ndarray,
    perm_arr: np.ndarray,
    feeder_neck_factors: np.ndarray,
) -> np.ndarray:
    pd_arr = (
        vis_arr * dist_m_arr * velocity_arr / np.maximum(perm_arr, 1e-20)
    )
    pd_arr = np.where(perm_arr < 1e-20, 1e10, pd_arr) * feeder_neck_factors
    return pd_arr

class ShrinkageFeedVelocityTests(unittest.TestCase):

    def test_sample_values_match_shrinkage_front_speed(self) -> None:
        rho_s = 7850.0
        rho_l = 7200.0
        beta = rho_s / rho_l - 1.0
        cooling_rate = 0.5
        gradient = 2.5
        n = 4

        vel = _shrinkage_feed_velocity(
            np.full(n, cooling_rate, dtype=np.float64),
            np.full(n, gradient, dtype=np.float64),
            beta,
            n,
        )

        expected = beta * (cooling_rate / (gradient * 1000.0))
        np.testing.assert_allclose(vel, expected, rtol=1e-12)
        self.assertAlmostEqual(expected, 1.8056e-5, places=8)
        self.assertTrue(np.all(vel >= 1e-6))
        self.assertTrue(np.all(vel <= 1e-2))

    def test_clamping(self) -> None:
        beta = 7850.0 / 7200.0 - 1.0
        n = 4

        steep = _shrinkage_feed_velocity(
            np.full(n, 0.5, dtype=np.float64),
            np.full(n, 1e-6, dtype=np.float64),
            beta,
            n,
        )
        self.assertTrue(np.all(steep <= 1e-2))
        self.assertAlmostEqual(float(np.max(steep)), 1e-2, places=12)

        stagnant = _shrinkage_feed_velocity(
            np.full(n, 0.0, dtype=np.float64),
            np.full(n, 2.5, dtype=np.float64),
            beta,
            n,
        )
        self.assertTrue(np.all(stagnant >= 1e-6))
        self.assertAlmostEqual(float(np.min(stagnant)), 1e-6, places=12)

    def test_fallback_when_gradients_missing(self) -> None:
        n = 5
        vel = _shrinkage_feed_velocity(None, None, 0.09, n)
        self.assertEqual(vel.shape, (n,))
        np.testing.assert_allclose(vel, np.full(n, 5.0e-4), rtol=0.0, atol=0.0)

    def test_fallback_on_length_mismatch(self) -> None:
        n = 3
        vel = _shrinkage_feed_velocity(
            np.zeros(2, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            0.09,
            n,
        )
        np.testing.assert_allclose(vel, np.full(n, 5.0e-4), rtol=0.0, atol=0.0)

class InlineDarcyPressureDropTests(unittest.TestCase):

    def test_matches_darcy_formula(self) -> None:
        vis = 0.005
        dist_m = 0.05
        velocity = 1.8e-5
        perm = 1e-11
        neck = 1.35

        pd = _inline_pressure_drop(
            np.array([vis]),
            np.array([dist_m]),
            np.array([velocity]),
            np.array([perm]),
            np.array([neck]),
        )[0]

        expected = vis * dist_m * velocity / perm * neck
        self.assertAlmostEqual(pd, expected, delta=abs(expected) * 1e-6)

    def test_zero_permeability_returns_1e10(self) -> None:
        pd = _inline_pressure_drop(
            np.array([0.005]),
            np.array([0.05]),
            np.array([1.8e-5]),
            np.array([0.0]),
            np.array([1.35]),
        )[0]
        self.assertAlmostEqual(pd, 1e10 * 1.35, places=2)

if __name__ == "__main__":
    unittest.main()
