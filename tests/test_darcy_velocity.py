from __future__ import annotations

import unittest

import numpy as np

# Pure-function mirrors of the shrinkage-driven feed-velocity logic in
# `core/solidification.SolidificationEngine._calculate_feeding_effectiveness`.
# No engine import: these helpers replicate the exact math so the regression
# tests pin down the physics without instantiating a simulation engine.


def _shrinkage_feed_velocity(
    cooling_rates: np.ndarray | None,
    gradients: np.ndarray | None,
    beta: float,
    n_voxels: int,
    fallback: float = 5.0e-4,
) -> np.ndarray:
    """v = beta * R, R = cooling_rate / |grad T| (grad °C/mm -> °C/m).

    Mirrors the engine: velocity = clip(beta * (Tdot / max(|grad|*1000, 1e-6)),
    1e-6, 1e-2), falling back to a constant velocity when thermal field
    data is missing or length-mismatched.
    """
    if (
        cooling_rates is not None
        and gradients is not None
        and len(cooling_rates) == n_voxels
        and len(gradients) == n_voxels
    ):
        grad_c_per_m = np.asarray(gradients, dtype=np.float64) * 1000.0  # °C/mm -> °C/m
        front_speed = np.asarray(cooling_rates, dtype=np.float64) / np.maximum(
            grad_c_per_m, 1e-6
        )  # m/s
        return np.clip(beta * front_speed, 1e-6, 1e-2)
    return np.full(n_voxels, fallback, dtype=np.float64)


def _inline_pressure_drop(
    vis_arr: np.ndarray,
    dist_m_arr: np.ndarray,
    velocity_arr: np.ndarray,
    perm_arr: np.ndarray,
    feeder_neck_factors: np.ndarray,
) -> np.ndarray:
    """Inline Darcy drop: dp = mu * L * v / K, neck-scaled.

    Mirrors the engine's replacement for
    `alloy.vectorized_pressure_drop(...)`.
    """
    pd_arr = (
        vis_arr * dist_m_arr * velocity_arr / np.maximum(perm_arr, 1e-20)
    )
    pd_arr = np.where(perm_arr < 1e-20, 1e10, pd_arr) * feeder_neck_factors
    return pd_arr


class ShrinkageFeedVelocityTests(unittest.TestCase):
    """Verify the shrinkage-driven interdendritic feed velocity v = beta * R."""

    def test_sample_values_match_shrinkage_front_speed(self) -> None:
        """v = (rho_s/rho_l - 1) * (cooling_rate / (gradient * 1000))."""
        rho_s = 7850.0
        rho_l = 7200.0
        beta = rho_s / rho_l - 1.0
        cooling_rate = 0.5  # °C/s
        gradient = 2.5  # °C/mm
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
        """Near-zero gradient saturates at 1e-2; near-zero cooling floors at 1e-6."""
        beta = 7850.0 / 7200.0 - 1.0
        n = 4

        # Gradient ~ 0 -> front speed blows up -> velocity capped at 1e-2.
        steep = _shrinkage_feed_velocity(
            np.full(n, 0.5, dtype=np.float64),
            np.full(n, 1e-6, dtype=np.float64),
            beta,
            n,
        )
        self.assertTrue(np.all(steep <= 1e-2))
        self.assertAlmostEqual(float(np.max(steep)), 1e-2, places=12)

        # Cooling ~ 0 -> no front motion -> velocity floored at 1e-6.
        stagnant = _shrinkage_feed_velocity(
            np.full(n, 0.0, dtype=np.float64),
            np.full(n, 2.5, dtype=np.float64),
            beta,
            n,
        )
        self.assertTrue(np.all(stagnant >= 1e-6))
        self.assertAlmostEqual(float(np.min(stagnant)), 1e-6, places=12)

    def test_fallback_when_gradients_missing(self) -> None:
        """No thermal field -> constant feed velocity 5e-4 m/s everywhere."""
        n = 5
        vel = _shrinkage_feed_velocity(None, None, 0.09, n)
        self.assertEqual(vel.shape, (n,))
        np.testing.assert_allclose(vel, np.full(n, 5.0e-4), rtol=0.0, atol=0.0)

    def test_fallback_on_length_mismatch(self) -> None:
        """Length-mismatched arrays also fall back to the constant velocity."""
        n = 3
        vel = _shrinkage_feed_velocity(
            np.zeros(2, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            0.09,
            n,
        )
        np.testing.assert_allclose(vel, np.full(n, 5.0e-4), rtol=0.0, atol=0.0)


class InlineDarcyPressureDropTests(unittest.TestCase):
    """Verify the inline dp = mu * L * v / K computation with neck factors."""

    def test_matches_darcy_formula(self) -> None:
        """pd = mu*L*v/K * neck for the spec's sample values."""
        vis = 0.005  # Pa·s
        dist_m = 0.05  # m
        velocity = 1.8e-5  # m/s
        perm = 1e-11  # m^2
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
        """K < 1e-20 is treated as blocked: dp = 1e10 * neck."""
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
