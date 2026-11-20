from __future__ import annotations

import unittest

import numpy as np

from physics.solidification_physics import (
    FractionSolidCalculator,
    SolidificationMethod,
    SolidificationModel,
    compute_fraction_solid_scheil_parallel,
)

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

class ScheilEquationRegressionTests(unittest.TestCase):

    T_LIQ = 1460.0
    T_SOL = 1399.0
    K = 0.17
    T_F = 1538.0

    def _scheil(self, T: float) -> float:
        return FractionSolidCalculator.scheil(
            T,
            self.T_LIQ,
            self.T_SOL,
            self.K,
            melting_point=self.T_F,
        )

    def test_wcb_analytic_values(self) -> None:
        self.assertEqual(self._scheil(1500.0), 0.0)
        self.assertAlmostEqual(self._scheil(1450.0), 0.135, delta=0.005)
        self.assertAlmostEqual(self._scheil(1430.0), 0.324, delta=0.01)
        self.assertEqual(self._scheil(1399.0), 1.0)

        temps = np.linspace(self.T_LIQ, self.T_SOL, 50)
        fractions = [self._scheil(float(t)) for t in temps]
        self.assertTrue(all(a <= b for a, b in zip(fractions, fractions[1:])))

    def test_bounds_no_clipping_hiding(self) -> None:
        rng = np.random.default_rng(7)
        temps = rng.uniform(self.T_SOL - 50.0, self.T_LIQ + 50.0, 200)
        fractions = np.array([self._scheil(float(t)) for t in temps])
        self.assertTrue(np.all(fractions >= 0.0))
        self.assertTrue(np.all(fractions <= 1.0))

    def test_eutectic_cap(self) -> None:
        fs = FractionSolidCalculator.scheil(
            400.0,
            self.T_LIQ,
            self.T_SOL,
            self.K,
            melting_point=self.T_F,
            eutectic_temperature=500.0,
            eutectic_fraction=0.4,
        )
        self.assertGreaterEqual(fs, 0.0)
        self.assertLessEqual(fs, 1.0 - 0.4)

    def test_fallback_when_no_melting_point(self) -> None:
        temps = np.linspace(self.T_LIQ, self.T_SOL, 21)
        fractions = [
            FractionSolidCalculator.scheil(float(t), self.T_LIQ, self.T_SOL, self.K)
            for t in temps
        ]
        self.assertEqual(fractions[0], 0.0)
        self.assertEqual(fractions[-1], 1.0)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in fractions))
        self.assertTrue(all(a <= b for a, b in zip(fractions, fractions[1:])))

    def test_k_one_uses_lever(self) -> None:
        for T in (1500.0, 1460.0, 1450.0, 1430.0, 1400.0, 1399.0, 1300.0):
            scheil = FractionSolidCalculator.scheil(
                T,
                self.T_LIQ,
                self.T_SOL,
                1.0,
                melting_point=self.T_F,
            )
            lever = FractionSolidCalculator.lever_rule(T, self.T_LIQ, self.T_SOL)
            self.assertAlmostEqual(scheil, lever, places=9)

    @unittest.skipUnless(NUMBA_AVAILABLE, "numba not installed")
    def test_numba_parity(self) -> None:
        rng = np.random.default_rng(11)
        temps = rng.uniform(self.T_SOL - 30.0, self.T_LIQ + 30.0, 64)
        expected = np.array(
            [
                FractionSolidCalculator.scheil(
                    float(t),
                    self.T_LIQ,
                    self.T_SOL,
                    self.K,
                    melting_point=self.T_F,
                )
                for t in temps
            ],
            dtype=np.float64,
        )
        actual = compute_fraction_solid_scheil_parallel(
            temps,
            self.T_LIQ,
            self.T_SOL,
            self.K,
            melting_point=self.T_F,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-8)

class _ScheilAlloy:

    liquidus_temp = 1460.0
    solidus_temp = 1399.0
    latent_heat = 272000.0
    density_liquid = 7200.0
    density_solid = 7850.0
    melting_point_solvent = 1538.0
    partition_coefficient = 0.17
    eutectic_temperature = 500.0
    eutectic_fraction = 0.4

class ScheilModelAlloyIntegrationTests(unittest.TestCase):

    def test_model_pulls_alloy_values(self) -> None:
        model = SolidificationModel(
            n_voxels=3,
            alloy=_ScheilAlloy(),
            method=SolidificationMethod.SCHEIL,
        )
        self.assertEqual(model.params.melting_point_solvent, 1538.0)
        self.assertEqual(model.params.partition_coefficient, 0.17)
        self.assertEqual(model.params.eutectic_temperature, 500.0)
        self.assertEqual(model.params.eutectic_fraction, 0.4)

    def test_model_scheil_matches_calculator(self) -> None:
        model = SolidificationModel(
            n_voxels=3,
            alloy=_ScheilAlloy(),
            method=SolidificationMethod.SCHEIL,
        )
        temps = np.array([1500.0, 1450.0, 1430.0, 1400.0, 1399.0, 400.0], dtype=np.float64)
        expected = np.array(
            [
                FractionSolidCalculator.scheil(
                    float(t),
                    model.T_liq,
                    model.T_sol,
                    model.params.partition_coefficient,
                    melting_point=model.params.melting_point_solvent,
                    eutectic_temperature=model.params.eutectic_temperature,
                    eutectic_fraction=model.params.eutectic_fraction,
                )
                for t in temps
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(model._scheil(temps), expected, atol=1e-12)

if __name__ == "__main__":
    unittest.main()
