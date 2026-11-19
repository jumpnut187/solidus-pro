"""Tests for the vectorized alloy property methods."""

from __future__ import annotations

import unittest

import numpy as np

from core.materials import ALLOY_DATABASE


class VectorizedAlloyMethodsTests(unittest.TestCase):
    """Compare vectorized implementations against the scalar originals."""

    def setUp(self) -> None:
        self.alloy = ALLOY_DATABASE["WCB"]
        self.rng = np.random.default_rng(42)
        self.n = 256

    def test_vectorized_dynamic_viscosity_matches_scalar(self) -> None:
        temps = self.rng.uniform(1300.0, 1500.0, self.n)
        fss = self.rng.uniform(0.0, 1.0, self.n)

        scalar = np.array(
            [
                self.alloy.get_dynamic_viscosity(temperature=t, fraction_solid=fs)
                for t, fs in zip(temps, fss)
            ],
            dtype=np.float64,
        )
        vectorized = self.alloy.vectorized_dynamic_viscosity(temps, fss)

        self.assertEqual(vectorized.shape, (self.n,))
        self.assertEqual(vectorized.dtype, np.float64)
        np.testing.assert_allclose(vectorized, scalar, rtol=1e-9, atol=1e-12)

    def test_vectorized_sdas_matches_scalar(self) -> None:
        rates = self.rng.uniform(0.001, 1000.0, self.n)

        scalar = np.array(
            [self.alloy.get_sdas_from_cooling_rate(cr) for cr in rates],
            dtype=np.float64,
        )
        vectorized = self.alloy.vectorized_sdas(rates)

        self.assertEqual(vectorized.shape, (self.n,))
        np.testing.assert_allclose(vectorized, scalar, rtol=1e-9, atol=1e-12)

    def test_vectorized_permeability_matches_scalar(self) -> None:
        fss = self.rng.uniform(0.0, 1.0, self.n)
        sdas_um = self.rng.uniform(10.0, 500.0, self.n)

        scalar = np.array(
            [
                self.alloy.get_permeability(float(fs), sdas_um=float(s))
                for fs, s in zip(fss, sdas_um)
            ],
            dtype=np.float64,
        )
        vectorized = self.alloy.vectorized_permeability(fss, sdas_um)

        self.assertEqual(vectorized.shape, (self.n,))
        np.testing.assert_allclose(vectorized, scalar, rtol=1e-6, atol=1e-20)

    def test_vectorized_pressure_drop_matches_scalar(self) -> None:
        fss = self.rng.uniform(0.3, 1.0, self.n)
        sdas_um = self.rng.uniform(20.0, 400.0, self.n)
        viscosities = self.rng.uniform(0.01, 5.0, self.n)
        path_lengths = self.rng.uniform(0.001, 0.2, self.n)
        feeder_neck_factors = self.rng.uniform(1.0, 1.5, self.n)

        scalar = np.array(
            [
                self.alloy.get_pressure_drop(
                    float(fs),
                    0.0005,
                    float(viscosities[i]),
                    float(path_lengths[i]),
                    sdas_um=float(sdas_um[i]),
                )
                * float(feeder_neck_factors[i])
                for i, fs in enumerate(fss)
            ],
            dtype=np.float64,
        )
        vectorized = self.alloy.vectorized_pressure_drop(
            fss,
            0.0005,
            viscosities,
            path_lengths,
            sdas_um,
            feeder_neck_factors,
        )

        self.assertEqual(vectorized.shape, (self.n,))
        # Allow a small relative tolerance — both formulas handle the
        # K -> 0 / 1e10 sentinel via slightly different code paths.
        np.testing.assert_allclose(vectorized, scalar, rtol=1e-5, atol=1.0)

    def test_vectorized_methods_handle_extreme_values(self) -> None:
        # All fully solid (fs = 1.0): permeability should clamp to a very low value
        fss = np.ones(10, dtype=np.float64)
        sdas = np.full(10, 100.0)
        perm = self.alloy.vectorized_permeability(fss, sdas)
        self.assertTrue(np.all(perm > 0.0))
        self.assertTrue(np.all(perm < 1e-9))

        # Very high cooling rates
        rates = np.full(5, 10000.0)
        sdas = self.alloy.vectorized_sdas(rates)
        self.assertTrue(np.all(sdas <= 500.0))
        self.assertTrue(np.all(sdas >= 10.0))


if __name__ == "__main__":
    unittest.main()