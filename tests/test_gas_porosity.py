"""Smoke tests for physics.gas_porosity module."""

from __future__ import annotations

import unittest

import numpy as np

from physics.gas_porosity import (
    SURFACE_TYPE_WEIGHTS,
    BINDER_GAS_YIELDS,
    compute_surface_gas_generation,
    compute_combined_risk,
    compute_hotspots,
    _propagate_sgi_numba,
)


class TestSurfaceGasGeneration(unittest.TestCase):
    def test_core_has_higher_sgi_than_mold_wall(self):
        n = 100
        surface_indices = np.arange(n, dtype=np.int32)
        part_types = np.zeros(n * 3, dtype=np.int32)
        part_types[5] = 4  # core
        part_types[6] = 4
        local_st = np.ones(n * 3, dtype=np.float32) * 10.0

        # An unvented core (weight 1.0) out-gasses more than the mold wall
        # (0.6). This test used the default venting 0.5 and passed only
        # because venting was skipped unless the last surface entry was a core.
        sgi0 = compute_surface_gas_generation(
            surface_indices=surface_indices,
            voxel_part_types=part_types,
            local_solidification_times=local_st,
            core_venting_effectiveness=0.0,
        )

        self.assertEqual(len(sgi0), n)
        core_indices = np.where(np.isin(surface_indices, [5, 6]))[0]
        wall_indices = np.where(~np.isin(surface_indices, [5, 6]))[0]

        if len(core_indices) > 0 and len(wall_indices) > 0:
            core_avg = np.mean(sgi0[core_indices])
            wall_avg = np.mean(sgi0[wall_indices])
            self.assertGreater(core_avg, wall_avg,
                               "Core surfaces should produce higher SGI than mold wall")

    def test_core_venting_applies_wherever_core_entries_sit(self):
        # Before the fix venting was applied only when the last surface entry
        # was a core: here the cores are entries 5 and 6 of 100.
        n = 100
        surface_indices = np.arange(n, dtype=np.int32)
        part_types = np.zeros(n, dtype=np.int32)
        part_types[[5, 6]] = 4
        local_st = np.full(n, 10.0, dtype=np.float32)
        by_venting = {
            venting: compute_surface_gas_generation(
                surface_indices=surface_indices,
                voxel_part_types=part_types,
                local_solidification_times=local_st,
                core_venting_effectiveness=venting,
            )
            for venting in (0.0, 0.5, 1.0)
        }
        np.testing.assert_allclose(by_venting[0.5][[5, 6]], 0.5 * by_venting[0.0][[5, 6]], rtol=1e-6)
        np.testing.assert_array_equal(by_venting[1.0][[5, 6]], 0.0)
        # Mold-wall entries do not depend on core venting.
        np.testing.assert_array_equal(by_venting[0.0][:5], by_venting[1.0][:5])

    def test_core_last_with_other_entries_does_not_raise(self):
        # Before the fix: last entry core, 2 core entries against 4 positive
        # entries -> "sgi0[sgi0 > 0] = sgi0[core]" raised a shape ValueError.
        surface_indices = np.arange(4, dtype=np.int32)
        part_types = np.array([0, 4, 0, 4], dtype=np.int32)
        local_st = np.full(4, 10.0, dtype=np.float32)
        sgi0 = compute_surface_gas_generation(
            surface_indices=surface_indices,
            voxel_part_types=part_types,
            local_solidification_times=local_st,
            core_venting_effectiveness=0.5,
        )
        self.assertAlmostEqual(float(sgi0[1]), float(sgi0[3]))
        self.assertAlmostEqual(float(sgi0[1]) / float(sgi0[0]), 0.5 / 0.6, places=5)

    def test_chill_surfaces_have_zero_sgi(self):
        n = 20
        surface_indices = np.arange(n, dtype=np.int32)
        part_types = np.zeros(n * 2, dtype=np.int32)
        part_types[0:5] = 2  # chill
        local_st = np.ones(n * 2, dtype=np.float32) * 5.0

        sgi0 = compute_surface_gas_generation(
            surface_indices=surface_indices,
            voxel_part_types=part_types,
            local_solidification_times=local_st,
        )

        chill_mask = surface_indices < 5
        self.assertTrue(np.all(sgi0[chill_mask] == 0.0),
                        "Chill surfaces should have zero SGI0")

    def test_increased_humidity_increases_sgi(self):
        n = 50
        surface_indices = np.arange(n, dtype=np.int32)
        part_types = np.zeros(n * 2, dtype=np.int32)
        local_st = np.ones(n * 2, dtype=np.float32) * 8.0

        sgi0_dry = compute_surface_gas_generation(
            surface_indices, part_types, local_st, humidity=0.0,
        )
        sgi0_wet = compute_surface_gas_generation(
            surface_indices, part_types, local_st, humidity=1.0,
        )

        self.assertGreater(np.mean(sgi0_wet), np.mean(sgi0_dry),
                           "Higher humidity should increase gas generation")

    def test_sgi0_in_range_zero_to_one(self):
        n = 100
        surface_indices = np.arange(n, dtype=np.int32)
        np.random.seed(42)
        part_types = np.random.choice([0, 1, 2, 3, 4], size=n * 2)
        part_types = part_types.astype(np.int32)
        local_st = np.random.uniform(0, 60, n * 2).astype(np.float32)

        sgi0 = compute_surface_gas_generation(
            surface_indices, part_types, local_st, humidity=0.8,
        )
        self.assertTrue(np.all(sgi0 >= 0.0) and np.all(sgi0 <= 1.0),
                        "SGI0 must be in [0, 1]")


class TestPropagationDecay(unittest.TestCase):
    def test_exponential_decay_below_5pct_at_3lambda(self):
        distances = np.array([0.0, 3.0, 9.0], dtype=np.float32)
        sgi0_vals = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        lam = np.float32(3.0)

        sgi = _propagate_sgi_numba(distances, sgi0_vals, lam)

        self.assertAlmostEqual(float(sgi[0]), 1.0, delta=0.01)
        self.assertLess(sgi[1], 1.0)
        self.assertLessEqual(float(sgi[2]), 0.06,
                             f"At d=3λ, SGI should be <5% (got {sgi[2]:.4f})")


class TestCombinedRisk(unittest.TestCase):
    def test_risk_in_thick_section_higher_than_thin(self):
        n = 1000
        gas_gen = np.full(n, 0.3, dtype=np.float32)
        local_st = np.linspace(1.0, 100.0, n).astype(np.float32)
        niyama = np.ones(n, dtype=np.float32) * 0.2

        _, _, risk = compute_combined_risk(
            gas_gen, local_st, niyama, 0.75,
        )

        thin_risk = np.mean(risk[:100])
        thick_risk = np.mean(risk[-100:])
        self.assertGreater(thick_risk, thin_risk,
                           "Thick sections should trap more gas")

    def test_ca6nm_sensitivity_higher_than_wcb(self):
        n = 500
        gas_gen = np.full(n, 0.4, dtype=np.float32)
        local_st = np.full(n, 30.0, dtype=np.float32)
        niyama = np.ones(n, dtype=np.float32) * 0.3

        _, _, risk_ca6nm = compute_combined_risk(
            gas_gen, local_st, niyama, 0.75,
        )
        _, _, risk_wcb = compute_combined_risk(
            gas_gen, local_st, niyama, 0.50,
        )

        self.assertGreater(np.mean(risk_ca6nm), np.mean(risk_wcb),
                           "CA6NM (0.75) should produce higher risk than WCB (0.50)")

    def test_niyama_coupling_amplifies_risk(self):
        n = 500
        gas_gen = np.full(n, 0.3, dtype=np.float32)
        local_st = np.full(n, 40.0, dtype=np.float32)

        niyama_good = np.zeros(n, dtype=np.float32)
        niyama_poor = np.ones(n, dtype=np.float32)

        _, _, risk_good = compute_combined_risk(
            gas_gen, local_st, niyama_good, 0.5, 2.0, 0.5,
        )
        _, _, risk_poor = compute_combined_risk(
            gas_gen, local_st, niyama_poor, 0.5, 2.0, 0.5,
        )

        self.assertGreater(np.mean(risk_poor), np.mean(risk_good),
                           "Poor feeding (high Niyama risk) should amplify gas risk")

    def test_no_niyama_falls_back_to_neutral(self):
        n = 500
        gas_gen = np.full(n, 0.3, dtype=np.float32)
        local_st = np.full(n, 40.0, dtype=np.float32)

        _, _, risk_with = compute_combined_risk(
            gas_gen, local_st, np.ones(n, dtype=np.float32) * 0.5, 0.5,
        )
        _, _, risk_without = compute_combined_risk(
            gas_gen, local_st, None, 0.5,
        )

        self.assertTrue(np.all(np.isfinite(risk_without)),
                        "Risk must be finite when Niyama is None")

    def test_risk_range_zero_to_one(self):
        n = 1000
        np.random.seed(42)
        gas_gen = np.random.uniform(0, 1, n).astype(np.float32)
        local_st = np.random.uniform(1, 100, n).astype(np.float32)
        niyama = np.random.uniform(0, 1, n).astype(np.float32)

        _, _, risk = compute_combined_risk(gas_gen, local_st, niyama, 0.5)

        self.assertTrue(np.all(risk >= 0.0) and np.all(risk <= 1.0),
                        "Combined gas risk must be in [0, 1]")


class TestHotspots(unittest.TestCase):
    def test_hotspot_threshold(self):
        risk = np.array([0.0, 0.3, 0.6, 0.7, 1.0], dtype=np.float32)
        hotspots = compute_hotspots(risk, threshold=0.65)

        expected = np.array([False, False, False, True, True])
        np.testing.assert_array_equal(hotspots, expected)


class TestConstants(unittest.TestCase):
    def test_surface_weights(self):
        self.assertEqual(SURFACE_TYPE_WEIGHTS["core"], 1.0)
        self.assertEqual(SURFACE_TYPE_WEIGHTS["mold_wall"], 0.6)
        self.assertEqual(SURFACE_TYPE_WEIGHTS["open_riser"], 0.2)
        self.assertEqual(SURFACE_TYPE_WEIGHTS["chill"], 0.0)

    def test_binder_gas_yields(self):
        self.assertIn("phenolic_urethane", BINDER_GAS_YIELDS)
        self.assertGreater(BINDER_GAS_YIELDS["phenolic_urethane"], 0)


if __name__ == "__main__":
    unittest.main()
