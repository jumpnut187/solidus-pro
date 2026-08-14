from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from core.acceleration import (
    update_solidification_with_feeding_connectivity,
    update_solidification_with_part_types_parallel,
)
from core.feeding_path import FeedingPathAnalyzer
from core.materials import get_alloy
from core.microstructure import (
    compute_mechanical_properties,
    compute_sdas_field,
    estimate_cooling_rate_from_solidification,
)
from core.solidification import SolidificationEngine
from core.stress_analysis import compute_distortion_magnitude
from physics.solidification_physics import (
    FractionSolidCalculator,
    SolidificationMethod,
    SolidificationModel,
    SolidificationParameters,
    compute_fraction_solid_scheil_parallel,
)


class _DummyAlloy:
    liquidus_temp = 100.0
    solidus_temp = 0.0
    latent_heat = 10.0
    density_liquid = 2.0
    density_solid = 2.0


class ScheilRegressionTests(unittest.TestCase):
    def test_scalar_scheil_stays_bounded_and_monotonic(self) -> None:
        temperatures = [100.0, 75.0, 50.0, 25.0, 0.0]
        fractions = [
            FractionSolidCalculator.scheil(temp, 100.0, 0.0, 0.14)
            for temp in temperatures
        ]

        self.assertEqual(fractions[0], 0.0)
        self.assertEqual(fractions[-1], 1.0)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in fractions))
        self.assertTrue(all(a <= b for a, b in zip(fractions, fractions[1:])))
        self.assertGreater(fractions[1], 0.05)
        self.assertLess(fractions[2], 0.95)

    def test_true_scheil_matches_analytic_values(self) -> None:
        # With the solvent melting point known, the Scheil option uses the
        # true microsegregation equation, not the bounded surrogate. These are
        # the analytic values for the WCB binary approximation (k = 0.17,
        # T_f = 1538, T_L = 1460, T_S = 1399): the surrogate previously
        # returned 0.194 / 0.558 here.
        T_liq, T_sol, k, T_f = 1460.0, 1399.0, 0.17, 1538.0
        self.assertEqual(
            FractionSolidCalculator.scheil(1500.0, T_liq, T_sol, k, melting_point=T_f),
            0.0,
        )
        self.assertAlmostEqual(
            FractionSolidCalculator.scheil(1450.0, T_liq, T_sol, k, melting_point=T_f),
            0.135,
            delta=0.005,
        )
        self.assertAlmostEqual(
            FractionSolidCalculator.scheil(1430.0, T_liq, T_sol, k, melting_point=T_f),
            0.324,
            delta=0.01,
        )
        # Raw Scheil leaves residual liquid at the solidus; the lever-rule
        # floor completes the model to exactly 1.0 there.
        self.assertEqual(
            FractionSolidCalculator.scheil(1399.0, T_liq, T_sol, k, melting_point=T_f),
            1.0,
        )
        # Bounded and monotone across the freezing range.
        temps = np.linspace(T_liq, T_sol, 50)
        fractions = [
            FractionSolidCalculator.scheil(t, T_liq, T_sol, k, melting_point=T_f)
            for t in temps
        ]
        self.assertTrue(all(0.0 <= value <= 1.0 for value in fractions))
        self.assertTrue(all(a <= b for a, b in zip(fractions, fractions[1:])))

    def test_parallel_scheil_matches_scalar_reference(self) -> None:
        temperatures = np.array([100.0, 80.0, 60.0, 40.0, 20.0, 0.0], dtype=np.float64)
        expected = np.array(
            [
                FractionSolidCalculator.scheil(temp, 100.0, 0.0, 0.14)
                for temp in temperatures
            ],
            dtype=np.float64,
        )

        actual = compute_fraction_solid_scheil_parallel(temperatures, 100.0, 0.0, 0.14)
        np.testing.assert_allclose(actual, expected, atol=1e-8)

    def test_back_diffusion_interpolates_between_lever_and_scheil(self) -> None:
        alloy = _DummyAlloy()
        params = SolidificationParameters(partition_coefficient=0.14, back_diffusion_alpha=0.25)
        model = SolidificationModel(
            n_voxels=3,
            alloy=alloy,
            method=SolidificationMethod.BACK_DIFFUSION,
            params=params,
        )
        temperatures = np.array([80.0, 50.0, 20.0], dtype=np.float64)

        lever = model._lever_rule(temperatures)
        scheil = model._scheil(temperatures)
        expected = 0.25 * lever + 0.75 * scheil

        actual = model._back_diffusion(temperatures)
        np.testing.assert_allclose(actual, expected, atol=1e-8)
        lower = np.minimum(lever, scheil)
        upper = np.maximum(lever, scheil)
        self.assertTrue(np.all(actual >= lower))
        self.assertTrue(np.all(actual <= upper))


class LatentHeatRegressionTests(unittest.TestCase):
    def test_get_state_reports_latest_latent_heat_rate(self) -> None:
        model = SolidificationModel(
            n_voxels=1,
            alloy=_DummyAlloy(),
            method=SolidificationMethod.LEVER_RULE,
        )

        first_rate = model.update(np.array([75.0], dtype=np.float64), dt=2.0)
        first_state = model.get_state()
        second_rate = model.update(np.array([50.0], dtype=np.float64), dt=2.0)
        second_state = model.get_state()

        np.testing.assert_allclose(first_state.latent_heat_rate, first_rate, atol=1e-8)
        np.testing.assert_allclose(second_state.latent_heat_rate, second_rate, atol=1e-8)
        np.testing.assert_allclose(first_rate, second_rate, atol=1e-8)

    def test_local_solidification_time_handles_inf_sentinels_without_nan(self) -> None:
        model = SolidificationModel(
            n_voxels=3,
            alloy=_DummyAlloy(),
            method=SolidificationMethod.LEVER_RULE,
        )
        model._solidification_start = np.array([1.0, np.inf, 3.0], dtype=np.float64)
        model._solidification_end = np.array([5.0, np.inf, np.inf], dtype=np.float64)

        local = model.local_solidification_time

        self.assertTrue(np.isfinite(local[0]))
        self.assertEqual(float(local[0]), 4.0)
        self.assertTrue(np.isinf(local[1]))
        self.assertTrue(np.isinf(local[2]))
        self.assertFalse(bool(np.any(np.isnan(local))))


class FeedingEffectivenessRegressionTests(unittest.TestCase):
    def test_ferrostatic_pressure_uses_traced_riser_component_head(self) -> None:
        alloy = get_alloy("WCB")
        dummy = SimpleNamespace()
        dummy.alloy = alloy
        dummy.mesh = SimpleNamespace(
            voxel_centers=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0],
                    [4.0, 0.0, 10.0],
                ],
                dtype=np.float32,
            )
        )
        dummy.feeding_path_result = SimpleNamespace(
            feeding_source=np.array([1, 1, 2], dtype=np.int32)
        )
        dummy._get_riser_voxel_indices = lambda: np.array([1, 2], dtype=np.int64)
        dummy._group_risers_by_component = lambda: [
            np.array([1], dtype=np.int64),
            np.array([2], dtype=np.int64),
        ]

        pressure = SolidificationEngine._calculate_ferrostatic_pressure(dummy)

        expected_low_riser_head = alloy.density_liquid * 9.81 * (2.0 * 0.0254)
        expected_global_head = alloy.density_liquid * 9.81 * (10.0 * 0.0254)
        self.assertAlmostEqual(float(pressure[0]), expected_low_riser_head, delta=1.0)
        self.assertLess(float(pressure[0]), expected_global_head * 0.25)

    def test_history_aware_feeding_path_uses_freezeoff_cutoff(self) -> None:
        alloy = get_alloy("WCB")

        class _Mesh:
            voxel_centers = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
            voxel_neighbors = np.array(
                [
                    [1, -1],
                    [0, 2],
                    [1, -1],
                ],
                dtype=np.int32,
            )

            @staticmethod
            def get_voxel_count() -> int:
                return 3

        analyzer = FeedingPathAnalyzer(_Mesh(), alloy)
        history = [
            np.array([0.0, 0.10, 0.10], dtype=np.float32),
            np.array([0.0, 0.995, 0.40], dtype=np.float32),
        ]
        result = analyzer.analyze(
            np.array([0], dtype=np.int64),
            history[-1],
            show_progress=False,
            history=history,
            history_times=[0.0, 10.0],
            reference_times=np.array([0.0, 5.0, 10.0], dtype=np.float32),
        )

        self.assertGreater(float(result.history_open_fraction[2]), 0.0)
        self.assertTrue(bool(result.blocked_voxels[2]))
        self.assertLess(float(result.feeding_quality[2]), 0.2)

    def test_effective_feeding_distance_prefers_traced_path_lengths(self) -> None:
        dummy = SimpleNamespace()
        dummy.surface_distances = np.array([0.25, 0.50, 0.75], dtype=np.float32)
        dummy.feeding_distances = dummy.surface_distances
        dummy.feeding_path_lengths = np.array([-1.0, 1.50, -1.0], dtype=np.float32)

        effective = SolidificationEngine._get_effective_feeding_distance_field(dummy)

        np.testing.assert_allclose(effective, np.array([0.25, 1.50, 0.75], dtype=np.float32))

    def test_feeding_effectiveness_uses_transient_fraction_solid(self) -> None:
        from types import SimpleNamespace

        alloy = get_alloy("WCB")
        dummy = SimpleNamespace()
        dummy.alloy = alloy
        dummy.mesh = SimpleNamespace(
            voxel_centers=np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.5],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            voxel_states=np.array([0.99, 0.99, 0.99], dtype=np.float32),
        )
        dummy.feeding_distances = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        dummy.feeding_fraction_solid = np.array([0.0, 0.40, 0.95], dtype=np.float32)
        dummy._thermal_latest_temperatures = np.array([200.0, 200.0, 200.0], dtype=np.float32)
        dummy._calculate_ferrostatic_pressure = lambda: np.array([0.0, 30000.0, 30000.0], dtype=np.float32)
        dummy._get_connected_feeder_mask = lambda: np.zeros(3, dtype=bool)
        dummy._ensure_feeding_path_result = lambda fs: None
        dummy._estimate_feeder_neck_resistance_factors = lambda: np.ones(3, dtype=np.float32)
        dummy.cooling_rates = None
        dummy.sdas = None
        dummy.stage_timings = {}
        dummy._record_stage = lambda name, start: dummy.stage_timings.update(
            {name: dummy.stage_timings.get(name, 0.0) + 0.0}
        )

        effectiveness = SolidificationEngine._calculate_feeding_effectiveness(dummy)

        self.assertEqual(effectiveness[0], 1.0)
        self.assertGreater(effectiveness[1], effectiveness[2])
        self.assertLess(effectiveness[2], 0.3)

    def test_feeding_effectiveness_blocked_path_returns_zero(self) -> None:
        from types import SimpleNamespace

        alloy = get_alloy("WCB")
        dummy = SimpleNamespace()
        dummy.alloy = alloy
        dummy.mesh = SimpleNamespace(
            voxel_centers=np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
            voxel_states=np.array([0.5], dtype=np.float32),
        )
        dummy.feeding_distances = np.array([1.0], dtype=np.float32)
        dummy.feeding_fraction_solid = np.array([0.5], dtype=np.float32)
        dummy._thermal_latest_temperatures = np.array([200.0], dtype=np.float32)
        dummy._calculate_ferrostatic_pressure = lambda: np.array([1000.0], dtype=np.float32)
        dummy._get_connected_feeder_mask = lambda: np.array([False], dtype=bool)
        dummy._ensure_feeding_path_result = lambda fs: SimpleNamespace(
            path_length=np.array([-1.0], dtype=np.float32),
            history_open_fraction=None,
        )
        dummy._estimate_feeder_neck_resistance_factors = lambda: np.array([1.0], dtype=np.float32)
        dummy.cooling_rates = None
        dummy.sdas = None
        dummy.stage_timings = {}
        dummy._record_stage = lambda name, start: dummy.stage_timings.update(
            {name: dummy.stage_timings.get(name, 0.0) + 0.0}
        )

        effectiveness = SolidificationEngine._calculate_feeding_effectiveness(dummy)

        self.assertLess(effectiveness[0], 0.2)

    def test_feeding_effectiveness_connected_feeder_returns_high(self) -> None:
        from types import SimpleNamespace

        alloy = get_alloy("WCB")
        dummy = SimpleNamespace()
        dummy.alloy = alloy
        dummy.mesh = SimpleNamespace(
            voxel_centers=np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
            voxel_states=np.array([0.5], dtype=np.float32),
        )
        dummy.feeding_distances = np.array([1.0], dtype=np.float32)
        dummy.feeding_fraction_solid = np.array([0.5], dtype=np.float32)
        dummy._thermal_latest_temperatures = np.array([200.0], dtype=np.float32)
        dummy._calculate_ferrostatic_pressure = lambda: np.array([1000.0], dtype=np.float32)
        dummy._get_connected_feeder_mask = lambda: np.array([True], dtype=bool)
        dummy._ensure_feeding_path_result = lambda fs: None
        dummy._estimate_feeder_neck_resistance_factors = lambda: np.array([1.0], dtype=np.float32)
        dummy.cooling_rates = None
        dummy.sdas = None
        dummy.stage_timings = {}
        dummy._record_stage = lambda name, start: dummy.stage_timings.update(
            {name: dummy.stage_timings.get(name, 0.0) + 0.0}
        )

        effectiveness = SolidificationEngine._calculate_feeding_effectiveness(dummy)

        self.assertGreater(effectiveness[0], 0.8)

    def test_feeding_effectiveness_surface_voxel_skipped(self) -> None:
        from types import SimpleNamespace

        alloy = get_alloy("WCB")
        dummy = SimpleNamespace()
        dummy.alloy = alloy
        dummy.mesh = SimpleNamespace(
            voxel_centers=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            voxel_states=np.array([0.5], dtype=np.float32),
        )
        dummy.feeding_distances = np.array([0.0], dtype=np.float32)
        dummy.feeding_fraction_solid = np.array([0.5], dtype=np.float32)
        dummy._thermal_latest_temperatures = np.array([200.0], dtype=np.float32)
        dummy._calculate_ferrostatic_pressure = lambda: np.array([0.0], dtype=np.float32)
        dummy._get_connected_feeder_mask = lambda: np.array([False], dtype=bool)
        dummy._ensure_feeding_path_result = lambda fs: None
        dummy._estimate_feeder_neck_resistance_factors = lambda: np.array([1.0], dtype=np.float32)
        dummy.cooling_rates = None
        dummy.sdas = None
        dummy.stage_timings = {}
        dummy._record_stage = lambda name, start: dummy.stage_timings.update(
            {name: dummy.stage_timings.get(name, 0.0) + 0.0}
        )

        effectiveness = SolidificationEngine._calculate_feeding_effectiveness(dummy)

        self.assertEqual(effectiveness[0], 1.0)


class VerticalSolidificationRegressionTests(unittest.TestCase):
    def test_part_type_update_freezes_lower_voxels_faster(self) -> None:
        states = np.array([0.2, 0.2, 1.0], dtype=np.float32)
        neighbors = np.array(
            [
                [2, -1],
                [2, -1],
                [-1, -1],
            ],
            dtype=np.int32,
        )
        moduli = np.ones(3, dtype=np.float32)
        heights = np.array([0.0, 1.0, 0.5], dtype=np.float32)
        chill_boost = np.ones(3, dtype=np.float32)
        riser_slowdown = np.ones(3, dtype=np.float32)

        updated = update_solidification_with_part_types_parallel(
            states,
            neighbors,
            moduli,
            heights,
            chill_boost,
            riser_slowdown,
            1.0,
            np.float32(1.0),
        )

        self.assertGreater(float(updated[0] - states[0]), float(updated[1] - states[1]))

    def test_feeding_connectivity_update_freezes_lower_voxels_faster(self) -> None:
        states = np.array([0.2, 0.2, 1.0], dtype=np.float32)
        neighbors = np.array(
            [
                [2, -1],
                [2, -1],
                [-1, -1],
            ],
            dtype=np.int32,
        )
        moduli = np.ones(3, dtype=np.float32)
        heights = np.array([0.0, 1.0, 0.5], dtype=np.float32)
        chill_boost = np.ones(3, dtype=np.float32)
        riser_slowdown = np.ones(3, dtype=np.float32)
        part_types = np.zeros(3, dtype=np.int32)

        updated = update_solidification_with_feeding_connectivity(
            states,
            neighbors,
            moduli,
            heights,
            chill_boost,
            riser_slowdown,
            part_types,
            1.0,
            np.float32(1.0),
        )

        self.assertGreater(float(updated[0] - states[0]), float(updated[1] - states[1]))

    def test_riser_pipe_modifier_concentrates_shrinkage_upward(self) -> None:
        dummy = SimpleNamespace()
        dummy.mesh = SimpleNamespace(
            voxel_centers=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0],
                ],
                dtype=np.float32,
            ),
            voxel_moduli=np.array([1.0, 1.0], dtype=np.float32),
        )
        dummy.solidification_times = np.array([100.0, 100.0], dtype=np.float32)
        dummy.hotspot_intensity = np.zeros(2, dtype=np.float32)
        dummy._group_risers_by_component = lambda: [np.array([0, 1], dtype=np.int64)]
        valid_indices = np.array([0, 1], dtype=np.int64)
        normalized_heights = np.array([0.0, 1.0], dtype=np.float32)
        local_section_size = np.array([1.0, 1.0], dtype=np.float32)
        riser_mask = np.array([True, True], dtype=bool)
        riser_post_mask = np.array([False, False], dtype=bool)

        base_risk, *_rest = SolidificationEngine._compute_shrinkage_base_risk(
            dummy,
            valid_indices,
            normalized_heights,
            local_section_size,
            riser_mask,
            riser_post_mask,
        )
        modifier = SolidificationEngine._compute_riser_pipe_modifier(
            dummy,
            valid_indices,
            riser_mask[valid_indices],
            {0: (0.0, 0.0, 1.0, 0.0, 2.0), 1: (0.0, 0.0, 1.0, 0.0, 2.0)},
        )

        final_risk = base_risk * modifier
        self.assertGreater(float(final_risk[1]), float(final_risk[0]))


class PorosityRegressionTests(unittest.TestCase):
    def test_shrinkage_evidence_damps_well_fed_cast_and_flags_feed_loss(self) -> None:
        dummy = SimpleNamespace(
            mesh=SimpleNamespace(
                voxel_centers=np.zeros((2, 3), dtype=np.float32),
            ),
            shrinkage_probability=np.array([0.6, 0.6], dtype=np.float32),
            solidification_times=np.array([10.0, 100.0], dtype=np.float32),
            niyama_severity=np.array([0.0, 0.85], dtype=np.float32),
            feeding_quality=np.array([1.0, 0.05], dtype=np.float32),
            feeding_open_fraction=np.array([1.0, 0.05], dtype=np.float32),
            feeding_blocked=np.array([False, True], dtype=bool),
            feeding_darcy_margin=np.array([1.0, -0.8], dtype=np.float32),
        )
        dummy._get_connected_feeder_mask = lambda: np.array([True, True], dtype=bool)

        evidence, feed_loss, healthy = SolidificationEngine._calculate_shrinkage_evidence(
            dummy,
            np.array([1.0, 0.1], dtype=np.float32),
        )

        self.assertLess(float(evidence[0]), 0.28)
        self.assertGreater(float(evidence[1]), 0.45)
        self.assertLess(float(feed_loss[0]), 0.1)
        self.assertGreater(float(feed_loss[1]), 0.9)
        self.assertTrue(bool(healthy[0]))
        self.assertFalse(bool(healthy[1]))

    def test_porosity_probability_is_clamped_to_one(self) -> None:
        dummy = SimpleNamespace(
            shrinkage_probability=np.array([0.8, 0.25], dtype=np.float32),
            porosity_probability=None,
        )

        SolidificationEngine._calculate_porosity(dummy)

        self.assertIsNotNone(dummy.porosity_probability)
        self.assertTrue(np.all(dummy.porosity_probability >= 0.0))
        self.assertTrue(np.all(dummy.porosity_probability <= 1.0))
        self.assertGreater(float(dummy.porosity_probability[0]), 0.0)

    def test_porosity_increases_with_niyama_severity(self) -> None:
        dummy = SimpleNamespace(
            shrinkage_probability=np.array([0.4, 0.4], dtype=np.float32),
            solidification_times=np.array([100.0, 100.0], dtype=np.float32),
            niyama_severity=np.array([0.0, 1.0], dtype=np.float32),
            mesh=SimpleNamespace(voxel_part_types=np.array([0, 0], dtype=np.int32)),
            porosity_probability=None,
        )

        SolidificationEngine._calculate_porosity(dummy)

        self.assertLess(
            float(dummy.porosity_probability[0]),
            float(dummy.porosity_probability[1]),
        )

    def test_blocked_feeding_path_increases_porosity(self) -> None:
        dummy = SimpleNamespace(
            shrinkage_probability=np.array([0.4, 0.4], dtype=np.float32),
            niyama_severity=np.array([0.2, 0.2], dtype=np.float32),
            feeding_quality=np.array([1.0, 0.0], dtype=np.float32),
            feeding_blocked=np.array([False, True], dtype=bool),
            mesh=SimpleNamespace(voxel_part_types=np.array([0, 0], dtype=np.int32)),
            porosity_probability=None,
        )

        SolidificationEngine._calculate_porosity(dummy)

        self.assertLess(
            float(dummy.porosity_probability[0]),
            float(dummy.porosity_probability[1]),
        )

    def test_poor_darcy_margin_increases_porosity(self) -> None:
        dummy = SimpleNamespace(
            shrinkage_probability=np.array([0.4, 0.4], dtype=np.float32),
            niyama_severity=np.array([0.3, 0.3], dtype=np.float32),
            feeding_quality=np.array([0.8, 0.8], dtype=np.float32),
            feeding_open_fraction=np.array([0.8, 0.8], dtype=np.float32),
            feeding_darcy_margin=np.array([0.8, -0.8], dtype=np.float32),
            mesh=SimpleNamespace(voxel_part_types=np.array([0, 0], dtype=np.int32)),
            porosity_probability=None,
        )

        SolidificationEngine._calculate_porosity(dummy)

        self.assertLess(
            float(dummy.porosity_probability[0]),
            float(dummy.porosity_probability[1]),
        )

    def test_connected_riser_post_relief_reduces_porosity(self) -> None:
        dummy = SimpleNamespace(
            shrinkage_probability=np.array([0.4, 0.4], dtype=np.float32),
            solidification_times=np.array([100.0, 100.0], dtype=np.float32),
            niyama_severity=np.array([1.0, 1.0], dtype=np.float32),
            mesh=SimpleNamespace(voxel_part_types=np.array([3, 3], dtype=np.int32)),
            porosity_probability=None,
        )
        dummy._get_connected_feeder_mask = lambda: np.array([False, True], dtype=bool)

        SolidificationEngine._calculate_porosity(dummy)

        self.assertGreater(
            float(dummy.porosity_probability[0]),
            float(dummy.porosity_probability[1]),
        )

    def test_blocked_connected_riser_post_does_not_get_full_relief(self) -> None:
        dummy = SimpleNamespace(
            shrinkage_probability=np.array([0.4, 0.4], dtype=np.float32),
            solidification_times=np.array([100.0, 100.0], dtype=np.float32),
            niyama_severity=np.array([0.4, 0.4], dtype=np.float32),
            feeding_quality=np.array([1.0, 0.05], dtype=np.float32),
            feeding_open_fraction=np.array([1.0, 0.05], dtype=np.float32),
            feeding_blocked=np.array([False, True], dtype=bool),
            feeding_darcy_margin=np.array([1.0, -0.8], dtype=np.float32),
            mesh=SimpleNamespace(voxel_part_types=np.array([3, 3], dtype=np.int32)),
            porosity_probability=None,
        )
        dummy._get_connected_feeder_mask = lambda: np.array([True, True], dtype=bool)

        SolidificationEngine._calculate_porosity(dummy)

        self.assertGreater(
            float(dummy.porosity_probability[1]),
            float(dummy.porosity_probability[0]),
        )


class DistortionRegressionTests(unittest.TestCase):
    def test_distortion_scales_linearly_with_weighted_vector(self) -> None:
        distortion_vectors = np.array(
            [
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        constraint_strain = np.array([0.1, 0.2], dtype=np.float64)

        magnitude = compute_distortion_magnitude(distortion_vectors, constraint_strain, voxel_size=2.0)

        np.testing.assert_allclose(magnitude, np.array([0.2, 0.4], dtype=np.float64), atol=1e-8)
        self.assertAlmostEqual(float(magnitude[1] / magnitude[0]), 2.0)


class MaterialPermeabilityEvolutionTests(unittest.TestCase):
    def test_dynamic_viscosity_increases_as_temperature_drops(self) -> None:
        alloy = get_alloy("WCB")
        hot_mu = alloy.get_dynamic_viscosity(temperature=alloy.pouring_temp, fraction_solid=0.1)
        mush_mu = alloy.get_dynamic_viscosity(temperature=alloy.solidus_temp, fraction_solid=0.7)

        self.assertGreater(mush_mu, hot_mu)
        self.assertGreater(mush_mu, alloy.dynamic_viscosity)

    def test_higher_cooling_rate_refines_sdas_and_reduces_permeability(self) -> None:
        alloy = get_alloy("WCB")
        fs = 0.45
        slow_rate = 0.2
        fast_rate = 10.0

        slow_sdas = alloy.get_sdas_from_cooling_rate(slow_rate)
        fast_sdas = alloy.get_sdas_from_cooling_rate(fast_rate)
        slow_perm = alloy.get_permeability(fs, cooling_rate=slow_rate)
        fast_perm = alloy.get_permeability(fs, cooling_rate=fast_rate)

        self.assertGreater(slow_sdas, fast_sdas)
        self.assertGreater(slow_perm, fast_perm)

    def test_feeder_neck_resistance_detects_cast_to_riser_post_interfaces(self) -> None:
        dummy = SimpleNamespace()
        dummy.mesh = SimpleNamespace(
            voxel_centers=np.array(
                [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            voxel_part_types=np.array([0, 3, 1], dtype=np.int32),
            voxel_neighbors=np.array(
                [[1, -1], [0, 2], [1, -1]],
                dtype=np.int32,
            ),
        )

        factors = SolidificationEngine._estimate_feeder_neck_resistance_factors(dummy)
        np.testing.assert_allclose(factors, np.array([1.35, 1.20, 1.0], dtype=np.float32))


class MicrostructureRegressionTests(unittest.TestCase):
    def test_microstructure_correlations_behave_monotonically(self) -> None:
        cooling_rates = np.array([0.1, 1.0, 10.0], dtype=np.float64)
        sdas = compute_sdas_field(cooling_rates, A=150.0, n=0.40)
        uts, elongation = compute_mechanical_properties(
            sdas,
            sigma_0=350.0,
            k_uts=250.0,
            epsilon_max=25.0,
            c_elong=0.003,
        )

        self.assertGreater(float(sdas[0]), float(sdas[1]))
        self.assertGreater(float(sdas[1]), float(sdas[2]))
        self.assertLess(float(uts[0]), float(uts[1]))
        self.assertLess(float(uts[1]), float(uts[2]))
        self.assertLess(float(elongation[0]), float(elongation[1]))
        self.assertLess(float(elongation[1]), float(elongation[2]))
        self.assertTrue(np.all(elongation >= 0.5))
        self.assertTrue(np.all(elongation <= 25.0))

    def test_cooling_rate_estimate_refinement_is_bounded(self) -> None:
        solidification_times = np.array([10.0, 20.0, 40.0], dtype=np.float64)
        neighbors = np.array(
            [
                [1, -1],
                [0, 2],
                [1, -1],
            ],
            dtype=np.int32,
        )

        cooling_rates = estimate_cooling_rate_from_solidification(
            solidification_times,
            neighbors,
            liquidus_temp=1460.0,
            solidus_temp=1399.0,
            voxel_size=0.5,
        )

        base_rates = 61.0 / solidification_times
        self.assertTrue(np.all(cooling_rates > 0.0))
        self.assertTrue(np.all(cooling_rates <= base_rates * 1.5 + 1e-8))
        self.assertGreater(float(cooling_rates[1]), float(base_rates[1]))


class FeedingPathCacheRegressionTests(unittest.TestCase):
    def test_ensure_feeding_path_result_called_once_per_run(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        alloy = get_alloy("WCB")
        dummy = SimpleNamespace()
        dummy.alloy = alloy
        dummy.mesh = SimpleNamespace(
            voxel_centers=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            voxel_neighbors=np.array([[-1, -1, -1, -1, -1, -1]], dtype=np.int32),
        )
        dummy.feeding_distances = np.array([0.0], dtype=np.float32)
        dummy.feeding_fraction_solid = np.array([0.5], dtype=np.float32)
        dummy._thermal_latest_temperatures = None
        dummy._get_connected_feeder_mask = lambda: np.zeros(1, dtype=bool)
        dummy._estimate_feeder_neck_resistance_factors = lambda: np.ones(1, dtype=np.float32)
        dummy.cooling_rates = None
        dummy.sdas = None
        dummy.stage_timings = {}
        dummy.fraction_liquid_history = None
        dummy.fraction_liquid_times = None
        dummy._estimate_feeding_reference_times = lambda: None
        dummy._get_riser_voxel_indices = lambda: np.array([0], dtype=np.int64)
        dummy._record_stage = lambda name, start: dummy.stage_timings.update(
            {name: dummy.stage_timings.get(name, 0.0) + 0.0}
        )

        analyzer = FeedingPathAnalyzer(dummy.mesh, alloy)
        path_result = SimpleNamespace(
            path_length=np.array([0.0], dtype=np.float32),
            feeding_quality=np.array([1.0], dtype=np.float32),
            blocked_voxels=np.array([False], dtype=bool),
            history_open_fraction=np.array([1.0], dtype=np.float32),
            latest_open_time=100.0,
        )

        call_count = 0

        def mock_analyze(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return path_result

        with patch.object(FeedingPathAnalyzer, "analyze", mock_analyze):
            SolidificationEngine._ensure_feeding_path_result(dummy, dummy.feeding_fraction_solid)
            SolidificationEngine._ensure_feeding_path_result(dummy, dummy.feeding_fraction_solid)

        self.assertEqual(call_count, 1)


if __name__ == "__main__":
    unittest.main()
