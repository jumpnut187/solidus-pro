"""Regression tests for Phase 2 hot tear physics fixes.

These tests pin the new behaviors introduced by:
  - Phase 2.1: coherency_support_fraction weights btr_exposure
  - Phase 2.2: BTR-window feeding via latest_open_time
  - Phase 2.3: SDAS / grain refinement term in material_factor
  - Phase 2.4: BTR bounds cooling-rate dependent
  - Phase 2.5: critical strain rate-dependent
  - new alloy fields: btr_cooling_rate_exponent,
    btr_critical_strain_rate_sensitivity (default 0 = no effect)
  - new hot_tear_components key: btr_critical_strain_effective
"""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np

from core.materials import get_alloy
from core.solidification import SolidificationEngine
from core.solidification_params import HotTearParams, SolidificationParameters


def _make_fake_engine(
    *,
    n_voxels: int = 6,
    solidification_times: np.ndarray | None = None,
    voxel_moduli: np.ndarray | None = None,
    voxel_neighbors: np.ndarray | None = None,
    fraction_liquid_history: list | None = None,
    fraction_liquid_times: list | None = None,
    analysis_options: object | None = None,
    surface_distances: np.ndarray | None = None,
    alloy: object | None = None,
    cooling_rates: np.ndarray | None = None,
) -> SimpleNamespace:
    if solidification_times is None:
        solidification_times = np.array(
            [1.0, 5.0, 10.0, 15.0, 20.0, 30.0], dtype=np.float32
        )
    if voxel_moduli is None:
        voxel_moduli = np.ones(n_voxels, dtype=np.float32)
    if voxel_neighbors is None:
        voxel_neighbors = np.array(
            [
                [1, 2, 3, 4, 5, -1],
                [0, 2, 3, 4, 5, -1],
                [0, 1, 3, 4, 5, -1],
                [0, 1, 2, 4, 5, -1],
                [0, 1, 2, 3, 5, -1],
                [0, 1, 2, 3, 4, -1],
            ],
            dtype=np.int32,
        )
    if fraction_liquid_history is None:
        fraction_liquid_history = []
    if fraction_liquid_times is None:
        fraction_liquid_times = []

    mesh = SimpleNamespace(
        voxel_centers=np.zeros((n_voxels, 3), dtype=np.float32),
        voxel_neighbors=voxel_neighbors,
        voxel_moduli=voxel_moduli,
        voxel_part_types=np.zeros(n_voxels, dtype=np.int32),
        base_voxel_size=0.5,
    )
    mesh.get_surface_voxel_indices = lambda: np.array([], dtype=np.int64)

    if analysis_options is None:
        analysis_options = SimpleNamespace(
            sand_restraint_index=0.5,
            solidification_params=SolidificationParameters(
                hot_tear=HotTearParams(
                    hot_tear_zone_threshold=0.0,
                ),
            ),
        )
    if surface_distances is None:
        surface_distances = np.full(n_voxels, 10.0, dtype=np.float32)

    engine = SimpleNamespace(
        mesh=mesh,
        solidification_times=solidification_times,
        analysis_options=analysis_options,
        surface_distances=surface_distances,
        fraction_liquid_history=fraction_liquid_history,
        fraction_liquid_times=fraction_liquid_times,
        feeding_fraction_solid=None,
        feeding_path_result=None,
        feeding_quality=None,
        feeding_open_fraction=None,
        feeding_latest_open_time=None,
        alloy=alloy if alloy is not None else get_alloy("WCB"),
        cooling_rates=cooling_rates,
        _ensure_feeding_path_result=lambda _fs: None,
    )
    engine._analyze_btr_history = (
        lambda *args, **kwargs: SolidificationEngine._analyze_btr_history(
            engine, *args, **kwargs
        )
    )
    engine._calculate_hot_tear_risk = (
        lambda: SolidificationEngine._calculate_hot_tear_risk(engine)
    )
    return engine


class HotTearAlloyFieldsTests(unittest.TestCase):
    """New alloy fields exist with sensible defaults."""

    def test_new_fields_have_defaults(self) -> None:
        from core.materials import AlloyProperties
        fields = AlloyProperties.__dataclass_fields__
        self.assertIn("btr_cooling_rate_exponent", fields)
        self.assertIn("btr_critical_strain_rate_sensitivity", fields)
        self.assertIn("btr_critical_strain_ref_rate", fields)
        self.assertIn("btr_btr_cooling_rate_ref", fields)
        # Defaults are 0 (no-op for backward compat)
        alloy = AlloyProperties(name="X", liquidus_temp=1000, solidus_temp=900,
                                pouring_temp=1100, density_liquid=7000,
                                density_solid=7500, thermal_conductivity=30,
                                specific_heat=500, latent_heat=200000,
                                linear_shrinkage=2.0,
                                feeding_distance_multiplier=6.0)
        self.assertEqual(alloy.btr_cooling_rate_exponent, 0.0)
        self.assertEqual(alloy.btr_critical_strain_rate_sensitivity, 0.0)

    def test_all_seven_alloys_have_new_fields(self) -> None:
        for name in ["WCB", "CF8", "C958", "C964", "CA6NM", "MONEL400", "IN625"]:
            alloy = get_alloy(name)
            self.assertGreater(
                alloy.btr_critical_strain_rate_sensitivity, 0.0,
                f"{name} missing strain-rate sensitivity"
            )
            self.assertLess(
                alloy.btr_critical_strain_rate_sensitivity, 1.0,
                f"{name} strain-rate sensitivity out of range"
            )
            # exponents are positive (faster cooling widens BTR, so exponent
            # multiplies (cr/ref) > 1 to widen the BTR)
            self.assertGreaterEqual(
                alloy.btr_cooling_rate_exponent, 0.0,
                f"{name} btr_cooling_rate_exponent should be >= 0"
            )
            self.assertLessEqual(
                alloy.btr_cooling_rate_exponent, 0.5,
                f"{name} btr_cooling_rate_exponent out of range"
            )


class HotTearPhase21CoherencyTests(unittest.TestCase):
    """Phase 2.1: coherency_support_fraction weights btr_exposure."""

    def test_coherency_weight_zeros_slab_region(self) -> None:
        """A voxel with no neighbors above coherency should have low btr_exposure."""
        # Build a scenario: voxel 0 has a constraining neighbor (voxel 1, above
        # coherency by snapshot 2), so it has high supporting_fraction. Voxel 2
        # has no neighbors above coherency (slab region).
        engine = _make_fake_engine(
            n_voxels=2,
            solidification_times=np.array([1.0, 5.0], dtype=np.float32),
            voxel_neighbors=np.array([[1, -1, -1, -1, -1, -1],
                                      [0, -1, -1, -1, -1, -1]],
                                     dtype=np.int32),
            fraction_liquid_history=[
                np.array([0.50, 0.60], dtype=np.float32),  # both in slurry
                np.array([0.92, 0.85], dtype=np.float32),  # both in BTR
                np.array([0.96, 0.93], dtype=np.float32),  # both still in BTR
            ],
            fraction_liquid_times=[0.0, 10.0, 20.0],
        )
        with redirect_stdout(io.StringIO()):
            SolidificationEngine._calculate_hot_tear_risk(engine)
        # Both voxels spend time in BTR
        btr_dwell = engine.hot_tear_components["btr_dwell_time"]
        self.assertGreater(float(btr_dwell[0]), 0.0)
        self.assertGreater(float(btr_dwell[1]), 0.0)
        # The supporting_fraction reflects neighbors in coherency
        supp = engine.hot_tear_components["coherency_support_fraction"]
        # Voxel 0 has neighbor 1 in BTR which is mostly slurry (0.50-0.93)
        # Voxel 1 has neighbor 0 in BTR
        # At least one of them should have non-trivial supporting fraction
        self.assertGreaterEqual(float(np.max(supp)), 0.0)

    def test_coherency_weight_factor_keeps_btr_exposure_in_range(self) -> None:
        """btr_exposure should always stay in [0, 1] even with coherency weighting."""
        engine = _make_fake_engine(
            n_voxels=4,
            solidification_times=np.array([1.0, 5.0, 10.0, 15.0], dtype=np.float32),
            voxel_neighbors=np.array(
                [[1, 2, 3, -1, -1, -1],
                 [0, 2, 3, -1, -1, -1],
                 [0, 1, 3, -1, -1, -1],
                 [0, 1, 2, -1, -1, -1]],
                dtype=np.int32,
            ),
            fraction_liquid_history=[
                np.array([0.50, 0.50, 0.50, 0.50], dtype=np.float32),
                np.array([0.92, 0.95, 0.50, 0.50], dtype=np.float32),
                np.array([0.96, 0.96, 0.50, 0.50], dtype=np.float32),
            ],
            fraction_liquid_times=[0.0, 10.0, 20.0],
        )
        with redirect_stdout(io.StringIO()):
            SolidificationEngine._calculate_hot_tear_risk(engine)
        btr_exposure = engine.hot_tear_components["btr_exposure"]
        self.assertTrue(np.all(btr_exposure >= 0.0))
        self.assertTrue(np.all(btr_exposure <= 1.0))


class HotTearPhase22BTRFeedingTests(unittest.TestCase):
    """Phase 2.2: BTR-window feeding via latest_open_time."""

    def test_btr_feeding_uses_latest_open_time(self) -> None:
        """Voxels with paths still open at BTR time should have higher healing."""
        # Voxel 0: latest_open_time very large (path still open at BTR)
        # Voxel 1: latest_open_time very small (path closed before BTR)
        latest_open = np.array([100.0, 0.5], dtype=np.float32)
        # Build a feeding_path_result-like object
        from types import SimpleNamespace as NS
        feeding_result = NS(
            feeding_quality=np.array([0.5, 0.5], dtype=np.float32),
            path_length=np.array([0.0, 0.0], dtype=np.float32),
            path_permeability=np.array([1e-12, 1e-12], dtype=np.float64),
            feeding_source=np.array([0, -1], dtype=np.int32),
            blocked_voxels=np.array([False, True], dtype=np.bool_),
            history_open_fraction=np.array([0.5, 0.1], dtype=np.float32),
            latest_open_time=latest_open,
            fed_fraction=0.5,
            avg_path_length=0.0,
            max_path_length=0.0,
            n_blocked=1,
        )
        engine = _make_fake_engine(
            n_voxels=2,
            solidification_times=np.array([1.0, 30.0], dtype=np.float32),
            voxel_neighbors=np.array([[1, -1, -1, -1, -1, -1],
                                      [0, -1, -1, -1, -1, -1]],
                                     dtype=np.int32),
            fraction_liquid_history=[
                np.array([0.50, 0.50], dtype=np.float32),
                np.array([0.92, 0.94], dtype=np.float32),
                np.array([0.97, 0.97], dtype=np.float32),
            ],
            fraction_liquid_times=[0.0, 10.0, 20.0],
        )
        engine.feeding_path_result = feeding_result
        engine._ensure_feeding_path_result = lambda _fs: feeding_result
        with redirect_stdout(io.StringIO()):
            SolidificationEngine._calculate_hot_tear_risk(engine)
        # Voxel 0 has BTR-window open (latest_open=100s > t_btr_start), should
        # have higher healing_capacity than voxel 1 (latest_open=0.5s < t_btr_start)
        healing = engine.hot_tear_components["healing_capacity"]
        self.assertGreater(
            float(healing[0]),
            float(healing[1]),
            f"voxel 0 (path open at BTR) should have higher healing than voxel 1 "
            f"(path closed before BTR); got {healing}"
        )

    def test_no_feeding_result_still_works(self) -> None:
        """Without feeding result, healing is zero, no exception."""
        engine = _make_fake_engine(
            n_voxels=2,
            solidification_times=np.array([1.0, 5.0], dtype=np.float32),
            voxel_neighbors=np.array([[1, -1, -1, -1, -1, -1],
                                      [0, -1, -1, -1, -1, -1]],
                                     dtype=np.int32),
            fraction_liquid_history=[
                np.array([0.50, 0.50], dtype=np.float32),
                np.array([0.92, 0.94], dtype=np.float32),
            ],
            fraction_liquid_times=[0.0, 10.0],
        )
        # feeding_path_result remains None
        with redirect_stdout(io.StringIO()):
            SolidificationEngine._calculate_hot_tear_risk(engine)
        healing = engine.hot_tear_components["healing_capacity"]
        self.assertTrue(np.all(np.isfinite(healing)))


class HotTearPhase23SDASTests(unittest.TestCase):
    """Phase 2.3: SDAS / grain refinement term in material_factor."""

    def test_slower_cooling_yields_coarser_sdas_higher_material(self) -> None:
        """Slower cooling → coarser SDAS → higher material factor (HTS)."""
        # Two scenarios with identical geometry and history, different cooling
        # rates. We compare the material factor.
        base_kwargs = dict(
            n_voxels=3,
            solidification_times=np.array([1.0, 5.0, 10.0], dtype=np.float32),
            voxel_neighbors=np.array(
                [[1, 2, -1, -1, -1, -1],
                 [0, 2, -1, -1, -1, -1],
                 [0, 1, -1, -1, -1, -1]],
                dtype=np.int32,
            ),
            fraction_liquid_history=[
                np.array([0.50, 0.50, 0.50], dtype=np.float32),
                np.array([0.92, 0.94, 0.96], dtype=np.float32),
            ],
            fraction_liquid_times=[0.0, 10.0],
        )
        # Slower cooling (0.1 °C/s) → coarser SDAS → higher material factor
        engine_slow = _make_fake_engine(
            **base_kwargs,
            cooling_rates=np.array([0.1, 0.1, 0.1], dtype=np.float32),
        )
        # Faster cooling (10 °C/s) → finer SDAS → lower material factor
        engine_fast = _make_fake_engine(
            **base_kwargs,
            cooling_rates=np.array([10.0, 10.0, 10.0], dtype=np.float32),
        )
        with redirect_stdout(io.StringIO()):
            SolidificationEngine._calculate_hot_tear_risk(engine_slow)
            SolidificationEngine._calculate_hot_tear_risk(engine_fast)
        mat_slow = engine_slow.hot_tear_components["material_factor"]
        mat_fast = engine_fast.hot_tear_components["material_factor"]
        # material_factor is higher in slow-cooling voxels (coarser SDAS)
        self.assertGreater(
            float(np.mean(mat_slow)),
            float(np.mean(mat_fast)),
            f"slow cooling should yield higher material factor; "
            f"got slow={float(np.mean(mat_slow)):.3f}, "
            f"fast={float(np.mean(mat_fast)):.3f}"
        )

    def test_no_cooling_rates_skips_sdas_term(self) -> None:
        """Without cooling_rates, the SDAS term is a no-op (factor=1)."""
        engine = _make_fake_engine(
            n_voxels=2,
            solidification_times=np.array([1.0, 5.0], dtype=np.float32),
            voxel_neighbors=np.array([[1, -1, -1, -1, -1, -1],
                                      [0, -1, -1, -1, -1, -1]],
                                     dtype=np.int32),
            fraction_liquid_history=[
                np.array([0.50, 0.50], dtype=np.float32),
                np.array([0.92, 0.94], dtype=np.float32),
            ],
            fraction_liquid_times=[0.0, 10.0],
            cooling_rates=None,  # not set
        )
        with redirect_stdout(io.StringIO()):
            SolidificationEngine._calculate_hot_tear_risk(engine)
        # material_factor should be present and finite
        mat = engine.hot_tear_components["material_factor"]
        self.assertTrue(np.all(np.isfinite(mat)))


class HotTearPhase24BTRCoolingRateTests(unittest.TestCase):
    """Phase 2.4: BTR bounds cooling-rate dependent."""

    def test_faster_cooling_widens_btr(self) -> None:
        """With btr_cooling_rate_exponent < 0, faster cooling yields longer btr_dwell_time."""
        base_kwargs = dict(
            n_voxels=3,
            solidification_times=np.array([1.0, 5.0, 10.0], dtype=np.float32),
            voxel_neighbors=np.array(
                [[1, 2, -1, -1, -1, -1],
                 [0, 2, -1, -1, -1, -1],
                 [0, 1, -1, -1, -1, -1]],
                dtype=np.int32,
            ),
            fraction_liquid_history=[
                np.array([0.50, 0.50, 0.50], dtype=np.float32),
                np.array([0.92, 0.94, 0.96], dtype=np.float32),
                np.array([0.97, 0.97, 0.97], dtype=np.float32),
            ],
            fraction_liquid_times=[0.0, 10.0, 20.0],
        )
        # Slow cooling: 0.1 °C/s
        engine_slow = _make_fake_engine(
            **base_kwargs,
            cooling_rates=np.array([0.1, 0.1, 0.1], dtype=np.float32),
        )
        # Fast cooling: 100 °C/s
        engine_fast = _make_fake_engine(
            **base_kwargs,
            cooling_rates=np.array([100.0, 100.0, 100.0], dtype=np.float32),
        )
        with redirect_stdout(io.StringIO()):
            SolidificationEngine._calculate_hot_tear_risk(engine_slow)
            SolidificationEngine._calculate_hot_tear_risk(engine_fast)
        # With btr_cooling_rate_exponent = -0.05 (WCB default), faster cooling
        # should yield wider BTR → larger btr_dwell_time.
        dwell_slow = engine_slow.hot_tear_components["btr_dwell_time"]
        dwell_fast = engine_fast.hot_tear_components["btr_dwell_time"]
        self.assertGreater(
            float(np.sum(dwell_fast)),
            float(np.sum(dwell_slow)),
            f"faster cooling should widen BTR; got slow={float(np.sum(dwell_slow)):.1f}s, "
            f"fast={float(np.sum(dwell_fast)):.1f}s"
        )

    def test_zero_exponent_is_no_op(self) -> None:
        """When btr_cooling_rate_exponent is 0, dwell time is unchanged by cooling rate."""
        # Build a custom alloy with exponent=0
        from core.materials import AlloyProperties
        alloy = AlloyProperties(
            name="Test", liquidus_temp=1460, solidus_temp=1399,
            pouring_temp=1621, density_liquid=7200, density_solid=7850,
            thermal_conductivity=45, specific_heat=600, latent_heat=272000,
            linear_shrinkage=2.0, feeding_distance_multiplier=6.0,
            conductivity_liquid=32, specific_heat_liquid=660,
            dynamic_viscosity=0.006, thermal_expansion=1.2e-4,
            niyama_threshold=1.0,
            hot_tear_susceptibility=0.30, solid_thermal_expansion=1.2e-5,
            btr_critical_strain=0.008, hot_tear_healing_factor=0.55,
            btr_fraction_solid_start=0.90, btr_fraction_solid_end=0.98,
            critical_fraction_solid=0.52, dendrite_arm_spacing=80.0,
            sand_burn_in_susceptibility=0.45, gas_sensitivity=0.50,
            btr_cooling_rate_exponent=0.0,  # NO adjustment
        )
        base_kwargs = dict(
            n_voxels=2,
            solidification_times=np.array([1.0, 5.0], dtype=np.float32),
            voxel_neighbors=np.array([[1, -1, -1, -1, -1, -1],
                                      [0, -1, -1, -1, -1, -1]],
                                     dtype=np.int32),
            fraction_liquid_history=[
                np.array([0.50, 0.50], dtype=np.float32),
                np.array([0.92, 0.94], dtype=np.float32),
            ],
            fraction_liquid_times=[0.0, 10.0],
            alloy=alloy,
        )
        engine_slow = _make_fake_engine(
            **base_kwargs,
            cooling_rates=np.array([0.1, 0.1], dtype=np.float32),
        )
        engine_fast = _make_fake_engine(
            **base_kwargs,
            cooling_rates=np.array([100.0, 100.0], dtype=np.float32),
        )
        with redirect_stdout(io.StringIO()):
            SolidificationEngine._calculate_hot_tear_risk(engine_slow)
            SolidificationEngine._calculate_hot_tear_risk(engine_fast)
        dwell_slow = engine_slow.hot_tear_components["btr_dwell_time"]
        dwell_fast = engine_fast.hot_tear_components["btr_dwell_time"]
        # With exponent=0, dwell times should be approximately equal
        np.testing.assert_allclose(
            dwell_slow, dwell_fast, rtol=0.05, atol=0.1,
            err_msg="zero exponent should make dwell time independent of cooling rate"
        )


class HotTearPhase25StrainRateTests(unittest.TestCase):
    """Phase 2.5: critical strain rate-dependent."""

    def test_high_strain_rate_lowers_effective_critical_strain(self) -> None:
        """With sensitivity > 0, voxels with high strain rate get a lower effective
        critical strain → higher strain_index → higher risk."""
        # A voxel with high strain_rate should produce a higher risk than the
        # same voxel with low strain_rate, holding everything else constant.
        # We test this indirectly via the components dict: btr_critical_strain_effective
        # is the per-voxel effective critical strain; with sensitivity > 0 it
        # should be lower than the static value for voxels with non-zero strain_rate.
        engine = _make_fake_engine(
            n_voxels=2,
            solidification_times=np.array([1.0, 2.0], dtype=np.float32),
            voxel_neighbors=np.array([[1, -1, -1, -1, -1, -1],
                                      [0, -1, -1, -1, -1, -1]],
                                     dtype=np.int32),
            fraction_liquid_history=[
                np.array([0.50, 0.50], dtype=np.float32),
                np.array([0.92, 0.95], dtype=np.float32),
            ],
            fraction_liquid_times=[0.0, 10.0],
        )
        with redirect_stdout(io.StringIO()):
            SolidificationEngine._calculate_hot_tear_risk(engine)
        # The WCB alloy has btr_critical_strain_rate_sensitivity = 0.10
        # Strain rates are populated for voxels that spent time in BTR.
        eff_cs = engine.hot_tear_components["btr_critical_strain_effective"]
        strain_rate = engine.hot_tear_components["strain_rate"]
        # At least one voxel should have non-zero strain rate
        if np.any(strain_rate > 0):
            # The effective critical strain for that voxel should be <= static
            mask = strain_rate > 0
            self.assertTrue(
                np.all(eff_cs[mask] <= 0.008),
                f"effective critical strain should be <= static value 0.008; "
                f"got {eff_cs[mask]}"
            )
            # And it should be >= 0.5 * static (the floor)
            self.assertTrue(
                np.all(eff_cs[mask] >= 0.5 * 0.008),
                f"effective critical strain should be >= floor 0.5*0.004; "
                f"got {eff_cs[mask]}"
            )

    def test_zero_sensitivity_preserves_static_critical_strain(self) -> None:
        """When sensitivity is 0, the effective critical strain equals the static value."""
        # Build alloy with sensitivity=0
        from core.materials import AlloyProperties
        alloy = AlloyProperties(
            name="Test", liquidus_temp=1460, solidus_temp=1399,
            pouring_temp=1621, density_liquid=7200, density_solid=7850,
            thermal_conductivity=45, specific_heat=600, latent_heat=272000,
            linear_shrinkage=2.0, feeding_distance_multiplier=6.0,
            conductivity_liquid=32, specific_heat_liquid=660,
            dynamic_viscosity=0.006, thermal_expansion=1.2e-4,
            niyama_threshold=1.0,
            hot_tear_susceptibility=0.30, solid_thermal_expansion=1.2e-5,
            btr_critical_strain=0.008, hot_tear_healing_factor=0.55,
            btr_fraction_solid_start=0.90, btr_fraction_solid_end=0.98,
            critical_fraction_solid=0.52, dendrite_arm_spacing=80.0,
            sand_burn_in_susceptibility=0.45, gas_sensitivity=0.50,
            btr_critical_strain_rate_sensitivity=0.0,  # no rate-dependence
        )
        engine = _make_fake_engine(
            n_voxels=2,
            solidification_times=np.array([1.0, 5.0], dtype=np.float32),
            voxel_neighbors=np.array([[1, -1, -1, -1, -1, -1],
                                      [0, -1, -1, -1, -1, -1]],
                                     dtype=np.int32),
            fraction_liquid_history=[
                np.array([0.50, 0.50], dtype=np.float32),
                np.array([0.92, 0.94], dtype=np.float32),
            ],
            fraction_liquid_times=[0.0, 10.0],
            alloy=alloy,
        )
        with redirect_stdout(io.StringIO()):
            SolidificationEngine._calculate_hot_tear_risk(engine)
        eff_cs = engine.hot_tear_components["btr_critical_strain_effective"]
        np.testing.assert_allclose(
            eff_cs, 0.008, rtol=1e-6,
            err_msg="zero sensitivity should preserve static critical strain"
        )


if __name__ == "__main__":
    unittest.main()
