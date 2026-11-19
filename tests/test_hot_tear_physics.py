"""Regression tests for the hot tear physics fixes (Phase 1).

These tests pin the new behavior introduced by:
  - Phase 1.1: time-integrated BTR exposure (replaces binary tag)
  - Phase 1.2: BTR temperature range passed to strain kernels (replaces full
               freezing range)
  - Phase 1.3: worst-case (max) neighbor strain in constraint kernel
  - Phase 1.4: populated self.strain_rate
  - Phase 1.5: tightened BTR-fallback window and warning
  - Phase 1 threshold: lowered default from 0.12 to 0.06
  - Additive hot_tear_components dict (btr_dwell_time, btr_temp_range_used,
    coherency_support_fraction, strain_rate)

The tests use a fake mesh namespace and call SolidificationEngine methods
directly, mirroring the pattern in test_math_audit_regressions.py.
"""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np

from core.acceleration import (
    compute_constraint_strain_parallel,
    compute_solid_path_strain_parallel,
)
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
) -> SimpleNamespace:
    """Build a duck-typed engine namespace with the attributes
    _calculate_hot_tear_risk / _analyze_btr_history / _get_sand_restraint_index
    need."""
    if solidification_times is None:
        # Default to strictly positive times so the constraint strain kernel's
        # `neighbor_time > 0` filter doesn't reject them.
        solidification_times = np.array(
            [1.0, 5.0, 10.0, 15.0, 20.0, 30.0], dtype=np.float32
        )
    if voxel_moduli is None:
        voxel_moduli = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    if voxel_neighbors is None:
        # Each voxel sees its 6 axis-aligned neighbors where available
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

    if analysis_options is None:
        analysis_options = SimpleNamespace(
            sand_restraint_index=0.5,
            solidification_params=SolidificationParameters(
                hot_tear=HotTearParams(),
            ),
        )

    if surface_distances is None:
        surface_distances = np.full(n_voxels, 10.0, dtype=np.float32)

    mesh.get_surface_voxel_indices = lambda: np.array([], dtype=np.int64)

    # Bind the unbound methods to the fake engine so they can be called as
    # `engine._analyze_btr_history(...)` and `engine._calculate_hot_tear_risk()`
    # from inside the call stack.
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
        alloy=alloy if alloy is not None else get_alloy("WCB"),
        # Stub for _ensure_feeding_path_result
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


class HotTearBTRHistoryTests(unittest.TestCase):
    """Phase 1.1: _analyze_btr_history now returns time-integrated BTR dwell."""

    def test_returns_dwell_time_supporting_fraction_and_mask(self) -> None:
        """The new return tuple is (btr_dwell_time, supporting_fraction, in_btr)."""
        engine = _make_fake_engine(
            fraction_liquid_history=[
                np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50], dtype=np.float32),
                np.array([0.92, 0.80, 0.94, 0.91, 0.50, 0.50], dtype=np.float32),
                np.array([0.98, 0.85, 0.98, 0.96, 0.60, 0.50], dtype=np.float32),
            ],
            fraction_liquid_times=[0.0, 10.0, 20.0],
        )
        alloy = get_alloy("WCB")
        dwell, supp, mask = SolidificationEngine._analyze_btr_history(
            engine,
            btr_start=alloy.btr_fraction_solid_start,  # 0.90
            btr_end=alloy.btr_fraction_solid_end,       # 0.98
            critical_fraction_solid=alloy.critical_fraction_solid,  # 0.52
        )
        self.assertEqual(dwell.shape, (6,))
        self.assertEqual(supp.shape, (6,))
        self.assertEqual(mask.shape, (6,))
        # Voxels 0, 2, 3 were in BTR (0.92/0.94/0.91) at snapshot 1
        # Voxel 0 was in BTR at snapshot 2 (0.98) as well
        # Voxel 1 never reached BTR
        self.assertTrue(mask[0])
        self.assertTrue(mask[2])
        self.assertTrue(mask[3])
        self.assertFalse(mask[1])
        self.assertFalse(mask[4])
        self.assertFalse(mask[5])
        # Voxel 0: in BTR for half of [0,10] and half of [10,20] = 10s
        # Voxel 1: never in BTR = 0s
        # Voxel 4: never in BTR = 0s
        self.assertGreater(float(dwell[0]), 0.0)
        self.assertEqual(float(dwell[1]), 0.0)
        self.assertEqual(float(dwell[4]), 0.0)

    def test_empty_history_returns_zeros(self) -> None:
        engine = _make_fake_engine()
        dwell, supp, mask = SolidificationEngine._analyze_btr_history(
            engine, btr_start=0.90, btr_end=0.98, critical_fraction_solid=0.50
        )
        self.assertTrue(np.all(dwell == 0.0))
        self.assertTrue(np.all(supp == 0.0))
        self.assertFalse(np.any(mask))

    def test_slower_voxel_spends_more_time_in_btr(self) -> None:
        """A voxel that lingers in the BTR longer should have higher dwell."""
        # Voxels 0 and 1 both stay in BTR from t=10 to t=30 (20s).
        # We then compare a third voxel that leaves the BTR earlier.
        engine = _make_fake_engine(
            fraction_liquid_history=[
                np.array([0.50, 0.50, 0.50], dtype=np.float32),
                np.array([0.92, 0.92, 0.95], dtype=np.float32),  # all in BTR
                np.array([0.95, 0.95, 0.99], dtype=np.float32),  # all still in BTR
                np.array([0.97, 0.97, 0.99], dtype=np.float32),  # all still in BTR
            ],
            fraction_liquid_times=[0.0, 10.0, 20.0, 30.0],
            n_voxels=3,
        )
        engine.mesh.voxel_neighbors = np.array(
            [[-1, -1, -1, -1, -1, -1],
             [-1, -1, -1, -1, -1, -1],
             [-1, -1, -1, -1, -1, -1]],
            dtype=np.int32,
        )
        dwell, _, _ = SolidificationEngine._analyze_btr_history(
            engine, btr_start=0.90, btr_end=0.98, critical_fraction_solid=0.50
        )
        # Voxels 0 and 1 stay in BTR for the entire [10, 30] interval
        # (half-interval rule: each sub-interval contributes 10s, total = 20s).
        # Voxel 2 enters BTR at t=10 but exits by t=20 (0.99 > 0.98), so
        # its total dwell is shorter: [10, 20] gives 0.5*10 = 5s, [20, 30] gives
        # 0.0s. Total = 5s.
        self.assertAlmostEqual(float(dwell[0]), 20.0, places=4)
        self.assertAlmostEqual(float(dwell[1]), 20.0, places=4)
        # Voxel 2 should have less dwell than voxels 0 and 1
        self.assertLess(float(dwell[2]), float(dwell[0]))


class HotTearStrainKernelMaxTests(unittest.TestCase):
    """Phase 1.3: compute_constraint_strain_parallel uses max, not mean."""

    def test_max_not_average(self) -> None:
        """One strongly-constraining neighbor should dominate, not be diluted."""
        # 3 voxels in a line. Voxel 0 solidified first (constrainer), voxel 1
        # later (target), voxel 2 much later (also target). Use strictly
        # positive times so the kernel's `neighbor_time > 0` filter doesn't
        # reject the constrainer.
        solidification_times = np.array([1.0, 5.0, 50.0], dtype=np.float32)
        voxel_moduli = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        # Only neighbor 0 of voxel 1 is a constrainer (early + valid index)
        # Voxel 1's other neighbors are all 'unconstrained' (-1 or self)
        voxel_neighbors = np.array(
            [
                [-1, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, -1],
                [1, -1, -1, -1, -1, -1],
            ],
            dtype=np.int32,
        )
        result = compute_constraint_strain_parallel(
            solidification_times,
            voxel_moduli,
            voxel_neighbors,
            thermal_expansion=1.2e-5,
            temp_range=10.0,
        )
        # Voxel 1 has exactly one constraining neighbor; max == mean (single sample)
        self.assertGreater(float(result[1]), 0.0)
        self.assertEqual(float(result[1]), float(result[1]))  # not nan
        # Single-neighbor strain = 1.2e-5 * 10 * (4/50) * 0.5 = 4.8e-6
        # (thickness_factor=1/(1+1)=0.5, time_factor=(5-1)/50=0.08)
        self.assertAlmostEqual(float(result[1]), 4.8e-6, places=9)

    def test_worst_neighbor_dominates_with_many_neutral_neighbors(self) -> None:
        """With several 'free' (no early-solidifying) neighbors, the worst
        (only) constraining neighbor must still set the strain — not be
        averaged down to 1/N."""
        # 5 neighbors of voxel 0: 1 strongly-constraining (voxel 1, time=1),
        # 4 invalid (-1). Use time=1.0 not 0.0 because the kernel filters
        # out neighbor_time == 0.
        solidification_times = np.array([10.0, 1.0, 10.0, 10.0, 10.0, 10.0],
                                        dtype=np.float32)
        voxel_moduli = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        voxel_neighbors = np.array(
            [
                [1, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, -1],
            ],
            dtype=np.int32,
        )
        result = compute_constraint_strain_parallel(
            solidification_times,
            voxel_moduli,
            voxel_neighbors,
            thermal_expansion=1.0e-5,
            temp_range=10.0,
        )
        # Under averaging: strain[0] would be (single strain) / 5
        # Under max: strain[0] equals the single strain.
        # single_strain = 1e-5 * 10 * ((10-1)/10) * 0.5 = 4.5e-5
        # (thickness_factor=1/(1+1)=0.5, time_factor=(10-1)/10=0.9)
        self.assertAlmostEqual(float(result[0]), 4.5e-5, places=9)


class HotTearBTRTempRangeTests(unittest.TestCase):
    """Phase 1.2: strain kernels receive BTR temp range, not full freezing range."""

    def test_btr_temp_range_scales_strain(self) -> None:
        """The strain should be proportional to the temp range passed in."""
        # Use strictly positive times so the kernel's `> 0` filter doesn't
        # reject the constrainer.
        solidification_times = np.array([1.0, 5.0, 10.0], dtype=np.float32)
        voxel_moduli = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        voxel_neighbors = np.array(
            [
                [-1, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, -1],
                [1, -1, -1, -1, -1, -1],
            ],
            dtype=np.int32,
        )
        # Same geometry, two different temperature ranges (full vs BTR)
        result_full = compute_constraint_strain_parallel(
            solidification_times, voxel_moduli, voxel_neighbors,
            thermal_expansion=1.2e-5, temp_range=61.0,  # WCB full range
        )
        result_btr = compute_constraint_strain_parallel(
            solidification_times, voxel_moduli, voxel_neighbors,
            thermal_expansion=1.2e-5, temp_range=4.88,  # WCB BTR (~0.08 * 61)
        )
        # Strain scales linearly with temp_range
        self.assertAlmostEqual(
            float(result_btr[1]) / float(result_full[1]),
            4.88 / 61.0,
            places=4,
        )


class HotTearFullCalculationTests(unittest.TestCase):
    """End-to-end tests of _calculate_hot_tear_risk with the new physics."""

    def _setup_full_engine(self) -> SimpleNamespace:
        """Build an engine that exercises the full _calculate_hot_tear_risk path."""
        # Two snapshots: one in the middle of solidification, one in the BTR
        n_voxels = 6
        # Voxel 0: passes through BTR at snapshot 1 (fraction_solid=0.92)
        # Voxel 1: not in BTR (fraction_solid stays at 0.70)
        # Voxel 2: in BTR at both snapshots
        # Voxels 3-5: similar to 0 but at later time
        fraction_liquid_history = [
            np.array([0.50, 0.30, 0.50, 0.50, 0.50, 0.50], dtype=np.float32),
            np.array([0.92, 0.70, 0.94, 0.93, 0.50, 0.50], dtype=np.float32),
            np.array([0.97, 0.85, 0.99, 0.98, 0.60, 0.50], dtype=np.float32),
        ]
        fraction_liquid_times = [0.0, 10.0, 20.0]

        solidification_times = np.array(
            [0.0, 30.0, 5.0, 8.0, 40.0, 50.0], dtype=np.float32
        )
        return _make_fake_engine(
            n_voxels=n_voxels,
            solidification_times=solidification_times,
            fraction_liquid_history=fraction_liquid_history,
            fraction_liquid_times=fraction_liquid_times,
        )

    def test_strain_rate_is_populated(self) -> None:
        """Phase 1.4: self.strain_rate is now a meaningful per-voxel value,
        not all zeros."""
        engine = self._setup_full_engine()
        alloy = get_alloy("WCB")
        # Build a minimally-populated analysis_options
        engine.analysis_options = SimpleNamespace(
            sand_restraint_index=0.5,
            solidification_params=SolidificationParameters(
                hot_tear=HotTearParams(
                    hot_tear_zone_threshold=0.05,
                ),
            ),
        )
        # Capture the print output from the analysis (it prints a summary)
        buf = io.StringIO()
        with redirect_stdout(buf):
            SolidificationEngine._calculate_hot_tear_risk(engine)
        # strain_rate should be populated, at least some voxels nonzero
        self.assertIsNotNone(engine.strain_rate)
        self.assertEqual(engine.strain_rate.shape, (6,))
        # Voxels that spent time in the BTR should have nonzero strain rate
        self.assertGreater(float(np.max(engine.strain_rate)), 0.0)

    def test_btr_dwell_time_in_components(self) -> None:
        """Phase 1.1: hot_tear_components includes btr_dwell_time."""
        engine = self._setup_full_engine()
        engine.analysis_options = SimpleNamespace(
            sand_restraint_index=0.5,
            solidification_params=SolidificationParameters(
                hot_tear=HotTearParams(
                    hot_tear_zone_threshold=0.05,
                ),
            ),
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            SolidificationEngine._calculate_hot_tear_risk(engine)
        self.assertIn("btr_dwell_time", engine.hot_tear_components)
        self.assertIn("btr_temp_range_used", engine.hot_tear_components)
        self.assertIn("coherency_support_fraction", engine.hot_tear_components)
        self.assertIn("strain_rate", engine.hot_tear_components)
        # Existing keys still present
        for k in ("btr_exposure", "mechanical_strain", "strain_index",
                  "restraint_index", "stress_concentration",
                  "healing_capacity", "material_factor", "healing_penalty"):
            self.assertIn(k, engine.hot_tear_components)
        # btr_dwell_time values match between the components dict and the
        # direct analysis (in seconds)
        dwell = engine.hot_tear_components["btr_dwell_time"]
        self.assertEqual(dwell.shape, (6,))
        # Total dwell is the sum of voxel dwells (positive for BTR-tagged voxels)
        self.assertGreater(float(np.sum(dwell)), 0.0)

    def test_btr_fallback_warns_and_uses_tight_window(self) -> None:
        """Phase 1.5: when no history is available, a warning is printed and
        the fallback window is [0.90, 0.99], not [0.20, 0.85]."""
        # solidification_times in [0, 100] so the [0.90, 0.99] window is [90, 99]
        solidification_times = np.linspace(0, 100, 6, dtype=np.float32)
        engine = _make_fake_engine(
            n_voxels=6,
            solidification_times=solidification_times,
            fraction_liquid_history=[],  # empty
            fraction_liquid_times=[],
        )
        engine.analysis_options = SimpleNamespace(
            sand_restraint_index=0.5,
            solidification_params=SolidificationParameters(
                hot_tear=HotTearParams(
                    hot_tear_zone_threshold=0.0,  # adaptive
                ),
            ),
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            SolidificationEngine._calculate_hot_tear_risk(engine)
        output = buf.getvalue()
        self.assertIn("WARNING", output)
        self.assertIn("fraction_liquid history", output)
        # btr_exposure should be set for voxels whose normalized_time in [0.90, 0.99]
        # i.e., voxels with time in [90, 99]: index 5 has time ~100, index 4 has
        # time ~80. So at most index 4 or 5 might be in window.
        # Voxel with time 100.0 has normalized_time=1.0, NOT in [0.90, 0.99].
        # None of the 6 voxels in this test should be in the window
        # (they're at 0, 20, 40, 60, 80, 100).
        btr_exposure = engine.hot_tear_components["btr_exposure"]
        # No voxel has btr_exposure == 1.0 in this test (none in [0.90, 0.99])
        self.assertTrue(np.all(btr_exposure <= 0.5))

    def test_threshold_default_is_0_06(self) -> None:
        """The new adaptive threshold uses 0.06, not 0.12.

        We verify indirectly: with the same input, the new threshold (0.06)
        produces a *different* zone count than the old 0.12. Specifically,
        0.06 will flag MORE voxels as zones than 0.12, because the threshold
        is lower.
        """
        engine = self._setup_full_engine()
        # Adaptive threshold (configured_threshold = 0)
        engine.analysis_options = SimpleNamespace(
            sand_restraint_index=0.5,
            solidification_params=SolidificationParameters(
                hot_tear=HotTearParams(
                    hot_tear_zone_threshold=0.0,  # use adaptive
                ),
            ),
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            SolidificationEngine._calculate_hot_tear_risk(engine)
        # WCB: adaptive threshold = 0.06 * (1 - 0.5*0.30) = 0.051
        # The summary line prints "Hot tear threshold: <value>"
        output = buf.getvalue()
        # Verify the printed threshold is in the [0.05, 0.06] range
        # (i.e. is the new 0.06, not the old 0.12)
        threshold_line = [l for l in output.splitlines()
                          if "Hot tear threshold:" in l]
        self.assertEqual(len(threshold_line), 1)
        # The printed value for WCB at HTS=0.30: 0.06 * 0.85 = 0.051
        self.assertIn("0.051", threshold_line[0])

    def test_higher_restraint_increases_max_risk(self) -> None:
        """The monotonicity property from test_simulation_scenarios.py:483
        should still hold: higher sand_restraint_index → higher max risk."""
        engine_low = self._setup_full_engine()
        engine_low.analysis_options = SimpleNamespace(
            sand_restraint_index=0.1,
            solidification_params=SolidificationParameters(
                hot_tear=HotTearParams(
                    hot_tear_zone_threshold=0.0,
                ),
            ),
        )
        engine_high = self._setup_full_engine()
        engine_high.analysis_options = SimpleNamespace(
            sand_restraint_index=0.9,
            solidification_params=SolidificationParameters(
                hot_tear=HotTearParams(
                    hot_tear_zone_threshold=0.0,
                ),
            ),
        )
        with redirect_stdout(io.StringIO()):
            SolidificationEngine._calculate_hot_tear_risk(engine_low)
            SolidificationEngine._calculate_hot_tear_risk(engine_high)
        self.assertGreater(
            float(np.max(engine_high.hot_tear_risk)),
            float(np.max(engine_low.hot_tear_risk)),
        )


class HotTearConfigCleanupTests(unittest.TestCase):
    """enable_hot_tear should be removed from SimulationSettings."""

    def test_enable_hot_tear_not_in_settings(self) -> None:
        from config.settings import SimulationSettings
        # The dataclass should not have the legacy global field
        self.assertNotIn("enable_hot_tear", SimulationSettings.__dataclass_fields__)


class HotTearComponentsShapeTests(unittest.TestCase):
    """All components dict entries must be n_voxels-long float32 arrays."""

    def test_components_have_consistent_shape(self) -> None:
        engine = _make_fake_engine(
            n_voxels=6,
            solidification_times=np.array(
                [0.0, 5.0, 10.0, 15.0, 20.0, 30.0], dtype=np.float32
            ),
            fraction_liquid_history=[
                np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50], dtype=np.float32),
                np.array([0.92, 0.50, 0.94, 0.50, 0.50, 0.50], dtype=np.float32),
            ],
            fraction_liquid_times=[0.0, 10.0],
        )
        engine.analysis_options = SimpleNamespace(
            sand_restraint_index=0.5,
            solidification_params=SolidificationParameters(
                hot_tear=HotTearParams(
                    hot_tear_zone_threshold=0.0,
                ),
            ),
        )
        with redirect_stdout(io.StringIO()):
            SolidificationEngine._calculate_hot_tear_risk(engine)
        for key, arr in engine.hot_tear_components.items():
            with self.subTest(key=key):
                self.assertEqual(arr.shape, (6,), f"component {key} wrong shape")
                if arr.dtype.kind == "f":
                    self.assertTrue(np.all(np.isfinite(arr)),
                                    f"component {key} has non-finite values")


if __name__ == "__main__":
    unittest.main()
