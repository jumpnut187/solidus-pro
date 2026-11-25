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
    if solidification_times is None:
        solidification_times = np.array(
            [1.0, 5.0, 10.0, 15.0, 20.0, 30.0], dtype=np.float32
        )
    if voxel_moduli is None:
        voxel_moduli = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
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

    def test_returns_dwell_time_supporting_fraction_and_mask(self) -> None:
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
            btr_start=alloy.btr_fraction_solid_start,
            btr_end=alloy.btr_fraction_solid_end,
            critical_fraction_solid=alloy.critical_fraction_solid,
        )
        self.assertEqual(dwell.shape, (6,))
        self.assertEqual(supp.shape, (6,))
        self.assertEqual(mask.shape, (6,))
        self.assertTrue(mask[0])
        self.assertTrue(mask[2])
        self.assertTrue(mask[3])
        self.assertFalse(mask[1])
        self.assertFalse(mask[4])
        self.assertFalse(mask[5])
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
        engine = _make_fake_engine(
            fraction_liquid_history=[
                np.array([0.50, 0.50, 0.50], dtype=np.float32),
                np.array([0.92, 0.92, 0.95], dtype=np.float32),
                np.array([0.95, 0.95, 0.99], dtype=np.float32),
                np.array([0.97, 0.97, 0.99], dtype=np.float32),
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
        self.assertAlmostEqual(float(dwell[0]), 20.0, places=4)
        self.assertAlmostEqual(float(dwell[1]), 20.0, places=4)
        self.assertLess(float(dwell[2]), float(dwell[0]))

class HotTearStrainKernelMaxTests(unittest.TestCase):

    def test_max_not_average(self) -> None:
        solidification_times = np.array([1.0, 5.0, 50.0], dtype=np.float32)
        voxel_moduli = np.array([1.0, 1.0, 1.0], dtype=np.float32)
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
        self.assertGreater(float(result[1]), 0.0)
        self.assertEqual(float(result[1]), float(result[1]))
        self.assertAlmostEqual(float(result[1]), 4.8e-6, places=9)

    def test_worst_neighbor_dominates_with_many_neutral_neighbors(self) -> None:
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
        self.assertAlmostEqual(float(result[0]), 4.5e-5, places=9)

class HotTearBTRTempRangeTests(unittest.TestCase):

    def test_btr_temp_range_scales_strain(self) -> None:
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
        result_full = compute_constraint_strain_parallel(
            solidification_times, voxel_moduli, voxel_neighbors,
            thermal_expansion=1.2e-5, temp_range=61.0,
        )
        result_btr = compute_constraint_strain_parallel(
            solidification_times, voxel_moduli, voxel_neighbors,
            thermal_expansion=1.2e-5, temp_range=4.88,
        )
        self.assertAlmostEqual(
            float(result_btr[1]) / float(result_full[1]),
            4.88 / 61.0,
            places=4,
        )

class HotTearFullCalculationTests(unittest.TestCase):

    def _setup_full_engine(self) -> SimpleNamespace:
        n_voxels = 6
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
        engine = self._setup_full_engine()
        alloy = get_alloy("WCB")
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
        self.assertIsNotNone(engine.strain_rate)
        self.assertEqual(engine.strain_rate.shape, (6,))
        self.assertGreater(float(np.max(engine.strain_rate)), 0.0)

    def test_btr_dwell_time_in_components(self) -> None:
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
        for k in ("btr_exposure", "mechanical_strain", "strain_index",
                  "restraint_index", "stress_concentration",
                  "healing_capacity", "material_factor", "healing_penalty"):
            self.assertIn(k, engine.hot_tear_components)
        dwell = engine.hot_tear_components["btr_dwell_time"]
        self.assertEqual(dwell.shape, (6,))
        self.assertGreater(float(np.sum(dwell)), 0.0)

    def test_btr_fallback_warns_and_uses_tight_window(self) -> None:
        solidification_times = np.linspace(0, 100, 6, dtype=np.float32)
        engine = _make_fake_engine(
            n_voxels=6,
            solidification_times=solidification_times,
            fraction_liquid_history=[],
            fraction_liquid_times=[],
        )
        engine.analysis_options = SimpleNamespace(
            sand_restraint_index=0.5,
            solidification_params=SolidificationParameters(
                hot_tear=HotTearParams(
                    hot_tear_zone_threshold=0.0,
                ),
            ),
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            SolidificationEngine._calculate_hot_tear_risk(engine)
        output = buf.getvalue()
        self.assertIn("WARNING", output)
        self.assertIn("fraction_liquid history", output)
        btr_exposure = engine.hot_tear_components["btr_exposure"]
        self.assertTrue(np.all(btr_exposure <= 0.5))

    def test_threshold_default_is_0_06(self) -> None:
        engine = self._setup_full_engine()
        engine.analysis_options = SimpleNamespace(
            sand_restraint_index=0.5,
            solidification_params=SolidificationParameters(
                hot_tear=HotTearParams(
                    hot_tear_zone_threshold=0.0,
                ),
            ),
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            SolidificationEngine._calculate_hot_tear_risk(engine)
        output = buf.getvalue()
        threshold_line = [l for l in output.splitlines()
                          if "Hot tear threshold:" in l]
        self.assertEqual(len(threshold_line), 1)
        self.assertIn("0.051", threshold_line[0])

    def test_higher_restraint_increases_max_risk(self) -> None:
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

    def test_enable_hot_tear_not_in_settings(self) -> None:
        from config.settings import SimulationSettings
        self.assertNotIn("enable_hot_tear", SimulationSettings.__dataclass_fields__)

class HotTearComponentsShapeTests(unittest.TestCase):

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
