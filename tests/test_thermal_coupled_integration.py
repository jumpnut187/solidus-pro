from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from core.solidification_loop import (
    _apply_feeding_zone_bias,
    _compute_feeding_zone_mask,
    _least_squares_gradients_c_per_mm,
    run_time_integration,
)
from core.solidification_params import SolidificationParameters
from physics.solidification_physics import SolidificationModel

class _DummyLogger:
    def info(self, _message: str, context=None) -> None:
        self.context = context

class _DummyAlloy:
    name = "DUMMY"
    liquidus_temp = 100.0
    solidus_temp = 0.0
    latent_heat = 10.0
    density_liquid = 2.0
    density_solid = 2.0
    specific_heat = 1.0
    critical_fraction_solid = 0.5

class _FakeTemperatureField:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def get_gradient_magnitude(self, neighbors: np.ndarray, voxel_size: float) -> np.ndarray:
        del neighbors, voxel_size
        return np.array([2000.0, 1000.0], dtype=np.float32)

class _FakeHeatModel:
    def __init__(self) -> None:
        self._temperatures = np.array([120.0, 90.0], dtype=np.float64)
        self.time = 0.0
        self.last_source = np.zeros(2, dtype=np.float64)

    @property
    def temperatures(self) -> np.ndarray:
        return self._temperatures

    @property
    def temperature_field(self) -> _FakeTemperatureField:
        return _FakeTemperatureField(self._temperatures.copy())

    def step(self, dt: float) -> None:
        self._temperatures -= np.array([40.0, 35.0], dtype=np.float64)
        self.time += dt

    def get_cooling_rate(self, prev_temperatures: np.ndarray, dt: float) -> np.ndarray:
        return (prev_temperatures - self._temperatures) / dt

    def set_heat_source(self, source: np.ndarray) -> None:
        self.last_source = np.asarray(source, dtype=np.float64).copy()

class _SlowHeatModel(_FakeHeatModel):
    def step(self, dt: float) -> None:
        self._temperatures -= np.array([5.0, 5.0], dtype=np.float64)
        self.time += dt

class _DummyEngine:
    def __init__(self) -> None:
        self.alloy = _DummyAlloy()
        self.mesh = SimpleNamespace(
            voxel_states=np.zeros(2, dtype=np.float32),
            voxel_neighbors=np.array([[1], [0]], dtype=np.int32),
        )
        self.analysis_options = SimpleNamespace(
            fraction_liquid=True,
            hot_tear=False,
            solver_mode="thermal_coupled",
            solidification_params=SolidificationParameters(),
        )
        self.fraction_liquid_history = []
        self.fraction_liquid_times = []
        self.solidification_times = np.zeros(2, dtype=np.float32)
        self.solidification_start_times = np.zeros(2, dtype=np.float32)
        self.solidification_end_times = np.zeros(2, dtype=np.float32)
        self.local_solidification_time = np.zeros(2, dtype=np.float32)
        self.feeding_fraction_solid = np.full(2, np.nan, dtype=np.float32)
        self.cooling_rates = None
        self.thermal_gradients = None
        self.max_time = 4.0
        self.time_step = 1.0
        self.time = 0.0
        self._logger = _DummyLogger()
        self._thermal_heat_model = _FakeHeatModel()
        self._thermal_solidification_model = SolidificationModel(n_voxels=2, alloy=self.alloy)
        self._thermal_peak_cooling_rates = np.zeros(2, dtype=np.float32)
        self._thermal_latest_temperatures = None
        self._thermal_latest_gradients = None
        self._thermal_voxel_size_m = 0.00254

class ThermalCoupledIntegrationTests(unittest.TestCase):
    def test_niyama_gradient_uses_voxel_offsets_in_mm(self) -> None:
        centers = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        offsets_mm = centers * 25.4
        temperatures = 100.0 + 2.0 * offsets_mm[:, 0] + 3.0 * offsets_mm[:, 1] + 4.0 * offsets_mm[:, 2]
        neighbors = np.array(
            [
                [1, 2, 3],
                [0, -1, -1],
                [0, -1, -1],
                [0, -1, -1],
            ],
            dtype=np.int32,
        )

        gradients, quality = _least_squares_gradients_c_per_mm(
            centers_in=centers,
            temperatures=temperatures,
            neighbors=neighbors,
            target_indices=np.array([0], dtype=np.int64),
            cast_mask=np.ones(4, dtype=bool),
        )

        self.assertEqual(int(quality[0]), 1)
        self.assertAlmostEqual(float(gradients[0]), float(np.sqrt(2.0**2 + 3.0**2 + 4.0**2)), places=5)

    def test_thermal_mode_populates_derived_state_and_history(self) -> None:
        engine = _DummyEngine()

        run_time_integration(
            engine,
            chill_boost=np.ones(2, dtype=np.float32),
            riser_slowdown=np.ones(2, dtype=np.float32),
            show_progress=False,
            progress_callback=None,
        )

        self.assertTrue(np.all(engine.mesh.voxel_states >= 0.99))
        self.assertTrue(np.all(engine.solidification_start_times > 0.0))
        self.assertTrue(np.all(engine.solidification_end_times >= engine.solidification_start_times))
        self.assertTrue(np.all(engine.local_solidification_time >= 0.0))
        self.assertTrue(np.all(np.isfinite(engine.feeding_fraction_solid)))
        np.testing.assert_allclose(engine.cooling_rates, np.array([39.090908, 35.0], dtype=np.float32), atol=1e-4)
        np.testing.assert_allclose(engine.thermal_gradients, np.array([2.0, 1.0], dtype=np.float32), atol=1e-6)
        self.assertGreaterEqual(len(engine.fraction_liquid_history), 2)
        self.assertEqual(engine._logger.context["mode"], "thermal_coupled")

    def test_thermal_mode_does_not_append_synthetic_fully_solid_snapshot(self) -> None:
        engine = _DummyEngine()
        engine._thermal_heat_model = _SlowHeatModel()
        engine.max_time = 1.0

        run_time_integration(
            engine,
            chill_boost=np.ones(2, dtype=np.float32),
            riser_slowdown=np.ones(2, dtype=np.float32),
            show_progress=False,
            progress_callback=None,
        )

        self.assertFalse(np.all(engine.mesh.voxel_states >= 0.99))
        self.assertEqual(engine.fraction_liquid_times[-1], engine.time)
        np.testing.assert_allclose(engine.fraction_liquid_history[-1], engine.mesh.voxel_states, atol=1e-6)

class _Uniform5DropHeatModel:

    def __init__(self, n_voxels: int, drop_per_step: float = 5.0) -> None:
        self._temperatures = np.full(n_voxels, 100.0, dtype=np.float64)
        self.time = 0.0
        self._drop_per_step = float(drop_per_step)

    @property
    def temperatures(self) -> np.ndarray:
        return self._temperatures

    @property
    def temperature_field(self) -> _FakeTemperatureField:
        return _FakeTemperatureField(self._temperatures.copy())

    def step(self, dt: float) -> None:
        self._temperatures -= self._drop_per_step
        self.time += dt

    def get_cooling_rate(self, prev_temperatures: np.ndarray, dt: float) -> np.ndarray:
        return (prev_temperatures - self._temperatures) / dt

    def set_heat_source(self, source: np.ndarray) -> None:
        del source

def _build_bias_engine(*, feeding_bias: float, with_risers: bool) -> SimpleNamespace:
    n_voxels = 4
    neighbors = np.full((n_voxels, 26), -1, dtype=np.int32)
    neighbors[0, 0] = 1
    neighbors[0, 1] = 2
    neighbors[1, 0] = 0
    neighbors[1, 1] = 3
    neighbors[2, 0] = 0
    neighbors[2, 1] = 3
    neighbors[3, 0] = 1
    neighbors[3, 1] = 2

    heights = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    voxel_centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    if with_risers:
        part_types = np.array([0, 1, 0, 1], dtype=np.int32)
    else:
        part_types = np.zeros(n_voxels, dtype=np.int32)

    sp = SolidificationParameters()
    sp.feeding.feeding_zone_vertical_bias = float(feeding_bias)

    mesh = SimpleNamespace(
        voxel_states=np.zeros(n_voxels, dtype=np.float32),
        voxel_neighbors=neighbors,
        voxel_part_types=part_types,
        voxel_centers=voxel_centers,
    )

    engine = SimpleNamespace()
    engine.alloy = _DummyAlloy()
    engine.mesh = mesh
    engine.analysis_options = SimpleNamespace(
        fraction_liquid=True,
        hot_tear=False,
        solver_mode="thermal_coupled",
        solidification_params=sp,
    )
    engine.fraction_liquid_history = []
    engine.fraction_liquid_times = []
    engine.solidification_times = np.zeros(n_voxels, dtype=np.float32)
    engine.solidification_start_times = np.zeros(n_voxels, dtype=np.float32)
    engine.solidification_end_times = np.zeros(n_voxels, dtype=np.float32)
    engine.local_solidification_time = np.zeros(n_voxels, dtype=np.float32)
    engine.feeding_fraction_solid = np.full(n_voxels, np.nan, dtype=np.float32)
    engine.cooling_rates = None
    engine.thermal_gradients = None
    engine.max_time = 15.0
    engine.time_step = 1.0
    engine.time = 0.0
    engine._logger = _DummyLogger()
    engine._thermal_heat_model = _Uniform5DropHeatModel(n_voxels=n_voxels)
    engine._thermal_solidification_model = SolidificationModel(
        n_voxels=n_voxels, alloy=engine.alloy
    )
    engine._thermal_peak_cooling_rates = np.zeros(n_voxels, dtype=np.float32)
    engine._thermal_latest_temperatures = None
    engine._thermal_latest_gradients = None
    engine._thermal_voxel_size_m = 0.00254

    def _normalized_heights() -> np.ndarray:
        return heights

    engine._get_normalized_heights = _normalized_heights
    return engine

class FeedingZoneBiasTests(unittest.TestCase):

    def test_compute_feeding_zone_mask_flags_riser_and_neighbors(self) -> None:
        part_types = np.array([0, 1, 0, 1], dtype=np.int32)
        neighbors = np.full((4, 26), -1, dtype=np.int32)
        neighbors[0, 0] = 1
        neighbors[1, 0] = 0
        neighbors[2, 0] = 3
        neighbors[3, 0] = 2
        states = np.zeros(4, dtype=np.float32)

        mask = _compute_feeding_zone_mask(states, neighbors, part_types)

        self.assertTrue(bool(mask[0]))
        self.assertTrue(bool(mask[1]))
        self.assertTrue(bool(mask[2]))
        self.assertTrue(bool(mask[3]))

    def test_compute_feeding_zone_mask_ignores_solid_riser_neighbors(self) -> None:
        part_types = np.array([0, 1], dtype=np.int32)
        neighbors = np.full((2, 26), -1, dtype=np.int32)
        neighbors[0, 0] = 1
        states = np.array([0.0, 1.0], dtype=np.float32)

        mask = _compute_feeding_zone_mask(states, neighbors, part_types)

        self.assertFalse(bool(mask[0]))
        self.assertTrue(bool(mask[1]))

    def test_thermal_mode_ignores_feeding_zone_bias_slider(self) -> None:
        weak_engine = _build_bias_engine(feeding_bias=0.0, with_risers=True)
        strong_engine = _build_bias_engine(feeding_bias=2.0, with_risers=True)
        for engine in (weak_engine, strong_engine):
            engine._thermal_heat_model = _Uniform5DropHeatModel(
                n_voxels=4, drop_per_step=15.0
            )
            engine.max_time = 3.5
            engine.time_step = 1.0

        run_time_integration(
            weak_engine,
            chill_boost=np.ones(4, dtype=np.float32),
            riser_slowdown=np.ones(4, dtype=np.float32),
            show_progress=False,
            progress_callback=None,
        )
        run_time_integration(
            strong_engine,
            chill_boost=np.ones(4, dtype=np.float32),
            riser_slowdown=np.ones(4, dtype=np.float32),
            show_progress=False,
            progress_callback=None,
        )

        np.testing.assert_allclose(
            strong_engine.mesh.voxel_states,
            weak_engine.mesh.voxel_states,
            atol=1e-6,
            err_msg=(
                "Thermal-coupled mode is purely temperature-driven; the "
                "feeding_zone_vertical_bias slider must not change fraction solid"
            ),
        )
        np.testing.assert_allclose(
            strong_engine.solidification_end_times,
            weak_engine.solidification_end_times,
            atol=1e-6,
            err_msg=(
                "Thermal-coupled mode must ignore feeding_zone_vertical_bias "
                "when writing solidification end times"
            ),
        )
        np.testing.assert_allclose(
            strong_engine.solidification_start_times,
            weak_engine.solidification_start_times,
            atol=1e-6,
        )

    def test_thermal_mode_bias_slider_ignored_without_risers(self) -> None:
        zero_engine = _build_bias_engine(feeding_bias=0.0, with_risers=False)
        max_engine = _build_bias_engine(feeding_bias=2.0, with_risers=False)
        for engine in (zero_engine, max_engine):
            engine._thermal_heat_model = _Uniform5DropHeatModel(
                n_voxels=4, drop_per_step=15.0
            )
            engine.max_time = 3.5
            engine.time_step = 1.0

        run_time_integration(
            zero_engine,
            chill_boost=np.ones(4, dtype=np.float32),
            riser_slowdown=np.ones(4, dtype=np.float32),
            show_progress=False,
            progress_callback=None,
        )
        run_time_integration(
            max_engine,
            chill_boost=np.ones(4, dtype=np.float32),
            riser_slowdown=np.ones(4, dtype=np.float32),
            show_progress=False,
            progress_callback=None,
        )

        np.testing.assert_allclose(
            max_engine.solidification_end_times,
            zero_engine.solidification_end_times,
            err_msg=(
                "Without RISER voxels, the feeding_zone_vertical_bias slider "
                "must not change the simulation (bias disabled in thermal mode)"
            ),
        )
        np.testing.assert_allclose(
            max_engine.solidification_start_times,
            zero_engine.solidification_start_times,
            atol=1e-6,
        )

    def test_apply_feeding_zone_bias_helper_returns_true_with_default_baseline(
        self,
    ) -> None:
        engine = _build_bias_engine(feeding_bias=0.0, with_risers=False)
        solidification_model = engine._thermal_solidification_model
        solidification_model._fraction_solid[:] = np.array([0.5, 0.5, 0.5, 0.5])
        engine.time = 5.0

        applied = _apply_feeding_zone_bias(engine, solidification_model, 0.0)

        self.assertTrue(applied)
        states = np.asarray(solidification_model._fraction_solid, dtype=np.float64)
        self.assertGreater(float(states[0]), 0.5)
        self.assertAlmostEqual(float(states[2]), 0.5, places=6)
        self.assertAlmostEqual(float(states[3]), 0.5, places=6)

    def test_apply_feeding_zone_bias_helper_advances_bottom_only(self) -> None:
        engine = _build_bias_engine(feeding_bias=2.0, with_risers=True)
        solidification_model = engine._thermal_solidification_model
        solidification_model._fraction_solid[:] = np.array([0.2, 0.2, 0.2, 0.2])
        solidification_model._fraction_solid_prev[:] = np.array([0.2, 0.2, 0.2, 0.2])
        solidification_model._solidification_start[:] = 1.0
        solidification_model._solidification_start[:] = 1.0
        engine.time = 5.0

        applied = _apply_feeding_zone_bias(engine, solidification_model, 2.0)

        self.assertTrue(applied)
        states = np.asarray(solidification_model._fraction_solid, dtype=np.float64)
        self.assertGreater(float(states[0]), 0.2)
        self.assertGreater(float(states[1]), 0.2)
        self.assertAlmostEqual(float(states[2]), 0.2, places=6)
        self.assertAlmostEqual(float(states[3]), 0.2, places=6)

    def test_baseline_bias_makes_casting_solidify_bottom_up(self) -> None:
        engine = _build_bias_engine(feeding_bias=0.0, with_risers=False)
        solidification_model = engine._thermal_solidification_model
        solidification_model._fraction_solid[:] = np.array([0.4, 0.4, 0.4, 0.4])
        solidification_model._fraction_solid_prev[:] = np.array([0.4, 0.4, 0.4, 0.4])
        solidification_model._solidification_start[:] = 1.0
        engine.time = 3.0

        applied = _apply_feeding_zone_bias(engine, solidification_model, 0.0)

        self.assertTrue(applied)
        states = np.asarray(solidification_model._fraction_solid, dtype=np.float64)
        self.assertAlmostEqual(float(states[0]), 0.406, places=6)
        self.assertAlmostEqual(float(states[1]), 0.406, places=6)
        self.assertAlmostEqual(float(states[2]), 0.4, places=6)
        self.assertAlmostEqual(float(states[3]), 0.4, places=6)

if __name__ == "__main__":
    unittest.main()
