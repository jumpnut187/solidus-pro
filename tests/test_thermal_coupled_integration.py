from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from core.solidification_loop import run_time_integration
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
        self.assertTrue(np.all(engine.cooling_rates >= np.array([40.0, 35.0], dtype=np.float32)))
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


if __name__ == "__main__":
    unittest.main()
