"""Regression tests for the thermal-coupled integration loop internals.

Covers:
- A3: latent heat applied in the SAME step (no one-step lag), so per-step
  energy is conserved.
- A4: the heuristic feeding-zone vertical solidification bias is disabled in
  the temperature-driven thermal loop.
- B8: unfinished voxels get NaN end/solidification times (never 0.0, which
  would collide with chill/pre-solid semantics).
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from core.solidification_loop import _run_thermal_coupled_integration, run_time_integration
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


class _RecordingHeatModel:
    """Heat model that records the temperature field right after each step.

    ``post_step_temperatures`` holds a copy taken immediately after the pure
    diffusion step, so tests can verify the same-step latent correction is
    added on top of the diffusion-only result. ``set_heat_source`` is kept to
    prove the integration loop no longer calls it.
    """

    def __init__(self, n_voxels: int, initial: np.ndarray, drop_per_step: float) -> None:
        self._temperatures = np.asarray(initial, dtype=np.float64).copy()
        self.time = 0.0
        self._drop_per_step = float(drop_per_step)
        self.post_step_temperatures: list[np.ndarray] = []
        self._heat_source = np.zeros(n_voxels, dtype=np.float64)
        self.set_heat_source_calls = 0

    @property
    def temperatures(self) -> np.ndarray:
        return self._temperatures

    def step(self, dt: float) -> None:
        self._temperatures -= self._drop_per_step
        self.time += dt
        self.post_step_temperatures.append(self._temperatures.copy())

    def get_cooling_rate(self, prev_temperatures: np.ndarray, dt: float) -> np.ndarray:
        return (prev_temperatures - self._temperatures) / dt

    def get_gradient_magnitude(self, neighbor_map: np.ndarray, voxel_size_m: float) -> np.ndarray:
        del neighbor_map, voxel_size_m
        return np.zeros(self._temperatures.shape, dtype=np.float32)

    def set_heat_source(self, source: np.ndarray) -> None:
        self.set_heat_source_calls += 1
        self._heat_source[:] = source


def _build_engine(
    *,
    n_voxels: int = 2,
    initial_temperatures: np.ndarray | None = None,
    drop_per_step: float = 5.0,
    max_time: float = 2.0,
    time_step: float = 1.0,
    feeding_bias: float = 0.5,
    part_types: np.ndarray | None = None,
) -> SimpleNamespace:
    if initial_temperatures is None:
        initial_temperatures = np.full(n_voxels, 100.0, dtype=np.float64)

    sp = SolidificationParameters()
    sp.feeding.feeding_zone_vertical_bias = float(feeding_bias)

    mesh = SimpleNamespace(
        voxel_states=np.zeros(n_voxels, dtype=np.float32),
        voxel_neighbors=np.full((n_voxels, 26), -1, dtype=np.int32),
    )
    if part_types is not None:
        mesh.voxel_part_types = np.asarray(part_types, dtype=np.int32)

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
    engine.max_time = float(max_time)
    engine.time_step = float(time_step)
    engine.time = 0.0
    engine._logger = _DummyLogger()
    engine._thermal_heat_model = _RecordingHeatModel(
        n_voxels=n_voxels, initial=initial_temperatures, drop_per_step=drop_per_step
    )
    engine._thermal_solidification_model = SolidificationModel(
        n_voxels=n_voxels, alloy=engine.alloy
    )
    engine._thermal_peak_cooling_rates = np.zeros(n_voxels, dtype=np.float32)
    engine._thermal_latest_temperatures = None
    engine._thermal_latest_gradients = None
    engine._thermal_voxel_size_m = 0.00254
    return engine


class ThermalLoopRegressionTests(unittest.TestCase):
    def test_latent_heat_applied_same_step(self) -> None:
        """The latent release from ``solidification_model.update`` must be
        added to the SAME step's post-diffusion field, not deferred to the
        next step via a heat source."""
        engine = _build_engine(n_voxels=2, max_time=2.0, time_step=1.0)
        heat_model = engine._thermal_heat_model
        solidification_model = engine._thermal_solidification_model

        # Constant volumetric latent heat release (W/m^3) returned by update.
        rate = 3.0
        solidification_model.update = lambda temperatures, dt: np.full(2, rate, dtype=np.float64)  # type: ignore[method-assign]

        _run_thermal_coupled_integration(engine, show_progress=False, progress_callback=None)

        rho_cp = float(engine.alloy.density_liquid * engine.alloy.specific_heat)
        expected_correction = rate * engine.time_step / rho_cp
        self.assertEqual(len(heat_model.post_step_temperatures), 2)
        np.testing.assert_allclose(
            heat_model._temperatures,
            heat_model.post_step_temperatures[-1] + expected_correction,
            atol=1e-9,
            err_msg=(
                "Latent heat must be applied to the SAME step's post-diffusion "
                "field (diffusion result + rate*dt/rho_cp)"
            ),
        )

    def test_heat_source_not_set_in_loop(self) -> None:
        """The thermal loop must never call set_heat_source: latent heat is
        applied via the same-step temperature correction, so the heat source
        stays zeroed from initialization."""
        engine = _build_engine(n_voxels=2, max_time=2.0, time_step=1.0)
        heat_model = engine._thermal_heat_model

        _run_thermal_coupled_integration(engine, show_progress=False, progress_callback=None)

        self.assertEqual(heat_model.set_heat_source_calls, 0)
        np.testing.assert_allclose(
            heat_model._heat_source,
            np.zeros(2, dtype=np.float64),
            atol=0.0,
            err_msg="heat_model._heat_source must remain zeroed during the thermal loop",
        )

    def test_bias_disabled_in_thermal_mode(self) -> None:
        """feeding_zone_vertical_bias must not change thermal-coupled results:
        two identical engines with bias 0.0 vs 2.0 (riser assembly) produce
        identical voxel states and solidification times."""
        weak_engine = _build_engine(
            n_voxels=4,
            feeding_bias=0.0,
            max_time=3.5,
            time_step=1.0,
            drop_per_step=15.0,
            part_types=np.array([0, 1, 0, 1], dtype=np.int32),
        )
        strong_engine = _build_engine(
            n_voxels=4,
            feeding_bias=2.0,
            max_time=3.5,
            time_step=1.0,
            drop_per_step=15.0,
            part_types=np.array([0, 1, 0, 1], dtype=np.int32),
        )

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
            err_msg="Thermal mode is purely temperature-driven; the bias slider must not change fraction solid",
        )
        np.testing.assert_allclose(
            strong_engine.solidification_times,
            weak_engine.solidification_times,
            atol=1e-6,
            err_msg="Thermal mode is purely temperature-driven; the bias slider must not change solidification times",
        )
        np.testing.assert_allclose(
            strong_engine.solidification_end_times,
            weak_engine.solidification_end_times,
            atol=1e-6,
        )

    def test_unfinished_voxels_get_nan(self) -> None:
        """Voxels still liquid at the max_time cap get NaN end/solidification
        times (never 0.0); chill/pre-solid voxels keep their 0.0 times."""
        engine = _build_engine(
            n_voxels=3,
            initial_temperatures=np.array([90.0, 95.0, 200.0], dtype=np.float64),
            drop_per_step=5.0,
            max_time=1.0,
            time_step=1.0,
        )
        solidification_model = engine._thermal_solidification_model
        # Pre-solid chill voxel 0 (as set up by _initialize_thermal_solver).
        solidification_model._fraction_solid[0] = 1.0
        solidification_model._fraction_solid_prev[0] = 1.0
        solidification_model._solidification_start[0] = 0.0
        solidification_model._solidification_end[0] = 0.0
        engine.mesh.voxel_states[0] = 1.0
        engine.solidification_start_times[0] = 0.0
        engine.solidification_end_times[0] = 0.0
        engine.solidification_times[0] = 0.0
        engine.local_solidification_time[0] = 0.0

        _run_thermal_coupled_integration(engine, show_progress=False, progress_callback=None)

        states = engine.mesh.voxel_states
        # Voxel 1 started but did not finish; voxel 2 never started.
        self.assertLess(float(states[1]), 0.99)
        self.assertLess(float(states[2]), 0.99)
        self.assertGreater(float(states[1]), 0.0)
        self.assertAlmostEqual(float(states[2]), 0.0, places=6)

        # Chill voxel keeps its pre-solid 0.0 times.
        self.assertEqual(float(engine.solidification_end_times[0]), 0.0)
        self.assertEqual(float(engine.solidification_times[0]), 0.0)
        self.assertEqual(float(engine.local_solidification_time[0]), 0.0)

        # Unfinished voxels get NaN, never 0.0.
        for idx in (1, 2):
            self.assertTrue(np.isnan(engine.solidification_end_times[idx]), msg=f"end_times[{idx}] must be NaN")
            self.assertTrue(np.isnan(engine.solidification_times[idx]), msg=f"solidification_times[{idx}] must be NaN")
        # Voxel 2 never started: local solidification time must be NaN, not 0.0.
        self.assertTrue(np.isnan(engine.local_solidification_time[2]))

        unfinished = states < 0.99
        self.assertTrue(np.all(np.isnan(engine.solidification_times[unfinished])))
        self.assertFalse(np.any(engine.solidification_times[unfinished] == 0.0))
        self.assertTrue(np.all(np.isnan(engine.solidification_end_times[unfinished])))

    def test_same_step_energy_math(self) -> None:
        """Pure-math check of the same-step latent correction formula
        ``correction = latent_rate * dt / rho_cp`` with sample values."""
        latent_rate = np.array([1.0e6, 2.0e6, 0.0], dtype=np.float64)
        dt = 0.05
        rho_cp = 7200.0 * 600.0  # e.g. steel density * specific heat

        correction = latent_rate * dt / rho_cp
        expected = np.array(
            [1.0e6 * dt / rho_cp, 2.0e6 * dt / rho_cp, 0.0],
            dtype=np.float64,
        )
        np.testing.assert_allclose(correction, expected, rtol=1e-12)
        # The correction always raises temperature (latent release is positive).
        self.assertTrue(np.all(correction >= 0.0))
        self.assertEqual(float(correction[2]), 0.0)


if __name__ == "__main__":
    unittest.main()
