"""Parity tests for the Numba parallel kernels that replaced serial NumPy.

Each kernel introduced to cut the thermal loop's single-threaded fraction has
a NumPy formulation it must agree with. These tests pin that agreement so a
future kernel edit cannot silently change the physics.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import core.solidification_loop as solidification_loop
import physics.heat_transfer as heat_transfer
import physics.solidification_physics as solidification_physics
from core.solidification_loop import _apply_enthalpy_correction_numpy
from physics.solidification_physics import SolidificationMethod, SolidificationModel

NUMBA_KERNELS = heat_transfer.NUMBA_AVAILABLE and solidification_physics.NUMBA_AVAILABLE


class _DummyAlloy:
    name = "DUMMY"
    liquidus_temp = 1500.0
    solidus_temp = 1450.0
    latent_heat = 272000.0
    density_liquid = 7200.0
    density_solid = 7800.0
    specific_heat = 600.0
    thermal_conductivity = 45.0
    thermal_diffusivity = 45.0 / (7200.0 * 600.0)
    critical_fraction_solid = 0.7
    melting_point_solvent = 1538.0
    partition_coefficient = 0.17
    eutectic_temperature = None
    eutectic_fraction = 0.0


def _production_curves() -> dict[str, np.ndarray]:
    """Curve set with the same shape core/solidification.py builds."""
    solidus = _DummyAlloy.solidus_temp
    liquidus = _DummyAlloy.liquidus_temp
    k_s = _DummyAlloy.thermal_conductivity
    cp = _DummyAlloy.specific_heat
    rho_s = _DummyAlloy.density_solid
    rho_l = _DummyAlloy.density_liquid
    return {
        "temp_points": np.array(
            [20.0, 0.5 * solidus, solidus, (solidus + liquidus) / 2, liquidus, liquidus + 100]
        ),
        "conductivity": k_s * np.array([1.0, 0.95, 0.85, 0.77, 0.70, 0.665]),
        "specific_heat": cp * np.array([0.85, 0.92, 1.0, 1.05, 1.10, 1.15]),
        "density": np.array(
            [rho_s, 0.98 * rho_s, 0.96 * rho_s, (rho_s + rho_l) / 2, rho_l, 0.98 * rho_l]
        ),
    }


@unittest.skipUnless(NUMBA_KERNELS, "Numba kernels unavailable")
class DiffusivityKernelParityTests(unittest.TestCase):
    """compute_diffusivity_from_curves_parallel vs three np.interp sweeps."""

    def _temperatures(self) -> np.ndarray:
        curves = _production_curves()
        points = curves["temp_points"]
        rng = np.random.default_rng(20260827)
        return np.concatenate(
            [
                rng.uniform(-200.0, 2200.0, 50_000),  # spans below/inside/above
                points,  # exact breakpoints
                np.nextafter(points, -np.inf),
                np.nextafter(points, np.inf),
                np.array([np.nan, np.inf, -np.inf]),
            ]
        )

    def test_matches_np_interp(self) -> None:
        curves = _production_curves()
        T = self._temperatures()

        expected = np.interp(T, curves["temp_points"], curves["conductivity"]) / (
            np.interp(T, curves["temp_points"], curves["density"])
            * np.interp(T, curves["temp_points"], curves["specific_heat"])
        )
        actual = heat_transfer.compute_diffusivity_from_curves_parallel(
            np.ascontiguousarray(T, dtype=np.float64),
            np.ascontiguousarray(curves["temp_points"], dtype=np.float64),
            np.ascontiguousarray(curves["conductivity"], dtype=np.float64),
            np.ascontiguousarray(curves["density"], dtype=np.float64),
            np.ascontiguousarray(curves["specific_heat"], dtype=np.float64),
        )

        finite = np.isfinite(expected)
        # The kernel reproduces np.interp's slope-then-offset arithmetic, but
        # exact bit-identity is not guaranteed across the two implementations.
        np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-12, atol=0.0)
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        np.testing.assert_array_equal(actual[~finite & ~np.isnan(expected)],
                                      expected[~finite & ~np.isnan(expected)])

    def test_model_dispatch_matches_numpy_fallback(self) -> None:
        """The model's kernel and NumPy branches agree end to end."""
        neighbors = np.array([[1, -1], [0, -1]], dtype=np.int32)
        model = heat_transfer.HeatTransferModel(
            n_voxels=2,
            voxel_size=1.0e-3,
            neighbors=neighbors,
            alloy=_DummyAlloy(),
            use_temperature_dependent=True,
            property_curves=_production_curves(),
        )
        T = np.array([1520.0, 1470.0], dtype=np.float64)

        with_kernel = model._get_temperature_dependent_diffusivity(T)
        with patch.object(heat_transfer, "NUMBA_AVAILABLE", False):
            with_numpy = model._get_temperature_dependent_diffusivity(T)

        np.testing.assert_allclose(with_kernel, with_numpy, rtol=1e-12, atol=0.0)


@unittest.skipUnless(NUMBA_KERNELS, "Numba kernels unavailable")
class SolidificationCommitParityTests(unittest.TestCase):
    """SolidificationModel.update's kernel path vs its NumPy path."""

    def _run_sequence(self, method: SolidificationMethod, *, use_numba: bool) -> dict[str, np.ndarray]:
        model = SolidificationModel(n_voxels=6, alloy=_DummyAlloy(), method=method)
        rates = []
        # Cooling sequence covering: staying liquid, partial mushy progress,
        # a single-step liquid -> solid crossing (voxel 5), and a temperature
        # rise that must not un-solidify anything.
        schedule = [
            np.array([1600.0, 1520.0, 1490.0, 1470.0, 1455.0, 1600.0]),
            np.array([1560.0, 1495.0, 1480.0, 1460.0, 1449.0, 1400.0]),
            np.array([1520.0, 1485.0, 1470.0, 1452.0, 1440.0, 1390.0]),
            np.array([1530.0, 1490.0, 1465.0, 1448.0, 1430.0, 1395.0]),
            np.array([1480.0, 1460.0, 1451.0, 1440.0, 1420.0, 1380.0]),
        ]
        for temperatures in schedule:
            with patch.object(solidification_physics, "NUMBA_AVAILABLE", use_numba):
                rates.append(model.update(temperatures.copy(), dt=0.5))
        return {
            "fraction_solid": np.asarray(model.fraction_solid).copy(),
            "fraction_solid_prev": model._fraction_solid_prev.copy(),
            "start": model._solidification_start.copy(),
            "end": model._solidification_end.copy(),
            "rates": np.asarray(rates),
            "time": np.array([model.time]),
        }

    def _assert_identical(self, method: SolidificationMethod) -> None:
        kernel = self._run_sequence(method, use_numba=True)
        numpy_ref = self._run_sequence(method, use_numba=False)
        for key in kernel:
            np.testing.assert_array_equal(
                kernel[key], numpy_ref[key], err_msg=f"{method} mismatch in {key}"
            )

    def test_lever_rule_identical(self) -> None:
        self._assert_identical(SolidificationMethod.LEVER_RULE)

    def test_scheil_identical(self) -> None:
        self._assert_identical(SolidificationMethod.SCHEIL)

    def test_back_diffusion_identical(self) -> None:
        self._assert_identical(SolidificationMethod.BACK_DIFFUSION)

    def test_single_step_crossing_spans_one_step(self) -> None:
        """A liquid -> solid jump inside one step keeps a non-zero interval."""
        model = SolidificationModel(
            n_voxels=1, alloy=_DummyAlloy(), method=SolidificationMethod.LEVER_RULE
        )
        model.update(np.array([1600.0]), dt=0.5)
        model.update(np.array([1000.0]), dt=0.5)
        self.assertEqual(model._solidification_start[0], 0.5)
        self.assertEqual(model._solidification_end[0], 1.0)
        self.assertEqual(model.local_solidification_time[0], 0.5)

    def test_returned_rate_is_independent_of_model_buffer(self) -> None:
        model = SolidificationModel(
            n_voxels=2, alloy=_DummyAlloy(), method=SolidificationMethod.LEVER_RULE
        )
        first = model.update(np.array([1480.0, 1600.0]), dt=1.0)
        snapshot = first.copy()
        model.update(np.array([1400.0, 1400.0]), dt=1.0)
        np.testing.assert_array_equal(first, snapshot)


@unittest.skipUnless(NUMBA_KERNELS, "Numba kernels unavailable")
class EnthalpyCorrectionParityTests(unittest.TestCase):
    """apply_enthalpy_correction_parallel vs its NumPy reference."""

    def test_all_branches_identical(self) -> None:
        T_liq = _DummyAlloy.liquidus_temp
        T_sol = _DummyAlloy.solidus_temp
        dT = T_liq - T_sol
        rho_cp = _DummyAlloy.density_liquid * _DummyAlloy.specific_heat
        beta = (
            _DummyAlloy.latent_heat
            * (_DummyAlloy.density_liquid + _DummyAlloy.density_solid)
            / 2.0
        ) / rho_cp

        rng = np.random.default_rng(4242)
        # Random field plus explicit boundary values so every branch is hit:
        # above liquidus, exactly at each bound, mushy, and below solidus.
        T_post = np.concatenate(
            [
                rng.uniform(T_sol - 200.0, T_liq + 200.0, 40_000),
                np.array([T_liq + 1.0, T_liq, T_liq - 1e-9, 1475.0, T_sol, T_sol - 1e-9, 1000.0]),
            ]
        )
        fs_prev = np.concatenate(
            [
                rng.uniform(0.0, 1.0, 40_000),
                np.array([0.0, 0.0, 0.0, 0.5, 0.25, 1.0, 0.9]),
            ]
        )

        expected = np.empty_like(T_post)
        _apply_enthalpy_correction_numpy(T_post, fs_prev, expected, T_liq, T_sol, dT, beta)
        actual = np.empty_like(T_post)
        solidification_physics.apply_enthalpy_correction_parallel(
            T_post, fs_prev, actual, T_liq, T_sol, dT, beta
        )

        np.testing.assert_array_equal(actual, expected)

    def test_recalescence_never_exceeds_solidus(self) -> None:
        T_liq, T_sol, dT, beta = 1500.0, 1450.0, 50.0, 85.0
        T_post = np.array([1400.0, 1449.0], dtype=np.float64)
        fs_prev = np.array([0.0, 0.1], dtype=np.float64)
        actual = np.empty_like(T_post)
        solidification_physics.apply_enthalpy_correction_parallel(
            T_post, fs_prev, actual, T_liq, T_sol, dT, beta
        )
        self.assertTrue(np.all(actual <= T_sol))


@unittest.skipUnless(NUMBA_KERNELS, "Numba kernels unavailable")
class ThermalLoopEndToEndParityTests(unittest.TestCase):
    """A full thermal-coupled run agrees between kernel and NumPy paths."""

    def _run(self, *, use_numba: bool) -> dict[str, np.ndarray]:
        # Imported lazily: building an engine pulls in the GPU manager.
        from tests.test_thermal_boundary_config import _build_engine

        engine = _build_engine(with_chill=True)
        engine.time_step = 0.1
        engine.max_time = 40.0
        engine._initialize_thermal_models(show_progress=False)

        patches = [
            patch.object(solidification_physics, "NUMBA_AVAILABLE", use_numba),
            patch.object(solidification_loop, "ENTHALPY_KERNEL_AVAILABLE", use_numba),
            patch.object(heat_transfer, "NUMBA_AVAILABLE", use_numba),
        ]
        for p in patches:
            p.start()
        try:
            solidification_loop._run_thermal_coupled_integration(
                engine,
                show_progress=False,
                progress_callback=None,
            )
        finally:
            for p in patches:
                p.stop()

        return {
            "states": np.asarray(engine.mesh.voxel_states).copy(),
            "start": np.asarray(engine.solidification_start_times).copy(),
            "end": np.asarray(engine.solidification_end_times).copy(),
            "local": np.asarray(engine.local_solidification_time).copy(),
            "temps": np.asarray(engine._thermal_latest_temperatures).copy(),
            "time": np.array([engine.time]),
        }

    def test_kernel_and_numpy_paths_agree(self) -> None:
        kernel = self._run(use_numba=True)
        numpy_ref = self._run(use_numba=False)
        for key in kernel:
            np.testing.assert_allclose(
                kernel[key],
                numpy_ref[key],
                rtol=1e-9,
                atol=1e-9,
                equal_nan=True,
                err_msg=f"end-to-end mismatch in {key}",
            )


if __name__ == "__main__":
    unittest.main()
