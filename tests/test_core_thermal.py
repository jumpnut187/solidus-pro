from __future__ import annotations

import contextlib
import io
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tests.helpers.ci_models import ci_model
import physics.heat_transfer as heat_transfer
from core.materials import get_alloy
from core.part_manager import PART_TYPE_CAST, PART_TYPE_CHILL, PART_TYPE_CORE, PartManager, PartType
from core.simulation_pipeline import PipelineInputs, run_pipeline
from core.solidification import AnalysisOptions, SolidificationEngine
from core.solidification_params import SolidificationParameters, ThermalBoundaryParams
from physics.heat_transfer import BoundaryType, HeatTransferModel
from physics.units import inches_to_meters
from tests.test_simulation_scenarios import _memory_monitor_check, _ok_validation

NUMBA = heat_transfer.NUMBA_AVAILABLE

def _line_engine(part_types: list[int]) -> SolidificationEngine:
    n_voxels = len(part_types)
    centers = np.asarray([[float(i), 0.0, 0.0] for i in range(n_voxels)], dtype=np.float32)
    neighbors = np.full((n_voxels, 26), -1, dtype=np.int32)
    for i in range(n_voxels):
        if i + 1 < n_voxels:
            neighbors[i, 0] = i + 1
        if i > 0:
            neighbors[i, 1] = i - 1
    mesh = SimpleNamespace(
        voxel_centers=centers,
        voxel_part_types=np.asarray(part_types, dtype=np.int32),
        voxel_states=np.zeros(n_voxels, dtype=np.float32),
        voxel_neighbors=neighbors,
        base_voxel_size=1.0,
    )
    mesh.get_surface_voxel_indices = lambda: np.arange(n_voxels, dtype=np.int32)
    engine = SolidificationEngine(mesh, get_alloy("WCB"), AnalysisOptions())
    engine.solidification_times = np.full(n_voxels, np.nan, dtype=np.float32)
    engine.solidification_start_times = np.full(n_voxels, np.nan, dtype=np.float32)
    engine.solidification_end_times = np.full(n_voxels, np.nan, dtype=np.float32)
    engine.local_solidification_time = np.full(n_voxels, np.nan, dtype=np.float32)
    engine.feeding_fraction_solid = np.full(n_voxels, np.nan, dtype=np.float32)
    return engine

LINE = [PART_TYPE_CAST, PART_TYPE_CAST, PART_TYPE_CORE, PART_TYPE_CORE, PART_TYPE_CAST, PART_TYPE_CHILL]
CORE_IDX = np.array([2, 3])

class CoreThermalModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = _line_engine(LINE)
        self.tb = self.engine.analysis_options.solidification_params.thermal_boundary

    def test_cores_get_sand_overrides_alongside_chill_overrides(self) -> None:
        self.engine._initialize_thermal_models(show_progress=False)
        model = self.engine._thermal_heat_model
        rho_cp = self.tb.core_density * self.tb.core_specific_heat
        alpha = dict(zip(*[a.tolist() for a in model._diffusivity_overrides]))
        capacity = dict(zip(*[a.tolist() for a in model._rho_cp_overrides]))
        for i in CORE_IDX:
            self.assertAlmostEqual(alpha[i], self.tb.core_thermal_conductivity / rho_cp, places=15)
            self.assertAlmostEqual(capacity[i], rho_cp)
        self.assertIn(5, alpha)
        self.assertAlmostEqual(alpha[5], 1.17e-4)

    def test_contact_conductance_marks_only_core_cells(self) -> None:
        self.engine._initialize_thermal_models(show_progress=False)
        hdx = self.engine._thermal_heat_model._contact_hdx
        expected = np.zeros(len(LINE))
        expected[CORE_IDX] = self.tb.htc_core_contact * inches_to_meters(1.0)
        np.testing.assert_allclose(hdx, expected)

    def test_cores_start_at_mold_temperature_with_and_without_a_fill(self) -> None:
        self.engine._initialize_thermal_models(show_progress=False)
        np.testing.assert_allclose(
            self.engine._thermal_heat_model.temperatures[CORE_IDX], self.tb.mold_temperature
        )
        filled = _line_engine(LINE)
        filled._initial_temperature_field = np.full(len(LINE), 1550.0)
        filled._initialize_thermal_models(show_progress=False)
        temperatures = filled._thermal_heat_model.temperatures
        np.testing.assert_allclose(temperatures[CORE_IDX], self.tb.mold_temperature)
        self.assertEqual(float(temperatures[0]), 1550.0)

    def test_cores_are_pre_solid_and_release_no_latent_heat(self) -> None:
        self.engine._initialize_thermal_models(show_progress=False)
        solid = self.engine._thermal_solidification_model
        np.testing.assert_array_equal(solid._fraction_solid[CORE_IDX], 1.0)
        np.testing.assert_array_equal(solid._fraction_solid_prev[CORE_IDX], 1.0)
        np.testing.assert_array_equal(self.engine.mesh.voxel_states[CORE_IDX], 1.0)
        np.testing.assert_array_equal(self.engine.local_solidification_time[CORE_IDX], 0.0)
        np.testing.assert_array_equal(self.engine.feeding_fraction_solid[CORE_IDX], 1.0)
        self.assertEqual(float(self.engine.mesh.voxel_states[0]), 0.0)

    def test_core_print_boundary_uses_its_own_htc_and_floor(self) -> None:
        self.engine._initialize_thermal_models(show_progress=False)
        print_bcs = [
            bc for bc in self.engine._thermal_heat_model._boundaries
            if bc.type == BoundaryType.INTERFACE and set(bc.indices.tolist()) == set(CORE_IDX.tolist())
        ]
        self.assertEqual(len(print_bcs), 1)
        bc = print_bcs[0]
        self.assertEqual(float(bc.htc), self.tb.htc_core_print)
        self.assertAlmostEqual(bc.time_function(1e9), self.tb.core_print_htc_floor_fraction)
        self.assertEqual(bc.time_function(0.0), 1.0)

    def test_contact_core_cells_list_cast_voxels_touching_a_core(self) -> None:
        np.testing.assert_array_equal(self.engine._get_core_contact_cast_indices(), [1, 4])

class CoreParamsTests(unittest.TestCase):
    def test_defaults_match_the_mold_sand_diffusivity(self) -> None:
        tb = ThermalBoundaryParams()
        self.assertAlmostEqual(
            tb.core_thermal_conductivity / (tb.core_density * tb.core_specific_heat),
            tb.mold_thermal_diffusivity,
            delta=0.01 * tb.mold_thermal_diffusivity,
        )

    def test_invalid_values_are_clamped(self) -> None:
        tb = ThermalBoundaryParams(core_density=-5.0, htc_core_contact="x", core_print_htc_floor_fraction=7.0)
        self.assertEqual(tb.core_density, 100.0)
        self.assertEqual(tb.htc_core_contact, 1000.0)
        self.assertEqual(tb.core_print_htc_floor_fraction, 1.0)

    def test_round_trip_through_dict(self) -> None:
        params = SolidificationParameters()
        params.thermal_boundary.htc_core_contact = 750.0
        params.thermal_boundary.core_specific_heat = 1000.0
        restored = SolidificationParameters.from_dict(params.to_dict())
        self.assertEqual(restored.thermal_boundary.htc_core_contact, 750.0)
        self.assertEqual(restored.thermal_boundary.core_specific_heat, 1000.0)

def _two_cell_model(h: float | None, *, numba: bool) -> HeatTransferModel:
    neighbors = np.full((2, 26), -1, dtype=np.int32)
    neighbors[0, 0] = 1
    neighbors[1, 1] = 0
    dx = 0.01
    sand_rho_cp = 1600.0 * 1140.0
    model = HeatTransferModel(
        n_voxels=2,
        voxel_size=dx,
        neighbors=neighbors,
        alloy=get_alloy("WCB"),
        diffusivity_overrides=(np.array([1]), np.array([0.6 / sand_rho_cp])),
        rho_cp_overrides=(np.array([1]), np.array([sand_rho_cp])),
        contact_conductance=None if h is None else (np.array([1]), h),
    )
    model.set_initial_temperature(np.array([1500.0, 20.0]))
    model._numba_parallel_eligible = False
    return model

class ContactConductanceTests(unittest.TestCase):
    def _step(self, h: float | None, numba: bool) -> tuple[HeatTransferModel, np.ndarray]:
        model = _two_cell_model(h, numba=numba)
        before = model.temperatures.copy()
        with patch.object(heat_transfer, "NUMBA_AVAILABLE", numba and NUMBA):
            model.step(1e-3)
        return model, model.temperatures - before

    def test_link_carries_h_in_series_with_the_pair_conductivity(self) -> None:
        model, d_free = self._step(None, numba=False)
        _, d_contact = self._step(500.0, numba=False)
        alloy = get_alloy("WCB")
        k_metal = alloy.thermal_diffusivity * model._rho_cp_scalar
        k_pair = 2.0 * k_metal * 0.6 / (k_metal + 0.6)
        expected_ratio = 1.0 / (1.0 + k_pair / (500.0 * 0.01))
        self.assertAlmostEqual(d_contact[0] / d_free[0], expected_ratio, places=9)
        self.assertLess(d_contact[1], d_free[1])

    def test_contact_conserves_energy(self) -> None:
        model, delta = self._step(500.0, numba=False)
        rho_cp = model.volumetric_heat_capacity()
        self.assertAlmostEqual(float(np.sum(rho_cp * delta)) / float(abs(rho_cp[0] * delta[0])), 0.0, places=9)

    @unittest.skipUnless(NUMBA, "Numba unavailable")
    def test_numba_kernel_matches_numpy_with_contact(self) -> None:
        _, d_numpy = self._step(500.0, numba=False)
        _, d_numba = self._step(500.0, numba=True)
        np.testing.assert_allclose(d_numba, d_numpy, rtol=1e-10)

class SandSlabHeatingTests(unittest.TestCase):

    def test_centre_reaches_half_the_face_rise_at_fo_0_379(self) -> None:
        n = 21
        dx = 0.002
        neighbors = np.full((n, 26), -1, dtype=np.int32)
        for i in range(n):
            if i + 1 < n:
                neighbors[i, 0] = i + 1
                neighbors[i, 18:22] = i + 1
            if i > 0:
                neighbors[i, 1] = i - 1
                neighbors[i, 22:26] = i - 1
        tb = ThermalBoundaryParams()
        rho_cp = tb.core_density * tb.core_specific_heat
        alpha = tb.core_thermal_conductivity / rho_cp
        every = np.arange(n)
        model = HeatTransferModel(
            n_voxels=n,
            voxel_size=dx,
            neighbors=neighbors,
            alloy=get_alloy("WCB"),
            diffusivity_overrides=(every, np.full(n, alpha)),
            rho_cp_overrides=(every, np.full(n, rho_cp)),
        )
        model.set_initial_temperature(np.zeros(n))
        model.add_boundary(np.array([0, n - 1]), BoundaryType.FIXED_TEMPERATURE, value=1000.0)
        t_half = 0.379 * (10 * dx) ** 2 / alpha
        model.step(t_half)
        self.assertAlmostEqual(float(model.temperatures[10]) / 1000.0, 0.5, delta=0.02)

def _cored_block_manager() -> PartManager:
    manager = PartManager()
    cast_id = manager.load_part(ci_model("block_2inch.stl"), name="cast")
    core_id = manager.load_part(ci_model("small_block.stl"), name="core")
    assert cast_id is not None and core_id is not None
    cast = manager.get_part(cast_id)
    core = manager.get_part(core_id)
    assert cast is not None and core is not None
    cast.set_part_type(PartType.CAST)
    core.set_part_type(PartType.CORE)
    cast.set_position(0.0, 0.0, 0.0)
    core.set_position(0.0, 0.0, 0.0)
    return manager

class EnclosedCorePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        inputs = PipelineInputs(
            alloy_name="WCB",
            voxel_size=0.2,
            refinement_level=0,
            memory_mode="conservative",
            analysis_options={"shrinkage": True, "hotspot": True, "niyama": True, "gas_porosity": True},
            max_voxels_for_visualization=5_000,
        )
        with ExitStack() as stack:
            stack.enter_context(patch("core.voxel_mesh.auto_adjust_voxel_size", return_value=(0.2, False, "")))
            stack.enter_context(patch("core.voxel_mesh.validate_mesh_parameters", side_effect=_ok_validation))
            stack.enter_context(patch("core.voxel_mesh.MemoryMonitor.start", lambda self: None))
            stack.enter_context(patch("core.voxel_mesh.MemoryMonitor.check", _memory_monitor_check))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            cls.engine = run_pipeline(inputs, part_manager=_cored_block_manager()).engine
        part_types = np.asarray(cls.engine.mesh.voxel_part_types, dtype=np.int32)
        cls.core = part_types == PART_TYPE_CORE
        cls.contact = cls.engine._get_core_contact_cast_indices()
        surface = np.asarray(cls.engine.mesh.get_surface_voxel_indices())
        cls.outer = surface[part_types[surface] == PART_TYPE_CAST]

    def test_fixture_has_an_enclosed_core(self) -> None:
        self.assertGreater(int(self.core.sum()), 0)
        self.assertGreater(len(self.contact), 0)
        surface = np.asarray(self.engine.mesh.get_surface_voxel_indices())
        self.assertFalse(np.any(self.core[surface]), "the core must not reach the outer surface")

    def test_core_voxels_are_never_metal_defects(self) -> None:
        for name in ("shrinkage_probability", "porosity_probability", "hotspot_intensity"):
            values = np.asarray(getattr(self.engine, name), dtype=float)
            self.assertEqual(float(values[self.core].max()), 0.0, name)
        np.testing.assert_array_equal(self.engine.local_solidification_time[self.core], 0.0)

    def test_the_core_heats_up(self) -> None:
        temperatures = np.asarray(self.engine._thermal_latest_temperatures, dtype=float)
        mold = self.engine.analysis_options.solidification_params.thermal_boundary.mold_temperature
        self.assertGreater(float(np.mean(temperatures[self.core])), mold + 500.0)

    def test_metal_against_the_saturated_core_freezes_after_the_outer_skin(self) -> None:
        end = np.asarray(self.engine.solidification_end_times, dtype=float)
        self.assertGreater(float(np.nanmean(end[self.contact])), float(np.nanmean(end[self.outer])))

    def test_enclosed_core_faces_are_gas_sources(self) -> None:
        generation = np.asarray(self.engine.gas_generation, dtype=float)
        self.assertGreater(float(np.max(generation[self.contact])), 0.0)

if __name__ == "__main__":
    unittest.main()
