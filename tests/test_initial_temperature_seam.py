from __future__ import annotations

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np

from tests.helpers.ci_models import ci_model
import tests.test_simulation_scenarios as scenarios
from core.materials import get_alloy
from core.part_manager import is_chill
from core.simulation_pipeline import (
    PipelineInputs,
    create_analysis_options,
    prepare_mesh_and_geometry,
)
from core.solidification import SolidificationEngine

def _build_engine(part_manager, analysis_options=None):
    inputs = PipelineInputs(
        alloy_name="WCB",
        voxel_size=0.5,
        refinement_level=0,
        memory_mode="conservative",
        analysis_options=analysis_options or {"shrinkage": True, "hotspot": True},
        max_voxels_for_visualization=5000,
    )
    with ExitStack() as stack:
        stack.enter_context(
            patch("core.voxel_mesh.auto_adjust_voxel_size", return_value=(0.5, False, ""))
        )
        stack.enter_context(
            patch("core.voxel_mesh.validate_mesh_parameters", side_effect=scenarios._ok_validation)
        )
        stack.enter_context(patch("core.voxel_mesh.MemoryMonitor.start", lambda self: None))
        stack.enter_context(
            patch("core.voxel_mesh.MemoryMonitor.check", scenarios._memory_monitor_check)
        )
        _, _, _, _, mesh, _ = prepare_mesh_and_geometry(
            inputs, part_manager=part_manager, stl_loader=None
        )

    alloy = get_alloy("WCB")
    options = create_analysis_options(
        inputs.analysis_options, use_gpu=False, use_incremental=False
    )
    engine = SolidificationEngine(mesh, alloy, options)
    engine.initialize()
    return engine

class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = _build_engine(
            scenarios._single_cast_manager(ci_model("block_2inch.stl"))
        )
        cls.n_voxels = cls.engine.mesh.get_voxel_count()

    def test_defaults_to_no_override(self) -> None:
        engine = _build_engine(
            scenarios._single_cast_manager(ci_model("block_2inch.stl"))
        )
        self.assertIsNone(engine._initial_temperature_field)
        self.assertIsNone(engine._fill_arrival_times)

    def test_wrong_length_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.engine.set_initial_temperature_field(np.zeros(self.n_voxels + 1))
        self.assertIn("expected", str(ctx.exception))

    def test_non_finite_values_are_rejected(self) -> None:
        field = np.full(self.n_voxels, 1500.0)
        field[0] = np.nan
        with self.assertRaises(ValueError):
            self.engine.set_initial_temperature_field(field)

    def test_wrong_fill_time_length_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.engine.set_initial_temperature_field(
                np.full(self.n_voxels, 1500.0),
                fill_arrival_times=np.zeros(self.n_voxels + 3),
            )

    def test_field_is_copied_not_aliased(self) -> None:
        field = np.full(self.n_voxels, 1500.0)
        self.engine.set_initial_temperature_field(field)
        field[:] = 9999.0
        self.assertTrue(np.allclose(self.engine._initial_temperature_field, 1500.0))

    def test_fill_arrival_times_are_stored(self) -> None:
        times = np.linspace(0.0, 4.0, self.n_voxels)
        self.engine.set_initial_temperature_field(
            np.full(self.n_voxels, 1500.0), fill_arrival_times=times
        )
        np.testing.assert_allclose(self.engine._fill_arrival_times, times)

class SeededSolveTests(unittest.TestCase):

    @staticmethod
    def _run(seed_offset):
        engine = _build_engine(
            scenarios._single_cast_manager(ci_model("block_2inch.stl"))
        )
        if seed_offset is not None:
            n = engine.mesh.get_voxel_count()
            engine.set_initial_temperature_field(
                np.full(n, engine.alloy.pouring_temp + seed_offset, dtype=np.float64)
            )
        engine.run_simulation(show_progress=False)
        return engine

    def test_hotter_seed_lengthens_solidification(self) -> None:
        baseline = float(np.mean(self._run(None).solidification_times))
        hotter = float(np.mean(self._run(+120.0).solidification_times))

        self.assertGreater(
            hotter,
            baseline,
            msg=(
                "seeding a hotter initial field did not lengthen solidification; "
                "the field is probably not reaching _initialize_thermal_models"
            ),
        )

    def test_seeding_pouring_temp_matches_the_default(self) -> None:
        baseline = self._run(None)
        seeded = self._run(0.0)

        np.testing.assert_allclose(
            np.asarray(seeded.solidification_times, dtype=np.float64),
            np.asarray(baseline.solidification_times, dtype=np.float64),
            rtol=1e-9,
            atol=1e-9,
        )

class ChillOverrideTests(unittest.TestCase):

    @staticmethod
    def _thermal_field_after_seeding(seed_value):
        engine = _build_engine(scenarios._cast_with_chill_manager())
        n = engine.mesh.get_voxel_count()
        if seed_value is not None:
            engine.set_initial_temperature_field(
                np.full(n, seed_value, dtype=np.float64)
            )
        engine._initialize_thermal_models(show_progress=False)
        return engine, np.asarray(engine._thermal_heat_model.temperatures[:n], dtype=np.float64)

    def test_chills_are_forced_to_mold_temperature(self) -> None:
        engine, field = self._thermal_field_after_seeding(1600.0)

        part_types = np.asarray(engine.mesh.voxel_part_types, dtype=np.int32)
        chill_mask = is_chill(part_types)
        self.assertTrue(np.any(chill_mask), "fixture produced no chill voxels")

        chill_temperature = (
            engine.analysis_options.solidification_params.thermal_boundary.chill_temperature
        )
        np.testing.assert_allclose(field[chill_mask], chill_temperature, atol=1e-9)

    def test_metal_keeps_the_seeded_value(self) -> None:
        engine, field = self._thermal_field_after_seeding(1600.0)

        part_types = np.asarray(engine.mesh.voxel_part_types, dtype=np.int32)
        metal_mask = ~is_chill(part_types)
        self.assertTrue(np.any(metal_mask))
        np.testing.assert_allclose(field[metal_mask], 1600.0, atol=1e-9)

    def test_without_a_seed_metal_starts_at_pouring_temp(self) -> None:
        engine, field = self._thermal_field_after_seeding(None)

        part_types = np.asarray(engine.mesh.voxel_part_types, dtype=np.int32)
        metal_mask = ~is_chill(part_types)
        np.testing.assert_allclose(
            field[metal_mask], engine.alloy.pouring_temp, atol=1e-9
        )

if __name__ == "__main__":
    unittest.main()
