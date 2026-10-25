"""Explicit sand mold, automatic mold box and cores, and the sand database.

Pins the mold stage (core/mold_setup.py), the solver pieces it relies on
(physics/heat_transfer.py: InterfaceLinks, TabulatedBody, the RADIATION
boundary), the sands (core/mold_materials.py), and the end-to-end effect:
the explicit mold freezes a steel cube in the time Chvorinov's rule gives
for real sand, open risers lose heat to the air, and auto cores appear where
a vertical pattern draw cannot form the sand.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import os
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import trimesh

from tests.helpers.ci_models import ci_model
from core.materials import get_alloy
from core.mold_materials import (
    SAND_MATERIALS,
    WATER_LATENT_HEAT,
    WATER_SPECIFIC_HEAT,
    get_sand,
    sand_choices,
    tabulated_body,
)
from core.mold_setup import (
    NEIGHBOR_SLOT_OFFSETS,
    Lattice,
    mask_surface,
    mold_box,
    plan_layout,
    preview,
)
from core.part_manager import (
    PART_TYPE_CAST,
    PART_TYPE_CORE,
    PART_TYPE_MOLD,
    PART_TYPE_POUR_CUP,
    PART_TYPE_RISER,
    PartManager,
    PartType,
    infer_part_type_from_name,
)
from core.simulation_pipeline import PipelineInputs, run_pipeline
from core.solidification_params import CORE_RULE_ENCLOSED, MoldParams, SolidificationParameters
from physics.heat_transfer import (
    STEFAN_BOLTZMANN,
    BoundaryType,
    HeatTransferModel,
    InterfaceLinks,
    TabulatedBody,
)
from tests.test_simulation_scenarios import _memory_monitor_check, _ok_validation


def _box(extents, centre) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(centre)
    return mesh


def _periodic_column(n_layers: int):
    """3 x 3 x n_layers lattice, periodic in x and y, so every cell has its full 26-cell stencil."""
    side = 3

    def cell(x, y, z):
        return ((z * side + y % side) * side + x % side)

    n = side * side * n_layers
    neighbors = np.full((n, 26), -1, dtype=np.int32)
    for z in range(n_layers):
        for y in range(side):
            for x in range(side):
                for slot, (dx, dy, dz) in enumerate(NEIGHBOR_SLOT_OFFSETS):
                    if 0 <= z + dz < n_layers:
                        neighbors[cell(x, y, z), slot] = cell(x + dx, y + dy, z + dz)
    return neighbors, cell


class MoldBoxTests(unittest.TestCase):
    cast = _box((6, 4, 2), (0, 0, 1))
    riser = _box((1.5, 1.5, 2), (0, 0, 3))

    def test_wall_on_the_sides_and_bottom(self) -> None:
        lo, hi = mold_box([(self.cast, PART_TYPE_CAST, False)], MoldParams(wall_thickness_in=3.0))
        np.testing.assert_allclose(lo, [-6, -5, -3])
        np.testing.assert_allclose(hi[:2], [6, 5])
        self.assertAlmostEqual(hi[2], 5.0)

    def test_closed_riser_gets_the_cover(self) -> None:
        params = MoldParams(wall_thickness_in=1.0, closed_riser_cover_in=2.0)
        _, hi = mold_box([(self.cast, PART_TYPE_CAST, False), (self.riser, PART_TYPE_RISER, False)], params)
        self.assertAlmostEqual(hi[2], 4.0 + 2.0)

    def test_open_riser_is_flush_with_the_cope(self) -> None:
        params = MoldParams(wall_thickness_in=4.0, closed_riser_cover_in=2.0)
        _, hi = mold_box([(self.cast, PART_TYPE_CAST, False), (self.riser, PART_TYPE_RISER, True)], params)
        self.assertAlmostEqual(hi[2], 4.0)

    def test_open_riser_below_a_covered_casting_gets_a_shaft(self) -> None:
        low_riser = _box((1.5, 1.5, 1.0), (2.0, 0, 2.5))  # top at 3, casting top at 2
        tall_closed = _box((1.0, 1.0, 3.0), (-2.0, 0, 3.5))  # top at 5 -> cope at 7
        parts = [
            (self.cast, PART_TYPE_CAST, False),
            (low_riser, PART_TYPE_RISER, True),
            (tall_closed, PART_TYPE_RISER, False),
        ]
        params = MoldParams(auto_mold=1, wall_thickness_in=1.0, closed_riser_cover_in=2.0)
        _, hi = mold_box(parts, params)
        self.assertAlmostEqual(hi[2], 7.0)
        layout, _ = preview(parts, [], params, voxel_size=0.25)
        self.assertGreater(int(layout.shaft.sum()), 0)

    def test_pour_cups_are_always_open(self) -> None:
        cup = _box((2, 2, 1), (0, 0, 3.5))
        _, hi = mold_box([(self.cast, PART_TYPE_CAST, False), (cup, PART_TYPE_POUR_CUP, False)], MoldParams())
        self.assertAlmostEqual(hi[2], 4.0)


class CoreRuleTests(unittest.TestCase):
    """A block with a horizontal through-hole, a vertical through-hole and a sealed cavity."""

    @classmethod
    def setUpClass(cls) -> None:
        labels = np.full((40, 30, 30), -1, np.int16)
        labels[5:35, 5:25, 5:15] = PART_TYPE_CAST
        labels[5:35, 13:17, 9:12] = -1  # horizontal hole along x
        labels[26:30, 13:17, :] = -1  # vertical hole
        labels[20:24, 6:9, 6:9] = -1  # sealed cavity
        cls.labels = labels
        cls.lattice = Lattice(origin=np.zeros(3), pitch=0.1, shape=labels.shape)
        cls.open = np.zeros(labels.shape, dtype=bool)

    def _layout(self, rule: int):
        return plan_layout(self.labels, self.open, self.lattice, MoldParams(auto_mold=1, auto_cores=1, core_rule_code=rule))

    def test_draw_rule_cores_the_horizontal_hole_and_the_cavity(self) -> None:
        layout = self._layout(0)
        # Horizontal hole less its crossing with the vertical hole, plus the cavity.
        self.assertEqual(int(layout.auto_core.sum()), 30 * 4 * 3 - 4 * 4 * 3 + 4 * 3 * 3)
        self.assertFalse(layout.auto_core[27, 14, 20], "a vertical through-hole is moldable")

    def test_enclosed_rule_cores_only_the_sealed_cavity(self) -> None:
        layout = self._layout(CORE_RULE_ENCLOSED)
        self.assertEqual(int(layout.auto_core.sum()), 4 * 3 * 3)
        self.assertTrue(layout.auto_core[21, 7, 7])

    def test_mold_and_core_do_not_overlap_the_assembly(self) -> None:
        layout = self._layout(0)
        blocked = self.labels >= 0
        self.assertFalse(np.any(layout.mold & blocked))
        self.assertFalse(np.any(layout.auto_core & blocked))
        self.assertFalse(np.any(layout.mold & layout.auto_core))

    def test_one_cell_gaps_are_neither_sand_nor_air(self) -> None:
        labels = np.full((10, 10, 10), -1, np.int16)
        labels[2:8, 2:8, 2:4] = PART_TYPE_CAST
        labels[2:8, 2:8, 5:7] = PART_TYPE_CAST  # one empty layer at z = 4
        layout = plan_layout(labels, np.zeros_like(labels, bool), Lattice(np.zeros(3), 0.1, labels.shape),
                             MoldParams(auto_mold=1, auto_cores=1))
        self.assertTrue(layout.gap[4, 4, 4])
        self.assertFalse(layout.mold[4, 4, 4])
        self.assertFalse(layout.auto_core[4, 4, 4])

    def test_core_surface_is_closed(self) -> None:
        layout = self._layout(0)
        mesh = mask_surface(layout.auto_core, self.lattice, smooth=0)
        self.assertTrue(mesh.is_watertight)
        self.assertAlmostEqual(mesh.volume, layout.auto_core.sum() * 0.1 ** 3, places=6)


class PartTypeTests(unittest.TestCase):
    def test_mold_code_and_names(self) -> None:
        self.assertEqual(PART_TYPE_MOLD, 10)
        self.assertIs(infer_part_type_from_name("Cope.stl"), PartType.MOLD)
        self.assertIs(infer_part_type_from_name("flask_body"), PartType.MOLD)
        self.assertIs(infer_part_type_from_name("molded_part.stl"), PartType.CAST)
        self.assertIs(infer_part_type_from_name("sand_core.stl"), PartType.CORE)

    def test_mold_parts_are_not_voxelised_with_the_metal(self) -> None:
        manager = PartManager()
        cast_id = manager.load_part(ci_model("block_2inch.stl"), name="cast")
        mold_id = manager.load_part(ci_model("block_2inch.stl"), name="mold")
        assert cast_id is not None and mold_id is not None
        manager.get_part(mold_id).set_part_type(PartType.MOLD)
        manager.get_part(mold_id).set_scale(4.0)
        self.assertEqual(len(manager.get_parts_with_types()), 1)
        self.assertEqual(len(manager.get_mold_meshes()), 1)
        self.assertEqual(len(manager.get_mold_setup_parts()), 1)


class SolverPieceTests(unittest.TestCase):
    alloy = get_alloy("WCB")

    def test_interface_link_matches_the_semi_infinite_sand_solution(self) -> None:
        """Heat into sand behind a 1000 °C face: 2 b dT sqrt(t / pi), b = sqrt(k rho cp)."""
        neighbors, cell = _periodic_column(200)
        n_sand = len(neighbors)
        n = n_sand + 9
        all_neighbors = np.full((n, 26), -1, dtype=np.int32)
        all_neighbors[9:] = np.where(neighbors >= 0, neighbors + 9, -1)
        dx, k, rho_cp = 0.001, 0.8, 1.6e6
        sand = np.arange(9, n)
        model = HeatTransferModel(
            n, dx, all_neighbors, self.alloy,
            tabulated_bodies=[TabulatedBody(sand, np.array([0.0, 2000.0]), np.array([k, k]), np.array([rho_cp, rho_cp]))],
            interface_links=InterfaceLinks(
                np.arange(9), 9 + np.array([cell(i % 3, i // 3, 0) for i in range(9)]),
                np.full(9, dx * dx), np.zeros(9, dtype=int), [1e9], [None],
            ),
        )
        start = np.zeros(n)
        start[:9] = 1000.0
        model.set_initial_temperature(start)
        t = 0.0
        for _ in range(400):
            model.step(0.5)
            model.temperatures[:9] = 1000.0
            t += 0.5
        absorbed = float(np.sum(model.temperatures[9:]) * rho_cp * dx ** 3) / (9 * dx * dx)
        analytic = 2.0 * np.sqrt(k * rho_cp) * 1000.0 * np.sqrt(t / np.pi)
        self.assertAlmostEqual(absorbed / analytic, 1.0, delta=0.01)

    def test_interface_links_conserve_energy(self) -> None:
        neighbors = np.full((2, 26), -1, dtype=np.int32)
        rho_cp = self.alloy.density_liquid * self.alloy.specific_heat
        model = HeatTransferModel(
            2, 0.005, neighbors, self.alloy,
            tabulated_bodies=[TabulatedBody(np.array([1]), np.array([0.0, 2000.0]), np.array([0.7, 0.7]), np.array([1.7e6, 1.7e6]))],
            interface_links=InterfaceLinks(np.array([0]), np.array([1]), np.array([2.5e-5]), np.array([0]), [1000.0], [None]),
        )
        model.set_initial_temperature(np.array([1500.0, 20.0]))
        before = 1500.0 * rho_cp + 20.0 * 1.7e6
        for _ in range(200):
            model.step(0.1)
        T = model.temperatures
        after = T[0] * model.volumetric_heat_capacity()[0] + T[1] * 1.7e6
        self.assertLess(T[0], 1500.0)
        self.assertGreater(T[1], 20.0)
        self.assertAlmostEqual(after / before, 1.0, places=6)

    def test_enthalpy_update_pays_the_moisture_peak_on_a_large_step(self) -> None:
        """Green sand heated past 100 °C in one step still absorbs the water's latent heat."""
        sand = get_sand(1)
        body = tabulated_body(np.array([1]), 1)
        neighbors = np.full((2, 26), -1, dtype=np.int32)
        model = HeatTransferModel(
            2, 0.005, neighbors, self.alloy, tabulated_bodies=[body],
            interface_links=InterfaceLinks(np.array([0]), np.array([1]), np.array([2.5e-5]), np.array([0]), [5000.0], [None]),
        )
        model.set_initial_temperature(np.array([1500.0, 20.0]))
        rho_cp_metal = self.alloy.density_liquid * self.alloy.specific_heat
        model.step(2.0)
        model.step(2.0)
        heat_out = (1500.0 - model.temperatures[0]) * rho_cp_metal
        heat_in = model.tabulated_enthalpy(0, model.temperatures[1:2])[0] - model.tabulated_enthalpy(0, np.array([20.0]))[0]
        self.assertAlmostEqual(heat_in / heat_out, 1.0, places=6)
        if model.temperatures[1] > 104.0:
            latent = sand.moisture * WATER_LATENT_HEAT * sand.bulk_density
            self.assertGreater(heat_in, latent)

    def test_coupled_mold_solver_conserves_energy(self) -> None:
        """Casting heat lost = mold enthalpy gained, with the mold on a slower clock."""
        from physics.mold_coupling import CoupledMoldSolver, MoldPairs

        dx = 0.005
        cast_nb = np.full((2, 26), -1, dtype=np.int32)
        cast_nb[0, 0], cast_nb[1, 1] = 1, 0
        mold_nb = np.full((3, 26), -1, dtype=np.int32)
        mold_nb[0, 0], mold_nb[1, 1], mold_nb[1, 0], mold_nb[2, 1] = 1, 0, 2, 1
        casting = HeatTransferModel(2, dx, cast_nb, self.alloy, use_temperature_dependent=False)
        sand = tabulated_body(np.arange(3), 2)
        mold = HeatTransferModel(3, dx, mold_nb, self.alloy, tabulated_bodies=[sand])
        casting.set_initial_temperature(1500.0)
        mold.set_initial_temperature(20.0)
        pairs = MoldPairs(
            first=np.array([1]), second=np.array([0]), area=np.array([dx * dx]), group=np.array([0]),
            stencil_factor=np.array([0.0]), htc=[1000.0], temperature_factors=[None],
        )
        solver = CoupledMoldSolver(casting, mold, pairs, mold_interval=0.5)
        rho_cp = self.alloy.density_liquid * self.alloy.specific_heat

        def energy() -> float:
            metal = float(np.sum(solver.temperatures)) * rho_cp * dx ** 3
            sand_h = float(np.sum(mold.tabulated_enthalpy(0, mold.temperatures))) * dx ** 3
            return metal + sand_h

        before = energy()
        for _ in range(37):
            solver.step(0.1)
        solver.flush()
        self.assertLess(float(solver.temperatures[1]), 1500.0)
        self.assertGreater(float(mold.temperatures[0]), 20.0)
        self.assertAlmostEqual(energy() / before, 1.0, places=9)

    def test_radiation_boundary_flux(self) -> None:
        neighbors = np.full((1, 26), -1, dtype=np.int32)
        model = HeatTransferModel(1, 0.01, neighbors, self.alloy)
        model.set_initial_temperature(1500.0)
        model.add_boundary(np.array([0]), BoundaryType.RADIATION, value=0.8, htc=10.0, T_infinity=20.0)
        dt = 0.01
        model.step(dt)
        rho_cp = self.alloy.density_liquid * self.alloy.specific_heat
        q = 0.8 * STEFAN_BOLTZMANN * ((1500 + 273.15) ** 4 - (20 + 273.15) ** 4) + 10.0 * 1480.0
        self.assertAlmostEqual(1500.0 - model.temperatures[0], dt * q / (rho_cp * 0.01), places=6)


class SandDatabaseTests(unittest.TestCase):
    def test_tables_are_complete_and_positive(self) -> None:
        for code, sand in SAND_MATERIALS.items():
            self.assertEqual(code, sand.code)
            self.assertEqual(len(sand.temp_points), len(sand.conductivity), sand.key)
            self.assertEqual(len(sand.temp_points), len(sand.specific_heat), sand.key)
            self.assertIn(len(sand.density), (1, len(sand.temp_points)), sand.key)
            temps, k, rho_cp = sand.tables()
            self.assertTrue(np.all(np.diff(temps) > 0), sand.key)
            self.assertTrue(np.all(k > 0) and np.all(rho_cp > 0), sand.key)

    def test_chilling_power_order(self) -> None:
        """Chromite and zircon chill harder than bonded silica; ceramic beads do not."""
        silica = get_sand(2).thermal_effusivity(600.0)
        self.assertGreater(get_sand(5).thermal_effusivity(600.0), 1.15 * silica)
        self.assertGreater(get_sand(6).thermal_effusivity(600.0), 1.15 * silica)
        self.assertLess(get_sand(4).thermal_effusivity(600.0), 1.05 * silica)

    def test_green_sand_carries_its_water(self) -> None:
        """Wet minus dry enthalpy to 200 °C is the water: latent heat plus its sensible heat to boiling."""
        wet = get_sand(1)
        dry = dataclasses.replace(wet, moisture=0.0)

        def enthalpy_to_200(sand):
            temps, _, rho_cp = sand.tables()
            keep = temps <= 200.0
            return np.trapezoid(rho_cp[keep], temps[keep])

        water = wet.moisture * wet.bulk_density * (WATER_LATENT_HEAT + WATER_SPECIFIC_HEAT * 80.0)
        self.assertAlmostEqual((enthalpy_to_200(wet) - enthalpy_to_200(dry)) / water, 1.0, delta=0.01)

    def test_choices_start_with_custom(self) -> None:
        self.assertEqual(sand_choices()[0][0], 0)
        self.assertEqual(len(sand_choices()), len(SAND_MATERIALS) + 1)

    def test_params_round_trip(self) -> None:
        params = SolidificationParameters.from_dict({"mold": {"auto_mold": 1, "mold_material_code": 6, "core_rule_code": 7}})
        self.assertEqual(params.mold.core_rule_code, 1)
        self.assertEqual(SolidificationParameters.from_dict(params.to_dict()).mold, params.mold)


def _run(manager: PartManager, mold: dict | None, voxel_size: float = 0.2):
    inputs = PipelineInputs(
        alloy_name="WCB",
        voxel_size=voxel_size,
        refinement_level=0,
        memory_mode="conservative",
        analysis_options={"shrinkage": True, "hotspot": True},
        max_voxels_for_visualization=5_000,
        solidification_params=None if mold is None else {"mold": mold},
    )
    with ExitStack() as stack:
        stack.enter_context(patch("core.voxel_mesh.auto_adjust_voxel_size", return_value=(voxel_size, False, "")))
        stack.enter_context(patch("core.voxel_mesh.validate_mesh_parameters", side_effect=_ok_validation))
        stack.enter_context(patch("core.voxel_mesh.MemoryMonitor.start", lambda self: None))
        stack.enter_context(patch("core.voxel_mesh.MemoryMonitor.check", _memory_monitor_check))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        return run_pipeline(inputs, part_manager=manager).engine


def _cube(with_riser: bool = False, open_top: bool = False) -> PartManager:
    manager = PartManager()
    cast = manager.get_part(manager.load_part(ci_model("block_2inch.stl"), name="cast"))
    cast.set_part_type(PartType.CAST)
    cast.set_position(0.0, 0.0, 0.0)
    if with_riser:
        riser = manager.get_part(manager.load_part(ci_model("cylinder.stl"), name="riser"))
        riser.set_part_type(PartType.RISER)
        riser.set_position(0.0, 0.0, cast.get_bounds()[2][1] - riser.get_bounds()[2][0])
        riser.is_open_top = open_top
    return manager


class ExplicitMoldPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.legacy = _run(_cube(), {"auto_mold": 0, "auto_cores": 0})
        cls.explicit = _run(_cube(), {"auto_mold": 1, "wall_thickness_in": 2.0})

    def test_metal_sand_contact_area_is_the_cube_surface(self) -> None:
        extension = self.explicit._mold_extension
        self.assertIsNotNone(extension)
        self.assertAlmostEqual(float(extension.pair_area.sum()), 24.0, places=3)

    def test_mesh_arrays_stay_mesh_sized(self) -> None:
        n_mesh = len(self.explicit.mesh.voxel_centers)
        self.assertEqual(len(self.explicit.solidification_end_times), n_mesh)
        self.assertEqual(len(self.explicit._thermal_latest_temperatures), n_mesh)
        self.assertEqual(len(self.explicit._thermal_heat_model.temperatures), n_mesh)
        self.assertEqual(len(self.explicit._thermal_heat_model.mold_temperatures), self.explicit._mold_extension.n_mold)

    def test_the_mold_heats_next_to_the_casting(self) -> None:
        extension = self.explicit._mold_extension
        mold_t = self.explicit._thermal_heat_model.mold_temperatures
        self.assertGreater(float(mold_t.max()), 800.0)

    def test_training_records_say_which_mold_they_ran(self) -> None:
        from core.surrogate.dataset import capture_run, dataset_summary, load_record

        with tempfile.TemporaryDirectory() as out_dir:
            explicit = capture_run(self.explicit, voxel_size=0.2, resolution=16, out_dir=out_dir)
            boundary = capture_run(self.legacy, voxel_size=0.2, resolution=16, out_dir=out_dir)
            self.assertTrue(load_record(explicit)["explicit_mold"])
            self.assertFalse(load_record(boundary)["explicit_mold"])
            self.assertEqual([r["explicit_mold"] for r in dataset_summary(out_dir)["runs"]], ["1", "0"])

    def test_explicit_mold_is_the_default(self) -> None:
        self.assertIsNotNone(_run(_cube(), None)._mold_extension)

    def test_freeze_time_is_chvorinov_for_real_sand(self) -> None:
        """2 in WCB cube: Chvorinov with hot PUNB silica (b^2 1.0-1.4e6) gives 180-252 s."""
        end = float(np.nanmax(self.explicit.solidification_end_times))
        self.assertGreater(end, 180.0)
        self.assertLess(end, 290.0)
        self.assertGreater(end, float(np.nanmax(self.legacy.solidification_end_times)))


class RiserTopTests(unittest.TestCase):
    def test_an_open_riser_freezes_before_a_closed_one(self) -> None:
        mold = {"auto_mold": 1, "wall_thickness_in": 2.0}
        closed = _run(_cube(with_riser=True, open_top=False), mold)
        opened = _run(_cube(with_riser=True, open_top=True), mold)

        def riser_end(engine):
            types = np.asarray(engine.mesh.voxel_part_types)
            return float(np.nanmax(np.asarray(engine.solidification_end_times)[types == PART_TYPE_RISER]))

        self.assertLess(riser_end(opened), riser_end(closed))
        self.assertGreater(len(opened._mold_extension.air_mesh_cells), 0)


class RiserSleeveTests(unittest.TestCase):
    def test_sleeved_riser_extends_the_modulus_by_about_1_3(self) -> None:
        """MEF = sqrt(t_riser / t_cast) of one 2 x 3 in cylinder, typed riser and then casting."""

        def freeze(part_type):
            manager = PartManager()
            part = manager.get_part(manager.load_part(ci_model("cylinder.stl"), name="c"))
            part.set_part_type(part_type)
            return float(np.nanmax(_run(manager, {}).solidification_end_times))

        mef = np.sqrt(freeze(PartType.RISER) / freeze(PartType.CAST))
        self.assertGreater(mef, 1.25)
        self.assertLess(mef, 1.36)


class AutoCorePipelineTests(unittest.TestCase):
    def test_horizontal_tunnel_is_cored_by_the_draw_rule_only(self) -> None:
        tmp = tempfile.mkdtemp()
        slabs = [((4, 3, 1), (0, 0, 0.5)), ((4, 3, 1), (0, 0, 2.5)), ((4, 1, 1), (0, -1, 1.5)), ((4, 1, 1), (0, 1, 1.5))]

        def manager():
            m = PartManager()
            for n, (extents, centre) in enumerate(slabs):
                path = os.path.join(tmp, f"slab{n}.stl")
                _box(extents, centre).export(path)
                m.get_part(m.load_part(path, name=f"cast{n}")).set_part_type(PartType.CAST)
            return m

        drawn = _run(manager(), {"auto_cores": 1})
        enclosed = _run(manager(), {"auto_cores": 1, "core_rule_code": CORE_RULE_ENCLOSED})
        cores = int(np.sum(np.asarray(drawn.mesh.voxel_part_types) == PART_TYPE_CORE))
        self.assertGreater(cores, 300)  # the 4 x 1 x 1 in tunnel is 500 cells at 0.2 in
        self.assertEqual(int(np.sum(np.asarray(enclosed.mesh.voxel_part_types) == PART_TYPE_CORE)), 0)
        np.testing.assert_array_equal(
            np.asarray(drawn.shrinkage_probability)[np.asarray(drawn.mesh.voxel_part_types) == PART_TYPE_CORE], 0.0
        )


class SuppliedGeometryOverrideTests(unittest.TestCase):
    def test_a_loaded_core_part_turns_auto_cores_off(self) -> None:
        from core.mold_setup import auto_cores_apply

        cast = _box((4, 3, 3), (0, 0, 1.5))
        core = _box((1, 1, 1), (0, 0, 1.5))
        params = MoldParams()
        self.assertTrue(auto_cores_apply(params, [(cast, PART_TYPE_CAST, False)]))
        self.assertFalse(auto_cores_apply(params, [(cast, PART_TYPE_CAST, False), (core, PART_TYPE_CORE, False)]))

    def test_a_loaded_mold_part_replaces_the_box(self) -> None:
        cast = _box((2, 2, 2), (0, 0, 1))
        flask = _box((10, 10, 10), (0, 0, 2))
        layout, meshes = preview([(cast, PART_TYPE_CAST, False)], [flask], MoldParams(), voxel_size=0.25)
        self.assertTrue(layout.from_parts)
        self.assertEqual(meshes["mold"], [])  # the loaded flask is drawn as itself
        lo, hi = layout.lattice.bounds()
        self.assertAlmostEqual(float(hi[2] - lo[2]), 10.0, delta=0.3)


class WebMoldRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        from pathlib import Path

        from flask import Flask

        from web import repository as repo
        from web.routes import api_bp
        from web.services.simulation_queue import SimulationQueueManager

        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        database = root / "web.db"
        repo.init_db(database)
        app = Flask(__name__)
        app.config["DATABASE"] = database
        app.config["UPLOAD_FOLDER"] = root
        app.config["TEMP_FOLDER"] = root
        app.extensions["simulation_queue_manager"] = SimulationQueueManager()
        app.register_blueprint(api_bp)
        self.client = app.test_client()
        self.project_id = self.client.post("/api/projects", json={"name": "Mold", "alloy": "WCB"}).get_json()["id"]
        import shutil

        shutil.copy(ci_model("block_2inch.stl"), root / "block_2inch.stl")
        repo.add_part(database, self.project_id, {
            "name": "cast",
            "part_type": "cast",
            "filepath": str(root / "block_2inch.stl"),
            "visible": True,
        }, upload_root=root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_sands_list_custom_first(self) -> None:
        sands = self.client.get("/api/sands").get_json()
        self.assertEqual(sands[0]["code"], 0)
        self.assertEqual({s["code"] for s in sands[1:]}, set(SAND_MATERIALS))

    def test_mold_preview_returns_the_box(self) -> None:
        off = self.client.post(
            f"/api/projects/{self.project_id}/mold-preview", json={"mold": {"auto_mold": 0, "auto_cores": 0}}
        ).get_json()
        self.assertEqual(off["mold"], [])
        on = self.client.post(
            f"/api/projects/{self.project_id}/mold-preview",
            json={"mold": {"auto_mold": 1, "wall_thickness_in": 2.0}, "voxel_size": 0.25},
        ).get_json()
        self.assertEqual(len(on["mold"]), 1)
        self.assertGreater(on["summary"]["mold_cells"], 0)
        vertices = np.asarray(on["mold"][0]["vertices"]).reshape(-1, 3)
        self.assertGreater(float(np.ptp(vertices[:, 0])), 5.5)


if __name__ == "__main__":
    unittest.main()
