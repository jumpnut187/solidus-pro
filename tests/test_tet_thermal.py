"""Tests for core/tet_thermal.py (contract C4 in docs/tet_migration/DESIGN.md).

SolidificationEngine runs its thermal_coupled solve on a TetVolumeMesh built
directly with build_tet_mesh. Meshes use edge length 0.2 in.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import io
import unittest
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence, Tuple
from unittest import mock

import numpy as np
import trimesh

from core.materials import get_alloy
from core.part_manager import (
    PART_TYPE_CAST,
    PART_TYPE_CHILL,
    PART_TYPE_CORE,
    PART_TYPE_FILTER,
    PART_TYPE_RISER,
    PART_TYPE_SPRUE,
)
from core.solidification import (
    CHILL_RHO_CP_J_M3_K,
    CHILL_THERMAL_DIFFUSIVITY_M2_S,
    AnalysisOptions,
    SolidificationEngine,
)
from core.solidification_loop import _apply_enthalpy_correction_numpy
from core.tet_mesh import TetVolumeMesh, build_tet_mesh
from config.settings import MeshSettings
from core.tet_thermal import (
    DEFAULT_TET_MAX_DELTA_T_C,
    DEFAULT_TET_MAX_TIME_STEP_S,
    invert_tet_enthalpy,
    tet_boundary_groups,
    tet_materials,
)
from core.voxel_mesh import AdaptiveVoxelMesh
from physics.solidification_physics import SolidificationModel
from physics.tet_heat_transfer import TetHeatTransferModel, TetMaterial

EDGE_IN = 0.2
#: Voxel pitch for the voxel-path comparison. At 0.2 in the voxel mesh holds
#: 49 % more metal than the CAD (7.97 vs 5.36 in^3); at 0.05 in it holds 10 % more.
VOXEL_COMPARE_IN = 0.05


def _box(extents, center) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def _cylinder(radius, height, center) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
    mesh.apply_translation(center)
    return mesh


CASTING = _box((2.0, 2.0, 1.0), (0.0, 0.0, 0.5))
RISER = _cylinder(0.6, 1.2, (0.0, 0.0, 1.6))
CHILL = _box((2.0, 2.0, 0.5), (0.0, 0.0, -0.25))
RISER_PARTS = [(CASTING, PART_TYPE_CAST), (RISER, PART_TYPE_RISER)]


def _run(
    mesh: Any,
    *,
    alloy=None,
    options: Optional[AnalysisOptions] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> SolidificationEngine:
    engine = SolidificationEngine(mesh, alloy or get_alloy("WCB"), options or AnalysisOptions())
    with contextlib.redirect_stdout(io.StringIO()):
        engine.initialize()
        engine.run_simulation(show_progress=False, progress_callback=progress_callback)
    return engine


def _tet_mesh(parts: Sequence[Tuple[trimesh.Trimesh, int]]) -> TetVolumeMesh:
    return build_tet_mesh(parts, EDGE_IN)


def _fresh(mesh: TetVolumeMesh) -> TetVolumeMesh:
    """Shallow copy with its own voxel_states, which engines write in place."""
    clone = copy.copy(mesh)
    clone.voxel_states = np.zeros(mesh.get_voxel_count(), dtype=np.float32)
    return clone


class TetEngineRiserAssemblyTest(unittest.TestCase):
    """2x2x1 in casting with a riser cylinder on top, default AnalysisOptions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mesh = _tet_mesh(RISER_PARTS)
        cls.engine = _run(cls.mesh)
        cls.n = cls.mesh.get_voxel_count()
        cls.types = np.asarray(cls.mesh.voxel_part_types)

    def test_heat_model_is_tet_and_solver_is_thermal_coupled(self) -> None:
        self.assertIsInstance(self.engine._thermal_heat_model, TetHeatTransferModel)
        self.assertEqual(self.engine._get_solver_mode(), "thermal_coupled")

    def test_engine_arrays_have_node_length_and_voxel_dtypes(self) -> None:
        e = self.engine
        expected = {
            "solidification_times": np.float32,
            "solidification_start_times": np.float32,
            "solidification_end_times": np.float32,
            "local_solidification_time": np.float32,
            "feeding_fraction_solid": np.float32,
            "cooling_rates": np.float32,
            "thermal_gradients": np.float32,
            "_thermal_peak_cooling_rates": np.float32,
            "_thermal_latest_temperatures": np.float32,
            "_thermal_latest_gradients": np.float32,
            "_thermal_niyama_gradient_at_eval": np.float32,
            "_thermal_niyama_cooling_rate_at_eval": np.float32,
            "_thermal_niyama_temperature_at_eval": np.float32,
            "_thermal_niyama_time_at_eval": np.float32,
            "_thermal_niyama_values_at_eval": np.float32,
            "_thermal_niyama_quality": np.int8,
            "_thermal_niyama_sampled": np.bool_,
            "niyama_values": np.float32,
            "niyama_quality": np.int8,
            "shrinkage_probability": np.float32,
            "hotspot_intensity": np.float32,
        }
        for name, dtype in expected.items():
            with self.subTest(array=name):
                array = np.asarray(getattr(e, name))
                self.assertEqual(array.shape, (self.n,))
                self.assertEqual(array.dtype, np.dtype(dtype))
        self.assertEqual(e.mesh.voxel_states.dtype, np.float32)
        self.assertEqual(e.mesh.voxel_states.shape, (self.n,))

    def test_everything_solidifies_and_riser_freezes_last(self) -> None:
        e = self.engine
        self.assertTrue(np.all(e.mesh.voxel_states >= 0.99))
        self.assertFalse(np.any(np.isnan(e.solidification_times)))
        self.assertLess(e.time, e.max_time)
        cast_last = float(np.max(e.solidification_times[self.types == PART_TYPE_CAST]))
        riser_last = float(np.max(e.solidification_times[self.types == PART_TYPE_RISER]))
        self.assertGreater(riser_last, cast_last)
        self.assertAlmostEqual(riser_last, e.time, delta=2.0)

    def test_default_step_control_within_2_percent_of_converged(self) -> None:
        # wave3_enthalpy.md: default (2 s, 15 C) 144.9 s, converged 143.5 s.
        with mock.patch("core.tet_thermal.tet_time_step_limits", return_value=(0.1, 0.5)):
            fine = _run(_fresh(self.mesh))
        self.assertAlmostEqual(self.engine.time / fine.time, 1.0, delta=0.02)

    def test_time_step_and_max_time_are_sane(self) -> None:
        e = self.engine
        self.assertGreaterEqual(e.max_time, 600.0)
        self.assertLessEqual(e.max_time, 14400.0)
        self.assertGreater(float(np.max(self.mesh.voxel_moduli)), 0.3)
        self.assertGreater(e.time_step, 0.0)
        self.assertLessEqual(e.time_step, 2.0)

    def test_start_precedes_end_and_local_time_matches(self) -> None:
        e = self.engine
        start = e.solidification_start_times
        end = e.solidification_end_times
        self.assertTrue(np.all(start <= end))
        np.testing.assert_allclose(e.local_solidification_time, end - start, atol=1e-3)

    def test_niyama_arrays_populated_for_cast_nodes(self) -> None:
        e = self.engine
        cast = self.types == PART_TYPE_CAST
        sampled = np.asarray(e._thermal_niyama_sampled)
        self.assertTrue(np.all(sampled[cast]))
        self.assertFalse(np.any(sampled[~cast]))
        self.assertTrue(np.all(e._thermal_niyama_quality[cast] == 2))
        self.assertTrue(np.all(np.isfinite(e.niyama_values[cast])))
        self.assertTrue(np.all(e.niyama_values[cast] >= 0.0))
        self.assertTrue(np.all(e._thermal_niyama_cooling_rate_at_eval[cast] > 0.0))
        self.assertGreater(float(np.median(e._thermal_niyama_gradient_at_eval[cast])), 0.0)
        # Gradients in degC/mm: a few C/mm at most across a sand-cast steel block.
        self.assertLess(float(np.max(e._thermal_niyama_gradient_at_eval[cast])), 50.0)

    def test_fraction_liquid_history_snapshots_have_node_length(self) -> None:
        e = self.engine
        history = e.fraction_liquid_history
        self.assertGreater(len(history), 10)
        self.assertEqual(len(history), len(e.fraction_liquid_times))
        for snapshot in history:
            self.assertEqual(snapshot.shape, (self.n,))
            self.assertEqual(snapshot.dtype, np.float32)
        self.assertTrue(np.all(history[0] == 0.0))
        self.assertTrue(np.all(history[-1] >= 0.99))
        self.assertEqual(e.fraction_liquid_times[-1], e.time)
        self.assertTrue(np.all(np.diff(e.fraction_liquid_times) >= 0.0))

    def test_feeding_fraction_solid_and_cooling_rates_resolved(self) -> None:
        e = self.engine
        self.assertFalse(np.any(np.isnan(e.feeding_fraction_solid)))
        self.assertTrue(np.all(e._thermal_peak_cooling_rates >= 0.0))
        self.assertGreater(float(np.max(e._thermal_peak_cooling_rates)), 0.0)

    def test_total_time_within_20_percent_of_voxel_path(self) -> None:
        combined = trimesh.util.concatenate([m for m, _ in RISER_PARTS])
        voxel_mesh = AdaptiveVoxelMesh(base_voxel_size=VOXEL_COMPARE_IN)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(
                voxel_mesh.generate(
                    combined,
                    np.asarray(combined.bounds).T,
                    show_progress=False,
                    part_meshes=[m for m, _ in RISER_PARTS],
                )
            )
            voxel_mesh.assign_part_types(RISER_PARTS)
        voxel_engine = _run(voxel_mesh)
        ratio = self.engine.time / voxel_engine.time
        self.assertGreater(ratio, 0.8, f"tet/voxel total time ratio {ratio:.3f}")
        self.assertLess(ratio, 1.2, f"tet/voxel total time ratio {ratio:.3f}")


class TetEngineChillTest(unittest.TestCase):
    """The same casting with and without a graphite chill under its bottom face."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.plain_mesh = _tet_mesh([(CASTING, PART_TYPE_CAST)])
        cls.chill_mesh = _tet_mesh([(CASTING, PART_TYPE_CAST), (CHILL, PART_TYPE_CHILL)])
        cls.plain = _run(cls.plain_mesh)
        cls.chilled = _run(cls.chill_mesh)

    @staticmethod
    def _near_chill_mean(engine: SolidificationEngine) -> float:
        mesh = engine.mesh
        near = (np.asarray(mesh.voxel_part_types) == PART_TYPE_CAST) & (mesh.node_xyz[:, 2] < 0.25)
        return float(np.mean(engine.solidification_times[near]))

    def test_chill_shortens_nearby_solidification(self) -> None:
        plain = self._near_chill_mean(self.plain)
        chilled = self._near_chill_mean(self.chilled)
        self.assertLess(chilled, 0.5 * plain)
        self.assertLess(self.chilled.time, self.plain.time)

    def test_chill_nodes_start_presolid_at_chill_temperature(self) -> None:
        e = self.chilled
        chill = np.asarray(e.mesh.voxel_part_types) == PART_TYPE_CHILL
        self.assertTrue(np.any(chill))
        self.assertTrue(np.all(e.mesh.voxel_states[chill] == 1.0))
        for name in (
            "solidification_start_times",
            "solidification_end_times",
            "local_solidification_time",
            "solidification_times",
        ):
            self.assertTrue(np.all(getattr(e, name)[chill] == 0.0), name)
        self.assertTrue(np.all(e.feeding_fraction_solid[chill] == 1.0))

        fresh = SolidificationEngine(_fresh(self.chill_mesh), get_alloy("WCB"), AnalysisOptions())
        with contextlib.redirect_stdout(io.StringIO()):
            fresh.initialize()
            fresh.set_initial_temperature_field(np.full(fresh.mesh.get_voxel_count(), 1550.0))
            fresh._initialize_thermal_models(show_progress=False)
        temps = fresh._thermal_heat_model.temperatures
        chill_t = fresh.analysis_options.solidification_params.thermal_boundary.chill_temperature
        np.testing.assert_allclose(temps[chill], chill_t)
        np.testing.assert_allclose(temps[~chill], 1550.0)

    def test_boundary_groups_follow_part_codes(self) -> None:
        groups = {g.label: g for g in tet_boundary_groups(self.chilled)}
        self.assertEqual(set(groups), {"cast", "chill"})
        tb = self.chilled.analysis_options.solidification_params.thermal_boundary
        self.assertEqual(groups["cast"].htc, tb.htc_cast_interface)
        self.assertIsNone(groups["cast"].floor)
        self.assertEqual(groups["chill"].htc, tb.chill_ambient_htc)
        self.assertEqual(groups["chill"].floor, tb.chill_htc_floor_fraction)
        n_faces = len(self.chill_mesh.boundary_faces)
        all_faces = np.concatenate([g.face_indices for g in groups.values()])
        self.assertEqual(np.sort(all_faces).tolist(), list(range(n_faces)))

    def test_chill_material_uses_voxel_path_constants(self) -> None:
        metal, chill = tet_materials(self.chilled)
        np.testing.assert_allclose(chill.rho_cp, [CHILL_RHO_CP_J_M3_K])
        np.testing.assert_allclose(
            chill.conductivity, [CHILL_THERMAL_DIFFUSIVITY_M2_S * CHILL_RHO_CP_J_M3_K]
        )
        self.assertEqual(chill.latent_heat_vol, 0.0)
        alloy = self.chilled.alloy
        self.assertAlmostEqual(
            metal.latent_heat_vol,
            alloy.latent_heat * (alloy.density_liquid + alloy.density_solid) / 2.0,
        )
        curves = self.chilled._alloy_property_curves()
        np.testing.assert_allclose(metal.rho_cp, curves["density"] * curves["specific_heat"])
        np.testing.assert_allclose(metal.conductivity, curves["conductivity"])

    def test_more_latent_heat_takes_longer(self) -> None:
        base = get_alloy("WCB")
        low = dataclasses.replace(base, latent_heat=0.5 * base.latent_heat)
        low_run = _run(_fresh(self.plain_mesh), alloy=low)
        self.assertLess(low_run.time, 0.9 * self.plain.time)

    def test_default_step_control_within_2_percent_of_converged(self) -> None:
        # wave3_enthalpy.md: default (2 s, 15 C) 101.3 s, converged 99.9 s.
        with mock.patch("core.tet_thermal.tet_time_step_limits", return_value=(0.1, 0.5)):
            fine = _run(_fresh(self.plain_mesh))
        self.assertLessEqual(fine.time_step, 0.1)
        self.assertAlmostEqual(self.plain.time / fine.time, 1.0, delta=0.02)

    def test_progress_callback_and_cancellation(self) -> None:
        calls = []
        _run(_fresh(self.plain_mesh), progress_callback=lambda c, t, s: calls.append((c, t, s)))
        percents = [c for c, _, _ in calls]
        self.assertEqual(percents, sorted(percents))
        self.assertEqual(calls[-1], (100, 100, "Solidification"))

        class Cancelled(Exception):
            pass

        def cancel_at_20(current: int, total: int, stage: str) -> None:
            if current >= 20:
                raise Cancelled()

        with self.assertRaises(Cancelled):
            _run(_fresh(self.plain_mesh), progress_callback=cancel_at_20)

    def test_schedule_legacy_request_runs_thermal_coupled_and_warns_once(self) -> None:
        engine = SolidificationEngine(
            _fresh(self.plain_mesh), get_alloy("WCB"), AnalysisOptions(solver_mode="schedule_legacy")
        )
        engine._logger = mock.Mock()
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(3):
                self.assertEqual(engine._get_solver_mode(), "thermal_coupled")
        self.assertEqual(engine._logger.warning.call_count, 1)


class TetBoundaryContactCodeTest(unittest.TestCase):
    """Faces against a removed CORE or FILTER take the CORE / gating h0."""

    def test_contact_codes_override_owner(self) -> None:
        owner = np.array([PART_TYPE_CAST, PART_TYPE_CAST, PART_TYPE_SPRUE, PART_TYPE_CHILL], dtype=np.int8)
        contact = np.array([-1, PART_TYPE_CORE, PART_TYPE_FILTER, PART_TYPE_CORE], dtype=np.int8)
        engine = SimpleNamespace(
            mesh=SimpleNamespace(boundary_face_part_types=owner, boundary_face_contact_codes=contact),
            analysis_options=AnalysisOptions(),
        )
        groups = {g.label: g for g in tet_boundary_groups(engine)}
        tb = engine.analysis_options.solidification_params.thermal_boundary
        self.assertEqual(groups["cast"].face_indices.tolist(), [0])
        self.assertEqual(groups["core_contact"].face_indices.tolist(), [1, 3])
        self.assertEqual(groups["core_contact"].htc, tb.htc_cast_interface)
        self.assertEqual(groups["filter_contact"].face_indices.tolist(), [2])
        self.assertEqual(groups["filter_contact"].htc, tb.htc_gating)
        self.assertNotIn("chill", groups)

    def test_unmapped_owner_code_raises(self) -> None:
        engine = SimpleNamespace(
            mesh=SimpleNamespace(
                boundary_face_part_types=np.array([PART_TYPE_CORE], dtype=np.int8),
                boundary_face_contact_codes=np.array([-1], dtype=np.int8),
            ),
            analysis_options=AnalysisOptions(),
        )
        with self.assertRaises(ValueError):
            tet_boundary_groups(engine)


class InvertTetEnthalpyTest(unittest.TestCase):
    """invert_tet_enthalpy against hand-computed single-node cases.

    T_liq 1460, T_sol 1400 (dT 60), beta 300 K. Enthalpy per unit capacity is
    h = T + beta * (1 - fs).
    """

    T_LIQ, T_SOL, BETA = 1460.0, 1400.0, 300.0

    def _invert(self, T_post: float, fs_prev: float) -> Tuple[float, float]:
        T_new, fs_new = invert_tet_enthalpy(
            np.array([T_post]), np.array([fs_prev]), np.array([self.BETA]), self.T_LIQ, self.T_SOL
        )
        h_before = T_post + self.BETA * (1.0 - fs_prev)
        self.assertAlmostEqual(float(T_new[0] + self.BETA * (1.0 - fs_new[0])), h_before, places=9)
        self.assertGreaterEqual(float(fs_new[0]), fs_prev)
        return float(T_new[0]), float(fs_new[0])

    def test_all_liquid_is_unchanged(self) -> None:
        self.assertEqual(self._invert(1500.0, 0.0), (1500.0, 0.0))

    def test_liquid_entering_mushy_zone(self) -> None:
        # T = (1450*60 + 300*1460) / 360 = 1458.333..., fs = (1460 - T) / 60.
        T_new, fs_new = self._invert(1450.0, 0.0)
        self.assertAlmostEqual(T_new, 525000.0 / 360.0, places=9)
        self.assertAlmostEqual(fs_new, (1460.0 - 525000.0 / 360.0) / 60.0, places=12)

    def test_mushy_step(self) -> None:
        # fs_prev 0.5 -> T_p 1430. T = (1420*60 + 300*1430) / 360 = 1428.333...
        T_new, fs_new = self._invert(1420.0, 0.5)
        self.assertAlmostEqual(T_new, 514200.0 / 360.0, places=9)
        self.assertAlmostEqual(fs_new, (1460.0 - 514200.0 / 360.0) / 60.0, places=12)

    def test_step_below_solidus_recalesces_into_mushy_zone(self) -> None:
        # T_post 1395 < T_sol but h = 1545 > T_sol: T = (1395*60 + 300*1430) / 360
        # = 1424.1667, fs 0.5972. The voxel correction caps this node at T_sol.
        T_new, fs_new = self._invert(1395.0, 0.5)
        self.assertAlmostEqual(T_new, 512700.0 / 360.0, places=9)
        self.assertAlmostEqual(fs_new, (1460.0 - 512700.0 / 360.0) / 60.0, places=12)

    def test_crossing_solidus_in_one_step(self) -> None:
        # fs_prev 0.9, T_post 1350: h = 1380 < T_sol, so T = 1380 and fs = 1.
        self.assertEqual(self._invert(1350.0, 0.9), (1380.0, 1.0))

    def test_liquid_crossing_liquidus_and_solidus_in_one_step(self) -> None:
        # fs_prev 0, T_post 1000: h = 1300 < T_sol, all latent released.
        self.assertEqual(self._invert(1000.0, 0.0), (1300.0, 1.0))

    def test_fully_solid_below_solidus_is_unchanged(self) -> None:
        self.assertEqual(self._invert(1300.0, 1.0), (1300.0, 1.0))

    def test_reheated_mushy_node_keeps_fraction_solid(self) -> None:
        # T_post 1440 > T_p 1430: fs never decreases, T stays.
        self.assertEqual(self._invert(1440.0, 0.5), (1440.0, 0.5))

    def test_fraction_solid_matches_solidification_model_update(self) -> None:
        alloy = get_alloy("WCB")
        rng = np.random.default_rng(11)
        n = 2000
        T_post = rng.uniform(alloy.solidus_temp - 200.0, alloy.liquidus_temp + 50.0, n)
        fs_prev = np.where(rng.uniform(size=n) < 0.3, 0.0, rng.uniform(size=n))
        beta = rng.uniform(1.0, 600.0, n)
        T_new, fs_new = invert_tet_enthalpy(
            T_post, fs_prev, beta, alloy.liquidus_temp, alloy.solidus_temp
        )
        np.testing.assert_allclose(
            T_new + beta * (1.0 - fs_new), T_post + beta * (1.0 - fs_prev), rtol=0.0, atol=1e-9
        )
        model = SolidificationModel(n_voxels=n, alloy=alloy)
        model._fraction_solid[:] = fs_prev
        model.update(T_new, 1.0)
        np.testing.assert_array_equal(model.fraction_solid, fs_new)


class TetAdiabaticEnthalpyConservationTest(unittest.TestCase):
    """Adiabatic box, constant properties, dt up to 20 s: sum(C T + L (1 - fs)) is conserved."""

    def test_full_solidification_conserves_enthalpy(self) -> None:
        mesh = _fresh(build_tet_mesh([(CASTING, PART_TYPE_CAST)], 0.25))
        alloy = get_alloy("WCB")
        rho_cp = float(alloy.density_solid * alloy.specific_heat)
        metal = TetMaterial(
            temp_points=np.zeros(1),
            conductivity=np.array([30.0]),
            rho_cp=np.array([rho_cp]),
            latent_heat_vol=float(alloy.latent_heat * (alloy.density_liquid + alloy.density_solid) / 2.0),
        )
        chill = TetMaterial(temp_points=np.zeros(1), conductivity=np.ones(1), rho_cp=np.ones(1))
        # Top half poured hot, bottom half cold: the mean enthalpy lies below
        # the solidus, so the closed box solidifies completely.
        initial = np.where(mesh.node_xyz[:, 2] > 0.5, 1600.0, 100.0)

        engine = SolidificationEngine(mesh, alloy, AnalysisOptions())
        with contextlib.redirect_stdout(io.StringIO()), mock.patch(
            "core.tet_thermal.tet_materials", return_value=[metal, chill]
        ), mock.patch("core.tet_thermal.tet_boundary_groups", return_value=[]), mock.patch(
            "core.tet_thermal.tet_time_step_limits", return_value=(20.0, 1e9)
        ):
            engine.initialize()
            engine.time_step = 20.0  # every step is 20 s: no growth ramp, no rejection
            engine.set_initial_temperature_field(initial)
            engine.run_simulation(show_progress=False)

        heat_model = engine._thermal_heat_model
        capacity = heat_model.node_heat_capacity()
        latent = heat_model.node_latent_capacity()
        fs = np.asarray(engine._thermal_solidification_model.fraction_solid)
        initial_enthalpy = float(np.dot(capacity, initial) + np.sum(latent))
        final_enthalpy = float(np.dot(capacity, heat_model.temperatures) + np.dot(latent, 1.0 - fs))

        self.assertAlmostEqual(final_enthalpy / initial_enthalpy, 1.0, delta=1e-6)
        self.assertTrue(np.all(fs >= 0.99))
        self.assertEqual(engine.time_step, 20.0)
        np.testing.assert_array_equal(engine.mesh.voxel_states, fs.astype(np.float32))


class TetStepControlDefaultsTest(unittest.TestCase):
    def test_fallbacks_match_mesh_settings_defaults(self) -> None:
        defaults = MeshSettings()
        self.assertEqual(DEFAULT_TET_MAX_TIME_STEP_S, defaults.tet_max_time_step_s)
        self.assertEqual(DEFAULT_TET_MAX_DELTA_T_C, defaults.tet_max_delta_t_c)


class PerNodeEnthalpyCorrectionTest(unittest.TestCase):
    """The NumPy correction accepts a per-node beta and matches the scalar form per node."""

    def setUp(self) -> None:
        rng = np.random.default_rng(3)
        self.n = 400
        self.T_liq, self.T_sol = 1500.0, 1450.0
        self.T_post = rng.uniform(1400.0, 1520.0, self.n)
        self.fs_prev = rng.uniform(0.0, 1.0, self.n) * (self.T_post < self.T_liq)
        self.beta = rng.uniform(0.0, 80.0, self.n)

    def test_array_beta_matches_scalar_beta_per_node(self) -> None:
        dT = self.T_liq - self.T_sol
        out = np.empty(self.n)
        _apply_enthalpy_correction_numpy(
            self.T_post, self.fs_prev, out, self.T_liq, self.T_sol, dT, self.beta
        )
        for i in range(0, self.n, 7):
            single = np.empty(1)
            _apply_enthalpy_correction_numpy(
                self.T_post[i : i + 1], self.fs_prev[i : i + 1], single,
                self.T_liq, self.T_sol, dT, float(self.beta[i]),
            )
            self.assertEqual(out[i], single[0])

    def test_mushy_correction_conserves_enthalpy_per_node(self) -> None:
        dT = self.T_liq - self.T_sol
        out = np.empty(self.n)
        _apply_enthalpy_correction_numpy(
            self.T_post, self.fs_prev, out, self.T_liq, self.T_sol, dT, self.beta
        )
        fs_new = np.clip((self.T_liq - out) / dT, 0.0, 1.0)
        mushy = (out > self.T_sol) & (out < self.T_liq) & (fs_new > self.fs_prev) & (self.T_post > self.T_sol)
        self.assertTrue(np.any(mushy))
        # Sensible heat gained (per unit C) equals latent released: dT_i = beta_i * dfs_i.
        np.testing.assert_allclose(
            (out - self.T_post)[mushy], (self.beta * (fs_new - self.fs_prev))[mushy], atol=1e-9
        )

    def test_zero_beta_is_identity(self) -> None:
        out = np.empty(self.n)
        _apply_enthalpy_correction_numpy(
            self.T_post, self.fs_prev, out, self.T_liq, self.T_sol, self.T_liq - self.T_sol, np.zeros(self.n)
        )
        np.testing.assert_array_equal(out, self.T_post)


if __name__ == "__main__":
    unittest.main()
