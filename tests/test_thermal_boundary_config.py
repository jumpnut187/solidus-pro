from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Optional

import numpy as np

from core.materials import get_alloy
from core.solidification import AnalysisOptions, SolidificationEngine
from physics.heat_transfer import BoundaryType, HeatTransferModel

CHILL_ALPHA = 1.17e-4

def _build_engine(
    *,
    with_chill: bool = False,
    voxel_sizes: Optional[np.ndarray] = None,
) -> SolidificationEngine:
    centers = []
    part_types = []
    for i in range(8):
        centers.append([float(i), 0.0, 0.0])
        part_types.append(2 if with_chill and i >= 6 else 0)

    centers_array = np.asarray(centers, dtype=np.float32)
    part_types_array = np.asarray(part_types, dtype=np.int32)
    n_voxels = len(centers_array)
    neighbors = np.full((n_voxels, 26), -1, dtype=np.int32)
    for idx, center in enumerate(centers_array):
        distances = np.linalg.norm(centers_array - center, axis=1)
        adjacent = [
            neighbor_idx
            for neighbor_idx, distance in sorted(
                enumerate(distances),
                key=lambda item: item[1],
            )
            if neighbor_idx != idx and distance <= 1.01
        ]
        neighbors[idx, : len(adjacent[:26])] = adjacent[:26]

    mesh = SimpleNamespace(
        voxel_centers=centers_array,
        voxel_part_types=part_types_array,
        voxel_states=np.zeros(n_voxels, dtype=np.float32),
        voxel_neighbors=neighbors,
        base_voxel_size=1.0,
    )
    if voxel_sizes is not None:
        mesh.voxel_sizes = np.asarray(voxel_sizes, dtype=np.float64)

    def _get_surface_voxel_indices() -> np.ndarray:
        return np.arange(n_voxels, dtype=np.int32)

    mesh.get_surface_voxel_indices = _get_surface_voxel_indices

    engine = SolidificationEngine(mesh, get_alloy("WCB"), AnalysisOptions())
    engine.solidification_times = np.zeros(n_voxels, dtype=np.float32)
    engine.solidification_start_times = np.zeros(n_voxels, dtype=np.float32)
    engine.solidification_end_times = np.zeros(n_voxels, dtype=np.float32)
    engine.local_solidification_time = np.zeros(n_voxels, dtype=np.float32)
    engine.feeding_fraction_solid = np.full(n_voxels, np.nan, dtype=np.float32)
    return engine

class ThermalBoundaryConfigTests(unittest.TestCase):
    def test_heat_model_uses_temperature_dependent_properties_and_voxel_sizes(self) -> None:
        n_voxels = 8
        engine = _build_engine(voxel_sizes=np.full(n_voxels, 0.5))

        engine._initialize_thermal_models(show_progress=False)

        model = engine._thermal_heat_model
        self.assertIsInstance(model, HeatTransferModel)
        self.assertTrue(model.use_temperature_dependent)
        self.assertIsNotNone(model._voxel_sizes)
        self.assertEqual(len(model._voxel_sizes), n_voxels)
        np.testing.assert_allclose(model._voxel_sizes, np.full(n_voxels, 0.5 * 0.0254), atol=1e-12)

    def test_voxel_sizes_fallback_to_uniform_scalar(self) -> None:
        n_voxels = 8
        engine = _build_engine()

        engine._initialize_thermal_models(show_progress=False)

        model = engine._thermal_heat_model
        self.assertIsNotNone(model._voxel_sizes)
        self.assertEqual(len(model._voxel_sizes), n_voxels)
        expected = 1.0 * 0.0254
        np.testing.assert_allclose(model._voxel_sizes, np.full(n_voxels, expected), atol=1e-12)

    def test_chill_voxels_become_diffusivity_overrides_with_interface_boundary(self) -> None:
        engine = _build_engine(with_chill=True)
        n_voxels = 8

        engine._initialize_thermal_models(show_progress=False)

        model = engine._thermal_heat_model
        chill_indices = np.where(np.asarray(engine.mesh.voxel_part_types) == 2)[0]
        self.assertEqual(len(chill_indices), 2)

        overrides = model._diffusivity_overrides
        self.assertIsNotNone(overrides)
        override_idx, override_vals = overrides
        np.testing.assert_array_equal(np.sort(override_idx), np.sort(chill_indices))
        self.assertEqual(override_idx.dtype, np.int64)
        np.testing.assert_allclose(
            override_vals,
            np.full(len(chill_indices), CHILL_ALPHA),
            rtol=1e-9,
            atol=1e-12,
        )

        chill_surface = np.intersect1d(
            chill_indices,
            np.arange(n_voxels, dtype=np.int64),
        )
        interface_bcs = [
            bc
            for bc in model._boundaries
            if bc.type == BoundaryType.INTERFACE and np.isin(chill_surface, bc.indices).all()
        ]
        self.assertTrue(interface_bcs, "Expected an INTERFACE boundary covering the chill surface")
        self.assertEqual(
            float(interface_bcs[0].htc),
            float(engine.analysis_options.solidification_params.thermal_boundary.chill_ambient_htc),
        )

        fixed_bcs = [bc for bc in model._boundaries if bc.type == BoundaryType.FIXED_TEMPERATURE]
        for bc in fixed_bcs:
            self.assertFalse(
                bool(np.isin(chill_indices, bc.indices).any()),
                "FIXED_TEMPERATURE boundary must not reference chill voxels",
            )

    def test_cast_interface_boundary_has_mold_penetration_time_function(self) -> None:
        engine = _build_engine()
        tb = engine.analysis_options.solidification_params.thermal_boundary

        engine._initialize_thermal_models(show_progress=False)

        model = engine._thermal_heat_model
        cast_bcs = [
            bc
            for bc in model._boundaries
            if bc.type == BoundaryType.INTERFACE and float(bc.htc) == float(tb.htc_cast_interface)
        ]
        self.assertEqual(len(cast_bcs), 1, "Expected exactly one cast-surface INTERFACE boundary")
        bc = cast_bcs[0]
        self.assertIsNotNone(bc.time_function)

        ap = engine.analysis_options.solidification_params.air_gap
        floor = float(ap.htc_floor_fraction)
        h0 = float(tb.htc_cast_interface)
        mold_k = float(tb.mold_thermal_conductivity)
        mold_alpha = float(tb.mold_thermal_diffusivity)

        self.assertAlmostEqual(float(bc.time_function(0.0)), 1.0, places=6)

        t = 1.0
        h_pen = mold_k / np.sqrt(np.pi * mold_alpha * t)
        h_eff = 1.0 / (1.0 / h0 + 1.0 / h_pen)
        expected = max(h_eff / h0, floor)
        self.assertAlmostEqual(float(bc.time_function(t)), expected, places=4)

        h_late_mult = float(bc.time_function(1.0e4))
        self.assertAlmostEqual(h_late_mult / floor, 1.0, delta=1e-3)

        t_grid = np.logspace(-2, 4, 100)
        h_grid = np.array([float(bc.time_function(tv)) for tv in t_grid])
        self.assertTrue(np.all(np.diff(h_grid) <= 1e-9))

    def test_property_curves_passed_to_heat_model(self) -> None:
        engine = _build_engine()

        engine._initialize_thermal_models(show_progress=False)

        curves = engine._thermal_heat_model._property_curves
        self.assertIsNotNone(curves)
        self.assertEqual(
            set(curves.keys()),
            {"temp_points", "conductivity", "specific_heat", "density"},
        )
        for key in ("temp_points", "conductivity", "specific_heat", "density"):
            values = np.asarray(curves[key], dtype=np.float64)
            self.assertEqual(values.shape, (6,), f"property curve '{key}' must have 6 points")
            self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(np.diff(np.asarray(curves["temp_points"])) > 0))

class AdaptiveTimestepTests(unittest.TestCase):

    def _init_models(self, *, env_value: Optional[str] = None) -> SolidificationEngine:
        engine = _build_engine()
        engine.time_step = 0.1
        import os as _os
        old = _os.environ.get("CASTSIM_THERMAL_ADAPTIVE_DT")
        if env_value is None:
            _os.environ.pop("CASTSIM_THERMAL_ADAPTIVE_DT", None)
        else:
            _os.environ["CASTSIM_THERMAL_ADAPTIVE_DT"] = env_value
        try:
            engine._initialize_thermal_models(show_progress=False)
        finally:
            if old is None:
                _os.environ.pop("CASTSIM_THERMAL_ADAPTIVE_DT", None)
            else:
                _os.environ["CASTSIM_THERMAL_ADAPTIVE_DT"] = old
        return engine

    def test_large_cfl_timestep_is_used_when_adaptive_enabled(self) -> None:
        engine = self._init_models()
        dt_stable = float(engine._thermal_heat_model.get_stable_timestep())
        self.assertGreater(dt_stable, 0.2)
        self.assertGreaterEqual(engine.time_step, 0.2)
        self.assertLessEqual(engine.time_step, 2.0)

    def test_fixed_timestep_when_adaptive_disabled(self) -> None:
        engine = self._init_models(env_value="0")
        self.assertEqual(engine.time_step, 0.1)

    def test_fixed_timestep_when_cfl_has_no_headroom(self) -> None:
        engine = _build_engine()
        engine.time_step = 0.1
        import os as _os
        import types as _types
        old = _os.environ.get("CASTSIM_THERMAL_ADAPTIVE_DT")
        _os.environ.pop("CASTSIM_THERMAL_ADAPTIVE_DT", None)
        try:
            engine._initialize_thermal_models(show_progress=False)
            engine._thermal_heat_model.get_stable_timestep = _types.MethodType(
                lambda self: 0.05,
                engine._thermal_heat_model,
            )
            engine.time_step = 0.1
            if _os.getenv("CASTSIM_THERMAL_ADAPTIVE_DT", "1").strip().lower() not in ("0", "false", "no"):
                dt_cfl = float(engine._thermal_heat_model.get_stable_timestep())
                if dt_cfl > 0.0 and np.isfinite(dt_cfl):
                    dt_cfl = float(np.clip(dt_cfl, 0.05, 2.0))
                    if dt_cfl >= 2.0 * engine.time_step:
                        engine.time_step = dt_cfl
        finally:
            if old is None:
                _os.environ.pop("CASTSIM_THERMAL_ADAPTIVE_DT", None)
            else:
                _os.environ["CASTSIM_THERMAL_ADAPTIVE_DT"] = old
        self.assertEqual(engine.time_step, 0.1)

if __name__ == "__main__":
    unittest.main()
