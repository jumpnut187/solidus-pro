from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Optional

import numpy as np

from core.materials import get_alloy
from core.solidification import AnalysisOptions, SolidificationEngine
from physics.heat_transfer import BoundaryType, HeatTransferModel

CHILL_ALPHA = 7.83e-5  # graphite thermal diffusivity (m^2/s) wired in solidification.py


def _build_engine(
    *,
    with_chill: bool = False,
    voxel_sizes: Optional[np.ndarray] = None,
) -> SolidificationEngine:
    """Build a tiny 8-voxel engine exercising the thermal-coupled model init.

    Voxel layout: 8 collinear voxels; the last two become CHILL (part type 2)
    when ``with_chill`` is set, otherwise everything is CAST. Every voxel is a
    surface voxel so boundary registration is trivially observable.
    """
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

    mesh.get_surface_voxel_indices = _get_surface_voxel_indices  # type: ignore[attr-defined]

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
        # Per-voxel sizes arrive in meters (inches * 0.0254).
        np.testing.assert_allclose(model._voxel_sizes, np.full(n_voxels, 0.5 * 0.0254), atol=1e-12)

    def test_voxel_sizes_fallback_to_uniform_scalar(self) -> None:
        n_voxels = 8
        engine = _build_engine()  # no mesh.voxel_sizes -> uniform fallback

        engine._initialize_thermal_models(show_progress=False)

        model = engine._thermal_heat_model
        self.assertIsNotNone(model._voxel_sizes)
        self.assertEqual(len(model._voxel_sizes), n_voxels)
        expected = 1.0 * 0.0254  # base_voxel_size = 1.0 inch
        np.testing.assert_allclose(model._voxel_sizes, np.full(n_voxels, expected), atol=1e-12)

    def test_chill_voxels_become_diffusivity_overrides_with_interface_boundary(self) -> None:
        engine = _build_engine(with_chill=True)
        n_voxels = 8

        engine._initialize_thermal_models(show_progress=False)

        model = engine._thermal_heat_model
        chill_indices = np.where(np.asarray(engine.mesh.voxel_part_types) == 2)[0]
        self.assertEqual(len(chill_indices), 2)

        # B4: per-voxel graphite diffusivity overrides cover exactly the chills.
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

        # Chill surface voxels carry an INTERFACE (ambient heat loss) boundary.
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

        # Chills must NOT be pinned with a FIXED_TEMPERATURE boundary.
        fixed_bcs = [bc for bc in model._boundaries if bc.type == BoundaryType.FIXED_TEMPERATURE]
        for bc in fixed_bcs:
            self.assertFalse(
                bool(np.isin(chill_indices, bc.indices).any()),
                "FIXED_TEMPERATURE boundary must not reference chill voxels",
            )

    def test_cast_interface_boundary_has_air_gap_time_function(self) -> None:
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
        tau = float(ap.time_constant)
        h0 = float(tb.htc_cast_interface)

        # h(t) = h0 * (floor + (1-floor)*exp(-t/tau)): at t=0 the decay is 1.0,
        # so the time function returns exactly h0.
        self.assertAlmostEqual(float(bc.time_function(0.0)), h0, places=6)
        # After 10 time constants the exponential term is negligible: h ~ h0*floor.
        h_late = float(bc.time_function(tau * 10.0))
        self.assertAlmostEqual(h_late / (h0 * floor), 1.0, delta=1e-3)

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
        # Reference points must be monotonically increasing.
        self.assertTrue(np.all(np.diff(np.asarray(curves["temp_points"])) > 0))


if __name__ == "__main__":
    unittest.main()
