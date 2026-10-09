"""Cut cells on the voxel grid (core/cut_cells.py, physics/heat_transfer.py, core/voxel_mesh.py).

Geometry: metal fraction phi and exterior area A per cell. Solver: capacity
rho_cp*phi*h^3, Robin flux through A/(phi*h^3), conductance scaling for
stability and merging of small cells. Flag ``settings.mesh.voxel_cut_cells``.
"""

from __future__ import annotations

import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import trimesh

import core.cut_cells as cut_cells
import physics.heat_transfer as heat_transfer
from config.settings import get_settings
from core.cut_cells import (
    compute_cut_cell_geometry,
    dominant_part_codes,
    merge_small_cut_cells,
    sample_metal_fractions,
)
from core.voxel_mesh import AdaptiveVoxelMesh
from physics.heat_transfer import BoundaryType, HeatTransferModel
from physics.units import inches_to_meters

NUMBA = heat_transfer.NUMBA_AVAILABLE


def _lattice_centers(meshes, pitch: float, shift: float = 0.013) -> np.ndarray:
    """Centre-containment voxelization (the chunked path's rule) on a shifted lattice."""
    lo = np.min([m.bounds[0] for m in meshes], axis=0) - pitch
    hi = np.max([m.bounds[1] for m in meshes], axis=0) + pitch
    axes = [np.arange(lo[d] + shift, hi[d], pitch) for d in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    inside = np.zeros(len(grid), dtype=bool)
    for mesh in meshes:
        inside |= mesh.contains(grid)
    return grid[inside]


def _canonical_neighbors(centers: np.ndarray, pitch: float) -> np.ndarray:
    """(N,26) neighbour slots in the core/voxel_mesh.py canonical layout."""
    origin = centers[0]
    ijk = np.rint((centers - origin) / pitch).astype(np.int64)
    index = {tuple(v): i for i, v in enumerate(ijk)}
    out = np.full((len(centers), 26), -1, dtype=np.int32)
    for i, (x, y, z) in enumerate(ijk):
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                for sz in (-1, 0, 1):
                    nz = (sx != 0) + (sy != 0) + (sz != 0)
                    if nz == 0:
                        continue
                    j = index.get((x + sx, y + sy, z + sz))
                    if j is None:
                        continue
                    if nz == 1:
                        slot = {(1, 0, 0): 0, (-1, 0, 0): 1, (0, 1, 0): 2, (0, -1, 0): 3, (0, 0, 1): 4, (0, 0, -1): 5}[(sx, sy, sz)]
                    elif nz == 2:
                        if sz == 0:
                            slot = 6 + ((0 if sx > 0 else 2) | (0 if sy > 0 else 1))
                        elif sy == 0:
                            slot = 10 + ((0 if sx > 0 else 2) | (0 if sz > 0 else 1))
                        else:
                            slot = 14 + ((0 if sy > 0 else 2) | (0 if sz > 0 else 1))
                    else:
                        slot = 18 + ((0 if sx > 0 else 4) | (0 if sy > 0 else 2) | (0 if sz > 0 else 1))
                    out[i, slot] = j
    return out


class _Alloy:
    """Constant-property material with a short time constant."""

    name = "TEST"
    liquidus_temp = 2000.0
    solidus_temp = 1990.0
    latent_heat = 0.0
    density_liquid = 1000.0
    density_solid = 1000.0
    specific_heat = 1000.0
    thermal_conductivity = 400.0
    thermal_diffusivity = 400.0 / (1000.0 * 1000.0)


class MetalFractionTest(unittest.TestCase):
    def test_voxel_cut_by_an_axis_aligned_plane(self) -> None:
        # Unit voxel centred at the origin, metal where x < 0.3: phi = 0.8.
        slab = trimesh.creation.box(bounds=[[-5.0, -5.0, -5.0], [0.3, 5.0, 5.0]])
        phi, counts = sample_metal_fractions(np.zeros((1, 3)), 1.0, [slab])
        self.assertAlmostEqual(float(phi[0]), 0.8, delta=1.0 / 64 + 1e-9)
        self.assertEqual(int(counts[0, 0]), int(round(phi[0] * 64)))

    def test_sphere_volume_and_area(self) -> None:
        sphere = trimesh.creation.icosphere(subdivisions=5, radius=1.0)
        geometry = compute_cut_cell_geometry(_lattice_centers([sphere], 0.1), 0.1, [sphere])
        volume = float(np.sum(geometry.metal_fractions, dtype=np.float64)) * 0.1**3
        area = float(np.sum(geometry.surface_areas, dtype=np.float64))
        self.assertLess(abs(volume / (4.0 / 3.0 * np.pi) - 1.0), 0.01)
        self.assertLess(abs(area / (4.0 * np.pi) - 1.0), 0.02)
        # Centre containment drops cut cells; the pass adds them back.
        self.assertGreater(geometry.n_added, 0)
        self.assertTrue(np.all(geometry.metal_fractions > 0.0))
        self.assertTrue(np.all(geometry.metal_fractions <= 1.0))

    def test_part_interface_faces_are_not_mold_contact(self) -> None:
        a = trimesh.creation.box(bounds=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        b = trimesh.creation.box(bounds=[[1.0, 0.0, 0.0], [2.0, 1.0, 1.0]])
        geometry = compute_cut_cell_geometry(_lattice_centers([a, b], 0.1), 0.1, [a, b])
        # Union is a 2 x 1 x 1 box: area 10, not 12.
        self.assertAlmostEqual(float(np.sum(geometry.surface_areas, dtype=np.float64)), 10.0, delta=0.05)
        self.assertAlmostEqual(float(np.sum(geometry.metal_fractions, dtype=np.float64)) * 1e-3, 2.0, delta=0.01)

    def test_touching_bodies_in_one_stl_are_split(self) -> None:
        # thick_thin_test.stl is two boxes sharing a face in one mesh; ray parity
        # on the combined mesh misclassifies the column behind the shared face.
        lower = trimesh.creation.box(bounds=[[-1.5, -1.5, 0.0], [1.5, 1.5, 2.0]])
        upper = trimesh.creation.box(bounds=[[-0.5, -0.5, 2.0], [0.5, 0.5, 5.0]])
        combined = trimesh.util.concatenate([lower, upper])
        geometry = compute_cut_cell_geometry(_lattice_centers([combined], 0.1), 0.1, [combined])
        self.assertAlmostEqual(float(np.sum(geometry.metal_fractions, dtype=np.float64)) * 1e-3, 21.0, delta=0.05)
        self.assertAlmostEqual(float(np.sum(geometry.surface_areas, dtype=np.float64)), 54.0, delta=0.1)

    def test_dominant_part_code_breaks_ties_toward_the_higher_code(self) -> None:
        counts = np.array([[10, 5, 0], [8, 8, 0], [0, 0, 0], [0, 3, 9]], dtype=np.uint16)
        codes = dominant_part_codes(counts, [0, 1, 2])
        np.testing.assert_array_equal(codes, [0, 1, -1, 2])


class MergeTest(unittest.TestCase):
    def test_merging_preserves_volume_and_area(self) -> None:
        sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.7)
        geometry = compute_cut_cell_geometry(_lattice_centers([sphere], 0.2), 0.2, [sphere])
        neighbors = _canonical_neighbors(geometry.centers.astype(np.float64), 0.2)
        merge = merge_small_cut_cells(geometry.metal_fractions, geometry.surface_areas, neighbors, None, 0.25)
        self.assertGreater(len(merge.slaves), 0)
        self.assertAlmostEqual(merge.capacity_weights.sum(), float(np.sum(geometry.metal_fractions, dtype=np.float64)), places=5)
        self.assertAlmostEqual(merge.boundary_areas.sum(), float(np.sum(geometry.surface_areas, dtype=np.float64)), places=5)
        np.testing.assert_array_equal(merge.capacity_weights[merge.slaves], 0.0)
        np.testing.assert_array_equal(merge.boundary_areas[merge.slaves], 0.0)
        self.assertTrue(np.all(geometry.metal_fractions[merge.masters] >= 0.25))
        self.assertTrue(np.all(merge.capacity_weights[merge.capacity_weights > 0] >= 0.25 - 1e-12) or merge.n_unmerged_small > 0)

    def test_merge_stays_within_one_part_code(self) -> None:
        phi = np.array([0.1, 1.0, 1.0])
        areas = np.array([0.5, 0.0, 0.0])
        neighbors = np.full((3, 26), -1, dtype=np.int32)
        neighbors[0, 0], neighbors[0, 1] = 1, 2
        merge = merge_small_cut_cells(phi, areas, neighbors, np.array([0, 2, 0]), 0.25)
        np.testing.assert_array_equal(merge.masters, [2])


class SolverTest(unittest.TestCase):
    def _cube_model(self, *, merge: bool, htc: float = 50.0):
        # 1 in cube offset from the 0.2 in lattice so the skin cells are cut
        # (smallest phi 0.15 per axis).
        cube = trimesh.creation.box(bounds=[[0.03, 0.03, 0.03], [1.03, 1.03, 1.03]])
        pitch = 0.2
        geometry = compute_cut_cell_geometry(_lattice_centers([cube], pitch, shift=0.0), pitch, [cube])
        neighbors = _canonical_neighbors(geometry.centers.astype(np.float64), pitch)
        phi_min = cut_cells.DEFAULT_PHI_MIN if merge else 0.0
        merged = merge_small_cut_cells(geometry.metal_fractions, geometry.surface_areas, neighbors, None, phi_min)
        n = len(geometry.centers)
        h_m = inches_to_meters(pitch)
        model = HeatTransferModel(
            n_voxels=n,
            voxel_size=h_m,
            neighbors=neighbors,
            alloy=_Alloy(),
            capacity_weights=merged.capacity_weights,
            boundary_areas=inches_to_meters(inches_to_meters(merged.boundary_areas)),
            slave_cells=(merged.slaves, merged.masters),
        )
        surface = np.flatnonzero(merged.boundary_areas > 0)
        model.add_boundary(surface, BoundaryType.CONVECTION, htc=htc, T_infinity=20.0)
        return model, geometry, merged

    def test_lumped_cube_cools_at_the_analytic_rate(self) -> None:
        model, geometry, merged = self._cube_model(merge=True)
        model.set_initial_temperature(1000.0)
        side = inches_to_meters(1.0)
        rho_cp = _Alloy.density_liquid * _Alloy.specific_heat
        tau = rho_cp * side**3 / (50.0 * 6.0 * side**2)  # Bi = 0.0016: lumped
        dt = model.get_stable_timestep()
        steps = int(round(tau / dt))
        for _ in range(steps):
            model.step(dt)
        weights = merged.capacity_weights
        mean_t = float(np.sum(weights * model.temperatures) / np.sum(weights))
        expected = 20.0 + 980.0 * np.exp(-steps * dt / tau)
        self.assertLess(abs(mean_t - expected) / (expected - 20.0), 0.01)
        # Slaves report their master's temperature.
        np.testing.assert_array_equal(model.temperatures[merged.slaves], model.temperatures[merged.masters])

    def test_tiny_cells_stay_stable_and_conserve_energy(self) -> None:
        # 3 x 3 x 3 block, phi 1e-4 on every face-centre cell, no merging.
        centers = np.stack(np.meshgrid(*[np.arange(3.0)] * 3, indexing="ij"), -1).reshape(-1, 3) * 0.1
        neighbors = _canonical_neighbors(centers, 0.1)
        weights = np.ones(len(centers))
        ijk = np.rint(centers / 0.1).astype(int)
        face_centre = np.sum(ijk == 1, axis=1) == 2
        weights[face_centre] = 1e-4
        h_m = inches_to_meters(0.1)
        rng = np.random.default_rng(3)
        model = HeatTransferModel(
            n_voxels=len(centers), voxel_size=h_m, neighbors=neighbors, alloy=_Alloy(),
            capacity_weights=weights, boundary_areas=np.zeros(len(centers)),
        )
        t0 = rng.uniform(100.0, 1500.0, len(centers))
        model.set_initial_temperature(t0)
        energy0 = float(np.sum(weights * t0))
        dt = model.get_stable_timestep()
        for _ in range(400):
            model.step(dt)
        T = model.temperatures
        self.assertTrue(np.all(np.isfinite(T)))
        self.assertGreaterEqual(T.min(), t0.min() - 1e-6)
        self.assertLessEqual(T.max(), t0.max() + 1e-6)
        self.assertAlmostEqual(float(np.sum(weights * T)) / energy0, 1.0, places=9)

    def test_tiny_cell_with_robin_area_stays_bounded(self) -> None:
        model, geometry, merged = self._cube_model(merge=False, htc=5000.0)
        model.set_initial_temperature(1000.0)
        dt = model.get_stable_timestep()
        for _ in range(300):
            model.step(dt)
        T = model.temperatures
        self.assertTrue(np.all(np.isfinite(T)))
        self.assertGreaterEqual(T.min(), 20.0 - 1e-6)
        self.assertLessEqual(T.max(), 1000.0 + 1e-6)

    def test_full_cells_reduce_to_the_plain_model(self) -> None:
        centers = np.stack(np.meshgrid(*[np.arange(4.0)] * 3, indexing="ij"), -1).reshape(-1, 3) * 0.1
        neighbors = _canonical_neighbors(centers, 0.1)
        n = len(centers)
        h_m = inches_to_meters(0.1)
        exposed = 6 - np.sum(neighbors[:, :6] >= 0, axis=1)
        surface = np.flatnonzero(exposed > 0)
        t0 = np.random.default_rng(5).uniform(1300.0, 1600.0, n)
        # Interior S_i = 4 everywhere only for full rows; the conductance
        # factor stays 1 because w = 1 >= S_i / 4 for every cell.
        plain = HeatTransferModel(n, h_m, neighbors, _Alloy(), rho_cp_overrides=(np.array([0]), np.array([1e6])))
        cut = HeatTransferModel(
            n, h_m, neighbors, _Alloy(), rho_cp_overrides=(np.array([0]), np.array([1e6])),
            capacity_weights=np.ones(n), boundary_areas=np.full(n, h_m * h_m),
        )
        for model in (plain, cut):
            model.set_initial_temperature(t0)
            model.add_boundary(surface, BoundaryType.INTERFACE, htc=500.0, T_infinity=20.0)
        # One substep per step on both models (their stable-dt rules differ).
        dt = 0.5 * min(plain.get_stable_timestep(), cut.get_stable_timestep())
        for model in (plain, cut):
            for _ in range(50):
                model.step(dt)
        np.testing.assert_allclose(cut.temperatures, plain.temperatures, rtol=1e-12, atol=1e-9)

    @unittest.skipUnless(NUMBA, "Numba unavailable")
    def test_numba_kernel_matches_numpy_reference(self) -> None:
        rng = np.random.default_rng(11)
        centers = np.stack(np.meshgrid(*[np.arange(6.0)] * 3, indexing="ij"), -1).reshape(-1, 3) * 0.1
        neighbors = _canonical_neighbors(centers, 0.1)
        n = len(centers)
        weights = rng.uniform(0.0, 1.2, n)
        weights[rng.random(n) < 0.1] = 0.0
        apertures = heat_transfer.cut_cell_link_apertures(weights)
        factors = heat_transfer.cut_cell_conductance_factors(weights, neighbors, apertures)
        args = (
            rng.uniform(20.0, 1600.0, n),
            neighbors,
            rng.uniform(5e-6, 2e-5, n),
            rng.uniform(1e6, 5e6, n),
            weights,
            factors,
            apertures,
            np.full(n, 1.0 / inches_to_meters(0.1) ** 2),
            rng.uniform(-1.0, 1.0, n),
            0.05,
        )
        kernel = heat_transfer.update_heterogeneous_cut_cell_parallel(*args)
        reference = heat_transfer.update_heterogeneous_cut_cell_numpy(*args)
        np.testing.assert_allclose(kernel, reference, rtol=1e-12, atol=1e-9)

    def test_conductance_factors_bound_every_cell_by_the_full_cell_rate(self) -> None:
        centers = np.stack(np.meshgrid(*[np.arange(5.0)] * 3, indexing="ij"), -1).reshape(-1, 3) * 0.1
        neighbors = _canonical_neighbors(centers, 0.1)
        weights = np.random.default_rng(2).uniform(0.01, 1.0, len(centers))
        a = heat_transfer.cut_cell_link_apertures(weights)
        f = heat_transfer.cut_cell_conductance_factors(weights, neighbors, a)
        slot_w = heat_transfer._stencil_slot_weights(26)
        safe = np.maximum(neighbors, 0)
        pair = np.where(neighbors >= 0, np.minimum(f[:, None], f[safe]) * 0.5 * (a[:, None] + a[safe]), 0.0)
        rate = (pair * slot_w[None, :]).sum(axis=1) / weights
        self.assertLessEqual(float(rate.max()), 4.0 + 1e-12)


class _Steel:
    """WCB-like alloy for the latent-heat tests (same fields the engine reads)."""

    name = "STEEL"
    liquidus_temp = 1500.0
    solidus_temp = 1450.0
    pouring_temp = 1560.0
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


def _steel_curves() -> dict:
    """The 6-point curves core/solidification.py builds (_alloy_property_curves)."""
    a = _Steel
    ts, tl = a.solidus_temp, a.liquidus_temp
    return {
        "temp_points": np.array([20.0, 0.5 * ts, ts, (ts + tl) / 2, tl, tl + 100]),
        "conductivity": a.thermal_conductivity * np.array([1.0, 0.95, 0.85, 0.77, 0.70, 0.665]),
        "specific_heat": a.specific_heat * np.array([0.85, 0.92, 1.0, 1.05, 1.10, 1.15]),
        "density": np.array(
            [a.density_solid, 0.98 * a.density_solid, 0.96 * a.density_solid,
             (a.density_solid + a.density_liquid) / 2, a.density_liquid, 0.98 * a.density_liquid]
        ),
    }


class LatentHeatTest(unittest.TestCase):
    def test_adiabatic_enthalpy_is_conserved_with_cut_cells(self) -> None:
        """Heat step + exact inversion through solidification, no boundaries: total enthalpy holds."""
        from core.solidification_loop import _invert_enthalpy_kernel, _invert_enthalpy_numpy
        from physics.solidification_physics import SolidificationModel

        sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.6)
        geometry = compute_cut_cell_geometry(_lattice_centers([sphere], 0.1), 0.1, [sphere])
        neighbors = _canonical_neighbors(geometry.centers.astype(np.float64), 0.1)
        merged = merge_small_cut_cells(geometry.metal_fractions, geometry.surface_areas, neighbors, None, 0.25)
        n = len(geometry.centers)
        curves = _steel_curves()
        model = HeatTransferModel(
            n, inches_to_meters(0.1), neighbors, _Steel(), use_temperature_dependent=True,
            property_curves=curves, capacity_weights=merged.capacity_weights,
            slave_cells=(merged.slaves, merged.masters),
        )
        fs_model = SolidificationModel(n_voxels=n, alloy=_Steel())
        # Hot core, cool skin: the inside solidifies late, the skin first.
        radius = np.linalg.norm(geometry.centers.astype(np.float64), axis=1)
        t0 = 1560.0 - 250.0 * (radius / radius.max()) ** 2
        t0[merged.slaves] = t0[merged.masters]
        model.set_initial_temperature(t0)
        fs_model.update(t0.copy(), 0.0)

        cap_curve = curves["density"] * curves["specific_heat"]
        grid = np.linspace(0.0, 1700.0, 170_001)
        sensible = np.concatenate([[0.0], np.cumsum(0.5 * np.diff(grid) * (np.interp(grid[1:], curves["temp_points"], cap_curve) + np.interp(grid[:-1], curves["temp_points"], cap_curve)))])
        l_vol = _Steel.latent_heat * (_Steel.density_liquid + _Steel.density_solid) / 2
        w = merged.capacity_weights

        def enthalpy() -> float:
            T = model.temperatures
            fs = np.asarray(fs_model.fraction_solid, dtype=np.float64)
            return float(np.sum(w * (np.interp(T, grid, sensible) + l_vol * (1.0 - fs))))

        h0 = enthalpy()
        dt = model.get_stable_timestep()
        T_final = np.empty(n)
        for step in range(3000):
            model.step(dt)
            T_post = np.array(model.temperatures, dtype=np.float64)
            fs_prev = np.asarray(fs_model.fraction_solid, dtype=np.float64).copy()
            beta = l_vol / model.volumetric_heat_capacity(T_post)
            invert = _invert_enthalpy_kernel if step % 2 else _invert_enthalpy_numpy
            invert(T_post, fs_prev, T_final, _Steel.liquidus_temp, _Steel.solidus_temp, beta)
            model.temperatures[:] = T_final
            fs_model.update(T_final.copy(), dt)
        fs = np.asarray(fs_model.fraction_solid)
        self.assertGreater(float(fs[w > 0].min()), 0.0)  # every cell went through the mushy zone
        released = l_vol * float(np.sum(w * fs)) + 1.0
        drift = abs(enthalpy() - h0)
        # Lagged rho(T)*cp(T): drift is O(dt); measured 3.8e-4 of the latent released.
        self.assertLess(drift / released, 2e-3)

    @unittest.skipUnless(NUMBA, "Numba unavailable")
    def test_exact_inversion_kernel_matches_numpy_and_tet(self) -> None:
        from core.solidification_loop import _invert_enthalpy_kernel, _invert_enthalpy_numpy
        from core.tet_thermal import invert_tet_enthalpy

        rng = np.random.default_rng(7)
        n = 50_000
        T_post = rng.uniform(1300.0, 1600.0, n)
        fs_prev = np.clip(rng.uniform(-0.2, 1.2, n), 0.0, 1.0)
        beta = rng.uniform(300.0, 600.0, n)
        kernel = np.empty(n)
        reference = np.empty(n)
        _invert_enthalpy_kernel(T_post, fs_prev, kernel, 1500.0, 1450.0, beta)
        _invert_enthalpy_numpy(T_post, fs_prev, reference, 1500.0, 1450.0, beta)
        tet, _ = invert_tet_enthalpy(T_post, fs_prev, beta, 1500.0, 1450.0)
        np.testing.assert_allclose(kernel, reference, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(reference, tet, rtol=1e-12, atol=1e-9)

    @unittest.skipUnless(NUMBA, "Numba unavailable")
    def test_capacity_and_rate_kernels_match_numpy(self) -> None:
        rng = np.random.default_rng(13)
        curves = _steel_curves()
        cap = curves["density"] * curves["specific_heat"]
        T = np.concatenate([rng.uniform(-100.0, 1800.0, 20_000), curves["temp_points"]])
        np.testing.assert_allclose(
            heat_transfer.interpolate_curve_parallel(T, curves["temp_points"], cap),
            np.interp(T, curves["temp_points"], cap), rtol=1e-12,
        )
        centers = np.stack(np.meshgrid(*[np.arange(6.0)] * 3, indexing="ij"), -1).reshape(-1, 3) * 0.1
        neighbors = _canonical_neighbors(centers, 0.1)
        n = len(centers)
        weights = rng.uniform(0.0, 1.3, n)
        weights[rng.random(n) < 0.1] = 0.0
        a = heat_transfer.cut_cell_link_apertures(weights)
        args = (neighbors, rng.uniform(5e-6, 2e-5, n), rng.uniform(1e6, 5e6, n), weights,
                heat_transfer.cut_cell_conductance_factors(weights, neighbors, a), a,
                np.full(n, 1.0 / inches_to_meters(0.1) ** 2))
        np.testing.assert_allclose(
            heat_transfer.cut_cell_conduction_rates_parallel(*args),
            heat_transfer.cut_cell_conduction_rates_numpy(*args), rtol=1e-12,
        )


def _generate(meshes, pitch: float, *, cut: bool, chunked: bool = False) -> AdaptiveVoxelMesh:
    mesh = AdaptiveVoxelMesh(base_voxel_size=pitch, use_chunked_voxelization=chunked)
    combined = trimesh.util.concatenate(meshes)
    bounds = np.asarray(combined.bounds).T
    with mock.patch.object(get_settings().mesh, "voxel_cut_cells", cut), contextlib.redirect_stdout(io.StringIO()):
        assert mesh.generate(combined, bounds, show_progress=False, auto_adjust=False, part_meshes=list(meshes))
    return mesh


class VoxelMeshCutCellTest(unittest.TestCase):
    def _assembly(self):
        cast = trimesh.creation.box(extents=(2.0, 2.0, 1.0))
        cast.apply_translation((0.0, 0.0, 0.5))
        riser = trimesh.creation.cylinder(radius=0.6, height=1.2, sections=48)
        riser.apply_translation((0.0, 0.0, 1.6))
        chill = trimesh.creation.box(extents=(1.0, 1.0, 0.4))
        chill.apply_translation((0.5, 0.0, -0.2))
        return [(cast, 0), (riser, 1), (chill, 2)]

    def test_assembly_volume_matches_cad_on_the_trimesh_path(self) -> None:
        parts = self._assembly()
        meshes = [m for m, _ in parts]
        cad = sum(abs(m.volume) for m in meshes)
        plain = _generate(meshes, 0.2, cut=False)
        cut = _generate(meshes, 0.2, cut=True)
        self.assertFalse(plain.cut_cells)
        self.assertIsNone(plain.voxel_metal_fractions)
        self.assertGreater(float(np.sum(plain.voxel_volumes)) / cad, 1.2)  # plain voxels: ~+40 %
        self.assertTrue(cut.cut_cells)
        self.assertLess(abs(float(np.sum(cut.voxel_volumes, dtype=np.float64)) / cad - 1.0), 0.01)
        np.testing.assert_array_equal(cut.surface_voxels, np.asarray(cut.voxel_surface_areas) > 0)
        with contextlib.redirect_stdout(io.StringIO()):
            cut.assign_part_types(parts)
        codes = np.asarray(cut.voxel_part_types)
        volumes = np.asarray(cut.voxel_volumes, dtype=np.float64)
        # A cell holds one code, so each part can be off by up to half a cell
        # layer over its part-to-part interface area (riser foot 1.13 in^2,
        # chill top 1.0 in^2); on this lattice both interfaces bisect a layer.
        interface_area = {0: 2.13, 1: 1.13, 2: 1.0}
        for (mesh, code) in parts:
            error = abs(volumes[codes == code].sum() - abs(mesh.volume))
            self.assertLessEqual(error, 0.5 * 0.2 * interface_area[code] + 0.02 * abs(mesh.volume), msg=f"code {code}")

    def test_chunked_path_adds_dropped_surface_cells(self) -> None:
        sphere = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
        cad = sphere.volume
        plain = _generate([sphere], 0.04, cut=False, chunked=True)  # 54^3 candidates: chunked
        cut = _generate([sphere], 0.04, cut=True, chunked=True)
        self.assertGreater(cut.get_voxel_count(), plain.get_voxel_count())
        self.assertLess(abs(float(np.sum(cut.voxel_volumes, dtype=np.float64)) / cad - 1.0), 0.005)
        self.assertGreater(cut.cut_cell_stats["added_voxels"], 0)


class EngineVolumeShareTest(unittest.TestCase):
    def test_shrinkage_percentage_is_volume_weighted_with_cut_cells(self) -> None:
        from core.solidification import SolidificationEngine

        mesh = SimpleNamespace(
            voxel_centers=np.zeros((4, 3)),
            voxel_volumes=np.array([1.0, 1.0, 0.1, 0.1]),
            voxel_metal_fractions=np.array([1.0, 1.0, 0.1, 0.1]),
            cut_cells=True,
        )
        engine = SimpleNamespace(
            mesh=mesh,
            affected_voxel_count=2,
            shrinkage_probability=np.array([0.0, 0.0, 0.9, 0.9]),
            porosity_probability=np.zeros(4),
        )
        self.assertAlmostEqual(SolidificationEngine.get_shrinkage_percentage(engine), 100 * 0.2 / 2.2)
        mesh.cut_cells = False
        self.assertAlmostEqual(SolidificationEngine.get_shrinkage_percentage(engine), 50.0)


class FlagOffTest(unittest.TestCase):
    def test_flag_off_never_reaches_cut_cell_code(self) -> None:
        import core.simulation_pipeline as sp
        from core.part_manager import Part, PartManager, PartType

        manager = PartManager()
        for name, (mesh, code) in zip(("cast", "riser", "chill"), VoxelMeshCutCellTest()._assembly()):
            part = Part(id=name, name=name, filepath=f"{name}.stl", trimesh_obj=mesh)
            part.set_part_type({0: PartType.CAST, 1: PartType.RISER, 2: PartType.CHILL}[code])
            manager.parts[name] = part
            manager._part_order.append(name)

        def boom(*_a, **_k):
            raise AssertionError("cut-cell code reached with the flag off")

        inputs = sp.PipelineInputs("WCB", 0.25, 0, mesh_type="voxel",
                                   analysis_options={"shrinkage": True, "hotspot": True, "filling": False})
        with mock.patch.object(get_settings().mesh, "voxel_cut_cells", False), \
                mock.patch.object(cut_cells, "compute_cut_cell_geometry", boom), \
                mock.patch.object(cut_cells, "merge_small_cut_cells", boom), \
                mock.patch.object(heat_transfer, "update_heterogeneous_cut_cell_numpy", boom), \
                mock.patch.object(heat_transfer, "update_heterogeneous_cut_cell_parallel", boom, create=True), \
                contextlib.redirect_stdout(io.StringIO()):
            result = sp.run_pipeline(inputs, part_manager=manager)
        model = result.engine._thermal_heat_model
        self.assertIsNone(model._capacity_weights)
        self.assertIsNone(model._boundary_area_per_volume)
        self.assertFalse(result.voxel_mesh.cut_cells)
        self.assertTrue(np.all(np.isfinite(np.asarray(result.engine.solidification_times)[
            np.asarray(result.voxel_mesh.voxel_part_types) != 2])))


if __name__ == "__main__":
    unittest.main()
