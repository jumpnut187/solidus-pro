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

    def test_clearance_gap_between_parts_is_not_mold_contact(self) -> None:
        # review_cut_cells.md P2: a 0.01 in CAD gap at 0.1 in is bridged by
        # lattice conduction, so its two faces must not also lose heat to sand.
        a = trimesh.creation.box(bounds=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        b = trimesh.creation.box(bounds=[[1.01, 0.0, 0.0], [2.01, 1.0, 1.0]])
        geometry = compute_cut_cell_geometry(_lattice_centers([a, b], 0.1), 0.1, [a, b])
        self.assertAlmostEqual(float(np.sum(geometry.surface_areas, dtype=np.float64)), 10.0, delta=0.05)
        self.assertAlmostEqual(float(np.sum(geometry.metal_fractions, dtype=np.float64)) * 1e-3, 2.0, delta=0.01)

    def test_part_with_an_open_piece_stays_one_unit(self) -> None:
        # review_cut_cells.md P2: an open piece's volume depends on the origin,
        # so it must not be split off and tested alone.
        # A sphere with its cap cut off: Trimesh.split's repair cannot close the
        # hole (it does close a missing box face), and the volume is > 0.
        closed = trimesh.creation.box(bounds=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        open_cap = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
        open_cap.apply_translation((3.0, 3.0, 3.0))
        open_cap.update_faces(open_cap.triangles_center[:, 2] < 3.2)
        open_cap.remove_unreferenced_vertices()
        self.assertFalse(open_cap.is_watertight)
        self.assertGreater(open_cap.volume, 0.0)  # the pre-fix split rule would accept it
        units, owner = cut_cells.split_into_bodies([trimesh.util.concatenate([closed, open_cap])])
        self.assertEqual(len(units), 1)
        np.testing.assert_array_equal(owner, [0])
        two_closed = trimesh.util.concatenate([closed, trimesh.creation.box(bounds=[[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])])
        self.assertEqual(len(cut_cells.split_into_bodies([two_closed])[0]), 2)
        # One missing triangle: split's repair closes it, so the part is used closed
        # whether it arrives alone or inside an assembly mesh
        # (tests/test_voxel_sealed_pockets.py::test_open_part_keeps_the_fill).
        holed = trimesh.creation.box(bounds=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        holed.update_faces(np.arange(len(holed.faces)) != 0)
        self.assertFalse(holed.is_watertight)
        units, _ = cut_cells.split_into_bodies([holed])
        self.assertEqual(len(units), 1)
        self.assertTrue(units[0].is_watertight)

    def test_open_mesh_containment_stays_on_the_winding_number(self) -> None:
        # review_cut_cells.md P2: above WINDING_NUMBER_MAX_PAIRS the winding
        # number fell back to the ray test; contains() now chunks the points.
        import core.tet_mesh as tet_mesh

        open_box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        open_box.update_faces(open_box.face_normals[:, 2] < 0.5)  # drop the top face
        self.assertFalse(open_box.is_watertight)
        points = np.random.default_rng(5).uniform(-0.8, 0.8, (500, 3))
        expected = tet_mesh.winding_number_contains(open_box, points)
        cap = 30 * len(open_box.faces)
        calls = []
        original = tet_mesh.winding_number_contains

        def spy(mesh, query, **kwargs):
            calls.append(len(query))
            return original(mesh, query, **kwargs)

        with mock.patch.object(tet_mesh, "WINDING_NUMBER_MAX_PAIRS", cap), \
                mock.patch.object(tet_mesh, "winding_number_contains", spy), \
                self.assertNoLogs(tet_mesh.logger, level="WARNING"):
            actual = cut_cells.contains(open_box, points)
        np.testing.assert_array_equal(actual, expected)
        self.assertGreater(len(calls), 1)
        self.assertTrue(all(n * len(open_box.faces) <= cap for n in calls))

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

    def test_chunked_path_off_pitch_extent_has_no_duplicate_cells(self) -> None:
        # Padded extent 5.13 in at 0.1 in: 52 chunked centres per axis spaced
        # 0.0986 in, so from k ~ 37 two centres round to one lattice cell.
        box = trimesh.creation.box(extents=(5.03, 5.03, 5.03))
        cad = box.volume
        cut = _generate([box], 0.1, cut=True, chunked=True)  # 52^3 candidates: chunked
        centers = np.asarray(cut.voxel_centers, dtype=np.float64)
        self.assertGreater(cut.cut_cell_stats["duplicate_voxels"], 0)
        self.assertEqual(len(np.unique(np.rint(centers / 0.1).astype(np.int64), axis=0)), len(centers))
        self.assertLess(abs(float(np.sum(cut.voxel_volumes, dtype=np.float64)) / cad - 1.0), 0.002)


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


class _MeshSettingsWithoutCutCellFlag:
    """``settings.mesh`` as it was before the flag existed: ``voxel_cut_cells`` raises AttributeError."""

    def __init__(self, inner) -> None:
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name: str):
        if name == "voxel_cut_cells":
            raise AttributeError(name)
        return getattr(self._inner, name)


def _output_arrays(engine) -> dict:
    """Every ndarray the engine, its mesh and its heat model hold (dict values one level down)."""
    out = {}
    sources = {"engine": engine, "mesh": engine.mesh, "heat": getattr(engine, "_thermal_heat_model", None)}
    for prefix, obj in sources.items():
        if obj is None:
            continue
        for name, value in vars(obj).items():
            if isinstance(value, np.ndarray):
                out[f"{prefix}.{name}"] = value
            elif isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(item, np.ndarray):
                        out[f"{prefix}.{name}[{key!r}]"] = item
    return out


class FlagOffBitIdentityTest(unittest.TestCase):
    def test_flag_off_matches_a_run_without_the_flag_bit_for_bit(self) -> None:
        """tests/test_simulation_scenarios.py cast + chill + riser, flag False vs no flag attribute."""
        from tests.test_simulation_scenarios import (
            _cast_with_chill_and_riser_manager,
            _run_pipeline_with_patched_memory,
        )

        settings = get_settings()
        absent = _MeshSettingsWithoutCutCellFlag(settings.mesh)
        self.assertFalse(hasattr(absent, "voxel_cut_cells"))
        with mock.patch.object(settings.mesh, "voxel_cut_cells", False), \
                contextlib.redirect_stdout(io.StringIO()):
            off = _output_arrays(_run_pipeline_with_patched_memory(_cast_with_chill_and_riser_manager()))
        with mock.patch.object(settings, "mesh", absent), contextlib.redirect_stdout(io.StringIO()):
            unset = _output_arrays(_run_pipeline_with_patched_memory(_cast_with_chill_and_riser_manager()))

        self.assertEqual(sorted(off), sorted(unset))
        self.assertIn("engine.solidification_times", off)
        self.assertIn("heat._temperatures", off)
        self.assertNotIn("mesh.voxel_metal_fractions", off)
        for name in off:
            a, b = off[name], unset[name]
            self.assertEqual((a.dtype, a.shape), (b.dtype, b.shape), msg=name)
            if a.dtype == object:
                self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a.ravel(), b.ravel())), msg=name)
            else:
                self.assertEqual(np.ascontiguousarray(a).tobytes(), np.ascontiguousarray(b).tobytes(), msg=name)


@unittest.skipUnless(NUMBA, "Numba unavailable")
class CutCellKernelParityTest(unittest.TestCase):
    """Cut-cell Numba kernels vs their NumPy references (tests/test_parallel_kernel_parity.py pattern)."""

    def test_update_kernel_matches_numpy_for_truncated_stencils(self) -> None:
        # 6 slots (faces only), 21 slots (odd corner remainder), 26 slots.
        rng = np.random.default_rng(17)
        centers = np.stack(np.meshgrid(*[np.arange(5.0)] * 3, indexing="ij"), -1).reshape(-1, 3) * 0.1
        full = _canonical_neighbors(centers, 0.1)
        n = len(centers)
        for slots in (6, 21, 26):
            neighbors = np.ascontiguousarray(full[:, :slots])
            weights = rng.uniform(0.0, 1.2, n)
            weights[rng.random(n) < 0.1] = 0.0
            a = heat_transfer.cut_cell_link_apertures(weights)
            f = heat_transfer.cut_cell_conductance_factors(weights, neighbors, a)
            args = (rng.uniform(20.0, 1600.0, n), neighbors, rng.uniform(5e-6, 2e-5, n), rng.uniform(1e6, 5e6, n),
                    weights, f, a, np.full(n, 1.0 / inches_to_meters(0.1) ** 2), rng.uniform(-1.0, 1.0, n), 0.05)
            np.testing.assert_allclose(
                heat_transfer.update_heterogeneous_cut_cell_parallel(*args),
                heat_transfer.update_heterogeneous_cut_cell_numpy(*args), rtol=1e-12, atol=1e-9, err_msg=f"{slots} slots",
            )
            rate_args = args[1:8]
            np.testing.assert_allclose(
                heat_transfer.cut_cell_conduction_rates_parallel(*rate_args),
                heat_transfer.cut_cell_conduction_rates_numpy(*rate_args), rtol=1e-12, err_msg=f"{slots} slots",
            )

    def test_interpolation_kernel_at_breakpoints_and_non_finite(self) -> None:
        curves = _steel_curves()
        points = curves["temp_points"]
        cap = curves["density"] * curves["specific_heat"]
        T = np.concatenate([points, np.nextafter(points, -np.inf), np.nextafter(points, np.inf),
                            np.array([np.nan, np.inf, -np.inf])])
        actual = heat_transfer.interpolate_curve_parallel(T, points, cap)
        expected = np.interp(T, points, cap)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0, equal_nan=True)

    def _stepped_model(self, *, use_numba: bool):
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.45)
        geometry = compute_cut_cell_geometry(_lattice_centers([sphere], 0.1), 0.1, [sphere])
        neighbors = _canonical_neighbors(geometry.centers.astype(np.float64), 0.1)
        merged = merge_small_cut_cells(geometry.metal_fractions, geometry.surface_areas, neighbors, None, 0.25)
        n = len(geometry.centers)
        with mock.patch.object(heat_transfer, "NUMBA_AVAILABLE", use_numba):
            model = HeatTransferModel(
                n, inches_to_meters(0.1), neighbors, _Steel(), use_temperature_dependent=True,
                property_curves=_steel_curves(), capacity_weights=merged.capacity_weights,
                boundary_areas=inches_to_meters(inches_to_meters(merged.boundary_areas)),
                slave_cells=(merged.slaves, merged.masters),
            )
            model.add_boundary(np.flatnonzero(merged.boundary_areas > 0), BoundaryType.INTERFACE,
                               htc=800.0, T_infinity=25.0, time_function=lambda t: 1.0 / (1.0 + t))
            model.set_initial_temperature(np.random.default_rng(23).uniform(1440.0, 1560.0, n))
            dts = []
            for _ in range(40):
                dt = model.get_stable_timestep()
                dts.append(dt)
                model.step(dt)
            capacity = model.volumetric_heat_capacity()
        return np.array(model.temperatures, dtype=np.float64), np.array(dts), capacity

    def test_model_dispatch_matches_numpy_fallback(self) -> None:
        kernel = self._stepped_model(use_numba=True)
        reference = self._stepped_model(use_numba=False)
        for name, a, b in zip(("temperatures", "stable dt", "rho_cp"), kernel, reference):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-9, err_msg=name)

    def _run_engine(self, *, use_numba: bool) -> dict:
        import core.solidification_loop as solidification_loop
        import physics.solidification_physics as solidification_physics
        from tests.test_thermal_boundary_config import _build_engine

        engine = _build_engine(with_chill=True)
        mesh = engine.mesh
        mesh.cut_cells = True
        mesh.voxel_metal_fractions = np.array([1.0, 0.7, 0.45, 0.15, 1.0, 0.9, 1.0, 1.0], dtype=np.float32)
        mesh.voxel_surface_areas = np.array([5.0, 3.6, 3.2, 1.2, 4.0, 4.0, 4.0, 5.0], dtype=np.float32)
        engine.time_step = 0.1
        engine.max_time = 40.0
        patches = [
            mock.patch.object(solidification_physics, "NUMBA_AVAILABLE", use_numba),
            mock.patch.object(solidification_loop, "ENTHALPY_KERNEL_AVAILABLE", use_numba),
            mock.patch.object(solidification_loop, "NUMBA_THREADING_AVAILABLE", use_numba),
            mock.patch.object(heat_transfer, "NUMBA_AVAILABLE", use_numba),
        ]
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            engine._initialize_thermal_models(show_progress=False)
            self.assertIsNotNone(engine._cut_cell_merge)
            self.assertEqual(engine._cut_cell_merge.slaves.tolist(), [3])
            solidification_loop._run_thermal_coupled_integration(engine, show_progress=False, progress_callback=None)
        return {
            "states": np.asarray(engine.mesh.voxel_states).copy(),
            "start": np.asarray(engine.solidification_start_times).copy(),
            "end": np.asarray(engine.solidification_end_times).copy(),
            "local": np.asarray(engine.local_solidification_time).copy(),
            "temps": np.asarray(engine._thermal_latest_temperatures).copy(),
            "time": np.array([engine.time]),
        }

    def test_thermal_loop_end_to_end_with_cut_cells(self) -> None:
        """Engine thermal loop with cut cells: kernels vs NumPy (exact inversion included)."""
        kernel = self._run_engine(use_numba=True)
        reference = self._run_engine(use_numba=False)
        self.assertGreater(float(np.max(kernel["states"])), 0.0)
        for key in kernel:
            np.testing.assert_allclose(kernel[key], reference[key], rtol=1e-9, atol=1e-9, equal_nan=True,
                                       err_msg=f"end-to-end mismatch in {key}")


class SlaveInitialStateTest(unittest.TestCase):
    def test_slaved_cell_starts_at_its_masters_temperature(self) -> None:
        """review_cut_cells.md P2: with a non-uniform initial field the slave starts at the master's T."""
        import core.solidification_loop as solidification_loop
        from tests.test_thermal_boundary_config import _build_engine

        engine = _build_engine(with_chill=True)
        mesh = engine.mesh
        mesh.cut_cells = True
        mesh.voxel_metal_fractions = np.array([1.0, 0.7, 0.45, 0.15, 1.0, 0.9, 1.0, 1.0], dtype=np.float32)
        mesh.voxel_surface_areas = np.array([5.0, 3.6, 3.2, 1.2, 4.0, 4.0, 4.0, 5.0], dtype=np.float32)
        mesh.get_voxel_count = lambda: 8
        engine.set_initial_temperature_field(np.linspace(1500.0, 1600.0, 8))
        engine.time_step = 0.1
        engine.max_time = 5.0
        engine._initialize_thermal_models(show_progress=False)
        merge = engine._cut_cell_merge
        self.assertEqual((merge.slaves.tolist(), merge.masters.tolist()), ([3], [4]))
        temperatures = np.asarray(engine._thermal_heat_model.temperatures)
        self.assertEqual(temperatures[3], temperatures[4])
        self.assertEqual(engine._thermal_latest_temperatures[3], engine._thermal_latest_temperatures[4])
        self.assertNotEqual(temperatures[2], temperatures[4])  # the field itself is untouched elsewhere
        solidification_loop._run_thermal_coupled_integration(engine, show_progress=False, progress_callback=None)
        # Without the copy, step 1 reports the slave's jump to the master's T as cooling.
        self.assertEqual(engine._thermal_peak_cooling_rates[3], engine._thermal_peak_cooling_rates[4])


def _sphere_manager():
    from core.part_manager import Part, PartManager, PartType

    manager = PartManager()
    part = Part(id="sphere", name="sphere", filepath="sphere.stl",
                trimesh_obj=trimesh.creation.icosphere(subdivisions=3, radius=0.6))
    part.set_part_type(PartType.CAST)
    manager.parts["sphere"] = part
    manager._part_order.append("sphere")
    return manager


def _run_sphere(voxel_cut_cells, setting: bool):
    import core.simulation_pipeline as sp

    inputs = sp.PipelineInputs("WCB", 0.2, 0, mesh_type="voxel", voxel_cut_cells=voxel_cut_cells,
                               analysis_options={"shrinkage": True, "hotspot": True, "filling": False})
    with mock.patch.object(get_settings().mesh, "voxel_cut_cells", setting), contextlib.redirect_stdout(io.StringIO()):
        return sp.run_pipeline(inputs, part_manager=_sphere_manager())


class PipelineInputsCutCellsTest(unittest.TestCase):
    def test_inputs_override_the_setting_and_none_follows_it(self) -> None:
        import core.simulation_pipeline as sp

        inputs = sp.PipelineInputs("WCB", 0.2, 0)
        self.assertIsNone(inputs.voxel_cut_cells)
        for setting in (False, True):
            with mock.patch.object(get_settings().mesh, "voxel_cut_cells", setting):
                self.assertIs(sp.resolve_voxel_cut_cells(inputs), setting)
                self.assertIs(sp.resolve_voxel_cut_cells(sp.PipelineInputs("WCB", 0.2, 0, voxel_cut_cells=not setting)),
                              not setting)
        self.assertTrue(_run_sphere(True, False).voxel_mesh.cut_cells)
        self.assertFalse(_run_sphere(False, True).voxel_mesh.cut_cells)
        self.assertTrue(_run_sphere(None, True).voxel_mesh.cut_cells)


class SurrogateRecordCutCellsTest(unittest.TestCase):
    def test_records_and_index_mark_cut_cell_runs(self) -> None:
        """review_cut_cells.md P2: capture_run writes ``cut_cells`` to the record and the index."""
        import csv
        import os
        import tempfile

        from core.surrogate.dataset import INDEX_FILENAME, capture_run, dataset_summary, load_record

        on = _run_sphere(True, False)
        off = _run_sphere(False, False)
        with tempfile.TemporaryDirectory() as tmp:
            # An index written before the cut_cells column: the value lands as a trailing field.
            with open(os.path.join(tmp, INDEX_FILENAME), "w", newline="") as fh:
                fh.write("sample_id,timestamp,alloy,voxel_count,resolution,path,mesh_type\r\n"
                         "aaaaaaaaaaaa,2026-01-01 00:00:00,WCB,125,16,/old/run_a.npz,voxel\r\n")
            on_path = capture_run(on.engine, voxel_size=0.2, resolution=16, out_dir=tmp)
            off_path = capture_run(off.engine, voxel_size=0.2, resolution=16, out_dir=tmp)
            self.assertIsNotNone(on_path)
            self.assertIsNotNone(off_path)
            self.assertTrue(load_record(on_path)["cut_cells"])
            self.assertFalse(load_record(off_path)["cut_cells"])
            runs = dataset_summary(tmp)["runs"]
            self.assertEqual([(r["mesh_type"], r["cut_cells"]) for r in runs],
                             [("voxel", "0"), ("voxel", "1"), ("voxel", "0")])

        with tempfile.TemporaryDirectory() as tmp:
            capture_run(on.engine, voxel_size=0.2, resolution=16, out_dir=tmp)
            with open(os.path.join(tmp, INDEX_FILENAME), newline="") as fh:
                header = next(csv.reader(fh))
            self.assertEqual(header[-2:], ["mesh_type", "cut_cells"])


if __name__ == "__main__":
    unittest.main()
