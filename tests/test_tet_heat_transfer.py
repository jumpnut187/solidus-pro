"""Verification of physics/tet_heat_transfer.py (contract C2).

Meshes are structured Kuhn (Freudenthal) 6-tet splits of a hexahedral grid,
built here so the tests do not depend on core/tet_mesh.py.
"""

from __future__ import annotations

import itertools
import math
import unittest
from unittest import mock

import numpy as np
from scipy.optimize import brentq

from physics import tet_heat_transfer
from physics.tet_heat_transfer import TetHeatTransferModel, TetMaterial


def kuhn_box_mesh(
    counts: tuple[int, int, int], lengths: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (nodes_m (N,3), tets (T,4)) for a box split into 6 tets per hex.

    Every hex uses the same Kuhn split, so the mesh is conforming. Tets are
    reordered to positive orientation.
    """
    nx, ny, nz = counts
    axes = [np.linspace(0.0, length, n + 1) for n, length in zip(counts, lengths)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    nodes = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    def node_id(i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
        return (i * (ny + 1) + j) * (nz + 1) + k

    ci, cj, ck = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    base = np.stack([ci.ravel(), cj.ravel(), ck.ravel()], axis=1)
    tets = []
    for perm in itertools.permutations(range(3)):
        corner = np.zeros(3, dtype=np.int64)
        path = [corner.copy()]
        for axis in perm:
            corner[axis] += 1
            path.append(corner.copy())
        tets.append(
            np.stack([node_id(*(base + offset).T) for offset in path], axis=1)
        )
    tets_arr = np.concatenate(tets, axis=0)
    x = nodes[tets_arr]
    signed = np.linalg.det(x[:, 1:] - x[:, :1])
    negative = signed < 0
    tets_arr[negative, 2], tets_arr[negative, 3] = (
        tets_arr[negative, 3].copy(),
        tets_arr[negative, 2].copy(),
    )
    return nodes, tets_arr


def boundary_faces_of(tets: np.ndarray) -> np.ndarray:
    """Return (F,3) triangles that belong to exactly one tet."""
    local = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    faces = np.sort(tets[:, local].reshape(-1, 3), axis=1)
    unique, counts = np.unique(faces, axis=0, return_counts=True)
    return unique[counts == 1]


def faces_on_plane(nodes: np.ndarray, faces: np.ndarray, axis: int, value: float) -> np.ndarray:
    """Return indices of faces whose three nodes lie on coordinate ``axis == value``."""
    on = np.isclose(nodes[faces][:, :, axis], value, atol=1e-12).all(axis=1)
    return np.flatnonzero(on)


def constant_material(k: float, rho_cp: float, latent: float = 0.0) -> TetMaterial:
    return TetMaterial(np.array([0.0]), np.array([k]), np.array([rho_cp]), latent)


def make_model(
    counts: tuple[int, int, int],
    lengths: tuple[float, float, float],
    materials: list[TetMaterial],
    tet_material: np.ndarray | None = None,
) -> tuple[TetHeatTransferModel, np.ndarray, np.ndarray]:
    nodes, tets = kuhn_box_mesh(counts, lengths)
    faces = boundary_faces_of(tets)
    if tet_material is None:
        tet_material = np.zeros(len(tets), dtype=np.int64)
    model = TetHeatTransferModel(nodes, tets, tet_material, materials, faces)
    return model, nodes, faces


def slab_centre_theta(biot: float, fourier: float, terms: int = 60) -> float:
    """Analytic centre (T-T_inf)/(T0-T_inf) of a plane wall, symmetric convection."""
    total = 0.0
    for n in range(terms):
        lo = n * math.pi + 1e-12
        hi = n * math.pi + 0.5 * math.pi - 1e-12
        zeta = brentq(lambda z: z * math.tan(z) - biot, lo, hi)
        coeff = 4.0 * math.sin(zeta) / (2.0 * zeta + math.sin(2.0 * zeta))
        total += coeff * math.exp(-zeta * zeta * fourier)
    return total


class TetHeatTransferAssemblyTest(unittest.TestCase):
    def test_stiffness_is_symmetric_psd_with_zero_row_sums(self) -> None:
        curve = TetMaterial(
            np.array([0.0, 500.0, 1000.0]),
            np.array([10.0, 40.0, 25.0]),
            np.array([3e6, 3.5e6, 4e6]),
        )
        model, nodes, _ = make_model((3, 2, 2), (0.03, 0.02, 0.02), [curve])
        rng = np.random.default_rng(0)
        model.set_initial_temperature(rng.uniform(0.0, 1000.0, model.n_nodes))
        K = model.stiffness_matrix().toarray()
        scale = np.abs(K).max()
        self.assertLess(np.abs(K - K.T).max(), 1e-12 * scale)
        # Row sums via K.sum: dense matmul on macOS Accelerate emits spurious FP warnings.
        self.assertLess(np.abs(K.sum(axis=1)).max(), 1e-12 * scale)
        self.assertGreaterEqual(np.linalg.eigvalsh(K).min(), -1e-10 * scale)

    def test_linear_field_gives_exact_gradient(self) -> None:
        nodes, tets = kuhn_box_mesh((4, 3, 3), (0.04, 0.03, 0.03))
        rng = np.random.default_rng(1)
        interior = np.all((nodes > 1e-9) & (nodes < np.array([0.04, 0.03, 0.03]) - 1e-9), axis=1)
        nodes = nodes.copy()
        nodes[interior] += rng.uniform(-2e-3, 2e-3, (int(interior.sum()), 3))
        model = TetHeatTransferModel(
            nodes, tets, np.zeros(len(tets), dtype=np.int64),
            [constant_material(30.0, 3e6)], boundary_faces_of(tets),
        )
        slope = np.array([120.0, -45.0, 300.0])
        model.set_initial_temperature(650.0 + nodes @ slope)
        grad = model.nodal_gradient()
        np.testing.assert_allclose(grad, np.broadcast_to(slope, grad.shape), rtol=0, atol=1e-8)
        np.testing.assert_allclose(model.gradient_magnitude(), np.linalg.norm(slope), rtol=1e-12)

    def test_masked_gradient_at_material_interface_is_one_sided(self) -> None:
        # Flat interface at z = 0.02 m between metal (k 25, above) and chill
        # (k 150, below). Flux continuity: G_chill = G_metal * k_m / k_c.
        k_metal, k_chill, g_metal = 25.0, 150.0, 2000.0
        g_chill = g_metal * k_metal / k_chill
        nodes, tets = kuhn_box_mesh((4, 2, 4), (0.04, 0.02, 0.04))
        metal_tets = nodes[tets].mean(axis=1)[:, 2] > 0.02
        model = TetHeatTransferModel(
            nodes, tets, np.where(metal_tets, 0, 1),
            [constant_material(k_metal, 5e6), constant_material(k_chill, 1.3e6)],
            boundary_faces_of(tets),
        )
        dz = nodes[:, 2] - 0.02
        model.set_initial_temperature(1000.0 + np.where(dz >= 0.0, g_metal, g_chill) * dz)
        interface = np.flatnonzero(np.isclose(nodes[:, 2], 0.02))

        metal_side = model.nodal_gradient(metal_tets)
        np.testing.assert_allclose(metal_side[interface], [[0.0, 0.0, g_metal]] * interface.size, atol=1e-9)
        # Nodes away from the interface keep their own one-material gradient.
        below = nodes[:, 2] < 0.02 - 1e-9
        np.testing.assert_allclose(metal_side[below, 2], g_chill, rtol=1e-12)
        # The unmasked average mixes in the chill's shallow gradient.
        mixed = model.gradient_magnitude()[interface]
        self.assertTrue(np.all(mixed < 0.8 * g_metal))
        # The node subset returns the same rows.
        np.testing.assert_allclose(
            model.nodal_gradient(metal_tets, nodes=interface), metal_side[interface], rtol=1e-12
        )
        np.testing.assert_allclose(model.gradient_magnitude(nodes=interface), mixed, rtol=1e-12)
        with self.assertRaises(ValueError):
            model.nodal_gradient(np.ones(3, dtype=bool))

    def test_numba_and_scipy_paths_agree(self) -> None:
        curve = TetMaterial(
            np.array([0.0, 500.0, 1000.0]), np.array([10.0, 40.0, 25.0]), np.array([3e6, 3.5e6, 4e6])
        )
        nodes, tets = kuhn_box_mesh((4, 3, 3), (0.04, 0.03, 0.03))
        faces = boundary_faces_of(tets)
        tet_material = (nodes[tets].mean(axis=1)[:, 0] > 0.02).astype(np.int64)
        rng = np.random.default_rng(5)
        initial = rng.uniform(100.0, 1100.0, len(nodes))
        mask = tet_material == 0

        def run() -> tuple:
            model = TetHeatTransferModel(
                nodes, tets, tet_material, [curve, constant_material(150.0, 1.3e6)], faces
            )
            model.add_boundary_group(np.arange(len(faces)), 500.0, 25.0, lambda t: 1.0 / (1.0 + t))
            model.set_initial_temperature(initial)
            for dt in (0.5, 1.0, 2.0):
                model.step(dt)
            return (
                model.temperatures.copy(),
                model.node_heat_capacity(),
                model.nodal_gradient(mask),
                model.stiffness_matrix().toarray(),
            )

        numba_result = run()
        with mock.patch.object(tet_heat_transfer, "NUMBA_AVAILABLE", False):
            scipy_result = run()
        for got, expected in zip(numba_result, scipy_result):
            np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)

    def test_latent_capacity_sums_to_material_latent_heat(self) -> None:
        nodes, tets = kuhn_box_mesh((2, 2, 2), (0.02, 0.02, 0.02))
        centroid_x = nodes[tets].mean(axis=1)[:, 0]
        tet_material = (centroid_x > 0.01).astype(np.int64)
        materials = [constant_material(30.0, 3e6, latent=1e9), constant_material(1.0, 2e6)]
        model = TetHeatTransferModel(nodes, tets, tet_material, materials, boundary_faces_of(tets))
        latent = model.node_latent_capacity()
        self.assertAlmostEqual(latent.sum() / (1e9 * 0.01 * 0.02 * 0.02), 1.0, places=12)
        self.assertTrue(np.all(latent[nodes[:, 0] > 0.0101] == 0.0))


class TetHeatTransferPhysicsTest(unittest.TestCase):
    def test_adiabatic_energy_conserved_and_relaxes_to_mean(self) -> None:
        model, _, _ = make_model((4, 4, 4), (0.01, 0.01, 0.01), [constant_material(10.0, 1e6)])
        rng = np.random.default_rng(2)
        model.set_initial_temperature(rng.uniform(200.0, 800.0, model.n_nodes))
        capacity = model.node_heat_capacity()
        mean = float(np.dot(capacity, model.temperatures) / capacity.sum())
        energy0 = model.total_energy()
        for _ in range(100):
            model.step(1.0)
            self.assertLess(abs(model.total_energy() - energy0), 1e-9 * abs(energy0))
        self.assertLess(np.abs(model.temperatures - mean).max(), 1e-6)
        self.assertLess(model.last_residual, 1e-9)
        self.assertFalse(model.last_used_direct_solver)

    def test_lumped_capacitance_cube(self) -> None:
        side, k, rho_cp, h, t0, t_inf = 0.01, 1000.0, 1e6, 10.0, 700.0, 25.0
        model, _, faces = make_model((2, 2, 2), (side, side, side), [constant_material(k, rho_cp)])
        model.add_boundary_group(np.arange(len(faces)), htc=h, t_inf=t_inf)
        model.set_initial_temperature(t0)
        self.assertLess(h * (side / 6.0) / k, 0.1)
        tau = rho_cp * side**3 / (h * 6.0 * side**2)
        dt = 0.5
        for n in range(1, 601):
            model.step(dt)
            if n % 100 == 0:
                exact = math.exp(-model.time / tau)
                numeric = (model.temperatures.mean() - t_inf) / (t0 - t_inf)
                self.assertLess(abs(numeric - exact) / exact, 0.01, msg=f"t={model.time}")

    def test_slab_centre_matches_series_solution(self) -> None:
        thickness, k, rho_cp, biot, t0, t_inf = 0.02, 20.0, 4e6, 1.0, 700.0, 100.0
        half = 0.5 * thickness
        h = biot * k / half
        alpha = k / rho_cp
        model, nodes, faces = make_model((20, 1, 1), (thickness, 0.002, 0.002), [constant_material(k, rho_cp)])
        x_faces = np.concatenate(
            [faces_on_plane(nodes, faces, 0, 0.0), faces_on_plane(nodes, faces, 0, thickness)]
        )
        model.add_boundary_group(x_faces, htc=h, t_inf=t_inf)
        model.set_initial_temperature(t0)
        centre = np.isclose(nodes[:, 0], half)
        dt = 0.01
        steps_per_tenth = round(0.1 * half**2 / alpha / dt)  # Fo increment 0.1
        for tenth in range(1, 11):
            for _ in range(steps_per_tenth):
                model.step(dt)
            fourier = alpha * model.time / half**2
            self.assertAlmostEqual(fourier, 0.1 * tenth, places=9)
            exact = slab_centre_theta(biot, fourier)
            numeric = (model.temperatures[centre].mean() - t_inf) / (t0 - t_inf)
            self.assertLess(abs(numeric - exact) / exact, 0.02, msg=f"Fo={fourier:.1f}")

    def test_huge_time_step_is_stable_and_bounded(self) -> None:
        model, _, faces = make_model((3, 3, 3), (0.03, 0.03, 0.03), [constant_material(30.0, 3e6)])
        model.add_boundary_group(np.arange(len(faces)), htc=500.0, t_inf=25.0)
        model.set_initial_temperature(700.0)
        for _ in range(10):
            model.step(1e4)
            temps = model.temperatures
            self.assertTrue(np.all(np.isfinite(temps)))
            self.assertGreaterEqual(temps.min(), 25.0 - 1e-9)
            self.assertLessEqual(temps.max(), 700.0 + 1e-9)
        self.assertLess(np.abs(model.temperatures - 25.0).max(), 1e-6)

    def test_time_function_is_evaluated_at_end_of_step(self) -> None:
        model, _, faces = make_model((2, 2, 2), (0.01, 0.01, 0.01), [constant_material(30.0, 3e6)])
        calls: list[float] = []

        def switch_on_at_one_second(t: float) -> float:
            calls.append(t)
            return 0.0 if t < 1.0 - 1e-12 else 1.0

        model.add_boundary_group(np.arange(len(faces)), 500.0, 25.0, switch_on_at_one_second)
        model.set_initial_temperature(700.0)
        model.step(0.5)
        self.assertEqual(calls, [0.5])
        np.testing.assert_allclose(model.temperatures, 700.0, rtol=0, atol=1e-9)
        model.step(0.5)
        self.assertEqual(calls, [0.5, 1.0])
        self.assertLess(model.temperatures.max(), 700.0 - 1e-3)

    def test_two_material_bar_series_resistance_flux(self) -> None:
        length, k1, k2, h, hot, cold = 0.01, 10.0, 50.0, 1e5, 500.0, 100.0
        nodes, tets = kuhn_box_mesh((20, 2, 2), (2 * length, 0.004, 0.004))
        centroid_x = nodes[tets].mean(axis=1)[:, 0]
        tet_material = (centroid_x > length).astype(np.int64)
        faces = boundary_faces_of(tets)
        model = TetHeatTransferModel(
            nodes, tets, tet_material,
            [constant_material(k1, 3e6), constant_material(k2, 4e6)], faces,
        )
        model.add_boundary_group(faces_on_plane(nodes, faces, 0, 0.0), h, hot)
        model.add_boundary_group(faces_on_plane(nodes, faces, 0, 2 * length), h, cold)
        model.set_initial_temperature(300.0)
        for _ in range(5):
            model.step(1e6)
        expected = (hot - cold) / (1.0 / h + length / k1 + length / k2 + 1.0 / h)
        hot_face = np.isclose(nodes[:, 0], 0.0)
        cold_face = np.isclose(nodes[:, 0], 2 * length)
        q_in = h * (hot - model.temperatures[hot_face].mean())
        q_out = h * (model.temperatures[cold_face].mean() - cold)
        self.assertLess(abs(q_in - expected) / expected, 0.01)
        self.assertLess(abs(q_out - expected) / expected, 0.01)
        interface = np.isclose(nodes[:, 0], length)
        t_interface = hot - expected * (1.0 / h + length / k1)
        np.testing.assert_allclose(model.temperatures[interface], t_interface, rtol=1e-6)

    def test_cooling_rate_sign(self) -> None:
        model, _, faces = make_model((2, 2, 2), (0.01, 0.01, 0.01), [constant_material(30.0, 3e6)])
        model.add_boundary_group(np.arange(len(faces)), 500.0, 25.0)
        model.set_initial_temperature(700.0)
        prev = model.temperatures.copy()
        model.step(1.0)
        self.assertTrue(np.all(model.get_cooling_rate(prev, 1.0) > 0.0))

    def test_in_place_temperature_edits_are_used_by_step(self) -> None:
        model, _, _ = make_model((2, 2, 2), (0.01, 0.01, 0.01), [constant_material(30.0, 3e6)])
        model.set_initial_temperature(100.0)
        buffer = model.temperatures
        buffer[:] = 400.0
        model.step(1.0)
        self.assertIs(model.temperatures, buffer)
        np.testing.assert_allclose(buffer, 400.0, rtol=1e-12)


class TetHeatTransferValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.nodes, self.tets = kuhn_box_mesh((1, 1, 1), (0.01, 0.01, 0.01))
        self.faces = boundary_faces_of(self.tets)
        self.mats = [constant_material(30.0, 3e6)]
        self.tet_material = np.zeros(len(self.tets), dtype=np.int64)

    def build(self, **overrides: object) -> TetHeatTransferModel:
        args = dict(
            nodes_m=self.nodes, tets=self.tets, tet_material=self.tet_material,
            materials=self.mats, boundary_faces=self.faces,
        )
        args.update(overrides)
        return TetHeatTransferModel(**args)  # type: ignore[arg-type]

    def test_rejects_bad_inputs(self) -> None:
        bad_nodes = self.nodes.copy()
        bad_nodes[0, 0] = np.nan
        cases = {
            "nodes_m must have shape": dict(nodes_m=self.nodes[:, :2]),
            "non-finite coordinates": dict(nodes_m=bad_nodes),
            "tets must have shape": dict(tets=self.tets[:, :3]),
            "integer dtype": dict(tets=self.tets.astype(float)),
            "tets indices": dict(tets=np.where(self.tets == 0, len(self.nodes), self.tets)),
            "tet_material length": dict(tet_material=self.tet_material[:-1]),
            "tet_material indices": dict(tet_material=self.tet_material + 1),
            "boundary_faces indices": dict(boundary_faces=self.faces - 1),
            "materials is empty": dict(materials=[]),
            "zero volume": dict(tets=np.vstack([self.tets, [[0, 1, 2, 2]]]),
                                tet_material=np.zeros(len(self.tets) + 1, dtype=np.int64)),
            "not referenced": dict(nodes_m=np.vstack([self.nodes, [[1.0, 1.0, 1.0]]])),
        }
        for message, override in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.build(**override)

    def test_rejects_bad_material_group_and_step_arguments(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly ascending"):
            TetMaterial(np.array([0.0, 0.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        with self.assertRaisesRegex(ValueError, "equal length"):
            TetMaterial(np.array([0.0, 1.0]), np.array([1.0]), np.array([1.0, 2.0]))
        with self.assertRaisesRegex(ValueError, "conductivity must be positive"):
            TetMaterial(np.array([0.0]), np.array([0.0]), np.array([1.0]))
        model = self.build()
        with self.assertRaisesRegex(ValueError, "face_indices indices"):
            model.add_boundary_group([len(self.faces)], 10.0, 25.0)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            model.add_boundary_group([0, 0], 10.0, 25.0)
        with self.assertRaisesRegex(ValueError, "htc must be finite"):
            model.add_boundary_group([0], np.inf, 25.0)
        with self.assertRaisesRegex(ValueError, "temperature field must have shape"):
            model.set_initial_temperature(np.zeros(3))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            model.set_initial_temperature(np.nan)
        with self.assertRaisesRegex(ValueError, "dt must be finite"):
            model.step(0.0)


if __name__ == "__main__":
    unittest.main()
