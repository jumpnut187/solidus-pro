"""The fused step kernels against the bit-for-bit path they replace.

``physics.kernel_dispatch.FUSED_KERNELS`` (``CASTSIM_FUSED_KERNELS``) switches
between the two. The fused kernels do the same arithmetic in fewer passes, in
a different order in places and without ``fastmath``, so they agree with the
reference path to rounding. Each test states how closely.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core import solidification_loop
from physics import heat_transfer, kernel_dispatch, mold_coupling, solidification_physics

NUMBA = heat_transfer.NUMBA_AVAILABLE and solidification_physics.NUMBA_AVAILABLE

CURVES = {
    "temp_points": np.array([25.0, 400.0, 800.0, 1200.0, 1450.0, 1600.0]),
    "conductivity": np.array([50.0, 42.0, 30.0, 28.0, 30.0, 32.0]),
    "density": np.array([7850.0, 7730.0, 7600.0, 7450.0, 7300.0, 7000.0]),
    "specific_heat": np.array([450.0, 600.0, 900.0, 650.0, 700.0, 800.0]),
}
ALLOY = SimpleNamespace(
    density_liquid=7000.0,
    density_solid=7300.0,
    specific_heat=700.0,
    thermal_diffusivity=6e-6,
    solidus_temp=1440.0,
    liquidus_temp=1500.0,
    latent_heat=270000.0,
)


def lattice_neighbors(shape: tuple[int, int, int]) -> np.ndarray:
    """26-slot neighbour array of a box lattice, in the mesh's slot order."""
    nx, ny, nz = shape
    index = np.arange(nx * ny * nz).reshape(shape)
    offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    offsets += [(sx, sy, 0) for sx in (1, -1) for sy in (1, -1)]
    offsets += [(sx, 0, sz) for sx in (1, -1) for sz in (1, -1)]
    offsets += [(0, sy, sz) for sy in (1, -1) for sz in (1, -1)]
    offsets += [(sx, sy, sz) for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)]
    neighbors = np.full((index.size, 26), -1, dtype=np.int32)
    ii, jj, kk = np.indices(shape)
    for slot, (dx, dy, dz) in enumerate(offsets):
        i2, j2, k2 = ii + dx, jj + dy, kk + dz
        ok = (i2 >= 0) & (i2 < nx) & (j2 >= 0) & (j2 < ny) & (k2 >= 0) & (k2 < nz)
        neighbors[index[ok], slot] = index[i2[ok], j2[ok], k2[ok]]
    return neighbors


def lattice_centers(shape: tuple[int, int, int], pitch: float) -> np.ndarray:
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1).reshape(-1, 3)
    return (grid * pitch).astype(np.float32)


@unittest.skipUnless(NUMBA, "numba not installed")
class CutCellFluxKernelTests(unittest.TestCase):
    def _problem(self, shape: tuple[int, int, int], seed: int):
        rng = np.random.default_rng(seed)
        neighbors = lattice_neighbors(shape)
        n = len(neighbors)
        T = rng.uniform(900.0, 1600.0, n)
        alpha = rng.uniform(4e-6, 9e-6, n)
        rho_cp = rng.uniform(4e6, 6e6, n)
        weights = np.where(rng.random(n) < 0.1, 0.0, rng.uniform(0.25, 1.0, n))
        apertures = heat_transfer.cut_cell_link_apertures(weights)
        factors = heat_transfer.cut_cell_conductance_factors(weights, neighbors, apertures)
        inv_dx2 = np.full(n, 1.0 / 0.00381**2)
        hdx = np.where(rng.random(n) < 0.15, 800.0 * 0.00381, 0.0)
        source = rng.uniform(-2.0, 2.0, n)
        return T, neighbors, alpha, rho_cp, weights, factors, apertures, inv_dx2, hdx, source

    def test_rates_are_the_rates_kernels_bits(self) -> None:
        for shape in ((5, 4, 3), (21, 17, 13)):
            T, nb, alpha, rho_cp, w, f, ap, inv_dx2, hdx, source = self._problem(shape, 1)
            expected = heat_transfer.cut_cell_conduction_rates_parallel(nb, alpha, rho_cp, w, f, ap, inv_dx2)
            for kernel in (heat_transfer.cut_cell_flux_parallel, heat_transfer.cut_cell_flux_serial):
                rates = np.full(len(T), -1.0)
                flux = np.empty(len(T))
                kernel(T, nb, alpha, rho_cp, w, f, ap, inv_dx2, hdx, True, rates, False, 0.0, source, flux)
                np.testing.assert_array_equal(rates, expected)

    def test_update_agrees_with_the_fastmath_kernel_to_rounding(self) -> None:
        dt = 0.21
        for shape in ((5, 4, 3), (21, 17, 13)):
            T, nb, alpha, rho_cp, w, f, ap, inv_dx2, hdx, source = self._problem(shape, 2)
            expected = heat_transfer.update_heterogeneous_cut_cell_parallel(
                T, nb, alpha, rho_cp, w, f, ap, inv_dx2, source, dt, hdx
            )
            direct = np.empty(len(T))
            heat_transfer.cut_cell_flux_parallel(
                T, nb, alpha, rho_cp, w, f, ap, inv_dx2, hdx, False, inv_dx2, True, dt, source, direct
            )
            # A step moves a cell by a few K at most; the two kernels differ in the last bit or two.
            np.testing.assert_allclose(direct, expected, rtol=1e-13, atol=0.0)

            # Flux now, step later (the first substep) is the same expression: same bits.
            flux = np.empty(len(T))
            rates = np.empty(len(T))
            heat_transfer.cut_cell_flux_parallel(
                T, nb, alpha, rho_cp, w, f, ap, inv_dx2, hdx, True, rates, False, 0.0, source, flux
            )
            for apply_kernel in (heat_transfer.apply_cut_cell_flux_parallel, heat_transfer.apply_cut_cell_flux_serial):
                later = np.empty(len(T))
                apply_kernel(T, flux, rho_cp, w, source, dt, later)
                np.testing.assert_array_equal(later, direct)

            # The serial twin, too.
            twin = np.empty(len(T))
            heat_transfer.cut_cell_flux_serial(
                T, nb, alpha, rho_cp, w, f, ap, inv_dx2, hdx, False, inv_dx2, True, dt, source, twin
            )
            np.testing.assert_array_equal(twin, direct)

    def test_empty_cells_keep_their_temperature(self) -> None:
        T, nb, alpha, rho_cp, w, f, ap, inv_dx2, hdx, source = self._problem((6, 5, 4), 3)
        out = np.empty(len(T))
        rates = np.empty(len(T))
        heat_transfer.cut_cell_flux_parallel(T, nb, alpha, rho_cp, w, f, ap, inv_dx2, hdx, True, rates, True, 0.3, source, out)
        empty = w <= 0.0
        self.assertTrue(np.any(empty))
        np.testing.assert_array_equal(out[empty], T[empty])
        np.testing.assert_array_equal(rates[empty], 0.0)


@unittest.skipUnless(NUMBA, "numba not installed")
class InvertAndCommitKernelTests(unittest.TestCase):
    def test_matches_capacity_then_inversion_then_commit(self) -> None:
        rng = np.random.default_rng(11)
        n = 6000
        capacity_curve = np.ascontiguousarray(CURVES["density"] * CURVES["specific_heat"])
        l_vol = ALLOY.latent_heat * (ALLOY.density_liquid + ALLOY.density_solid) / 2.0
        T_post = rng.uniform(1380.0, 1560.0, n)
        fs0 = np.clip(rng.uniform(-0.3, 1.3, n), 0.0, 1.0)
        # A fifth of the cells carry a heat-capacity override (chill, sand).
        override_cells = np.sort(rng.choice(n, n // 5, replace=False))
        override_values = rng.uniform(1.5e6, 3.0e6, len(override_cells))
        positions = np.full(n, -1, dtype=np.int64)
        positions[override_cells] = np.arange(len(override_cells))

        for use_curve in (True, False):
            for use_overrides in (True, False):
                # Reference: the three steps of the bit-for-bit path.
                if use_curve:
                    rho_cp = heat_transfer.interpolate_curve_parallel(T_post, CURVES["temp_points"], capacity_curve)
                else:
                    rho_cp = np.full(n, 4.9e6)
                if use_overrides:
                    rho_cp[override_cells] = override_values
                T_ref = np.empty(n)
                solidification_loop._invert_enthalpy_numpy(
                    T_post, fs0, T_ref, ALLOY.liquidus_temp, ALLOY.solidus_temp, l_vol / rho_cp
                )
                ref = [fs0.copy(), np.zeros(n), np.full(n, np.inf), np.full(n, np.inf), np.zeros(n)]
                solidification_physics.commit_solidification_update_parallel(
                    T_ref, T_ref, ref[0], ref[1], ref[2], ref[3], ref[4], True, 1500.0, 60.0, 12.0, 0.5, l_vol
                )

                results = []
                for kernel in (
                    solidification_physics.invert_enthalpy_and_commit_parallel,
                    solidification_physics.invert_enthalpy_and_commit_serial,
                ):
                    T = T_post.copy()
                    state = [fs0.copy(), np.zeros(n), np.full(n, np.inf), np.full(n, np.inf), np.zeros(n)]
                    kernel(
                        T, state[0], state[1], state[2], state[3], state[4],
                        ALLOY.liquidus_temp, ALLOY.solidus_temp, l_vol,
                        use_curve, CURVES["temp_points"], capacity_curve, 4.9e6,
                        use_overrides, positions, override_values,
                        1500.0, 60.0, 12.0, 0.5, l_vol,
                    )
                    results.append([T] + state)
                with self.subTest(use_curve=use_curve, use_overrides=use_overrides):
                    T, fs, fs_prev, start, end, latent = results[0]
                    np.testing.assert_allclose(T, T_ref, rtol=1e-13)
                    np.testing.assert_allclose(fs, ref[0], rtol=1e-11, atol=1e-13)
                    np.testing.assert_array_equal(fs_prev, ref[1])
                    np.testing.assert_array_equal(start, ref[2])
                    np.testing.assert_array_equal(end, ref[3])
                    np.testing.assert_allclose(latent, ref[4], rtol=1e-9, atol=1e-3)
                    for parallel, serial in zip(*results):
                        np.testing.assert_array_equal(serial, parallel)

    def test_model_method_matches_update(self) -> None:
        """``finish_step_from_enthalpy`` leaves the model where inversion + ``update`` leaves it."""
        rng = np.random.default_rng(12)
        neighbors = lattice_neighbors((7, 6, 5))
        n = len(neighbors)
        weights = rng.uniform(0.3, 1.0, n)
        l_vol = ALLOY.latent_heat * (ALLOY.density_liquid + ALLOY.density_solid) / 2.0
        heat = heat_transfer.HeatTransferModel(
            n_voxels=n, voxel_size=0.004, neighbors=neighbors, alloy=ALLOY,
            use_temperature_dependent=True, property_curves=CURVES, capacity_weights=weights,
        )
        models = [solidification_physics.SolidificationModel(n_voxels=n, alloy=ALLOY) for _ in range(2)]
        fields = []
        for step in range(6):
            T_post = rng.uniform(1400.0, 1540.0, n) - 12.0 * step
            heat.set_initial_temperature(T_post)
            # reference
            T_ref = np.empty(n)
            solidification_loop._invert_enthalpy_numpy(
                T_post, np.asarray(models[0].fraction_solid, dtype=np.float64).copy(), T_ref,
                ALLOY.liquidus_temp, ALLOY.solidus_temp, l_vol / heat.volumetric_heat_capacity(),
            )
            models[0].update(T_ref, 0.5)
            # fused
            T = T_post.copy()
            self.assertTrue(models[1].can_finish_step_from_enthalpy())
            models[1].finish_step_from_enthalpy(
                T, heat.heat_capacity_lookup(), ALLOY.liquidus_temp, ALLOY.solidus_temp, l_vol, 0.5
            )
            fields.append((T, T_ref))
        for T, T_ref in fields:
            np.testing.assert_allclose(T, T_ref, rtol=1e-13)
        np.testing.assert_allclose(models[1].fraction_solid, models[0].fraction_solid, rtol=1e-11, atol=1e-13)
        np.testing.assert_array_equal(models[1].solidification_time, models[0].solidification_time)
        np.testing.assert_array_equal(models[1]._solidification_end, models[0]._solidification_end)
        self.assertEqual(models[1].time, models[0].time)


@unittest.skipUnless(NUMBA, "numba not installed")
class LeastSquaresGradientKernelTests(unittest.TestCase):
    def _compare(self, shape, keep, seed, rtol):
        rng = np.random.default_rng(seed)
        centers_all = lattice_centers(shape, 0.15)
        neighbors_all = lattice_neighbors(shape)
        kept = rng.random(len(centers_all)) < keep
        # Drop cells: re-index the neighbours of the kept ones.
        new_index = np.full(len(centers_all), -1, dtype=np.int32)
        new_index[kept] = np.arange(int(kept.sum()), dtype=np.int32)
        centers = centers_all[kept]
        neighbors = np.where(neighbors_all[kept] >= 0, new_index[np.maximum(neighbors_all[kept], 0)], -1).astype(np.int32)
        n = len(centers)
        temperatures = 1500.0 - 40.0 * centers[:, 0].astype(np.float64) + rng.uniform(-3.0, 3.0, n)
        cast_mask = rng.random(n) < 0.9
        excluded = rng.random(n) < 0.05
        targets = np.arange(n, dtype=np.int64)
        reference = solidification_loop._least_squares_gradients_c_per_mm(
            centers_in=centers.astype(np.float64), temperatures=temperatures, neighbors=neighbors,
            target_indices=targets, cast_mask=cast_mask, excluded_neighbor_mask=excluded,
        )
        fused = solidification_loop._least_squares_gradients_fused(
            centers=centers, temperatures=temperatures, neighbors=neighbors,
            target_indices=targets, cast_mask=cast_mask, excluded_neighbor_mask=excluded,
        )
        np.testing.assert_array_equal(fused[1], reference[1])
        np.testing.assert_allclose(fused[0], reference[0], rtol=rtol, atol=0.0)
        return reference

    def test_full_lattice_and_lattice_with_gaps(self) -> None:
        # float32 results: one unit in the last place is 1.2e-7 relative.
        self._compare((9, 8, 7), 1.0, 0, 5e-7)
        self._compare((10, 9, 8), 0.6, 1, 5e-7)

    def test_flat_and_sparse_stencils_go_to_the_reference(self) -> None:
        """One layer of voxels: no stencil spans three directions, so every value is the reference's."""
        reference = self._compare((12, 11, 1), 1.0, 2, 0.0)
        self.assertTrue(np.all(reference[1][reference[0] > 0] == 1))
        # Very sparse: isolated voxels and collinear pairs.
        self._compare((8, 8, 8), 0.12, 3, 5e-7)

    def test_no_exclusion_mask(self) -> None:
        centers = lattice_centers((6, 5, 4), 0.2)
        neighbors = lattice_neighbors((6, 5, 4))
        n = len(centers)
        temperatures = np.random.default_rng(4).uniform(1400.0, 1500.0, n)
        cast_mask = np.ones(n, dtype=bool)
        targets = np.arange(0, n, 3, dtype=np.int64)
        reference = solidification_loop._least_squares_gradients_c_per_mm(
            centers_in=centers.astype(np.float64), temperatures=temperatures, neighbors=neighbors,
            target_indices=targets, cast_mask=cast_mask,
        )
        fused = solidification_loop._least_squares_gradients_fused(
            centers=centers, temperatures=temperatures, neighbors=neighbors, target_indices=targets, cast_mask=cast_mask,
        )
        np.testing.assert_array_equal(fused[1], reference[1])
        np.testing.assert_allclose(fused[0], reference[0], rtol=5e-7)


@unittest.skipUnless(NUMBA, "numba not installed")
class CoupleKernelTests(unittest.TestCase):
    """``CoupledMoldSolver._couple`` through the kernel and through NumPy."""

    def _solver(self, seed: int, stencil_contact: float) -> mold_coupling.CoupledMoldSolver:
        rng = np.random.default_rng(seed)
        casting_neighbors = lattice_neighbors((5, 4, 3))
        mold_neighbors = lattice_neighbors((6, 5, 4))
        n_c, n_m = len(casting_neighbors), len(mold_neighbors)
        casting = heat_transfer.HeatTransferModel(
            n_voxels=n_c, voxel_size=0.004, neighbors=casting_neighbors, alloy=ALLOY,
            use_temperature_dependent=True, property_curves=CURVES,
            capacity_weights=rng.uniform(0.4, 1.0, n_c),
        )
        sand = SimpleNamespace(
            density_liquid=1600.0, density_solid=1600.0, specific_heat=1100.0, thermal_diffusivity=4e-7,
            solidus_temp=1e9, liquidus_temp=2e9,
        )
        mold = heat_transfer.HeatTransferModel(n_voxels=n_m, voxel_size=0.004, neighbors=mold_neighbors, alloy=sand)
        casting.set_initial_temperature(rng.uniform(1300.0, 1560.0, n_c))
        mold.set_initial_temperature(rng.uniform(25.0, 900.0, n_m))
        n_pairs = 90
        first = rng.integers(0, n_c, n_pairs)
        second = rng.integers(0, n_m, n_pairs)
        is_stencil = rng.random(n_pairs) < 0.25
        pairs = mold_coupling.MoldPairs(
            first=first,
            second=second,
            area=np.where(is_stencil, 0.0, rng.uniform(4e-6, 1.6e-5, n_pairs)),
            group=np.where(is_stencil, -1, rng.integers(0, 3, n_pairs)),
            stencil_factor=np.where(is_stencil, rng.uniform(1e-3, 4e-3, n_pairs), 0.0),
            htc=[900.0, 400.0, 0.0],
            temperature_factors=[
                (np.array([20.0, 600.0, 1200.0, 1450.0, 1600.0]), np.array([0.3, 0.45, 0.8, 1.0, 1.0])),
                None,
                None,
            ],
            stencil_contact_htc=stencil_contact,
        )
        return mold_coupling.CoupledMoldSolver(casting, mold, pairs)

    def test_kernel_matches_numpy(self) -> None:
        for seed, stencil_contact in ((0, 0.0), (1, 650.0)):
            with patch.object(kernel_dispatch, "FUSED_KERNELS", False):
                reference = self._solver(seed, stencil_contact)
            with patch.object(kernel_dispatch, "FUSED_KERNELS", True):
                fused = self._solver(seed, stencil_contact)
            with self.subTest(seed=seed):
                self.assertTrue(np.any(reference._G > 0.0))
                np.testing.assert_allclose(fused._G, reference._G, rtol=1e-13)
                np.testing.assert_array_equal(fused._T_second, reference._T_second)
                np.testing.assert_allclose(fused.casting._sink_G, reference.casting._sink_G, rtol=1e-13)
                np.testing.assert_allclose(fused.casting._sink_T, reference.casting._sink_T, rtol=1e-13)


@unittest.skipUnless(NUMBA, "numba not installed")
class FusedRunTests(unittest.TestCase):
    """A whole run with the fused kernels against the same run on the bit-for-bit path."""

    def _run(self, fused: bool):
        from core.part_manager import build_part_manager
        from core.simulation_pipeline import PipelineInputs, run_pipeline
        from tests.helpers.ci_models import ci_model

        manager = build_part_manager(
            [
                {"filepath": ci_model("block_2inch.stl"), "part_type": "cast"},
                {"filepath": ci_model("small_block.stl"), "part_type": "chill",
                 "position_x": 1.5, "position_y": 0.0, "position_z": 0.5},
                {"filepath": ci_model("small_block.stl"), "part_type": "core",
                 "position_x": -1.5, "position_y": 0.0, "position_z": 0.5},
            ]
        )
        inputs = PipelineInputs(
            alloy_name="WCB", voxel_size=0.25, refinement_level=0, solidification_params={},
            analysis_options={"shrinkage": True, "niyama": True, "solver_mode": "thermal_coupled"},
            show_progress=False,
        )
        with patch.object(kernel_dispatch, "FUSED_KERNELS", fused):
            result = run_pipeline(inputs, part_manager=manager)
        engine = result.engine
        return {
            "temperatures": np.array(engine._thermal_heat_model.temperatures),
            "mold": np.array(engine._thermal_heat_model.mold_temperatures),
            "end": np.array(engine.solidification_end_times),
            "start": np.array(engine.solidification_start_times),
            "niyama_gradient": np.array(engine._thermal_niyama_gradient_at_eval),
            "states": np.array(engine.mesh.voxel_states),
            "time": engine.time,
        }

    def test_fused_run_agrees_with_reference_run(self) -> None:
        reference = self._run(False)
        fused = self._run(True)
        self.assertEqual(fused["time"], reference["time"])
        np.testing.assert_allclose(fused["temperatures"], reference["temperatures"], rtol=1e-9)
        np.testing.assert_allclose(fused["mold"], reference["mold"], rtol=1e-9)
        # Times are whole steps, stored as float32.
        np.testing.assert_array_equal(fused["end"], reference["end"])
        np.testing.assert_array_equal(fused["start"], reference["start"])
        np.testing.assert_allclose(fused["states"], reference["states"], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(fused["niyama_gradient"], reference["niyama_gradient"], rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
