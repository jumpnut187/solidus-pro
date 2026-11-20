"""Serial twins of the Numba parallel kernels (physics/kernel_dispatch.py).

A twin must give the same bits as its parallel kernel, since which one runs
depends only on the mesh size. That holds for kernels compiled without
``fastmath`` and is checked here on every twin the solver uses.
"""

from __future__ import annotations

import unittest

import numpy as np

from physics import heat_transfer, kernel_dispatch, solidification_physics

NUMBA = heat_transfer.NUMBA_AVAILABLE and solidification_physics.NUMBA_AVAILABLE


def _twins(module) -> list[tuple[str, object, object]]:
    """(name, parallel kernel, serial twin) for every twin defined in ``module``."""
    found = []
    for name in sorted(vars(module)):
        if not name.endswith("_serial"):
            continue
        parallel = getattr(module, name[: -len("_serial")] + "_parallel", None)
        if parallel is not None:
            found.append((name, parallel, getattr(module, name)))
    return found


@unittest.skipUnless(NUMBA, "numba not installed")
class SerialTwinRulesTests(unittest.TestCase):
    def test_every_twin_is_of_a_kernel_without_fastmath(self) -> None:
        twins = _twins(heat_transfer) + _twins(solidification_physics)
        self.assertGreaterEqual(len(twins), 8)
        for name, parallel, serial in twins:
            with self.subTest(kernel=name):
                self.assertTrue(parallel.targetoptions.get("parallel"))
                self.assertFalse(parallel.targetoptions.get("fastmath"))
                self.assertFalse(serial.targetoptions.get("parallel"))
                self.assertFalse(serial.targetoptions.get("fastmath"))

    def test_fastmath_kernel_is_refused(self) -> None:
        for kernel in (
            heat_transfer.update_heterogeneous_parallel,
            heat_transfer.update_heterogeneous_cut_cell_parallel,
        ):
            with self.assertRaises(ValueError):
                kernel_dispatch.serial_twin(kernel)

    def test_runs_serial_follows_the_budget(self) -> None:
        from unittest.mock import patch

        base = kernel_dispatch.SERIAL_KERNEL_BUDGET_NS
        per_thread = kernel_dispatch.SERIAL_KERNEL_BUDGET_PER_THREAD_NS
        for threads in (2, 8, 18):
            budget = base + per_thread * (threads - 2)
            with patch.object(kernel_dispatch, "get_num_threads", return_value=threads):
                self.assertTrue(kernel_dispatch.runs_serial(int(budget // 2), 2.0))
                self.assertFalse(kernel_dispatch.runs_serial(int(budget // 2) + 1, 2.0))
                self.assertTrue(kernel_dispatch.runs_serial(0, 1e9))
        # One thread: the pool cannot help, whatever the size.
        with patch.object(kernel_dispatch, "get_num_threads", return_value=1):
            self.assertTrue(kernel_dispatch.runs_serial(10**9, 100.0))
        # A zero budget turns the twins off, at any thread count.
        with patch.object(kernel_dispatch, "SERIAL_KERNEL_BUDGET_NS", 0.0):
            self.assertFalse(kernel_dispatch.runs_serial(1, 1.0))
            with patch.object(kernel_dispatch, "get_num_threads", return_value=1):
                self.assertFalse(kernel_dispatch.runs_serial(1, 1.0))


@unittest.skipUnless(NUMBA, "numba not installed")
class SerialTwinBitEqualityTests(unittest.TestCase):
    """Each twin against its parallel kernel, same inputs, exact comparison."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(20261120)
        self.curve_t = np.array([25.0, 400.0, 800.0, 1200.0, 1450.0, 1600.0])
        self.curve_k = np.array([50.0, 42.0, 30.0, 28.0, 30.0, 32.0])
        self.curve_rho = np.array([7850.0, 7730.0, 7600.0, 7450.0, 7300.0, 7000.0])
        self.curve_cp = np.array([450.0, 600.0, 900.0, 650.0, 700.0, 800.0])
        self.sand_t = np.array([20.0, 100.0, 105.0, 300.0, 573.0, 580.0, 800.0, 1200.0, 1600.0])
        self.sand_k = np.array([0.6, 0.62, 0.9, 0.7, 0.75, 0.9, 1.0, 1.3, 1.5])
        self.sand_c = np.array([1.3e6, 1.5e6, 9e6, 1.6e6, 1.7e6, 4e6, 1.8e6, 1.9e6, 2.0e6])
        self.fine_t = np.arange(-100.0, 2500.5, 0.5)
        fine_c = np.interp(self.fine_t, self.sand_t, self.sand_c)
        self.fine_h = np.concatenate(
            ([0.0], np.cumsum(0.5 * (fine_c[1:] + fine_c[:-1]) * np.diff(self.fine_t)))
        )

    def _field(self, n: int) -> np.ndarray:
        """Temperatures spanning the tables, with table points, end points and a NaN."""
        T = self.rng.uniform(-50.0, 1700.0, n)
        T[:6] = self.curve_t
        T[6] = np.nan
        return T

    def _lattice_neighbors(self, shape: tuple[int, int, int]) -> np.ndarray:
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

    def test_zero_denominator_gives_the_same_non_finite_value(self) -> None:
        # rho * cp = 0 at a curve point: k / 0 inside the loop.
        T = self._field(500)
        rho = self.curve_rho.copy()
        rho[2] = 0.0
        args = (T, self.curve_t, self.curve_k, rho, self.curve_cp)
        parallel = heat_transfer.compute_diffusivity_from_curves_parallel(*args)
        np.testing.assert_array_equal(heat_transfer.compute_diffusivity_from_curves_serial(*args), parallel)
        self.assertTrue(np.any(np.isinf(parallel)))

    def test_curve_kernels(self) -> None:
        for n in (7, 1000, 40000):
            T = self._field(n)
            args = (T, self.curve_t, self.curve_k, self.curve_rho, self.curve_cp)
            np.testing.assert_array_equal(
                heat_transfer.compute_diffusivity_from_curves_serial(*args),
                heat_transfer.compute_diffusivity_from_curves_parallel(*args),
            )
            np.testing.assert_array_equal(
                heat_transfer.interpolate_curve_serial(T, self.curve_t, self.curve_cp),
                heat_transfer.interpolate_curve_parallel(T, self.curve_t, self.curve_cp),
            )

    def test_fused_property_kernel_matches_the_two_it_replaces(self) -> None:
        capacity = np.ascontiguousarray(self.curve_rho * self.curve_cp)
        for n in (7, 1000, 40000):
            T = self._field(n)
            expected_alpha = heat_transfer.compute_diffusivity_from_curves_parallel(
                T, self.curve_t, self.curve_k, self.curve_rho, self.curve_cp
            )
            expected_capacity = heat_transfer.interpolate_curve_parallel(T, self.curve_t, capacity)
            for kernel in (
                heat_transfer.compute_diffusivity_and_capacity_parallel,
                heat_transfer.compute_diffusivity_and_capacity_serial,
            ):
                alpha, rho_cp = kernel(T, self.curve_t, self.curve_k, self.curve_rho, self.curve_cp, capacity)
                np.testing.assert_array_equal(alpha, expected_alpha)
                np.testing.assert_array_equal(rho_cp, expected_capacity)

    def test_enthalpy_correction_matches_binary_search(self) -> None:
        """The hinted table lookup against the same interpolation done by ``np.searchsorted``."""

        def lookup(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
            j = np.clip(np.searchsorted(xp, x, side="right") - 1, 0, len(xp) - 2)
            inside = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]) * (x - xp[j]) + fp[j]
            return np.where(x <= xp[0], fp[0], np.where(x >= xp[-1], fp[-1], inside))

        for n, spread in ((400, 4.0), (6000, 0.05), (6000, 40.0), (6000, 900.0)):
            cells = 3 * n
            indices = np.sort(self.rng.choice(cells, n, replace=False)).astype(np.int64)
            T_old = self.rng.uniform(-150.0, 2600.0, cells)
            # Table points, both ends, and steps that cross the latent peaks or leave the table.
            T_old[indices[:40]] = self.rng.choice(self.fine_t, 40)
            T_old[indices[40:44]] = [self.fine_t[0], self.fine_t[-1], -100.25, 2499.75]
            T_new = T_old + self.rng.uniform(-spread, spread, cells) * (self.rng.random(cells) < 0.9)
            rcp = np.interp(T_old[indices], self.sand_t, self.sand_c)

            expected = T_new.copy()
            step = T_new[indices] - T_old[indices]
            moved = np.abs(step) >= 1e-3
            h = lookup(T_old[indices][moved], self.fine_t, self.fine_h) + step[moved] * rcp[moved]
            expected[indices[moved]] = lookup(h, self.fine_h, self.fine_t)

            for kernel in (heat_transfer.correct_tabulated_parallel, heat_transfer.correct_tabulated_serial):
                field = T_new.copy()
                kernel(field, T_old, indices, rcp, 0, self.fine_t, self.fine_h, 1e-3)
                with self.subTest(n=n, spread=spread):
                    np.testing.assert_array_equal(field, expected)

    def test_tabulated_kernels(self) -> None:
        for n in (50, 3000, 20000):
            cells = 4 * n
            indices = np.sort(self.rng.choice(cells, n, replace=False)).astype(np.int64)
            T = self.rng.uniform(15.0, 1650.0, cells)
            last = np.where(self.rng.random(n) < 0.5, T[indices] + self.rng.uniform(-0.02, 0.02, n), np.nan)
            results = []
            for kernel in (heat_transfer.update_tabulated_parallel, heat_transfer.update_tabulated_serial):
                diff_vals, rcp_vals, last_T = np.zeros(n + 3), np.zeros(n + 5), last.copy()
                kernel(T, indices, last_T, self.sand_t, self.sand_k, self.sand_c, diff_vals, 3, rcp_vals, 5, 0.01)
                results.append((diff_vals, rcp_vals, last_T))
            for parallel, serial in zip(*results):
                np.testing.assert_array_equal(serial, parallel)

            T_new = T + self.rng.uniform(-4.0, 4.0, cells) * (self.rng.random(cells) < 0.8)
            rcp = np.interp(T[indices], self.sand_t, self.sand_c)
            corrected = []
            for kernel in (heat_transfer.correct_tabulated_parallel, heat_transfer.correct_tabulated_serial):
                field = T_new.copy()
                kernel(field, T, indices, rcp, 0, self.fine_t, self.fine_h, 1e-3)
                corrected.append(field)
            np.testing.assert_array_equal(corrected[1], corrected[0])

    def test_neighbor_kernels(self) -> None:
        for shape in ((4, 3, 5), (20, 17, 13)):
            neighbors = self._lattice_neighbors(shape)
            n = len(neighbors)
            T = self.rng.uniform(900.0, 1600.0, n)
            alpha = self.rng.uniform(4e-6, 9e-6, n)
            rho_cp = self.rng.uniform(4e6, 6e6, n)
            weights = np.where(self.rng.random(n) < 0.1, 0.0, self.rng.uniform(0.25, 1.0, n))
            apertures = heat_transfer.cut_cell_link_apertures(weights)
            factors = heat_transfer.cut_cell_conductance_factors(weights, neighbors, apertures)
            inv_dx2 = np.full(n, 1.0 / 0.00381**2)
            args = (neighbors, alpha, rho_cp, weights, factors, apertures, inv_dx2)
            np.testing.assert_array_equal(
                heat_transfer.cut_cell_conduction_rates_serial(*args),
                heat_transfer.cut_cell_conduction_rates_parallel(*args),
            )
            np.testing.assert_array_equal(
                heat_transfer.compute_gradient_magnitude_serial(T, neighbors, 0.00381),
                heat_transfer.compute_gradient_magnitude_parallel(T, neighbors, 0.00381),
            )

    def test_solidification_kernels(self) -> None:
        for n in (11, 5000, 40000):
            T = self.rng.uniform(1380.0, 1560.0, n)
            fs = np.clip(self.rng.uniform(-0.2, 1.2, n), 0.0, 1.0)
            committed = []
            for kernel in (
                solidification_physics.commit_solidification_update_parallel,
                solidification_physics.commit_solidification_update_serial,
            ):
                state = [fs.copy(), np.zeros(n), np.full(n, np.inf), np.full(n, np.inf), np.zeros(n)]
                kernel(T, T, state[0], state[1], state[2], state[3], state[4], True, 1500.0, 60.0, 10.0, 0.5, 1.9e9)
                committed.append(state)
            for parallel, serial in zip(*committed):
                np.testing.assert_array_equal(serial, parallel)

            # dt = 0 (the first update of a run): the parallel loop gives
            # inf/NaN latent rates and does not raise; the twin must do the same.
            zero_dt = []
            for kernel in (
                solidification_physics.commit_solidification_update_parallel,
                solidification_physics.commit_solidification_update_serial,
            ):
                state = [fs.copy(), np.zeros(n), np.full(n, np.inf), np.full(n, np.inf), np.zeros(n)]
                kernel(T, T, state[0], state[1], state[2], state[3], state[4], True, 1500.0, 60.0, 10.0, 0.0, 1.9e9)
                zero_dt.append(state)
            for parallel, serial in zip(*zero_dt):
                np.testing.assert_array_equal(serial, parallel)
            self.assertTrue(np.any(~np.isfinite(zero_dt[0][4])))

            corrected = []
            for kernel in (
                solidification_physics.apply_enthalpy_correction_parallel,
                solidification_physics.apply_enthalpy_correction_serial,
            ):
                out = np.empty(n)
                kernel(T, fs, out, 1500.0, 1440.0, 60.0, 310.0)
                corrected.append(out)
            np.testing.assert_array_equal(corrected[1], corrected[0])


@unittest.skipUnless(NUMBA, "numba not installed")
class ModelUsesTwinTests(unittest.TestCase):
    """A model small enough for the twins gives the same field as one forced onto the pool."""

    def _run(self, budget_ns: float) -> np.ndarray:
        from types import SimpleNamespace
        from unittest.mock import patch

        n_side = 9
        n = n_side**3
        index = np.arange(n).reshape(n_side, n_side, n_side)
        neighbors = np.full((n, 26), -1, dtype=np.int32)
        ii, jj, kk = np.indices(index.shape)
        for slot, (dx, dy, dz) in enumerate([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]):
            i2, j2, k2 = ii + dx, jj + dy, kk + dz
            ok = (i2 >= 0) & (i2 < n_side) & (j2 >= 0) & (j2 < n_side) & (k2 >= 0) & (k2 < n_side)
            neighbors[index[ok], slot] = index[i2[ok], j2[ok], k2[ok]]
        alloy = SimpleNamespace(
            density_liquid=7000.0, specific_heat=700.0, thermal_diffusivity=6e-6,
            solidus_temp=1440.0, liquidus_temp=1500.0,
        )
        curves = {
            "temp_points": np.array([25.0, 400.0, 800.0, 1200.0, 1450.0, 1600.0]),
            "conductivity": np.array([50.0, 42.0, 30.0, 28.0, 30.0, 32.0]),
            "density": np.array([7850.0, 7730.0, 7600.0, 7450.0, 7300.0, 7000.0]),
            "specific_heat": np.array([450.0, 600.0, 900.0, 650.0, 700.0, 800.0]),
        }
        rng = np.random.default_rng(3)
        weights = rng.uniform(0.3, 1.0, n)
        with patch.object(kernel_dispatch, "SERIAL_KERNEL_BUDGET_NS", budget_ns):
            model = heat_transfer.HeatTransferModel(
                n_voxels=n, voxel_size=0.005, neighbors=neighbors, alloy=alloy,
                use_temperature_dependent=True, property_curves=curves,
                capacity_weights=weights, boundary_areas=np.where(weights < 0.6, 2.5e-5, 0.0),
            )
            model.set_initial_temperature(rng.uniform(1300.0, 1560.0, n))
            model.add_boundary(
                np.flatnonzero(weights < 0.6), heat_transfer.BoundaryType.CONVECTION, htc=400.0, T_infinity=25.0
            )
            for _ in range(25):
                model.step(0.5)
            gradient = model.get_gradient_magnitude(neighbors, 0.005)
            return np.concatenate([model.temperatures, gradient.astype(np.float64)])

    def test_twin_and_pool_give_the_same_field(self) -> None:
        np.testing.assert_array_equal(self._run(kernel_dispatch.SERIAL_KERNEL_BUDGET_NS), self._run(0.0))


if __name__ == "__main__":
    unittest.main()
