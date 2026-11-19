"""
Regression tests for the heat-transfer physics fixes in
``physics/heat_transfer.py``.

Covers:
- Correct isotropic 27-point weighted Laplacian (exact for quadratics)
- Per-voxel voxel sizes and per-voxel diffusivity overrides
- Correct Robin (convection) boundary treatment
- Stability criterion for the weighted stencil
- Numba / NumPy kernel parity
"""

from __future__ import annotations

import itertools
import unittest
from unittest.mock import patch

import numpy as np

import physics.heat_transfer as heat_transfer


class _FakeAlloy:
    """Unit-alloy so alpha = 1.0 and rho_cp = 1.0 make checks arithmetic."""

    density_liquid = 1.0
    density_solid = 1.0
    specific_heat = 1.0
    thermal_conductivity = 1.0
    latent_heat = 0.0
    liquidus_temp = 999.0
    solidus_temp = 1.0
    pouring_temp = 990.0

    @property
    def thermal_diffusivity(self) -> float:
        return 1.0


def _build_grid(n: int = 9):
    """
    Build an n x n x n canonical neighbor array.

    Returns (neighbors, n_vox, idx) where idx(i, j, k) is the linear voxel
    index. Slots follow the canonical layout: 0-5 faces, 6-17 edges,
    18-25 corners. Neighbor index is -1 when out of bounds.
    """
    n_vox = n * n * n
    idx = lambda i, j, k: i * n * n + j * n + k
    neighbors = np.full((n_vox, 26), -1, dtype=np.int32)

    faces = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]
    edges = [
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
        (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
    ]
    corners = list(itertools.product([1, -1], [1, -1], [1, -1]))

    for i in range(n):
        for j in range(n):
            for k in range(n):
                vox = idx(i, j, k)
                slot = 0
                for di, dj, dk in faces + edges + corners:
                    ni, nj, nk = i + di, j + dj, k + dk
                    if 0 <= ni < n and 0 <= nj < n and 0 <= nk < n:
                        neighbors[vox, slot] = idx(ni, nj, nk)
                    slot += 1
    return neighbors, n_vox, idx


def _build_model(n: int = 9, **kwargs) -> heat_transfer.HeatTransferModel:
    """Build a HeatTransferModel on the canonical grid with dx = 1.0."""
    neighbors, n_vox, _ = _build_grid(n)
    return heat_transfer.HeatTransferModel(
        n_voxels=n_vox,
        voxel_size=1.0,
        neighbors=neighbors,
        alloy=_FakeAlloy(),
        **kwargs,
    )


def _quadratic_field(n: int = 9) -> np.ndarray:
    """T = x^2 + y^2 + z^2 with float voxel-center coordinates 0..n-1."""
    T = np.zeros(n * n * n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                T[i * n * n + j * n + k] = float(i * i + j * j + k * k)
    return T


class HeatTransferPhysicsTests(unittest.TestCase):
    def test_quadratic_laplacian(self) -> None:
        """Laplacian of x^2+y^2+z^2 must give 6 at interior voxels."""
        n = 9
        neighbors, n_vox, idx = _build_grid(n)
        model = heat_transfer.HeatTransferModel(
            n_voxels=n_vox,
            voxel_size=1.0,
            neighbors=neighbors,
            alloy=_FakeAlloy(),
        )
        T0 = _quadratic_field(n)
        model.set_initial_temperature(T0)
        dt = 1.0e-4
        model.step(dt)

        interior = idx(4, 4, 4)
        dT_dt = (model.temperatures[interior] - T0[interior]) / dt
        self.assertGreaterEqual(dT_dt, 5.4)
        self.assertLessEqual(dT_dt, 6.6)

    def test_1d_diffusion_laplacian(self) -> None:
        """Laplacian of (x-4)^2 must give 2 at interior voxels."""
        n = 9
        neighbors, n_vox, idx = _build_grid(n)
        model = heat_transfer.HeatTransferModel(
            n_voxels=n_vox,
            voxel_size=1.0,
            neighbors=neighbors,
            alloy=_FakeAlloy(),
        )
        T0 = np.zeros(n_vox, dtype=np.float64)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    T0[idx(i, j, k)] = (i - 4.0) ** 2
        model.set_initial_temperature(T0)
        dt = 1.0e-4
        model.step(dt)

        interior = idx(4, 4, 4)
        dT_dt = (model.temperatures[interior] - T0[interior]) / dt
        self.assertGreaterEqual(dT_dt, 1.8)
        self.assertLessEqual(dT_dt, 2.2)

    def test_per_voxel_sizes_accepted(self) -> None:
        """Uniform per-voxel sizes must not change the quadratic result."""
        n = 9
        neighbors, n_vox, idx = _build_grid(n)
        model = heat_transfer.HeatTransferModel(
            n_voxels=n_vox,
            voxel_size=1.0,
            neighbors=neighbors,
            alloy=_FakeAlloy(),
            voxel_sizes=np.full(n_vox, 1.0, dtype=np.float64),
        )
        T0 = _quadratic_field(n)
        model.set_initial_temperature(T0)
        dt = 1.0e-4
        model.step(dt)  # must not raise

        interior = idx(4, 4, 4)
        dT_dt = (model.temperatures[interior] - T0[interior]) / dt
        self.assertGreaterEqual(dT_dt, 5.4)
        self.assertLessEqual(dT_dt, 6.6)

    def test_diffusivity_overrides(self) -> None:
        """A 10x alpha override voxel responds faster than base, bounded by
        the conservative two-point flux: the interface conductance is the
        harmonic mean of the pair (k_ij = 2*k_i*k_j/(k_i+k_j)), so an
        overridden voxel surrounded by base voxels responds at most 2x the
        true base rate (harmonic mean of 10 and 1 = 1.818), and never pumps
        energy (the interface cannot exceed the slow-side conductance)."""
        n = 9
        neighbors, n_vox, idx = _build_grid(n)
        interior = idx(4, 4, 4)
        override_idx = np.array([interior], dtype=np.int64)
        override_vals = np.array([10.0], dtype=np.float64)
        model = heat_transfer.HeatTransferModel(
            n_voxels=n_vox,
            voxel_size=1.0,
            neighbors=neighbors,
            alloy=_FakeAlloy(),
            diffusivity_overrides=(override_idx, override_vals),
        )
        T0 = _quadratic_field(n)
        model.set_initial_temperature(T0)
        dt = 1.0e-4
        model.step(dt)

        # True base rate for the quadratic field is 6.0 K/s (alpha = 1).
        dT_dt_override = (model.temperatures[interior] - T0[interior]) / dt
        self.assertGreater(dT_dt_override, 6.0)
        self.assertLessEqual(dT_dt_override, 2.0 * 6.0 + 0.05)
        # Harmonic mean of (10, 1) = 1.818 -> expected ~10.909 K/s.
        self.assertAlmostEqual(dT_dt_override, 10.909, delta=0.05)

    def test_robin_boundary_direction(self) -> None:
        """Convection cools the boundary voxel only, by the explicit flux."""
        n = 9
        neighbors, n_vox, idx = _build_grid(n)
        model = heat_transfer.HeatTransferModel(
            n_voxels=n_vox,
            voxel_size=1.0,
            neighbors=neighbors,
            alloy=_FakeAlloy(),
        )
        model.set_initial_temperature(500.0)

        boundary_voxel = idx(0, 4, 4)
        model.add_boundary(
            np.array([boundary_voxel], dtype=np.int32),
            heat_transfer.BoundaryType.CONVECTION,
            htc=100.0,
            T_infinity=25.0,
        )
        dt = 1.0e-4
        model.step(dt)

        # Boundary voxel cooled; interior voxel (uniform field) unchanged.
        self.assertLess(model.temperatures[boundary_voxel], 500.0)
        interior = idx(4, 4, 4)
        self.assertAlmostEqual(model.temperatures[interior], 500.0, delta=1.0e-9)

        # Magnitude matches dt*h*(T-T_inf)/(rho_cp*dx) within 5%.
        expected = dt * 100.0 * (500.0 - 25.0) / (1.0 * 1.0 * 1.0)
        actual = 500.0 - model.temperatures[boundary_voxel]
        self.assertAlmostEqual(actual, expected, delta=0.05 * expected)

    def test_stable_timestep_positive_and_scales(self) -> None:
        """dt_stable > 0 and shrinks when the voxel size is halved."""
        n = 9
        neighbors, n_vox, _ = _build_grid(n)
        model_full = heat_transfer.HeatTransferModel(
            n_voxels=n_vox,
            voxel_size=1.0,
            neighbors=neighbors,
            alloy=_FakeAlloy(),
        )
        model_half = heat_transfer.HeatTransferModel(
            n_voxels=n_vox,
            voxel_size=1.0,
            neighbors=neighbors,
            alloy=_FakeAlloy(),
            voxel_sizes=np.full(n_vox, 0.5, dtype=np.float64),
        )
        dt_full = model_full.get_stable_timestep()
        dt_half = model_half.get_stable_timestep()
        self.assertGreater(dt_full, 0.0)
        self.assertLess(dt_half, dt_full)

    def test_numba_numpy_parity(self) -> None:
        """Numba and NumPy Laplacian paths must agree to 1e-9."""
        n = 9
        neighbors, n_vox, _ = _build_grid(n)
        T0 = _quadratic_field(n)
        dt = 1.0e-4

        def _run() -> np.ndarray:
            model = _build_model(n)
            model.set_initial_temperature(T0)
            model.step(dt)
            return model.temperatures.copy()

        expected = _run()
        with patch.object(heat_transfer, "NUMBA_AVAILABLE", False):
            actual = _run()

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-9)

    @unittest.skipUnless(heat_transfer.NUMBA_AVAILABLE, "numba not installed")
    def test_heterogeneous_numba_numpy_parity(self) -> None:
        """Chill-override steps (diffusivity + rho_cp arrays) must be identical
        whether the conservative flux runs through the Numba kernel or the
        NumPy reference (the existing NumPy fallback)."""
        n = 5
        neighbors, n_vox, idx = _build_grid(n)
        chill_mask = np.zeros(n_vox, dtype=bool)
        for i in range(n):
            for j in range(n):
                chill_mask[idx(i, j, 0)] = True
        chill_idx = np.flatnonzero(chill_mask).astype(np.int64)
        T0 = np.full(n_vox, 1000.0, dtype=np.float64)
        T0[chill_idx] = 20.0
        source = np.full(n_vox, 5.0, dtype=np.float64)

        def _build() -> heat_transfer.HeatTransferModel:
            return heat_transfer.HeatTransferModel(
                n_voxels=n_vox,
                voxel_size=1.0,
                neighbors=neighbors,
                alloy=_FakeAlloy(),
                diffusivity_overrides=(chill_idx, np.full(len(chill_idx), 10.4)),
                rho_cp_overrides=(chill_idx, np.full(len(chill_idx), 0.296)),
            )

        def _run() -> np.ndarray:
            model = _build()
            model.set_initial_temperature(T0)
            model.set_heat_source(source)
            for _ in range(10):
                model.step(1.0e-3)
            return model.temperatures.copy()

        # Activation assertion: the default path must use the Numba kernel.
        expected_model = _build()
        expected_model.set_initial_temperature(T0)
        expected_model.set_heat_source(source)
        for _ in range(10):
            expected_model.step(1.0e-3)
        self.assertTrue(expected_model.last_step_used_numba)
        expected = expected_model.temperatures.copy()

        with patch.object(heat_transfer, "NUMBA_AVAILABLE", False):
            actual = _run()

        np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)

    @unittest.skipUnless(heat_transfer.NUMBA_AVAILABLE, "numba not installed")
    def test_temperature_dependent_numba_numpy_parity(self) -> None:
        """Temperature-dependent alpha steps must be identical whether the
        conservative flux runs through the Numba kernel or the NumPy
        reference."""
        n = 5
        neighbors, n_vox, _ = _build_grid(n)
        curves = {
            "temp_points": np.array([20.0, 600.0, 1000.0, 1200.0, 1450.0, 1500.0]),
            "conductivity": np.array([35.0, 33.0, 30.0, 28.0, 26.0, 25.0]),
            "specific_heat": np.array([500.0, 550.0, 600.0, 630.0, 660.0, 680.0]),
            "density": np.array([7850.0, 7800.0, 7700.0, 7600.0, 7400.0, 7300.0]),
        }
        T0 = np.linspace(300.0, 1500.0, n_vox)

        def _build() -> heat_transfer.HeatTransferModel:
            return heat_transfer.HeatTransferModel(
                n_voxels=n_vox,
                voxel_size=1.0,
                neighbors=neighbors,
                alloy=_FakeAlloy(),
                use_temperature_dependent=True,
                property_curves=curves,
            )

        def _run() -> np.ndarray:
            model = _build()
            model.set_initial_temperature(T0)
            for _ in range(10):
                model.step(1.0e-4)
            return model.temperatures.copy()

        expected_model = _build()
        expected_model.set_initial_temperature(T0)
        for _ in range(10):
            expected_model.step(1.0e-4)
        self.assertTrue(expected_model.last_step_used_numba)
        expected = expected_model.temperatures.copy()

        with patch.object(heat_transfer, "NUMBA_AVAILABLE", False):
            actual = _run()

        np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)

    @unittest.skipUnless(heat_transfer.NUMBA_AVAILABLE, "numba not installed")
    def test_rho_cp_only_override_numba_numpy_parity(self) -> None:
        """rho_cp overrides alone (scalar alpha promoted to an array for the
        kernel) must match the NumPy reference."""
        n = 5
        neighbors, n_vox, idx = _build_grid(n)
        override_idx = np.array([idx(2, 2, 2)], dtype=np.int64)
        T0 = _quadratic_field(n)
        source = np.full(n_vox, 1.0, dtype=np.float64)

        def _build() -> heat_transfer.HeatTransferModel:
            return heat_transfer.HeatTransferModel(
                n_voxels=n_vox,
                voxel_size=1.0,
                neighbors=neighbors,
                alloy=_FakeAlloy(),
                rho_cp_overrides=(override_idx, np.array([4.0], dtype=np.float64)),
            )

        def _run() -> np.ndarray:
            model = _build()
            model.set_initial_temperature(T0)
            model.set_heat_source(source)
            for _ in range(10):
                model.step(1.0e-4)
            return model.temperatures.copy()

        expected_model = _build()
        expected_model.set_initial_temperature(T0)
        expected_model.set_heat_source(source)
        for _ in range(10):
            expected_model.step(1.0e-4)
        self.assertTrue(expected_model.last_step_used_numba)
        expected = expected_model.temperatures.copy()

        with patch.object(heat_transfer, "NUMBA_AVAILABLE", False):
            actual = _run()

        np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)

    def test_heterogeneous_interface_conserves_energy(self) -> None:
        """
        Regression: at a material interface (cast vs graphite chill) the
        update must conserve energy. The old per-voxel-diffusivity scheme
        applied each voxel's OWN alpha to the shared interface gradient, so
        the high-diffusivity side gained energy at k_hi/k_lo times the rate
        the low side lost it (a numerical heat pump that self-heated the
        chill and insulated the casting). The conservative two-point flux
        uses the harmonic-mean pair conductivity and per-voxel heat capacity.
        """
        n = 5
        neighbors, n_vox, idx = _build_grid(n)
        # Make the z=0 plane (indices with k == 0) a "chill" region.
        chill_mask = np.zeros(n_vox, dtype=bool)
        for i in range(n):
            for j in range(n):
                chill_mask[idx(i, j, 0)] = True
        chill_idx = np.flatnonzero(chill_mask).astype(np.int64)

        alpha_chill = 1.17e-4 / 1.12e-5  # ~10.4x steel diffusivity
        rho_cp_chill = 1800.0 * 710.0 / (7200.0 * 600.0)  # graphite/steel ratio
        model = heat_transfer.HeatTransferModel(
            n_voxels=n_vox,
            voxel_size=1.0,
            neighbors=neighbors,
            alloy=_FakeAlloy(),
            diffusivity_overrides=(chill_idx, np.full(len(chill_idx), alpha_chill)),
            rho_cp_overrides=(chill_idx, np.full(len(chill_idx), rho_cp_chill)),
        )

        T0 = np.full(n_vox, 1000.0, dtype=np.float64)
        T0[chill_idx] = 20.0
        model.set_initial_temperature(T0)
        model.set_heat_source(np.zeros(n_vox))

        rho_cp_cast = 1.0  # unit alloy
        rho_cp_chill_scale = float(rho_cp_chill)  # graphite/steel ratio applied by the model
        V = 1.0  # unit voxel volume (dx = 1)

        def _total_energy(T: np.ndarray) -> float:
            return float(
                np.sum(T[~chill_mask] * rho_cp_cast * V)
                + np.sum(T[chill_mask] * rho_cp_chill_scale * V)
            )

        E_before = _total_energy(model.temperatures)
        for _ in range(200):
            model.step(1.0e-3)
        E_after = _total_energy(model.temperatures)

        # Total energy must be conserved exactly: the cast voxels lose
        # exactly what the chill voxels gain (each side booked with its own
        # heat capacity, matching the model's per-voxel rho_cp overrides).
        np.testing.assert_allclose(E_after, E_before, rtol=1e-9, atol=1e-9)

        # And the system must have moved toward equilibrium (heat flowed
        # from the hot cast into the cold chill).
        self.assertLess(np.mean(model.temperatures[chill_mask]), 1000.0)
        self.assertGreater(np.mean(model.temperatures[chill_mask]), 20.0)

    def test_stability_includes_diffusivity_overrides(self) -> None:
        """
        Regression: get_stable_timestep() must account for per-voxel
        diffusivity overrides (e.g., graphite chills at ~10x the alloy alpha).
        Before this fix the CFL bound ignored the overrides, so the explicit
        scheme stepped at dt larger than the chill stability limit and the
        chill voxels blew up to +/-inf within a few hundred steps (which then
        surfaced as NaN fraction solid and a ValueError in the progress
        callback).
        """
        n = 9
        neighbors, n_vox, _ = _build_grid(n)
        override_idx = np.array([0], dtype=np.int64)
        override_alpha = 10.0  # 10x the alloy diffusivity (alpha = 1.0)
        model = heat_transfer.HeatTransferModel(
            n_voxels=n_vox,
            voxel_size=1.0,
            neighbors=neighbors,
            alloy=_FakeAlloy(),
            diffusivity_overrides=(override_idx, np.array([override_alpha])),
        )

        self.assertAlmostEqual(model._get_alpha_max(), override_alpha)
        dt_stable = model.get_stable_timestep()
        self.assertAlmostEqual(dt_stable, 0.9 / (4.0 * override_alpha))

        # A step of dt=1e-2 (fine for alpha=1 but far beyond the chill
        # stability limit) must subdivide internally and remain finite.
        T0 = np.full(n_vox, 500.0, dtype=np.float64)
        T0[0] = 20.0  # sharp gradient at the override voxel
        model.set_initial_temperature(T0)
        model.step(1.0e-2)
        self.assertTrue(np.isfinite(model.temperatures).all())

        # And repeated stepping at the loop's 0.1 s cadence stays finite.
        for _ in range(50):
            model.step(0.1)
            self.assertTrue(np.isfinite(model.temperatures).all())


if __name__ == "__main__":
    unittest.main()
