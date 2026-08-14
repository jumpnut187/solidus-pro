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
        """10x alpha override must produce ~10x dT/dt at that voxel."""
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

        neighbor = idx(4, 5, 4)  # +Y neighbor, un-overridden
        dT_dt_override = (model.temperatures[interior] - T0[interior]) / dt
        dT_dt_base = (model.temperatures[neighbor] - T0[neighbor]) / dt
        self.assertAlmostEqual(dT_dt_override / dT_dt_base, 10.0, delta=0.05)

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
