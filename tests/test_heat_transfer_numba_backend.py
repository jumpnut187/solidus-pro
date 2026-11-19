from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import physics.heat_transfer as heat_transfer


class _DummyAlloy:
    thermal_diffusivity = 1.0e-5
    density_liquid = 7200.0
    specific_heat = 600.0
    liquidus_temp = 1500.0
    solidus_temp = 1450.0


class HeatTransferNumbaBackendTests(unittest.TestCase):
    def _build_model(
        self,
        *,
        use_temperature_dependent: bool = False,
        diffusivity_overrides=None,
        rho_cp_overrides=None,
    ) -> heat_transfer.HeatTransferModel:
        neighbors = np.array(
            [
                [1, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, -1],
            ],
            dtype=np.int32,
        )
        model = heat_transfer.HeatTransferModel(
            n_voxels=2,
            voxel_size=1.0e-3,
            neighbors=neighbors,
            alloy=_DummyAlloy(),
            use_temperature_dependent=use_temperature_dependent,
            diffusivity_overrides=diffusivity_overrides,
            rho_cp_overrides=rho_cp_overrides,
        )
        model.set_initial_temperature(np.array([1500.0, 1490.0], dtype=np.float64))
        return model

    def test_uses_numba_kernels_when_parallel_eligible(self) -> None:
        model = self._build_model(use_temperature_dependent=False)

        def _fake_laplacian(
            temperatures: np.ndarray,
            neighbors: np.ndarray,
            inv_dx2: np.ndarray,
        ) -> np.ndarray:
            del temperatures, neighbors, inv_dx2
            return np.array([-1000.0, -1000.0], dtype=np.float64)

        def _fake_update(
            temperatures: np.ndarray,
            laplacian: np.ndarray,
            source: np.ndarray,
            alpha: float,
            dt: float,
        ) -> np.ndarray:
            return temperatures + dt * (alpha * laplacian + source)

        with (
            patch.object(heat_transfer, "NUMBA_AVAILABLE", True),
            patch.object(heat_transfer, "compute_laplacian_parallel", side_effect=_fake_laplacian, create=True) as lap_mock,
            patch.object(heat_transfer, "update_temperatures_parallel", side_effect=_fake_update, create=True) as upd_mock,
            patch.object(heat_transfer, "update_heterogeneous_parallel", create=True) as het_mock,
        ):
            model._numba_parallel_eligible = True
            model.step(1.0e-3)

        self.assertTrue(lap_mock.called)
        self.assertTrue(upd_mock.called)
        self.assertFalse(het_mock.called)
        self.assertTrue(model.last_step_used_numba)

    def test_temperature_dependent_uses_heterogeneous_kernel(self) -> None:
        """Temperature-dependent alpha (a per-voxel array) now dispatches the
        conservative Numba flux kernel; the uniform Laplacian/update kernels
        are not called."""
        model = self._build_model(use_temperature_dependent=True)

        def _fake_heterogeneous(
            temperatures: np.ndarray,
            neighbors: np.ndarray,
            alpha: np.ndarray,
            rho_cp: np.ndarray,
            inv_dx2: np.ndarray,
            source: np.ndarray,
            dt: float,
            contact_hdx: np.ndarray,
        ) -> np.ndarray:
            del neighbors, rho_cp, inv_dx2, source, dt, contact_hdx
            return temperatures + 1.0 * alpha

        with (
            patch.object(heat_transfer, "NUMBA_AVAILABLE", True),
            patch.object(heat_transfer, "compute_laplacian_parallel", create=True) as lap_mock,
            patch.object(heat_transfer, "update_temperatures_parallel", create=True) as upd_mock,
            patch.object(
                heat_transfer,
                "update_heterogeneous_parallel",
                side_effect=_fake_heterogeneous,
                create=True,
            ) as het_mock,
        ):
            model._numba_parallel_eligible = False
            model.step(1.0e-3)

        self.assertFalse(lap_mock.called)
        self.assertFalse(upd_mock.called)
        self.assertTrue(het_mock.called)
        self.assertTrue(model.last_step_used_numba)

    def test_chill_overrides_use_heterogeneous_kernel(self) -> None:
        """Chill diffusivity/rho_cp overrides make the field heterogeneous and
        dispatch the conservative Numba flux kernel, not the uniform one."""
        override_idx = np.array([1], dtype=np.int64)
        model = self._build_model(
            diffusivity_overrides=(override_idx, np.array([1.0e-4], dtype=np.float64)),
            rho_cp_overrides=(override_idx, np.array([2.0 * 7200.0 * 600.0], dtype=np.float64)),
        )

        def _fake_heterogeneous(
            temperatures: np.ndarray,
            neighbors: np.ndarray,
            alpha: np.ndarray,
            rho_cp: np.ndarray,
            inv_dx2: np.ndarray,
            source: np.ndarray,
            dt: float,
            contact_hdx: np.ndarray,
        ) -> np.ndarray:
            del neighbors, alpha, rho_cp, inv_dx2, source, dt, contact_hdx
            return temperatures.copy()

        with (
            patch.object(heat_transfer, "NUMBA_AVAILABLE", True),
            patch.object(heat_transfer, "compute_laplacian_parallel", create=True) as lap_mock,
            patch.object(heat_transfer, "update_temperatures_parallel", create=True) as upd_mock,
            patch.object(
                heat_transfer,
                "update_heterogeneous_parallel",
                side_effect=_fake_heterogeneous,
                create=True,
            ) as het_mock,
        ):
            # Uniform eligibility alone must NOT route an alpha-array step into
            # the uniform kernels.
            model._numba_parallel_eligible = True
            model.step(1.0e-3)

        self.assertFalse(lap_mock.called)
        self.assertFalse(upd_mock.called)
        self.assertTrue(het_mock.called)
        self.assertTrue(model.last_step_used_numba)


if __name__ == "__main__":
    unittest.main()
