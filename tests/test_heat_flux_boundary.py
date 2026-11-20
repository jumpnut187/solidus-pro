from __future__ import annotations

import unittest

import numpy as np

from core.materials import get_alloy
from physics.heat_transfer import BoundaryType, HeatTransferModel

def _make_model(n_voxels: int = 8) -> HeatTransferModel:
    neighbors = np.full((n_voxels, 2), -1, dtype=np.int32)
    for i in range(n_voxels):
        if i > 0:
            neighbors[i, 0] = i - 1
        if i < n_voxels - 1:
            neighbors[i, 1] = i + 1

    return HeatTransferModel(
        n_voxels=n_voxels,
        voxel_size=0.01,
        neighbors=neighbors,
        alloy=get_alloy("WCB"),
    )

class HeatFluxBoundaryTests(unittest.TestCase):
    def test_positive_flux_heats_the_boundary_cell(self) -> None:
        model = _make_model()
        model.set_initial_temperature(1000.0)
        model.add_boundary(
            indices=np.array([0], dtype=np.int32),
            boundary_type=BoundaryType.HEAT_FLUX,
            value=5.0e5,
        )

        before = float(model.temperatures[0])
        model.step(0.01)
        after = float(model.temperatures[0])

        self.assertGreater(
            after,
            before,
            msg="HEAT_FLUX boundary did not change the temperature at all",
        )

    def test_negative_flux_cools_the_boundary_cell(self) -> None:
        model = _make_model()
        model.set_initial_temperature(1000.0)
        model.add_boundary(
            indices=np.array([0], dtype=np.int32),
            boundary_type=BoundaryType.HEAT_FLUX,
            value=-5.0e5,
        )

        model.step(0.01)
        self.assertLess(float(model.temperatures[0]), 1000.0)

    def test_zero_flux_matches_no_boundary(self) -> None:
        with_bc = _make_model()
        with_bc.set_initial_temperature(1000.0)
        with_bc.add_boundary(
            indices=np.array([0], dtype=np.int32),
            boundary_type=BoundaryType.HEAT_FLUX,
            value=0.0,
        )
        with_bc.step(0.01)

        without = _make_model()
        without.set_initial_temperature(1000.0)
        without.step(0.01)

        np.testing.assert_allclose(with_bc.temperatures, without.temperatures, rtol=1e-12)

    def test_flux_scales_with_magnitude(self) -> None:
        deltas = []
        for q in (2.0e5, 4.0e5):
            model = _make_model()
            model.set_initial_temperature(1000.0)
            model.add_boundary(
                indices=np.array([0], dtype=np.int32),
                boundary_type=BoundaryType.HEAT_FLUX,
                value=q,
            )
            model.step(0.001)
            deltas.append(float(model.temperatures[0]) - 1000.0)

        self.assertAlmostEqual(deltas[1] / deltas[0], 2.0, places=6)

    def test_time_function_modulates_the_flux(self) -> None:
        model = _make_model()
        model.set_initial_temperature(1000.0)
        model.add_boundary(
            indices=np.array([0], dtype=np.int32),
            boundary_type=BoundaryType.HEAT_FLUX,
            value=5.0e5,
            time_function=lambda t: 0.0,
        )

        model.step(0.01)
        self.assertAlmostEqual(float(model.temperatures[0]), 1000.0, places=9)

class BoundaryMutationTests(unittest.TestCase):
    def test_boundaries_are_exposed_read_only(self) -> None:
        model = _make_model()
        model.add_boundary(
            indices=np.array([0], dtype=np.int32),
            boundary_type=BoundaryType.HEAT_FLUX,
            value=1.0,
        )
        listed = model.boundaries
        self.assertEqual(len(listed), 1)

        listed.clear()
        self.assertEqual(len(model.boundaries), 1)

    def test_clear_boundaries_removes_all(self) -> None:
        model = _make_model()
        idx = np.array([0], dtype=np.int32)
        model.add_boundary(indices=idx, boundary_type=BoundaryType.HEAT_FLUX, value=1.0)
        model.add_boundary(
            indices=idx, boundary_type=BoundaryType.CONVECTION, htc=100.0, T_infinity=25.0
        )

        self.assertEqual(model.clear_boundaries(), 2)
        self.assertEqual(model.boundaries, [])

    def test_clear_boundaries_by_type(self) -> None:
        model = _make_model()
        idx = np.array([0], dtype=np.int32)
        model.add_boundary(indices=idx, boundary_type=BoundaryType.HEAT_FLUX, value=1.0)
        model.add_boundary(
            indices=idx, boundary_type=BoundaryType.CONVECTION, htc=100.0, T_infinity=25.0
        )

        self.assertEqual(model.clear_boundaries(BoundaryType.HEAT_FLUX), 1)
        remaining = model.boundaries
        self.assertEqual(len(remaining), 1)
        self.assertIs(remaining[0].type, BoundaryType.CONVECTION)

    def test_readding_replaces_the_index_set(self) -> None:
        model = _make_model()
        model.set_initial_temperature(1000.0)
        model.add_boundary(
            indices=np.array([0], dtype=np.int32),
            boundary_type=BoundaryType.HEAT_FLUX,
            value=5.0e5,
        )
        model.step(0.001)

        model.clear_boundaries(BoundaryType.HEAT_FLUX)
        model.add_boundary(
            indices=np.array([7], dtype=np.int32),
            boundary_type=BoundaryType.HEAT_FLUX,
            value=5.0e5,
        )
        before_tail = float(model.temperatures[7])
        model.step(0.001)

        self.assertGreater(float(model.temperatures[7]), before_tail)

if __name__ == "__main__":
    unittest.main()
