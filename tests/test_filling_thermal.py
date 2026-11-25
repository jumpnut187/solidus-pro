from __future__ import annotations

import unittest

import numpy as np

from physics.fluid_flow import equilibrium, lattice_units
from physics.filling import (
    FLAG_FLUID,
    FLAG_GAS,
    FLAG_INTERFACE,
    NO_TEMPERATURE,
    REFERENCE_DENSITY,
    TRANSITION_HYSTERESIS,
    advect_temperature,
    apply_transitions,
    apply_wall_heat_loss,
    create_state,
    step,
)

WCB_NU = 0.006 / 7200.0
DX_M = 0.0152
K_WCB = 45.0
RHO_WCB = 7200.0
CP_WCB = 600.0

POUR_TEMP = 1600.0
MOLD_TEMP = 21.0
HTC_CAST = 500.0

def _units(dx_m: float = DX_M, u_max_physical_m_s: float = 2.8):
    return lattice_units(dx_m, WCB_NU, u_max_physical_m_s=u_max_physical_m_s, u_max_lattice=0.05)

def _alpha_lattice(units, *, k=K_WCB, rho=RHO_WCB, cp=CP_WCB, dx_m=DX_M) -> float:
    alpha_si = k / (rho * cp)
    return alpha_si * units.dt_s / (dx_m * dx_m)

def _closed_box(n: int = 10) -> np.ndarray:
    obstacle = np.zeros((n, n, n), dtype=bool)
    obstacle[0] = obstacle[-1] = True
    obstacle[:, 0] = obstacle[:, -1] = True
    obstacle[:, :, 0] = obstacle[:, :, -1] = True
    return obstacle

def _open_channel(nx: int = 6, ny: int = 6, nz: int = 24) -> np.ndarray:
    obstacle = np.zeros((nx, ny, nz), dtype=bool)
    obstacle[0] = obstacle[-1] = True
    obstacle[:, 0] = obstacle[:, -1] = True
    obstacle[:, :, 0] = True
    return obstacle

def _driven_fill(steps: int, *, wall_loss: bool = True, htc: float = HTC_CAST, nz: int = 24):
    obstacle = _open_channel(nz=nz)
    units = _units()
    seed_z = nz - 4
    seed = np.zeros(obstacle.shape, dtype=bool)
    seed[1:-1, 1:-1, seed_z] = True
    state = create_state(
        obstacle,
        dx_m=DX_M,
        dt_s=units.dt_s,
        initial_fluid=seed,
        initial_temperature_c=POUR_TEMP,
    )
    inlet = np.argwhere(seed)
    ix, iy, iz = inlet[:, 0], inlet[:, 1], inlet[:, 2]
    alpha_lattice = _alpha_lattice(units)

    for _ in range(steps):
        state.flags[ix, iy, iz] = FLAG_FLUID
        state.mass[ix, iy, iz] = REFERENCE_DENSITY
        state.f[:, ix, iy, iz] = equilibrium(
            np.full(len(inlet), REFERENCE_DENSITY, dtype=np.float64),
            state.u[:, ix, iy, iz].astype(np.float64),
        ).astype(np.float32)
        state.temperature[ix, iy, iz] = POUR_TEMP

        step(state, units, track_free_surface=True)
        advect_temperature(state, alpha_lattice)
        if wall_loss:
            apply_wall_heat_loss(
                state,
                htc_w_m2k=htc,
                mold_temperature_c=MOLD_TEMP,
                density_kg_m3=RHO_WCB,
                specific_heat_j_kgk=CP_WCB,
            )
    return state, seed_z

class StabilityMarginTests(unittest.TestCase):

    def test_advective_cfl_matches_the_lbm_mach_guard(self) -> None:
        units = _units()
        self.assertAlmostEqual(units.u_max_lattice, 0.05)
        self.assertLess(units.u_max_lattice, 1.0)

    def test_lattice_thermal_diffusivity_is_order_1em5(self) -> None:
        units = _units()
        alpha_lattice = _alpha_lattice(units)
        self.assertGreater(alpha_lattice, 1e-6)
        self.assertLess(alpha_lattice, 1e-4)

    def test_lattice_thermal_diffusivity_is_within_the_laplacian_limit(self) -> None:
        units = _units()
        alpha_lattice = _alpha_lattice(units)
        self.assertLess(alpha_lattice, 1.0 / 6.0)

    def test_combined_scheme_keeps_a_nonnegative_centre_weight(self) -> None:
        units = _units()
        alpha_lattice = _alpha_lattice(units)
        worst_case_cfl_sum = 3.0 * units.u_max_lattice
        centre_weight = 1.0 - worst_case_cfl_sum - 6.0 * alpha_lattice
        self.assertGreater(centre_weight, 0.0)

class ConservativeTests(unittest.TestCase):
    def test_a_uniform_field_stays_uniform_under_advection_alone(self) -> None:
        n = 10
        obstacle = _closed_box(n)
        units = _units()
        state = create_state(
            obstacle, dx_m=DX_M, dt_s=units.dt_s, initial_fluid=~obstacle,
            initial_temperature_c=POUR_TEMP,
        )
        alpha_lattice = _alpha_lattice(units)

        for _ in range(50):
            step(state, units, track_free_surface=False)
            advect_temperature(state, alpha_lattice)

        interior = state.temperature[~obstacle]
        self.assertTrue(bool(np.all(np.isfinite(interior))))
        self.assertAlmostEqual(float(interior.min()), POUR_TEMP, places=3)
        self.assertAlmostEqual(float(interior.max()), POUR_TEMP, places=3)

class DrivenFillInvariantTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.state, cls.seed_z = _driven_fill(steps=2500)
        cls.wet = (cls.state.flags == FLAG_FLUID) | (cls.state.flags == FLAG_INTERFACE)
        cls.gas = cls.state.flags == FLAG_GAS

    def test_wet_temperatures_never_exceed_pour_temperature(self) -> None:
        self.assertTrue(bool(np.all(self.state.temperature[self.wet] <= POUR_TEMP + 1e-6)))

    def test_wet_temperatures_never_undershoot_mold_temperature(self) -> None:
        self.assertTrue(bool(np.all(self.state.temperature[self.wet] >= MOLD_TEMP - 1e-6)))

    def test_metal_further_from_the_inlet_is_cooler(self) -> None:
        nz = self.state.shape[2]
        bottom = self.wet[:, :, 1 : self.seed_z // 2 + 1]
        top = self.wet[:, :, self.seed_z - 2 : self.seed_z + 1]
        self.assertTrue(bool(bottom.any()))
        self.assertTrue(bool(top.any()))

        bottom_mean = float(self.state.temperature[:, :, 1 : self.seed_z // 2 + 1][bottom].mean())
        top_mean = float(self.state.temperature[:, :, self.seed_z - 2 : self.seed_z + 1][top].mean())
        self.assertLess(bottom_mean, top_mean)

    def test_gas_cells_carry_no_temperature(self) -> None:
        self.assertTrue(bool(np.all(np.isnan(self.state.temperature[self.gas]))))

    def test_no_nan_leaks_into_the_wet_region(self) -> None:
        self.assertTrue(bool(np.all(np.isfinite(self.state.temperature[self.wet]))))

class WallHeatLossTests(unittest.TestCase):
    WALL_CELL = (1, 3, 3)
    OPEN_CELL = (3, 3, 3)

    @staticmethod
    def _fresh_state(n: int = 6):
        obstacle = np.zeros((n, n, n), dtype=bool)
        obstacle[0] = True
        units = _units()
        state = create_state(
            obstacle, dx_m=DX_M, dt_s=units.dt_s, initial_fluid=~obstacle,
            initial_temperature_c=POUR_TEMP,
        )
        return state

    def _apply(self, state, htc: float) -> None:
        apply_wall_heat_loss(
            state,
            htc_w_m2k=htc,
            mold_temperature_c=MOLD_TEMP,
            density_kg_m3=RHO_WCB,
            specific_heat_j_kgk=CP_WCB,
        )

    def test_a_cell_against_the_wall_cools(self) -> None:
        state = self._fresh_state()
        self._apply(state, HTC_CAST)
        self.assertLess(float(state.temperature[self.WALL_CELL]), POUR_TEMP)
        self.assertGreater(float(state.temperature[self.WALL_CELL]), MOLD_TEMP)

    def test_a_cell_in_open_metal_does_not_lose_heat(self) -> None:
        state = self._fresh_state()
        self._apply(state, HTC_CAST)
        self.assertEqual(float(state.temperature[self.OPEN_CELL]), POUR_TEMP)

    def test_doubling_h_roughly_doubles_the_loss(self) -> None:
        low = self._fresh_state()
        before = float(low.temperature[self.WALL_CELL])
        self._apply(low, HTC_CAST)
        delta_low = float(low.temperature[self.WALL_CELL]) - before

        high = self._fresh_state()
        self._apply(high, 2.0 * HTC_CAST)
        delta_high = float(high.temperature[self.WALL_CELL]) - before

        self.assertLess(delta_low, 0.0)
        self.assertLess(delta_high, 0.0)
        self.assertAlmostEqual(delta_high / delta_low, 2.0, delta=0.15)

class PromotedCellTemperatureTests(unittest.TestCase):

    def test_promoted_cell_inherits_its_only_donors_temperature(self) -> None:
        n = 5
        obstacle = np.zeros((n, n, n), dtype=bool)
        units = _units()
        state = create_state(obstacle, dx_m=DX_M, dt_s=units.dt_s)

        a = (2, 2, 2)
        b = (2, 2, 3)
        c = (2, 2, 4)

        state.flags[a] = FLAG_FLUID
        state.mass[a] = 1.0
        state.rho[a] = 1.0
        state.temperature[a] = 1600.0

        state.flags[b] = FLAG_INTERFACE
        state.mass[b] = 1.0 + 2 * TRANSITION_HYSTERESIS
        state.rho[b] = 1.0
        state.temperature[b] = 1400.0

        self.assertTrue(bool(np.isnan(state.temperature[c])))

        filled, _emptied = apply_transitions(state, time_s=1.0)

        self.assertEqual(filled, 1)
        self.assertEqual(int(state.flags[c]), int(FLAG_INTERFACE))
        self.assertTrue(bool(np.isfinite(state.temperature[c])))
        self.assertAlmostEqual(float(state.temperature[c]), 1400.0, places=3)
        self.assertGreater(float(state.temperature[c]), MOLD_TEMP)
        self.assertLessEqual(float(state.temperature[c]), POUR_TEMP)

    def test_an_emptied_cell_loses_its_temperature(self) -> None:
        n = 5
        obstacle = np.zeros((n, n, n), dtype=bool)
        units = _units()
        state = create_state(obstacle, dx_m=DX_M, dt_s=units.dt_s)

        cell = (2, 2, 2)
        state.flags[cell] = FLAG_INTERFACE
        state.mass[cell] = -2 * TRANSITION_HYSTERESIS
        state.rho[cell] = 1.0
        state.temperature[cell] = 1234.0

        _filled, emptied = apply_transitions(state, time_s=1.0)

        self.assertEqual(emptied, 1)
        self.assertEqual(int(state.flags[cell]), int(FLAG_GAS))
        self.assertTrue(bool(np.isnan(state.temperature[cell])))

if __name__ == "__main__":
    unittest.main()

class HandoverFiniteTests(unittest.TestCase):

    def test_drained_cells_do_not_leak_nan_to_the_handover(self) -> None:
        import numpy as np

        from physics.filling import NO_TEMPERATURE

        fill_time = np.array([0.1, 0.4, -1.0, 0.2], dtype=np.float64)
        temperature_fill = np.array([1600.0, NO_TEMPERATURE, np.nan, 1580.0])
        filled_voxels = fill_time >= 0.0
        voxel_metal = np.ones(4, dtype=bool)
        mold_temp = 21.0

        mask = filled_voxels & voxel_metal & np.isfinite(temperature_fill)
        out = np.full(4, mold_temp, dtype=np.float64)
        out[mask] = temperature_fill[mask]

        self.assertTrue(bool(np.all(np.isfinite(out))))
        self.assertAlmostEqual(out[1], mold_temp)
        self.assertAlmostEqual(out[2], mold_temp)
        self.assertAlmostEqual(out[0], 1600.0)

    def test_fill_time_alone_is_not_sufficient(self) -> None:
        import numpy as np

        from physics.filling import NO_TEMPERATURE

        fill_time = np.array([0.4], dtype=np.float64)
        temperature_fill = np.array([NO_TEMPERATURE])
        naive = np.full(1, 21.0)
        naive[fill_time >= 0.0] = temperature_fill[fill_time >= 0.0]
        self.assertFalse(
            bool(np.all(np.isfinite(naive))),
            "the naive mask no longer reproduces the bug; the guard may be moot",
        )
