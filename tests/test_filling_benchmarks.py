from __future__ import annotations

import os
import unittest
from dataclasses import replace

import numpy as np

from physics.fluid_flow import CS2, equilibrium, lattice_units, omega_minus_from_magic
from physics.filling import (
    FLAG_FLUID,
    FLAG_GAS,
    FLAG_INTERFACE,
    FLAG_OBSTACLE,
    create_state,
    fluid_fraction,
    step,
    total_mass,
)

WCB_NU = 0.006 / 7200.0
DX_M = 0.00254

def closed_box(n: int = 12) -> np.ndarray:
    obstacle = np.zeros((n, n, n), dtype=bool)
    obstacle[0] = obstacle[-1] = True
    obstacle[:, 0] = obstacle[:, -1] = True
    obstacle[:, :, 0] = obstacle[:, :, -1] = True
    return obstacle

def hydrostatic_state(n: int = 12):
    obstacle = closed_box(n)
    units = lattice_units(DX_M, WCB_NU, u_max_physical_m_s=1.0)
    state = create_state(obstacle, dx_m=DX_M, dt_s=units.dt_s, initial_fluid=~obstacle)

    z = np.arange(n, dtype=np.float64)
    rho_z = 1.0 - units.gravity_lattice * (z - z[n // 2]) / CS2
    rho = np.broadcast_to(rho_z[None, None, :], (n, n, n)).copy()
    rho[obstacle] = 0.0

    state.rho = rho.astype(np.float32)
    state.f = equilibrium(rho, np.zeros((3, n, n, n))).astype(np.float32)
    state.f[:, obstacle] = 0.0
    return state, units, obstacle

class QuiescenceTests(unittest.TestCase):

    def test_quiescent_fluid_stays_exactly_still(self) -> None:
        obstacle = closed_box(10)
        units = lattice_units(DX_M, WCB_NU, u_max_physical_m_s=1.0)
        state = create_state(obstacle, dx_m=DX_M, dt_s=units.dt_s, initial_fluid=~obstacle)
        for _ in range(50):
            step(state, units, gravity_direction=(0.0, 0.0, 0.0), track_free_surface=False)
        self.assertEqual(float(np.abs(state.u).max()), 0.0)

    def test_quiescent_density_stays_exactly_uniform(self) -> None:
        obstacle = closed_box(10)
        units = lattice_units(DX_M, WCB_NU, u_max_physical_m_s=1.0)
        state = create_state(obstacle, dx_m=DX_M, dt_s=units.dt_s, initial_fluid=~obstacle)
        for _ in range(50):
            step(state, units, gravity_direction=(0.0, 0.0, 0.0), track_free_surface=False)
        interior = state.rho[~obstacle]
        self.assertAlmostEqual(float(interior.min()), 1.0, places=6)
        self.assertAlmostEqual(float(interior.max()), 1.0, places=6)

    def test_solid_cells_never_hold_populations(self) -> None:
        obstacle = closed_box(10)
        units = lattice_units(DX_M, WCB_NU, u_max_physical_m_s=1.0)
        state = create_state(obstacle, dx_m=DX_M, dt_s=units.dt_s, initial_fluid=~obstacle)
        self.assertEqual(float(np.abs(state.f[:, obstacle]).max()), 0.0)
        for _ in range(10):
            step(state, units, track_free_surface=False)
            self.assertEqual(float(np.abs(state.f[:, obstacle]).max()), 0.0)

    def test_walls_are_impermeable(self) -> None:
        obstacle = closed_box(10)
        units = lattice_units(DX_M, WCB_NU, u_max_physical_m_s=1.0)
        state = create_state(obstacle, dx_m=DX_M, dt_s=units.dt_s, initial_fluid=~obstacle)
        before = total_mass(state)
        for _ in range(100):
            step(state, units, track_free_surface=False)
        self.assertAlmostEqual(total_mass(state) / before, 1.0, places=9)

class HydrostaticTests(unittest.TestCase):

    def test_analytic_profile_is_preserved(self) -> None:
        state, units, obstacle = hydrostatic_state()
        n = state.shape[2]
        column = slice(1, n - 1)
        initial = state.rho[n // 2, n // 2, column].astype(np.float64)
        initial_range = float(initial.max() - initial.min())
        for _ in range(2000):
            step(state, units, track_free_surface=False)
        final = state.rho[n // 2, n // 2, column].astype(np.float64)
        final_range = float(final.max() - final.min())
        self.assertAlmostEqual(final_range / initial_range, 1.0, delta=0.10)

    def test_gradient_matches_the_analytic_value(self) -> None:
        state, units, obstacle = hydrostatic_state()
        n = state.shape[2]
        for _ in range(500):
            step(state, units, track_free_surface=False)
        column = state.rho[n // 2, n // 2, 1 : n - 1].astype(np.float64)
        measured = float(column.max() - column.min())
        expected = units.gravity_lattice * (n - 3) / CS2
        self.assertAlmostEqual(measured / expected, 1.0, delta=0.15)

    def test_spurious_currents_stay_far_below_the_mach_guard(self) -> None:
        state, units, obstacle = hydrostatic_state()
        for _ in range(2000):
            step(state, units, track_free_surface=False)
        peak = float(np.abs(state.u[:, ~obstacle]).max())
        self.assertLess(peak, 0.01 * units.u_max_lattice)

    def test_spurious_currents_do_not_grow(self) -> None:
        state, units, obstacle = hydrostatic_state()
        for _ in range(200):
            step(state, units, track_free_surface=False)
        early = float(np.abs(state.u[:, ~obstacle]).max())
        for _ in range(1800):
            step(state, units, track_free_surface=False)
        late = float(np.abs(state.u[:, ~obstacle]).max())
        self.assertLess(late, 5.0 * early + 1e-6)

    def test_gravity_direction_is_respected(self) -> None:
        obstacle = closed_box(10)
        units = lattice_units(DX_M, WCB_NU, u_max_physical_m_s=1.0)
        state = create_state(obstacle, dx_m=DX_M, dt_s=units.dt_s, initial_fluid=~obstacle)
        for _ in range(200):
            step(state, units, gravity_direction=(0.0, 0.0, -1.0), track_free_surface=False)
        interior = ~obstacle
        lateral = max(
            float(np.abs(state.u[0][interior]).max()),
            float(np.abs(state.u[1][interior]).max()),
        )
        vertical = float(np.abs(state.u[2][interior]).max())
        self.assertLess(lateral, 0.1 * vertical + 1e-12)

    def test_the_cold_start_builds_the_gradient(self) -> None:
        obstacle = closed_box(10)
        units = lattice_units(DX_M, WCB_NU, u_max_physical_m_s=1.0)
        state = create_state(obstacle, dx_m=DX_M, dt_s=units.dt_s, initial_fluid=~obstacle)
        for _ in range(400):
            step(state, units, track_free_surface=False)
        column = state.rho[5, 5, 1:-1].astype(np.float64)
        measured = float(column.max() - column.min())
        expected = units.gravity_lattice * 7 / CS2
        self.assertGreater(measured, 0.3 * expected)

class PoiseuilleTests(unittest.TestCase):

    NZ = 21
    TAU = 0.8
    FORCE = 1e-6

    STEPS = 60000 if os.environ.get("CASTSIM_FILLING_FULL_BENCH") == "1" else 12000

    @classmethod
    def setUpClass(cls) -> None:
        obstacle = np.zeros((4, 4, cls.NZ), dtype=bool)
        obstacle[:, :, 0] = obstacle[:, :, -1] = True
        base = lattice_units(DX_M, 1e-6, u_max_physical_m_s=1.0)
        units = replace(
            base,
            tau=cls.TAU,
            omega_plus=1.0 / cls.TAU,
            omega_minus=omega_minus_from_magic(1.0 / cls.TAU),
            gravity_lattice=cls.FORCE,
        )
        state = create_state(obstacle, dx_m=DX_M, dt_s=units.dt_s, initial_fluid=~obstacle)
        for _ in range(cls.STEPS):
            step(state, units, gravity_direction=(1.0, 0.0, 0.0), track_free_surface=False)
        cls.profile = state.u[0, 2, 2, :].astype(np.float64)
        cls.nu = (cls.TAU - 0.5) / 3.0
        cls.wall_low, cls.wall_high = 0.5, cls.NZ - 1.5

    def _analytic(self) -> np.ndarray:
        z = np.arange(1, self.NZ - 1, dtype=np.float64)
        half = (self.wall_high - self.wall_low) / 2.0
        centre = (self.wall_low + self.wall_high) / 2.0
        return (self.FORCE / (2.0 * self.nu)) * (half ** 2 - (z - centre) ** 2)

    def test_peak_velocity_matches_the_analytic_value(self) -> None:
        analytic = self._analytic()
        self.assertAlmostEqual(
            self.profile.max() / analytic.max(), 1.0, delta=0.02
        )

    def test_profile_matches_the_parabola(self) -> None:
        analytic = self._analytic()
        error = np.abs(self.profile[1:-1] - analytic).max() / analytic.max()
        self.assertLess(error, 0.02)

    def test_bounce_back_wall_sits_halfway_between_nodes(self) -> None:
        z = np.arange(1, self.NZ - 1, dtype=np.float64)
        roots = np.sort(np.roots(np.polyfit(z, self.profile[1:-1], 2)))
        self.assertAlmostEqual(roots[0], self.wall_low, delta=0.05)
        self.assertAlmostEqual(roots[1], self.wall_high, delta=0.05)

    def test_flow_is_unidirectional(self) -> None:
        self.assertGreater(self.profile.max(), 0.0)

    def test_no_slip_at_the_solid_nodes(self) -> None:
        self.assertEqual(self.profile[0], 0.0)
        self.assertEqual(self.profile[-1], 0.0)

class MassConservationTests(unittest.TestCase):

    def test_closed_box_conserves_mass_with_the_free_surface_on(self) -> None:
        obstacle = closed_box(10)
        units = lattice_units(DX_M, WCB_NU, u_max_physical_m_s=1.0)
        state = create_state(obstacle, dx_m=DX_M, dt_s=units.dt_s, initial_fluid=~obstacle)
        before = total_mass(state)
        for _ in range(200):
            step(state, units, track_free_surface=True)
        drift = abs(total_mass(state) - before) / before
        self.assertLess(drift, 1e-3)

    def test_partially_filled_box_conserves_mass(self) -> None:
        n = 12
        obstacle = closed_box(n)
        initial = np.zeros((n, n, n), dtype=bool)
        initial[1:-1, 1:-1, 1 : n // 2] = True
        units = lattice_units(DX_M, WCB_NU, u_max_physical_m_s=1.0)
        state = create_state(
            obstacle, dx_m=DX_M, dt_s=units.dt_s, initial_fluid=initial
        )
        before = total_mass(state)
        self.assertGreater(before, 0.0)
        for _ in range(200):
            step(state, units, track_free_surface=True)
        drift = abs(total_mass(state) - before) / before
        self.assertLess(drift, 1e-3)

    def test_fluid_fraction_stays_bounded(self) -> None:
        n = 12
        obstacle = closed_box(n)
        initial = np.zeros((n, n, n), dtype=bool)
        initial[1:-1, 1:-1, 1 : n // 2] = True
        units = lattice_units(DX_M, WCB_NU, u_max_physical_m_s=1.0)
        state = create_state(
            obstacle, dx_m=DX_M, dt_s=units.dt_s, initial_fluid=initial
        )
        for _ in range(200):
            step(state, units, track_free_surface=True)
            eps = fluid_fraction(state)[state.flags == FLAG_FLUID]
            if eps.size:
                self.assertLess(float(np.abs(eps - 1.0).max()), 0.5)

    def test_fill_times_are_recorded_for_filled_cells(self) -> None:
        n = 12
        obstacle = closed_box(n)
        initial = np.zeros((n, n, n), dtype=bool)
        initial[1:-1, 1:-1, 1 : n // 2] = True
        units = lattice_units(DX_M, WCB_NU, u_max_physical_m_s=1.0)
        state = create_state(
            obstacle, dx_m=DX_M, dt_s=units.dt_s, initial_fluid=initial
        )
        filled = state.flags == FLAG_FLUID
        self.assertTrue(np.all(state.fill_time_s[filled] >= 0.0))
        gas = state.flags != FLAG_FLUID
        self.assertTrue(np.all(state.fill_time_s[gas & ~obstacle] < 0.0))

if __name__ == "__main__":
    unittest.main()

class FreeSurfaceAdvectionTests(unittest.TestCase):

    @classmethod
    def _falling_column(cls, nz: int = 24):
        nx = ny = 5
        obstacle = np.zeros((nx, ny, nz), dtype=bool)
        obstacle[0] = obstacle[-1] = True
        obstacle[:, 0] = obstacle[:, -1] = True
        obstacle[:, :, 0] = True
        units = lattice_units(DX_M, WCB_NU, u_max_physical_m_s=2.8)
        seed = np.zeros((nx, ny, nz), dtype=bool)
        seed[1:-1, 1:-1, 15:21] = True
        state = create_state(
            obstacle, dx_m=DX_M, dt_s=units.dt_s, initial_fluid=seed
        )
        return state, units

    def test_a_suspended_column_conserves_mass(self) -> None:
        state, units = self._falling_column()
        before = total_mass(state)
        for _ in range(500):
            step(state, units, track_free_surface=True)
        self.assertLess(abs(total_mass(state) - before) / before, 1e-5)

    def test_gas_cells_never_hold_populations(self) -> None:
        state, units = self._falling_column()
        for _ in range(50):
            step(state, units, track_free_surface=True)
            gas = state.flags == FLAG_GAS
            if gas.any():
                self.assertEqual(float(np.abs(state.f[:, gas]).max()), 0.0)

    def test_a_suspended_column_falls(self) -> None:
        state, units = self._falling_column()
        wet = (state.flags == FLAG_FLUID) | (state.flags == FLAG_INTERFACE)
        start_front = int(np.argwhere(wet)[:, 2].min())
        for _ in range(500):
            step(state, units, track_free_surface=True)
        wet = (state.flags == FLAG_FLUID) | (state.flags == FLAG_INTERFACE)
        end_front = int(np.argwhere(wet)[:, 2].min())
        self.assertLess(end_front, start_front)

class DrivenFillTests(unittest.TestCase):

    @staticmethod
    def _open_channel(nz: int = 20):
        nx = ny = 6
        obstacle = np.zeros((nx, ny, nz), dtype=bool)
        obstacle[0] = obstacle[-1] = True
        obstacle[:, 0] = obstacle[:, -1] = True
        obstacle[:, :, 0] = True
        return obstacle

    def test_an_inlet_fills_a_channel(self) -> None:
        from physics.fluid_flow import equilibrium as _eq

        obstacle = self._open_channel()
        units = lattice_units(DX_M, WCB_NU, u_max_physical_m_s=2.8)
        seed = np.zeros(obstacle.shape, dtype=bool)
        seed[1:-1, 1:-1, -2:-1] = True
        state = create_state(
            obstacle, dx_m=DX_M, dt_s=units.dt_s, initial_fluid=seed
        )
        inlet = np.argwhere(seed)
        ix, iy, iz = inlet[:, 0], inlet[:, 1], inlet[:, 2]
        target = int(np.count_nonzero(~obstacle))

        for _ in range(4000):
            state.flags[ix, iy, iz] = FLAG_FLUID
            state.mass[ix, iy, iz] = 1.0
            state.f[:, ix, iy, iz] = _eq(
                np.full(len(inlet), 1.0), state.u[:, ix, iy, iz].astype(np.float64)
            ).astype(np.float32)
            step(state, units, track_free_surface=True)

        reached = int(np.count_nonzero(state.flags == FLAG_FLUID))
        self.assertGreater(reached, target // 2)
