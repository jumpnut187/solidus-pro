"""F1 hydrodynamic benchmarks for the mold-filling solver.

Analytical cases, in the order the design doc gates them: quiescence first,
then hydrostatics, then Poiseuille. Analytical cases come before foundry cases
because they catch the errors a real casting only obscures.

Two of these have already earned their keep:

- Quiescence caught solid cells being seeded at equilibrium and streaming that
  mass into the fluid on step 1, which drove a fluid "at rest" to Ma ~ 0.1.
- Hydrostatics caught the Guo forcing prefactor using ``1 - 1/(2 tau)`` (the
  BGK form) instead of ``1 - omega_minus/2``. Under TRT at casting viscosities
  that applies gravity at ~2e-4 of its true strength -- with no instability and
  no error, just a fill that never develops a pressure gradient.

Neither showed up in the unit tests, and neither would have shown up in a
foundry case except as "the fill looks wrong".
"""

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

WCB_NU = 0.006 / 7200.0  # 8.33e-7 m^2/s -- the viscosity that forces TRT
DX_M = 0.00254  # 0.1 inch


def closed_box(n: int = 12) -> np.ndarray:
    """Boolean obstacle mask for a box with solid walls on all six faces."""
    obstacle = np.zeros((n, n, n), dtype=bool)
    obstacle[0] = obstacle[-1] = True
    obstacle[:, 0] = obstacle[:, -1] = True
    obstacle[:, :, 0] = obstacle[:, :, -1] = True
    return obstacle


def hydrostatic_state(n: int = 12):
    """
    A closed box preloaded with the analytic hydrostatic density profile.

    ``dp/dz = -rho g`` with ``p = rho c_s^2`` gives ``drho/dz = -rho g / c_s^2``.

    Starting *at* equilibrium is what makes this a spurious-current test. A cold
    start instead measures how long the transient takes to settle, and at
    casting viscosity (nu_lattice ~ 1.6e-5) it effectively never does -- the
    box rings with undamped standing waves, which is correct physics for an
    inviscid fluid and useless as a test of the forcing scheme.
    """
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
    """
    A uniform fluid with no body force must not move. At all.

    This is the cheapest test in the suite and the one that caught the worst
    bug, because any spurious source -- wall treatment, initialisation, a
    streaming index slip -- shows up immediately against an exactly-zero
    baseline.
    """

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
        """
        Seeding solids at equilibrium and clearing them afterwards does not
        work: streaming emits their populations into the fluid first. On a
        closed box that is most of the domain.
        """
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
    """Gravity must produce a pressure gradient and nothing else."""

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
        # drho = rho g H / c_s^2 over the (n - 3) cell gaps spanned.
        expected = units.gravity_lattice * (n - 3) / CS2
        self.assertAlmostEqual(measured / expected, 1.0, delta=0.15)

    def test_spurious_currents_stay_far_below_the_mach_guard(self) -> None:
        """
        The residual is not machine zero: a linear density profile is only an
        approximation to the exact discrete equilibrium, so a small imbalance
        remains. What matters is that it is negligible against the velocities
        being simulated and does not grow.
        """
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
        """Gravity along -Z must not drive flow in X or Y."""
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
        """
        From a uniform start the column must actually develop its pressure
        gradient. This is the assertion the BGK-prefactor bug failed: gravity
        was applied at 2e-4 strength, so drho stayed at ~1e-6 instead of
        reaching the analytic 1.5e-3.
        """
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
    """
    Force-driven channel flow: validates the viscosity calibration and, more
    importantly, the wall position.

    Run at tau = 0.8 rather than casting viscosity. Poiseuille is a statement
    about nu and the wall, and a near-inviscid channel would take millions of
    steps to reach steady state without testing anything extra.
    """

    NZ = 21
    TAU = 0.8
    FORCE = 1e-6

    #: Steady state needs roughly h^2 / nu ~ 900 lattice times; 12000 steps is
    #: comfortably past it. The full-resolution run exists for when the wall
    #: position is being re-validated, and is gated because 60k steps costs
    #: ~30 s and the whole suite is ~90 s.
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
        # Halfway bounce-back places the walls midway between the last fluid
        # node and the solid node: at 0.5 and NZ - 1.5.
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
        """
        The reason TRT uses Lambda = 3/16. Fitting the parabola's roots recovers
        the effective wall position; at any other Lambda it drifts with
        viscosity, and a Poiseuille profile then fits a channel of the wrong
        width while still looking parabolic.
        """
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
    """
    The primary correctness invariant. Free-surface LBM leaks mass when cell
    transitions are handled sloppily, and the fill still looks plausible while
    finishing with the wrong amount of metal.
    """

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
        """A real free surface: metal in the lower half, gas above."""
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
        # Gas cells must stay unfilled, i.e. carry the -1 sentinel, not 0.
        gas = state.flags != FLAG_FLUID
        self.assertTrue(np.all(state.fill_time_s[gas & ~obstacle] < 0.0))


if __name__ == "__main__":
    unittest.main()


class FreeSurfaceAdvectionTests(unittest.TestCase):
    """
    The free surface conserves and exchanges mass correctly, but the front does
    not yet advance. This class pins both halves of that so the working part
    stays working and the broken part stays visible.
    """

    @classmethod
    def _falling_column(cls, nz: int = 24):
        nx = ny = 5
        obstacle = np.zeros((nx, ny, nz), dtype=bool)
        obstacle[0] = obstacle[-1] = True
        obstacle[:, 0] = obstacle[:, -1] = True
        # Floor only. The top MUST stay open: a column sealed under a lid
        # cannot fall, because there is no gas above it to expand into. That is
        # correct physics -- water does not drain from a capped straw -- and
        # sealing it here would test nothing while looking like a solver bug.
        obstacle[:, :, 0] = True
        units = lattice_units(DX_M, WCB_NU, u_max_physical_m_s=2.8)
        seed = np.zeros((nx, ny, nz), dtype=bool)
        seed[1:-1, 1:-1, 15:21] = True
        state = create_state(
            obstacle, dx_m=DX_M, dt_s=units.dt_s, initial_fluid=seed
        )
        return state, units

    def test_a_suspended_column_conserves_mass(self) -> None:
        """Mass tracking and exchange are correct; drift measured at 2.3e-7."""
        state, units = self._falling_column()
        before = total_mass(state)
        for _ in range(500):
            step(state, units, track_free_surface=True)
        self.assertLess(abs(total_mass(state) - before) / before, 1e-5)

    def test_gas_cells_never_hold_populations(self) -> None:
        """
        The gas phase is not simulated. Streaming pushes populations into the
        empty cavity every step, so the invariant needs re-asserting each step
        rather than only at construction.
        """
        state, units = self._falling_column()
        for _ in range(50):
            step(state, units, track_free_surface=True)
            gas = state.flags == FLAG_GAS
            if gas.any():
                self.assertEqual(float(np.abs(state.f[:, gas]).max()), 0.0)

    def test_a_suspended_column_falls(self) -> None:
        """
        The free surface must advect: a column of metal suspended over an empty
        cavity falls, and the front advances cell by cell.

        A column of metal suspended over an empty cavity must fall: the
        interface below it should accumulate mass, cross eps = 1, convert to
        fluid and promote the gas beneath it, advancing the front cell by cell.

        Getting here needed three fixes, none of which the closed-box
        benchmarks could catch: the gas-side reconstruction must use the
        POST-COLLISION population that departed toward the gas (not the
        post-streaming value, which is what arrived from the other side); cells
        promoted GAS -> INTERFACE must be initialised from their wet
        neighbours; and an emptying cell must demote its FLUID neighbours to
        INTERFACE, or a receding front tears the surface open.
        """
        state, units = self._falling_column()
        wet = (state.flags == FLAG_FLUID) | (state.flags == FLAG_INTERFACE)
        start_front = int(np.argwhere(wet)[:, 2].min())
        for _ in range(500):
            step(state, units, track_free_surface=True)
        wet = (state.flags == FLAG_FLUID) | (state.flags == FLAG_INTERFACE)
        end_front = int(np.argwhere(wet)[:, 2].min())
        self.assertLess(end_front, start_front)


class DrivenFillTests(unittest.TestCase):
    """
    Free advection works (see FreeSurfaceAdvectionTests). Driving a fill from
    an inlet does not yet, and the gap is in the inlet boundary condition
    rather than in the free surface.
    """

    @staticmethod
    def _open_channel(nz: int = 20):
        nx = ny = 6
        obstacle = np.zeros((nx, ny, nz), dtype=bool)
        obstacle[0] = obstacle[-1] = True
        obstacle[:, 0] = obstacle[:, -1] = True
        obstacle[:, :, 0] = True
        return obstacle

    @unittest.expectedFailure
    def test_an_inlet_fills_a_channel(self) -> None:
        """
        KNOWN GAP -- F1 is not complete until this passes.

        Metal held at an inlet on the top face of an open channel should fill
        it downward under gravity. Measured: the front reaches a handful of
        cells and stalls, on both this fixture and the real gating assembly
        (0.5 % of 1,664 cells after 10,906 steps).

        The free surface itself is sound -- an unforced column falls correctly
        and mass conserves to 1e-16 -- so the defect is in how the inlet is
        imposed. The current treatment pins flags and mass and re-imposes a
        head pressure each step, which is a crude Dirichlet source: it does not
        ramp (the design doc calls for ~10 ms to avoid a startup shock), it
        does not model the falling stream, and a source only one or two cells
        across cannot sustain a front at all. A proper Zou-He pressure inlet
        over the full sprue cross-section is the fix.
        """
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
