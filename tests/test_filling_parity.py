"""The Numba filling kernels must agree with the NumPy reference.

Mirrors tests/test_parallel_kernel_parity.py, which guards the solidification
kernels the same way. The NumPy path is the readable implementation the
analytical benchmarks were validated against; the kernels are the fast one.
Only a parity test keeps the second honest, because a fused kernel can be
subtly wrong in ways that still look like a plausible fill.

Two agreement levels are asserted deliberately:

- One step from an identical state must agree to float64 round-off. Any real
  divergence in the algebra shows up immediately here.
- Many steps are allowed to drift, because the two paths sum in different
  orders and the scheme runs within 1e-5 of its stability limit. What must NOT
  drift is anything conserved or discrete: total mass, and the cell flags.
"""

from __future__ import annotations

import unittest

import numpy as np

from physics.fluid_flow import NUMBA_AVAILABLE, LATTICE_VELOCITIES, Q, lattice_units
from physics.filling import (
    FLAG_FLUID,
    FLAG_GAS,
    FLAG_INTERFACE,
    FLAG_OBSTACLE,
    NO_TEMPERATURE,
    PorousZone,
    advect_temperature,
    apply_wall_heat_loss,
    create_state,
    heal_submerged_interface,
    reset_detached_interface_cells,
    step,
    total_mass,
)
from physics.filling import _dilate, _interface_neighbourhood

WCB_NU = 0.006 / 7200.0
DX_M = 0.0152


def _fixture(
    n: int = 14, *, with_filter: bool = False, submerge: bool = False, seed: int = 7
):
    """
    A partly filled box, optionally with a filter slab across the middle.

    ``submerge`` fills *over* that slab and marks it primed, so the filter
    cells are wet. The default fixture leaves them dry, which means it never
    reaches the porosity branch of the mass exchange -- the branch that keeps
    the eps average for pairs touching a filter, because a full porous cell
    holds ``rho * porosity`` rather than ``rho``. See
    :func:`physics.filling.advance_mass`.
    """
    obstacle = np.zeros((n, n, n), dtype=bool)
    obstacle[0] = obstacle[-1] = True
    obstacle[:, 0] = obstacle[:, -1] = True
    obstacle[:, :, 0] = True

    units = lattice_units(DX_M, WCB_NU, u_max_physical_m_s=2.8)
    fluid = np.zeros((n, n, n), dtype=bool)
    if submerge:
        fluid[1:-1, 1:-1, 1 : n - 3] = True
    else:
        fluid[1:-1, 1:-1, n // 2 :] = True

    porous = None
    if with_filter:
        porous = PorousZone.inert((n, n, n))
        band = slice(n // 3, n // 3 + 2)
        porous.porosity[1:-1, 1:-1, band] = 0.85
        porous.drag_c0[1:-1, 1:-1, band] = 0.62
        porous.drag_c1[1:-1, 1:-1, band] = 1.9
        porous.entry_offset[1:-1, 1:-1, band] = 3.6e-4

    state = create_state(
        obstacle, dx_m=DX_M, dt_s=units.dt_s, initial_fluid=fluid, porous=porous
    )
    if submerge:
        state.porous.primed[...] = True
    # Perturb so the comparison is not made on a trivially symmetric field.
    rng = np.random.default_rng(seed)
    state.f = (state.f * (1.0 + 0.01 * rng.standard_normal(state.f.shape))).astype(
        np.float32
    )
    state.f[:, obstacle] = 0.0
    state.f[:, state.flags == FLAG_GAS] = 0.0
    return state, units


def _twin(**kwargs):
    """Two independent states seeded identically."""
    a, units = _fixture(**kwargs)
    b, _ = _fixture(**kwargs)
    np.testing.assert_array_equal(a.f, b.f)
    return a, b, units


@unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
class SingleStepParityTests(unittest.TestCase):
    """One step must agree to round-off. Anything worse is an algebra bug."""

    def test_populations_agree(self) -> None:
        a, b, units = _twin()
        step(a, units, use_numba=True)
        step(b, units, use_numba=False)
        np.testing.assert_allclose(a.f, b.f, rtol=1e-5, atol=1e-7)

    def test_density_agrees(self) -> None:
        a, b, units = _twin()
        step(a, units, use_numba=True)
        step(b, units, use_numba=False)
        np.testing.assert_allclose(a.rho, b.rho, rtol=1e-5, atol=1e-7)

    def test_velocity_agrees(self) -> None:
        a, b, units = _twin()
        step(a, units, use_numba=True)
        step(b, units, use_numba=False)
        np.testing.assert_allclose(a.u, b.u, rtol=1e-4, atol=1e-9)

    def test_mass_agrees(self) -> None:
        a, b, units = _twin()
        step(a, units, use_numba=True)
        step(b, units, use_numba=False)
        np.testing.assert_allclose(a.mass, b.mass, rtol=1e-5, atol=1e-9)

    def test_flags_agree_exactly(self) -> None:
        """Cell state is discrete; it has no tolerance to spend."""
        a, b, units = _twin()
        step(a, units, use_numba=True)
        step(b, units, use_numba=False)
        np.testing.assert_array_equal(a.flags, b.flags)

    def test_single_phase_agrees(self) -> None:
        a, b, units = _twin()
        step(a, units, use_numba=True, track_free_surface=False)
        step(b, units, use_numba=False, track_free_surface=False)
        np.testing.assert_allclose(a.f, b.f, rtol=1e-5, atol=1e-7)

    def test_smagorinsky_off_agrees(self) -> None:
        """The eddy-viscosity branch must not be the only path that matches."""
        a, b, units = _twin()
        step(a, units, use_numba=True, smagorinsky=False)
        step(b, units, use_numba=False, smagorinsky=False)
        np.testing.assert_allclose(a.f, b.f, rtol=1e-5, atol=1e-7)


@unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
class PorousParityTests(unittest.TestCase):
    """The filter path has its own branches in the kernel; cover it separately."""

    def test_populations_agree_with_a_filter(self) -> None:
        a, b, units = _twin(with_filter=True)
        step(a, units, use_numba=True)
        step(b, units, use_numba=False)
        np.testing.assert_allclose(a.f, b.f, rtol=1e-5, atol=1e-7)

    def test_velocity_agrees_with_a_filter(self) -> None:
        a, b, units = _twin(with_filter=True)
        step(a, units, use_numba=True)
        step(b, units, use_numba=False)
        np.testing.assert_allclose(a.u, b.u, rtol=1e-4, atol=1e-9)

    def test_drag_actually_engaged(self) -> None:
        """Guard the fixture: a parity test over an inert filter proves nothing."""
        state, _ = _fixture(with_filter=True)
        self.assertFalse(state.porous.is_inert)
        self.assertGreater(float(state.porous.drag_c1.max()), 0.0)

    def test_the_filter_is_actually_wet_when_submerged(self) -> None:
        """
        Guard the other half: the mass exchange's porosity branch is only
        reached by a pair where one cell has ``porosity < 1`` *and* holds
        metal. The default fixture starts the metal above the slab, so it
        never gets there.
        """
        dry, _ = _fixture(with_filter=True)
        wet, _ = _fixture(with_filter=True, submerge=True)
        porous = np.asarray(dry.porous.porosity) < 1.0
        self.assertGreater(int(np.count_nonzero(porous)), 0)

        def _wet_porous(state):
            holds_metal = (state.flags == FLAG_FLUID) | (
                state.flags == FLAG_INTERFACE
            )
            return int(np.count_nonzero(porous & holds_metal))

        self.assertEqual(_wet_porous(dry), 0)
        self.assertGreater(_wet_porous(wet), 0)

    def test_populations_agree_through_a_wet_filter(self) -> None:
        a, b, units = _twin(with_filter=True, submerge=True)
        step(a, units, use_numba=True)
        step(b, units, use_numba=False)
        np.testing.assert_allclose(a.f, b.f, rtol=1e-5, atol=1e-7)

    def test_mass_agrees_through_a_wet_filter(self) -> None:
        """
        The pair weighting inside a filter, in both paths, held to round-off.

        This is the only test that reaches the porosity branch, and it is a
        branch the two implementations have to agree on exactly or the fast
        path is quietly solving a different problem inside every filter.
        """
        a, b, units = _twin(with_filter=True, submerge=True)
        for _ in range(30):
            step(a, units, use_numba=True)
            step(b, units, use_numba=False)
        np.testing.assert_allclose(a.mass, b.mass, rtol=1e-9, atol=1e-12)
        np.testing.assert_array_equal(a.flags, b.flags)
        self.assertAlmostEqual(total_mass(a), total_mass(b), places=9)


@unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
class MultiStepParityTests(unittest.TestCase):
    """
    Over many steps the two paths sum in different orders, and the scheme runs
    within 1e-5 of its stability limit, so bitwise agreement is not a
    reasonable thing to demand. Conserved and discrete quantities still are.
    """

    STEPS = 60

    def test_total_mass_tracks(self) -> None:
        a, b, units = _twin()
        start = total_mass(a)
        for _ in range(self.STEPS):
            step(a, units, use_numba=True)
            step(b, units, use_numba=False)
        self.assertAlmostEqual(total_mass(a) / start, total_mass(b) / start, places=6)

    def test_both_paths_conserve_mass(self) -> None:
        a, b, units = _twin()
        start = total_mass(a)
        for _ in range(self.STEPS):
            step(a, units, use_numba=True)
            step(b, units, use_numba=False)
        for label, state in (("numba", a), ("numpy", b)):
            with self.subTest(path=label):
                self.assertLess(abs(total_mass(state) - start) / start, 1e-5)

    def test_fill_fronts_stay_together(self) -> None:
        a, b, units = _twin()
        for _ in range(self.STEPS):
            step(a, units, use_numba=True)
            step(b, units, use_numba=False)
        filled_a = int(np.count_nonzero(a.flags == FLAG_FLUID))
        filled_b = int(np.count_nonzero(b.flags == FLAG_FLUID))
        self.assertLessEqual(abs(filled_a - filled_b), 2)

    def test_neither_path_diverges(self) -> None:
        a, b, units = _twin()
        for _ in range(self.STEPS):
            step(a, units, use_numba=True)
            step(b, units, use_numba=False)
        for label, state in (("numba", a), ("numpy", b)):
            with self.subTest(path=label):
                self.assertTrue(bool(np.all(np.isfinite(state.f))))
                self.assertTrue(bool(np.all(np.isfinite(state.rho))))


@unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
class KernelInvariantTests(unittest.TestCase):
    def test_kernel_flag_constants_match_the_solver(self) -> None:
        """
        physics/fluid_flow.py duplicates the cell-flag values so the kernels
        stay free of physics.filling. Duplicated constants drift; this is what
        stops them.
        """
        from physics import fluid_flow
        from physics import filling

        self.assertEqual(fluid_flow._FLAG_OBSTACLE, int(filling.FLAG_OBSTACLE))
        self.assertEqual(fluid_flow._FLAG_GAS, int(filling.FLAG_GAS))
        self.assertEqual(fluid_flow._FLAG_INTERFACE, int(filling.FLAG_INTERFACE))
        self.assertEqual(fluid_flow._FLAG_FLUID, int(filling.FLAG_FLUID))

    def test_obstacle_cells_hold_nothing_on_the_numba_path(self) -> None:
        state, units = _fixture()
        obstacle = state.flags == 0
        for _ in range(5):
            step(state, units, use_numba=True)
            self.assertEqual(float(np.abs(state.f[:, obstacle]).max()), 0.0)

    def test_numpy_fallback_is_reachable(self) -> None:
        """The reference path must keep working even where numba is present."""
        state, units = _fixture()
        step(state, units, use_numba=False)
        self.assertTrue(bool(np.all(np.isfinite(state.f))))


class SmagorinskyConstantTests(unittest.TestCase):
    """
    ``smagorinsky_cs`` reached both collision paths, or it reached neither.

    It used to reach neither: the dialog offered it and the project file stored
    it while the kernels ran at the hardcoded module constant. Parity alone
    could never have caught that -- the two paths agreed precisely *because*
    both were ignoring the value -- so the liveness check below comes first,
    and the cross-path check is then repeated at a Cs that is not the default.
    """

    #: Both inside the range the dialog offers, and far enough apart that the
    #: eddy viscosity (which goes as Cs^2) cannot land on the same tau_eff.
    CS_LOW = 0.05
    CS_HIGH = 0.40

    def _stepped(self, cs: float, *, use_numba: bool) -> np.ndarray:
        state, units = _fixture()
        step(state, units, use_numba=use_numba, smagorinsky_cs=cs)
        return state.f.astype(np.float64)

    def test_the_constant_changes_the_numpy_result(self) -> None:
        """The knob is live. Without this, everything else here is vacuous."""
        spread = float(
            np.max(
                np.abs(
                    self._stepped(self.CS_HIGH, use_numba=False)
                    - self._stepped(self.CS_LOW, use_numba=False)
                )
            )
        )
        self.assertGreater(
            spread,
            1e-6,
            "changing smagorinsky_cs left the populations untouched -- the "
            "knob is being ignored again",
        )

    @unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
    def test_the_constant_changes_the_numba_result(self) -> None:
        spread = float(
            np.max(
                np.abs(
                    self._stepped(self.CS_HIGH, use_numba=True)
                    - self._stepped(self.CS_LOW, use_numba=True)
                )
            )
        )
        self.assertGreater(spread, 1e-6)

    @unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
    def test_paths_agree_at_a_non_default_constant(self) -> None:
        """A kernel that quietly kept the constant would show up right here."""
        np.testing.assert_allclose(
            self._stepped(self.CS_HIGH, use_numba=True),
            self._stepped(self.CS_HIGH, use_numba=False),
            rtol=1e-5,
            atol=1e-7,
        )

    def test_zero_degrades_to_plain_trt(self) -> None:
        """
        Cs = 0 must be a usable setting, not a division by zero or a silent
        fallback to the default: nu_t = Cs^2 |S| vanishes, tau_eff collapses to
        the molecular tau, and the collision is exactly the one taken with the
        subgrid model switched off. That equivalence is what lets the stage
        accept 0 from settings.
        """
        zero, units = _fixture()
        step(zero, units, use_numba=False, smagorinsky_cs=0.0)
        disabled, units = _fixture()
        step(disabled, units, use_numba=False, smagorinsky=False)
        np.testing.assert_allclose(zero.f, disabled.f, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# The thermal and free-surface-scan kernels
# ---------------------------------------------------------------------------
#
# advect_temperature, apply_wall_heat_loss, reset_detached_interface_cells and
# heal_submerged_interface each gained a Numba kernel and each kept its NumPy
# body as the fallback. Unlike the collision kernels these are *supposed* to be
# bit-identical: same stencil, same accumulation order, float64 arithmetic on
# both sides, one float32 rounding at the end. So they are asserted exactly.
#
# Exactness is not pedantry here. Converting an ``np.roll`` to index arithmetic
# flips the sign of the shift -- roll(shift=-1) brings index+1 down to index --
# and getting that backwards reverses the upwind direction while leaving the
# symmetric Laplacian looking perfectly fine. A tolerance wide enough to
# absorb round-off is also wide enough to absorb a plausible-looking wrong
# answer on a smooth field.


TEMPERATURE_MIX_SEED = 4103


def _mixed_state(n: int = 14, *, steps: int = 8, seed: int = TEMPERATURE_MIX_SEED):
    """
    A state carrying all four cell flags, a developed velocity field, and
    temperatures that are NaN exactly where there is no metal.

    Built by actually running the solver rather than by assembling flags by
    hand, so the flag topology is one the scheme can really produce: a curved
    front with interface cells of every neighbour count, obstacle walls, and a
    gas cavity above. The temperatures are then stamped on with noise, because
    a smooth field cannot distinguish an upwind stencil from a downwind one.
    """
    state, units = _fixture(n=n, seed=seed)
    for _ in range(steps):
        step(state, units, use_numba=True)

    rng = np.random.default_rng(seed)
    wet = (state.flags == FLAG_FLUID) | (state.flags == FLAG_INTERFACE)
    temperature = np.full(state.shape, NO_TEMPERATURE, dtype=np.float32)
    temperature[wet] = (
        1400.0 + 80.0 * rng.standard_normal(int(np.count_nonzero(wet)))
    ).astype(np.float32)
    state.temperature = temperature
    return state, units


def _plant_pathological_cells(state):
    """
    Add the two cells :func:`heal_submerged_interface` and
    :func:`reset_detached_interface_cells` exist for.

    A short run of a healthy fill produces neither: a submerged interface cell
    takes a closing front, and an airborne droplet takes a fast one. Without
    planting them the end-to-end parity tests below would compare two paths
    that both correctly did nothing, which is agreement about nothing.

    Deterministic -- first cell in index order -- so twin states plant the
    same ones.

    Returns:
        ``(buried, droplet)`` index tuples.
    """
    gas = state.flags == FLAG_GAS
    wet = (state.flags == FLAG_FLUID) | (state.flags == FLAG_INTERFACE)

    # A FLUID cell with no gas anywhere in its 18-neighbourhood. Demoting it to
    # INTERFACE cannot create a gas neighbour, so it is gasless by construction
    # -- and under-full, so healing it has real mass bookkeeping to do.
    buried_mask = (state.flags == FLAG_FLUID) & ~_dilate(gas)
    buried = tuple(int(v) for v in np.argwhere(buried_mask)[0])
    state.flags[buried] = FLAG_INTERFACE
    state.mass[buried] = 0.42

    # A GAS cell with no wet neighbour at all. Promoting it to INTERFACE leaves
    # it with zero wet support: the sub-grid droplet.
    droplet_mask = gas & ~_dilate(wet)
    droplet = tuple(int(v) for v in np.argwhere(droplet_mask)[0])
    state.flags[droplet] = FLAG_INTERFACE
    state.mass[droplet] = 0.31
    state.rho[droplet] = np.float32(1.0)
    state.temperature[droplet] = np.float32(1380.0)
    return buried, droplet


def _mixed_twin(*, plant: bool = False, **kwargs):
    """Two independent mixed states seeded identically."""
    a, units = _mixed_state(**kwargs)
    b, _ = _mixed_state(**kwargs)
    if plant:
        _plant_pathological_cells(a)
        _plant_pathological_cells(b)
    np.testing.assert_array_equal(a.flags, b.flags)
    np.testing.assert_array_equal(a.u, b.u)
    np.testing.assert_array_equal(a.mass, b.mass)
    np.testing.assert_array_equal(a.temperature, b.temperature, strict=True)
    return a, b, units


class MixedStateFixtureTests(unittest.TestCase):
    """
    Guard the fixture. A parity test over a state that never reaches a branch
    proves the two paths agree about nothing in particular.
    """

    def test_all_four_flags_are_present(self) -> None:
        state, _ = _mixed_state()
        for flag in (FLAG_OBSTACLE, FLAG_GAS, FLAG_INTERFACE, FLAG_FLUID):
            with self.subTest(flag=int(flag)):
                self.assertGreater(int(np.count_nonzero(state.flags == flag)), 0)

    def test_nan_sits_exactly_where_there_is_no_metal(self) -> None:
        state, _ = _mixed_state()
        wet = (state.flags == FLAG_FLUID) | (state.flags == FLAG_INTERFACE)
        np.testing.assert_array_equal(np.isnan(state.temperature), ~wet)

    def test_the_velocity_field_has_both_signs_on_every_axis(self) -> None:
        """Otherwise only one arm of the upwind branch is ever taken."""
        state, _ = _mixed_state()
        for axis in range(3):
            with self.subTest(axis=axis):
                component = state.u[axis]
                self.assertGreater(float(component.max()), 0.0)
                self.assertLess(float(component.min()), 0.0)

    def test_some_cell_actually_touches_the_mold(self) -> None:
        state, _ = _mixed_state()
        obstacle = state.flags == FLAG_OBSTACLE
        exposed = np.zeros(state.shape, dtype=np.int32)
        for axis in range(3):
            for shift in (-1, 1):
                exposed += np.roll(obstacle, shift=shift, axis=axis)
        wet = (state.flags == FLAG_FLUID) | (state.flags == FLAG_INTERFACE)
        self.assertGreater(int(np.count_nonzero(wet & (exposed > 0))), 0)


@unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
class ThermalKernelParityTests(unittest.TestCase):
    """
    ``advect_temperature`` and ``apply_wall_heat_loss``, kernel against the
    NumPy reference, on the same state.
    """

    ALPHA = 0.037
    MOLD_C = 25.0
    RHO = 7200.0
    CP = 750.0

    def test_advection_agrees_exactly(self) -> None:
        a, b, _ = _mixed_twin()
        fast = advect_temperature(a, self.ALPHA, use_numba=True)
        slow = advect_temperature(b, self.ALPHA, use_numba=False)
        np.testing.assert_array_equal(fast, slow)

    def test_advection_leaves_the_sentinel_alone_on_both_paths(self) -> None:
        a, b, _ = _mixed_twin()
        dry = ~((a.flags == FLAG_FLUID) | (a.flags == FLAG_INTERFACE))
        advect_temperature(a, self.ALPHA, use_numba=True)
        advect_temperature(b, self.ALPHA, use_numba=False)
        self.assertTrue(bool(np.all(np.isnan(a.temperature[dry]))))
        self.assertTrue(bool(np.all(np.isnan(b.temperature[dry]))))
        self.assertTrue(bool(np.all(np.isfinite(a.temperature[~dry]))))

    def test_advection_agrees_over_many_steps(self) -> None:
        """
        One step cannot tell a stencil that is wrong only near the front from
        one that is right. Twenty can: the two fields have to stay identical
        while the surface moves under them.
        """
        a, b, units = _mixed_twin()
        for _ in range(20):
            step(a, units, use_numba=True)
            step(b, units, use_numba=True)
            advect_temperature(a, self.ALPHA, use_numba=True)
            advect_temperature(b, self.ALPHA, use_numba=False)
        np.testing.assert_array_equal(a.temperature, b.temperature)

    def _wall_loss(self, state, htc, *, use_numba):
        return apply_wall_heat_loss(
            state,
            htc_w_m2k=htc,
            mold_temperature_c=self.MOLD_C,
            density_kg_m3=self.RHO,
            specific_heat_j_kgk=self.CP,
            use_numba=use_numba,
        )

    def test_wall_loss_agrees_exactly_for_a_scalar_htc(self) -> None:
        a, b, _ = _mixed_twin()
        fast = self._wall_loss(a, 1500.0, use_numba=True)
        slow = self._wall_loss(b, 1500.0, use_numba=False)
        np.testing.assert_array_equal(fast, slow)

    def test_wall_loss_agrees_exactly_for_a_per_cell_htc(self) -> None:
        """
        The gating and casting surfaces carry different HTCs, so the field form
        is the one the real stage takes -- and it is the form the kernel has to
        fake, because Numba cannot branch on an argument's type.
        """
        a, b, _ = _mixed_twin()
        rng = np.random.default_rng(TEMPERATURE_MIX_SEED + 1)
        htc = 500.0 + 2000.0 * rng.random(a.shape)
        fast = self._wall_loss(a, htc, use_numba=True)
        slow = self._wall_loss(b, htc, use_numba=False)
        np.testing.assert_array_equal(fast, slow)

    def test_wall_loss_actually_changes_something(self) -> None:
        """Two paths that both do nothing would agree perfectly."""
        a, _, _ = _mixed_twin()
        before = a.temperature.copy()
        self._wall_loss(a, 1500.0, use_numba=True)
        changed = np.isfinite(before) & (a.temperature != before)
        self.assertGreater(int(np.count_nonzero(changed)), 0)

    def test_the_cached_face_count_matches_a_fresh_one(self) -> None:
        """
        The NumPy path caches the exposed-face count because the obstacle set
        is static. If anything ever does convert a cell to or from
        FLAG_OBSTACLE mid-run, this is the assertion that should have caught it.
        """
        from physics.filling import _wall_exposed_faces

        state, units = _mixed_state()
        cached = _wall_exposed_faces(state).copy()
        for _ in range(10):
            step(state, units, use_numba=True)
        obstacle = state.flags == FLAG_OBSTACLE
        fresh = np.zeros(state.shape, dtype=np.float64)
        for axis in range(3):
            for shift in (-1, 1):
                fresh += np.roll(obstacle, shift=shift, axis=axis)
        np.testing.assert_array_equal(_wall_exposed_faces(state), fresh)
        np.testing.assert_array_equal(cached, fresh)


@unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
class InterfaceNeighbourhoodParityTests(unittest.TestCase):
    """
    The 18-direction scan kernel against the ``np.roll`` expressions it
    replaces. Boolean masks, so exact equality is the only acceptable result.
    """

    @staticmethod
    def _reference_masks(state):
        """
        The two expressions as ``heal_submerged_interface`` and
        ``reset_detached_interface_cells`` spell them in their NumPy fallbacks.
        Deliberately written out here rather than imported: this is the
        specification the kernel is being held to, and a shared helper could be
        edited to match a broken kernel.
        """
        interface = state.flags == FLAG_INTERFACE
        gasless = interface & ~_dilate(state.flags == FLAG_GAS)

        wet = interface | (state.flags == FLAG_FLUID)
        support = np.zeros(state.shape, dtype=np.int16)
        for i in range(1, Q):
            c = LATTICE_VELOCITIES[i]
            support += np.roll(
                wet, shift=(int(c[0]), int(c[1]), int(c[2])), axis=(0, 1, 2)
            ).astype(np.int16)
        detached = interface & (support == 0)
        return gasless, detached

    def test_masks_agree_on_a_developed_front(self) -> None:
        state, _ = _mixed_state()
        gasless, detached = _interface_neighbourhood(state)
        ref_gasless, ref_detached = self._reference_masks(state)
        np.testing.assert_array_equal(gasless, ref_gasless)
        np.testing.assert_array_equal(detached, ref_detached)

    #: Two flag mixtures, because the two masks need opposite ones. A gas-heavy
    #: lattice strands interface cells with no wet support (``detached``); a
    #: wet-heavy one buries them with no gas neighbour (``gasless``). A single
    #: balanced mixture produces neither: at 18 independent neighbours, "all of
    #: them are the minority phase" is a 1-in-10^8 event.
    RANDOM_MIXTURES = (
        ("gas heavy", (0.10, 0.85, 0.04, 0.01), "detached"),
        ("wet heavy", (0.10, 0.02, 0.50, 0.38), "gasless"),
    )

    def test_masks_agree_on_randomised_flags(self) -> None:
        """
        A physical front carries only a handful of gasless or detached cells.
        Random flags carry thousands, and reach neighbour counts a real fill
        would take a long run to produce.
        """
        shape = (13, 11, 12)
        for label, probabilities, expected in self.RANDOM_MIXTURES:
            with self.subTest(mixture=label):
                rng = np.random.default_rng(19)
                state = create_state(
                    np.zeros(shape, dtype=bool), dx_m=DX_M, dt_s=1e-4
                )
                state.flags = rng.choice(
                    np.array(
                        [FLAG_OBSTACLE, FLAG_GAS, FLAG_INTERFACE, FLAG_FLUID],
                        dtype=np.uint8,
                    ),
                    size=shape,
                    p=probabilities,
                ).astype(np.uint8)

                gasless, detached = _interface_neighbourhood(state)
                ref_gasless, ref_detached = self._reference_masks(state)
                np.testing.assert_array_equal(gasless, ref_gasless)
                np.testing.assert_array_equal(detached, ref_detached)
                # Guard the mixture: the branch it exists for must fire.
                fired = ref_detached if expected == "detached" else ref_gasless
                self.assertGreater(int(np.count_nonzero(fired)), 0)

    def test_a_buried_interface_cell_is_gasless_and_supported(self) -> None:
        """
        The two masks are not the same question. A cell the front closed over
        has no gas neighbour *and* plenty of wet ones; an airborne droplet is
        the exact opposite. A kernel that conflated them would still pass a
        test where only one case existed.
        """
        n = 7
        obstacle = np.zeros((n, n, n), dtype=bool)
        state = create_state(obstacle, dx_m=DX_M, dt_s=1e-4)
        state.flags[...] = FLAG_FLUID
        state.flags[3, 3, 3] = FLAG_INTERFACE

        gasless, detached = _interface_neighbourhood(state)
        self.assertTrue(bool(gasless[3, 3, 3]))
        self.assertFalse(bool(detached[3, 3, 3]))
        self.assertEqual(int(np.count_nonzero(gasless)), 1)
        self.assertEqual(int(np.count_nonzero(detached)), 0)
        np.testing.assert_array_equal(gasless, self._reference_masks(state)[0])

    def test_an_isolated_droplet_is_detached_and_not_gasless(self) -> None:
        n = 7
        obstacle = np.zeros((n, n, n), dtype=bool)
        state = create_state(obstacle, dx_m=DX_M, dt_s=1e-4)
        state.flags[...] = FLAG_GAS
        state.flags[3, 3, 3] = FLAG_INTERFACE

        gasless, detached = _interface_neighbourhood(state)
        self.assertTrue(bool(detached[3, 3, 3]))
        self.assertFalse(bool(gasless[3, 3, 3]))
        self.assertEqual(int(np.count_nonzero(detached)), 1)
        np.testing.assert_array_equal(detached, self._reference_masks(state)[1])

    def test_reset_detached_agrees_end_to_end(self) -> None:
        a, b, _ = _mixed_twin(plant=True)
        self.assertEqual(
            reset_detached_interface_cells(a, use_numba=True),
            reset_detached_interface_cells(b, use_numba=False),
        )
        np.testing.assert_array_equal(a.f, b.f)
        np.testing.assert_array_equal(a.rho, b.rho)
        np.testing.assert_array_equal(a.u, b.u)

    def test_heal_agrees_end_to_end(self) -> None:
        """
        Driven with ``dwell_steps=1`` so the conversion fires on the first
        gasless step: at the real 200-step dwell nothing converts in a short
        test and the two paths would agree vacuously.
        """
        a, b, _ = _mixed_twin(plant=True)
        healed_fast = heal_submerged_interface(a, 0.5, 1, use_numba=True)
        healed_slow = heal_submerged_interface(b, 0.5, 1, use_numba=False)
        self.assertEqual(healed_fast, healed_slow)
        np.testing.assert_array_equal(a.flags, b.flags)
        np.testing.assert_array_equal(a.mass, b.mass)
        np.testing.assert_array_equal(a.fill_time_s, b.fill_time_s)

    def test_both_rules_actually_fire_on_the_planted_fixture(self) -> None:
        """Guard the planting; without it both tests above are vacuous."""
        a, _, _ = _mixed_twin(plant=True)
        self.assertGreater(heal_submerged_interface(a, 0.5, 1, use_numba=True), 0)
        b, _, _ = _mixed_twin(plant=True)
        self.assertGreater(reset_detached_interface_cells(b, use_numba=True), 0)


if __name__ == "__main__":
    unittest.main()
