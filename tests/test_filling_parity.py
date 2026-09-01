"""The Numba filling kernels must agree with the NumPy reference.

Mirrors tests/test_parallel_kernel_parity.py, which guards the solidification
kernels the same way. The NumPy path is the readable implementation the
analytical benchmarks were validated against; the kernels are the fast one.
Only a parity test keeps the second honest, because a fused kernel can be
subtly wrong in ways that still look like a plausible fill.

Three agreement levels are asserted deliberately:

- One step from an identical state must agree to round-off. Any real
  divergence in the algebra shows up immediately here.
- Many steps are allowed to drift, because the two paths sum in different
  orders and the scheme runs within 1e-5 of its stability limit. What must NOT
  drift is anything conserved or discrete: total mass, and the cell flags.
- The bookkeeping kernels -- the thermal pair, the neighbourhood masks, the
  whole of ``apply_transitions`` -- are asserted with ``assert_array_equal``
  and no tolerance at all. They are supposed to be bit-identical, and a
  tolerance wide enough to absorb round-off is also wide enough to absorb a
  plausible-looking wrong stencil.

The step paths are no longer bit-identical, and that is deliberate
-----------------------------------------------------------------
Until the float32-storage pass they were: every number the two paths produced
matched to the last bit, and the "round-off" tolerances above were slack that
never got spent. They now round populations at different points -- the numba
path stores ``post`` and ``streamed`` as float32 and reconstructs into float32,
the NumPy reference stays float64 throughout because it is the numerics
reference rather than a performance path -- so the first two levels above
became real tolerances, set from measurement.

The measurements they are set from, taken on the four fixtures below
(quiescent and provoked, each with and without a wet filter), numba against
NumPy from an identical start:

===================  ===============  ==================
field                after 1 step     peak over 100
===================  ===============  ==================
``f``                6.9e-8 rel       1.4e-6 rel
``rho``              0 (exact)        4.8e-7 rel
``u``                0 (exact)        1.6e-7 abs
``mass``             1.7e-8 abs       4.6e-6 abs
``total_mass``       0 (exact)        3.7e-16 rel
flags/fill_time/     exact            exact
primed
===================  ===============  ==================

``rho`` and ``u`` come straight out of ``collide_kernel``'s float64 registers,
so the *first* step cannot move them at all; they only diverge once a rounded
``f`` feeds back into the next step's moments. ``total_mass`` staying at
round-off while the per-cell ``mass`` field drifts by 4.6e-6 is not luck: the
Koerner exchange is pairwise antisymmetric in whatever the populations are
stored as, so the per-cell errors sum to zero -- measured at 2e-15 over 1,584
disagreeing cells. See :class:`Float32StorageDivergenceTests`, which pins every
number in that table, and
``physics.fluid_flow.free_surface_kernel`` for the argument.
"""

from __future__ import annotations

import unittest

import numpy as np

from physics.fluid_flow import (
    NUMBA_AVAILABLE,
    LATTICE_VELOCITIES,
    Q,
    equilibrium,
    lattice_units,
)
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
from physics.filling import (
    REFERENCE_DENSITY,
    TRANSITION_HYSTERESIS,
    apply_transitions,
    ensure_interface_skin,
    fluid_fraction,
)
from physics.filling import (
    _dilate,
    _initialise_promoted_cells,
    _initialise_promoted_temperature,
    _interface_neighbourhood,
    _mask_indices,
    _redistribute,
)

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
    """
    One step must agree to round-off. Anything worse is an algebra bug.

    "Round-off" now means *float32* round-off on the populations and on
    anything derived from them within the step -- see the module docstring.
    The tolerances below are unchanged from when the two paths were
    bit-identical, and they are kept rather than tightened because they were
    always sized for a float32 field; what changed is that they now have
    something to absorb. The measured single-step divergence is recorded next
    to each assertion so the headroom is a number and not a guess.
    """

    def test_populations_agree(self) -> None:
        a, b, units = _twin()
        step(a, units, use_numba=True)
        step(b, units, use_numba=False)
        # Measured: 6.9e-8 relative, 3.7e-9 absolute -- about half a float32
        # ULP, which is exactly what one extra rounding of ``post`` costs.
        # rtol has ~145x headroom over that, atol ~27x.
        np.testing.assert_allclose(a.f, b.f, rtol=1e-5, atol=1e-7)

    def test_density_agrees(self) -> None:
        a, b, units = _twin()
        step(a, units, use_numba=True)
        step(b, units, use_numba=False)
        # Still *exactly* equal after one step, and asserted as such below;
        # the moments are read out of collide_kernel's float64 registers
        # before anything is rounded, so the first step cannot move them.
        np.testing.assert_allclose(a.rho, b.rho, rtol=1e-5, atol=1e-7)
        np.testing.assert_array_equal(a.rho, b.rho)

    def test_velocity_agrees(self) -> None:
        a, b, units = _twin()
        step(a, units, use_numba=True)
        step(b, units, use_numba=False)
        np.testing.assert_allclose(a.u, b.u, rtol=1e-4, atol=1e-9)
        # As with rho: the velocity comes out of the same float64 registers,
        # so one step from an identical state still matches bit for bit.
        np.testing.assert_array_equal(a.u, b.u)

    def test_mass_agrees(self) -> None:
        a, b, units = _twin()
        step(a, units, use_numba=True)
        step(b, units, use_numba=False)
        # One step moves the per-cell mass apart by at most 1.7e-8 absolute /
        # 1.5e-7 relative across the four fixtures -- the float32 rounding of
        # the two populations the Koerner difference reads. Bounds set at
        # ~12x and ~13x that. The old atol of 1e-9 was *below* the measured
        # absolute divergence and survived only because rtol carried the
        # cells with mass near 1; it is not a bound anyone could reason with.
        np.testing.assert_allclose(a.mass, b.mass, rtol=2e-6, atol=2e-7)

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
        # 30 steps accumulate 7.2e-7 absolute / 1.3e-6 relative on the per-cell
        # mass inside the filter; bounds at ~14x and ~7.5x that. This used to
        # be rtol=1e-9, atol=1e-12, which is what bit-identical paths cost --
        # the float32 population buffers are what spent it, and the number
        # above is what they spent.
        np.testing.assert_allclose(a.mass, b.mass, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(a.flags, b.flags)
        # The *total* is not a tolerance question, though. Measured difference
        # after 30 steps: 0.0, on a total of 1512.0 -- the exchange's pairwise
        # cancellation is exact in float32 storage too, so places=9 (5e-10 on
        # that total) is three orders of headroom over anything observed.
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

    def test_the_population_buffers_are_float32_and_mass_is_not(self) -> None:
        """
        The dtype split the whole float32-storage pass rests on.

        Nothing in the step *reads* a dtype and behaves differently, so a
        silent slide back to float64 buffers would cost 500 MB and 8 ms a step
        at 120^3 while every other test in the suite carried on passing. The
        reverse slide -- mass or its scratch going float32 -- would be worse:
        accumulating a conservation sum over millions of cells in float32
        loses exactly the digits the 0.1% invariant is checking, and it would
        show up as a plausible-looking leak rather than as an error.
        """
        state, units = _fixture()
        step(state, units, use_numba=True)

        self.assertEqual(state.f.dtype, np.float32)
        self.assertEqual(state.rho.dtype, np.float32)
        self.assertEqual(state.u.dtype, np.float32)
        self.assertIsNotNone(state._scratch_post)
        self.assertIsNotNone(state._scratch_out)
        self.assertEqual(state._scratch_post.dtype, np.float32)
        self.assertEqual(state._scratch_out.dtype, np.float32)

        self.assertEqual(state.mass.dtype, np.float64)
        self.assertIsNotNone(state._scratch_mass_out)
        self.assertEqual(state._scratch_mass_out.dtype, np.float64)
        # The float64 moment buffers the free-surface pass reads are the one
        # place a float64 population-sized array would still be justified, and
        # they are per-cell, not per-direction.
        self.assertEqual(state._scratch_rho64.dtype, np.float64)
        self.assertEqual(state._scratch_u64.dtype, np.float64)

        # free_surface_kernel used to reconstruct into a float64 (19, nx, ny,
        # nz) buffer and read it back for the mass exchange. With float32
        # populations there is nothing for it to preserve, so it is gone -- and
        # a reintroduced one would be 250 MB at 120^3 that no test would miss
        # except this line.
        self.assertFalse(hasattr(state, "_scratch_f_out"))

    def test_the_numpy_reference_path_still_computes_in_float64(self) -> None:
        """
        The fallback is the numerics reference, not a second performance path.

        It reads the same float32 ``state.f`` the kernels do and must widen it
        immediately; if it ever starts carrying float32 intermediates, the
        thing the parity tolerances are measured *against* has moved, and the
        tolerances stop meaning anything.
        """
        from physics.filling import (
            advance_mass,
            macroscopic,
            reconstruct_gas_populations,
            _stream_with_bounce_back,
        )

        state, units = _fixture()
        rho, u = macroscopic(state)
        self.assertEqual(rho.dtype, np.float64)
        self.assertEqual(u.dtype, np.float64)

        post = np.asarray(state.f, dtype=np.float64)
        streamed = _stream_with_bounce_back(post, state.flags == FLAG_OBSTACLE)
        self.assertEqual(streamed.dtype, np.float64)

        reconstructed = reconstruct_gas_populations(state, streamed, post, u)
        self.assertEqual(reconstructed.dtype, np.float64)
        self.assertEqual(advance_mass(state, post, reconstructed).dtype, np.float64)

        step(state, units, use_numba=False)
        self.assertEqual(state.mass.dtype, np.float64)

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


# ---------------------------------------------------------------------------
# The cell-transition bookkeeping
# ---------------------------------------------------------------------------
#
# apply_transitions and its helpers are the free-surface scheme's mass
# bookkeeping, so these are the strictest parity tests in the file: every field
# is asserted **exactly**, and total mass is asserted equal to the last bit.
# The kernels were written to preserve accumulation order direction by
# direction precisely so that this is achievable -- an allclose here would let
# a reordered sum through, and a reordered sum in a redistribution loop is how
# a conservation bug looks before it becomes a leak.
#
# The one documented exception is in redistribute_gather_kernel's docstring and
# concerns NaN placement after divergence; no test asserts on that, because a
# run that has diverged has already been aborted.

#: A state has to be *provoked* into converting cells. Every synthetic box in
#: this solver parks its interface at eps ~ 0.5 and stays there -- measured 0
#: firing steps in 400 at 64^3 and 120^3, settling, falling and inlet-driven --
#: so the transition paths are unreachable from a plain fixture and would go
#: untested by accident rather than by choice.
PROVOKE_SEED = 811


def _provoked(n: int = 16, *, with_filter: bool = False, submerge: bool = False,
              steps: int = 6, seed: int = PROVOKE_SEED):
    """
    A developed state with a deterministic scatter of cells pushed outside the
    hysteresis band, so ``apply_transitions`` converts in both directions.

    Two thirds of the picked cells are pushed over full and one third under
    empty, which exercises the advancing front, the retreating front, and the
    ``_redistribute`` call that has to balance both at once.
    """
    state, units = _fixture(n=n, with_filter=with_filter, submerge=submerge, seed=seed)
    for _ in range(steps):
        step(state, units, use_numba=True)
        advect_temperature(state, 0.03)

    rng = np.random.default_rng(seed)
    wet = (state.flags == FLAG_FLUID) | (state.flags == FLAG_INTERFACE)
    state.temperature = np.full(state.shape, NO_TEMPERATURE, dtype=np.float32)
    state.temperature[wet] = (
        1400.0 + 80.0 * rng.standard_normal(int(np.count_nonzero(wet)))
    ).astype(np.float32)

    idx = np.argwhere(state.flags == FLAG_INTERFACE)
    pick = idx[rng.choice(len(idx), size=max(6, len(idx) // 5), replace=False)]
    full = state.rho.astype(np.float64) * state.porous.porosity_f64
    for j, cell in enumerate(pick):
        at = (int(cell[0]), int(cell[1]), int(cell[2]))
        state.mass[at] = full[at] * 1.08 + 0.01 if j % 3 else -0.02
    return state, units


def _provoked_twin(**kwargs):
    a, units = _provoked(**kwargs)
    b, _ = _provoked(**kwargs)
    _assert_states_identical(a, b, "fixture")
    return a, b, units


#: Everything apply_transitions is allowed to touch. Asserting on a subset
#: would let a kernel corrupt whatever was left out.
STATE_FIELDS = ("flags", "mass", "f", "rho", "u", "temperature", "fill_time_s")


def _assert_states_identical(a, b, label: str) -> None:
    for name in STATE_FIELDS:
        np.testing.assert_array_equal(
            getattr(a, name), getattr(b, name), err_msg=f"{label}: {name}"
        )
    np.testing.assert_array_equal(
        a.porous.primed, b.porous.primed, err_msg=f"{label}: primed"
    )


class ProvokedFixtureTests(unittest.TestCase):
    """Guard the fixture: none of the tests below mean anything without this."""

    def test_transitions_actually_fire_in_both_directions(self) -> None:
        state, _ = _provoked()
        filled, emptied = apply_transitions(state, 0.1, use_numba=False)
        self.assertGreater(filled, 0)
        self.assertGreater(emptied, 0)

    def test_the_fronts_are_not_empty(self) -> None:
        """
        Both the advancing and the retreating mask must have cells in them, or
        transition_front_kernel is being compared against nothing.
        """
        state, _ = _provoked()
        eps = fluid_fraction(state)
        interface = state.flags == FLAG_INTERFACE
        to_fluid = interface & (eps > 1.0 + TRANSITION_HYSTERESIS)
        to_gas = interface & (eps < -TRANSITION_HYSTERESIS)
        state.flags[to_fluid] = FLAG_FLUID
        state.flags[to_gas] = FLAG_GAS
        promote = _dilate(to_fluid) & (state.flags == FLAG_GAS)
        demote = _dilate(to_gas) & (state.flags == FLAG_FLUID)
        self.assertGreater(int(np.count_nonzero(promote)), 0)
        self.assertGreater(int(np.count_nonzero(demote)), 0)

    def test_a_plain_fixture_converts_nothing(self) -> None:
        """
        The measurement behind PROVOKE_SEED's docstring, kept as a test so the
        provocation is not quietly carrying a fixture that would have worked
        anyway.
        """
        state, units = _fixture()
        for _ in range(20):
            step(state, units, use_numba=True)
        self.assertEqual(apply_transitions(state, 0.1, use_numba=False), (0, 0))


# ---------------------------------------------------------------------------
# What float32 population storage actually costs
# ---------------------------------------------------------------------------


#: Regimes the divergence envelope is measured over. Both halves matter: the
#: quiescent fixture never converts a cell (see PROVOKE_SEED), so on its own it
#: would measure only the collide/stream/exchange chain and say nothing about
#: whether a rounded fill fraction can flip a hysteresis crossing. The filter
#: variants reach the porosity branch of the mass exchange, which weights by
#: eps rather than 1 and so multiplies the rounding rather than cancelling it.
DIVERGENCE_REGIMES: tuple[tuple[str, dict, bool], ...] = (
    ("quiescent", {}, False),
    ("quiescent + wet filter", {"with_filter": True, "submerge": True}, False),
    ("firing", {}, True),
    ("firing + wet filter", {"with_filter": True, "submerge": True}, True),
)

#: Steps each regime is walked for. 100 is where the measurements in the module
#: docstring were taken; the envelope grows roughly as sqrt(steps), so a longer
#: walk would need its own numbers rather than these ones.
DIVERGENCE_STEPS = 100


def _divergence_walk(kwargs, provoked, steps):
    """
    Step a numba/NumPy twin side by side and collect the worst divergence.

    Returns a dict of peaks over the whole walk, plus the step at which each
    discrete field first stopped being exactly equal (absent = never).
    """
    if provoked:
        a, units = _provoked(**kwargs)
        b, _ = _provoked(**kwargs)
    else:
        a, b, units = _twin(**kwargs)
    _assert_states_identical(a, b, "divergence fixture")

    start = total_mass(a)
    peak = {name: {"abs": 0.0, "rel": 0.0} for name in ("f", "rho", "u", "mass")}
    first = {"abs": {}, "rel": {}}
    broke: dict[str, int] = {}
    total_gap = 0.0
    conservation = 0.0

    for s in range(1, steps + 1):
        step(a, units, use_numba=True)
        step(b, units, use_numba=False)
        for name in peak:
            x = np.asarray(getattr(a, name), dtype=np.float64)
            y = np.asarray(getattr(b, name), dtype=np.float64)
            d = np.abs(x - y)
            scale = np.maximum(np.abs(x), np.abs(y))
            live = scale > 0
            amax = float(d.max())
            rmax = float((d[live] / scale[live]).max()) if np.any(live) else 0.0
            if s == 1:
                first["abs"][name] = amax
                first["rel"][name] = rmax
            peak[name]["abs"] = max(peak[name]["abs"], amax)
            peak[name]["rel"] = max(peak[name]["rel"], rmax)
        total_gap = max(total_gap, abs(total_mass(a) - total_mass(b)) / start)
        conservation = max(conservation, abs(total_mass(a) - start) / start)
        for name, x, y in (
            ("flags", a.flags, b.flags),
            ("fill_time_s", a.fill_time_s, b.fill_time_s),
            ("primed", a.porous.primed, b.porous.primed),
        ):
            if name not in broke and not np.array_equal(x, y):
                broke[name] = s

    return {
        "peak": peak,
        "first_step": first,
        "total_gap": total_gap,
        "conservation": conservation,
        "broke": broke,
    }


@unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
class Float32StorageDivergenceTests(unittest.TestCase):
    """
    The price of storing the population buffers float32, measured and pinned.

    ``collide_kernel`` used to write a float64 ``post`` that streamed into a
    float64 buffer and was reconstructed into a float64 ``f_out``; all three are
    float32 now, and the arithmetic that reads them stays float64. Before that
    change every assertion in this file could have been ``assert_array_equal``:
    the numba and NumPy step paths agreed to the last bit on ``f``, ``rho``,
    ``u``, ``mass``, ``flags``, ``fill_time_s`` and ``primed``, in all four
    regimes, for as long as anyone cared to run them. That is no longer true,
    and the point of this class is that the *amount* by which it is no longer
    true is a measured number with a test around it rather than a shrug.

    Every bound below is the measured peak over 100 steps of all four regimes,
    rounded up and given the headroom stated beside it. They are ceilings, not
    targets: if a future change makes them fail, the change moved the numerics,
    and the answer is to find out why -- not to raise the ceiling.
    """

    envelopes: dict

    @classmethod
    def setUpClass(cls) -> None:
        # One walk per regime, shared by every test below. Each walk is 100
        # paired steps on a 14^3 or 16^3 grid; doing it per test method would
        # multiply that by the number of assertions for no extra coverage.
        cls.envelopes = {
            label: _divergence_walk(kwargs, provoked, DIVERGENCE_STEPS)
            for label, kwargs, provoked in DIVERGENCE_REGIMES
        }

    def test_the_first_step_leaves_the_moments_untouched(self) -> None:
        """
        ``rho`` and ``u`` cannot move on step one, and that is a structural
        claim rather than a tolerance.

        Both come out of ``collide_kernel``'s float64 registers and are stored
        float32 exactly as the NumPy path stores them, before anything in the
        step has been rounded to float32 twice. If this ever fails, the
        narrowing has crept into the moment arithmetic rather than staying at
        the population store, which is precisely the line this pass drew.
        """
        for label in self.envelopes:
            with self.subTest(regime=label):
                first = self.envelopes[label]["first_step"]
                self.assertEqual(first["abs"]["rho"], 0.0)
                self.assertEqual(first["abs"]["u"], 0.0)

    def test_the_first_step_costs_about_half_a_float32_ulp_on_f(self) -> None:
        """
        Measured across the four regimes: 6.95e-8 relative, 3.73e-9 absolute.

        float32 has a relative spacing of 1.19e-7, so a single rounding of
        ``post`` costs half of one, and that is what this is (the measured
        figure sits a little over 5.96e-8 because the relative error is scored
        against ``max(|a|, |b|)`` and the spacing widens just under a power of
        two). It is the evidence that the narrowing happens once per step
        rather than compounding within it.

        Bounds at 2e-7 relative (~3x) and 4e-8 absolute (~11x). Deliberately
        the tightest assertions in this file, but honest about their reach: a
        *second* rounding inside the step would land near 1.4e-7 relative,
        which is still inside 2e-7, so this catches an order-of-magnitude
        widening rather than a doubling. The thing that would catch a doubling
        is the sibling test above -- rho and u are asserted exactly equal after
        one step, and any extra rounding early enough to double ``f`` would
        have to pass through the moments first.
        """
        for label in self.envelopes:
            with self.subTest(regime=label):
                first = self.envelopes[label]["first_step"]
                self.assertLess(first["rel"]["f"], 2e-7)
                self.assertLess(first["abs"]["f"], 4e-8)

    def test_a_hundred_steps_stay_inside_the_measured_envelope(self) -> None:
        """
        Peaks over 100 steps, worst of the four regimes:
        ``f`` 1.42e-6 rel / 2.98e-7 abs, ``rho`` 4.77e-7 rel, ``u`` 1.58e-7
        abs, ``mass`` 4.56e-6 abs. Every bound below is 10x the measurement.

        ``u`` and ``mass`` are bounded absolutely, not relatively, and that is
        not a dodge: both fields have entries legitimately at or near zero --
        a stationary cell, an interface cell that has just given up nearly all
        its metal -- where a relative measure divides by noise. The worst
        relative disagreement on ``mass`` reaches 1.6 on such a cell while its
        absolute disagreement is 5e-6, and it is the absolute figure that says
        whether the fill is the same fill.
        """
        bounds = {
            ("f", "rel"): 1.5e-5,
            ("f", "abs"): 3e-6,
            ("rho", "rel"): 5e-6,
            ("u", "abs"): 1.6e-6,
            ("mass", "abs"): 5e-5,
        }
        for label, env in self.envelopes.items():
            for (name, kind), bound in bounds.items():
                with self.subTest(regime=label, field=name, measure=kind):
                    self.assertLess(env["peak"][name][kind], bound)

    def test_the_discrete_fields_never_diverge_at_all(self) -> None:
        """
        Flags, arrival times and the filter's primed latch stay *exactly* equal
        for the whole 100 steps, in every regime.

        This is the assertion the float32 pass was most likely to break and the
        one worth the most: cell transitions fire on an eps threshold crossing,
        so a rounded fill fraction could in principle land on the far side of
        ``1 + TRANSITION_HYSTERESIS`` on one path and the near side on the
        other, and the two fills would then differ by a whole cell rather than
        by 1e-6 of one. It does not happen here, and the hysteresis band is why:
        at 1e-3 it is four orders of magnitude wider than the 1e-7 the rounding
        moves eps by, so a crossing is only at risk for a cell whose eps sits
        within 1e-7 of the band edge and stays there.

        The band is not a guarantee, only a very long lever, so this test
        records a horizon rather than a theorem. If a future fixture does flip
        a crossing, the honest fix is to report the first divergent step here
        and assert exact equality up to it -- not to relax the assertion.
        """
        for label, env in self.envelopes.items():
            with self.subTest(regime=label):
                self.assertEqual(
                    env["broke"], {},
                    f"{label}: a discrete field diverged; first divergent step "
                    f"per field is {env['broke']}",
                )

    def test_total_mass_stays_exact_across_both_paths(self) -> None:
        """
        The per-cell mass field drifts by up to 4.6e-6; its *sum* does not
        drift at all.

        Measured: 3.7e-16 relative between the paths, and the same for the
        numba path against its own starting total, over 100 steps of every
        regime. That is float64 round-off on a sum of ~1,500, not a
        conservation error, and it is the structural consequence of the Koerner
        exchange being pairwise antisymmetric in whatever the populations are
        stored as -- the per-cell errors are equal and opposite across each
        pair, so they cancel in the sum. Directly measured on the wet-filter
        fixture: 1,584 cells disagree by up to 7.2e-7 each, and the sum of
        those disagreements is 2.2e-15.

        Bound at 1e-12 relative -- ~2,700x the measurement, and still six
        orders inside the module's 0.1% invariant. A real leak would blow
        through it immediately.
        """
        for label, env in self.envelopes.items():
            with self.subTest(regime=label):
                self.assertLess(env["total_gap"], 1e-12)
                self.assertLess(env["conservation"], 1e-12)


@unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
class InletLedgerConservationTests(unittest.TestCase):
    """
    Mass delivered in equals mass in the domain, on the numba path, over a long
    provoked pour.

    The closed-box conservation tests elsewhere in the suite start with all the
    metal already present and only have to avoid losing it. This one runs the
    boundary that a real fill runs on: ``core/filling_stage.py`` holds the inlet
    cells full every step and books the difference between what it writes and
    what it overwrote as delivered mass. That ledger is the actual definition of
    the 0.1% invariant in the module docstring, and nothing in ``physics/``
    tested it, because ``physics/`` may not import ``core/``. So the pin is
    reproduced here, minus the mesh.

    Every write into ``state.mass`` from outside the solver is booked the same
    way, including the provocation -- which is what makes it a ledger rather
    than a hope.
    """

    N = 12
    NZ = 20
    STEPS = 600

    def _pour(self, use_numba: bool):
        obstacle = np.zeros((self.N, self.N, self.NZ), dtype=bool)
        obstacle[0] = obstacle[-1] = True
        obstacle[:, 0] = obstacle[:, -1] = True
        obstacle[:, :, 0] = True
        units = lattice_units(DX_M, WCB_NU, u_max_physical_m_s=2.8)
        state = create_state(obstacle, dx_m=DX_M, dt_s=units.dt_s)

        inlet = np.zeros(state.shape, dtype=bool)
        mid = self.N // 2
        inlet[mid - 1 : mid + 1, mid - 1 : mid + 1, self.NZ - 2] = True
        ix, iy, iz = np.nonzero(inlet)
        # The cells the inlet can stream into, kept wet the way the stage keeps
        # them wet -- otherwise the cell above the inlet drains to gas and the
        # inlet stops receiving anything arriving downward.
        halo = _dilate(inlet) & ~inlet & ~obstacle
        n_in = int(inlet.sum())

        # A pressure boundary, 2% over reference, like the stage's head mode.
        inlet_rho = np.full(n_in, REFERENCE_DENSITY * 1.02, dtype=np.float64)
        inlet_u = np.zeros((3, n_in), dtype=np.float64)
        inlet_full = np.full(n_in, float(REFERENCE_DENSITY), dtype=np.float64)

        rng = np.random.default_rng(PROVOKE_SEED)
        delivered = 0.0
        worst = 0.0
        converted = [0, 0]  # INTERFACE -> FLUID, INTERFACE -> GAS

        for s in range(1, self.STEPS + 1):
            state.flags[ix, iy, iz] = FLAG_FLUID
            ensure_interface_skin(state, halo, use_numba=use_numba)
            state.f[:, ix, iy, iz] = equilibrium(inlet_rho, inlet_u).astype(
                np.float32
            )
            delivered += float(inlet_full.sum()) - float(
                state.mass[ix, iy, iz].sum()
            )
            state.mass[ix, iy, iz] = inlet_full
            state.temperature[ix, iy, iz] = np.float32(1400.0)

            # Provocation, booked into the ledger exactly like the pin above:
            # shove a scatter of interface cells outside the hysteresis band in
            # both directions, so promote, demote, _redistribute and the
            # promoted-cell seeding all fire repeatedly over the run instead of
            # only where the front happens to reach.
            if s % 5 == 0:
                idx = np.argwhere(state.flags == FLAG_INTERFACE)
                if len(idx):
                    pick = idx[
                        rng.choice(
                            len(idx),
                            size=min(len(idx), max(2, len(idx) // 8)),
                            replace=False,
                        )
                    ]
                    full = state.rho.astype(np.float64) * state.porous.porosity_f64
                    for j, cell in enumerate(pick):
                        at = (int(cell[0]), int(cell[1]), int(cell[2]))
                        value = full[at] * 1.07 + 0.01 if j % 3 else -0.02
                        delivered += value - float(state.mass[at])
                        state.mass[at] = value

            before = state.flags.copy()
            step(state, units, track_free_surface=True, use_numba=use_numba)
            # Conversions actually made this step, counted off the flags rather
            # than taken on trust from apply_transitions' return value -- this
            # is the fixture guard, and a pour that quietly converted nothing
            # would make the ledger assertion vacuous.
            was_interface = before == FLAG_INTERFACE
            converted[0] += int(
                np.count_nonzero(was_interface & (state.flags == FLAG_FLUID))
            )
            converted[1] += int(
                np.count_nonzero(was_interface & (state.flags == FLAG_GAS))
            )
            if delivered > 0.0:
                worst = max(
                    worst, abs(total_mass(state) - delivered) / delivered
                )

        return state, delivered, worst, tuple(converted)

    def test_the_numba_path_holds_the_ledger(self) -> None:
        """
        Measured over 600 steps: worst-in-run discrepancy 1.92e-15 relative,
        final 3.56e-16, against 1,916.36 lattice mass units delivered through
        4 inlet cells -- float64 round-off, twelve orders of magnitude inside
        the 0.1% the module promises. 1,882 cells end FLUID, and the run makes
        11,281 INTERFACE -> FLUID conversions and 1,367 the other way along the
        route -- identical counts on both paths -- so the transition
        bookkeeping and ``_redistribute`` are carrying real traffic rather than
        idling.

        Asserted at 1e-3, the invariant itself, rather than at the measurement:
        this test exists to catch a *leak*, and a bound tight enough to catch
        round-off would be a bound that fails for reasons no foundry cares
        about. The measurement is recorded here so a future drift to, say,
        1e-6 is visibly a regression even while the assertion still passes.
        """
        state, delivered, worst, converted = self._pour(use_numba=True)
        self.assertGreater(delivered, 0.0)
        self.assertGreater(
            int(np.count_nonzero(state.flags == FLAG_FLUID)), 100,
            "the pour never filled anything",
        )
        self.assertGreater(converted[0], 1000, "the front never advanced")
        self.assertGreater(converted[1], 100, "the front never retreated")
        self.assertLess(
            worst, 1e-3,
            f"inlet ledger broke: worst {worst:.3e} over {self.STEPS} steps",
        )
        self.assertLess(
            abs(total_mass(state) - delivered) / delivered, 1e-3
        )

    def test_the_numpy_path_holds_the_same_ledger(self) -> None:
        """
        The reference path, so the test above is known to be measuring the
        solver rather than the harness. Measured: worst 1.51e-15, final
        1.07e-15, on the same 600 steps.
        """
        state, delivered, worst, _ = self._pour(use_numba=False)
        self.assertGreater(delivered, 0.0)
        self.assertLess(worst, 1e-3)
        self.assertLess(
            abs(total_mass(state) - delivered) / delivered, 1e-3
        )


@unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
class MaskCompactionParityTests(unittest.TestCase):
    """``compact_mask_kernel`` is a parallel ``np.nonzero``; prove it."""

    def _check(self, mask) -> None:
        want = np.nonzero(mask)
        got = _mask_indices(mask)
        for axis, (g, w) in enumerate(zip(got, want)):
            np.testing.assert_array_equal(g, w, err_msg=f"axis {axis}")

    def test_matches_numpy_on_random_masks(self) -> None:
        rng = np.random.default_rng(31)
        for density in (0.0, 1e-4, 0.02, 0.5, 1.0):
            with self.subTest(density=density):
                mask = rng.random((13, 11, 9)) < density
                self._check(mask)

    def test_matches_numpy_on_degenerate_shapes(self) -> None:
        """
        The prefix sum runs over the x axis, so a single x slab is the case
        where an off-by-one in the offsets would hide.
        """
        rng = np.random.default_rng(32)
        for shape in ((1, 6, 5), (2, 1, 1), (7, 1, 3), (1, 1, 1), (5, 4, 1)):
            with self.subTest(shape=shape):
                self._check(rng.random(shape) < 0.4)

    def test_the_order_is_the_one_mask_assignment_uses(self) -> None:
        """
        The whole point of matching C order: the caller writes through the
        index list where the reference path wrote through the mask, so the two
        have to visit cells in the same sequence.
        """
        rng = np.random.default_rng(33)
        mask = rng.random((9, 8, 7)) < 0.3
        n = int(np.count_nonzero(mask))
        by_mask = np.zeros((9, 8, 7), dtype=np.float64)
        by_mask[mask] = np.arange(n, dtype=np.float64)
        ix, iy, iz = _mask_indices(mask)
        by_index = np.zeros((9, 8, 7), dtype=np.float64)
        by_index[ix, iy, iz] = np.arange(n, dtype=np.float64)
        np.testing.assert_array_equal(by_mask, by_index)


@unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
class TransitionFrontParityTests(unittest.TestCase):
    """
    ``transition_front_kernel`` scatters from the converting cells where the
    reference dilates towards them. Boolean masks, so exact or nothing.
    """

    @staticmethod
    def _reference(state, to_fluid, to_gas):
        """
        The two expressions apply_transitions' NumPy fallback uses. Written out
        rather than imported: this is the specification, and a shared helper
        could be edited to agree with a broken kernel.
        """
        promote = _dilate(to_fluid) & (state.flags == FLAG_GAS)
        demote = _dilate(to_gas) & (state.flags == FLAG_FLUID)
        return promote, demote

    def _kernel(self, state, to_fluid, to_gas):
        from physics.fluid_flow import lattice_tables, transition_front_kernel

        _C, Ci, _W, _OPP = lattice_tables()
        promote = np.empty(state.shape, dtype=bool)
        demote = np.empty(state.shape, dtype=bool)
        transition_front_kernel(state.flags, to_fluid, to_gas, Ci, promote, demote)
        return promote, demote

    @staticmethod
    def _split(state):
        eps = fluid_fraction(state)
        interface = state.flags == FLAG_INTERFACE
        to_fluid = interface & (eps > 1.0 + TRANSITION_HYSTERESIS)
        to_gas = interface & (eps < -TRANSITION_HYSTERESIS)
        # The masks are read after the flag writes on both paths.
        state.flags[to_fluid] = FLAG_FLUID
        state.flags[to_gas] = FLAG_GAS
        return to_fluid, to_gas

    def test_masks_agree_on_a_provoked_state(self) -> None:
        state, _ = _provoked()
        to_fluid, to_gas = self._split(state)
        kp, kd = self._kernel(state, to_fluid, to_gas)
        rp, rd = self._reference(state, to_fluid, to_gas)
        np.testing.assert_array_equal(kp, rp)
        np.testing.assert_array_equal(kd, rd)
        self.assertGreater(int(np.count_nonzero(rp)), 0)
        self.assertGreater(int(np.count_nonzero(rd)), 0)

    def test_masks_agree_on_randomised_flags(self) -> None:
        """
        Dense converting sets, which a real front never produces: every gas
        cell then has several to_fluid neighbours and the scatter's duplicate
        writes all collide.
        """
        rng = np.random.default_rng(41)
        shape = (12, 10, 11)
        state = create_state(np.zeros(shape, dtype=bool), dx_m=DX_M, dt_s=1e-4)
        state.flags = rng.choice(
            np.array(
                [FLAG_OBSTACLE, FLAG_GAS, FLAG_INTERFACE, FLAG_FLUID], dtype=np.uint8
            ),
            size=shape,
            p=(0.1, 0.35, 0.2, 0.35),
        ).astype(np.uint8)
        to_fluid = (rng.random(shape) < 0.15) & (state.flags == FLAG_FLUID)
        to_gas = (rng.random(shape) < 0.15) & (state.flags == FLAG_GAS)

        kp, kd = self._kernel(state, to_fluid, to_gas)
        rp, rd = self._reference(state, to_fluid, to_gas)
        np.testing.assert_array_equal(kp, rp)
        np.testing.assert_array_equal(kd, rd)

    def test_masks_agree_when_the_x_axis_is_too_short_to_have_three_planes(self) -> None:
        """
        The scatter parallelises over x and reads source planes x-1, x, x+1. At
        nx < 3 the periodic wrap makes those collide, so the same source cell is
        visited two or three times -- harmless only because the writes are
        idempotent, which is exactly the kind of claim that needs a test.
        """
        rng = np.random.default_rng(42)
        for nx in (1, 2, 3):
            with self.subTest(nx=nx):
                shape = (nx, 7, 6)
                state = create_state(
                    np.zeros(shape, dtype=bool), dx_m=DX_M, dt_s=1e-4
                )
                state.flags = rng.choice(
                    np.array(
                        [FLAG_OBSTACLE, FLAG_GAS, FLAG_INTERFACE, FLAG_FLUID],
                        dtype=np.uint8,
                    ),
                    size=shape,
                    p=(0.1, 0.35, 0.2, 0.35),
                ).astype(np.uint8)
                to_fluid = (rng.random(shape) < 0.2) & (state.flags == FLAG_FLUID)
                to_gas = (rng.random(shape) < 0.2) & (state.flags == FLAG_GAS)
                kp, kd = self._kernel(state, to_fluid, to_gas)
                rp, rd = self._reference(state, to_fluid, to_gas)
                np.testing.assert_array_equal(kp, rp)
                np.testing.assert_array_equal(kd, rd)


@unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
class RedistributeParityTests(unittest.TestCase):
    """
    ``_redistribute`` moves leftover mass between cells, so it is the one
    helper where a reordered sum would show up as a conservation error rather
    than as noise.
    """

    @staticmethod
    def _excess(state, seed: int):
        """A signed excess field on a scatter of cells, donors and takers both."""
        rng = np.random.default_rng(seed)
        excess = np.zeros(state.shape, dtype=np.float64)
        idx = np.argwhere(
            (state.flags == FLAG_INTERFACE) | (state.flags == FLAG_FLUID)
        )
        pick = idx[rng.choice(len(idx), size=max(8, len(idx) // 8), replace=False)]
        values = rng.standard_normal(len(pick)) * 0.05
        for cell, v in zip(pick, values):
            excess[int(cell[0]), int(cell[1]), int(cell[2])] = float(v)
        return excess

    def test_agrees_exactly(self) -> None:
        a, b, _ = _provoked_twin()
        excess = self._excess(a, 51)
        _redistribute(a, excess.copy(), use_numba=True)
        _redistribute(b, excess.copy(), use_numba=False)
        np.testing.assert_array_equal(a.mass, b.mass)

    def test_conserves_the_excess_to_the_ulp_on_both_paths(self) -> None:
        """
        Mass in equals mass out. Not "to 0.1%" -- the redistribution is a
        rebalance, so the only honest tolerance is the float64 sum's own.
        """
        for label, use_numba in (("numba", True), ("numpy", False)):
            with self.subTest(path=label):
                state, _ = _provoked()
                excess = self._excess(state, 52)
                before = float(state.mass.sum())
                handed_out = float(excess.sum())
                _redistribute(state, excess, use_numba=use_numba)
                after = float(state.mass.sum())
                self.assertAlmostEqual(
                    after - before, handed_out, delta=1e-9 * max(abs(before), 1.0)
                )

    def test_agrees_when_a_donor_has_no_interface_neighbour(self) -> None:
        """
        The stranded-mass branch: the excess stays on the donor rather than
        being dropped, which is what keeps conservation true for an isolated
        cell. Both paths must strand the same amount in the same place.
        """
        n = 7
        for use_numba in (True, False):
            state = create_state(
                np.zeros((n, n, n), dtype=bool), dx_m=DX_M, dt_s=1e-4
            )
            state.flags[...] = FLAG_GAS
            state.flags[3, 3, 3] = FLAG_FLUID
            state.mass[3, 3, 3] = 0.25
            excess = np.zeros((n, n, n), dtype=np.float64)
            excess[3, 3, 3] = 0.125
            _redistribute(state, excess, use_numba=use_numba)
            with self.subTest(use_numba=use_numba):
                self.assertEqual(float(state.mass[3, 3, 3]), 0.375)
                self.assertEqual(float(state.mass.sum()), 0.375)


@unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
class PromotedSeedingParityTests(unittest.TestCase):
    """
    The donor-averaged seeding of freshly promoted cells, kernel against
    reference. This is the pair that used to build a full-domain equilibrium.
    """

    @staticmethod
    def _promote_mask(state):
        gas = state.flags == FLAG_GAS
        wet = (state.flags == FLAG_FLUID) | (state.flags == FLAG_INTERFACE)
        return gas & _dilate(wet)

    def test_populations_agree_exactly(self) -> None:
        a, b, _ = _provoked_twin()
        promote = self._promote_mask(a)
        self.assertGreater(int(np.count_nonzero(promote)), 0)
        _initialise_promoted_cells(a, promote, use_numba=True)
        _initialise_promoted_cells(b, promote, use_numba=False)
        np.testing.assert_array_equal(a.f, b.f)
        np.testing.assert_array_equal(a.rho, b.rho)

    def test_populations_agree_through_a_wet_filter(self) -> None:
        """
        The compressed equilibrium takes ``porosity`` per cell. Outside a filter
        that is exactly 1.0 and the porosity term is inert, so a fixture without
        one cannot tell a correct gather of it from no gather at all.
        """
        a, b, _ = _provoked_twin(with_filter=True, submerge=True)
        promote = self._promote_mask(a)
        self.assertGreater(int(np.count_nonzero(promote)), 0)
        _initialise_promoted_cells(a, promote, use_numba=True)
        _initialise_promoted_cells(b, promote, use_numba=False)
        np.testing.assert_array_equal(a.f, b.f)
        np.testing.assert_array_equal(a.rho, b.rho)

    def test_temperatures_agree_exactly(self) -> None:
        a, b, _ = _provoked_twin()
        promote = self._promote_mask(a)
        _initialise_promoted_temperature(a, promote, use_numba=True)
        _initialise_promoted_temperature(b, promote, use_numba=False)
        np.testing.assert_array_equal(a.temperature, b.temperature)

    def test_the_donorless_fallback_agrees(self) -> None:
        """
        A promoted cell whose only wet neighbours were promoted alongside it has
        no donor, and falls back to reference density at rest. Sheet promotion
        makes that routine rather than exotic -- see
        ``_initialise_promoted_cells`` -- so both paths need the same fallback.
        """
        n = 7
        a = create_state(np.zeros((n, n, n), dtype=bool), dx_m=DX_M, dt_s=1e-4)
        b = create_state(np.zeros((n, n, n), dtype=bool), dx_m=DX_M, dt_s=1e-4)
        promote = np.zeros((n, n, n), dtype=bool)
        promote[2:5, 2:5, 2:5] = True
        for state in (a, b):
            state.flags[...] = FLAG_GAS
            state.flags[promote] = FLAG_INTERFACE
            state.rho[...] = 0.0
            state.f[...] = 0.0
        _initialise_promoted_cells(a, promote, use_numba=True)
        _initialise_promoted_cells(b, promote, use_numba=False)
        np.testing.assert_array_equal(a.f, b.f)
        np.testing.assert_array_equal(a.rho, b.rho)
        self.assertEqual(
            float(a.rho[3, 3, 3]), float(np.float32(REFERENCE_DENSITY))
        )
        self.assertEqual(float(np.abs(a.u[:, 3, 3, 3]).max()), 0.0)

    def test_the_temperature_fallback_agrees(self) -> None:
        """
        The no-local-donor temperature falls back to the whole-domain mean, and
        both paths compute that mean with ``np.mean`` on purpose -- a prange
        reduction would sum in a different order and this is where it would
        show. So it must be exactly equal, not merely close.
        """
        n = 9
        states = []
        for _ in range(2):
            state = create_state(
                np.zeros((n, n, n), dtype=bool), dx_m=DX_M, dt_s=1e-4
            )
            state.flags[...] = FLAG_FLUID
            state.temperature[...] = np.arange(n ** 3, dtype=np.float32).reshape(
                (n, n, n)
            )
            states.append(state)
        promote = np.zeros((n, n, n), dtype=bool)
        # An island far from any donor, so every cell in it takes the fallback.
        promote[3:6, 3:6, 3:6] = True
        for state in states:
            state.flags[promote] = FLAG_INTERFACE
            state.temperature[promote] = NO_TEMPERATURE
        _initialise_promoted_temperature(states[0], promote, use_numba=True)
        _initialise_promoted_temperature(states[1], promote, use_numba=False)
        np.testing.assert_array_equal(states[0].temperature, states[1].temperature)
        self.assertTrue(bool(np.all(np.isfinite(states[0].temperature[promote]))))


@unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
class TransitionKernelParityTests(unittest.TestCase):
    """``apply_transitions`` end to end, every field, exactly."""

    def test_agrees_exactly_on_a_provoked_state(self) -> None:
        a, b, _ = _provoked_twin()
        self.assertEqual(
            apply_transitions(a, 0.25, use_numba=True),
            apply_transitions(b, 0.25, use_numba=False),
        )
        _assert_states_identical(a, b, "apply_transitions")
        self.assertEqual(total_mass(a), total_mass(b))

    def test_agrees_exactly_through_a_wet_filter(self) -> None:
        """
        The porosity branch: ``full = rho * porosity``, so the eps scan, the
        mass snap and the promoted-cell equilibrium all read porosity, and a
        fixture with none of them below 1.0 proves nothing about any of it.
        """
        a, b, _ = _provoked_twin(with_filter=True, submerge=True)
        self.assertEqual(
            apply_transitions(a, 0.25, use_numba=True),
            apply_transitions(b, 0.25, use_numba=False),
        )
        _assert_states_identical(a, b, "apply_transitions with a filter")
        self.assertEqual(total_mass(a), total_mass(b))

    def test_a_quiescent_state_converts_nothing_on_either_path(self) -> None:
        """
        The scan's early return. The kernel path runs the sweep and then bails
        on the counts, so it must leave the state untouched -- including
        ``state.mass``, which the kernel is otherwise allowed to write.
        """
        a, b, units = _twin()
        for _ in range(4):
            step(a, units, use_numba=True)
            step(b, units, use_numba=True)
        before = a.mass.copy()
        self.assertEqual(apply_transitions(a, 0.1, use_numba=True), (0, 0))
        self.assertEqual(apply_transitions(b, 0.1, use_numba=False), (0, 0))
        np.testing.assert_array_equal(a.mass, before)
        _assert_states_identical(a, b, "quiescent")

    def test_agrees_over_many_provoked_steps(self) -> None:
        """
        One call cannot distinguish a front kernel that is right at the first
        transition from one that is right only there. Repeated provocation walks
        the fill through advance and retreat many times over.
        """
        a, b, units = _provoked_twin()
        full_a = a.rho.astype(np.float64) * a.porous.porosity_f64
        rng = np.random.default_rng(PROVOKE_SEED + 5)
        for _ in range(12):
            step(a, units, use_numba=True)
            step(b, units, use_numba=True)
            # Same provocation on both, from the same draw.
            idx = np.argwhere(a.flags == FLAG_INTERFACE)
            if len(idx):
                pick = idx[rng.choice(len(idx), size=max(2, len(idx) // 8),
                                      replace=False)]
                for j, cell in enumerate(pick):
                    at = (int(cell[0]), int(cell[1]), int(cell[2]))
                    value = full_a[at] * 1.07 + 0.01 if j % 3 else -0.02
                    a.mass[at] = value
                    b.mass[at] = value
            ra = apply_transitions(a, 0.3, use_numba=True)
            rb = apply_transitions(b, 0.3, use_numba=False)
            self.assertEqual(ra, rb)
            _assert_states_identical(a, b, "repeated provocation")

    def test_ensure_interface_skin_agrees_exactly(self) -> None:
        """
        The other caller of the promoted-cell seeding. It writes FLAG_FLUID by
        hand, which is outside apply_transitions' bookkeeping, so it has its own
        path through the same helpers.
        """
        a, b, _ = _provoked_twin()
        halo = _dilate(a.flags == FLAG_FLUID)
        self.assertEqual(
            ensure_interface_skin(a, halo, use_numba=True),
            ensure_interface_skin(b, halo, use_numba=False),
        )
        _assert_states_identical(a, b, "ensure_interface_skin")


if __name__ == "__main__":
    unittest.main()
