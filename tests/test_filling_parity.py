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

from physics.fluid_flow import NUMBA_AVAILABLE, lattice_units
from physics.filling import (
    FLAG_FLUID,
    FLAG_GAS,
    PorousZone,
    create_state,
    step,
    total_mass,
)

WCB_NU = 0.006 / 7200.0
DX_M = 0.0152


def _fixture(n: int = 14, *, with_filter: bool = False, seed: int = 7):
    """A partly filled box, optionally with a filter slab across the middle."""
    obstacle = np.zeros((n, n, n), dtype=bool)
    obstacle[0] = obstacle[-1] = True
    obstacle[:, 0] = obstacle[:, -1] = True
    obstacle[:, :, 0] = True

    units = lattice_units(DX_M, WCB_NU, u_max_physical_m_s=2.8)
    fluid = np.zeros((n, n, n), dtype=bool)
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


if __name__ == "__main__":
    unittest.main()
