"""Unit tests for the D3Q19 lattice primitives and the filter closure.

No mesh, no geometry, no I/O -- everything here is a property of the lattice or
of a single cell, so these run in milliseconds and are the first thing to check
when the filling solver misbehaves.

The lattice-isotropy tests look like arithmetic trivia but are not: a single
wrong weight or a mismatched opposite index produces a scheme that runs, looks
stable, and silently solves the wrong equations.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from physics.fluid_flow import (
    CS2,
    ERGUN_PREFACTOR,
    FORCHHEIMER_WARD_CF,
    GRAVITY_M_S2,
    LATTICE_VELOCITIES,
    LATTICE_WEIGHTS,
    NOMINAL_FOAM_POROSITY,
    OPPOSITE,
    Q,
    TAU_MINUS_MAX,
    TRT_MAGIC_PARAMETER,
    bounce_back,
    capillary_entry_density_offset,
    capillary_entry_pressure,
    ceramic_foam_permeability,
    collide_trt,
    density,
    equilibrium,
    forchheimer_inverse_length,
    forchheimer_pressure_drop,
    guo_force_terms,
    guo_velocity,
    lattice_units,
    momentum,
    omega_minus_from_magic,
    porous_drag_coefficients,
    priming_head_m,
    resolve_porous_velocity,
    smagorinsky_tau,
    stream,
)

# WCB, the reference alloy: the one whose viscosity forces TRT.
WCB_RHO = 7200.0
WCB_MU = 0.006
WCB_NU = WCB_MU / WCB_RHO  # 8.33e-7 m^2/s


class LatticeStructureTests(unittest.TestCase):
    def test_nineteen_velocities_and_weights(self) -> None:
        self.assertEqual(LATTICE_VELOCITIES.shape, (Q, 3))
        self.assertEqual(LATTICE_WEIGHTS.shape, (Q,))
        self.assertEqual(OPPOSITE.shape, (Q,))

    def test_weights_sum_to_one(self) -> None:
        self.assertAlmostEqual(float(LATTICE_WEIGHTS.sum()), 1.0, places=15)

    def test_weight_classes_are_the_d3q19_values(self) -> None:
        self.assertAlmostEqual(LATTICE_WEIGHTS[0], 1.0 / 3.0)
        for i in range(1, 7):
            self.assertAlmostEqual(LATTICE_WEIGHTS[i], 1.0 / 18.0)
        for i in range(7, Q):
            self.assertAlmostEqual(LATTICE_WEIGHTS[i], 1.0 / 36.0)

    def test_velocity_classes_match_their_weights(self) -> None:
        """Speed-squared must partition as 0 / 1 / 2 in the same order."""
        speeds = np.sum(LATTICE_VELOCITIES.astype(int) ** 2, axis=1)
        self.assertEqual(speeds[0], 0)
        np.testing.assert_array_equal(speeds[1:7], np.ones(6, dtype=int))
        np.testing.assert_array_equal(speeds[7:], 2 * np.ones(12, dtype=int))

    def test_opposites_negate_the_velocity(self) -> None:
        c = LATTICE_VELOCITIES.astype(int)
        np.testing.assert_array_equal(c[OPPOSITE.astype(int)], -c)

    def test_opposite_is_an_involution(self) -> None:
        opp = OPPOSITE.astype(int)
        np.testing.assert_array_equal(opp[opp], np.arange(Q))

    def test_velocities_are_distinct(self) -> None:
        self.assertEqual(len({tuple(v) for v in LATTICE_VELOCITIES.tolist()}), Q)

    def test_first_moment_vanishes(self) -> None:
        first = (LATTICE_WEIGHTS[:, None] * LATTICE_VELOCITIES).sum(axis=0)
        np.testing.assert_allclose(first, np.zeros(3), atol=1e-15)

    def test_second_moment_is_isotropic(self) -> None:
        """sum_i w_i c_ia c_ib = c_s^2 delta_ab -- the lattice's whole point."""
        c = LATTICE_VELOCITIES.astype(float)
        second = np.einsum("i,ia,ib->ab", LATTICE_WEIGHTS, c, c)
        np.testing.assert_allclose(second, CS2 * np.eye(3), atol=1e-15)

    def test_third_moment_vanishes(self) -> None:
        c = LATTICE_VELOCITIES.astype(float)
        third = np.einsum("i,ia,ib,ic->abc", LATTICE_WEIGHTS, c, c, c)
        np.testing.assert_allclose(third, np.zeros((3, 3, 3)), atol=1e-15)

    def test_fourth_moment_has_the_required_isotropy(self) -> None:
        """
        sum w c_a c_b c_c c_d = c_s^4 (d_ab d_cd + d_ac d_bd + d_ad d_bc).
        This is what makes the recovered Navier-Stokes stress isotropic; D3Q19
        satisfies it and a mistyped weight would not.
        """
        c = LATTICE_VELOCITIES.astype(float)
        fourth = np.einsum("i,ia,ib,ic,id->abcd", LATTICE_WEIGHTS, c, c, c, c)
        d = np.eye(3)
        expected = CS2 ** 2 * (
            np.einsum("ab,cd->abcd", d, d)
            + np.einsum("ac,bd->abcd", d, d)
            + np.einsum("ad,bc->abcd", d, d)
        )
        np.testing.assert_allclose(fourth, expected, atol=1e-15)


class EquilibriumTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(20260830)
        self.rho = 1.0 + 0.05 * rng.standard_normal(64)
        self.u = 0.02 * rng.standard_normal((3, 64))

    def test_equilibrium_conserves_density(self) -> None:
        feq = equilibrium(self.rho, self.u)
        np.testing.assert_allclose(density(feq), self.rho, rtol=1e-12)

    def test_equilibrium_conserves_momentum(self) -> None:
        feq = equilibrium(self.rho, self.u)
        np.testing.assert_allclose(momentum(feq), self.rho * self.u, rtol=1e-11, atol=1e-15)

    def test_equilibrium_at_rest_is_the_weights(self) -> None:
        feq = equilibrium(np.array([1.0]), np.zeros((3, 1)))
        np.testing.assert_allclose(feq[:, 0], LATTICE_WEIGHTS, rtol=1e-15)

    def test_equilibrium_scales_with_density(self) -> None:
        a = equilibrium(np.array([1.0]), self.u[:, :1])
        b = equilibrium(np.array([2.0]), self.u[:, :1])
        np.testing.assert_allclose(2.0 * a, b, rtol=1e-14)

    def test_porosity_one_matches_the_standard_equilibrium(self) -> None:
        np.testing.assert_allclose(
            equilibrium(self.rho, self.u, porosity=1.0),
            equilibrium(self.rho, self.u),
            rtol=1e-15,
        )

    def test_porosity_below_one_changes_the_equilibrium(self) -> None:
        base = equilibrium(self.rho, self.u)
        porous = equilibrium(self.rho, self.u, porosity=0.85)
        self.assertFalse(np.allclose(base, porous))
        # It must still conserve both moments.
        np.testing.assert_allclose(density(porous), self.rho, rtol=1e-12)
        np.testing.assert_allclose(momentum(porous), self.rho * self.u, rtol=1e-11, atol=1e-15)


class TrtCollisionTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(4242)
        self.rho = 1.0 + 0.02 * rng.standard_normal(32)
        self.u = 0.01 * rng.standard_normal((3, 32))
        # Perturb first, then rebuild feq from the perturbed field's own
        # moments. In a real step feq is always computed from f, so
        # sum(feq) == sum(f) by construction; perturbing f while holding feq
        # fixed would ask collision to conserve a density it is relaxing.
        base = equilibrium(self.rho, self.u)
        self.f = base * (1.0 + 0.01 * rng.standard_normal(base.shape))
        rho_f = density(self.f)
        self.feq = equilibrium(rho_f, momentum(self.f) / rho_f)
        self.omega_plus = 1.0 / 0.5001
        self.omega_minus = omega_minus_from_magic(self.omega_plus)

    def test_collision_conserves_mass(self) -> None:
        out = collide_trt(self.f, self.feq, self.omega_plus, self.omega_minus)
        np.testing.assert_allclose(density(out), density(self.f), rtol=1e-12)

    def test_collision_conserves_momentum(self) -> None:
        out = collide_trt(self.f, self.feq, self.omega_plus, self.omega_minus)
        np.testing.assert_allclose(momentum(out), momentum(self.f), rtol=1e-10, atol=1e-16)

    def test_equilibrium_is_a_fixed_point(self) -> None:
        out = collide_trt(self.feq, self.feq, self.omega_plus, self.omega_minus)
        np.testing.assert_allclose(out, self.feq, rtol=1e-14)

    def test_collision_relaxes_towards_equilibrium(self) -> None:
        out = collide_trt(self.f, self.feq, self.omega_plus, self.omega_minus)
        self.assertLess(
            np.abs(out - self.feq).sum(), np.abs(self.f - self.feq).sum()
        )

    def test_equal_rates_reproduce_bgk(self) -> None:
        """With omega+ == omega- TRT degenerates to BGK exactly."""
        omega = 1.2
        trt = collide_trt(self.f, self.feq, omega, omega)
        bgk = self.f - omega * (self.f - self.feq)
        np.testing.assert_allclose(trt, bgk, rtol=1e-13)

    def test_source_term_is_added(self) -> None:
        source = np.full_like(self.f, 1e-6)
        without = collide_trt(self.f, self.feq, self.omega_plus, self.omega_minus)
        with_source = collide_trt(
            self.f, self.feq, self.omega_plus, self.omega_minus, source=source
        )
        # Differencing two O(0.03) fields to recover a 1e-6 term costs about
        # five digits to cancellation, so rtol here is bounded by that, not by
        # anything the collision does.
        np.testing.assert_allclose(with_source - without, source, rtol=1e-8)


class MagicParameterTests(unittest.TestCase):
    def test_magic_parameter_round_trips_where_the_clamp_does_not_bind(self) -> None:
        """
        ``Lambda`` is recovered exactly above ``tau = 0.541667``.

        The tau list lost 0.5001 and 0.51 when ``TAU_MINUS_MAX`` arrived: below
        0.541667 the clamp caps ``tau_minus`` at 5 and ``Lambda`` is
        deliberately no longer 3/16. That is the whole point of the clamp and it
        is asserted by :meth:`test_the_clamp_binds_at_a_casting_tau`; asserting
        the round trip there would be asserting the mode back into existence.
        The exact relation is what the boundary-accuracy benchmarks depend on,
        and they all run above the threshold.
        """
        threshold = TRT_MAGIC_PARAMETER / (TAU_MINUS_MAX - 0.5) + 0.5
        for tau in (0.5417, 0.55, 0.8, 1.0, 1.9):
            with self.subTest(tau=tau):
                self.assertGreater(tau, threshold, "tau is inside the clamp")
                omega_plus = 1.0 / tau
                omega_minus = omega_minus_from_magic(omega_plus)
                recovered = (1.0 / omega_plus - 0.5) * (1.0 / omega_minus - 0.5)
                self.assertAlmostEqual(recovered, TRT_MAGIC_PARAMETER, places=12)

    def test_magic_parameter_is_three_sixteenths(self) -> None:
        self.assertAlmostEqual(TRT_MAGIC_PARAMETER, 0.1875, places=15)

    def test_the_clamp_binds_at_a_casting_tau(self) -> None:
        """
        At ``tau_plus = 0.500010`` -- ``Test 2``'s whole domain --
        ``tau_minus`` is 5, not 18,060.

        Unclamped, the odd momentum-carrying moments relax over 18,060 steps,
        which is no relaxation at all over the 200 steps a mode needs to double.
        A period-2 checkerboard on one lattice diagonal grew undamped for
        ~5,000 steps in the runner cells outside a filter inlet and took the
        run non-finite at step 20,192. See
        :data:`~physics.fluid_flow.TAU_MINUS_MAX`.
        """
        omega_plus = 1.0 / 0.500010
        clamped = omega_minus_from_magic(omega_plus)
        unclamped = omega_minus_from_magic(
            omega_plus, tau_minus_max=float("inf")
        )
        self.assertAlmostEqual(1.0 / clamped, TAU_MINUS_MAX, places=12)
        self.assertAlmostEqual(clamped, 1.0 / TAU_MINUS_MAX, places=15)
        self.assertGreater(
            1.0 / unclamped, 18_000.0,
            "the unclamped relation no longer freezes the odd moments, so this "
            "test no longer measures anything",
        )

    def test_the_clamp_does_not_bind_at_the_benchmark_tau(self) -> None:
        """
        At ``tau = 0.6`` the clamp is inert and ``Lambda`` is exactly 3/16.

        This is the property that lets the wall-location benchmarks keep
        testing the magic parameter: ``tests/test_filling_benchmarks.py`` runs
        its Poiseuille profile at ``tau = 0.8``, and the threshold is 0.541667.
        """
        for tau in (0.6, 0.8):
            with self.subTest(tau=tau):
                omega_plus = 1.0 / tau
                clamped = omega_minus_from_magic(omega_plus)
                unclamped = omega_minus_from_magic(
                    omega_plus, tau_minus_max=float("inf")
                )
                self.assertEqual(clamped, unclamped)
                self.assertLess(1.0 / clamped, TAU_MINUS_MAX)
                recovered = (tau - 0.5) * (1.0 / clamped - 0.5)
                self.assertAlmostEqual(recovered, TRT_MAGIC_PARAMETER, places=12)

    def test_the_clamp_threshold_is_where_it_is_documented(self) -> None:
        """The boundary itself, so the docstring's 0.541667 cannot go stale."""
        threshold = TRT_MAGIC_PARAMETER / (TAU_MINUS_MAX - 0.5) + 0.5
        self.assertAlmostEqual(threshold, 0.5416666666666666, places=15)
        just_below = omega_minus_from_magic(1.0 / (threshold - 1e-6))
        just_above = omega_minus_from_magic(1.0 / (threshold + 1e-6))
        self.assertAlmostEqual(1.0 / just_below, TAU_MINUS_MAX, places=12)
        self.assertLess(1.0 / just_above, TAU_MINUS_MAX)

    def test_omega_minus_stays_in_the_stable_range(self) -> None:
        """
        omega- must remain in (0, 2) even as tau approaches 1/2.

        The clamp makes the lower end trivial -- omega- cannot fall below
        ``1 / TAU_MINUS_MAX`` -- so the floor is asserted explicitly rather
        than left as "greater than zero", which is now unfalsifiable.
        """
        for tau in (0.500001, 0.5001, 0.51, 1.0, 1.99):
            with self.subTest(tau=tau):
                om = omega_minus_from_magic(1.0 / tau)
                self.assertGreaterEqual(om, 1.0 / TAU_MINUS_MAX)
                self.assertLess(om, 2.0)

    def test_lattice_units_carries_the_clamp(self) -> None:
        """
        The clamp reaches the solver through ``LatticeUnits.omega_minus``, which
        is what the non-Smagorinsky path relaxes with, so it is asserted at that
        seam and not only on the helper.
        """
        wcb_nu = 0.006 / 7200.0
        lu = lattice_units(0.0043, wcb_nu, u_max_physical_m_s=2.8)
        raw = lattice_units(
            0.0043, wcb_nu, u_max_physical_m_s=2.8, tau_minus_max=float("inf")
        )
        self.assertLess(lu.tau, 0.5416666666666666)
        self.assertAlmostEqual(1.0 / lu.omega_minus, TAU_MINUS_MAX, places=12)
        self.assertGreater(1.0 / raw.omega_minus, 18_000.0)
        self.assertEqual(lu.tau, raw.tau)
        self.assertEqual(lu.omega_plus, raw.omega_plus)

    def test_invalid_omega_plus_raises(self) -> None:
        for bad in (0.0, -1.0, 2.0, 2.5):
            with self.subTest(omega_plus=bad):
                with self.assertRaises(ValueError):
                    omega_minus_from_magic(bad)


class LatticeUnitTests(unittest.TestCase):
    """The conversion happens in one place, so it gets tested in one place."""

    def test_documented_configurations_reproduce_their_tau(self) -> None:
        """
        The design doc tabulates tau for four real configurations. They all sit
        within 1e-4 of the 1/2 stability limit -- which is the entire argument
        for TRT, so it is worth pinning numerically.
        """
        cases = [
            (0.00254, 0.05, 127e-6, 0.500049),   # 0.1 in, u_lat 0.05
            (0.00254, 0.10, 254e-6, 0.500098),   # 0.1 in, u_lat 0.10
            (0.00508, 0.05, 254e-6, 0.500025),   # 0.2 in, u_lat 0.05
        ]
        for dx, u_lat, expected_dt, expected_tau in cases:
            with self.subTest(dx=dx, u_lat=u_lat):
                lu = lattice_units(
                    dx, WCB_NU, u_max_physical_m_s=1.0, u_max_lattice=u_lat
                )
                self.assertAlmostEqual(lu.dt_s, expected_dt, delta=1e-6)
                self.assertAlmostEqual(lu.tau, expected_tau, places=5)

    def test_tau_is_always_above_one_half(self) -> None:
        for dx in (0.00254, 0.00508, 0.00635):
            for u_phys in (0.5, 1.0, 3.0):
                with self.subTest(dx=dx, u_phys=u_phys):
                    lu = lattice_units(dx, WCB_NU, u_max_physical_m_s=u_phys)
                    self.assertGreater(lu.tau, 0.5)

    def test_shrinking_dt_pushes_tau_towards_the_limit(self) -> None:
        """
        The doc's warning, as an executable statement: halving dt (by halving
        u_max_lattice) halves nu_lattice and moves tau *closer* to 1/2. Anyone
        tempted to stabilise the solver that way should see this fail their
        intuition, not their simulation.
        """
        coarse = lattice_units(0.00254, WCB_NU, u_max_physical_m_s=1.0, u_max_lattice=0.10)
        fine = lattice_units(0.00254, WCB_NU, u_max_physical_m_s=1.0, u_max_lattice=0.05)
        self.assertLess(fine.dt_s, coarse.dt_s)
        self.assertLess(fine.tau - 0.5, coarse.tau - 0.5)

    def test_conversion_relations_hold(self) -> None:
        dx, u_phys, u_lat = 0.00254, 1.5, 0.04
        lu = lattice_units(dx, WCB_NU, u_max_physical_m_s=u_phys, u_max_lattice=u_lat)
        self.assertAlmostEqual(lu.dt_s, u_lat * dx / u_phys, places=15)
        self.assertAlmostEqual(lu.nu_lattice, WCB_NU * lu.dt_s / dx ** 2, places=18)
        self.assertAlmostEqual(lu.tau, 3.0 * lu.nu_lattice + 0.5, places=15)
        self.assertAlmostEqual(
            lu.gravity_lattice, GRAVITY_M_S2 * lu.dt_s ** 2 / dx, places=18
        )

    def test_velocity_round_trip(self) -> None:
        lu = lattice_units(0.00254, WCB_NU, u_max_physical_m_s=1.0)
        u_phys = np.array([0.3, -0.1, 0.7])
        np.testing.assert_allclose(
            lu.to_physical_velocity(lu.to_lattice_velocity(u_phys)), u_phys, rtol=1e-14
        )

    def test_max_physical_velocity_maps_to_the_mach_guard(self) -> None:
        lu = lattice_units(0.00254, WCB_NU, u_max_physical_m_s=2.0, u_max_lattice=0.05)
        self.assertAlmostEqual(
            float(lu.to_lattice_velocity(np.array([2.0]))[0]), 0.05, places=15
        )

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            lattice_units(0.0, WCB_NU, u_max_physical_m_s=1.0)
        with self.assertRaises(ValueError):
            lattice_units(0.00254, 0.0, u_max_physical_m_s=1.0)
        with self.assertRaises(ValueError):
            lattice_units(0.00254, WCB_NU, u_max_physical_m_s=0.0)
        with self.assertRaises(ValueError):
            lattice_units(0.00254, WCB_NU, u_max_physical_m_s=1.0, u_max_lattice=0.6)


class ForcingTests(unittest.TestCase):
    def test_guo_velocity_includes_the_half_force(self) -> None:
        rho = np.array([1.0])
        f = equilibrium(rho, np.zeros((3, 1)))
        force = np.array([[0.001], [0.0], [0.0]])
        u = guo_velocity(f, rho, force)
        self.assertAlmostEqual(float(u[0, 0]), 0.0005, places=15)

    def test_forcing_injects_the_discounted_momentum(self) -> None:
        """
        The Guo source contributes ``(1 - 1/(2 tau)) F`` to the momentum, not
        ``F``. The missing half is supplied by the ``+F/2`` term in the velocity
        definition (:func:`guo_velocity`); together they give exactly F per step.

        Pinning the discounted form matters because dropping the prefactor
        still "works" -- the flow accelerates, nothing errors -- and simply
        applies gravity wrong by a tau-dependent factor. At tau = 0.6 that
        factor is 1/6.
        """
        u = np.zeros((3, 1))
        force = np.array([[0.0], [0.0], [-1e-4]])
        for tau in (0.5001, 0.6, 1.0, 1.8):
            with self.subTest(tau=tau):
                omega_minus = omega_minus_from_magic(1.0 / tau)
                added = momentum(guo_force_terms(u, force, omega_minus))
                expected = (1.0 - 0.5 * omega_minus) * force
                np.testing.assert_allclose(added, expected, rtol=1e-12, atol=1e-20)

    def test_collision_injects_exactly_the_body_force(self) -> None:
        """
        The identity that makes hydrostatic equilibrium reachable at all:

            omega_minus * F/2  +  (1 - omega_minus/2) * F  ==  F

        The first term is the antisymmetric relaxation acting on the +F/2 that
        guo_velocity folded into the equilibrium; the second is the explicit
        source. They must sum to exactly F for *any* pair of relaxation rates.

        Using the BGK prefactor ``1 - 1/(2 tau)`` here instead satisfies this
        only when omega_minus == 1/tau. Under TRT at casting viscosities
        omega_minus is 0.2 -- the ``TAU_MINUS_MAX`` floor -- while 1/tau is
        ~2.0, so gravity would be applied at a tenth of its true strength,
        silently, with no instability to give it away. Before the clamp the same
        mismatch was 4.84e-5 on the lattice these numbers are quoted for
        (tau = 0.500010, omega_minus 5.54e-5), which is how large the
        discrepancy this test guards can get. The identity is what is asserted,
        not the magnitude -- it holds for any pair of rates, which is the point.
        """
        u = np.zeros((3, 1))
        force = np.array([[0.0], [0.0], [-1e-4]])
        for tau in (0.500049, 0.5001, 0.6, 1.0, 1.8):
            with self.subTest(tau=tau):
                omega_minus = omega_minus_from_magic(1.0 / tau)
                injected = momentum(guo_force_terms(u, force, omega_minus))
                relaxed = 0.5 * omega_minus * force
                np.testing.assert_allclose(
                    injected + relaxed, force, rtol=1e-12, atol=1e-20
                )

    def test_forcing_adds_no_mass(self) -> None:
        rng = np.random.default_rng(7)
        u = 0.01 * rng.standard_normal((3, 16))
        force = 1e-5 * rng.standard_normal((3, 16))
        source = guo_force_terms(u, force, omega_minus_from_magic(1.0 / 0.55))
        np.testing.assert_allclose(density(source), np.zeros(16), atol=1e-18)


class StreamingTests(unittest.TestCase):
    def test_streaming_conserves_mass(self) -> None:
        rng = np.random.default_rng(11)
        f = rng.random((Q, 6, 5, 4))
        self.assertAlmostEqual(float(stream(f).sum()), float(f.sum()), places=10)

    def test_streaming_moves_each_population_by_its_velocity(self) -> None:
        f = np.zeros((Q, 5, 5, 5))
        f[:, 2, 2, 2] = 1.0
        out = stream(f)
        for i in range(Q):
            c = LATTICE_VELOCITIES[i]
            with self.subTest(direction=i):
                self.assertAlmostEqual(
                    out[i, 2 + int(c[0]), 2 + int(c[1]), 2 + int(c[2])], 1.0
                )

    def test_streaming_then_reverse_streaming_is_identity(self) -> None:
        rng = np.random.default_rng(13)
        f = rng.random((Q, 6, 6, 6))
        opp = OPPOSITE.astype(int)
        # Streaming the opposite-permuted field undoes the shift.
        once = stream(f)
        back = stream(once[opp])[opp]
        np.testing.assert_allclose(back, f, rtol=1e-14)


class BounceBackTests(unittest.TestCase):
    def test_no_obstacle_is_a_no_op(self) -> None:
        rng = np.random.default_rng(3)
        f = rng.random((Q, 4, 4, 4))
        np.testing.assert_array_equal(bounce_back(f, np.zeros((4, 4, 4), bool)), f)

    def test_obstacle_cells_are_reflected(self) -> None:
        f = np.zeros((Q, 3, 3, 3))
        f[1, 1, 1, 1] = 1.0  # +X population on the obstacle cell
        obstacle = np.zeros((3, 3, 3), bool)
        obstacle[1, 1, 1] = True
        out = bounce_back(f, obstacle)
        self.assertAlmostEqual(out[int(OPPOSITE[1]), 1, 1, 1], 1.0)
        self.assertAlmostEqual(out[1, 1, 1, 1], 0.0)

    def test_bounce_back_conserves_mass(self) -> None:
        rng = np.random.default_rng(5)
        f = rng.random((Q, 4, 4, 4))
        obstacle = rng.random((4, 4, 4)) < 0.3
        self.assertAlmostEqual(float(bounce_back(f, obstacle).sum()), float(f.sum()), places=10)

    def test_bounce_back_is_an_involution_on_obstacles(self) -> None:
        rng = np.random.default_rng(9)
        f = rng.random((Q, 4, 4, 4))
        obstacle = rng.random((4, 4, 4)) < 0.4
        np.testing.assert_allclose(bounce_back(bounce_back(f, obstacle), obstacle), f)


class FilterClosureTests(unittest.TestCase):
    """The ceramic filter closure, and the arithmetic that justifies it."""

    LITERATURE_K1 = {30: 6.29e-8, 50: 2.0e-8, 80: 8.0e-9}

    @staticmethod
    def _cell_diameter_m(ppi: int) -> float:
        return 0.0254 / ppi

    def test_ergun_ratios_are_constant_across_pore_count(self) -> None:
        """
        The real justification for a one-parameter fit: uncalibrated Ergun is
        off by a nearly constant factor across a 2.7x span in cell diameter, so
        the d^2 scaling transfers and only the prefactor is wrong.
        """
        ratios = [
            ceramic_foam_permeability(
                NOMINAL_FOAM_POROSITY, self._cell_diameter_m(ppi), ergun_prefactor=1.0
            )
            / k
            for ppi, k in self.LITERATURE_K1.items()
        ]
        spread = (max(ratios) - min(ratios)) / min(ratios)
        self.assertLess(spread, 0.15, f"Ergun ratios not constant: {ratios}")
        self.assertAlmostEqual(float(np.mean(ratios)), ERGUN_PREFACTOR, delta=0.15)

    def test_fitted_permeability_matches_literature_within_supplier_scatter(self) -> None:
        """
        3x, not 2x: the literature itself documents ~2x supplier-to-supplier
        scatter at identical nominal ppi, so a tighter bound would assert more
        than the source supports.
        """
        for ppi, k_lit in self.LITERATURE_K1.items():
            with self.subTest(ppi=ppi):
                k = ceramic_foam_permeability(
                    NOMINAL_FOAM_POROSITY, self._cell_diameter_m(ppi)
                )
                self.assertGreater(k, k_lit / 3.0)
                self.assertLess(k, k_lit * 3.0)

    def test_two_independent_routes_agree_at_ten_ppi(self) -> None:
        """
        No 10 ppi permeability is published anywhere, and 10 ppi is the grade
        used for steel. Ergun with the fitted prefactor and a 1/ppi^2
        extrapolation from the 30 ppi anchor are unrelated routes; agreeing to
        under 15 % is the only evidence available that the 10 ppi value is sane.
        """
        ergun = ceramic_foam_permeability(NOMINAL_FOAM_POROSITY, self._cell_diameter_m(10))
        scaled = self.LITERATURE_K1[30] * (30.0 / 10.0) ** 2
        self.assertLess(abs(ergun - scaled) / scaled, 0.15)

    def test_permeability_falls_with_pore_count(self) -> None:
        previous = float("inf")
        for ppi in (10, 30, 50, 80):
            k = ceramic_foam_permeability(NOMINAL_FOAM_POROSITY, self._cell_diameter_m(ppi))
            with self.subTest(ppi=ppi):
                self.assertLess(k, previous)
            previous = k

    def test_forchheimer_pairs_with_its_own_permeability(self) -> None:
        k1 = ceramic_foam_permeability(NOMINAL_FOAM_POROSITY, 2.54e-3)
        inv_k2 = forchheimer_inverse_length(k1)
        self.assertAlmostEqual(inv_k2, FORCHHEIMER_WARD_CF / math.sqrt(k1), places=9)
        self.assertAlmostEqual(inv_k2, 752.9, delta=5.0)

    def test_inertial_term_dominates_at_foundry_velocity(self) -> None:
        """
        The reason the inertial coefficient is the one that must be right: at
        the rated face velocity the Darcy term carries under 2 % of the drop.
        Calibrating k1 while leaving k2 at a packed-bed value gets this exactly
        backwards.
        """
        k1 = ceramic_foam_permeability(NOMINAL_FOAM_POROSITY, 2.54e-3)
        inv_k2 = forchheimer_inverse_length(k1)
        u = 0.1389  # carbon/lip rated superficial velocity at rho = 7200
        viscous = WCB_MU / k1 * u
        inertial = WCB_RHO * inv_k2 * u * u
        self.assertLess(viscous / (viscous + inertial), 0.02)

    def test_pressure_drop_at_the_rated_velocity_is_one_to_two_inches(self) -> None:
        """
        Validation, not calibration. Nothing here was fitted to reproduce this
        number -- k1 comes from Ergun-on-alumina and 1/k2 from Ward -- so
        landing on the 1.5 in of metal that foundry practice quotes is real
        corroboration. Fitting the coefficients to hit it would instead make the
        filter the flow-controlling restriction by construction.
        """
        k1 = ceramic_foam_permeability(NOMINAL_FOAM_POROSITY, 2.54e-3)
        dp = forchheimer_pressure_drop(
            0.1389,
            0.025,
            k1_m2=k1,
            inv_k2_per_m=forchheimer_inverse_length(k1),
            dynamic_viscosity_pa_s=WCB_MU,
            density_kg_m3=WCB_RHO,
        )
        head_in = dp / (WCB_RHO * GRAVITY_M_S2) / 0.0254
        self.assertGreater(head_in, 1.3)
        self.assertLess(head_in, 1.6)

    def test_pressure_drop_grows_faster_than_linearly(self) -> None:
        """
        Quadratic dominance is what lets the solver *discover* an undersized
        filter. A flux cap would clamp the velocity, make the run look healthy,
        and destroy the diagnostic.
        """
        k1 = ceramic_foam_permeability(NOMINAL_FOAM_POROSITY, 2.54e-3)
        kwargs = dict(
            k1_m2=k1,
            inv_k2_per_m=forchheimer_inverse_length(k1),
            dynamic_viscosity_pa_s=WCB_MU,
            density_kg_m3=WCB_RHO,
        )
        single = forchheimer_pressure_drop(0.14, 0.025, **kwargs)
        double = forchheimer_pressure_drop(0.28, 0.025, **kwargs)
        self.assertGreater(double / single, 3.5)

    def test_invalid_filter_inputs_raise(self) -> None:
        for porosity in (0.0, 1.0, -0.1, 1.5):
            with self.subTest(porosity=porosity):
                with self.assertRaises(ValueError):
                    ceramic_foam_permeability(porosity, 2.54e-3)
        with self.assertRaises(ValueError):
            ceramic_foam_permeability(0.85, 0.0)
        with self.assertRaises(ValueError):
            forchheimer_inverse_length(0.0)


class PorousDragTests(unittest.TestCase):
    def test_clear_cells_degenerate_to_no_drag(self) -> None:
        c0, c1 = porous_drag_coefficients(
            1.0, float("inf"), 0.0, nu_lattice=1e-5, dx_m=0.00254
        )
        self.assertEqual(c0, 0.5)
        self.assertEqual(c1, 0.0)

    def test_clear_cells_leave_the_velocity_exactly_unchanged(self) -> None:
        """
        u = v / (0.5 + sqrt(0.25 + 0)) = v / 1 = v, bit-exact. This is what
        makes a no-filter build identical to a filter build with zero drag, and
        keeps a single branch-free kernel honest.
        """
        rng = np.random.default_rng(17)
        v = 0.05 * rng.standard_normal((3, 40))
        u = resolve_porous_velocity(v, np.float64(0.5), np.float64(0.0))
        np.testing.assert_array_equal(u, v)

    def test_drag_never_reverses_or_amplifies_the_velocity(self) -> None:
        """
        A sign slip here is catastrophic and otherwise invisible. Swept over
        coefficients well past anything physical, including a very stiff c1.
        """
        rng = np.random.default_rng(23)
        v = rng.standard_normal((3, 200))
        for c0 in (0.5, 0.6, 1.0, 5.0):
            for c1 in (0.0, 0.1, 2.0, 100.0):
                with self.subTest(c0=c0, c1=c1):
                    u = resolve_porous_velocity(v, np.float64(c0), np.float64(c1))
                    speed_v = np.linalg.norm(v, axis=0)
                    speed_u = np.linalg.norm(u, axis=0)
                    self.assertTrue(np.all(speed_u <= speed_v + 1e-12))
                    dots = np.einsum("i...,i...->...", u, v)
                    self.assertTrue(np.all(dots >= -1e-15))

    def test_stronger_drag_slows_the_flow_more(self) -> None:
        v = np.array([[0.05], [0.0], [0.0]])
        weak = resolve_porous_velocity(v, np.float64(0.5), np.float64(0.5))
        strong = resolve_porous_velocity(v, np.float64(0.5), np.float64(5.0))
        self.assertLess(abs(strong[0, 0]), abs(weak[0, 0]))

    def test_filter_cells_produce_real_drag(self) -> None:
        k1 = ceramic_foam_permeability(NOMINAL_FOAM_POROSITY, 2.54e-3)
        c0, c1 = porous_drag_coefficients(
            NOMINAL_FOAM_POROSITY,
            k1,
            forchheimer_inverse_length(k1),
            nu_lattice=1.64e-5,
            dx_m=0.00254,
        )
        self.assertGreaterEqual(c0, 0.5)
        self.assertGreater(c1, 0.0)

    def test_invalid_drag_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            porous_drag_coefficients(0.0, 1e-7, 100.0, nu_lattice=1e-5, dx_m=0.00254)
        with self.assertRaises(ValueError):
            porous_drag_coefficients(0.85, 1e-7, 100.0, nu_lattice=1e-5, dx_m=0.0)


class CapillaryPrimingTests(unittest.TestCase):
    SIGMA = 1.8
    THETA = 120.0
    D_CELL = 2.54e-3
    WINDOW = 0.45

    def _entry_pressure(self) -> float:
        return capillary_entry_pressure(self.SIGMA, self.THETA, self.D_CELL, self.WINDOW)

    def test_entry_pressure_matches_the_window_formula(self) -> None:
        self.assertAlmostEqual(self._entry_pressure(), 3149.6, delta=5.0)

    def test_priming_head_is_in_the_practice_band(self) -> None:
        """
        Foundry practice quotes 50-75 mm of head to prime a 10 ppi steel filter.
        Asserted as a band rather than a point: three uncertain inputs multiply,
        so a point assertion would be false precision.
        """
        head = priming_head_m(self._entry_pressure(), WCB_RHO)
        self.assertGreater(head, 0.040)
        self.assertLess(head, 0.080)

    def test_using_the_cell_radius_instead_of_the_window_halves_the_head(self) -> None:
        """
        Guards the choice of length scale. The full-pore form lands at roughly
        half the practice band, which is how you tell the two apart.
        """
        window_head = priming_head_m(self._entry_pressure(), WCB_RHO)
        full_pore = capillary_entry_pressure(self.SIGMA, self.THETA, self.D_CELL, 1.0)
        full_head = priming_head_m(full_pore, WCB_RHO)
        self.assertAlmostEqual(full_head / window_head, self.WINDOW, places=9)
        self.assertLess(full_head, 0.040)

    def test_lattice_density_offset_matches_the_head_identity(self) -> None:
        """
        d_rho == (h_prime / dx) * g_lattice / c_s^2.

        The two expressions are algebraically identical but reach the answer by
        different routes, so this single assertion catches every unit error on
        the capillary path -- a term wrong by a factor of dx is invisible in the
        output and merely makes filters prime at the wrong depth.
        """
        lu = lattice_units(self.D_CELL, WCB_NU, u_max_physical_m_s=3.0)
        dp = self._entry_pressure()
        direct = capillary_entry_density_offset(
            dp, density_kg_m3=WCB_RHO, dt_s=lu.dt_s, dx_m=lu.dx_m
        )
        via_head = (
            priming_head_m(dp, WCB_RHO) / lu.dx_m
        ) * lu.gravity_lattice / CS2
        self.assertAlmostEqual(direct, via_head, delta=abs(direct) * 1e-12)

    def test_density_offset_has_float32_headroom(self) -> None:
        """
        The offset is applied to a float32 density of order 1. It must be far
        enough above the float32 epsilon to survive, or density-based priming
        would silently do nothing.
        """
        lu = lattice_units(self.D_CELL, WCB_NU, u_max_physical_m_s=3.0)
        offset = capillary_entry_density_offset(
            self._entry_pressure(), density_kg_m3=WCB_RHO, dt_s=lu.dt_s, dx_m=lu.dx_m
        )
        self.assertGreater(offset, 1000.0 * np.finfo(np.float32).eps)

    def test_a_wetting_liquid_needs_no_priming_head(self) -> None:
        self.assertAlmostEqual(
            capillary_entry_pressure(self.SIGMA, 90.0, self.D_CELL, self.WINDOW), 0.0
        )

    def test_invalid_capillary_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            capillary_entry_pressure(0.0, 120.0, 2.54e-3, 0.45)
        with self.assertRaises(ValueError):
            capillary_entry_pressure(1.8, 120.0, 2.54e-3, 0.0)
        with self.assertRaises(ValueError):
            capillary_entry_density_offset(100.0, density_kg_m3=0.0, dt_s=1e-4, dx_m=1e-3)


class SmagorinskyTests(unittest.TestCase):
    def test_zero_strain_leaves_tau_unchanged(self) -> None:
        self.assertAlmostEqual(smagorinsky_tau(0.5001, 0.0), 0.5001, places=15)

    def test_strain_raises_tau(self) -> None:
        self.assertGreater(smagorinsky_tau(0.5001, 1.0), 0.5001)

    def test_eddy_viscosity_lifts_tau_off_the_stability_limit(self) -> None:
        """
        Helpful but not sufficient: even a large strain rate only lifts tau to
        ~0.5-0.6, which is why TRT is not optional.
        """
        lifted = smagorinsky_tau(0.500049, 1.0)
        self.assertGreater(lifted - 0.5, 10.0 * (0.500049 - 0.5))
        self.assertLess(lifted, 0.7)


if __name__ == "__main__":
    unittest.main()
