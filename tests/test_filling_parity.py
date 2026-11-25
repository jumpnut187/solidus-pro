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
    rng = np.random.default_rng(seed)
    state.f = (state.f * (1.0 + 0.01 * rng.standard_normal(state.f.shape))).astype(
        np.float32
    )
    state.f[:, obstacle] = 0.0
    state.f[:, state.flags == FLAG_GAS] = 0.0
    return state, units

def _twin(**kwargs):
    a, units = _fixture(**kwargs)
    b, _ = _fixture(**kwargs)
    np.testing.assert_array_equal(a.f, b.f)
    return a, b, units

@unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
class SingleStepParityTests(unittest.TestCase):

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
        np.testing.assert_array_equal(a.rho, b.rho)

    def test_velocity_agrees(self) -> None:
        a, b, units = _twin()
        step(a, units, use_numba=True)
        step(b, units, use_numba=False)
        np.testing.assert_allclose(a.u, b.u, rtol=1e-4, atol=1e-9)
        np.testing.assert_array_equal(a.u, b.u)

    def test_mass_agrees(self) -> None:
        a, b, units = _twin()
        step(a, units, use_numba=True)
        step(b, units, use_numba=False)
        np.testing.assert_allclose(a.mass, b.mass, rtol=2e-6, atol=2e-7)

    def test_flags_agree_exactly(self) -> None:
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
        a, b, units = _twin()
        step(a, units, use_numba=True, smagorinsky=False)
        step(b, units, use_numba=False, smagorinsky=False)
        np.testing.assert_allclose(a.f, b.f, rtol=1e-5, atol=1e-7)

@unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
class PorousParityTests(unittest.TestCase):

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
        state, _ = _fixture(with_filter=True)
        self.assertFalse(state.porous.is_inert)
        self.assertGreater(float(state.porous.drag_c1.max()), 0.0)

    def test_the_filter_is_actually_wet_when_submerged(self) -> None:
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
        a, b, units = _twin(with_filter=True, submerge=True)
        for _ in range(30):
            step(a, units, use_numba=True)
            step(b, units, use_numba=False)
        np.testing.assert_allclose(a.mass, b.mass, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(a.flags, b.flags)
        self.assertAlmostEqual(total_mass(a), total_mass(b), places=9)

@unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
class MultiStepParityTests(unittest.TestCase):

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

    def test_the_numba_path_is_deterministic(self) -> None:
        fields = ("f", "rho", "u", "mass", "flags", "temperature", "fill_time_s")

        def run(fixture):
            state, units = fixture()
            for _ in range(12):
                step(state, units, use_numba=True)
                advect_temperature(state, 0.03)
            return {n: np.asarray(getattr(state, n)).tobytes() for n in fields}

        for label, fixture in (
            ("quiescent", _fixture),
            ("firing", _provoked),
            ("wet filter", lambda: _fixture(with_filter=True, submerge=True)),
        ):
            with self.subTest(fixture=label):
                first = run(fixture)
                for attempt in range(2):
                    again = run(fixture)
                    for name in fields:
                        self.assertEqual(
                            first[name], again[name],
                            f"{label}: {name} differed between identical runs "
                            f"on attempt {attempt + 1} -- a kernel is racing",
                        )

    def test_the_population_buffers_are_float32_and_mass_is_not(self) -> None:
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
        self.assertEqual(state._scratch_rho64.dtype, np.float64)
        self.assertEqual(state._scratch_u64.dtype, np.float64)

        self.assertFalse(hasattr(state, "_scratch_f_out"))

    def test_the_numpy_reference_path_still_computes_in_float64(self) -> None:
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
        state, units = _fixture()
        step(state, units, use_numba=False)
        self.assertTrue(bool(np.all(np.isfinite(state.f))))

class SmagorinskyConstantTests(unittest.TestCase):

    CS_LOW = 0.05
    CS_HIGH = 0.40

    def _stepped(self, cs: float, *, use_numba: bool) -> np.ndarray:
        state, units = _fixture()
        step(state, units, use_numba=use_numba, smagorinsky_cs=cs)
        return state.f.astype(np.float64)

    def test_the_constant_changes_the_numpy_result(self) -> None:
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
        np.testing.assert_allclose(
            self._stepped(self.CS_HIGH, use_numba=True),
            self._stepped(self.CS_HIGH, use_numba=False),
            rtol=1e-5,
            atol=1e-7,
        )

    def test_zero_degrades_to_plain_trt(self) -> None:
        zero, units = _fixture()
        step(zero, units, use_numba=False, smagorinsky_cs=0.0)
        disabled, units = _fixture()
        step(disabled, units, use_numba=False, smagorinsky=False)
        np.testing.assert_allclose(zero.f, disabled.f, rtol=1e-6, atol=1e-9)

TEMPERATURE_MIX_SEED = 4103

def _mixed_state(n: int = 14, *, steps: int = 8, seed: int = TEMPERATURE_MIX_SEED):
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
    gas = state.flags == FLAG_GAS
    wet = (state.flags == FLAG_FLUID) | (state.flags == FLAG_INTERFACE)

    buried_mask = (state.flags == FLAG_FLUID) & ~_dilate(gas)
    buried = tuple(int(v) for v in np.argwhere(buried_mask)[0])
    state.flags[buried] = FLAG_INTERFACE
    state.mass[buried] = 0.42

    droplet_mask = gas & ~_dilate(wet)
    droplet = tuple(int(v) for v in np.argwhere(droplet_mask)[0])
    state.flags[droplet] = FLAG_INTERFACE
    state.mass[droplet] = 0.31
    state.rho[droplet] = np.float32(1.0)
    state.temperature[droplet] = np.float32(1380.0)
    return buried, droplet

def _mixed_twin(*, plant: bool = False, **kwargs):
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
        a, b, _ = _mixed_twin()
        rng = np.random.default_rng(TEMPERATURE_MIX_SEED + 1)
        htc = 500.0 + 2000.0 * rng.random(a.shape)
        fast = self._wall_loss(a, htc, use_numba=True)
        slow = self._wall_loss(b, htc, use_numba=False)
        np.testing.assert_array_equal(fast, slow)

    def test_wall_loss_actually_changes_something(self) -> None:
        a, _, _ = _mixed_twin()
        before = a.temperature.copy()
        self._wall_loss(a, 1500.0, use_numba=True)
        changed = np.isfinite(before) & (a.temperature != before)
        self.assertGreater(int(np.count_nonzero(changed)), 0)

    def test_the_cached_face_count_matches_a_fresh_one(self) -> None:
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

    @staticmethod
    def _reference_masks(state):
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

    RANDOM_MIXTURES = (
        ("gas heavy", (0.10, 0.85, 0.04, 0.01), "detached"),
        ("wet heavy", (0.10, 0.02, 0.50, 0.38), "gasless"),
    )

    def test_masks_agree_on_randomised_flags(self) -> None:
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
                fired = ref_detached if expected == "detached" else ref_gasless
                self.assertGreater(int(np.count_nonzero(fired)), 0)

    def test_a_buried_interface_cell_is_gasless_and_supported(self) -> None:
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
        a, b, _ = _mixed_twin(plant=True)
        healed_fast = heal_submerged_interface(a, 0.5, 1, use_numba=True)
        healed_slow = heal_submerged_interface(b, 0.5, 1, use_numba=False)
        self.assertEqual(healed_fast, healed_slow)
        np.testing.assert_array_equal(a.flags, b.flags)
        np.testing.assert_array_equal(a.mass, b.mass)
        np.testing.assert_array_equal(a.fill_time_s, b.fill_time_s)

    def test_both_rules_actually_fire_on_the_planted_fixture(self) -> None:
        a, _, _ = _mixed_twin(plant=True)
        self.assertGreater(heal_submerged_interface(a, 0.5, 1, use_numba=True), 0)
        b, _, _ = _mixed_twin(plant=True)
        self.assertGreater(reset_detached_interface_cells(b, use_numba=True), 0)

PROVOKE_SEED = 811

def _provoked(n: int = 16, *, with_filter: bool = False, submerge: bool = False,
              steps: int = 6, seed: int = PROVOKE_SEED):
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

    def test_transitions_actually_fire_in_both_directions(self) -> None:
        state, _ = _provoked()
        filled, emptied = apply_transitions(state, 0.1, use_numba=False)
        self.assertGreater(filled, 0)
        self.assertGreater(emptied, 0)

    def test_the_fronts_are_not_empty(self) -> None:
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
        state, units = _fixture()
        for _ in range(20):
            step(state, units, use_numba=True)
        self.assertEqual(apply_transitions(state, 0.1, use_numba=False), (0, 0))

DIVERGENCE_REGIMES: tuple[tuple[str, dict, bool], ...] = (
    ("quiescent", {}, False),
    ("quiescent + wet filter", {"with_filter": True, "submerge": True}, False),
    ("firing", {}, True),
    ("firing + wet filter", {"with_filter": True, "submerge": True}, True),
)

DIVERGENCE_STEPS = 100

def _divergence_walk(kwargs, provoked, steps):
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

    envelopes: dict

    @classmethod
    def setUpClass(cls) -> None:
        cls.envelopes = {
            label: _divergence_walk(kwargs, provoked, DIVERGENCE_STEPS)
            for label, kwargs, provoked in DIVERGENCE_REGIMES
        }

    def test_the_first_step_leaves_the_moments_untouched(self) -> None:
        for label in self.envelopes:
            with self.subTest(regime=label):
                first = self.envelopes[label]["first_step"]
                self.assertEqual(first["abs"]["rho"], 0.0)
                self.assertEqual(first["abs"]["u"], 0.0)

    def test_the_first_step_costs_about_half_a_float32_ulp_on_f(self) -> None:
        for label in self.envelopes:
            with self.subTest(regime=label):
                first = self.envelopes[label]["first_step"]
                self.assertLess(first["rel"]["f"], 2e-7)
                self.assertLess(first["abs"]["f"], 4e-8)

    def test_a_hundred_steps_stay_inside_the_measured_envelope(self) -> None:
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
        for label, env in self.envelopes.items():
            with self.subTest(regime=label):
                self.assertEqual(
                    env["broke"], {},
                    f"{label}: a discrete field diverged; first divergent step "
                    f"per field is {env['broke']}",
                )

    def test_total_mass_stays_exact_across_both_paths(self) -> None:
        for label, env in self.envelopes.items():
            with self.subTest(regime=label):
                self.assertLess(env["total_gap"], 1e-12)
                self.assertLess(env["conservation"], 1e-12)

@unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
class InletLedgerConservationTests(unittest.TestCase):

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
        halo = _dilate(inlet) & ~inlet & ~obstacle
        n_in = int(inlet.sum())

        inlet_rho = np.full(n_in, REFERENCE_DENSITY * 1.02, dtype=np.float64)
        inlet_u = np.zeros((3, n_in), dtype=np.float64)
        inlet_full = np.full(n_in, float(REFERENCE_DENSITY), dtype=np.float64)

        rng = np.random.default_rng(PROVOKE_SEED)
        delivered = 0.0
        worst = 0.0
        converted = [0, 0]

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
        state, delivered, worst, _ = self._pour(use_numba=False)
        self.assertGreater(delivered, 0.0)
        self.assertLess(worst, 1e-3)
        self.assertLess(
            abs(total_mass(state) - delivered) / delivered, 1e-3
        )

@unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
class MaskCompactionParityTests(unittest.TestCase):

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
        rng = np.random.default_rng(32)
        for shape in ((1, 6, 5), (2, 1, 1), (7, 1, 3), (1, 1, 1), (5, 4, 1)):
            with self.subTest(shape=shape):
                self._check(rng.random(shape) < 0.4)

    def test_the_order_is_the_one_mask_assignment_uses(self) -> None:
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

    @staticmethod
    def _reference(state, to_fluid, to_gas):
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

    @staticmethod
    def _excess(state, seed: int):
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

    def test_agrees_exactly_on_a_provoked_state(self) -> None:
        a, b, _ = _provoked_twin()
        self.assertEqual(
            apply_transitions(a, 0.25, use_numba=True),
            apply_transitions(b, 0.25, use_numba=False),
        )
        _assert_states_identical(a, b, "apply_transitions")
        self.assertEqual(total_mass(a), total_mass(b))

    def test_agrees_exactly_through_a_wet_filter(self) -> None:
        a, b, _ = _provoked_twin(with_filter=True, submerge=True)
        self.assertEqual(
            apply_transitions(a, 0.25, use_numba=True),
            apply_transitions(b, 0.25, use_numba=False),
        )
        _assert_states_identical(a, b, "apply_transitions with a filter")
        self.assertEqual(total_mass(a), total_mass(b))

    def test_a_quiescent_state_converts_nothing_on_either_path(self) -> None:
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
        a, b, units = _provoked_twin()
        full_a = a.rho.astype(np.float64) * a.porous.porosity_f64
        rng = np.random.default_rng(PROVOKE_SEED + 5)
        for _ in range(12):
            step(a, units, use_numba=True)
            step(b, units, use_numba=True)
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
        a, b, _ = _provoked_twin()
        halo = _dilate(a.flags == FLAG_FLUID)
        self.assertEqual(
            ensure_interface_skin(a, halo, use_numba=True),
            ensure_interface_skin(b, halo, use_numba=False),
        )
        _assert_states_identical(a, b, "ensure_interface_skin")

if __name__ == "__main__":
    unittest.main()
