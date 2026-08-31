"""Tests for the filling solver's stability guards and honest reporting.

These exist because of a specific failure that every test in the suite passed
straight through. The inlet used to rebuild its populations from ``state.u``,
which is the *Guo force-corrected* velocity -- it carries half of that step's
gravitational impulse by construction. Feeding it back into ``equilibrium()``
re-imposed that half-impulse every step, so the inlet gained ``0.5 *
gravity_lattice`` of velocity per step forever, crossed the Mach guard the
timestep was sized for, and detonated the whole solve.

Nothing caught it because the completion check counts *flags*: a blow-up
promotes every cell to FLUID at once, which reads as a flawless 100 % fill.
On the real test assembly it produced exactly that, and the result was
recorded in the docs as proof the stage worked.

So the tests here assert two things the old suite never did: that the solver
conserves mass while it runs, and that the inlet does not accelerate.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.filling_stage import (
    DIVERGENCE_MASS_FACTOR,
    MIN_CHOKE_THICKNESS_CELLS,
    STATUS_COMPLETE,
    STATUS_DIVERGED,
    STATUS_STALLED,
    STATUS_TIME_CAPPED,
    FillFrame,
    _min_flow_slice_cells,
    FillingResult,
    measure_choke_thickness_cells,
)
from core.part_manager import PART_TYPE_CAST, PART_TYPE_GATE, PART_TYPE_RUNNER
from tests.test_filling_stage import (  # reuse the established fixtures
    EMPTY_CELL,
    _Alloy,
    _FakeMesh,
    _Inputs,
    _Options,
    _column_mesh,
    lattice_centers,
)

DX = 0.25


def _tower(nx=2, ny=2, nz=6, part_type=PART_TYPE_CAST):
    """A metal column of a chosen cross-section; the pour goes in on top."""
    centers = lattice_centers((nx, ny, nz), dx=DX, origin=(0.0, 0.0, 0.0))
    mesh = _FakeMesh(centers, np.full(len(centers), part_type), DX)
    mesh._top_z = (nz - 1) * DX
    return mesh


def _pour_at_top(mesh):
    from core.pour_location import PourLocation

    return PourLocation(point_in=(0.0, 0.0, getattr(mesh, "_top_z", 5 * DX)))


class _FakeFilterZone:
    """Only the front face matters to the flow rating."""

    def __init__(self, front_face_area_m2: float):
        self.front_face_area_m2 = front_face_area_m2


class _NamedAlloy(_Alloy):
    """_Alloy plus the name the Foseco steel table is keyed on."""

    def __init__(self, name: str):
        self.name = name


def _run(mesh=None, options=None, messages=None, alloy=None,
         detailed_progress_callback=None, **filling_overrides):
    """Drive the real stage, optionally overriding FillingParams fields."""
    from core.filling_stage import maybe_run_filling

    mesh = mesh if mesh is not None else _column_mesh()
    options = options if options is not None else _Options()
    for key, value in filling_overrides.items():
        setattr(options.solidification_params.filling, key, value)
    return maybe_run_filling(
        mesh,
        alloy if alloy is not None else _NamedAlloy("WCB"),
        options,
        _Inputs(_pour_at_top(mesh)),
        progress_message_callback=(
            messages.append if messages is not None else None
        ),
        detailed_progress_callback=detailed_progress_callback,
    )


class InletMomentumTests(unittest.TestCase):
    """
    The regression test for the detonation.

    A boundary condition supplies a *condition*. If the velocity it imposes
    grows without bound then it is a pump, and the only question is how many
    steps the solver survives.
    """

    def _velocity_history(self, steps=1200):
        """Max |u| in the domain, sampled while a real fill runs."""
        import physics.filling as pf

        history: list[float] = []
        original = pf.step

        def traced(state, units, **kwargs):
            original(state, units, **kwargs)
            history.append(float(np.max(np.abs(state.u))))

        pf.step = traced
        try:
            # A tall column so the run lasts long enough for linear growth to
            # be unmistakable rather than arguable.
            _run(mesh=_tower(nz=40), max_fill_time_s=0.5)
        finally:
            pf.step = original
        return np.asarray(history[:steps], dtype=np.float64)

    def test_velocity_does_not_grow_without_bound(self) -> None:
        history = self._velocity_history()
        self.assertGreater(history.size, 200, "fill ended too early to judge")
        early = float(np.max(history[: history.size // 4]))
        late = float(np.max(history[-history.size // 4 :]))
        # The old inlet grew strictly linearly, so the last quarter ran about
        # four times the first. Physical transients settle instead.
        self.assertLess(
            late,
            4.0 * early + 1e-6,
            f"inlet velocity is still accelerating: {early:.5g} -> {late:.5g}",
        )

    def test_velocity_stays_under_the_mach_guard(self) -> None:
        """
        The timestep is sized so nothing exceeds ``lattice_velocity_max``.
        Running past it is not a small error -- it is outside the range where
        the collision operator means anything.
        """
        history = self._velocity_history()
        self.assertLess(
            float(np.max(history)),
            0.05 * 3.0,
            "velocity ran far past the Mach guard the timestep was sized for",
        )


class MassConservationTests(unittest.TestCase):
    """
    Mass is the health check the stage never applied to itself.

    The free surface can hold at most one cell's worth of mass per wet cell,
    so a total above that ceiling is not drift -- it is the bookkeeping having
    come apart.
    """

    def test_wet_mass_stays_under_the_physical_ceiling(self) -> None:
        import physics.filling as pf

        breaches: list[tuple[int, float, int]] = []
        original = pf.step
        counter = [0]

        def traced(state, units, **kwargs):
            original(state, units, **kwargs)
            counter[0] += 1
            wet = (state.flags == pf.FLAG_FLUID) | (
                state.flags == pf.FLAG_INTERFACE
            )
            wet_cells = int(np.count_nonzero(wet))
            wet_mass = float(np.sum(state.mass[wet])) if wet_cells else 0.0
            if wet_mass > DIVERGENCE_MASS_FACTOR * max(wet_cells, 1):
                breaches.append((counter[0], wet_mass, wet_cells))

        pf.step = traced
        try:
            _run(max_fill_time_s=0.05)
        finally:
            pf.step = original

        self.assertEqual(
            breaches[:3], [], f"mass conservation broke at {breaches[:3]}"
        )

    def test_state_stays_finite(self) -> None:
        import physics.filling as pf

        original = pf.step
        first_bad: list = []

        def traced(state, units, **kwargs):
            original(state, units, **kwargs)
            if not first_bad and not (
                np.all(np.isfinite(state.f))
                and np.all(np.isfinite(state.mass))
                and np.all(np.isfinite(state.u))
            ):
                first_bad.append(True)

        pf.step = traced
        try:
            _run(max_fill_time_s=0.05)
        finally:
            pf.step = original
        self.assertEqual(first_bad, [], "solver produced a non-finite field")


class StatusReportingTests(unittest.TestCase):
    """
    "Could not get there", "needed longer" and "the answer is garbage" used to
    surface identically, as a misrun count.
    """

    def test_a_full_fill_reports_complete(self) -> None:
        result = _run(max_fill_time_s=2.0)
        self.assertIsNotNone(result)
        self.assertEqual(result.status, STATUS_COMPLETE)
        self.assertEqual(result.misrun_voxel_count, 0)

    def test_a_truncated_fill_reports_the_time_cap(self) -> None:
        """A cap is not a defect, and must not read as one."""
        messages: list = []
        result = _run(messages=messages, max_fill_time_s=0.004)
        self.assertIsNotNone(result)
        self.assertEqual(result.status, STATUS_TIME_CAPPED)
        self.assertTrue(
            any("max_fill_time_s" in w for w in result.warnings),
            f"time cap was not explained: {result.warnings}",
        )

    def test_status_values_are_distinct(self) -> None:
        self.assertEqual(
            len({
                STATUS_COMPLETE,
                STATUS_TIME_CAPPED,
                STATUS_STALLED,
                STATUS_DIVERGED,
            }),
            4,
        )


class TrustworthyTests(unittest.TestCase):
    """
    A diverged run still returns arrays of the right shape and dtype. They are
    simply meaningless, and seeding solidification from them yields a
    complete, plausible-looking casting result built on debris.
    """

    def _result(self, status):
        empty = np.zeros(3, dtype=np.float32)
        return FillingResult(
            fill_time_s=empty,
            temperature_c=empty,
            inlet_voxel_count=1,
            filled_voxel_count=3,
            misrun_voxel_count=0,
            choke_area_cm2=0.0,
            filter_zones=[],
            steps=10,
            fill_seconds=0.1,
            coarsening=1,
            min_fill_temperature_c=1500.0,
            mean_fill_temperature_c=1550.0,
            status=status,
        )

    def test_diverged_is_not_trustworthy(self) -> None:
        self.assertFalse(self._result(STATUS_DIVERGED).trustworthy)

    def test_every_other_status_is_trustworthy(self) -> None:
        for status in (STATUS_COMPLETE, STATUS_TIME_CAPPED, STATUS_STALLED):
            with self.subTest(status=status):
                self.assertTrue(self._result(status).trustworthy)


class ChokeResolutionTests(unittest.TestCase):
    """
    An under-resolved choke is not a small inaccuracy. Halfway bounce-back
    puts the walls at +-0.5 cells, so a one-cell channel has no interior and
    seals shut -- and the fill then stalls for a reason invisible in the
    output.
    """

    def test_thickness_finds_the_narrowest_section(self) -> None:
        dense = np.full((10, 10, 10), EMPTY_CELL, dtype=np.int8)
        dense[0:6, 4:7, 4:7] = PART_TYPE_RUNNER   # 3 cells thick
        dense[6:10, 5:6, 5:6] = PART_TYPE_GATE    # 1 cell thick
        self.assertLess(measure_choke_thickness_cells(dense), 2.0)

    def test_a_wide_thin_gate_is_not_excused_by_its_area(self) -> None:
        """
        The case the old area-based check got wrong. This gate has a 10-cell
        cross-section -- comfortably "big" -- while being one cell thick, so
        halfway bounce-back seals it. Thickness must catch it; area cannot.
        """
        dense = np.full((12, 12, 12), EMPTY_CELL, dtype=np.int8)
        dense[0:6, 1:11, 4:7] = PART_TYPE_RUNNER
        dense[6:12, 1:11, 5:6] = PART_TYPE_GATE   # 10 wide x 1 thick
        self.assertGreaterEqual(_min_flow_slice_cells(dense), 10)
        self.assertLess(measure_choke_thickness_cells(dense), 2.0)

    def test_no_gating_metal_gives_zero(self) -> None:
        dense = np.full((4, 4, 4), PART_TYPE_CAST, dtype=np.int8)
        self.assertEqual(measure_choke_thickness_cells(dense), 0.0)

    def test_an_under_resolved_choke_is_warned_about(self) -> None:
        """A 1x1 gate is exactly the channel bounce-back seals shut."""
        mesh = _tower(nx=1, ny=1, nz=8, part_type=PART_TYPE_GATE)
        result = _run(mesh=mesh, max_fill_time_s=0.02)
        self.assertIsNotNone(result)
        self.assertLess(result.choke_thickness_cells, MIN_CHOKE_THICKNESS_CELLS)
        self.assertTrue(
            any("cells thick" in w for w in result.warnings),
            f"under-resolved choke was not warned about: {result.warnings}",
        )

    def test_a_resolved_choke_is_not_warned_about(self) -> None:
        """The warning must stay quiet when the geometry is fine."""
        mesh = _tower(nx=4, ny=4, nz=8, part_type=PART_TYPE_GATE)
        result = _run(mesh=mesh, max_fill_time_s=0.02)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.choke_thickness_cells, MIN_CHOKE_THICKNESS_CELLS)
        self.assertFalse(
            any("cells thick" in w for w in result.warnings),
            f"warned about a well-resolved choke: {result.warnings}",
        )


class FrameCaptureTests(unittest.TestCase):
    """
    The solver overwrites its whole state every step, so without capture the
    only record of the pour is one write-once arrival time per cell -- enough
    for "when did this fill?", not enough to show metal moving.
    """

    def setUp(self) -> None:
        self.mesh = _column_mesh()
        self.result = _run(mesh=self.mesh, max_fill_time_s=0.05)
        self.assertIsNotNone(self.result)

    def test_frames_are_captured(self) -> None:
        self.assertGreater(len(self.result.frames), 1)

    def test_frames_are_voxel_shaped(self) -> None:
        n = self.mesh.get_voxel_count()
        for i, frame in enumerate(self.result.frames):
            with self.subTest(frame=i):
                self.assertIsInstance(frame, FillFrame)
                self.assertEqual(frame.fill_fraction.shape, (n,))
                self.assertEqual(frame.speed_m_s.shape, (n,))

    def test_fill_fraction_is_a_fraction(self) -> None:
        """Outside [0, 1] it is not a fill level, it is a symptom."""
        for i, frame in enumerate(self.result.frames):
            with self.subTest(frame=i):
                self.assertTrue(bool(np.all(frame.fill_fraction >= 0.0)))
                self.assertTrue(bool(np.all(frame.fill_fraction <= 1.0)))

    def test_frames_advance_in_time(self) -> None:
        times = [f.time_s for f in self.result.frames]
        self.assertEqual(times, sorted(times))

    def test_the_pour_gains_metal_over_the_run(self) -> None:
        """The front advances; the last frame holds more metal than the first."""
        first = float(self.result.frames[0].fill_fraction.sum())
        last = float(self.result.frames[-1].fill_fraction.sum())
        self.assertGreater(last, first)

    def test_frame_count_stays_within_the_budget(self) -> None:
        """
        The stride halves whenever the buffer doubles, so the count is bounded
        above however long the run lasts -- and, more importantly, bounded
        *below* on a short one. Sizing the stride off max_steps instead gave a
        handful of frames for a quick pour, which is exactly the run worth
        watching.
        """
        from core.filling_stage import FILL_ANIMATION_FRAMES

        for cap in (0.02, 0.2, 1.0):
            with self.subTest(max_fill_time_s=cap):
                result = _run(mesh=_tower(nz=30), max_fill_time_s=cap)
                self.assertIsNotNone(result)
                self.assertLessEqual(
                    len(result.frames), 2 * FILL_ANIMATION_FRAMES + 1
                )

    def test_frames_stay_evenly_spaced_after_decimation(self) -> None:
        """Playback must not run dense then sparse."""
        result = _run(mesh=_tower(nz=30), max_fill_time_s=1.0)
        times = np.array([f.time_s for f in result.frames], dtype=np.float64)
        if times.size < 4:
            self.skipTest("run too short to judge spacing")
        # The final frame is captured off-cadence, at whatever step the run
        # actually ended on, so it is excluded from the spacing check.
        gaps = np.diff(times[:-1])
        self.assertLess(
            float(gaps.max() - gaps.min()),
            float(gaps.mean()) * 0.5,
            f"frame spacing is uneven: {gaps.min():.5g}..{gaps.max():.5g}",
        )

    def test_speeds_are_finite_and_non_negative(self) -> None:
        for i, frame in enumerate(self.result.frames):
            with self.subTest(frame=i):
                self.assertTrue(bool(np.all(np.isfinite(frame.speed_m_s))))
                self.assertTrue(bool(np.all(frame.speed_m_s >= 0.0)))


class PromotedCellSeedingTests(unittest.TestCase):
    """
    A front advancing into open space promotes in sheets, and the donor mask
    excludes cells promoted in the same step. Every cell behind the leading
    edge of such a sheet therefore has no donor -- and used to be left at
    rho = 0 with no populations, which is precisely the state
    _initialise_promoted_cells exists to prevent.
    """

    def _state(self, shape=(7, 7, 7)):
        from physics.filling import create_state

        obstacle = np.zeros(shape, dtype=bool)
        fluid = np.zeros(shape, dtype=bool)
        fluid[0, 0, 0] = True  # one seed cell, far from the block below
        return create_state(
            obstacle,
            dx_m=0.005,
            dt_s=1e-4,
            initial_fluid=fluid,
            initial_temperature_c=1500.0,
        )

    def test_a_sheet_of_promoted_cells_is_all_seeded(self) -> None:
        from physics.filling import _initialise_promoted_cells

        state = self._state()
        # A solid 3x3x3 block promoted at once. Its centre cell touches only
        # other cells in the block, so it has no donor by construction.
        promote = np.zeros(state.shape, dtype=bool)
        promote[3:6, 3:6, 3:6] = True

        _initialise_promoted_cells(state, promote)

        self.assertTrue(
            bool(np.all(state.rho[promote] > 0.0)),
            "a promoted cell was left at rho = 0, which the very next "
            "macroscopic pass divides by",
        )

    def test_the_donorless_centre_is_seeded_at_rest(self) -> None:
        from physics.filling import REFERENCE_DENSITY, _initialise_promoted_cells

        state = self._state()
        promote = np.zeros(state.shape, dtype=bool)
        promote[3:6, 3:6, 3:6] = True
        _initialise_promoted_cells(state, promote)

        self.assertAlmostEqual(
            float(state.rho[4, 4, 4]), float(REFERENCE_DENSITY), places=4
        )

    def test_populations_are_finite_everywhere_after_seeding(self) -> None:
        from physics.filling import _initialise_promoted_cells

        state = self._state()
        promote = np.zeros(state.shape, dtype=bool)
        promote[3:6, 3:6, 3:6] = True
        _initialise_promoted_cells(state, promote)
        self.assertTrue(bool(np.all(np.isfinite(state.f))))
        self.assertTrue(bool(np.all(np.isfinite(state.rho))))

    def test_promoted_temperature_is_never_left_nan(self) -> None:
        """
        A NaN here survives until a neighbour reads across into it, then
        spreads through the advection stencil one cell per step.
        """
        from physics.filling import _initialise_promoted_temperature

        state = self._state()
        promote = np.zeros(state.shape, dtype=bool)
        promote[3:6, 3:6, 3:6] = True
        _initialise_promoted_temperature(state, promote)
        self.assertTrue(
            bool(np.all(np.isfinite(state.temperature[promote]))),
            "a promoted cell kept NO_TEMPERATURE and will poison advection",
        )


class RetreatClearsCellTests(unittest.TestCase):
    """
    The scheme's stated invariant is that a gas cell holds nothing.

    Demotion cleared a retreating cell's mass, flag and temperature but left
    its density behind: ``macroscopic()`` zeroes a gas cell's velocity and
    computes its density unconditionally from ``sum(f)``, so nothing else
    reset it. Populations were already covered by both streaming paths.

    This is an invariant repair, not the cause of the fine-resolution
    divergence -- that was measured separately and is unaffected by it.
    """

    def _retreating_state(self):
        from physics.filling import FLAG_INTERFACE, create_state

        shape = (5, 5, 5)
        obstacle = np.zeros(shape, dtype=bool)
        fluid = np.zeros(shape, dtype=bool)
        fluid[2, 2, 1] = True
        state = create_state(
            obstacle,
            dx_m=0.005,
            dt_s=1e-4,
            initial_fluid=fluid,
            initial_temperature_c=1500.0,
        )
        # Drive one interface cell past the empty threshold so it retreats,
        # and give it populations worth leaking.
        state.flags[2, 2, 2] = FLAG_INTERFACE
        state.rho[2, 2, 2] = 1.0
        state.f[:, 2, 2, 2] = 0.25
        state.mass[2, 2, 2] = -1.0
        return state

    def test_a_retreated_cell_holds_no_populations(self) -> None:
        from physics.filling import FLAG_GAS, apply_transitions

        state = self._retreating_state()
        apply_transitions(state, time_s=0.01)

        self.assertEqual(int(state.flags[2, 2, 2]), int(FLAG_GAS))
        self.assertEqual(
            float(np.max(np.abs(state.f[:, 2, 2, 2]))),
            0.0,
            "a gas cell kept its interface-era populations",
        )

    def test_a_retreated_cell_holds_no_density(self) -> None:
        from physics.filling import apply_transitions

        state = self._retreating_state()
        apply_transitions(state, time_s=0.01)
        self.assertEqual(
            float(state.rho[2, 2, 2]),
            0.0,
            "a gas cell kept a non-zero density, which is the invariant "
            "violation the divergence grew out of",
        )

    def test_every_gas_cell_is_empty_after_a_real_fill(self) -> None:
        """The invariant, checked across a whole running solve."""
        import physics.filling as pf

        violations: list = []
        original = pf.step
        counter = [0]

        def traced(state, units, **kwargs):
            original(state, units, **kwargs)
            counter[0] += 1
            gas = state.flags == pf.FLAG_GAS
            if np.any(gas) and float(np.max(np.abs(state.rho[gas]))) > 1e-6:
                violations.append(counter[0])

        pf.step = traced
        try:
            _run(mesh=_tower(nz=20), max_fill_time_s=0.3)
        finally:
            pf.step = original

        self.assertEqual(
            violations[:5], [], f"gas cells carried density at steps {violations[:5]}"
        )


class InletModeTests(unittest.TestCase):
    """
    ``inlet_mode_code`` and ``pour_rate_lb_s`` were exposed in the UI,
    serialized and round-trip tested, and never read by the stage. They drive
    the inlet now, so a change in them must change the pour.
    """

    def _reported_inlet_velocity(self, messages):
        for message in messages:
            if "inlet" in message and "m/s" in message:
                return float(message.split("m/s")[0].split()[-1])
        self.fail(f"no inlet velocity reported in {messages}")

    def test_head_mode_uses_torricelli(self) -> None:
        messages: list = []
        _run(messages=messages, inlet_mode_code=0, initial_head_in=4.0)
        expected = float(np.sqrt(2.0 * 9.81 * 4.0 * 0.0254))
        self.assertAlmostEqual(
            self._reported_inlet_velocity(messages), expected, places=2
        )

    def test_a_deeper_head_pours_faster(self) -> None:
        shallow, deep = [], []
        _run(messages=shallow, inlet_mode_code=0, initial_head_in=2.0)
        _run(messages=deep, inlet_mode_code=0, initial_head_in=8.0)
        self.assertGreater(
            self._reported_inlet_velocity(deep),
            self._reported_inlet_velocity(shallow),
        )

    def test_velocity_mode_follows_the_pour_rate(self) -> None:
        slow, fast = [], []
        _run(messages=slow, inlet_mode_code=1, pour_rate_lb_s=2.0)
        _run(messages=fast, inlet_mode_code=1, pour_rate_lb_s=8.0)
        self.assertGreater(
            self._reported_inlet_velocity(fast),
            self._reported_inlet_velocity(slow),
        )


class PourRateRuleTests(unittest.TestCase):
    """
    The square-root rule: a casting of W pounds pours in sqrt(W) seconds, so
    the rate is W / sqrt(W) = sqrt(W) lb/s. Pour time in seconds and pour rate
    in lb/s are the same number.

    This replaces a fixed 5 lb/s default that took no account of the casting.
    The mesh already knows the metal volume and the alloy knows its density,
    so there was never a reason to ask.
    """

    def test_the_rate_is_the_square_root_of_the_weight(self) -> None:
        from core.filling_stage import recommended_pour_rate_lb_s

        for weight, expected in ((100.0, 10.0), (25.0, 5.0), (2500.0, 50.0)):
            with self.subTest(weight_lb=weight):
                self.assertAlmostEqual(
                    recommended_pour_rate_lb_s(weight), expected, places=6
                )

    def test_pour_time_and_rate_are_the_same_number(self) -> None:
        """The convenience of the rule, and worth pinning so it stays true."""
        from core.filling_stage import recommended_pour_rate_lb_s

        weight = 400.0
        rate = recommended_pour_rate_lb_s(weight)
        pour_time = weight / rate
        self.assertAlmostEqual(rate, pour_time, places=6)

    def test_a_non_positive_weight_gives_no_rate(self) -> None:
        from core.filling_stage import recommended_pour_rate_lb_s

        self.assertEqual(recommended_pour_rate_lb_s(0.0), 0.0)
        self.assertEqual(recommended_pour_rate_lb_s(-5.0), 0.0)

    def test_weight_counts_metal_only(self) -> None:
        """Mold, cores and chills take no metal, so they take no weight."""
        from core.part_manager import PART_TYPE_CHILL
        from core.filling_stage import pour_weight_lb

        metal = _tower(nx=2, ny=2, nz=6)
        mixed = _tower(nx=2, ny=2, nz=6)
        # Half the same tower re-declared as chill: the weight must halve.
        types = np.array(mixed.voxel_part_types, dtype=np.int8)
        types[: len(types) // 2] = PART_TYPE_CHILL
        mixed.voxel_part_types = types

        alloy = _Alloy()
        self.assertAlmostEqual(
            pour_weight_lb(mixed, alloy),
            pour_weight_lb(metal, alloy) / 2.0,
            places=4,
        )

    def test_weight_is_physically_plausible(self) -> None:
        """
        A 2x2x6 tower of quarter-inch cells is 24 * 0.25^3 = 0.375 in^3.
        At 7200 kg/m^3 that is about 0.0975 lb -- checked against the arithmetic
        rather than against itself, so a unit slip cannot pass.
        """
        from core.filling_stage import pour_weight_lb

        volume_in3 = 24 * (DX ** 3)
        expected_lb = 7200.0 * volume_in3 * 1.6387064e-5 * 2.20462262
        self.assertAlmostEqual(
            pour_weight_lb(_tower(nx=2, ny=2, nz=6), _Alloy()),
            expected_lb,
            places=5,
        )

    def test_a_zero_rate_derives_from_the_casting(self) -> None:
        """0 means derive, the convention pour_time_s already uses."""
        from core.filling_stage import pour_weight_lb, recommended_pour_rate_lb_s

        mesh = _tower(nz=12)
        messages: list = []
        _run(mesh=mesh, messages=messages, inlet_mode_code=1, pour_rate_lb_s=0.0)
        expected = recommended_pour_rate_lb_s(pour_weight_lb(mesh, _Alloy()))
        line = next(m for m in messages if "inlet" in m and "lb/s" in m)
        self.assertIn("sqrt of", line)
        self.assertIn(f"{expected:.1f} lb/s", line)

    def test_a_filter_caps_the_pour_rate(self) -> None:
        """
        A ceramic foam filter is a rated restriction. Above its rating it
        blinds off or breaks up rather than throttling, so the pour rate has
        a hard ceiling whenever one is in the flow path.
        """
        from core.filling_stage import filter_flow_limit_kg_s

        # 50x50 mm face = 25 cm^2. Carbon/low-alloy, lip poured: 0.100 kg/s
        # per cm^2 -> 2.5 kg/s.
        zone = _FakeFilterZone(front_face_area_m2=0.05 * 0.05)
        limit, note = filter_flow_limit_kg_s([zone], _NamedAlloy("WCB"), 0)
        self.assertIsNone(note)
        self.assertAlmostEqual(limit, 2.5, places=6)

    def test_bottom_pouring_is_rated_higher_than_lip(self) -> None:
        from core.filling_stage import filter_flow_limit_kg_s

        zone = _FakeFilterZone(front_face_area_m2=0.05 * 0.05)
        lip, _ = filter_flow_limit_kg_s([zone], _NamedAlloy("WCB"), 0)
        bottom, _ = filter_flow_limit_kg_s([zone], _NamedAlloy("WCB"), 1)
        self.assertGreater(bottom, lip)

    def test_filters_in_parallel_add(self) -> None:
        """One filter per ingate is the normal layout, so capacities sum."""
        from core.filling_stage import filter_flow_limit_kg_s

        one = [_FakeFilterZone(front_face_area_m2=0.05 * 0.05)]
        two = one * 2
        single, _ = filter_flow_limit_kg_s(one, _NamedAlloy("WCB"), 0)
        double, _ = filter_flow_limit_kg_s(two, _NamedAlloy("WCB"), 0)
        self.assertAlmostEqual(double, 2.0 * single, places=6)

    def test_no_filter_means_no_limit(self) -> None:
        from core.filling_stage import filter_flow_limit_kg_s

        limit, note = filter_flow_limit_kg_s([], _NamedAlloy("WCB"), 0)
        self.assertIsNone(limit)
        self.assertIsNone(note)

    def test_an_unrated_alloy_is_reported_not_guessed(self) -> None:
        """
        Substituting a steel figure for a non-ferrous alloy would give a
        plausible rating wrong by tens of percent. It must decline instead.
        """
        from core.filling_stage import filter_flow_limit_kg_s

        zone = _FakeFilterZone(front_face_area_m2=0.05 * 0.05)
        limit, note = filter_flow_limit_kg_s([zone], _NamedAlloy("BRONZE"), 0)
        self.assertIsNone(limit)
        self.assertIsNotNone(note)
        self.assertIn("not covered", note)

    def test_a_filter_in_the_geometry_caps_the_running_pour(self) -> None:
        """
        End to end: the rating has to bind the pour the solver actually runs,
        not just be computable in isolation.
        """
        from core.part_manager import PART_TYPE_FILTER, PART_TYPE_SPRUE

        mesh = _tower(nx=2, ny=2, nz=12, part_type=PART_TYPE_SPRUE)
        # A one-cell-thick filter band partway down the sprue.
        types = np.array(mesh.voxel_part_types, dtype=np.int8)
        z = mesh.voxel_centers[:, 2]
        band = np.isclose(z, np.unique(z)[6])
        types[band] = PART_TYPE_FILTER
        mesh.voxel_part_types = types

        messages: list = []
        result = _run(
            mesh=mesh, messages=messages, max_fill_time_s=0.02,
            inlet_mode_code=1, pour_rate_lb_s=40.0,
        )
        self.assertIsNotNone(result)
        self.assertTrue(
            any("limited by the filter" in w for w in result.warnings),
            f"a 40 lb/s pour through a small filter was not capped: "
            f"{result.warnings}",
        )
        line = next(m for m in messages if "inlet" in m and "m/s" in m)
        self.assertIn("filter-limited", line)

    def test_no_filter_leaves_the_pour_uncapped(self) -> None:
        """The cap must stay out of the way when there is nothing to cap."""
        result = _run(
            mesh=_tower(nz=12), max_fill_time_s=0.02,
            inlet_mode_code=1, pour_rate_lb_s=40.0,
        )
        self.assertIsNotNone(result)
        self.assertFalse(
            any("limited by the filter" in w for w in result.warnings),
            f"capped a pour with no filter present: {result.warnings}",
        )

    def test_an_explicit_rate_is_honoured(self) -> None:
        messages: list = []
        _run(mesh=_tower(nz=12), messages=messages, inlet_mode_code=1,
             pour_rate_lb_s=7.5)
        line = next(m for m in messages if "inlet" in m and "lb/s" in m)
        self.assertIn("7.5 lb/s", line)
        self.assertIn("set", line)


if __name__ == "__main__":
    unittest.main()


class DetachedInterfaceCellTests(unittest.TestCase):
    """
    An interface cell with no wet neighbour at all is a droplet smaller than
    the grid can represent. Every one of its populations comes from the
    gas-side reconstruction, which at that topology has nothing real to close
    against, so its velocity is decided entirely by the closure.

    Measured on the reference assembly: 5-12 such cells at any moment, 0.1-0.3%
    of the metal, and the *only* cells exceeding the Mach guard -- while FLUID
    velocities stayed flat at 0.038-0.046 across 108,000 steps. That split is
    why the fix belongs here and not in a Mach limiter on the equilibrium,
    which would have damped a bulk that was behaving perfectly.
    """

    def _state_with_a_detached_cell(self):
        from physics.filling import (
            FLAG_GAS,
            FLAG_INTERFACE,
            create_state,
        )

        shape = (7, 7, 7)
        state = create_state(
            np.zeros(shape, dtype=bool),
            dx_m=0.005,
            dt_s=1e-4,
            initial_fluid=np.zeros(shape, dtype=bool),
            initial_temperature_c=1500.0,
        )
        state.flags[...] = FLAG_GAS
        # One interface cell, surrounded entirely by gas.
        state.flags[3, 3, 3] = FLAG_INTERFACE
        state.mass[3, 3, 3] = 0.4
        state.f[:, 3, 3, 3] = 0.9          # nonsense populations
        state.u[:, 3, 3, 3] = 0.5          # far past the Mach guard
        state.rho[3, 3, 3] = 2.0
        return state

    def test_a_detached_cell_is_returned_to_rest(self) -> None:
        from physics.filling import reset_detached_interface_cells

        state = self._state_with_a_detached_cell()
        self.assertEqual(reset_detached_interface_cells(state), 1)
        self.assertAlmostEqual(float(np.max(np.abs(state.u[:, 3, 3, 3]))), 0.0)

    def test_its_mass_is_left_alone(self) -> None:
        """
        Momentum is the problem, not the metal. Removing the mass would be a
        leak dressed up as a stability fix.
        """
        from physics.filling import reset_detached_interface_cells

        state = self._state_with_a_detached_cell()
        before = float(state.mass[3, 3, 3])
        reset_detached_interface_cells(state)
        self.assertAlmostEqual(float(state.mass[3, 3, 3]), before)

    def test_a_supported_interface_cell_is_untouched(self) -> None:
        """The intervention must stay off the ordinary free surface."""
        from physics.filling import (
            FLAG_FLUID,
            reset_detached_interface_cells,
        )

        state = self._state_with_a_detached_cell()
        state.flags[3, 3, 2] = FLAG_FLUID  # give it one wet neighbour
        self.assertEqual(reset_detached_interface_cells(state), 0)
        self.assertAlmostEqual(float(np.max(np.abs(state.u[:, 3, 3, 3]))), 0.5)

    def test_it_does_nothing_when_there_is_no_interface(self) -> None:
        from physics.filling import FLAG_GAS, reset_detached_interface_cells

        state = self._state_with_a_detached_cell()
        state.flags[...] = FLAG_GAS
        self.assertEqual(reset_detached_interface_cells(state), 0)

    def test_a_real_fill_does_not_run_away_in_velocity(self) -> None:
        """
        The end-to-end assertion, made on the *settled* part of the run.

        Before this fix the reference assembly climbed steadily to 0.44 -- nine
        times the guard the timestep was sized for -- and kept climbing. After
        it, the late-run figure sits at 0.043 against a 0.05 design target.
        The startup transient is deliberately excluded: the inlet establishing
        itself produces a brief legitimate spike, and asserting on the whole-run
        peak would test that instead of the runaway this fix addresses.
        """
        import physics.filling as pf

        history: list[float] = []
        original = pf.step

        def traced(state, units, **kwargs):
            original(state, units, **kwargs)
            history.append(float(np.max(np.abs(state.u))))

        pf.step = traced
        try:
            _run(mesh=_tower(nz=30), max_fill_time_s=0.5)
        finally:
            pf.step = original

        self.assertGreater(len(history), 400, "run too short to judge")
        late = np.asarray(history[len(history) // 2:], dtype=np.float64)
        # Not growing: the second half's late quarter must not exceed its
        # early quarter by a wide margin.
        quarter = max(1, late.size // 4)
        self.assertLess(
            float(np.median(late[-quarter:])),
            3.0 * float(np.median(late[:quarter])) + 0.02,
            f"velocity is still climbing across the settled run: "
            f"{np.median(late[:quarter]):.4g} -> {np.median(late[-quarter:]):.4g}",
        )


class ProgressReportingTests(unittest.TestCase):
    """
    A fill takes minutes and used to drive no progress bar at all.

    Its messages went to a Qt status label and nothing else, so the console
    printed the mesh stage, then `Interface bridges added...`, then nothing
    for the whole solve. That reads as a hang rather than as work, and the
    bridging stage -- which takes 0.4 s -- got the blame for it.
    """

    def test_the_fill_drives_a_progress_bar(self) -> None:
        updates: list = []
        _run(
            mesh=_tower(nz=20),
            max_fill_time_s=0.3,
            detailed_progress_callback=lambda c, t, s: updates.append((c, t, s)),
        )
        self.assertTrue(updates, "the fill reported no progress at all")
        self.assertTrue(all(stage == "Filling" for _, _, stage in updates))

    def test_progress_is_monotonic(self) -> None:
        """
        Reported against the step budget, not filled cells. Metal sloshes back
        and forth across the same cells while a front is stalled, so a
        fill-percentage bar would run backwards.
        """
        updates: list = []
        _run(
            mesh=_tower(nz=20),
            max_fill_time_s=0.3,
            detailed_progress_callback=lambda c, t, s: updates.append((c, t, s)),
        )
        current = [c for c, _, _ in updates]
        self.assertEqual(current, sorted(current))

    def test_progress_never_exceeds_its_total(self) -> None:
        updates: list = []
        _run(
            mesh=_tower(nz=20),
            max_fill_time_s=0.3,
            detailed_progress_callback=lambda c, t, s: updates.append((c, t, s)),
        )
        for current, total, _ in updates:
            self.assertLessEqual(current, total)
            self.assertGreater(total, 0)

    def test_no_callback_is_fine(self) -> None:
        """Every other caller passes nothing; it must stay optional."""
        self.assertIsNotNone(_run(mesh=_tower(nz=12), max_fill_time_s=0.05))
