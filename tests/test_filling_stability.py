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

The prescribed-velocity inlet that replaced it failed a second time, for the
opposite reason: a fixed-velocity equilibrium pin cannot feel back-pressure, so
it kept overwriting whatever the domain streamed back into it. Measured on
Test_Model/Filling_Test, delivery fell to 6% of the prescribed rate and then
went negative, the sprue sat in suction at rho 0.99995 against a hydrostatic
requirement of ~1.0028, and the inlet layer oscillated to +-218 per-cell mass
before the divergence guard fired. Head mode is a pressure boundary now, and
:class:`PressureInletTests` asserts the three properties that failure taught:
delivery falls away when the column backs up, a standing column is hydrostatic
rather than in suction, and the pinned inlet keeps its interface skin.
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
from core.part_manager import (
    PART_TYPE_CAST,
    PART_TYPE_CORE,
    PART_TYPE_GATE,
    PART_TYPE_RUNNER,
)
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


def _walled_tower(nz=12, width=3):
    """
    The same column, but inside a mold block so the domain has walls.

    ``_tower`` produces a *fully occupied* dense grid: every cell is metal, so
    ``_obstacle_mask`` finds nothing solid and the lattice has no boundary at
    all. Both streaming paths wrap periodically -- ``stream_kernel``'s own
    docstring says "real domains are closed by obstacle cells, so the wrap
    never fires" -- which makes that fixture a torus. Metal leaving the bottom
    re-enters at the top, and gravity, having nothing to press against,
    accelerates the whole fluid without limit: free fall, forever, with no
    hydrostatic gradient and no steady state.

    That was invisible while the inlet prescribed its own constant velocity.
    A pressure inlet lets the flow decide, so anything about settling,
    back-pressure or hydrostatics has to be asked of a bounded domain. Cells
    outside the column are cores, which ``_obstacle_mask`` makes solid, so the
    metal sits in a closed channel with a floor.
    """
    n = width + 2
    centers = lattice_centers((n, n, nz + 2), dx=DX, origin=(0.0, 0.0, 0.0))
    index = np.rint(np.asarray(centers) / DX).astype(int)
    types = np.full(len(centers), PART_TYPE_CORE, dtype=np.int8)
    inside = (
        (index[:, 0] >= 1) & (index[:, 0] <= width)
        & (index[:, 1] >= 1) & (index[:, 1] <= width)
        & (index[:, 2] >= 1) & (index[:, 2] <= nz)
    )
    types[inside] = PART_TYPE_CAST
    mesh = _FakeMesh(centers, types, DX)
    metal_centers = np.asarray(centers)[inside]
    mesh._top_z = float(metal_centers[:, 2].max())
    mesh._pour_xy = (
        float(metal_centers[:, 0].mean()),
        float(metal_centers[:, 1].mean()),
    )
    return mesh


def _pour_at_top(mesh):
    from core.pour_location import PourLocation

    x, y = getattr(mesh, "_pour_xy", (0.0, 0.0))
    return PourLocation(point_in=(x, y, getattr(mesh, "_top_z", 5 * DX)))


def _inlet_cells(mesh, coarsening=1):
    """The lattice cells ``run_filling_stage`` will pin, for a top pour."""
    from core.filling_stage import (
        build_grid_mapping,
        coarsen_part_types,
        dense_part_types,
        _resolve_inlet_cells,
    )

    centers = np.asarray(mesh.voxel_centers, dtype=np.float64)
    dx_in = float(np.asarray(mesh.voxel_sizes).ravel()[0])
    mapping = build_grid_mapping(centers, dx_in)
    types = coarsen_part_types(
        dense_part_types(mapping, mesh.voxel_part_types), coarsening
    )
    return _resolve_inlet_cells(_pour_at_top(mesh), mapping, types, coarsening)


def _head_mode(**overrides):
    """FillingParams overrides for a plain head-driven pour."""
    settings = dict(
        fill_grid_coarsening=1,
        lattice_velocity_max=0.05,
        initial_head_in=4.0,
        inlet_mode_code=0,
        pour_rate_lb_s=0.0,
        pour_temperature_c=0.0,
    )
    settings.update(overrides)
    return settings


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
        """
        Max |u| in the domain, sampled while a real fill runs.

        Measured on a *walled* tower. The unwalled ``_tower`` is a torus (see
        that fixture's docstring): gravity there has nothing to press against,
        so the fluid free-falls for the whole run and any growth ratio measures
        free fall rather than the inlet. On the walled column the flow reaches a
        steady state -- quartile peaks 0.042, 0.041, 0.024, 0.004 over a
        50,000-step fill -- which is the behaviour these assertions are about.
        """
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
            _run(mesh=_walled_tower(nz=40, width=4), max_fill_time_s=0.5)
        finally:
            pf.step = original
        return np.asarray(history[:steps], dtype=np.float64)

    def test_velocity_does_not_grow_without_bound(self) -> None:
        history = self._velocity_history()
        self.assertGreater(history.size, 200, "fill ended too early to judge")
        early = float(np.max(history[: history.size // 4]))
        late = float(np.max(history[-history.size // 4 :]))
        # The old inlet grew strictly linearly, so the last quarter ran about
        # four times the first. Physical transients settle instead: measured
        # 0.0022 -> 0.0056 here, a factor of 2.6 and flattening.
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


class FluidMassTracksDensityTests(unittest.TestCase):
    """
    The regression test for the mass-exchange weighting.

    ``advance_mass`` used to weight *every* wet-wet exchange by
    ``(eps_x + eps_nb)/2``. For a FLUID cell ``eps = m / (rho * porosity)`` is
    1 only to round-off, and because the weight scales the exchange amplitude,
    any drift in eps amplified its own exchanges: a random walk whose step size
    was proportional to the error. Measured with the sprue of
    ``Test_Model/Filling_Test`` running completely full (332 of 332 cells
    FLUID), per-cell FLUID mass reached +-70,000 and wet mass alternated
    sample-to-sample until the divergence guard fired at step 251,000.

    Koerner et al. (2005), Table 1, averages the fill fraction over
    interface-interface pairs only; a pair involving a FLUID cell exchanges at
    weight 1. At weight 1 the direction sum telescopes into the cell's own
    density change, so a FLUID cell's mass tracks its density and cannot drift.
    These tests assert that identity directly, on lattices built so it is the
    only thing being measured.
    """

    NU = 0.006 / 7200.0
    DX_M = 0.0152

    def _units(self):
        from physics.fluid_flow import lattice_units

        return lattice_units(self.DX_M, self.NU, u_max_physical_m_s=2.8)

    def _closed_box(self, n=10):
        """Walls on all six faces, every interior cell FLUID, no interface."""
        import physics.filling as pf

        obstacle = np.zeros((n, n, n), dtype=bool)
        obstacle[0] = obstacle[-1] = True
        obstacle[:, 0] = obstacle[:, -1] = True
        obstacle[:, :, 0] = obstacle[:, :, -1] = True
        units = self._units()
        state = pf.create_state(
            obstacle, dx_m=self.DX_M, dt_s=units.dt_s, initial_fluid=~obstacle
        )
        # The point of the fixture: no free surface anywhere, so nothing but
        # the FLUID-FLUID weighting is under test.
        self.assertEqual(int(np.count_nonzero(state.flags == pf.FLAG_INTERFACE)), 0)
        self.assertEqual(int(np.count_nonzero(state.flags == pf.FLAG_GAS)), 0)
        return state, units

    def test_fluid_mass_stays_equal_to_its_density(self) -> None:
        """
        ``|m - rho * porosity|`` over 2,000 steps of a closed, all-FLUID box.

        Measured: 1.6e-4 worst case, and it does not accumulate -- sampled out
        to 32,000 steps it stays at the same 1e-4, because it is the float32
        storage of ``rho`` against a float64 ``mass``, not drift. The bound
        below is an order of magnitude above that and nine orders below the
        +-70,000 the old weighting reached.
        """
        import physics.filling as pf

        state, units = self._closed_box()
        worst = 0.0
        for k in range(2000):
            pf.step(state, units)
            if (k + 1) % 100 == 0:
                full = (
                    state.rho.astype(np.float64)
                    * state.porous.porosity.astype(np.float64)
                )
                fluid = state.flags == pf.FLAG_FLUID
                worst = max(worst, float(np.abs(state.mass - full)[fluid].max()))
        self.assertLess(
            worst, 1e-3,
            f"FLUID mass drifted from its density by {worst:.3e}",
        )

    def test_a_fluid_cells_mass_change_equals_its_density_change(self) -> None:
        """
        Koerner's Table 1 asserted as the identity it implies, on a field
        deliberately pushed away from ``eps = 1``.

        At weight 1 the direction sum telescopes -- everything streaming in
        minus everything streaming out -- so ``dm`` is the cell's own density
        change no matter what its fill fraction says. Under the old
        eps-weighting ``dm`` is that same quantity multiplied by a number that
        *is* the drift, which is how the drift amplified itself.

        Measured over one step from eps in [0.39, 1.67]: ``max|dm - drho|`` is
        2.2e-7 here against a mean ``|drho|`` of 1.1e-5 -- a ratio of 0.02, and
        pure float32 storage noise. The old weighting gives 1.3e-5 on the same
        state, a ratio of 1.14: the error is as large as the signal.
        """
        import physics.filling as pf

        state, units = self._closed_box()
        fluid = state.flags == pf.FLAG_FLUID
        rng = np.random.default_rng(3)
        state.mass[fluid] += 0.2 * rng.standard_normal(int(np.count_nonzero(fluid)))
        eps = state.mass[fluid] / state.rho.astype(np.float64)[fluid]
        self.assertLess(eps.min(), 0.6, "fixture is not actually perturbed")
        self.assertGreater(eps.max(), 1.4, "fixture is not actually perturbed")

        rho_before = state.f.astype(np.float64).sum(axis=0)
        mass_before = state.mass.copy()
        pf.step(state, units)
        rho_after = state.f.astype(np.float64).sum(axis=0)

        dm = (state.mass - mass_before)[fluid]
        drho = (rho_after - rho_before)[fluid]
        scale = float(np.abs(drho).mean())
        self.assertGreater(scale, 0.0, "nothing moved, so nothing was tested")
        self.assertLess(
            float(np.abs(dm - drho).max()), 0.1 * scale,
            "a FLUID cell's mass change is not its density change: "
            f"max|dm - drho| = {float(np.abs(dm - drho).max()):.3e} against a "
            f"mean |drho| of {scale:.3e}",
        )

    def test_a_closed_box_conserves_its_mass_exactly(self) -> None:
        """No inlet, no free surface: total mass is a constant of the motion."""
        import physics.filling as pf

        state, units = self._closed_box()
        start = pf.total_mass(state)
        wet_cells = int(np.count_nonzero(state.flags != pf.FLAG_OBSTACLE))
        for _ in range(2000):
            pf.step(state, units)
            self.assertLess(
                pf.total_mass(state), DIVERGENCE_MASS_FACTOR * wet_cells
            )
        self.assertAlmostEqual(pf.total_mass(state) / start, 1.0, places=9)

    def test_no_interface_cell_is_created_out_of_nothing(self) -> None:
        """A box with no free surface must not grow one."""
        import physics.filling as pf

        state, units = self._closed_box()
        for _ in range(2000):
            pf.step(state, units)
        self.assertEqual(
            int(np.count_nonzero(state.flags == pf.FLAG_INTERFACE)), 0
        )

    def test_a_column_under_an_interface_layer_stays_bounded(self) -> None:
        """
        The same identity with a real free surface on top.

        This is the configuration the ``Test_Model/Filling_Test`` sprue was in
        when per-cell mass reached +-70,000: a full channel with an interface
        skin above it, run long. Measured here over 5,000 steps, FLUID mass
        stays inside [0.9997, 1.0035]; the assertion is deliberately looser so
        it fails on a runaway rather than on the third digit.
        """
        import physics.filling as pf

        n, nz = 8, 16
        obstacle = np.zeros((n, n, nz), dtype=bool)
        obstacle[0] = obstacle[-1] = True
        obstacle[:, 0] = obstacle[:, -1] = True
        obstacle[:, :, 0] = True
        fluid = np.zeros((n, n, nz), dtype=bool)
        fluid[1:-1, 1:-1, 1:nz - 3] = True
        units = self._units()
        state = pf.create_state(
            obstacle, dx_m=self.DX_M, dt_s=units.dt_s, initial_fluid=fluid
        )
        self.assertGreater(
            int(np.count_nonzero(state.flags == pf.FLAG_INTERFACE)), 0,
            "fixture has no free surface to test",
        )

        low, high = 1e9, -1e9
        for k in range(5000):
            pf.step(state, units)
            if (k + 1) % 50 == 0:
                fl = state.flags == pf.FLAG_FLUID
                if fl.any():
                    low = min(low, float(state.mass[fl].min()))
                    high = max(high, float(state.mass[fl].max()))
        self.assertGreater(low, -0.1, f"a FLUID cell emptied to {low:.4g}")
        self.assertLess(high, 1.6, f"a FLUID cell overfilled to {high:.4g}")


class StatusReportingTests(unittest.TestCase):
    """
    "Could not get there", "needed longer" and "the answer is garbage" used to
    surface identically, as a misrun count.
    """

    def test_a_full_fill_reports_complete(self) -> None:
        """
        A cavity that genuinely fills reports ``complete`` and no misrun.

        The fixture is deliberately shallow. Under Koerner's mass-exchange
        weighting -- see :class:`FluidMassTracksDensityTests` -- an interface
        cell whose neighbours are all wet or solid exchanges at weight 1 in
        every direction, so its mass change is exactly its own density change
        and it stops filling: a *submerged* interface cell is frozen wherever
        the front left it. Measured on ``_walled_tower(nz=4, width=4)``, the
        run ends with 44 FLUID cells and 20 INTERFACE cells, and all twenty of
        those have no gas neighbour left, at eps between 0.05 and 0.84.

        That is a gap in the flag bookkeeping, not in the weighting: Koerner's
        scheme reinitialises a gasless interface cell to FLUID and this
        implementation has no such rule. It was invisible while the old
        weighting let FLUID mass drift upward past the transition threshold
        and convert those cells anyway -- and that drift is the +-70,000
        runaway. Removing the drift exposed the gap; it did not cause it.
        """
        result = _run(mesh=_walled_tower(nz=2, width=2),
                      **_head_mode(max_fill_time_s=2.0))
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

    def _reported_inlet_density(self, messages):
        for message in messages:
            if "inlet" in message and "rho" in message:
                return float(message.split("rho")[1].split()[0])
        self.fail(f"no inlet density reported in {messages}")

    def test_head_mode_imposes_a_pressure_not_a_velocity(self) -> None:
        """
        The contract change. Head mode used to prescribe Torricelli's
        sqrt(2 g h) as a constant entry velocity; it now imposes the
        metallostatic head as a *pressure* and measures the velocity from the
        flow. A prescribed velocity is what the message must no longer claim.
        """
        messages: list = []
        _run(messages=messages, inlet_mode_code=0, initial_head_in=4.0)
        line = next(m for m in messages if "inlet" in m and "head" in m)
        self.assertIn("pressure boundary", line)
        self.assertIn("velocity follows the flow", line)
        self.assertGreater(self._reported_inlet_density(messages), 1.0)

    def test_a_deeper_head_raises_the_inlet_pressure(self) -> None:
        """
        rho = rho_0 (1 + g h / c_s^2): the head enters as pressure, so a deeper
        basin means a denser inlet, not a faster one.
        """
        shallow, deep = [], []
        _run(messages=shallow, inlet_mode_code=0, initial_head_in=2.0)
        _run(messages=deep, inlet_mode_code=0, initial_head_in=8.0)
        self.assertGreater(
            self._reported_inlet_density(deep),
            self._reported_inlet_density(shallow),
        )

    def test_velocity_mode_follows_the_pour_rate(self) -> None:
        slow, fast = [], []
        _run(messages=slow, inlet_mode_code=1, pour_rate_lb_s=2.0)
        _run(messages=fast, inlet_mode_code=1, pour_rate_lb_s=8.0)
        self.assertGreater(
            self._reported_inlet_velocity(fast),
            self._reported_inlet_velocity(slow),
        )


class PressureInletTests(unittest.TestCase):
    """
    What the head-mode inlet must do now that it imposes pressure, not velocity.

    Each of these is a measured failure of the prescribed-velocity design,
    turned into an assertion. See the module docstring for the numbers.
    """

    def _poured_lb(self, messages):
        """Cumulative poured weight from the last progress report."""
        reports = [m for m in messages if "lb poured" in m]
        self.assertTrue(reports, f"no poured weight reported in {messages}")
        return float(reports[-1].split("lb poured")[0].split(",")[-1])

    def test_a_head_driven_pour_fills_and_delivers_metal(self) -> None:
        """
        The whole point: gravity plus a head pressure has to move metal.
        Delivery is reported as weight, so a pour that reports 0 lb has not
        happened however many cells its flag count claims.

        Measured: 84 % of cells FLUID, 96 % by mass, 0.3 lb delivered. It does
        not report ``complete`` and cannot. ``_tower`` has no obstacle cells at
        all, so both streaming paths wrap periodically -- see
        :func:`_walled_tower`'s docstring -- and the bottom layer drains into
        the inlet pinned at the top. The 13 cells left over are that seam. The
        run used to report ``complete`` because FLUID mass drifted past the
        transition threshold and converted them; see
        :class:`FluidMassTracksDensityTests`.
        """
        messages: list = []
        result = _run(
            mesh=_tower(nz=20), messages=messages, **_head_mode(max_fill_time_s=0.5)
        )
        self.assertIsNotNone(result)
        self.assertNotEqual(result.status, STATUS_DIVERGED)
        self.assertGreater(
            result.filled_fraction, 0.8,
            f"the pour barely moved: {result.filled_fraction:.3f} of cells",
        )
        self.assertGreater(self._poured_lb(messages), 0.0)

    def test_delivery_falls_away_when_the_column_backs_up(self) -> None:
        """
        Back-pressure, which is the property a fixed-velocity pin cannot have.

        Total domain mass changes only through the inlet pin -- the solver
        conserves the rest, which :class:`MassConservationTests` checks -- so
        its per-step increment *is* the net injection. In a closed tower with
        the metal nowhere left to go, that increment must collapse. Measured:
        0.0151 lattice-mass/step over the first quarter of the run against
        0.000025 over the last, a factor of 600. The old pin instead kept
        writing its prescribed velocity and deleted the metal streaming back
        into it, which is how delivery went negative on the real assembly.
        """
        import physics.filling as pf

        mesh = _walled_tower(nz=12)
        totals: list[float] = []
        original = pf.step

        def traced(state, units, **kwargs):
            original(state, units, **kwargs)
            totals.append(pf.total_mass(state))

        pf.step = traced
        try:
            result = _run(mesh=mesh, **_head_mode(max_fill_time_s=1.5))
        finally:
            pf.step = original

        self.assertIsNotNone(result)
        self.assertNotEqual(result.status, STATUS_DIVERGED)
        delivery = np.diff(np.asarray(totals, dtype=np.float64))
        self.assertGreater(delivery.size, 400, "run too short to judge")
        quarter = delivery.size // 4
        early = float(delivery[:quarter].mean())
        late = float(delivery[-quarter:].mean())
        self.assertGreater(early, 0.0, "the inlet never delivered anything")
        self.assertLess(
            late,
            0.2 * early,
            f"the inlet kept delivering into a full tower: "
            f"{early:.5g} -> {late:.5g} lattice-mass/step",
        )

    def test_a_standing_column_is_hydrostatic_not_in_suction(self) -> None:
        """
        The suction-column regression.

        With delivery choked, the whole sprue sat *below* reference density --
        mean rho 0.99995 at the sprue bottom against the ~1.0028 the head
        requires -- so nothing pushed the metal onward and the front crept.
        A column fed by a real head stands on its own weight instead: density
        rises downward at 3 g_lattice per cell.

        Measured on this fixture at the end of the fill: 1.00333 over the
        bottom quarter against 1.00211 over the top, a difference of 1.2e-3
        where the hydrostatic prediction over that span is 7.8e-4. Asserted as
        a sign, not a value, because the free surface is still moving when the
        clock runs out.

        The unpinned column *rings*, and the ring is the same size as the
        gradient. This fixture is a sealed box that only fills to about 40 %,
        so what settles is a half-wet column with a sloshing mode. Sampled
        every 250 steps out to 10,000, bottom-minus-top is positive in 37 of
        40 samples and negative in three; the single 300-step snapshot this
        test used to take landed on one of the three. The sign is therefore
        taken over a window now, after the launch transient has decayed --
        which is what "stands on its own weight" can mean for a column that
        has not stopped moving. A clean, monotone gradient is available from a
        column that is actually full: a bounded 3x3 bore filled to 4/5 and run
        4,000 steps grades 1.001982 at the floor to 0.999904 at the surface,
        with FLUID mass equal to density at every layer.
        """
        import physics.filling as pf

        mesh = _walled_tower(nz=12)
        held: dict = {}
        original = pf.step

        def traced(state, units, **kwargs):
            original(state, units, **kwargs)
            held["state"] = state
            held["units"] = units

        pf.step = traced
        try:
            result = _run(mesh=mesh, **_head_mode(max_fill_time_s=1.0))
        finally:
            pf.step = original
        self.assertIsNotNone(result)
        self.assertNotEqual(result.status, STATUS_DIVERGED)

        # Let the column settle with no inlet holding it: a pressure gradient
        # that only exists while something is pinning the top is not
        # hydrostatic, it is the boundary condition showing through.
        state, units = held["state"], held["units"]
        for _ in range(3000):
            pf.step(state, units, track_free_surface=True)

        def _gradient() -> float:
            wet = (state.flags == pf.FLAG_FLUID) | (
                state.flags == pf.FLAG_INTERFACE
            )
            layers = np.nonzero(np.any(wet, axis=(0, 1)))[0]
            low, high = int(layers.min()), int(layers.max())
            span = high - low + 1
            self.assertGreater(
                span, 4, "column too short to have a top and a bottom"
            )
            bottom = wet.copy()
            bottom[:, :, low + span // 4:] = False
            top = wet.copy()
            top[:, :, : high - span // 4 + 1] = False
            # Density, not mass: mass is the free surface's fill level,
            # density is the pressure. Gas cells hold neither, hence the wet
            # mask.
            return float(state.rho[bottom].mean()) - float(state.rho[top].mean())

        samples = [_gradient()]
        for _ in range(20):
            for _ in range(100):
                pf.step(state, units, track_free_surface=True)
            samples.append(_gradient())
        mean_gradient = float(np.mean(samples))
        self.assertGreater(
            mean_gradient,
            0.0,
            f"the column is not standing on its own weight: mean "
            f"bottom-minus-top {mean_gradient:.3e} over {len(samples)} samples "
            f"(min {min(samples):.3e}, max {max(samples):.3e})",
        )

    def test_the_pinned_inlet_keeps_its_interface_skin(self) -> None:
        """
        No FLUID cell may touch a GAS cell -- apply_transitions' own invariant,
        which the inlet pin sits outside because it writes FLAG_FLUID by hand.

        When it was violated, the inlet stopped receiving every population
        arriving with c_z = -1 (a sixth of the weight) and its measured
        z-momentum jumped from -0.009 to +0.147 in one step: the column
        reversed and ran upward at the Mach cap. A boundary that measures its
        own velocity cannot measure it through a hole.
        """
        import physics.filling as pf
        from physics.filling import _dilate

        mesh = _walled_tower(nz=12)
        inlet = _inlet_cells(mesh)
        self.assertGreater(inlet.shape[0], 0)
        breaches: list[int] = []
        original = pf.step
        counter = [0]

        def traced(state, units, **kwargs):
            original(state, units, **kwargs)
            counter[0] += 1
            pinned = np.zeros(state.shape, dtype=bool)
            pinned[inlet[:, 0], inlet[:, 1], inlet[:, 2]] = True
            if np.any(_dilate(pinned) & ~pinned & (state.flags == pf.FLAG_GAS)):
                breaches.append(counter[0])

        pf.step = traced
        try:
            _run(mesh=mesh, **_head_mode(max_fill_time_s=0.3))
        finally:
            pf.step = original
        # The check runs after step() and before the next pin, so a breach here
        # is one the pin is about to read through.
        self.assertEqual(
            breaches[-3:], [], f"the inlet touched gas at steps {breaches[-3:]}"
        )

    def test_an_unreachable_head_is_called_out_before_the_clock_runs(self) -> None:
        """
        Metal does not climb above the free surface of the basin feeding it.
        A pour point plus head below the highest metal cell cannot finish the
        casting, and saying so costs one comparison; not saying so costs the
        whole step budget and then reports a time cap, which reads as "needed
        longer" when it means "cannot get there from here".
        """
        result = _run(
            mesh=_tower(nz=20), **_head_mode(initial_head_in=0.1, max_fill_time_s=0.02)
        )
        self.assertIsNotNone(result)
        self.assertTrue(
            any("cannot reach the top of the metal" in w for w in result.warnings),
            f"an unreachable head was not called out: {result.warnings}",
        )

    def test_a_reachable_head_is_not_warned_about(self) -> None:
        result = _run(
            mesh=_tower(nz=20), **_head_mode(initial_head_in=4.0, max_fill_time_s=0.02)
        )
        self.assertIsNotNone(result)
        self.assertFalse(
            any("cannot reach the top of the metal" in w for w in result.warnings),
            f"warned about a head that reaches: {result.warnings}",
        )

    def test_velocity_mode_still_prescribes_a_constant_velocity(self) -> None:
        """
        The redesign is head mode's alone. In velocity mode the user is
        asserting a delivery rate, and a boundary that meters itself would
        quietly ignore them.
        """
        messages: list = []
        _run(
            mesh=_tower(nz=12), messages=messages,
            **_head_mode(inlet_mode_code=1, pour_rate_lb_s=7.5, max_fill_time_s=0.02)
        )
        line = next(m for m in messages if "inlet" in m and "m/s" in m)
        self.assertIn("velocity (7.5 lb/s", line)
        self.assertNotIn("pressure boundary", line)


class MassProgressReportingTests(unittest.TestCase):
    """
    A cell count alone cannot tell a slow fill from a stopped one.

    Interface cells fill continuously from eps 0 to eps 1 and only flip to
    FLUID at the top of that band, so a front creeping through them takes on
    mass every report while flipping no flags. Measured on
    Test_Model/Filling_Test: 428 casting cells held at eps ~ 0.5, advancing,
    with the cell count flat -- which the old stall guard read as stopped.
    """

    def _reports(self, messages):
        # Keyed on "lb poured", not on "by mass": the stall notice also quotes
        # a mass fraction ("holding at 147/160 cells (92%, 98% by mass)") and
        # carries no weight, so a run that ends stalled used to leave that
        # line as the last match and break the parse below.
        return [m for m in messages if "lb poured" in m]

    def _poured_lb(self, messages):
        reports = self._reports(messages)
        self.assertTrue(reports, f"no poured weight reported in {messages}")
        return float(reports[-1].split("lb poured")[0].split(",")[-1])

    def test_progress_reports_cells_mass_and_weight(self) -> None:
        messages: list = []
        _run(mesh=_tower(nz=20), messages=messages, **_head_mode(max_fill_time_s=0.2))
        reports = self._reports(messages)
        self.assertTrue(reports, f"no mass-based progress reported: {messages}")
        for line in reports:
            with self.subTest(line=line):
                self.assertIn("cells", line)
                self.assertIn("% by mass", line)
                self.assertIn("lb poured", line)

    def test_the_mass_fraction_is_a_fraction(self) -> None:
        """Above 100% it is not a fill level, it is a symptom."""
        messages: list = []
        _run(mesh=_tower(nz=20), messages=messages, **_head_mode(max_fill_time_s=0.3))
        for line in self._reports(messages):
            percent = int(line.split("% by mass")[0].split(",")[-1])
            with self.subTest(line=line):
                self.assertGreaterEqual(percent, 0)
                self.assertLessEqual(percent, 100)

    def test_poured_weight_only_grows_while_metal_is_arriving(self) -> None:
        """
        Cumulative delivery, so it must never run backwards over a fill that
        is still taking metal on.
        """
        messages: list = []
        _run(mesh=_tower(nz=20), messages=messages, **_head_mode(max_fill_time_s=0.4))
        poured = [
            float(m.split("lb poured")[0].split(",")[-1])
            for m in self._reports(messages)
        ]
        self.assertEqual(poured, sorted(poured), f"poured weight went backwards: {poured}")

    def test_the_poured_weight_matches_the_castings_own_weight(self) -> None:
        """
        The unit conversion, checked against arithmetic rather than against
        itself. One lattice mass unit at REFERENCE_DENSITY is one cell volume
        of liquid metal -- ``total_mass_kg``'s convention -- so a completed
        pour must have delivered about the casting's weight.

        Measured: a 2x2x40 tower of quarter-inch cells is 2.5 in^3, which
        ``pour_weight_lb`` puts at 0.65 lb of steel, and the run reports
        0.6 lb delivered. The balance is the seed ``create_state`` starts with,
        which came in before the inlet did anything.

        The run gets there at 92 % of cells and 98 % by mass rather than at
        ``complete``: ``_tower`` is a torus, and its bottom layer drains into
        the inlet at the top -- see
        :meth:`PressureInletTests.test_a_head_driven_pour_fills_and_delivers_metal`.
        The delivered weight is what this test is about, and it is unchanged.
        """
        from core.filling_stage import pour_weight_lb

        mesh = _tower(nz=40)
        messages: list = []
        result = _run(mesh=mesh, messages=messages, **_head_mode(max_fill_time_s=1.0))
        self.assertIsNotNone(result)
        self.assertNotEqual(result.status, STATUS_DIVERGED)
        self.assertGreater(result.filled_fraction, 0.9)
        weight_lb = pour_weight_lb(mesh, _Alloy())
        poured_lb = self._poured_lb(messages)
        self.assertGreater(
            poured_lb, 0.5 * weight_lb,
            f"reported {poured_lb:.2f} lb poured into a {weight_lb:.2f} lb casting",
        )
        self.assertLess(
            poured_lb, 1.2 * weight_lb,
            f"reported {poured_lb:.2f} lb poured into a {weight_lb:.2f} lb casting",
        )

    def test_a_stall_needs_both_measures_flat(self) -> None:
        """
        The epsilon exists so that a front advancing only in mass is not
        declared stopped. Pinning it here keeps the two measures from
        collapsing back into one.
        """
        from core.filling_stage import STALL_MASS_FRACTION_EPSILON

        self.assertGreater(STALL_MASS_FRACTION_EPSILON, 0.0)
        self.assertLess(STALL_MASS_FRACTION_EPSILON, 0.01)

    def test_a_stall_is_reported_with_both_measures(self) -> None:
        """A stall message that gives only a cell count hides the disagreement."""
        messages: list = []
        _run(mesh=_walled_tower(nz=12), messages=messages,
             **_head_mode(max_fill_time_s=1.5))
        stall = [m for m in messages if "has not advanced" in m]
        if not stall:
            self.skipTest("this fixture did not stall on this run")
        self.assertIn("by mass", stall[0])


class InterfaceSkinTests(unittest.TestCase):
    """
    ``ensure_interface_skin`` is the hand-maintained half of an invariant
    ``apply_transitions`` maintains for itself.
    """

    def _state(self, shape=(5, 5, 5)):
        from physics.filling import create_state

        fluid = np.zeros(shape, dtype=bool)
        fluid[2, 2, 2] = True
        return create_state(
            np.zeros(shape, dtype=bool),
            dx_m=0.005,
            dt_s=1e-4,
            initial_fluid=fluid,
            initial_temperature_c=1500.0,
        )

    def test_a_gas_cell_in_the_halo_is_promoted(self) -> None:
        from physics.filling import FLAG_GAS, FLAG_INTERFACE, ensure_interface_skin

        state = self._state()
        state.flags[state.flags == FLAG_INTERFACE] = FLAG_GAS
        halo = np.zeros(state.shape, dtype=bool)
        halo[2, 2, 3] = True
        self.assertEqual(ensure_interface_skin(state, halo), 1)
        self.assertEqual(int(state.flags[2, 2, 3]), int(FLAG_INTERFACE))

    def test_a_promoted_cell_is_given_populations(self) -> None:
        """rho = 0 is what the very next macroscopic pass divides by."""
        from physics.filling import FLAG_GAS, FLAG_INTERFACE, ensure_interface_skin

        state = self._state()
        state.flags[state.flags == FLAG_INTERFACE] = FLAG_GAS
        halo = np.zeros(state.shape, dtype=bool)
        halo[2, 2, 3] = True
        ensure_interface_skin(state, halo)
        self.assertGreater(float(state.rho[2, 2, 3]), 0.0)
        self.assertTrue(bool(np.all(np.isfinite(state.f[:, 2, 2, 3]))))
        self.assertTrue(bool(np.isfinite(state.temperature[2, 2, 3])))

    def test_it_leaves_wet_cells_alone(self) -> None:
        """Idempotent, because it runs every step of every fill."""
        from physics.filling import FLAG_GAS, ensure_interface_skin

        state = self._state()
        before = state.flags.copy()
        halo = state.flags != FLAG_GAS      # everything already wet or solid
        self.assertEqual(ensure_interface_skin(state, halo), 0)
        self.assertTrue(bool(np.array_equal(state.flags, before)))

    def test_it_takes_no_mass(self) -> None:
        """A promotion is a topology repair, not a delivery."""
        from physics.filling import FLAG_GAS, FLAG_INTERFACE, ensure_interface_skin

        state = self._state()
        state.flags[state.flags == FLAG_INTERFACE] = FLAG_GAS
        total = float(state.mass.sum())
        halo = np.zeros(state.shape, dtype=bool)
        halo[2, 2, 3] = True
        ensure_interface_skin(state, halo)
        self.assertAlmostEqual(float(state.mass.sum()), total, places=12)


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


class SmagorinskyConstantTests(unittest.TestCase):
    """
    ``smagorinsky_cs`` now reaches the collision operator, which means the
    stage is the last place a nonsense value can be caught.

    It is refused rather than clamped. Pulling 1.2 back to 0.5 would still run
    a subgrid viscosity many times what was asked for and report nothing, and
    a settings field that quietly does something else is the exact failure
    this parameter was hidden for in the first place.
    """

    def _warning(self, result):
        return next(
            (w for w in result.warnings if "smagorinsky_cs" in w), None
        )

    def test_a_usable_constant_passes_without_comment(self) -> None:
        result = _run(mesh=_tower(nz=12), max_fill_time_s=0.05, smagorinsky_cs=0.2)
        self.assertIsNotNone(result)
        self.assertIsNone(self._warning(result))

    def test_zero_is_accepted(self) -> None:
        """0 means no eddy viscosity, which is a legitimate ask, not an error."""
        result = _run(mesh=_tower(nz=12), max_fill_time_s=0.05, smagorinsky_cs=0.0)
        self.assertIsNotNone(result)
        self.assertIsNone(self._warning(result))

    def test_a_negative_constant_falls_back_and_says_so(self) -> None:
        result = _run(mesh=_tower(nz=12), max_fill_time_s=0.05, smagorinsky_cs=-0.5)
        self.assertIsNotNone(result)
        message = self._warning(result)
        self.assertIsNotNone(message, f"no warning raised: {result.warnings}")
        self.assertIn("0.16", message)

    def test_an_absurd_constant_falls_back_and_says_so(self) -> None:
        result = _run(mesh=_tower(nz=12), max_fill_time_s=0.05, smagorinsky_cs=5.0)
        self.assertIsNotNone(result)
        self.assertIsNotNone(
            self._warning(result), f"no warning raised: {result.warnings}"
        )

    def test_the_stage_constant_reaches_the_solver(self) -> None:
        """
        End to end: the value read from settings has to be the value ``step``
        is called with, not merely validated on the way past.
        """
        import physics.filling as pf

        seen: list = []
        original = pf.step

        def traced(state, units, **kwargs):
            seen.append(kwargs.get("smagorinsky_cs"))
            return original(state, units, **kwargs)

        pf.step = traced
        try:
            _run(mesh=_tower(nz=12), max_fill_time_s=0.02, smagorinsky_cs=0.23)
        finally:
            pf.step = original

        self.assertTrue(seen, "the stage never stepped the solver")
        self.assertTrue(all(cs == 0.23 for cs in seen), f"got {set(seen)}")
