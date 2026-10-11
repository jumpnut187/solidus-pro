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
    PART_TYPE_FILTER,
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


def _filtered_tower(nz=14, width=3, band=(5, 7)):
    """
    ``_walled_tower`` with a ceramic slab laid across the channel.

    The band is three cells deep on purpose. Two cells trips the
    "``the filter is 1.0 cells thick``" seal warning -- ``cells_along_flow``
    divides the cell count by the front-face area, so a two-cell band on a
    3x3 channel measures one cell along the flow and halfway bounce-back
    closes it. Three cells is the marginal-but-usable resolution the stage
    warns about rather than refuses, which is also what the real assembly's
    filters resolve to.

    Filter cells are flow restrictors, not metal: ``_obstacle_mask`` lets the
    metal through them and ``_metal_mask`` leaves them out of the fill target,
    exactly as on a real assembly. So the tower still reports 100 % when the
    metal has crossed the slab and filled the chamber below.
    """
    mesh = _walled_tower(nz=nz, width=width)
    index = np.rint(np.asarray(mesh.voxel_centers) / DX).astype(int)
    in_band = (
        (mesh.voxel_part_types == PART_TYPE_CAST)
        & (index[:, 2] >= band[0])
        & (index[:, 2] <= band[1])
    )
    assert int(in_band.sum()) == width * width * (band[1] - band[0] + 1)
    mesh.voxel_part_types[in_band] = PART_TYPE_FILTER
    return mesh


def _walled_tower_with_a_sealed_pocket(nz=12, width=3):
    """
    ``_walled_tower`` plus one metal cell the metal can never reach.

    The tower alone can no longer be used to watch an inlet back up. Since
    :func:`~physics.filling.heal_submerged_interface`, a walled tower fills
    and the stage exits on the step it reports ``complete`` -- measured, 4,939
    steps at 0.82 s against a 1.5 s cap -- so the run ends at the moment the
    column reaches the inlet and the backed-up regime never happens inside it.
    (Before the heal the same fixture ran the full cap with 24 of its cells
    stuck as frozen interface, which is what made it look backed up; that
    "collapse" was the bug being measured, not the boundary.)

    One sealed pocket restores the regime honestly: the corner cell of the
    mold block is marked metal, every one of its 18 neighbours is core, so the
    fill can never reach it, the run cannot report ``complete``, and the tower
    goes on being poured into long after it is full. That is a genuine misrun,
    not a contrivance -- an unvented pocket is exactly the geometry the misrun
    count exists to report.
    """
    mesh = _walled_tower(nz=nz, width=width)
    index = np.rint(np.asarray(mesh.voxel_centers) / DX).astype(int)
    corner = (index[:, 0] == 0) & (index[:, 1] == 0) & (index[:, 2] == 0)
    assert int(corner.sum()) == 1, "the fixture lost its corner cell"
    mesh.voxel_part_types[corner] = PART_TYPE_CAST
    return mesh


def _gasless_interface_cells(state):
    """
    Interface cells with no gas neighbour left, in the neighbourhood
    ``_dilate`` uses to advance the front.

    This is the mechanism :func:`~physics.filling.heal_submerged_interface`
    exists to remove, measured directly rather than through a status string.
    """
    import physics.filling as pf

    gas = state.flags == pf.FLAG_GAS
    has_gas_neighbour = np.zeros_like(gas)
    for i in range(1, pf.Q):
        c = pf.LATTICE_VELOCITIES[i]
        has_gas_neighbour |= np.roll(
            gas, shift=(int(c[0]), int(c[1]), int(c[2])), axis=(0, 1, 2)
        )
    return (state.flags == pf.FLAG_INTERFACE) & ~has_gas_neighbour


def _closed_box_state(n=10):
    """A sealed box of FLUID, with no free surface anywhere to start from."""
    import physics.filling as pf
    from physics.fluid_flow import lattice_units

    nu, dx_m = 0.006 / 7200.0, 0.0152
    obstacle = np.zeros((n, n, n), dtype=bool)
    obstacle[0] = obstacle[-1] = True
    obstacle[:, 0] = obstacle[:, -1] = True
    obstacle[:, :, 0] = obstacle[:, :, -1] = True
    units = lattice_units(dx_m, nu, u_max_physical_m_s=2.8)
    state = pf.create_state(
        obstacle, dx_m=dx_m, dt_s=units.dt_s, initial_fluid=~obstacle
    )
    return state, units


def _submerged_pair():
    """
    A closed box holding one submerged interface cell and one that is not.

    ``(5, 5, 5)`` is under-full at eps 0.40 with nothing but fluid around it --
    the frozen cell. ``(5, 5, 6)`` sits at eps 0.80 and keeps a gas neighbour
    above it, so the heal must leave it alone; it is the only cell that can
    take ``(5, 5, 5)``'s deficit, which makes the transfer readable in one
    subtraction.
    """
    import physics.filling as pf

    state, units = _closed_box_state()
    state.flags[5, 5, 5] = pf.FLAG_INTERFACE
    state.mass[5, 5, 5] = 0.40
    state.flags[5, 5, 6] = pf.FLAG_INTERFACE
    state.mass[5, 5, 6] = 0.80
    state.flags[5, 5, 7] = pf.FLAG_GAS
    state.mass[5, 5, 7] = 0.0
    state.f[:, 5, 5, 7] = 0.0
    return state, units


def _final_state(run_kwargs):
    """Run the real stage and hand back the solver state it finished on."""
    import physics.filling as pf

    held = {}
    original = pf.step

    def traced(state, units, **kwargs):
        original(state, units, **kwargs)
        held["state"] = state
        return state

    pf.step = traced
    try:
        result = _run(**run_kwargs)
    finally:
        pf.step = original
    return result, held["state"]


def _pour_at_top(mesh):
    from core.pour_location import PourLocation

    x, y = getattr(mesh, "_pour_xy", (0.0, 0.0))
    return PourLocation(point_in=(x, y, getattr(mesh, "_top_z", 5 * DX)))


def _inlet_cells(mesh, coarsening=1):
    """
    The lattice cells ``run_filling_stage`` will pin, for a top pour.

    Shifted by :data:`~core.filling_stage.MOLD_PAD_CELLS`, because the stage
    resolves the pour point against the bounding-box grid the mapping knows and
    then pads that grid with a cell of mold on every face before it is solved.
    Any caller comparing these indices against a live ``state`` array is
    indexing the padded grid, so without the shift it reads a cell diagonally
    adjacent to the real inlet on all three axes.
    """
    from core.filling_stage import (
        MOLD_PAD_CELLS,
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
    resolved = _resolve_inlet_cells(_pour_at_top(mesh), mapping, types, coarsening)
    return resolved + int(MOLD_PAD_CELLS)


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

    def test_a_closed_box_conserves_its_mass_exactly_while_healing(self) -> None:
        """
        The same invariant with
        :func:`~physics.filling.heal_submerged_interface` actually firing.

        The plain closed box has no interface cells at all, so the heal is
        inert in it and the test above cannot see this. Seeding a 27-cell
        submerged pocket at random fill levels makes the heal convert every one
        of them on the first step, hand out the deficits, and strand what it
        cannot place. Measured: mass ratio 1.000000000000000 over 2,000 steps,
        i.e. 2.2e-16 -- one ulp, not a leak.
        """
        import physics.filling as pf

        state, units = self._closed_box()
        pocket = np.zeros(state.shape, dtype=bool)
        pocket[4:7, 4:7, 4:7] = True
        pocket &= state.flags == pf.FLAG_FLUID
        state.flags[pocket] = pf.FLAG_INTERFACE
        rng = np.random.default_rng(0)
        state.mass[pocket] = rng.uniform(0.05, 0.9, int(np.count_nonzero(pocket)))
        self.assertEqual(int(np.count_nonzero(pocket)), 27, "fixture drifted")

        start = pf.total_mass(state)
        for _ in range(2000):
            pf.step(state, units)
        self.assertEqual(
            int(np.count_nonzero(state.flags == pf.FLAG_INTERFACE)), 0,
            "the submerged pocket was not healed",
        )
        self.assertAlmostEqual(pf.total_mass(state) / start, 1.0, places=13)

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


def _table1_advance_mass(state, f_post_collision, f_streamed):
    """
    Koerner Table 1, written out independently of ``physics.filling``.

    The reference for the bit-identity test below. Weight 1 whenever either
    cell of the pair is FLUID, the clamped fill-fraction average otherwise, and
    no porosity term anywhere -- which is what the shipped exchange must reduce
    to on a lattice with no filter in it, since that is all it ever was there.
    """
    import physics.filling as pf
    from physics.filling import FLAG_FLUID, FLAG_INTERFACE, LATTICE_VELOCITIES, Q
    from physics.fluid_flow import OPPOSITE

    outgoing = np.asarray(f_post_collision, dtype=np.float64)
    incoming = np.asarray(f_streamed, dtype=np.float64)
    eps = np.clip(pf.fluid_fraction(state), 0.0, 1.0)
    fluid = state.flags == FLAG_FLUID
    wet = (state.flags == FLAG_INTERFACE) | fluid
    opp = OPPOSITE.astype(np.intp)
    axes = (0, 1, 2)
    delta = np.zeros(state.shape, dtype=np.float64)
    for i in range(1, Q):
        c = LATTICE_VELOCITIES[i]
        shift = (-int(c[0]), -int(c[1]), -int(c[2]))
        ibar = int(opp[i])
        weight = np.where(
            fluid | np.roll(fluid, shift=shift, axis=axes),
            1.0,
            0.5 * (eps + np.roll(eps, shift=shift, axis=axes)),
        )
        delta += np.where(
            wet & np.roll(wet, shift=shift, axis=axes),
            (incoming[ibar] - outgoing[i]) * weight,
            0.0,
        )
    return state.mass + delta


def _porous_weight_advance_mass(state, f_post_collision, f_streamed):
    """
    The exchange as it stood at f1daf97, porous override included.

    Kept only so the tests below can measure the gap they close. Any pair with
    a porous cell on either side was diverted to ``0.5 * (eps_A + eps_B)`` with
    eps **unclamped**, overriding Table 1. Reproduced here rather than in the
    solver, so a future edit cannot quietly restore it and still pass.
    """
    import physics.filling as pf
    from physics.filling import FLAG_FLUID, FLAG_INTERFACE, LATTICE_VELOCITIES, Q
    from physics.fluid_flow import OPPOSITE

    outgoing = np.asarray(f_post_collision, dtype=np.float64)
    incoming = np.asarray(f_streamed, dtype=np.float64)
    eps_raw = pf.fluid_fraction(state)
    eps = np.clip(eps_raw, 0.0, 1.0)
    fluid = state.flags == FLAG_FLUID
    wet = (state.flags == FLAG_INTERFACE) | fluid
    porous = state.porous.porosity_f64 < 1.0
    opp = OPPOSITE.astype(np.intp)
    axes = (0, 1, 2)
    delta = np.zeros(state.shape, dtype=np.float64)
    for i in range(1, Q):
        c = LATTICE_VELOCITIES[i]
        shift = (-int(c[0]), -int(c[1]), -int(c[2]))
        ibar = int(opp[i])
        weight = np.where(
            fluid | np.roll(fluid, shift=shift, axis=axes),
            1.0,
            0.5 * (eps + np.roll(eps, shift=shift, axis=axes)),
        )
        weight = np.where(
            porous | np.roll(porous, shift=shift, axis=axes),
            0.5 * (eps_raw + np.roll(eps_raw, shift=shift, axis=axes)),
            weight,
        )
        delta += np.where(
            wet & np.roll(wet, shift=shift, axis=axes),
            (incoming[ibar] - outgoing[i]) * weight,
            0.0,
        )
    return state.mass + delta


class PorousPairWeightTests(unittest.TestCase):
    """
    The regression test for the filter-outlet mass overload.

    :class:`FluidMassTracksDensityTests` above asserts Koerner Table 1 on a
    clear lattice. Inside and around a filter the solver used to override
    Table 1 with ``0.5 * (eps_A + eps_B)`` on *unclamped* eps, for any pair
    with a porous cell on either side. The weight scales the exchange
    amplitude and is a function of the cell's own eps, so an eps error
    amplified its own exchanges -- the same positive feedback Table 1 was
    adopted to remove, re-enabled everywhere a filter touches.

    Nothing arrests it. ``transition_scan_kernel``, ``apply_transitions`` and
    ``heal_submerged_interface`` all select on FLAG_INTERFACE, so a FLUID
    cell's mass is written by the exchange and by nothing else. Measured on the
    real ``Test 2`` assembly at the filter outlet: the effective weight, which
    the scheme requires to be 1, reached +10.7 and -13.0; 786 of 2,009 FLUID
    samples of the filter's bottom plane were over full against 1 of 754
    INTERFACE samples of the same cells; eps_max in the porous zone climbed
    1.046 -> 7.76 while eps_min fell to -9.5, and the run died on a
    negative-mass cell 8,800 steps later.

    Mass is conserved throughout -- the weight is symmetric under the pair
    swap, measured at exactly zero residual over 36,550 pairs -- so these
    tests assert the fill *fraction*, not the total. The phantom fill is stolen
    from real cells, which is why the total looks healthy the whole way down.
    """

    NU = 0.006 / 7200.0
    DX_M = 0.0152
    #: Enough to separate the two weight rules without putting any one test
    #: past a few seconds.
    #:
    #: **Recalibrated when TAU_MINUS_MAX landed.** The deleted override's gain
    #: comes from the exchange *amplitude*, and clamping ``tau_minus`` from
    #: 18,060 to 5 relaxes the odd moments hard enough to cut that amplitude on
    #: a quiescent fixture -- so on this box the override no longer runs away at
    #: all. It settles at a fixed bias instead: measured over 8,000 steps the
    #: shipped rule holds a 5.0000e-02 seed at 5.0027e-02 (+0.055 %) and the
    #: override at 5.0253e-02 (+0.51 %), both dead flat from step 2,000 onward.
    #: The tests below therefore measure that 9x bias separation rather than a
    #: runaway, and their bounds moved with it: pre-clamp the same fixture gave
    #: +4.2 % at 2,000 steps and +14 % at 4,000.
    #:
    #: The runaway itself is still real -- it needs a live front and the
    #: exchange amplitudes of a real pour, where the diagnosis measured
    #: eps 7.76 at the filter outlet. It is no longer reproducible on a
    #: two-second fixture, which is a fact about the fixture, not about the
    #: defect.
    STEPS = 2000

    def _units(self):
        from physics.fluid_flow import lattice_units

        return lattice_units(self.DX_M, self.NU, u_max_physical_m_s=2.8)

    def _submerged_filter(self, seed_eps=1.05):
        """
        A closed, all-FLUID box with a three-cell porosity-0.85 band across it.

        :class:`FluidMassTracksDensityTests`' fixture plus a filter. There is
        no GAS and no INTERFACE cell anywhere, so ``apply_transitions``,
        ``heal_submerged_interface`` and ``_redistribute`` are no-ops by
        construction and the only thing that writes ``state.mass`` is the
        exchange. The filter is also *fully submerged*, which is the state the
        real assembly's filter is in for most of a pour and the one where the
        re-pin repair that was considered instead does nothing at all --
        ``_redistribute`` has no INTERFACE neighbour to donate to and strands
        the correction back on the donor.

        The band is seeded 5 % over full because the defect does not
        self-start: seeded at exactly eps = 1 it stays there for 6,000 steps.
        It is an amplifier, and 5 % stands in for the 1-2 % that
        ``state.rho`` being the start-of-step density puts on every FLUID cell
        for free.
        """
        import physics.filling as pf
        from physics.filling import PorousZone

        shape = (5, 5, 16)
        obstacle = np.zeros(shape, dtype=bool)
        obstacle[0] = obstacle[-1] = True
        obstacle[:, 0] = obstacle[:, -1] = True
        obstacle[:, :, 0] = obstacle[:, :, -1] = True
        units = self._units()
        zone = PorousZone.inert(shape)
        band = np.zeros(shape, dtype=bool)
        band[1:4, 1:4, 6:9] = True
        zone.porosity[band] = np.float32(0.85)
        state = pf.create_state(
            obstacle,
            dx_m=self.DX_M,
            dt_s=units.dt_s,
            initial_fluid=~obstacle,
            porous=zone,
        )
        self.assertEqual(
            int(np.count_nonzero(state.flags == pf.FLAG_INTERFACE)), 0,
            "the filter must be submerged, with no free surface anywhere",
        )
        self.assertEqual(int(np.count_nonzero(state.flags == pf.FLAG_GAS)), 0)
        full = state.rho.astype(np.float64) * state.porous.porosity_f64
        # create_state now carries the porosity factor, so the band starts at
        # eps = 1 rather than 1/0.85 = 1.176. Assert that before overwriting it:
        # a fixture that starts 18 % over full cannot say what the seed did.
        self.assertAlmostEqual(
            float(np.abs(state.mass[band] / full[band] - 1.0).max()), 0.0,
            places=6,
            msg="create_state seeded the filter without its porosity factor",
        )
        state.mass[band] = (full * seed_eps)[band]
        return state, units, band

    def _worst_fill_fraction_error(self, state, band):
        """``max|eps - 1|`` over the band, from the state's own fields."""
        full = state.rho.astype(np.float64) * state.porous.porosity_f64
        with np.errstate(invalid="ignore", divide="ignore"):
            eps = np.where(full > 0, state.mass / full, 1.0)
        return float(np.abs(eps[band] - 1.0).max())

    def _drift(self, use_numba, advance_mass=None):
        """``(seeded, final, dM/M0)`` for one weight rule on one step path."""
        import physics.filling as pf

        state, units, band = self._submerged_filter()
        seeded = self._worst_fill_fraction_error(state, band)
        start = float(state.mass.sum())
        original = pf.advance_mass
        if advance_mass is not None:
            pf.advance_mass = advance_mass
        try:
            for _ in range(self.STEPS):
                pf.step(state, units, use_numba=use_numba)
        finally:
            pf.advance_mass = original
        final = self._worst_fill_fraction_error(state, band)
        return seeded, final, (float(state.mass.sum()) - start) / abs(start)

    def test_a_seeded_error_does_not_grow_numpy(self) -> None:
        """The witness, on the numerics reference path."""
        seeded, final, drift = self._drift(use_numba=False)
        self.assertAlmostEqual(seeded, 0.05, places=6)
        # 1.002, not 1.01: see STEPS. The override now settles at +0.51 % on
        # this fixture, so a 1 % bound would pass under the very rule this test
        # exists to reject. Measured shipped value is +0.055 %.
        self.assertLess(
            final, 1.002 * seeded,
            f"the fill-fraction error grew {seeded:.4e} -> {final:.4e} over "
            f"{self.STEPS} steps: the exchange weight is feeding a cell's own "
            f"eps back into its exchange amplitude",
        )
        self.assertLess(abs(drift), 1e-12, f"mass drifted by {drift:.3e}")

    def test_a_seeded_error_does_not_grow_numba(self) -> None:
        """
        The same on the kernel path.

        Both paths carried the same override, in the same precedence, with the
        same clamping choice, so a one-sided fix would leave every real solve
        broken while this file's NumPy tests passed.
        """
        from physics.fluid_flow import NUMBA_AVAILABLE

        if not NUMBA_AVAILABLE:
            self.skipTest("numba is not installed")
        seeded, final, drift = self._drift(use_numba=True)
        self.assertAlmostEqual(seeded, 0.05, places=6)
        self.assertLess(
            final, 1.002 * seeded,
            f"the fill-fraction error grew {seeded:.4e} -> {final:.4e} over "
            f"{self.STEPS} steps on the kernel path",
        )
        self.assertLess(abs(drift), 1e-12, f"mass drifted by {drift:.3e}")

    def test_the_deleted_weighting_is_what_grew_it(self) -> None:
        """
        The discriminator: the same fixture under the rule that was removed.

        Without this the two tests above could pass on a fixture that had lost
        the ability to fail.

        **What it measures changed with ``TAU_MINUS_MAX``** -- see ``STEPS``.
        It used to assert a runaway: the override grew the error monotonically
        at 3.83e-5 per step from a 5 % seed and 2.99e-5 from a 17.6 % one, a
        rate independent of the seed, which is the signature of linear positive
        feedback. Clamping ``tau_minus`` cut the exchange amplitude the gain
        feeds on, so on this quiescent box the override now settles at a fixed
        bias instead of diverging. The separation it leaves is still decisive
        and is what is asserted: the override's excess over the seed is **9x**
        the shipped rule's (+0.51 % against +0.055 %), and the bound below asks
        for 4x.

        Mass is conserved to round-off under both rules, which is the point
        this test has always made: the defect was never a leak.
        """
        _, fixed, _ = self._drift(use_numba=False)
        seeded, broken, drift = self._drift(
            use_numba=False, advance_mass=_porous_weight_advance_mass
        )
        # The 4x bound below is a ratio of two small differences, so its
        # denominator has to be established first: if a future change made the
        # shipped rule *decay* the seed, ``fixed - seeded`` would go negative
        # and the comparison would pass no matter what the override did.
        self.assertGreater(
            fixed - seeded, 0.0,
            f"Table 1 no longer grows the seed at all on this fixture "
            f"({seeded:.6f} -> {fixed:.6f}), so the separation bound below "
            f"cannot be read as a ratio; re-derive it",
        )
        self.assertGreater(
            broken - seeded, 4.0 * (fixed - seeded),
            f"the fixture no longer discriminates: the porous override held "
            f"{seeded:.6f} -> {broken:.6f} against Table 1's {fixed:.6f}, so a "
            f"passing test above proves nothing",
        )
        self.assertGreater(
            broken, 1.003 * seeded,
            f"the override's bias has vanished on this fixture "
            f"({seeded:.6f} -> {broken:.6f}); it needs re-deriving before the "
            f"bounds above mean anything",
        )
        self.assertLess(abs(drift), 1e-12, "the override leaked mass after all")

    def test_a_free_surface_through_a_filter_stays_bounded(self) -> None:
        """
        The other regime: a real front crossing the band, where the transition
        and heal scans are live and the band is only partly wet.

        The submerged fixture isolates the weight. This one is the geometry the
        defect was found in -- a filter with a partly filled channel below it,
        where the eps gradient across the outlet face points the bias into the
        cell.

        **This is now an invariant test, not a discriminator, and the change is
        measured.** Before ``TAU_MINUS_MAX`` the two rules separated cleanly
        here: 5.010e-02 shipped against 5.917e-02 (+18 %) for the override.
        Clamping ``tau_minus`` cut the exchange amplitude the override's gain
        feeds on, and on this configuration the two rules now agree to six
        figures -- both reach 5.0035e-02. So it no longer proves anything about
        the weight rule; it asserts what it says on the tin, that a FLUID cell
        in or beside a filter with a live front stays near full, and the
        discrimination lives in
        :meth:`test_the_deleted_weighting_is_what_grew_it` and in the diagnosis
        report's real-assembly measurements.

        FLUID cells only, deliberately. An INTERFACE cell in the same band
        reaches ``|eps - 1|`` of 0.53 under both rules, because
        ``_redistribute`` applies no capacity check to its recipients and can
        push a receiving interface cell over full -- documented, unchanged here,
        and harmless because that cell is re-examined by
        ``apply_transitions`` on the next step. A FLUID cell has no such
        re-examination, which is the whole reason the defect lived there.
        """
        import physics.filling as pf
        from physics.filling import PorousZone

        shape = (5, 5, 18)
        obstacle = np.zeros(shape, dtype=bool)
        obstacle[0] = obstacle[-1] = True
        obstacle[:, 0] = obstacle[:, -1] = True
        obstacle[:, :, 0] = obstacle[:, :, -1] = True
        units = self._units()
        zone = PorousZone.inert(shape)
        band = np.zeros(shape, dtype=bool)
        band[1:4, 1:4, 5:8] = True
        zone.porosity[band] = np.float32(0.85)
        initial = ~obstacle & (np.arange(shape[2])[None, None, :] <= 5)
        state = pf.create_state(
            obstacle,
            dx_m=self.DX_M,
            dt_s=units.dt_s,
            initial_fluid=initial,
            porous=zone,
        )
        self.assertGreater(
            int(np.count_nonzero(state.flags == pf.FLAG_INTERFACE)), 0,
            "fixture has no free surface",
        )
        full = state.rho.astype(np.float64) * state.porous.porosity_f64
        seed = band & (state.flags == pf.FLAG_FLUID)
        self.assertGreater(int(np.count_nonzero(seed)), 0, "the band is dry")
        state.mass[seed] = (full * 1.05)[seed]

        seeded = self._worst_fill_fraction_error(state, seed)
        worst = 0.0
        for k in range(self.STEPS):
            pf.step(state, units)
            if (k + 1) % 100 == 0:
                fluid_band = band & (state.flags == pf.FLAG_FLUID)
                if fluid_band.any():
                    worst = max(
                        worst, self._worst_fill_fraction_error(state, fluid_band)
                    )
        self.assertGreater(worst, 0.0, "no FLUID cell stayed in the band")
        # 1.01, not 1.05: the measured value is +0.07 % and a 5 % bound on a
        # test that no longer discriminates is a bound that asserts nothing.
        self.assertLess(
            worst, 1.01 * seeded,
            f"a FLUID cell in or beside the filter reached eps "
            f"{1 + worst:.4g} against a seeded {1 + seeded:.4g}; a full cell "
            f"holds exactly its pore volume, so back-pressure belongs in rho, "
            f"not here",
        )

    def test_the_exchange_is_unchanged_where_there_is_no_filter(self) -> None:
        """
        Bit-identity against plain Koerner Table 1 on a clear lattice.

        The override was guarded on "any porous cell in the domain", so on a
        model with no filter it never executed and its deletion cannot have
        changed anything. Asserted rather than argued, and asserted against
        :func:`_table1_advance_mass` -- an independent transcription of the
        rule -- so the claim survives the old code being gone. A real free
        surface, with transitions firing, because an all-FLUID box only
        exercises the weight-1 branch.
        """
        import copy

        import physics.filling as pf

        shape = (5, 5, 18)
        obstacle = np.zeros(shape, dtype=bool)
        obstacle[0] = obstacle[-1] = True
        obstacle[:, 0] = obstacle[:, -1] = True
        obstacle[:, :, 0] = obstacle[:, :, -1] = True
        units = self._units()
        initial = ~obstacle & (np.arange(shape[2])[None, None, :] <= 5)
        shipped = pf.create_state(
            obstacle, dx_m=self.DX_M, dt_s=units.dt_s, initial_fluid=initial
        )
        self.assertGreater(
            int(np.count_nonzero(shipped.flags == pf.FLAG_INTERFACE)), 0
        )
        self.assertGreater(int(np.count_nonzero(shipped.flags == pf.FLAG_GAS)), 0)
        reference = copy.deepcopy(shipped)

        original = pf.advance_mass
        for k in range(300):
            pf.step(shipped, units, use_numba=False)
            pf.advance_mass = _table1_advance_mass
            try:
                pf.step(reference, units, use_numba=False)
            finally:
                pf.advance_mass = original
            self.assertTrue(
                np.array_equal(shipped.mass, reference.mass),
                f"mass diverged from Table 1 at step {k + 1}",
            )
        self.assertTrue(np.array_equal(shipped.flags, reference.flags))
        self.assertTrue(np.array_equal(shipped.f, reference.f))

    def test_a_prefilled_filter_starts_exactly_full(self) -> None:
        """
        ``create_state`` seeds mass at ``rho * porosity``, not ``rho``.

        Full is ``rho * porosity`` everywhere else in the solver -- in
        ``fluid_fraction``, in ``apply_transitions``' threshold, in the heal, in
        the stage's inlet top-up. Seeding a flat REFERENCE_DENSITY put a cell
        that *starts* FLUID inside a filter at eps = 1/0.85 = 1.176, so every
        porous fixture began 18 % over full and the exchange had to work it off.
        The real assembly starts empty and its filter cells arrive through
        ``apply_transitions``, which snaps them correctly -- this bit a restart
        and every fixture, not Test 2.
        """
        import physics.filling as pf
        from physics.filling import PorousZone

        shape = (5, 5, 8)
        obstacle = np.zeros(shape, dtype=bool)
        obstacle[0] = obstacle[-1] = True
        obstacle[:, 0] = obstacle[:, -1] = True
        obstacle[:, :, 0] = obstacle[:, :, -1] = True
        zone = PorousZone.inert(shape)
        band = np.zeros(shape, dtype=bool)
        band[1:4, 1:4, 3:6] = True
        zone.porosity[band] = np.float32(0.85)
        units = self._units()
        state = pf.create_state(
            obstacle,
            dx_m=self.DX_M,
            dt_s=units.dt_s,
            initial_fluid=~obstacle,
            porous=zone,
        )
        eps = pf.fluid_fraction(state)
        fluid = state.flags == pf.FLAG_FLUID
        np.testing.assert_allclose(eps[fluid & band], 1.0, rtol=1e-6)
        np.testing.assert_allclose(eps[fluid & ~band], 1.0, rtol=1e-6)

    def test_an_over_full_fluid_cell_aborts_the_run(self) -> None:
        """
        The precursor that would have named this defect where it happened.

        A FLUID cell's fill fraction is unconstrained *and*, until now,
        unobserved: the flag scans all select on FLAG_INTERFACE, so the only
        thing that ever wrote a FLUID cell's mass was the exchange and nothing
        checked what it wrote. On ``Test 2`` the first filter cell crossed
        eps = 1 at step 7,650 and the run surfaced as a NaN at 16,493 --
        8,800 steps of the mass ledger visibly wrong, reported as nothing.

        Poisoning one cell mid-fill stands in for that. The abort has to name
        the count, the part and the z-range, because "diverged" alone is what
        made the original take a session to locate.

        It also has to wait: the abort needs
        :data:`~core.filling_stage.FLUID_FILL_FRACTION_CHECKS` consecutive
        checks, so a single 200-step hydraulic transient cannot kill a correct
        run. Poisoning a FLUID cell is exactly the shape that does *not* clear
        on its own -- nothing writes a FLUID cell's mass but the exchange -- so
        the breach is still there at the next check and the run dies one check
        late, which is what this asserts.
        """
        import physics.filling as pf
        from core.filling_stage import (
            FLUID_FILL_FRACTION_CHECKS,
            FLUID_FILL_FRACTION_MAX,
            STATUS_DIVERGED,
            DIVERGENCE_CHECK_INTERVAL,
        )

        poison_at = 2 * DIVERGENCE_CHECK_INTERVAL
        abort_at = poison_at + (
            FLUID_FILL_FRACTION_CHECKS - 1
        ) * DIVERGENCE_CHECK_INTERVAL
        original = pf.step
        poisoned = []

        def traced(state, units, **kwargs):
            original(state, units, **kwargs)
            if state.step_count == poison_at and not poisoned:
                fluid = np.nonzero(state.flags == pf.FLAG_FLUID)
                if len(fluid[0]):
                    cell = tuple(int(a[0]) for a in fluid)
                    state.mass[cell] = 5.0
                    poisoned.append(cell)
            return state

        messages: list = []
        pf.step = traced
        try:
            result = _run(
                mesh=_walled_tower(nz=12), messages=messages,
                **_head_mode(max_fill_time_s=0.3),
            )
        finally:
            pf.step = original

        self.assertTrue(poisoned, "the fill never produced a FLUID cell to poison")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.status, STATUS_DIVERGED)
        self.assertEqual(
            result.diverged_at_step, abort_at,
            f"the guard fired at step {result.diverged_at_step} against "
            f"{abort_at}: it needs {FLUID_FILL_FRACTION_CHECKS} consecutive "
            f"checks, no more and no fewer",
        )
        aborts = [m for m in messages if "ABORTED" in m and "fill fraction" in m]
        self.assertTrue(aborts, f"the abort did not name the cause: {messages}")
        self.assertIn("1 FLUID cell holds", aborts[-1])
        self.assertIn(f"{FLUID_FILL_FRACTION_MAX:.1f}", aborts[-1])
        self.assertIn("cast", aborts[-1])
        self.assertIn("solve z", aborts[-1])

    def test_a_single_check_transient_does_not_abort(self) -> None:
        """
        The other half of the two-check rule: one bad window must not kill a
        run.

        For a FLUID cell ``eps`` is very nearly ``1 + drho/rho`` -- ``state.rho``
        is the start-of-step density and under Table 1 a FLUID cell's mass
        increment is exactly its own ``drho`` -- so the band is also a
        "no FLUID cell changed density by 10 % in one step" meter. A front
        free-falling into a closed gate is a real water hammer and is entitled
        to one such window; only a breach that is still there 200 steps later is
        the bookkeeping coming apart.

        Poisoned at one check and repaired before the next, so exactly one check
        sees it. With a one-check abort this run dies; with two it finishes.
        """
        import physics.filling as pf
        from core.filling_stage import DIVERGENCE_CHECK_INTERVAL, STATUS_DIVERGED

        poison_at = 2 * DIVERGENCE_CHECK_INTERVAL
        original = pf.step
        held: dict = {}

        def traced(state, units, **kwargs):
            original(state, units, **kwargs)
            if state.step_count == poison_at and "cell" not in held:
                fluid = np.nonzero(state.flags == pf.FLAG_FLUID)
                if len(fluid[0]):
                    cell = tuple(int(a[0]) for a in fluid)
                    held["cell"] = cell
                    held["mass"] = float(state.mass[cell])
                    state.mass[cell] = 5.0
            elif state.step_count == poison_at + 1 and "cell" in held:
                state.mass[held["cell"]] = held["mass"]
                held["repaired"] = True
            return state

        messages: list = []
        pf.step = traced
        try:
            result = _run(
                mesh=_walled_tower(nz=12), messages=messages,
                **_head_mode(max_fill_time_s=0.3),
            )
        finally:
            pf.step = original

        self.assertIn("cell", held, "the fill never produced a FLUID cell")
        self.assertTrue(held.get("repaired"), "the poison was never removed")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertNotEqual(
            result.status, STATUS_DIVERGED,
            f"a one-check transient aborted the run: {result.warnings}",
        )
        self.assertEqual(
            [m for m in messages if "fill fraction outside" in m], []
        )

    def test_a_healthy_fill_does_not_trip_the_fill_fraction_guard(self) -> None:
        """
        The band has to clear the two harmless excursions that live just
        outside [0, 1], or it is a guard that kills working pours.

        ``state.rho`` is the start-of-step density while ``mass`` is the
        end-of-step value, so every FLUID cell's reported eps is high by
        ``drho/rho`` -- measured at 1-2 % on several hundred cells of a healthy
        Test 2 run, plateauing, never growing. ``heal_submerged_interface``
        separately flags a cell full at slightly under full mass. Both are
        inside the band by a factor of five, and this asserts it on a fill that
        actually completes.
        """
        from core.filling_stage import STATUS_DIVERGED

        messages: list = []
        result = _run(
            mesh=_walled_tower(nz=12), messages=messages,
            **_head_mode(max_fill_time_s=0.3),
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertNotEqual(result.status, STATUS_DIVERGED, result.warnings)
        self.assertEqual(
            [m for m in messages if "fill fraction outside" in m], []
        )


class PeriodicWrapTests(unittest.TestCase):
    """
    Why the solve grid is padded: an unwalled array face makes the lattice a
    torus, on both step paths.

    Every neighbour access in the solver is periodic -- Numba ``% n``, NumPy
    ``np.roll`` -- and both are documented as relying on an obstacle shell to
    make the wrap unreachable (``physics.fluid_flow.stream_kernel``: "real
    domains are closed by obstacle cells, so the wrap never fires"). The stage
    used to hand them a grid sized exactly to the geometry's bounding box, where
    no such shell exists.

    These tests assert the *solver's* half of the contract rather than the
    stage's: given an unwalled face, the wrap fires and it is not a bug in the
    solver. The stage-side guarantee that no such grid is ever built is
    ``tests/test_filling_stage.py::MoldPadTests``.

    The controls are what redirect the diagnosis. An open top **alone** is
    completely clean -- the wrap needs metal on the *opposite* face -- which is
    why the fix could keep every riser vented instead of sealing riser tops.
    """

    DX_M = 0.004
    NU = 6.5e-7
    NZ = 14
    WIDTH = 3
    STEPS = 40

    def _channel(self, *, floor: bool, lid: bool):
        """
        A 3x3 channel walled in x and y, with the z faces optional.

        Metal is seeded at the *bottom*. The top plane stands in for a riser
        top: 12 planes of empty gas separate it from the metal, so any wet cell
        there arrived through the array edge.
        """
        import physics.filling as pf
        from physics.fluid_flow import lattice_units

        n = self.WIDTH + 2
        obstacle = np.ones((n, n, self.NZ), dtype=bool)
        obstacle[1:1 + self.WIDTH, 1:1 + self.WIDTH, :] = False
        if floor:
            obstacle[:, :, 0] = True
        if lid:
            obstacle[:, :, self.NZ - 1] = True
        fluid = np.zeros_like(obstacle)
        z_lo = 1 if floor else 0
        fluid[1:1 + self.WIDTH, 1:1 + self.WIDTH, z_lo:z_lo + 2] = True
        units = lattice_units(self.DX_M, self.NU, u_max_physical_m_s=3.0)
        state = pf.create_state(
            obstacle, dx_m=self.DX_M, dt_s=units.dt_s, initial_fluid=fluid
        )
        top = np.zeros_like(obstacle)
        top[:, :, self.NZ - 1] = True
        top &= ~obstacle
        return state, units, top

    def _first_top_wet_step(self, *, floor, lid, use_numba):
        """Step until the top plane wets, and report when and how many."""
        import physics.filling as pf

        state, units, top = self._channel(floor=floor, lid=lid)
        first, count = None, 0
        for s in range(1, self.STEPS + 1):
            pf.step(state, units, use_numba=use_numba)
            wet = (state.flags == pf.FLAG_FLUID) | (
                state.flags == pf.FLAG_INTERFACE
            )
            n = int(np.count_nonzero(wet & top))
            if first is None and n:
                first, count = s, n
        wet = (state.flags == pf.FLAG_FLUID) | (state.flags == pf.FLAG_INTERFACE)
        return first, count, int(np.count_nonzero(wet & top))

    def _both_paths(self):
        from physics.fluid_flow import NUMBA_AVAILABLE

        return (False, True) if NUMBA_AVAILABLE else (False,)

    def test_metal_on_both_z_faces_floods_the_far_face_immediately(self) -> None:
        """
        The mechanism, on both paths: 9 cells of the top plane are INTERFACE at
        step 1, 12 empty planes above the metal.

        ``mass = 0.5`` on them names the writer exactly -- that is
        ``mass[interface] = 0.5 * REFERENCE_DENSITY`` in ``create_state``'s
        initial interface layer, whose ``_dilate`` rolls the ``z = 0`` plane
        onto ``z = nz - 1``. On the real assembly the same ``_dilate`` is
        reached later, through ``apply_transitions``' promote: measured at step
        12,156, the step the front first occupied ``z = 0``, five RISER cells
        created at ``z = 93`` with every wet donor a wrapped cast-floor cell 92
        planes away.
        """
        for use_numba in self._both_paths():
            with self.subTest(numba=use_numba):
                first, count, _ = self._first_top_wet_step(
                    floor=False, lid=False, use_numba=use_numba
                )
                self.assertEqual(
                    first, 1,
                    "the far z face did not wet, so this fixture no longer "
                    "demonstrates the wrap the mold pad exists to remove",
                )
                self.assertEqual(count, self.WIDTH ** 2)

    def test_an_open_top_alone_is_clean(self) -> None:
        """
        The control that redirected the diagnosis.

        With an obstacle floor and the top still open the top plane stays GAS
        for the whole run: the wrap needs metal on the *opposite* face, so an
        open riser top is harmless by itself and the fix does not have to seal
        one. Under the mold pad the riser keeps its atmosphere boundary and
        ``_static_vent_mask`` keeps finding its vents, because the pad above a
        riser top is still ``EMPTY_CELL``.
        """
        for use_numba in self._both_paths():
            with self.subTest(numba=use_numba):
                first, _, final = self._first_top_wet_step(
                    floor=True, lid=False, use_numba=use_numba
                )
                self.assertIsNone(first, f"the top plane wet at step {first}")
                self.assertEqual(final, 0)

    def test_a_closed_channel_is_clean(self) -> None:
        """The second control: floor and lid both obstacle, nothing wraps."""
        for use_numba in self._both_paths():
            with self.subTest(numba=use_numba):
                first, _, final = self._first_top_wet_step(
                    floor=True, lid=True, use_numba=use_numba
                )
                self.assertIsNone(first, f"the top plane wet at step {first}")
                self.assertEqual(final, 0)

    def test_the_stage_never_builds_an_unwalled_grid(self) -> None:
        """
        The stage's side of the contract, on the fixture that used to be a
        torus.

        ``_tower`` is fully dense -- every cell metal, so ``_obstacle_mask``
        finds nothing solid and before the mold pad the lattice had no boundary
        at all. That is exactly the condition the three tests above measure, and
        it must now be unrepresentable: every face of the solved grid is
        obstacle, and gravity has something to press against.
        """
        import physics.filling as pf

        _, state = _final_state(dict(mesh=_tower(nz=8), max_fill_time_s=0.05))
        obstacle = state.flags == pf.FLAG_OBSTACLE
        for axis in (0, 1, 2):
            for face in (0, -1):
                plane = np.take(obstacle, face, axis=axis)
                self.assertTrue(
                    bool(plane.all()),
                    f"axis {axis} face {face} has "
                    f"{int(np.count_nonzero(~plane))} non-obstacle cells: the "
                    f"lattice is a torus",
                )


class GhostModeDampingTests(unittest.TestCase):
    """
    The regression test for the step-20,192 momentum blow-up.

    In TRT ``omega_plus`` sets the shear viscosity and ``omega_minus`` does not
    -- it enters only through ``Lambda = (1/omega_plus - 1/2)(1/omega_minus -
    1/2)``, which fixes the bounce-back wall position *and* how fast the odd,
    momentum-carrying "ghost" modes are damped. Holding ``Lambda`` at the
    wall-optimal 3/16 as ``tau_plus -> 1/2`` sends ``omega_minus -> 0``, and a
    casting lattice sits within 1e-5 of 1/2. Measured on ``Test 2``:
    ``tau_minus = 18,060`` steps, so over the ~200 steps a mode needs to double
    the odd part of ``f`` has no restoring term at all.

    What grew in that hole was a period-2 checkerboard on one lattice diagonal,
    in the *non-porous* runner cells one cell outside a filter's inlet face: the
    argmax of ``|u|`` alternating between two diagonal neighbours for 31
    consecutive steps without a break, ``u`` converging on ``(-a, +a, 0)``,
    growing from ``|u| = 0.016`` at step 19,000 to 0.94 at 20,180 and taking the
    field non-finite at 20,192. Neither of the two dissipation mechanisms the
    solver has could touch it: Smagorinsky senses the *symmetric* stress and is
    blind to an odd-parity mode, and the Guo-Zhao porous drag -- a strict
    contraction at every velocity -- is identically absent in the cells it grew
    in. See :data:`~physics.fluid_flow.TAU_MINUS_MAX`.

    These tests seed that mode directly rather than waiting for a jet to excite
    it, which is what makes them cheap and deterministic. Smagorinsky is off, so
    ``omega_minus`` is the only channel that can damp the seed and the test
    measures exactly the quantity the clamp restores.
    """

    DX_M = 0.0043          # Test 2's own fill grid
    NU = 0.006 / 7200.0
    SHAPE = (8, 8, 10)
    AMPLITUDE = 0.01       # a fifth of the 0.05 design velocity
    STEPS = 600

    def _closed_box(self, tau_minus_max):
        """A sealed box of FLUID on Test 2's lattice, at the requested clamp."""
        import physics.filling as pf
        from physics.fluid_flow import lattice_units

        obstacle = np.zeros(self.SHAPE, dtype=bool)
        obstacle[0] = obstacle[-1] = True
        obstacle[:, 0] = obstacle[:, -1] = True
        obstacle[:, :, 0] = obstacle[:, :, -1] = True
        units = lattice_units(
            self.DX_M, self.NU, u_max_physical_m_s=2.8,
            tau_minus_max=tau_minus_max,
        )
        state = pf.create_state(
            obstacle, dx_m=self.DX_M, dt_s=units.dt_s, initial_fluid=~obstacle
        )
        self.assertEqual(
            int(np.count_nonzero(state.flags == pf.FLAG_INTERFACE)), 0,
            "the box must be sealed, so only the momentum path is under test",
        )
        return state, units, ~obstacle

    def _seed_checkerboard(self, state, wet):
        """
        Put the mode in by hand: ``u = (+a, -a, 0)`` flipping with ``(x + y)``
        parity, which is the ``(+1, -1, 0)`` diagonal the real mode chose.

        Returns the sign pattern, so the mode's amplitude can be projected back
        out of ``state.u`` afterwards instead of read off a global maximum -- a
        global maximum cannot separate this mode from the mean flow, which is
        exactly why it went unseen for 5,000 steps on the real assembly.
        """
        from physics.fluid_flow import equilibrium

        grid = np.meshgrid(*[np.arange(n) for n in self.SHAPE], indexing="ij")
        sign = np.where((grid[0] + grid[1]) % 2 == 0, 1.0, -1.0)
        u = np.zeros((3,) + self.SHAPE, dtype=np.float64)
        u[0] = self.AMPLITUDE * sign * wet
        u[1] = -self.AMPLITUDE * sign * wet
        seeded = equilibrium(np.ones(self.SHAPE, dtype=np.float64), u)
        state.f = np.where(
            wet, seeded, np.asarray(state.f, dtype=np.float64)
        ).astype(np.float32)
        return sign

    @staticmethod
    def _amplitude(state, wet, sign):
        """The staggered mode's amplitude, projected out of ``state.u``."""
        u = np.asarray(state.u, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            return float(np.mean((sign * (u[0] - u[1]))[wet]) * 0.5)

    def _trajectory(self, *, use_numba, tau_minus_max):
        """
        ``(seed amplitude, [(step, amplitude), ...])`` for one clamp.

        The clamp travels on ``units`` and nowhere else. It used to need a
        monkeypatch of ``physics.filling.TAU_MINUS_MAX`` alongside the
        ``lattice_units`` argument, because the kernel and the Smagorinsky path
        read the module global while only the non-Smagorinsky NumPy branch read
        ``units`` -- so asking for a non-default ceiling moved one of the three
        relaxation sites and silently put the two step paths on different
        schemes. ``LatticeUnits.tau_minus_max`` is now the single authority and
        the workaround is gone.
        """
        import physics.filling as pf

        state, units, wet = self._closed_box(tau_minus_max)
        self.assertEqual(units.tau_minus_max, tau_minus_max)
        sign = self._seed_checkerboard(state, wet)
        kwargs = dict(
            use_numba=use_numba, track_free_surface=False, smagorinsky=False
        )
        with np.errstate(over="ignore", invalid="ignore"):
            pf.step(state, units, **kwargs)
            seed = self._amplitude(state, wet, sign)
            samples = []
            for k in range(1, self.STEPS + 1):
                pf.step(state, units, **kwargs)
                if k % 150 == 0:
                    samples.append((k, self._amplitude(state, wet, sign)))
        return seed, samples

    def _both_paths(self):
        from physics.fluid_flow import NUMBA_AVAILABLE

        return (False, True) if NUMBA_AVAILABLE else (False,)

    def test_the_shipped_clamp_damps_the_mode(self) -> None:
        """
        Seeded at 0.01 lattice, the mode decays and stays decayed.

        Measured: 0.43 of the seed by step 150 and 0.37 by step 600, monotone,
        finite throughout, identically on both paths. ``tau_minus = 5`` gives
        the odd moments a five-step relaxation time, which is what a period-2
        mode needs to see.
        """
        from physics.fluid_flow import TAU_MINUS_MAX

        self.assertEqual(TAU_MINUS_MAX, 5.0, "the clamp value moved")
        for use_numba in self._both_paths():
            with self.subTest(numba=use_numba):
                seed, samples = self._trajectory(
                    use_numba=use_numba, tau_minus_max=TAU_MINUS_MAX
                )
                self.assertAlmostEqual(seed, self.AMPLITUDE, delta=2e-3)
                ratios = [a / seed for _, a in samples]
                self.assertTrue(
                    all(np.isfinite(r) for r in ratios),
                    f"the mode went non-finite: {samples}",
                )
                self.assertLess(
                    max(ratios), 0.6,
                    f"the mode was not damped: amplitude ratios {ratios}",
                )
                self.assertLess(
                    ratios[-1], ratios[0] + 1e-9,
                    f"the mode stopped decaying: {ratios}",
                )

    def test_without_the_clamp_the_mode_survives_and_detonates(self) -> None:
        """
        The discriminator, so a passing test above cannot be a passing fixture.

        At ``Lambda = 3/16`` exactly -- ``tau_minus = 18,060`` on this lattice
        -- the same seed does not decay at all: it holds above its seed
        amplitude through step 450 and the field is **non-finite by step 600**,
        on both paths. That is the mechanism, in 600 steps of an 8x8x10 box,
        with no jet and no filter needed to excite it.
        """
        for use_numba in self._both_paths():
            with self.subTest(numba=use_numba):
                seed, samples = self._trajectory(
                    use_numba=use_numba, tau_minus_max=float("inf")
                )
                ratios = [a / seed for _, a in samples]
                grew = max((r for r in ratios if np.isfinite(r)), default=0.0)
                detonated = any(not np.isfinite(r) for r in ratios)
                self.assertTrue(
                    grew > 1.0 or detonated,
                    f"the unclamped scheme damped the mode too, so this "
                    f"fixture no longer discriminates: ratios {ratios}",
                )

    def test_the_units_carry_the_clamp_into_both_step_paths(self) -> None:
        """
        ``lattice_units(..., tau_minus_max=inf)`` really does reach the solver
        -- in both paths, and with Smagorinsky in its production default.

        This is the regression test for a seam, not for the physics. The clamp
        is applied at three places: ``omega_minus_from_magic`` (which fills
        ``units.omega_minus``, read only when Smagorinsky is off),
        ``smagorinsky_relaxation`` (the production NumPy path) and inlined in
        ``collide_kernel`` (both Numba paths). The latter two used to read the
        module global ``TAU_MINUS_MAX``, so ``lattice_units``' argument reached
        one site out of three: the documented ``float("inf")`` escape hatch did
        nothing in the default configuration, and asking for it with
        ``smagorinsky=False`` put the NumPy path on ``omega_minus = 5.5e-5``
        while the kernel stayed at 0.2 -- a 3,600x split between paths that
        ``tests/test_filling_parity.py`` cannot see, because its own fixture
        uses the default ceiling and so both paths clamp and agree.

        Measured by stepping the same seeded state under each ceiling and
        differencing the populations: at ``inf`` the two ceilings must disagree
        (or the ceiling is not reaching the solver at all), and at each ceiling
        the two *paths* must agree.
        """
        import copy

        import physics.filling as pf
        from physics.fluid_flow import (
            NUMBA_AVAILABLE, TAU_MINUS_MAX, lattice_units,
            omega_minus_from_magic,
        )

        def units_at(ceiling):
            return lattice_units(
                self.DX_M, self.NU, u_max_physical_m_s=2.8,
                tau_minus_max=ceiling,
            )

        clamped, unclamped = units_at(TAU_MINUS_MAX), units_at(float("inf"))
        # The two ceilings really are different schemes on this lattice.
        self.assertAlmostEqual(1.0 / clamped.omega_minus, TAU_MINUS_MAX, places=12)
        self.assertGreater(1.0 / unclamped.omega_minus, 18_000.0)
        self.assertEqual(
            unclamped.omega_minus,
            omega_minus_from_magic(
                unclamped.omega_plus, tau_minus_max=float("inf")
            ),
        )

        def stepped(units, *, use_numba, smagorinsky):
            state, _, wet = self._closed_box(units.tau_minus_max)
            self._seed_checkerboard(state, wet)
            for _ in range(4):
                pf.step(
                    state, units, use_numba=use_numba,
                    track_free_surface=False, smagorinsky=smagorinsky,
                )
            return np.asarray(state.f, dtype=np.float64)

        paths = (False, True) if NUMBA_AVAILABLE else (False,)
        for smagorinsky in (True, False):
            for use_numba in paths:
                with self.subTest(smagorinsky=smagorinsky, numba=use_numba):
                    a = stepped(clamped, use_numba=use_numba,
                                smagorinsky=smagorinsky)
                    b = stepped(unclamped, use_numba=use_numba,
                                smagorinsky=smagorinsky)
                    self.assertFalse(
                        np.allclose(a, b, rtol=1e-9, atol=1e-12),
                        "the ceiling on units made no difference to the "
                        "populations, so this path is reading the clamp from "
                        "somewhere other than the units it was handed",
                    )

        if not NUMBA_AVAILABLE:
            self.skipTest("numba is not installed; path agreement not checkable")
        for ceiling, label in ((TAU_MINUS_MAX, "clamped"),
                               (float("inf"), "unclamped")):
            for smagorinsky in (True, False):
                with self.subTest(ceiling=label, smagorinsky=smagorinsky):
                    units = units_at(ceiling)
                    np.testing.assert_allclose(
                        stepped(units, use_numba=False,
                                smagorinsky=smagorinsky),
                        stepped(units, use_numba=True,
                                smagorinsky=smagorinsky),
                        rtol=1e-5, atol=1e-7,
                        err_msg=(
                            f"the two step paths disagree at "
                            f"tau_minus_max={label}: one of them is not "
                            f"reading units.tau_minus_max"
                        ),
                    )

    def test_both_paths_agree_on_the_clamped_trajectory(self) -> None:
        """
        The clamp is inlined in the kernel and applied by
        ``omega_minus_from_magic`` in the NumPy path, so a one-sided edit is
        possible and would be invisible to the amplitude bounds above.

        ``tests/test_filling_parity.py`` covers this generally -- its fixture
        runs at ``tau = 0.500003``, deep inside the clamp -- but it does not
        seed this mode, so the agreement is pinned here too.
        """
        from physics.fluid_flow import NUMBA_AVAILABLE, TAU_MINUS_MAX

        if not NUMBA_AVAILABLE:
            self.skipTest("numba is not installed")
        seed_np, numpy_samples = self._trajectory(
            use_numba=False, tau_minus_max=TAU_MINUS_MAX
        )
        seed_nb, numba_samples = self._trajectory(
            use_numba=True, tau_minus_max=TAU_MINUS_MAX
        )
        self.assertAlmostEqual(seed_np, seed_nb, places=6)
        for (step, a), (_, b) in zip(numpy_samples, numba_samples):
            with self.subTest(step=step):
                self.assertAlmostEqual(a, b, places=6)



class NegativeCascadePersistenceTests(unittest.TestCase):
    """
    Requirement: one bad sample must not abort a run; two consecutive must.

    The cascade trigger is ``>= max(8 cells, 2 % of wet cells)``, and the
    8-cell floor was chosen against models with tens of thousands of wet cells,
    where 8 is negligible. On a small model it is not: at ~400 wet cells the
    2 % fraction is 8 cells too, so both halves are satisfied by 8 cells -- and
    the *first* divergence check lands at step 200, while a splat rim still
    routinely carries that many transient negatives. One sample was enough to
    discard a run that was about to be fine.

    Two consecutive checks, the same shape as
    :data:`~core.filling_stage.FLUID_FILL_FRACTION_CHECKS`, rather than a
    warm-up window: a warm-up blinds the guard for its first N steps, and a
    genuine cascade at step 200 is exactly as fatal as one at step 20,000. It
    should still abort, 200 steps later, rather than not at all.

    The negatives are injected rather than provoked, so what is under test is
    the guard's persistence logic and not a solver trajectory that might or
    might not produce them.
    """

    def _run_with_injected_negatives(self, at_steps):
        """
        Drive the real stage, forcing negative populations at chosen steps.

        Returns ``(result, messages, samples)`` where each sample records the
        wet-cell count and how many cells were made negative, so the test can
        assert the trigger's arithmetic was actually satisfied.
        """
        import physics.filling as pf
        from core.filling_stage import (
            DIVERGENCE_CHECK_INTERVAL,
            NEGATIVE_POPULATION_CASCADE_FRACTION,
            NEGATIVE_POPULATION_CASCADE_MIN_CELLS,
        )

        samples: list = []
        original = pf.step

        def traced(state, units, **kwargs):
            original(state, units, **kwargs)
            step_index = int(state.step_count)
            wet = (state.flags == pf.FLAG_FLUID) | (
                state.flags == pf.FLAG_INTERFACE
            )
            if step_index in at_steps:
                ix, iy, iz = np.nonzero(wet)
                # Exactly the trigger's own threshold, so the test cannot pass
                # by injecting too few cells to fire it.
                want = max(
                    NEGATIVE_POPULATION_CASCADE_MIN_CELLS,
                    int(np.ceil(
                        NEGATIVE_POPULATION_CASCADE_FRACTION * max(ix.size, 1)
                    )),
                )
                take = min(want, ix.size)
                state.f[0, ix[:take], iy[:take], iz[:take]] = np.float32(-1e-6)
                samples.append(
                    {"step": step_index, "wet": int(ix.size), "negative": take}
                )
            elif step_index - 1 in at_steps:
                # Clear them again, so only the checks in ``at_steps`` see a
                # cascade. Without this the injection would persist and every
                # later check would count it.
                state.f[0][state.f[0] < 0.0] = np.float32(1e-9)
            return state

        messages: list = []
        pf.step = traced
        try:
            result = _run(
                mesh=_walled_tower(nz=12, width=5), messages=messages,
                **_head_mode(max_fill_time_s=0.2),
            )
        finally:
            pf.step = original
        return result, messages, samples

    def _cascade_aborts(self, messages):
        return [
            m for m in messages
            if "ABORTED" in m and "cascade in progress" in m
        ]

    def test_one_check_of_negatives_does_not_abort(self) -> None:
        """
        The regression: a single 200-step sample, cleared before the next
        check, must leave the run alone.
        """
        from core.filling_stage import (
            DIVERGENCE_CHECK_INTERVAL, STATUS_DIVERGED,
        )

        result, messages, samples = self._run_with_injected_negatives(
            {DIVERGENCE_CHECK_INTERVAL}
        )
        self.assertEqual(len(samples), 1, "the injection never happened")
        self.assertGreaterEqual(samples[0]["negative"], 8)
        # The regime the finding is about: a small model, where the 8-cell
        # floor is a large share of the wet set rather than a rounding error.
        self.assertLess(
            samples[0]["wet"], 500,
            "the fixture grew out of the small-model regime this guards",
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertNotEqual(
            result.status, STATUS_DIVERGED,
            f"one sample of {samples[0]['negative']} negative cells out of "
            f"{samples[0]['wet']} wet aborted the run: {result.warnings}",
        )
        self.assertEqual(self._cascade_aborts(messages), [])

    def test_two_consecutive_checks_do_abort(self) -> None:
        """
        And the guard must still work. Same fixture, same injection, held for
        one more check -- the run dies at the second one, naming the
        persistence so the message cannot be read as a single-sample abort.
        """
        from core.filling_stage import (
            DIVERGENCE_CHECK_INTERVAL,
            NEGATIVE_POPULATION_CASCADE_CHECKS,
            STATUS_DIVERGED,
        )

        checks = {
            DIVERGENCE_CHECK_INTERVAL * (k + 1)
            for k in range(NEGATIVE_POPULATION_CASCADE_CHECKS)
        }
        result, messages, samples = self._run_with_injected_negatives(checks)
        self.assertEqual(len(samples), NEGATIVE_POPULATION_CASCADE_CHECKS)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.status, STATUS_DIVERGED)
        self.assertEqual(
            result.diverged_at_step,
            DIVERGENCE_CHECK_INTERVAL * NEGATIVE_POPULATION_CASCADE_CHECKS,
            "the abort did not land on the last of the consecutive checks",
        )
        aborts = self._cascade_aborts(messages)
        self.assertTrue(aborts, f"the cascade guard never fired: {messages}")
        self.assertIn("consecutive checks", aborts[-1])

class StatusReportingTests(unittest.TestCase):
    """
    "Could not get there", "needed longer" and "the answer is garbage" used to
    surface identically, as a misrun count.
    """

    def test_a_full_fill_reports_complete(self) -> None:
        """A cavity that genuinely fills reports ``complete`` and no misrun."""
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


class SubmergedInterfaceHealingTests(unittest.TestCase):
    """
    The regression test for the frozen submerged interface cell.

    Under Koerner's mass-exchange weighting -- see
    :class:`FluidMassTracksDensityTests` -- an interface cell whose neighbours
    are all wet or solid exchanges at weight 1 in every direction, so its mass
    change is exactly its own density change and ``eps`` freezes wherever the
    passing front left it. It can never reach the transition threshold, so
    without a flag rule to convert it, it is stranded for the rest of the run.

    Measured on ``_walled_tower(nz=4, width=4)`` before
    :func:`~physics.filling.heal_submerged_interface` existed: the run hit the
    2 s cap after 10,189 steps with 40 cells FLUID and 24 INTERFACE, **all
    twenty-four** of them with no gas neighbour left, at eps between 0.02 and
    0.99 -- reported as ``time_capped`` with 24 misruns, 37.5 % of the metal,
    in a cavity that was physically full. On ``Test_Model/Full_Gating_Test``
    the same freeze stranded 796 cells, 8.6 % of the metal.

    That is a gap in the flag bookkeeping, not in the weighting: Koerner's
    scheme reinitialises a gasless interface cell to FLUID and this
    implementation had no such rule. It was invisible while the old weighting
    let FLUID mass drift upward past the transition threshold and convert
    those cells anyway -- and that drift is the +-70,000 runaway. Removing the
    drift exposed the gap; it did not cause it.
    """

    def test_a_shallow_walled_tower_fills_completely(self) -> None:
        """
        The discriminating case. With the heal removed this fails on every
        assertion below: ``time_capped``, 24 misruns, 24 gasless cells.
        """
        result, state = _final_state(dict(
            mesh=_walled_tower(nz=4, width=4),
            **_head_mode(max_fill_time_s=2.0),
        ))
        self.assertIsNotNone(result)
        self.assertEqual(result.status, STATUS_COMPLETE)
        self.assertEqual(result.misrun_voxel_count, 0)

    def test_no_interface_cell_is_left_without_a_gas_neighbour(self) -> None:
        """
        The mechanism, asserted directly rather than through a status string.

        A status can come out right for the wrong reason; a gasless interface
        cell is the thing that cannot happen. Measured before the heal: 24 of
        24 surviving interface cells on this fixture had no gas neighbour.
        """
        _, state = _final_state(dict(
            mesh=_walled_tower(nz=4, width=4),
            **_head_mode(max_fill_time_s=2.0),
        ))
        stranded = _gasless_interface_cells(state)
        self.assertEqual(
            int(np.count_nonzero(stranded)), 0,
            f"{int(np.count_nonzero(stranded))} interface cells were left "
            f"submerged, which no fill level can ever convert",
        )

    def test_a_deeper_tower_leaves_none_either(self) -> None:
        """The same, on a fixture deep enough to have a running front."""
        result, state = _final_state(dict(
            mesh=_walled_tower(nz=12),
            **_head_mode(max_fill_time_s=1.5),
        ))
        self.assertEqual(result.status, STATUS_COMPLETE)
        self.assertEqual(
            int(np.count_nonzero(_gasless_interface_cells(state))), 0
        )

    def test_a_submerged_cell_converts_and_pays_its_neighbours(self) -> None:
        """
        A cell built submerged by hand, on a lattice small enough that the
        bookkeeping can be read off directly.

        The cell is under-full, so what it hands :func:`_redistribute` is a
        *deficit*: the neighbouring interface cells give the metal up rather
        than receive it. Measured here: the healed cell goes from 0.40 to a
        full 1.0 and its single interface neighbour from 0.80 to 0.20, an
        exchange of exactly 0.60, with the domain total unchanged to the last
        bit.
        """
        import physics.filling as pf

        state, units = _submerged_pair()
        neighbour_before = float(state.mass[5, 5, 6])
        total_before = pf.total_mass(state)

        healed = pf.heal_submerged_interface(state, 0.0, dwell_steps=1)

        self.assertEqual(healed, 1, "the submerged cell was not converted")
        self.assertEqual(int(state.flags[5, 5, 5]), int(pf.FLAG_FLUID))
        self.assertAlmostEqual(
            float(state.mass[5, 5, 5]), float(state.rho[5, 5, 5]), places=6,
            msg="a healed cell was not snapped to a full cell's mass",
        )
        self.assertEqual(
            int(state.flags[5, 5, 6]), int(pf.FLAG_INTERFACE),
            "the cell that still has a gas neighbour must not be healed",
        )
        self.assertAlmostEqual(
            float(state.mass[5, 5, 6]) - neighbour_before, -0.60, places=9,
            msg="the deficit did not land on the neighbouring interface cell",
        )
        self.assertAlmostEqual(
            pf.total_mass(state), total_before, places=9,
            msg="healing moved mass in or out of the domain",
        )

    def test_it_converts_within_a_dwell_of_a_real_run(self) -> None:
        """
        The same cell, reached through :func:`~physics.filling.step`.

        The dwell (:data:`~physics.filling.HEAL_DWELL_STEPS`) holds the rule
        back until the cell has been gasless for that many consecutive steps,
        so through the real loop conversion lands just after the dwell -- a
        permanently stranded cell only ever pays it once.
        """
        import physics.filling as pf

        state, units = _submerged_pair()
        for _ in range(pf.HEAL_DWELL_STEPS + 5):
            pf.step(state, units)
            if int(state.flags[5, 5, 5]) == int(pf.FLAG_FLUID):
                break
        self.assertEqual(
            int(state.flags[5, 5, 5]), int(pf.FLAG_FLUID),
            "a submerged interface cell survived a full dwell",
        )

    def test_a_transient_submersion_never_heals(self) -> None:
        """
        The dwell's whole point: a cell whose submersion is interrupted loses
        its streak and starts over, so a sloshing front is never healed.
        Measured before the dwell existed: 828,368 heal events over 327,200
        steps on Test_Model/Filling_Test (~190 per cell) and 34% less metal
        delivered -- the rule was fighting the retreat path for the front.
        """
        import physics.filling as pf

        state, _ = _submerged_pair()
        for _ in range(4):
            self.assertEqual(pf.heal_submerged_interface(state, 0.0, dwell_steps=5), 0)
        # Re-open the surface beside it for one step: the streak must reset.
        was = np.uint8(state.flags[5, 5, 4])
        state.flags[5, 5, 4] = pf.FLAG_GAS
        self.assertEqual(pf.heal_submerged_interface(state, 0.0, dwell_steps=5), 0)
        state.flags[5, 5, 4] = was
        # Four more consecutive gasless steps still do not reach the dwell.
        for _ in range(4):
            self.assertEqual(pf.heal_submerged_interface(state, 0.0, dwell_steps=5), 0)
        self.assertEqual(int(state.flags[5, 5, 5]), int(pf.FLAG_INTERFACE))
        # The fifth consecutive one does.
        self.assertEqual(pf.heal_submerged_interface(state, 0.0, dwell_steps=5), 1)
        self.assertEqual(int(state.flags[5, 5, 5]), int(pf.FLAG_FLUID))

    def test_it_leaves_a_cell_with_a_gas_neighbour_alone(self) -> None:
        """The rule keys on topology, not on fill level."""
        import physics.filling as pf

        state, _ = _submerged_pair()
        state.mass[5, 5, 5] = 0.40
        before = state.mass[5, 5, 6]
        # Open the submerged cell to the cavity as well.
        state.flags[5, 5, 4] = pf.FLAG_GAS
        state.mass[5, 5, 4] = 0.0
        state.f[:, 5, 5, 4] = 0.0
        self.assertEqual(pf.heal_submerged_interface(state, 0.0, dwell_steps=1), 0)
        self.assertEqual(int(state.flags[5, 5, 5]), int(pf.FLAG_INTERFACE))
        self.assertEqual(float(state.mass[5, 5, 6]), float(before))

    def test_a_pocket_with_no_recipient_strands_rather_than_leaks(self) -> None:
        """
        The one case ``_redistribute`` cannot balance, measured rather than
        assumed.

        When a whole submerged pocket heals at once there is no interface cell
        left next to it to take the deficit -- which is the common case, not a
        corner one. ``_redistribute`` strands the deficit back on the donor, so
        nothing is created and nothing is lost: the price is a FLUID cell
        holding less than a full cell's metal, and the measured leakage is
        exactly zero. Inventing the missing metal instead would be a leak
        dressed up as a fill.
        """
        import physics.filling as pf

        state, _ = _closed_box_state()
        state.flags[5, 5, 5] = pf.FLAG_INTERFACE
        state.mass[5, 5, 5] = 0.40
        before = pf.total_mass(state)

        self.assertEqual(pf.heal_submerged_interface(state, 0.0, dwell_steps=1), 1)
        self.assertEqual(int(state.flags[5, 5, 5]), int(pf.FLAG_FLUID))
        self.assertEqual(
            float(state.mass[5, 5, 5]), 0.40,
            "the stranded deficit was not returned to the cell it came from",
        )
        self.assertEqual(
            pf.total_mass(state), before,
            f"healing leaked {pf.total_mass(state) - before:+.3e} lattice mass",
        )

    def test_conversions_decay_rather_than_plateau(self) -> None:
        """
        Ping-pong check. A healed cell is FLUID, and only a genuinely draining
        neighbour sends it back to the interface through
        ``apply_transitions``' retreat path -- so the count must fall away, not
        settle onto a per-step floor.

        Measured on a closed box seeded with a 27-cell submerged pocket: zero
        conversions through the dwell, all 27 in one burst the step it
        expires, and zero on every step after -- once per cell, ever, is the
        contract.
        """
        import physics.filling as pf

        state, units = _closed_box_state()
        pocket = np.zeros(state.shape, dtype=bool)
        pocket[4:7, 4:7, 4:7] = True
        pocket &= state.flags == pf.FLAG_FLUID
        state.flags[pocket] = pf.FLAG_INTERFACE
        rng = np.random.default_rng(0)
        state.mass[pocket] = rng.uniform(0.05, 0.9, int(np.count_nonzero(pocket)))

        counts: list[int] = []
        original = pf.heal_submerged_interface

        def traced(s, t, **kw):
            healed = original(s, t, **kw)
            counts.append(healed)
            return healed

        pf.heal_submerged_interface = traced
        try:
            for _ in range(pf.HEAL_DWELL_STEPS + 100):
                pf.step(state, units)
        finally:
            pf.heal_submerged_interface = original

        dwell = pf.HEAL_DWELL_STEPS
        self.assertEqual(
            sum(counts[: dwell - 1]), 0,
            "healing fired before the dwell expired",
        )
        self.assertEqual(counts[dwell - 1], int(np.count_nonzero(pocket)))
        self.assertEqual(
            sum(counts[dwell:]), 0,
            f"healing kept firing after the pocket closed: {counts[dwell:dwell+20]}",
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


def _plate_welded_to_a_block():
    """
    The masking case, in miniature.

    A wide, thin gate plate -- 11 x 14 cells across and **two** thick, the
    shape the ``Test 2`` gates take on their solved 8.6 mm grid -- fed by a
    fat runner block, with the gate's own root swelling into a lump where the
    two meet. That lump is the whole problem: it belongs to the gate, so a
    per-part maximum reads the lump and calls the plate 3.5 cells thick.
    """
    dense = np.full((20, 18, 14), EMPTY_CELL, dtype=np.int8)
    dense[1:8, 5:12, 3:10] = PART_TYPE_RUNNER      # 7 x 7 x 7 junction block
    dense[8:19, 2:16, 6:8] = PART_TYPE_GATE        # 11 x 14 x 2 plate
    dense[8:11, 7:10, 4:10] = PART_TYPE_GATE       # the fat root, 3 x 3 x 6
    return dense


def _generous_channel():
    """A runner and gate both five cells across -- resolved, and silent."""
    dense = np.full((16, 12, 12), EMPTY_CELL, dtype=np.int8)
    dense[1:8, 3:8, 3:8] = PART_TYPE_RUNNER
    dense[8:15, 3:8, 3:8] = PART_TYPE_GATE
    return dense


class ChokeMaskingTests(unittest.TestCase):
    """
    The regression test for a check that measured the right quantity and then
    took the wrong end of its distribution.

    On ``Test 2`` at its solved 8.6 mm grid the gates voxelize to 11 x 13 x 2
    plates -- sealed -- and the stage reported 3.00 cells and said nothing,
    because one fat cell where each gate met the runner set the maximum for
    the whole plate. 85 % of those gate cells had a single cell of wall
    clearance.
    """

    def setUp(self) -> None:
        from core.filling_stage import (
            measure_choke_thickness_by_part,
            narrowest_choke_part_code,
        )

        self.dense = _plate_welded_to_a_block()
        self.by_part = measure_choke_thickness_by_part(self.dense)
        self.narrowest = narrowest_choke_part_code(self.by_part)

    def test_the_block_really_does_mask_the_plate(self) -> None:
        """
        Without this the rest of the class could pass on a fixture that had
        no fat junction to be fooled by. The old statistic was a per-part
        maximum over the distance transform; compute it here and watch it
        wave the plate through.
        """
        from scipy import ndimage

        flow = np.isin(self.dense, (PART_TYPE_RUNNER, PART_TYPE_GATE))
        distance = ndimage.distance_transform_edt(np.pad(flow, 1))[1:-1, 1:-1, 1:-1]
        old_gate_thickness = float(distance[self.dense == PART_TYPE_GATE].max()) * 2 - 1
        self.assertGreaterEqual(old_gate_thickness, 3.0)

    def test_the_robust_measure_reports_the_plate(self) -> None:
        """
        Two cells across reads as 1.0: ``distance_transform_edt`` measures the
        gap to the wall, not the metal, so the measure is ``2*d - 1`` -- exact
        for odd thicknesses, one cell conservative for even ones. What matters
        is that it is the plate being reported and not the block.
        """
        self.assertEqual(self.by_part[PART_TYPE_GATE], 1.0)
        self.assertLessEqual(self.by_part[PART_TYPE_GATE], MIN_CHOKE_THICKNESS_CELLS)

    def test_the_fat_junction_is_not_reported_as_thin(self) -> None:
        """The 7-cell block is 7 cells; robustness must cut only one way."""
        self.assertGreaterEqual(self.by_part[PART_TYPE_RUNNER], 5.0)

    def test_the_narrowest_part_is_named(self) -> None:
        self.assertEqual(self.narrowest, PART_TYPE_GATE)

    def test_the_scalar_summary_is_the_thinnest_part(self) -> None:
        self.assertEqual(
            measure_choke_thickness_cells(self.dense),
            min(self.by_part.values()),
        )

    def test_a_generous_channel_measures_five_cells(self) -> None:
        from core.filling_stage import measure_choke_thickness_by_part

        by_part = measure_choke_thickness_by_part(_generous_channel())
        for code, thickness in by_part.items():
            with self.subTest(part=code):
                self.assertGreaterEqual(thickness, 5.0)


class ChokeWarningTests(unittest.TestCase):
    """
    Whether the user hears about it, through the channel the Foseco filter
    warning already uses.
    """

    def _run_dense(self, dense):
        """Drive the real stage over an arbitrary dense part-type grid."""
        filled = dense != EMPTY_CELL
        centers = np.asarray(np.nonzero(filled)).T.astype(np.float64) * DX
        mesh = _FakeMesh(centers, dense[filled], DX)
        mesh._top_z = float(centers[:, 2].max())
        highest = centers[int(np.argmax(centers[:, 2]))]
        mesh._pour_xy = (float(highest[0]), float(highest[1]))
        return _run(mesh=mesh, max_fill_time_s=0.02)

    def test_a_two_cell_gate_warns_and_names_the_part(self) -> None:
        result = self._run_dense(_plate_welded_to_a_block())
        self.assertIsNotNone(result)
        self.assertEqual(result.narrowest_choke_part, "gate")
        self.assertEqual(result.choke_thickness_by_part["gate"], 1.0)
        hard = [w for w in result.warnings if "cells thick" in w]
        self.assertTrue(hard, f"no choke warning raised: {result.warnings}")
        self.assertIn("gate", hard[0])
        self.assertIn("cannot be trusted", hard[0])

    def test_a_five_cell_channel_says_nothing(self) -> None:
        """
        Neither the hard warning nor the marginal note. ``Full_Gating_Test``
        at its shipped defaults measures 5, 7 and 5 cells across its sprue,
        runner and gate, and a warning there would be noise.
        """
        result = self._run_dense(_generous_channel())
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.choke_thickness_cells, 5.0)
        self.assertFalse(
            [w for w in result.warnings if "cells thick" in w or "cells across" in w],
            f"warned about a five-cell channel: {result.warnings}",
        )

    def test_the_choke_area_is_the_grid_that_was_solved(self) -> None:
        """
        The stage used to report the *mesh* grid's choke area next to results
        produced on the coarsened one. On ``Test 2`` those are 2.98 and
        7.46 cm^2 -- not a rounding difference, and the Foseco filter ratio is
        checked against this number.

        A 5 x 5 column, so the two grids genuinely disagree: 25 fine cells
        against 9 coarse ones of four times the area, 36/25 rather than 1.
        """
        mesh = _tower(nx=5, ny=5, nz=8, part_type=PART_TYPE_GATE)
        result = _run(mesh=mesh, max_fill_time_s=0.02, fill_grid_coarsening=2)
        self.assertIsNotNone(result)
        self.assertEqual(result.coarsening, 2)
        self.assertGreater(result.choke_area_cm2, result.choke_area_fine_cm2)
        self.assertAlmostEqual(
            result.choke_area_cm2 / result.choke_area_fine_cm2, 36.0 / 25.0, places=6
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

        Measured: ``complete`` in 1.10 s, every metal cell FLUID, 0.6 lb
        delivered against a 0.73 lb tower.

        **The fixture changed from ``_tower`` to ``_walled_tower`` when the
        solve grid gained its mold pad**
        (:data:`~core.filling_stage.MOLD_PAD_CELLS`), and the cap with it,
        0.5 s -> 2.0 s. ``_tower`` is a fully dense 2x2 column: before the pad
        ``_obstacle_mask`` found nothing solid, so it was periodic in x and y as
        well as z and the "fill" partly recirculated its own metal through the
        domain edge. The pad does not just give it a floor -- it wraps a 2x2
        channel in halfway bounce-back, which is the geometry the stage itself
        refuses to trust (``MIN_CHOKE_THICKNESS_CELLS``), and the 4.4x slowdown
        is what creep through a 2-cell no-slip channel costs, not what removing
        a recirculation path costs. ``_walled_tower`` is the fixture that was
        already written for this: a 3-cell channel inside a mold block, walled
        by cores rather than by the pad, so the pad changes nothing about it and
        the pour it measures is one the production code will stand behind.

        The full assertion is back. It was weakened to "84 % of cells" for one
        round, when the corrected Koerner pair weighting stopped FLUID mass
        drifting past the transition threshold and the 13 cells the front had
        closed over were left frozen as submerged interface -- a gap in the
        flag bookkeeping, not in the weighting. Now that
        :func:`~physics.filling.heal_submerged_interface` reinitialises them,
        this fixture completes on its own merits and the test can ask for
        completion again rather than for 80 % of it.
        """
        messages: list = []
        result = _run(
            mesh=_walled_tower(nz=20), messages=messages,
            **_head_mode(max_fill_time_s=2.0),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.status, STATUS_COMPLETE)
        self.assertEqual(result.misrun_voxel_count, 0)
        self.assertEqual(
            result.filled_fraction, 1.0,
            f"the pour barely moved: {result.filled_fraction:.3f} of cells",
        )
        self.assertGreater(self._poured_lb(messages), 0.0)

    def test_delivery_falls_away_when_the_column_backs_up(self) -> None:
        """
        Back-pressure, which is the property a fixed-velocity pin cannot have.

        Total domain mass changes only through the inlet pin -- the solver
        conserves the rest, which :class:`MassConservationTests` checks -- so
        its per-step increment *is* the net injection. In a closed tower with
        the metal nowhere left to go, that increment must collapse. The old pin
        instead kept writing its prescribed velocity and deleted the metal
        streaming back into it, which is how delivery went negative on the real
        assembly.

        The fixture is :func:`_walled_tower_with_a_sealed_pocket`, not the
        plain tower: since the submerged-interface heal a plain walled tower
        reports ``complete`` and the stage exits on that step, so the run now
        ends at the moment the column reaches the inlet and never spends any
        time backed up. One unreachable pocket keeps the pour running after the
        tower is full. Measured on it: 0.0158 lattice-mass/step over the first
        quarter against 3.3e-7 over the last, a factor of 49,000 -- and the
        decile trace shows delivery holding near 0.02 through the fill and
        falling off a cliff the moment the column closes on the inlet.
        """
        import physics.filling as pf

        mesh = _walled_tower_with_a_sealed_pocket(nz=12)
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
        self.assertNotEqual(
            result.status, STATUS_COMPLETE,
            "the sealed pocket filled, so the run never backed up",
        )
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

        Measured: a 3x3x40 channel of quarter-inch cells is 5.6 in^3, which
        ``pour_weight_lb`` puts at 1.46 lb of steel, and the run reports
        1.3 lb delivered -- 89 %. The balance is the seed ``create_state``
        starts with, which came in before the inlet did anything.

        The run reports ``complete`` at 1.50 s against a 3.0 s cap.

        **The fixture changed from ``_tower`` to ``_walled_tower``, for the
        reason spelled out in**
        :meth:`PressureInletTests.test_a_head_driven_pour_fills_and_delivers_metal`:
        ``_tower`` is fully dense, so before the mold pad it was periodic on all
        three axes, and after the pad it is a 2x2 no-slip channel at the
        thickness the stage warns is unresolvable. ``_walled_tower`` is walled by
        cores rather than by the pad, so the pad changes nothing about it. The
        ratio this test is actually about -- delivered weight against the
        casting's own weight -- lands in the same place on either fixture.

        It was reading 92 % of cells for one round while the front's own
        leavings sat frozen as submerged interface cells -- see
        :meth:`PressureInletTests.test_a_head_driven_pour_fills_and_delivers_metal`
        -- and the completion assertion is restored with the heal. The
        delivered weight is what this test is about, and it is unchanged
        either way.
        """
        from core.filling_stage import pour_weight_lb

        mesh = _walled_tower(nz=40)
        messages: list = []
        result = _run(mesh=mesh, messages=messages, **_head_mode(max_fill_time_s=3.0))
        self.assertIsNotNone(result)
        self.assertEqual(result.status, STATUS_COMPLETE)
        self.assertEqual(result.filled_fraction, 1.0)
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
        """
        A stall message that gives only a cell count hides the disagreement.

        The fixture is :func:`_walled_tower_with_a_sealed_pocket`, whose one
        unreachable cell makes the stall deterministic. A plain walled tower
        cannot be used: since the submerged-interface heal it fills and reports
        ``complete``, so the assertion had to be guarded by a skip. The skip is
        gone -- the pocket stalls every run, at 108/109 cells.
        """
        messages: list = []
        result = _run(mesh=_walled_tower_with_a_sealed_pocket(nz=12),
                      messages=messages, **_head_mode(max_fill_time_s=1.5))
        self.assertIsNotNone(result)
        self.assertEqual(result.status, STATUS_STALLED)
        stall = [m for m in messages if "has not advanced" in m]
        self.assertTrue(stall, f"a stalled run reported no stall: {messages}")
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


class PorousDragForceTests(unittest.TestCase):
    """
    A filter cell has to have a terminal velocity.

    The regression test for the round-3 failure. ``resolve_porous_velocity``
    answers "given the provisional velocity, what does the Guo-Zhao drag leave?"
    and the solver was using that answer only to build the equilibrium. The drag
    never reached the Guo source, so it never reached the collision's momentum
    budget -- over one collision the populations gain exactly the source force,
    and the source was gravity alone.

    A filter cell therefore accelerated for ever. Measured here before the fix,
    on the block below: ``sum_i c_i f_i`` grew by ``gravity_lattice`` every step
    without bound, reached ``u = 0.11`` by step 10,000 -- twice the Mach guard
    the timestep was sized for -- and by step 50,000 held 1,080 negative
    populations, five in every one of the block's 216 cells, which never healed.

    That is what put the negatives inside the real assembly's ceramic filters:
    60x more of them than the filter's 1.4 % of the mesh would predict, all in
    interface cells, persistent across thousands of steps, aborting the fill on
    the precursor guard at step 12,000. No floor on the reconstruction can fix
    it -- at the time this was measured ``omega_minus`` was ~5.6e-5 on this
    lattice, so the TRT barely relaxed the antisymmetric part, the gap between a
    braked equilibrium and an unbraked momentum was frozen in, and it grew until
    ``|f_asym|`` passed ``f_sym``. ``TAU_MINUS_MAX`` has since bounded
    ``omega_minus`` at 0.2 for exactly that reason -- see
    :class:`GhostModeDampingTests` -- so the freezing is gone; the momentum
    source this fixture tests is not, and it is what the assertions below are
    about.

    The fixture is deliberately *not* a free surface: no interface cells, no
    reconstruction, no mass exchange. It is the collision on its own, so a
    failure here cannot be blamed on the Koerner closure.
    """

    #: 10 ppi ceramic foam at the nominal porosity, on the real assembly's
    #: 4.3 mm fill grid. Not arbitrary: this is the filter Test 2 ships.
    DX_M = 0.0043
    NU_SI = 0.006 / 7200.0
    POROSITY = 0.85
    PORE_D_M = 2.54e-3

    def _block(self, n=6):
        """A periodic block of filter, full of metal, with nothing else in it."""
        from physics.filling import PorousZone, create_state
        from physics.fluid_flow import (
            ceramic_foam_permeability,
            forchheimer_inverse_length,
            lattice_units,
            porous_drag_coefficients,
        )

        units = lattice_units(
            self.DX_M, self.NU_SI, u_max_physical_m_s=float(np.sqrt(2 * 9.81 * 0.4))
        )
        k1 = ceramic_foam_permeability(self.POROSITY, self.PORE_D_M)
        c0, c1 = porous_drag_coefficients(
            self.POROSITY,
            k1,
            forchheimer_inverse_length(k1),
            nu_lattice=units.nu_lattice,
            dx_m=self.DX_M,
        )
        zone = PorousZone.inert((n, n, n))
        zone.porosity[...] = np.float32(self.POROSITY)
        zone.drag_c0[...] = np.float32(c0)
        zone.drag_c1[...] = np.float32(c1)
        state = create_state(
            np.zeros((n, n, n), dtype=bool),
            dx_m=self.DX_M,
            dt_s=units.dt_s,
            initial_fluid=np.ones((n, n, n), dtype=bool),
            porous=zone,
        )
        return state, units, c0, c1

    @staticmethod
    def _terminal_u(gravity_lattice, c0, c1):
        """
        The velocity at which the drag force cancels gravity.

        ``F_drag / rho = -2 [ (2 c0 - 1) + c1 |u| ] u``, so the balance is
        ``g = 2 (2 c0 - 1) u + 2 c1 u^2``. Written out rather than measured
        against the previous sample, because "it stopped changing" is also true
        of a solve that has quietly stalled.
        """
        a = 2.0 * c1
        b = 2.0 * (2.0 * c0 - 1.0)
        return float((-b + np.sqrt(b * b + 4.0 * a * gravity_lattice)) / (2.0 * a))

    def _momentum_z(self, state):
        from physics.fluid_flow import LATTICE_VELOCITIES, Q

        f = np.asarray(state.f, dtype=np.float64)
        out = np.zeros(state.shape)
        for i in range(Q):
            if LATTICE_VELOCITIES[i, 2]:
                out += LATTICE_VELOCITIES[i, 2] * f[i]
        return out

    def _settles(self, use_numba, steps):
        from physics.filling import step as filling_step

        state, units, c0, c1 = self._block()
        for _ in range(steps):
            filling_step(
                state, units, track_free_surface=False, use_numba=use_numba
            )
        f = np.asarray(state.f, dtype=np.float64)
        return (
            float(self._momentum_z(state).mean()),
            self._terminal_u(units.gravity_lattice, c0, c1),
            int(np.count_nonzero(f < 0.0)),
        )

    def test_the_numba_path_settles_at_the_algebraic_terminal_velocity(self) -> None:
        settled, expected, negatives = self._settles(True, 20_000)
        self.assertEqual(negatives, 0, "a porous cell went negative")
        # Downward, and within 1 % of the balance point. Before the fix this
        # was -0.246 against the same -0.00219, and still falling.
        self.assertAlmostEqual(settled, -expected, delta=0.01 * expected)

    def test_the_numpy_path_settles_at_the_same_place(self) -> None:
        """
        Fewer steps because the NumPy reference is ~40x slower; it converges in
        well under a thousand, so this is not a weaker assertion, only a
        cheaper one. The point is that the fix landed in both paths -- a
        one-sided edit here is the failure ``tests/test_filling_parity`` exists
        to catch, and this is the second lock on it.
        """
        settled, expected, negatives = self._settles(False, 2_000)
        self.assertEqual(negatives, 0, "a porous cell went negative")
        self.assertAlmostEqual(settled, -expected, delta=0.01 * expected)

    def test_the_drag_force_is_exactly_zero_off_a_filter(self) -> None:
        """
        The whole no-filter regression argument rests on this being *exactly*
        zero rather than merely small: an inert zone is ``c0 = 0.5, c1 = 0``, so
        adding this to the gravity force must not change a single bit of a
        filterless run.
        """
        from physics.fluid_flow import porous_drag_force

        rng = np.random.default_rng(11)
        rho = 1.0 + 0.01 * rng.standard_normal((4, 4, 4))
        u = 0.05 * rng.standard_normal((3, 4, 4, 4))
        force = porous_drag_force(
            rho, u, np.full((4, 4, 4), 0.5), np.zeros((4, 4, 4))
        )
        np.testing.assert_array_equal(np.abs(force), np.zeros((3, 4, 4, 4)))

    def test_the_drag_force_always_opposes_the_flow(self) -> None:
        """A drag term that can push is catastrophic and easy to sign-slip."""
        from physics.fluid_flow import porous_drag_force

        rng = np.random.default_rng(3)
        rho = np.abs(rng.standard_normal((5, 5, 5))) + 0.5
        u = rng.standard_normal((3, 5, 5, 5))
        force = porous_drag_force(
            rho, u, np.full((5, 5, 5), 0.71), np.full((5, 5, 5), 1.4)
        )
        along = np.einsum("i...,i...->...", force, u)
        self.assertTrue(bool(np.all(along <= 0.0)), "the drag pushed the flow")


class MetalThroughAFilterTests(unittest.TestCase):
    """
    End to end: metal has to cross a ceramic slab and keep filling.

    :class:`PorousDragForceTests` pins the mechanism on a bare collision. This
    pins the consequence on the real stage, which is where it was found. On the
    same fixture before the fix, at these settings, the run went **non-finite at
    step 3,800** and reported ``diverged`` with 36 % of the metal missing; the
    two-cell variant of the same slab reached 18 of 18 filter cells negative and
    ``|u|`` of 1.47 -- twenty-nine times the Mach guard -- before the precursor
    guard stopped it.

    One run, shared across the assertions: the stage is the expensive part and
    every question below is about the same trajectory.
    """

    @classmethod
    def setUpClass(cls) -> None:
        import physics.filling as pf

        cls.samples: list[dict] = []
        original = pf.step

        def traced(state, units, **kwargs):
            original(state, units, **kwargs)
            if state.step_count % 100:
                return state
            f = np.asarray(state.f, dtype=np.float64)
            wet = (state.flags == pf.FLAG_FLUID) | (
                state.flags == pf.FLAG_INTERFACE
            )
            porous = np.asarray(state.porous.porosity) < 1.0
            negative = np.any(f < 0.0, axis=0) & wet
            wet_cells = int(np.count_nonzero(wet))
            cls.samples.append(
                {
                    "step": state.step_count,
                    "negative_porous": int(np.count_nonzero(negative & porous)),
                    "negative_clear": int(np.count_nonzero(negative & ~porous)),
                    "wet_porous": int(np.count_nonzero(porous & wet)),
                    "wet_cells": wet_cells,
                    "wet_mass": float(np.sum(state.mass[wet])) if wet_cells else 0.0,
                    "finite": bool(
                        np.all(np.isfinite(f)) and np.all(np.isfinite(state.mass))
                    ),
                    "max_u": float(np.abs(np.asarray(state.u)).max()),
                }
            )
            return state

        pf.step = traced
        try:
            cls.result = _run(mesh=_filtered_tower(), max_fill_time_s=1.0)
        finally:
            pf.step = original

    def test_the_fill_completes_through_the_slab(self) -> None:
        self.assertIsNotNone(self.result)
        self.assertEqual(self.result.status, STATUS_COMPLETE, self.result.warnings)
        self.assertEqual(self.result.filled_fraction, 1.0)

    def test_the_slab_wets_all_the_way_through(self) -> None:
        """
        "Completed" is not the same as "the metal went through the filter" --
        a slab the front merely closed around would also finish. Every porous
        cell has to end up holding metal.
        """
        wet_porous = [s["wet_porous"] for s in self.samples]
        self.assertTrue(self.samples, "the stage never stepped the solver")
        self.assertEqual(max(wet_porous), 27, "the slab never fully wetted")

    def test_no_porous_cell_ever_holds_a_negative_population(self) -> None:
        offenders = [s for s in self.samples if s["negative_porous"]]
        self.assertEqual(
            [(s["step"], s["negative_porous"]) for s in offenders[:5]],
            [],
            "porous cells went negative",
        )

    def test_nothing_else_goes_negative_either(self) -> None:
        """
        Asserted separately so a future failure says *where*. The drag fix is
        about filter cells; if clear cells start going negative that is a
        different bug and should not be read as this one recurring.
        """
        offenders = [s for s in self.samples if s["negative_clear"]]
        self.assertEqual(
            [(s["step"], s["negative_clear"]) for s in offenders[:5]], []
        )

    def test_the_field_stays_finite_and_subsonic(self) -> None:
        self.assertEqual([s["step"] for s in self.samples if not s["finite"]], [])
        worst = max(s["max_u"] for s in self.samples)
        # The Mach guard the timestep was sized for, with room for the front.
        self.assertLess(worst, 0.05, f"|u| reached {worst}")

    def test_wet_mass_stays_under_the_physical_ceiling_through_the_slab(self) -> None:
        """
        Mass conservation across the porous band. The free surface can hold at
        most one cell's worth per wet cell, so a total above that is the
        bookkeeping having come apart.

        The exchange's porous override was exactly the place it could, and is
        gone -- see :class:`PorousPairWeightTests`. This total is the wrong
        statistic for that defect and stays here for a different reason: the
        override conserved mass *exactly* while eps ran to 7.76, so a passing
        ceiling check said nothing about it. Fill fraction is the statistic
        that sees it.
        """
        breaches = [
            (s["step"], s["wet_mass"], s["wet_cells"])
            for s in self.samples
            if s["wet_mass"] > DIVERGENCE_MASS_FACTOR * max(s["wet_cells"], 1)
        ]
        self.assertEqual(breaches[:3], [])
