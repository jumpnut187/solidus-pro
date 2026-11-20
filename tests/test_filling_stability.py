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
from tests.test_filling_stage import (
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
    centers = lattice_centers((nx, ny, nz), dx=DX, origin=(0.0, 0.0, 0.0))
    mesh = _FakeMesh(centers, np.full(len(centers), part_type), DX)
    mesh._top_z = (nz - 1) * DX
    return mesh

def _walled_tower(nz=12, width=3):
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
    mesh = _walled_tower(nz=nz, width=width)
    index = np.rint(np.asarray(mesh.voxel_centers) / DX).astype(int)
    corner = (index[:, 0] == 0) & (index[:, 1] == 0) & (index[:, 2] == 0)
    assert int(corner.sum()) == 1, "the fixture lost its corner cell"
    mesh.voxel_part_types[corner] = PART_TYPE_CAST
    return mesh

def _gasless_interface_cells(state):
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

    def __init__(self, front_face_area_m2: float):
        self.front_face_area_m2 = front_face_area_m2

class _NamedAlloy(_Alloy):

    def __init__(self, name: str):
        self.name = name

def _run(mesh=None, options=None, messages=None, alloy=None,
         detailed_progress_callback=None, **filling_overrides):
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

    def _velocity_history(self, steps=1200):
        import physics.filling as pf

        history: list[float] = []
        original = pf.step

        def traced(state, units, **kwargs):
            original(state, units, **kwargs)
            history.append(float(np.max(np.abs(state.u))))

        pf.step = traced
        try:
            _run(mesh=_walled_tower(nz=40, width=4), max_fill_time_s=0.5)
        finally:
            pf.step = original
        return np.asarray(history[:steps], dtype=np.float64)

    def test_velocity_does_not_grow_without_bound(self) -> None:
        history = self._velocity_history()
        self.assertGreater(history.size, 200, "fill ended too early to judge")
        early = float(np.max(history[: history.size // 4]))
        late = float(np.max(history[-history.size // 4 :]))
        self.assertLess(
            late,
            4.0 * early + 1e-6,
            f"inlet velocity is still accelerating: {early:.5g} -> {late:.5g}",
        )

    def test_velocity_stays_under_the_mach_guard(self) -> None:
        history = self._velocity_history()
        self.assertLess(
            float(np.max(history)),
            0.05 * 3.0,
            "velocity ran far past the Mach guard the timestep was sized for",
        )

class MassConservationTests(unittest.TestCase):

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

    NU = 0.006 / 7200.0
    DX_M = 0.0152

    def _units(self):
        from physics.fluid_flow import lattice_units

        return lattice_units(self.DX_M, self.NU, u_max_physical_m_s=2.8)

    def _closed_box(self, n=10):
        import physics.filling as pf

        obstacle = np.zeros((n, n, n), dtype=bool)
        obstacle[0] = obstacle[-1] = True
        obstacle[:, 0] = obstacle[:, -1] = True
        obstacle[:, :, 0] = obstacle[:, :, -1] = True
        units = self._units()
        state = pf.create_state(
            obstacle, dx_m=self.DX_M, dt_s=units.dt_s, initial_fluid=~obstacle
        )
        self.assertEqual(int(np.count_nonzero(state.flags == pf.FLAG_INTERFACE)), 0)
        self.assertEqual(int(np.count_nonzero(state.flags == pf.FLAG_GAS)), 0)
        return state, units

    def test_fluid_mass_stays_equal_to_its_density(self) -> None:
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
        import physics.filling as pf

        state, units = self._closed_box()
        for _ in range(2000):
            pf.step(state, units)
        self.assertEqual(
            int(np.count_nonzero(state.flags == pf.FLAG_INTERFACE)), 0
        )

    def test_a_column_under_an_interface_layer_stays_bounded(self) -> None:
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

    NU = 0.006 / 7200.0
    DX_M = 0.0152
    STEPS = 2000

    def _units(self):
        from physics.fluid_flow import lattice_units

        return lattice_units(self.DX_M, self.NU, u_max_physical_m_s=2.8)

    def _submerged_filter(self, seed_eps=1.05):
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
        self.assertAlmostEqual(
            float(np.abs(state.mass[band] / full[band] - 1.0).max()), 0.0,
            places=6,
            msg="create_state seeded the filter without its porosity factor",
        )
        state.mass[band] = (full * seed_eps)[band]
        return state, units, band

    def _worst_fill_fraction_error(self, state, band):
        full = state.rho.astype(np.float64) * state.porous.porosity_f64
        with np.errstate(invalid="ignore", divide="ignore"):
            eps = np.where(full > 0, state.mass / full, 1.0)
        return float(np.abs(eps[band] - 1.0).max())

    def _drift(self, use_numba, advance_mass=None):
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
        seeded, final, drift = self._drift(use_numba=False)
        self.assertAlmostEqual(seeded, 0.05, places=6)
        self.assertLess(
            final, 1.002 * seeded,
            f"the fill-fraction error grew {seeded:.4e} -> {final:.4e} over "
            f"{self.STEPS} steps: the exchange weight is feeding a cell's own "
            f"eps back into its exchange amplitude",
        )
        self.assertLess(abs(drift), 1e-12, f"mass drifted by {drift:.3e}")

    def test_a_seeded_error_does_not_grow_numba(self) -> None:
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
        _, fixed, _ = self._drift(use_numba=False)
        seeded, broken, drift = self._drift(
            use_numba=False, advance_mass=_porous_weight_advance_mass
        )
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
        self.assertLess(
            worst, 1.01 * seeded,
            f"a FLUID cell in or beside the filter reached eps "
            f"{1 + worst:.4g} against a seeded {1 + seeded:.4g}; a full cell "
            f"holds exactly its pore volume, so back-pressure belongs in rho, "
            f"not here",
        )

    def test_the_exchange_is_unchanged_where_there_is_no_filter(self) -> None:
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

    DX_M = 0.004
    NU = 6.5e-7
    NZ = 14
    WIDTH = 3
    STEPS = 40

    def _channel(self, *, floor: bool, lid: bool):
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
        for use_numba in self._both_paths():
            with self.subTest(numba=use_numba):
                first, _, final = self._first_top_wet_step(
                    floor=True, lid=False, use_numba=use_numba
                )
                self.assertIsNone(first, f"the top plane wet at step {first}")
                self.assertEqual(final, 0)

    def test_a_closed_channel_is_clean(self) -> None:
        for use_numba in self._both_paths():
            with self.subTest(numba=use_numba):
                first, _, final = self._first_top_wet_step(
                    floor=True, lid=True, use_numba=use_numba
                )
                self.assertIsNone(first, f"the top plane wet at step {first}")
                self.assertEqual(final, 0)

    def test_the_stage_never_builds_an_unwalled_grid(self) -> None:
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

    DX_M = 0.0043
    NU = 0.006 / 7200.0
    SHAPE = (8, 8, 10)
    AMPLITUDE = 0.01
    STEPS = 600

    def _closed_box(self, tau_minus_max):
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
        u = np.asarray(state.u, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            return float(np.mean((sign * (u[0] - u[1]))[wet]) * 0.5)

    def _trajectory(self, *, use_numba, tau_minus_max):
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

    def _run_with_injected_negatives(self, at_steps):
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
        from core.filling_stage import (
            DIVERGENCE_CHECK_INTERVAL, STATUS_DIVERGED,
        )

        result, messages, samples = self._run_with_injected_negatives(
            {DIVERGENCE_CHECK_INTERVAL}
        )
        self.assertEqual(len(samples), 1, "the injection never happened")
        self.assertGreaterEqual(samples[0]["negative"], 8)
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

    def test_a_full_fill_reports_complete(self) -> None:
        result = _run(mesh=_walled_tower(nz=2, width=2),
                      **_head_mode(max_fill_time_s=2.0))
        self.assertIsNotNone(result)
        self.assertEqual(result.status, STATUS_COMPLETE)
        self.assertEqual(result.misrun_voxel_count, 0)

    def test_a_truncated_fill_reports_the_time_cap(self) -> None:
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

    def test_a_shallow_walled_tower_fills_completely(self) -> None:
        result, state = _final_state(dict(
            mesh=_walled_tower(nz=4, width=4),
            **_head_mode(max_fill_time_s=2.0),
        ))
        self.assertIsNotNone(result)
        self.assertEqual(result.status, STATUS_COMPLETE)
        self.assertEqual(result.misrun_voxel_count, 0)

    def test_no_interface_cell_is_left_without_a_gas_neighbour(self) -> None:
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
        result, state = _final_state(dict(
            mesh=_walled_tower(nz=12),
            **_head_mode(max_fill_time_s=1.5),
        ))
        self.assertEqual(result.status, STATUS_COMPLETE)
        self.assertEqual(
            int(np.count_nonzero(_gasless_interface_cells(state))), 0
        )

    def test_a_submerged_cell_converts_and_pays_its_neighbours(self) -> None:
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
        import physics.filling as pf

        state, _ = _submerged_pair()
        for _ in range(4):
            self.assertEqual(pf.heal_submerged_interface(state, 0.0, dwell_steps=5), 0)
        was = np.uint8(state.flags[5, 5, 4])
        state.flags[5, 5, 4] = pf.FLAG_GAS
        self.assertEqual(pf.heal_submerged_interface(state, 0.0, dwell_steps=5), 0)
        state.flags[5, 5, 4] = was
        for _ in range(4):
            self.assertEqual(pf.heal_submerged_interface(state, 0.0, dwell_steps=5), 0)
        self.assertEqual(int(state.flags[5, 5, 5]), int(pf.FLAG_INTERFACE))
        self.assertEqual(pf.heal_submerged_interface(state, 0.0, dwell_steps=5), 1)
        self.assertEqual(int(state.flags[5, 5, 5]), int(pf.FLAG_FLUID))

    def test_it_leaves_a_cell_with_a_gas_neighbour_alone(self) -> None:
        import physics.filling as pf

        state, _ = _submerged_pair()
        state.mass[5, 5, 5] = 0.40
        before = state.mass[5, 5, 6]
        state.flags[5, 5, 4] = pf.FLAG_GAS
        state.mass[5, 5, 4] = 0.0
        state.f[:, 5, 5, 4] = 0.0
        self.assertEqual(pf.heal_submerged_interface(state, 0.0, dwell_steps=1), 0)
        self.assertEqual(int(state.flags[5, 5, 5]), int(pf.FLAG_INTERFACE))
        self.assertEqual(float(state.mass[5, 5, 6]), float(before))

    def test_a_pocket_with_no_recipient_strands_rather_than_leaks(self) -> None:
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

    def test_thickness_finds_the_narrowest_section(self) -> None:
        dense = np.full((10, 10, 10), EMPTY_CELL, dtype=np.int8)
        dense[0:6, 4:7, 4:7] = PART_TYPE_RUNNER
        dense[6:10, 5:6, 5:6] = PART_TYPE_GATE
        self.assertLess(measure_choke_thickness_cells(dense), 2.0)

    def test_a_wide_thin_gate_is_not_excused_by_its_area(self) -> None:
        dense = np.full((12, 12, 12), EMPTY_CELL, dtype=np.int8)
        dense[0:6, 1:11, 4:7] = PART_TYPE_RUNNER
        dense[6:12, 1:11, 5:6] = PART_TYPE_GATE
        self.assertGreaterEqual(_min_flow_slice_cells(dense), 10)
        self.assertLess(measure_choke_thickness_cells(dense), 2.0)

    def test_no_gating_metal_gives_zero(self) -> None:
        dense = np.full((4, 4, 4), PART_TYPE_CAST, dtype=np.int8)
        self.assertEqual(measure_choke_thickness_cells(dense), 0.0)

    def test_an_under_resolved_choke_is_warned_about(self) -> None:
        mesh = _tower(nx=1, ny=1, nz=8, part_type=PART_TYPE_GATE)
        result = _run(mesh=mesh, max_fill_time_s=0.02)
        self.assertIsNotNone(result)
        self.assertLess(result.choke_thickness_cells, MIN_CHOKE_THICKNESS_CELLS)
        self.assertTrue(
            any("cells thick" in w for w in result.warnings),
            f"under-resolved choke was not warned about: {result.warnings}",
        )

    def test_a_resolved_choke_is_not_warned_about(self) -> None:
        mesh = _tower(nx=4, ny=4, nz=8, part_type=PART_TYPE_GATE)
        result = _run(mesh=mesh, max_fill_time_s=0.02)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.choke_thickness_cells, MIN_CHOKE_THICKNESS_CELLS)
        self.assertFalse(
            any("cells thick" in w for w in result.warnings),
            f"warned about a well-resolved choke: {result.warnings}",
        )

def _plate_welded_to_a_block():
    dense = np.full((20, 18, 14), EMPTY_CELL, dtype=np.int8)
    dense[1:8, 5:12, 3:10] = PART_TYPE_RUNNER
    dense[8:19, 2:16, 6:8] = PART_TYPE_GATE
    dense[8:11, 7:10, 4:10] = PART_TYPE_GATE
    return dense

def _generous_channel():
    dense = np.full((16, 12, 12), EMPTY_CELL, dtype=np.int8)
    dense[1:8, 3:8, 3:8] = PART_TYPE_RUNNER
    dense[8:15, 3:8, 3:8] = PART_TYPE_GATE
    return dense

class ChokeMaskingTests(unittest.TestCase):

    def setUp(self) -> None:
        from core.filling_stage import (
            measure_choke_thickness_by_part,
            narrowest_choke_part_code,
        )

        self.dense = _plate_welded_to_a_block()
        self.by_part = measure_choke_thickness_by_part(self.dense)
        self.narrowest = narrowest_choke_part_code(self.by_part)

    def test_the_block_really_does_mask_the_plate(self) -> None:
        from scipy import ndimage

        flow = np.isin(self.dense, (PART_TYPE_RUNNER, PART_TYPE_GATE))
        distance = ndimage.distance_transform_edt(np.pad(flow, 1))[1:-1, 1:-1, 1:-1]
        old_gate_thickness = float(distance[self.dense == PART_TYPE_GATE].max()) * 2 - 1
        self.assertGreaterEqual(old_gate_thickness, 3.0)

    def test_the_robust_measure_reports_the_plate(self) -> None:
        self.assertEqual(self.by_part[PART_TYPE_GATE], 1.0)
        self.assertLessEqual(self.by_part[PART_TYPE_GATE], MIN_CHOKE_THICKNESS_CELLS)

    def test_the_fat_junction_is_not_reported_as_thin(self) -> None:
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

    def _run_dense(self, dense):
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
        result = self._run_dense(_generous_channel())
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.choke_thickness_cells, 5.0)
        self.assertFalse(
            [w for w in result.warnings if "cells thick" in w or "cells across" in w],
            f"warned about a five-cell channel: {result.warnings}",
        )

    def test_the_choke_area_is_the_grid_that_was_solved(self) -> None:
        mesh = _tower(nx=5, ny=5, nz=8, part_type=PART_TYPE_GATE)
        result = _run(mesh=mesh, max_fill_time_s=0.02, fill_grid_coarsening=2)
        self.assertIsNotNone(result)
        self.assertEqual(result.coarsening, 2)
        self.assertGreater(result.choke_area_cm2, result.choke_area_fine_cm2)
        self.assertAlmostEqual(
            result.choke_area_cm2 / result.choke_area_fine_cm2, 36.0 / 25.0, places=6
        )

class FrameCaptureTests(unittest.TestCase):

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
        for i, frame in enumerate(self.result.frames):
            with self.subTest(frame=i):
                self.assertTrue(bool(np.all(frame.fill_fraction >= 0.0)))
                self.assertTrue(bool(np.all(frame.fill_fraction <= 1.0)))

    def test_frames_advance_in_time(self) -> None:
        times = [f.time_s for f in self.result.frames]
        self.assertEqual(times, sorted(times))

    def test_the_pour_gains_metal_over_the_run(self) -> None:
        first = float(self.result.frames[0].fill_fraction.sum())
        last = float(self.result.frames[-1].fill_fraction.sum())
        self.assertGreater(last, first)

    def test_frame_count_stays_within_the_budget(self) -> None:
        from core.filling_stage import FILL_ANIMATION_FRAMES

        for cap in (0.02, 0.2, 1.0):
            with self.subTest(max_fill_time_s=cap):
                result = _run(mesh=_tower(nz=30), max_fill_time_s=cap)
                self.assertIsNotNone(result)
                self.assertLessEqual(
                    len(result.frames), 2 * FILL_ANIMATION_FRAMES + 1
                )

    def test_frames_stay_evenly_spaced_after_decimation(self) -> None:
        result = _run(mesh=_tower(nz=30), max_fill_time_s=1.0)
        times = np.array([f.time_s for f in result.frames], dtype=np.float64)
        if times.size < 4:
            self.skipTest("run too short to judge spacing")
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

    def _state(self, shape=(7, 7, 7)):
        from physics.filling import create_state

        obstacle = np.zeros(shape, dtype=bool)
        fluid = np.zeros(shape, dtype=bool)
        fluid[0, 0, 0] = True
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
        messages: list = []
        _run(messages=messages, inlet_mode_code=0, initial_head_in=4.0)
        line = next(m for m in messages if "inlet" in m and "head" in m)
        self.assertIn("pressure boundary", line)
        self.assertIn("velocity follows the flow", line)
        self.assertGreater(self._reported_inlet_density(messages), 1.0)

    def test_a_deeper_head_raises_the_inlet_pressure(self) -> None:
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

    def _poured_lb(self, messages):
        reports = [m for m in messages if "lb poured" in m]
        self.assertTrue(reports, f"no poured weight reported in {messages}")
        return float(reports[-1].split("lb poured")[0].split(",")[-1])

    def test_a_head_driven_pour_fills_and_delivers_metal(self) -> None:
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
        self.assertEqual(
            breaches[-3:], [], f"the inlet touched gas at steps {breaches[-3:]}"
        )

    def test_an_unreachable_head_is_called_out_before_the_clock_runs(self) -> None:
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
        messages: list = []
        _run(
            mesh=_tower(nz=12), messages=messages,
            **_head_mode(inlet_mode_code=1, pour_rate_lb_s=7.5, max_fill_time_s=0.02)
        )
        line = next(m for m in messages if "inlet" in m and "m/s" in m)
        self.assertIn("velocity (7.5 lb/s", line)
        self.assertNotIn("pressure boundary", line)

class MassProgressReportingTests(unittest.TestCase):

    def _reports(self, messages):
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
        messages: list = []
        _run(mesh=_tower(nz=20), messages=messages, **_head_mode(max_fill_time_s=0.3))
        for line in self._reports(messages):
            percent = int(line.split("% by mass")[0].split(",")[-1])
            with self.subTest(line=line):
                self.assertGreaterEqual(percent, 0)
                self.assertLessEqual(percent, 100)

    def test_poured_weight_only_grows_while_metal_is_arriving(self) -> None:
        messages: list = []
        _run(mesh=_tower(nz=20), messages=messages, **_head_mode(max_fill_time_s=0.4))
        poured = [
            float(m.split("lb poured")[0].split(",")[-1])
            for m in self._reports(messages)
        ]
        self.assertEqual(poured, sorted(poured), f"poured weight went backwards: {poured}")

    def test_the_poured_weight_matches_the_castings_own_weight(self) -> None:
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
        from core.filling_stage import STALL_MASS_FRACTION_EPSILON

        self.assertGreater(STALL_MASS_FRACTION_EPSILON, 0.0)
        self.assertLess(STALL_MASS_FRACTION_EPSILON, 0.01)

    def test_a_stall_is_reported_with_both_measures(self) -> None:
        messages: list = []
        result = _run(mesh=_walled_tower_with_a_sealed_pocket(nz=12),
                      messages=messages, **_head_mode(max_fill_time_s=1.5))
        self.assertIsNotNone(result)
        self.assertEqual(result.status, STATUS_STALLED)
        stall = [m for m in messages if "has not advanced" in m]
        self.assertTrue(stall, f"a stalled run reported no stall: {messages}")
        self.assertIn("by mass", stall[0])

class InterfaceSkinTests(unittest.TestCase):

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
        from physics.filling import FLAG_GAS, ensure_interface_skin

        state = self._state()
        before = state.flags.copy()
        halo = state.flags != FLAG_GAS
        self.assertEqual(ensure_interface_skin(state, halo), 0)
        self.assertTrue(bool(np.array_equal(state.flags, before)))

    def test_it_takes_no_mass(self) -> None:
        from physics.filling import FLAG_GAS, FLAG_INTERFACE, ensure_interface_skin

        state = self._state()
        state.flags[state.flags == FLAG_INTERFACE] = FLAG_GAS
        total = float(state.mass.sum())
        halo = np.zeros(state.shape, dtype=bool)
        halo[2, 2, 3] = True
        ensure_interface_skin(state, halo)
        self.assertAlmostEqual(float(state.mass.sum()), total, places=12)

class PourRateRuleTests(unittest.TestCase):

    def test_the_rate_is_the_square_root_of_the_weight(self) -> None:
        from core.filling_stage import recommended_pour_rate_lb_s

        for weight, expected in ((100.0, 10.0), (25.0, 5.0), (2500.0, 50.0)):
            with self.subTest(weight_lb=weight):
                self.assertAlmostEqual(
                    recommended_pour_rate_lb_s(weight), expected, places=6
                )

    def test_pour_time_and_rate_are_the_same_number(self) -> None:
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
        from core.part_manager import PART_TYPE_CHILL
        from core.filling_stage import pour_weight_lb

        metal = _tower(nx=2, ny=2, nz=6)
        mixed = _tower(nx=2, ny=2, nz=6)
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
        from core.filling_stage import pour_weight_lb

        volume_in3 = 24 * (DX ** 3)
        expected_lb = 7200.0 * volume_in3 * 1.6387064e-5 * 2.20462262
        self.assertAlmostEqual(
            pour_weight_lb(_tower(nx=2, ny=2, nz=6), _Alloy()),
            expected_lb,
            places=5,
        )

    def test_a_zero_rate_derives_from_the_casting(self) -> None:
        from core.filling_stage import pour_weight_lb, recommended_pour_rate_lb_s

        mesh = _tower(nz=12)
        messages: list = []
        _run(mesh=mesh, messages=messages, inlet_mode_code=1, pour_rate_lb_s=0.0)
        expected = recommended_pour_rate_lb_s(pour_weight_lb(mesh, _Alloy()))
        line = next(m for m in messages if "inlet" in m and "lb/s" in m)
        self.assertIn("sqrt of", line)
        self.assertIn(f"{expected:.1f} lb/s", line)

    def test_a_filter_caps_the_pour_rate(self) -> None:
        from core.filling_stage import filter_flow_limit_kg_s

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
        from core.filling_stage import filter_flow_limit_kg_s

        zone = _FakeFilterZone(front_face_area_m2=0.05 * 0.05)
        limit, note = filter_flow_limit_kg_s([zone], _NamedAlloy("BRONZE"), 0)
        self.assertIsNone(limit)
        self.assertIsNotNone(note)
        self.assertIn("not covered", note)

    def test_a_filter_in_the_geometry_caps_the_running_pour(self) -> None:
        from core.part_manager import PART_TYPE_FILTER, PART_TYPE_SPRUE

        mesh = _tower(nx=2, ny=2, nz=12, part_type=PART_TYPE_SPRUE)
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
        state.flags[3, 3, 3] = FLAG_INTERFACE
        state.mass[3, 3, 3] = 0.4
        state.f[:, 3, 3, 3] = 0.9
        state.u[:, 3, 3, 3] = 0.5
        state.rho[3, 3, 3] = 2.0
        return state

    def test_a_detached_cell_is_returned_to_rest(self) -> None:
        from physics.filling import reset_detached_interface_cells

        state = self._state_with_a_detached_cell()
        self.assertEqual(reset_detached_interface_cells(state), 1)
        self.assertAlmostEqual(float(np.max(np.abs(state.u[:, 3, 3, 3]))), 0.0)

    def test_its_mass_is_left_alone(self) -> None:
        from physics.filling import reset_detached_interface_cells

        state = self._state_with_a_detached_cell()
        before = float(state.mass[3, 3, 3])
        reset_detached_interface_cells(state)
        self.assertAlmostEqual(float(state.mass[3, 3, 3]), before)

    def test_a_supported_interface_cell_is_untouched(self) -> None:
        from physics.filling import (
            FLAG_FLUID,
            reset_detached_interface_cells,
        )

        state = self._state_with_a_detached_cell()
        state.flags[3, 3, 2] = FLAG_FLUID
        self.assertEqual(reset_detached_interface_cells(state), 0)
        self.assertAlmostEqual(float(np.max(np.abs(state.u[:, 3, 3, 3]))), 0.5)

    def test_it_does_nothing_when_there_is_no_interface(self) -> None:
        from physics.filling import FLAG_GAS, reset_detached_interface_cells

        state = self._state_with_a_detached_cell()
        state.flags[...] = FLAG_GAS
        self.assertEqual(reset_detached_interface_cells(state), 0)

    def test_a_real_fill_does_not_run_away_in_velocity(self) -> None:
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
        quarter = max(1, late.size // 4)
        self.assertLess(
            float(np.median(late[-quarter:])),
            3.0 * float(np.median(late[:quarter])) + 0.02,
            f"velocity is still climbing across the settled run: "
            f"{np.median(late[:quarter]):.4g} -> {np.median(late[-quarter:]):.4g}",
        )

class ProgressReportingTests(unittest.TestCase):

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
        self.assertIsNotNone(_run(mesh=_tower(nz=12), max_fill_time_s=0.05))

class SmagorinskyConstantTests(unittest.TestCase):

    def _warning(self, result):
        return next(
            (w for w in result.warnings if "smagorinsky_cs" in w), None
        )

    def test_a_usable_constant_passes_without_comment(self) -> None:
        result = _run(mesh=_tower(nz=12), max_fill_time_s=0.05, smagorinsky_cs=0.2)
        self.assertIsNotNone(result)
        self.assertIsNone(self._warning(result))

    def test_zero_is_accepted(self) -> None:
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

    DX_M = 0.0043
    NU_SI = 0.006 / 7200.0
    POROSITY = 0.85
    PORE_D_M = 2.54e-3

    def _block(self, n=6):
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
        self.assertAlmostEqual(settled, -expected, delta=0.01 * expected)

    def test_the_numpy_path_settles_at_the_same_place(self) -> None:
        settled, expected, negatives = self._settles(False, 2_000)
        self.assertEqual(negatives, 0, "a porous cell went negative")
        self.assertAlmostEqual(settled, -expected, delta=0.01 * expected)

    def test_the_drag_force_is_exactly_zero_off_a_filter(self) -> None:
        from physics.fluid_flow import porous_drag_force

        rng = np.random.default_rng(11)
        rho = 1.0 + 0.01 * rng.standard_normal((4, 4, 4))
        u = 0.05 * rng.standard_normal((3, 4, 4, 4))
        force = porous_drag_force(
            rho, u, np.full((4, 4, 4), 0.5), np.zeros((4, 4, 4))
        )
        np.testing.assert_array_equal(np.abs(force), np.zeros((3, 4, 4, 4)))

    def test_the_drag_force_always_opposes_the_flow(self) -> None:
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
        offenders = [s for s in self.samples if s["negative_clear"]]
        self.assertEqual(
            [(s["step"], s["negative_clear"]) for s in offenders[:5]], []
        )

    def test_the_field_stays_finite_and_subsonic(self) -> None:
        self.assertEqual([s["step"] for s in self.samples if not s["finite"]], [])
        worst = max(s["max_u"] for s in self.samples)
        self.assertLess(worst, 0.05, f"|u| reached {worst}")

    def test_wet_mass_stays_under_the_physical_ceiling_through_the_slab(self) -> None:
        breaches = [
            (s["step"], s["wet_mass"], s["wet_cells"])
            for s in self.samples
            if s["wet_mass"] > DIVERGENCE_MASS_FACTOR * max(s["wet_cells"], 1)
        ]
        self.assertEqual(breaches[:3], [])
