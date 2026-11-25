from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, cast

import numpy as np

from physics.fluid_flow import (
    CS2,
    SMAGORINSKY_CS,
    TRT_MAGIC_PARAMETER,
    LATTICE_VELOCITIES,
    LATTICE_WEIGHTS,
    OPPOSITE,
    Q,
    LatticeUnits,
    collide_trt,
    equilibrium,
    guo_force_terms,
    porous_drag_force,
    resolve_porous_velocity,
    smagorinsky_relaxation,
)
from physics.fluid_flow import NUMBA_AVAILABLE, lattice_tables

if NUMBA_AVAILABLE:
    from physics.fluid_flow import (
        advect_temperature_kernel,
        collide_kernel,
        compact_mask_kernel,
        free_surface_kernel,
        interface_neighbourhood_kernel,
        promoted_donor_average_kernel,
        promoted_donor_temperature_mask_kernel,
        promoted_temperature_average_kernel,
        redistribute_gather_kernel,
        redistribute_share_kernel,
        stream_kernel,
        transition_front_kernel,
        transition_scan_kernel,
        wall_heat_loss_kernel,
    )

FLAG_OBSTACLE = np.uint8(0)
FLAG_GAS = np.uint8(1)
FLAG_INTERFACE = np.uint8(2)
FLAG_FLUID = np.uint8(3)

TRANSITION_HYSTERESIS = 1e-3

HEAL_DWELL_STEPS = 200

GAS_DENSITY = 1.0

REFERENCE_DENSITY = 1.0

NO_TEMPERATURE = np.float32(np.nan)

@dataclass
class PorousZone:

    porosity: np.ndarray
    drag_c0: np.ndarray
    drag_c1: np.ndarray
    entry_offset: np.ndarray
    primed: np.ndarray

    _porosity_f64: Optional[np.ndarray] = field(
        default=None, repr=False, compare=False
    )
    _drag_c0_f64: Optional[np.ndarray] = field(
        default=None, repr=False, compare=False
    )
    _drag_c1_f64: Optional[np.ndarray] = field(
        default=None, repr=False, compare=False
    )
    _entry_offset_f64: Optional[np.ndarray] = field(
        default=None, repr=False, compare=False
    )

    @classmethod
    def inert(cls, shape: Tuple[int, int, int]) -> "PorousZone":
        return cls(
            porosity=np.ones(shape, dtype=np.float32),
            drag_c0=np.full(shape, 0.5, dtype=np.float32),
            drag_c1=np.zeros(shape, dtype=np.float32),
            entry_offset=np.zeros(shape, dtype=np.float32),
            primed=np.zeros(shape, dtype=bool),
        )

    @property
    def is_inert(self) -> bool:
        return not bool(np.any(self.drag_c1)) and not bool(
            np.any(self.drag_c0 != 0.5)
        )

    @property
    def porosity_f64(self) -> np.ndarray:
        cached = self._porosity_f64
        if cached is None or cached.shape != self.porosity.shape:
            cached = self.porosity.astype(np.float64)
            self._porosity_f64 = cached
        return cached

    @property
    def drag_c0_f64(self) -> np.ndarray:
        cached = self._drag_c0_f64
        if cached is None or cached.shape != self.drag_c0.shape:
            cached = self.drag_c0.astype(np.float64)
            self._drag_c0_f64 = cached
        return cached

    @property
    def drag_c1_f64(self) -> np.ndarray:
        cached = self._drag_c1_f64
        if cached is None or cached.shape != self.drag_c1.shape:
            cached = self.drag_c1.astype(np.float64)
            self._drag_c1_f64 = cached
        return cached

    @property
    def entry_offset_f64(self) -> np.ndarray:
        cached = self._entry_offset_f64
        if cached is None or cached.shape != self.entry_offset.shape:
            cached = self.entry_offset.astype(np.float64)
            self._entry_offset_f64 = cached
        return cached

@dataclass
class FillingState:

    shape: Tuple[int, int, int]
    dx_m: float
    dt_s: float
    f: np.ndarray
    rho: np.ndarray
    u: np.ndarray
    mass: np.ndarray
    flags: np.ndarray
    fill_time_s: np.ndarray
    temperature: np.ndarray
    porous: PorousZone
    step_count: int = 0
    inlet_mass: float = 0.0

    _scratch_post: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _scratch_out: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _scratch_mass_out: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _scratch_rho64: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _scratch_u64: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _submerged_streak: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _scratch_exposed_faces: Optional[np.ndarray] = field(
        default=None, repr=False, compare=False
    )

    @property
    def cell_volume_m3(self) -> float:
        return self.dx_m ** 3

def create_state(
    obstacle: np.ndarray,
    *,
    dx_m: float,
    dt_s: float,
    initial_fluid: Optional[np.ndarray] = None,
    porous: Optional[PorousZone] = None,
    initial_temperature_c: Optional[float] = None,
) -> FillingState:
    obstacle = np.asarray(obstacle, dtype=bool)
    if obstacle.ndim != 3:
        raise ValueError(f"obstacle must be 3-D, got shape {obstacle.shape}")
    shape = (int(obstacle.shape[0]), int(obstacle.shape[1]), int(obstacle.shape[2]))

    flags = np.where(obstacle, FLAG_OBSTACLE, FLAG_GAS).astype(np.uint8)
    mass = np.zeros(shape, dtype=np.float64)
    temperature = np.full(shape, NO_TEMPERATURE, dtype=np.float32)
    zone = porous if porous is not None else PorousZone.inert(shape)

    if initial_fluid is not None:
        fluid = np.asarray(initial_fluid, dtype=bool) & ~obstacle
        flags[fluid] = FLAG_FLUID
        por = zone.porosity
        mass[fluid] = REFERENCE_DENSITY * por[fluid].astype(np.float64)
        neighbours = _dilate(fluid)
        interface = neighbours & ~fluid & ~obstacle
        flags[interface] = FLAG_INTERFACE
        mass[interface] = (
            0.5 * REFERENCE_DENSITY * por[interface].astype(np.float64)
        )
        if initial_temperature_c is not None:
            temperature[fluid | interface] = np.float32(initial_temperature_c)

    rho = np.full(shape, REFERENCE_DENSITY, dtype=np.float32)
    u = np.zeros((3,) + shape, dtype=np.float32)
    f = equilibrium(rho.astype(np.float64), u.astype(np.float64)).astype(np.float32)

    f[:, obstacle] = 0.0
    rho[obstacle] = 0.0

    empty = flags == FLAG_GAS
    f[:, empty] = 0.0
    rho[empty] = 0.0

    fill_time = np.full(shape, -1.0, dtype=np.float32)
    fill_time[flags == FLAG_FLUID] = 0.0

    return FillingState(
        shape=shape,
        dx_m=float(dx_m),
        dt_s=float(dt_s),
        f=f,
        rho=rho,
        u=u,
        mass=mass,
        flags=flags,
        fill_time_s=fill_time,
        temperature=temperature,
        porous=zone,
    )

def _dilate(mask: np.ndarray) -> np.ndarray:
    out: np.ndarray = mask.copy()
    for i in range(1, Q):
        c = LATTICE_VELOCITIES[i]
        out |= np.roll(mask, shift=(int(c[0]), int(c[1]), int(c[2])), axis=(0, 1, 2))
    return out

def fluid_fraction(state: FillingState) -> np.ndarray:
    denominator = state.rho.astype(np.float64) * state.porous.porosity_f64
    with np.errstate(divide="ignore", invalid="ignore"):
        eps = np.where(denominator > 0, state.mass / denominator, 0.0)
    return eps

def total_mass(state: FillingState) -> float:
    return float(state.mass[state.flags != FLAG_OBSTACLE].sum())

def total_mass_kg(state: FillingState, density_kg_m3: float) -> float:
    return total_mass(state) * density_kg_m3 * state.cell_volume_m3 / REFERENCE_DENSITY

def macroscopic(
    state: FillingState, force_lattice: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    f = state.f.astype(np.float64)
    rho = f.sum(axis=0)

    mom = np.zeros((3,) + state.shape, dtype=np.float64)
    for i in range(Q):
        for d in range(3):
            c = LATTICE_VELOCITIES[i, d]
            if c:
                mom[d] += c * f[i]

    if force_lattice is not None:
        mom = mom + 0.5 * np.asarray(force_lattice, dtype=np.float64)

    safe_rho = np.where(rho > 1e-12, rho, 1.0)
    v = mom / safe_rho

    if not state.porous.is_inert:
        v = resolve_porous_velocity(
            v,
            state.porous.drag_c0_f64,
            state.porous.drag_c1_f64,
        )

    dead = (state.flags == FLAG_GAS) | (state.flags == FLAG_OBSTACLE)
    v[:, dead] = 0.0
    return rho, v

def gas_density(state: FillingState) -> np.ndarray:
    rho_g = np.full(state.shape, GAS_DENSITY, dtype=np.float64)
    unprimed = ~state.porous.primed
    rho_g[unprimed] += state.porous.entry_offset[unprimed]
    return rho_g

def reconstruct_gas_populations(
    state: FillingState,
    f_streamed: np.ndarray,
    f_post_collision: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    streamed = np.asarray(f_streamed, dtype=np.float64)
    interface = state.flags == FLAG_INTERFACE
    if not np.any(interface):
        return streamed

    post = np.asarray(f_post_collision, dtype=np.float64)
    rho_g = gas_density(state)
    feq_gas = equilibrium(rho_g, u, porosity=state.porous.porosity_f64)
    opp = OPPOSITE.astype(np.intp)

    out = streamed.copy()
    is_gas = state.flags == FLAG_GAS
    for i in range(1, Q):
        c = LATTICE_VELOCITIES[i]
        neighbour_is_gas = np.roll(
            is_gas, shift=(-int(c[0]), -int(c[1]), -int(c[2])), axis=(0, 1, 2)
        )
        target = interface & neighbour_is_gas
        if not np.any(target):
            continue
        ibar = int(opp[i])
        out[ibar][target] = np.maximum(
            feq_gas[i][target] + feq_gas[ibar][target] - post[i][target],
            0.0,
        )
    return cast("np.ndarray", out)

def advance_mass(
    state: FillingState,
    f_post_collision: np.ndarray,
    f_streamed: np.ndarray,
) -> np.ndarray:
    outgoing = np.asarray(f_post_collision, dtype=np.float64)
    incoming = np.asarray(f_streamed, dtype=np.float64)
    eps = np.clip(fluid_fraction(state), 0.0, 1.0)
    fluid = state.flags == FLAG_FLUID
    wet = (state.flags == FLAG_INTERFACE) | fluid
    opp = OPPOSITE.astype(np.intp)

    axes = (0, 1, 2)
    delta = np.zeros(state.shape, dtype=np.float64)
    for i in range(1, Q):
        c = LATTICE_VELOCITIES[i]
        shift = (-int(c[0]), -int(c[1]), -int(c[2]))
        neighbour_wet = np.roll(wet, shift=shift, axis=axes)
        ibar = int(opp[i])

        weight = np.where(
            fluid | np.roll(fluid, shift=shift, axis=axes),
            1.0,
            0.5 * (eps + np.roll(eps, shift=shift, axis=axes)),
        )

        exchange = (incoming[ibar] - outgoing[i]) * weight
        delta += np.where(wet & neighbour_wet, exchange, 0.0)

    return state.mass + delta

def apply_transitions(
    state: FillingState, time_s: float, *, use_numba: bool = True
) -> Tuple[int, int]:
    to_fluid: np.ndarray
    to_gas: np.ndarray
    excess: np.ndarray
    if use_numba and NUMBA_AVAILABLE:
        to_fluid = np.empty(state.shape, dtype=bool)
        to_gas = np.empty(state.shape, dtype=bool)
        excess = np.empty(state.shape, dtype=np.float64)
        n_filled, n_emptied = transition_scan_kernel(
            state.mass,
            state.rho,
            state.porous.porosity_f64,
            state.flags,
            1.0 + TRANSITION_HYSTERESIS,
            -TRANSITION_HYSTERESIS,
            to_fluid,
            to_gas,
            excess,
        )
        filled = int(n_filled)
        emptied = int(n_emptied)
        if filled == 0 and emptied == 0:
            return 0, 0
    else:
        eps = fluid_fraction(state)
        interface = state.flags == FLAG_INTERFACE

        to_fluid = interface & (eps > 1.0 + TRANSITION_HYSTERESIS)
        to_gas = interface & (eps < -TRANSITION_HYSTERESIS)

        filled = int(np.count_nonzero(to_fluid))
        emptied = int(np.count_nonzero(to_gas))
        if filled == 0 and emptied == 0:
            return 0, 0

        full_mass = state.rho.astype(np.float64) * state.porous.porosity_f64

        excess = np.zeros(state.shape, dtype=np.float64)
        excess[to_fluid] = state.mass[to_fluid] - full_mass[to_fluid]
        excess[to_gas] = state.mass[to_gas]

        state.mass[to_fluid] = full_mass[to_fluid]
        state.mass[to_gas] = 0.0

    state.flags[to_fluid] = FLAG_FLUID
    state.flags[to_gas] = FLAG_GAS
    state.temperature[to_gas] = NO_TEMPERATURE
    state.f[:, to_gas] = 0.0
    state.rho[to_gas] = 0.0

    newly_filled = to_fluid & (state.fill_time_s < 0)
    state.fill_time_s[newly_filled] = np.float32(time_s)
    state.porous.primed[to_fluid] = True

    promote: Optional[np.ndarray] = None
    demote: Optional[np.ndarray] = None
    if use_numba and NUMBA_AVAILABLE:
        _C, Ci, _W, _OPP = lattice_tables()
        promote = np.empty(state.shape, dtype=bool)
        demote = np.empty(state.shape, dtype=bool)
        transition_front_kernel(state.flags, to_fluid, to_gas, Ci, promote, demote)
    else:
        if filled:
            promote = _dilate(to_fluid) & (state.flags == FLAG_GAS)
        if emptied:
            demote = _dilate(to_gas) & (state.flags == FLAG_FLUID)

    if promote is not None:
        state.flags[promote] = FLAG_INTERFACE
        _seed_promoted_cells(state, promote, use_numba=use_numba)

    if demote is not None and np.any(demote):
        state.flags[demote] = FLAG_INTERFACE

    _redistribute(state, excess, use_numba=use_numba)
    return filled, emptied

def _interface_neighbourhood(state: FillingState) -> Tuple[np.ndarray, np.ndarray]:
    _C, Ci, _W, _OPP = lattice_tables()
    gasless = np.empty(state.shape, dtype=bool)
    detached = np.empty(state.shape, dtype=bool)
    interface_neighbourhood_kernel(state.flags, Ci, gasless, detached)
    return gasless, detached

def heal_submerged_interface(
    state: FillingState,
    time_s: float,
    dwell_steps: int = HEAL_DWELL_STEPS,
    *,
    use_numba: bool = True,
) -> int:
    interface = state.flags == FLAG_INTERFACE
    streak = state._submerged_streak
    if streak is None or streak.shape != state.shape:
        streak = np.zeros(state.shape, dtype=np.int16)
        state._submerged_streak = streak
    if not np.any(interface):
        streak[...] = 0
        return 0

    if use_numba and NUMBA_AVAILABLE:
        submerged, _detached = _interface_neighbourhood(state)
    else:
        submerged = interface & ~_dilate(state.flags == FLAG_GAS)
    streak[...] = np.where(
        submerged, np.minimum(streak.astype(np.int32) + 1, dwell_steps), 0
    ).astype(np.int16)
    ripe = submerged & (streak >= dwell_steps)
    count = int(np.count_nonzero(ripe))
    if count == 0:
        return 0
    submerged = ripe
    streak[ripe] = 0

    full_mass = state.rho.astype(np.float64) * state.porous.porosity_f64
    deficit = np.zeros(state.shape, dtype=np.float64)
    deficit[submerged] = state.mass[submerged] - full_mass[submerged]

    state.mass[submerged] = full_mass[submerged]
    state.flags[submerged] = FLAG_FLUID
    newly_filled = submerged & (state.fill_time_s < 0)
    state.fill_time_s[newly_filled] = np.float32(time_s)
    state.porous.primed[submerged] = True

    _redistribute(state, deficit, use_numba=use_numba)
    return count

def ensure_interface_skin(
    state: FillingState, halo: np.ndarray, *, use_numba: bool = True
) -> int:
    promote = np.asarray(halo, dtype=bool) & (state.flags == FLAG_GAS)
    count = int(np.count_nonzero(promote))
    if count:
        state.flags[promote] = FLAG_INTERFACE
        _seed_promoted_cells(state, promote, use_numba=use_numba)
    return count

def reset_detached_interface_cells(
    state: FillingState, *, use_numba: bool = True
) -> int:
    interface = state.flags == FLAG_INTERFACE
    if not np.any(interface):
        return 0

    if use_numba and NUMBA_AVAILABLE:
        _gasless, detached = _interface_neighbourhood(state)
    else:
        wet = interface | (state.flags == FLAG_FLUID)
        support = np.zeros(state.shape, dtype=np.int16)
        for i in range(1, Q):
            c = LATTICE_VELOCITIES[i]
            support += np.roll(
                wet, shift=(int(c[0]), int(c[1]), int(c[2])), axis=(0, 1, 2)
            ).astype(np.int16)
        detached = interface & (support == 0)

    count = int(np.count_nonzero(detached))
    if count == 0:
        return 0

    rest = equilibrium(
        np.full(count, REFERENCE_DENSITY, dtype=np.float64),
        np.zeros((3, count), dtype=np.float64),
        porosity=state.porous.porosity[detached].astype(np.float64),
    )
    state.f[:, detached] = rest.astype(np.float32)
    state.rho[detached] = np.float32(REFERENCE_DENSITY)
    state.u[:, detached] = 0.0
    return count

def _mask_indices(
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(np.count_nonzero(mask))
    counts = np.empty(mask.shape[0] + 1, dtype=np.int64)
    ix = np.empty(n, dtype=np.int64)
    iy = np.empty(n, dtype=np.int64)
    iz = np.empty(n, dtype=np.int64)
    total = int(compact_mask_kernel(mask, counts, ix, iy, iz))
    return ix[:total], iy[:total], iz[:total]

def _seed_promoted_cells(
    state: FillingState, promote: np.ndarray, *, use_numba: bool = True
) -> None:
    if not np.any(promote):
        return
    indices: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    if use_numba and NUMBA_AVAILABLE:
        indices = _mask_indices(promote)
    _initialise_promoted_cells(
        state, promote, use_numba=use_numba, indices=indices
    )
    _initialise_promoted_temperature(
        state, promote, use_numba=use_numba, indices=indices
    )

def _initialise_promoted_cells(
    state: FillingState,
    promote: np.ndarray,
    *,
    use_numba: bool = True,
    indices: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> None:
    if not np.any(promote):
        return

    if use_numba and NUMBA_AVAILABLE:
        px, py, pz = indices if indices is not None else _mask_indices(promote)
        _C, Ci, _W, _OPP = lattice_tables()
        rho_seed = np.empty(px.size, dtype=np.float64)
        u_seed = np.empty((3, px.size), dtype=np.float64)
        promoted_donor_average_kernel(
            px, py, pz,
            state.flags,
            promote,
            state.rho,
            state.u,
            Ci,
            float(REFERENCE_DENSITY),
            rho_seed,
            u_seed,
        )
        seeded = equilibrium(
            rho_seed, u_seed, porosity=state.porous.porosity_f64[px, py, pz]
        )
        state.f[:, px, py, pz] = seeded.astype(np.float32)
        state.rho[px, py, pz] = rho_seed.astype(np.float32)
        return

    donor = (
        (state.flags == FLAG_FLUID) | (state.flags == FLAG_INTERFACE)
    ) & ~promote

    weight = donor.astype(np.float64)
    rho = state.rho.astype(np.float64) * weight
    u = state.u.astype(np.float64) * weight[None, :, :, :]

    rho_sum = np.zeros(state.shape, dtype=np.float64)
    u_sum = np.zeros((3,) + state.shape, dtype=np.float64)
    count = np.zeros(state.shape, dtype=np.float64)

    for i in range(1, Q):
        c = LATTICE_VELOCITIES[i]
        shift = (-int(c[0]), -int(c[1]), -int(c[2]))
        count += np.roll(weight, shift=shift, axis=(0, 1, 2))
        rho_sum += np.roll(rho, shift=shift, axis=(0, 1, 2))
        for d in range(3):
            u_sum[d] += np.roll(u[d], shift=shift, axis=(0, 1, 2))

    has_donor = promote & (count > 0)

    safe = np.where(count > 0, count, 1.0)
    rho_avg = np.where(has_donor, rho_sum / safe, REFERENCE_DENSITY)
    u_avg = np.where(has_donor[None, :, :, :], u_sum / safe[None, :, :, :], 0.0)

    seeded = equilibrium(rho_avg, u_avg, porosity=state.porous.porosity_f64)
    state.f[:, promote] = seeded[:, promote].astype(np.float32)
    state.rho[promote] = rho_avg[promote].astype(np.float32)

def _initialise_promoted_temperature(
    state: FillingState,
    promote: np.ndarray,
    *,
    use_numba: bool = True,
    indices: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> None:
    if not np.any(promote):
        return

    donor_has_temp: np.ndarray
    if use_numba and NUMBA_AVAILABLE:
        donor_has_temp = np.empty(state.shape, dtype=bool)
        promoted_donor_temperature_mask_kernel(
            state.flags, promote, state.temperature, donor_has_temp
        )
    else:
        donor = (
            (state.flags == FLAG_FLUID) | (state.flags == FLAG_INTERFACE)
        ) & ~promote
        donor_has_temp = donor & np.isfinite(state.temperature)
    if not np.any(donor_has_temp):
        return

    fallback_temp = float(np.mean(state.temperature[donor_has_temp]))

    if use_numba and NUMBA_AVAILABLE:
        px, py, pz = indices if indices is not None else _mask_indices(promote)
        _C, Ci, _W, _OPP = lattice_tables()
        avg_c = np.empty(px.size, dtype=np.float64)
        promoted_temperature_average_kernel(
            px, py, pz, donor_has_temp, state.temperature, Ci,
            float(fallback_temp), avg_c,
        )
        state.temperature[px, py, pz] = avg_c.astype(np.float32)
        return

    weight = donor_has_temp.astype(np.float64)
    temp = np.where(donor_has_temp, state.temperature.astype(np.float64), 0.0) * weight

    temp_sum = np.zeros(state.shape, dtype=np.float64)
    count = np.zeros(state.shape, dtype=np.float64)

    for i in range(1, Q):
        c = LATTICE_VELOCITIES[i]
        shift = (-int(c[0]), -int(c[1]), -int(c[2]))
        count += np.roll(weight, shift=shift, axis=(0, 1, 2))
        temp_sum += np.roll(temp, shift=shift, axis=(0, 1, 2))

    has_donor = promote & (count > 0)

    safe = np.where(count > 0, count, 1.0)
    avg = np.where(has_donor, temp_sum / safe, fallback_temp)
    state.temperature[promote] = avg[promote].astype(np.float32)

def _redistribute(
    state: FillingState, excess: np.ndarray, *, use_numba: bool = True
) -> None:
    if not np.any(excess):
        return

    if use_numba and NUMBA_AVAILABLE:
        _C, Ci, _W, _OPP = lattice_tables()
        share_buf = np.empty(state.shape, dtype=np.float64)
        stranded_buf = np.empty(state.shape, dtype=np.float64)
        redistribute_share_kernel(state.flags, excess, Ci, share_buf, stranded_buf)
        redistribute_gather_kernel(
            state.flags, share_buf, stranded_buf, Ci, state.mass
        )
        return

    interface = (state.flags == FLAG_INTERFACE).astype(np.float64)
    neighbour_count = np.zeros(state.shape, dtype=np.float64)
    for i in range(1, Q):
        c = LATTICE_VELOCITIES[i]
        neighbour_count += np.roll(
            interface, shift=(-int(c[0]), -int(c[1]), -int(c[2])), axis=(0, 1, 2)
        )

    share = np.where(neighbour_count > 0, excess / np.maximum(neighbour_count, 1.0), 0.0)
    stranded = np.where(neighbour_count > 0, 0.0, excess)

    for i in range(1, Q):
        c = LATTICE_VELOCITIES[i]
        contribution = np.roll(
            share * interface_shifted(interface, c),
            shift=(int(c[0]), int(c[1]), int(c[2])),
            axis=(0, 1, 2),
        )
        state.mass += contribution

    state.mass += stranded

def interface_shifted(interface: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.roll(interface, shift=(-int(c[0]), -int(c[1]), -int(c[2])), axis=(0, 1, 2))

def _neighbour_or_self(
    field: np.ndarray, wet: np.ndarray, axis: int, shift: int
) -> np.ndarray:
    rolled = np.roll(field, shift=shift, axis=axis)
    rolled_wet = np.roll(wet, shift=shift, axis=axis)
    return np.where(rolled_wet, rolled, field)

def advect_temperature(
    state: FillingState, alpha_lattice: float, *, use_numba: bool = True
) -> np.ndarray:
    if use_numba and NUMBA_AVAILABLE:
        out = np.empty(state.shape, dtype=np.float32)
        advect_temperature_kernel(
            state.temperature, state.u, state.flags, float(alpha_lattice), out
        )
        state.temperature = out
        return state.temperature

    wet = (state.flags == FLAG_INTERFACE) | (state.flags == FLAG_FLUID)
    if not np.any(wet):
        return state.temperature

    T = state.temperature.astype(np.float64)
    u = state.u.astype(np.float64)

    convective = np.zeros(state.shape, dtype=np.float64)
    laplacian = np.zeros(state.shape, dtype=np.float64)

    for axis in range(3):
        t_plus = _neighbour_or_self(T, wet, axis, -1)
        t_minus = _neighbour_or_self(T, wet, axis, 1)

        u_axis = u[axis]
        gradient = np.where(u_axis >= 0.0, T - t_minus, t_plus - T)
        convective += u_axis * gradient
        laplacian += t_plus + t_minus - 2.0 * T

    T_new = T - convective + alpha_lattice * laplacian
    state.temperature = np.where(wet, T_new, T).astype(np.float32)
    return state.temperature

def _wall_exposed_faces(state: FillingState) -> np.ndarray:
    cached = state._scratch_exposed_faces
    if cached is not None and cached.shape == state.shape:
        return cached

    obstacle = state.flags == FLAG_OBSTACLE
    exposed = np.zeros(state.shape, dtype=np.float64)
    for axis in range(3):
        for shift in (-1, 1):
            exposed += np.roll(obstacle, shift=shift, axis=axis)
    state._scratch_exposed_faces = exposed
    return exposed

def apply_wall_heat_loss(
    state: FillingState,
    *,
    htc_w_m2k,
    mold_temperature_c: float,
    density_kg_m3: float,
    specific_heat_j_kgk: float,
    use_numba: bool = True,
) -> np.ndarray:
    rho_cp_dx = density_kg_m3 * specific_heat_j_kgk * state.dx_m

    if use_numba and NUMBA_AVAILABLE:
        h_arr = np.asarray(htc_w_m2k, dtype=np.float64)
        h_is_field = h_arr.ndim == 3
        if h_is_field:
            h_field = np.ascontiguousarray(h_arr)
            h_scalar = 0.0
        else:
            h_field = np.zeros((1, 1, 1), dtype=np.float64)
            h_scalar = float(h_arr)
        out = np.empty(state.shape, dtype=np.float32)
        wall_heat_loss_kernel(
            state.temperature,
            state.flags,
            h_scalar,
            h_field,
            h_is_field,
            float(mold_temperature_c),
            float(state.dt_s),
            float(rho_cp_dx),
            out,
        )
        state.temperature = out
        return state.temperature

    wet = (state.flags == FLAG_INTERFACE) | (state.flags == FLAG_FLUID)
    exposed_faces = _wall_exposed_faces(state)

    losing = wet & (exposed_faces > 0)
    if not np.any(losing):
        return state.temperature

    h = np.broadcast_to(np.asarray(htc_w_m2k, dtype=np.float64), state.shape)
    T = state.temperature.astype(np.float64)
    coeff = h * exposed_faces * state.dt_s / rho_cp_dx
    delta = np.where(losing, -coeff * (T - mold_temperature_c), 0.0)
    state.temperature = np.where(wet, (T + delta), T).astype(np.float32)
    return state.temperature

def gravity_force(
    state: FillingState, units: LatticeUnits, direction: Tuple[float, float, float] = (0.0, 0.0, -1.0)
) -> np.ndarray:
    force = np.zeros((3,) + state.shape, dtype=np.float64)
    wet = (state.flags == FLAG_INTERFACE) | (state.flags == FLAG_FLUID)
    magnitude = units.gravity_lattice * np.sum(
        np.asarray(state.f, dtype=np.float64), axis=0
    )
    for d in range(3):
        if direction[d]:
            force[d] = np.where(wet, magnitude * direction[d], 0.0)
    return force

def step(
    state: FillingState,
    units: LatticeUnits,
    *,
    gravity_direction: Tuple[float, float, float] = (0.0, 0.0, -1.0),
    track_free_surface: bool = True,
    smagorinsky: bool = True,
    smagorinsky_cs: float = SMAGORINSKY_CS,
    use_numba: bool = True,
) -> FillingState:
    if use_numba and NUMBA_AVAILABLE:
        post, streamed, u64 = _collide_stream_numba(
            state, units, gravity_direction, smagorinsky, track_free_surface,
            smagorinsky_cs,
        )
        if track_free_surface:
            mass_out = _free_surface_scratch(state)
            C, Ci, W, OPP = lattice_tables()
            f32_out = np.empty((Q,) + state.shape, dtype=np.float32)
            free_surface_kernel(
                post,
                streamed,
                state.flags,
                state.rho,
                u64,
                state.porous.porosity_f64,
                state.porous.entry_offset_f64,
                state.porous.primed,
                C, Ci, W, OPP,
                float(GAS_DENSITY),
                state.mass,
                mass_out,
                f32_out,
            )
            state.f = f32_out
            state.mass[...] = mass_out
            state.step_count += 1
            apply_transitions(state, state.step_count * state.dt_s)
            heal_submerged_interface(state, state.step_count * state.dt_s)
            reset_detached_interface_cells(state)
        else:
            state.f = streamed.astype(np.float32)
            state.step_count += 1
        return state

    force = gravity_force(state, units, gravity_direction)
    rho, u = macroscopic(state, force)
    state.rho = rho.astype(np.float32)
    state.u = u.astype(np.float32)

    if not state.porous.is_inert:
        force = force + porous_drag_force(
            rho, u, state.porous.drag_c0_f64, state.porous.drag_c1_f64
        )

    porosity = state.porous.porosity_f64
    feq = equilibrium(rho, u, porosity=porosity)

    f_now = state.f.astype(np.float64)
    omega_plus: float | np.ndarray
    omega_minus: float | np.ndarray
    if smagorinsky:
        omega_plus, omega_minus = smagorinsky_relaxation(
            f_now,
            feq,
            rho,
            units.tau,
            smagorinsky_cs=smagorinsky_cs,
            suppress=state.porous.drag_c1 > 0.0,
            tau_minus_max=units.tau_minus_max,
        )
    else:
        omega_plus = units.omega_plus
        omega_minus = units.omega_minus

    source = guo_force_terms(u, force, omega_minus)
    post = collide_trt(f_now, feq, omega_plus, omega_minus, source=source)

    obstacle = state.flags == FLAG_OBSTACLE
    streamed = _stream_with_bounce_back(post, obstacle)

    if track_free_surface:
        streamed[:, state.flags == FLAG_GAS] = 0.0

    state.f = streamed.astype(np.float32)

    if track_free_surface:
        reconstructed = reconstruct_gas_populations(state, streamed, post, u)
        state.f = reconstructed.astype(np.float32)
        state.mass = advance_mass(state, post, reconstructed)
        state.step_count += 1
        apply_transitions(state, state.step_count * state.dt_s)
        heal_submerged_interface(state, state.step_count * state.dt_s)
        reset_detached_interface_cells(state)
    else:
        state.step_count += 1

    return state

def _collide_stream_numba(
    state: FillingState,
    units: LatticeUnits,
    gravity_direction: Tuple[float, float, float],
    smagorinsky: bool,
    track_free_surface: bool,
    smagorinsky_cs: float = SMAGORINSKY_CS,
):
    C, _c_int, W, OPP = lattice_tables()
    shape = state.shape

    post = state._scratch_post
    out = state._scratch_out
    if post is None or out is None or post.shape[1:] != shape:
        post = np.empty((Q,) + shape, dtype=np.float32)
        out = np.empty((Q,) + shape, dtype=np.float32)
        state._scratch_post = post
        state._scratch_out = out

    rho = state._scratch_rho64
    u = state._scratch_u64
    if rho is None or u is None or rho.shape != shape:
        rho = np.empty(shape, dtype=np.float64)
        u = np.empty((3,) + shape, dtype=np.float64)
        state._scratch_rho64 = rho
        state._scratch_u64 = u

    rho32 = np.empty(shape, dtype=np.float32)
    u32 = np.empty((3,) + shape, dtype=np.float32)

    collide_kernel(
        state.f,
        state.flags,
        state.porous.porosity_f64,
        state.porous.drag_c0_f64,
        state.porous.drag_c1_f64,
        C, W, OPP,
        float(units.tau),
        float(units.gravity_lattice),
        float(gravity_direction[0]),
        float(gravity_direction[1]),
        float(gravity_direction[2]),
        float(smagorinsky_cs),
        bool(smagorinsky),
        float(TRT_MAGIC_PARAMETER),
        float(units.tau_minus_max),
        rho, u, post, rho32, u32,
    )
    state.rho = rho32
    state.u = u32

    stream_kernel(post, state.flags, _c_int, OPP, bool(track_free_surface), out)
    return post, out, u

def _free_surface_scratch(state: FillingState) -> np.ndarray:
    shape = state.shape
    mass_out = state._scratch_mass_out
    if mass_out is None or mass_out.shape != shape:
        mass_out = np.empty(shape, dtype=np.float64)
        state._scratch_mass_out = mass_out
    return mass_out

def find_entrapped_gas(flags: np.ndarray, vent_mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    gas = flags == FLAG_GAS
    if not np.any(gas):
        return np.zeros_like(gas, dtype=bool)

    structure = ndimage.generate_binary_structure(3, 1)
    labels, n_components = ndimage.label(gas, structure=structure)
    if n_components == 0:
        return np.zeros_like(gas, dtype=bool)

    vented = np.asarray(vent_mask, dtype=bool) & gas
    vented_labels = (
        np.unique(labels[vented]) if np.any(vented) else np.empty(0, dtype=labels.dtype)
    )
    result: np.ndarray = gas & ~np.isin(labels, vented_labels)
    return result

def find_cold_shuts(
    fill_time_s: np.ndarray,
    temperature: np.ndarray,
    flags: np.ndarray,
    *,
    liquidus_c: float,
    arrival_gap_s: float,
    temperature_margin_c: float,
) -> np.ndarray:
    wet = (flags == FLAG_INTERFACE) | (flags == FLAG_FLUID)
    fill_time = np.asarray(fill_time_s, dtype=np.float64)
    temp = np.asarray(temperature, dtype=np.float64)

    valid = wet & (fill_time >= 0.0) & np.isfinite(temp)
    if not np.any(valid):
        return np.zeros_like(wet, dtype=bool)

    earlier_directions = np.zeros(flags.shape, dtype=np.int32)
    for axis in range(3):
        for shift in (1, -1):
            nbr_time = np.roll(fill_time, shift=shift, axis=axis)
            nbr_valid = np.roll(valid, shift=shift, axis=axis)
            earlier = valid & nbr_valid & ((fill_time - nbr_time) > arrival_gap_s)
            earlier_directions += earlier.astype(np.int32)

    meeting = valid & (earlier_directions >= 2)
    cold = valid & (temp <= liquidus_c + temperature_margin_c)
    result: np.ndarray = meeting & cold
    return result

def _stream_with_bounce_back(post: np.ndarray, obstacle: np.ndarray) -> np.ndarray:
    post = np.asarray(post, dtype=np.float64)
    out = np.zeros_like(post)
    opp = OPPOSITE.astype(np.intp)

    out[0] = post[0]
    for i in range(1, Q):
        c = LATTICE_VELOCITIES[i]
        forward = (int(c[0]), int(c[1]), int(c[2]))
        backward = (-int(c[0]), -int(c[1]), -int(c[2]))

        blocked = np.roll(obstacle, shift=backward, axis=(0, 1, 2))

        out[i] += np.roll(
            np.where(blocked, 0.0, post[i]), shift=forward, axis=(0, 1, 2)
        )
        out[int(opp[i])] += np.where(blocked, post[i], 0.0)

    out[:, obstacle] = 0.0
    return out
