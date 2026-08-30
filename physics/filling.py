"""
Free-surface lattice Boltzmann solver for mold filling.

Owns the solver state, the free-surface mass tracking, boundary conditions and
the step loop. The lattice primitives it is built from live in
:mod:`physics.fluid_flow`; the mesh-to-grid mapping and parameter translation
live in ``core/filling_stage.py``.

Layering
--------
``physics/`` may not import ``core/`` or ``config/``. This module therefore
knows nothing about voxel meshes, alloys or parameters -- it takes a dense
occupancy grid and SI scalars. That is what lets it be tested on synthetic
lattices with no mesh at all.

The free-surface method
-----------------------
Koerner-style volume tracking (Koerner et al., *J. Stat. Phys.* 121 (2005)
179-196), developed specifically for foundry filling. Each cell carries a mass
``m`` alongside its populations, and the fluid fraction ``eps = m / rho``
distinguishes empty gas, partially filled interface, and full fluid cells::

    dm_i(x, t+1) = (f_ibar(x + c_i, t) - f_i(x, t)) (eps(x) + eps(x + c_i)) / 2
    m(x, t+1)    = m(x, t) + sum_i dm_i

Interface cells have no neighbour populations on the gas side, so the missing
ones are reconstructed from the gas density.

**Mass conservation is the primary correctness invariant.** Free-surface LBM
leaks mass when cell transitions are handled sloppily, and it is the failure
mode most likely to go unnoticed -- the fill still looks plausible, it just
finishes with the wrong amount of metal. :func:`total_mass` exists to be
asserted every step in tests, and the transition bookkeeping deliberately
redistributes excess and deficit mass rather than discarding it.

Cell transitions use hysteresis so a cell hovering at eps = 1 does not
flip-flop between FLUID and INTERFACE on alternate steps, which would shed a
little mass each time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from physics.fluid_flow import (
    CS2,
    LATTICE_VELOCITIES,
    LATTICE_WEIGHTS,
    OPPOSITE,
    Q,
    LatticeUnits,
    collide_trt,
    equilibrium,
    guo_force_terms,
    resolve_porous_velocity,
    smagorinsky_relaxation,
)

# ---------------------------------------------------------------------------
# Cell flags
# ---------------------------------------------------------------------------

#: Solid: mold, chill or core. Bounce-back applies; no mass is tracked.
FLAG_OBSTACLE = np.uint8(0)
#: Empty cavity ahead of the melt front.
FLAG_GAS = np.uint8(1)
#: Partially filled. Carries the free surface.
FLAG_INTERFACE = np.uint8(2)
#: Completely filled with metal.
FLAG_FLUID = np.uint8(3)

#: Hysteresis band on the fluid fraction for cell transitions.
#:
#: A cell converts INTERFACE -> FLUID only above ``1 + kappa`` and
#: INTERFACE -> GAS only below ``-kappa``. Without the band a cell sitting
#: exactly at eps = 1 oscillates between states on alternate steps, and each
#: round trip sheds a little mass -- a slow leak that looks like a physical
#: result rather than a bug.
TRANSITION_HYSTERESIS = 1e-3

#: Reference gas density (atmospheric) in lattice units.
GAS_DENSITY = 1.0

#: Reference fluid density in lattice units. LBM is weakly compressible and
#: works in density perturbations about 1.
REFERENCE_DENSITY = 1.0


@dataclass
class PorousZone:
    """
    Per-cell porous-medium drag, for ceramic filter cells.

    All arrays are dense ``(nx, ny, nz)`` and take *inert* values outside
    filters, so the solver runs one branch-free path. ``drag_c0 = 0.5`` and
    ``drag_c1 = 0.0`` give ``u = v`` exactly, which makes a no-filter run
    bit-identical to a filter run with the drag switched off.

    Attributes:
        porosity: Void fraction; 1.0 outside filters.
        drag_c0: Guo-Zhao linear coefficient; 0.5 outside.
        drag_c1: Guo-Zhao quadratic coefficient; 0.0 outside.
        entry_offset: Capillary entry density offset; 0.0 outside.
        primed: Latched flag, set once a filter cell has filled.
    """

    porosity: np.ndarray
    drag_c0: np.ndarray
    drag_c1: np.ndarray
    entry_offset: np.ndarray
    primed: np.ndarray

    @classmethod
    def inert(cls, shape: Tuple[int, int, int]) -> "PorousZone":
        """A zone with no filters anywhere."""
        return cls(
            porosity=np.ones(shape, dtype=np.float32),
            drag_c0=np.full(shape, 0.5, dtype=np.float32),
            drag_c1=np.zeros(shape, dtype=np.float32),
            entry_offset=np.zeros(shape, dtype=np.float32),
            primed=np.zeros(shape, dtype=bool),
        )

    @property
    def is_inert(self) -> bool:
        """True when no cell imposes any drag."""
        return not bool(np.any(self.drag_c1)) and not bool(
            np.any(self.drag_c0 != 0.5)
        )


@dataclass
class FillingState:
    """
    Dense solver state on a uniform lattice.

    Populations are ``float32``: LBM is bandwidth-bound and halving the traffic
    is the single biggest constant-factor win. Mass is ``float64`` because it is
    the conserved quantity, and accumulating a global sum over millions of
    cells in float32 loses precisely the digits the conservation assertion is
    checking.

    Attributes:
        shape: Grid shape ``(nx, ny, nz)``.
        dx_m: Cell size, metres.
        dt_s: Timestep, seconds.
        f: Populations, ``(19, nx, ny, nz)`` float32.
        rho: Density, ``(nx, ny, nz)`` float32.
        u: Velocity, ``(3, nx, ny, nz)`` float32.
        mass: Cell mass, ``(nx, ny, nz)`` float64.
        flags: Cell state, ``(nx, ny, nz)`` uint8.
        fill_time_s: Arrival time per cell; -1 where never filled.
        porous: Filter drag; inert when there are no filters.
        step_count: Steps taken.
        inlet_mass: Total mass delivered through the inlet, for the
            conservation check.
    """

    shape: Tuple[int, int, int]
    dx_m: float
    dt_s: float
    f: np.ndarray
    rho: np.ndarray
    u: np.ndarray
    mass: np.ndarray
    flags: np.ndarray
    fill_time_s: np.ndarray
    porous: PorousZone
    step_count: int = 0
    inlet_mass: float = 0.0

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
) -> FillingState:
    """
    Build a solver state from an occupancy grid.

    Args:
        obstacle: Boolean ``(nx, ny, nz)``; True where the cell is solid.
        dx_m: Cell size in metres.
        dt_s: Timestep in seconds.
        initial_fluid: Optional boolean mask of cells that start full of metal.
            Cells adjacent to the resulting fluid region become INTERFACE.
        porous: Optional filter drag. Defaults to inert.

    Returns:
        An initialised :class:`FillingState` at rest.
    """
    obstacle = np.asarray(obstacle, dtype=bool)
    if obstacle.ndim != 3:
        raise ValueError(f"obstacle must be 3-D, got shape {obstacle.shape}")
    shape = (int(obstacle.shape[0]), int(obstacle.shape[1]), int(obstacle.shape[2]))

    flags = np.where(obstacle, FLAG_OBSTACLE, FLAG_GAS).astype(np.uint8)
    mass = np.zeros(shape, dtype=np.float64)

    if initial_fluid is not None:
        fluid = np.asarray(initial_fluid, dtype=bool) & ~obstacle
        flags[fluid] = FLAG_FLUID
        mass[fluid] = REFERENCE_DENSITY
        # Gas cells touching fluid become the initial interface layer.
        neighbours = _dilate(fluid)
        interface = neighbours & ~fluid & ~obstacle
        flags[interface] = FLAG_INTERFACE
        mass[interface] = 0.5 * REFERENCE_DENSITY

    rho = np.full(shape, REFERENCE_DENSITY, dtype=np.float32)
    u = np.zeros((3,) + shape, dtype=np.float32)
    f = equilibrium(rho.astype(np.float64), u.astype(np.float64)).astype(np.float32)

    # Solid cells hold no populations, ever. Seeding them at equilibrium and
    # relying on the solver to clear them afterwards does not work: the very
    # first streaming step pushes those populations out into the neighbouring
    # fluid before anything zeroes them, injecting one cell's worth of mass and
    # momentum per solid cell. On a closed box that is most of the domain, and
    # it shows up as a fluid "at rest" spontaneously reaching Ma ~ 0.1.
    f[:, obstacle] = 0.0
    rho[obstacle] = 0.0

    # Gas cells hold nothing either. This is not a tidiness point: leaving them
    # at equilibrium simulates the empty cavity as a fluid of density 1, which
    # then pushes back against the arriving metal. The cavity does not fill, the
    # pressure climbs, and the run diverges. In the Koerner scheme the gas phase
    # is not solved at all -- interface cells reconstruct what they need from
    # the gas density instead.
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
        porous=porous if porous is not None else PorousZone.inert(shape),
    )


def _dilate(mask: np.ndarray) -> np.ndarray:
    """Grow a boolean mask by one cell along every lattice direction."""
    out = mask.copy()
    for i in range(1, Q):
        c = LATTICE_VELOCITIES[i]
        out |= np.roll(mask, shift=(int(c[0]), int(c[1]), int(c[2])), axis=(0, 1, 2))
    return out


# ---------------------------------------------------------------------------
# Macroscopic quantities
# ---------------------------------------------------------------------------


def fluid_fraction(state: FillingState) -> np.ndarray:
    """
    Fill level per cell, ``eps = m / (rho * porosity)``.

    The porosity divisor matters inside a filter: a filter cell is full when its
    *pores* are full, not when it holds a whole cell's worth of metal. Omitting
    it loses about 15 % of each filter cell's volume from the mass budget, which
    on a 100 kg pour is a quarter-kilogram discrepancy -- enough to break the
    0.1 % conservation assertion and look like a leak elsewhere.
    """
    denominator = state.rho.astype(np.float64) * state.porous.porosity.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        eps = np.where(denominator > 0, state.mass / denominator, 0.0)
    return eps


def total_mass(state: FillingState) -> float:
    """
    Total metal in the domain, in lattice mass units.

    Obstacle cells are excluded; they never carry mass. This is the quantity
    that must track inlet-delivered mass to within 0.1 %.
    """
    return float(state.mass[state.flags != FLAG_OBSTACLE].sum())


def total_mass_kg(state: FillingState, density_kg_m3: float) -> float:
    """Total metal in kilograms."""
    return total_mass(state) * density_kg_m3 * state.cell_volume_m3 / REFERENCE_DENSITY


def macroscopic(
    state: FillingState, force_lattice: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recompute density and velocity from the populations.

    Applies the Guo half-force correction when a body force is present, and the
    Guo-Zhao porous drag solve where a filter imposes one.

    Args:
        state: Solver state.
        force_lattice: Optional body force, shape ``(3, nx, ny, nz)``.

    Returns:
        ``(rho, u)`` as float64 arrays.
    """
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
            state.porous.drag_c0.astype(np.float64),
            state.porous.drag_c1.astype(np.float64),
        )

    # Empty and solid cells carry no meaningful velocity.
    dead = (state.flags == FLAG_GAS) | (state.flags == FLAG_OBSTACLE)
    v[:, dead] = 0.0
    return rho, v


# ---------------------------------------------------------------------------
# Free surface
# ---------------------------------------------------------------------------


def gas_density(state: FillingState) -> np.ndarray:
    """
    Gas density seen by each interface cell.

    Atmospheric everywhere, plus the capillary entry offset where a filter cell
    has not yet primed. Once primed a cell latches and never resists again: a
    non-wetting liquid does not resist *leaving* into a larger space, and
    letting it re-arm would chatter the front.
    """
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
    """
    Supply the populations an interface cell is missing on its gas side.

    ``f_ibar(x) = f_i^eq(rho_G, u) + f_ibar^eq(rho_G, u) - f_i*(x)``

    Gas cells are not simulated, so nothing streams from them. Without this the
    interface would see a vacuum on its gas side; with it, the interface sees
    atmospheric pressure and is free to advance.

    ``f_i*(x)`` is the **post-collision** population at x -- the one that just
    departed toward the gas. It is *not* the post-streaming value at x, which is
    what arrived from ``x - c_i`` and is an unrelated quantity. Substituting the
    post-streaming array leaves the interface momentum balance wrong in a way
    that makes the free surface behave like a wall: a suspended column of metal
    simply refuses to fall, while mass conservation stays perfect because
    nothing is lost.

    Args:
        state: Solver state, for the flags and the gas density.
        f_streamed: Post-streaming populations, the array being corrected.
        f_post_collision: Populations after collision, before streaming.
        u: Cell velocity used for the gas-side equilibrium.

    Returns:
        The corrected populations.
    """
    streamed = np.asarray(f_streamed, dtype=np.float64)
    interface = state.flags == FLAG_INTERFACE
    if not np.any(interface):
        return streamed

    post = np.asarray(f_post_collision, dtype=np.float64)
    rho_g = gas_density(state)
    feq_gas = equilibrium(rho_g, u, porosity=state.porous.porosity.astype(np.float64))
    opp = OPPOSITE.astype(np.intp)

    out = streamed.copy()
    is_gas = state.flags == FLAG_GAS
    for i in range(1, Q):
        c = LATTICE_VELOCITIES[i]
        # Is the neighbour at x + c_i a gas cell?
        neighbour_is_gas = np.roll(
            is_gas, shift=(-int(c[0]), -int(c[1]), -int(c[2])), axis=(0, 1, 2)
        )
        target = interface & neighbour_is_gas
        if not np.any(target):
            continue
        ibar = int(opp[i])
        out[ibar][target] = (
            feq_gas[i][target] + feq_gas[ibar][target] - post[i][target]
        )
    return out


def advance_mass(
    state: FillingState,
    f_post_collision: np.ndarray,
    f_streamed: np.ndarray,
) -> np.ndarray:
    """
    Koerner mass exchange over one step.

    ``dm_i = (f_ibar(x + c_i, t) - f_i(x, t)) (eps(x) + eps(x + c_i)) / 2``

    Both populations in that difference are **post-collision, pre-streaming**
    values: ``f_ibar`` at the neighbour is the mass heading *into* x, and
    ``f_i`` at x is the mass heading *out*. After streaming the incoming one has
    arrived and is readable as ``f_ibar(x)``, but the outgoing one has already
    left, so it must be read from the pre-streaming array. Hence two arguments.

    Taking both from the post-streaming array instead makes every opposite pair
    contribute ``(f_ibar - f_i)`` and ``(f_i - f_ibar)``, which cancel exactly:
    the total mass change is then identically zero and the front never advances,
    while everything else about the run looks healthy.

    Exchange happens only between cells that both hold metal (FLUID or
    INTERFACE). A FLUID cell surrounded by fluid nets zero, so only the front
    actually moves mass.

    Args:
        state: Solver state, before the mass update.
        f_post_collision: Populations after collision, before streaming.
        f_streamed: Populations after streaming.

    Returns:
        The new mass field.
    """
    outgoing = np.asarray(f_post_collision, dtype=np.float64)
    incoming = np.asarray(f_streamed, dtype=np.float64)
    eps = fluid_fraction(state)
    wet = (state.flags == FLAG_INTERFACE) | (state.flags == FLAG_FLUID)
    opp = OPPOSITE.astype(np.intp)

    delta = np.zeros(state.shape, dtype=np.float64)
    for i in range(1, Q):
        c = LATTICE_VELOCITIES[i]
        shift = (-int(c[0]), -int(c[1]), -int(c[2]))
        neighbour_wet = np.roll(wet, shift=shift, axis=(0, 1, 2))
        eps_neighbour = np.roll(eps, shift=shift, axis=(0, 1, 2))
        ibar = int(opp[i])
        exchange = (incoming[ibar] - outgoing[i]) * 0.5 * (eps + eps_neighbour)
        delta += np.where(wet & neighbour_wet, exchange, 0.0)

    return state.mass + delta


def apply_transitions(state: FillingState, time_s: float) -> Tuple[int, int]:
    """
    Convert cells whose fill level has crossed the hysteresis band.

    Excess mass above full, and deficit below empty, is redistributed to
    neighbouring interface cells rather than clipped. Clipping is the classic
    free-surface mass leak: each conversion silently discards a sliver, and over
    tens of thousands of steps the fill quietly loses metal.

    Newly filled cells convert their gas neighbours into interface cells, which
    is how the front advances.

    Args:
        state: Solver state, mutated in place.
        time_s: Current physical time, recorded as the arrival time.

    Returns:
        ``(filled, emptied)`` counts.
    """
    eps = fluid_fraction(state)
    interface = state.flags == FLAG_INTERFACE

    to_fluid = interface & (eps > 1.0 + TRANSITION_HYSTERESIS)
    to_gas = interface & (eps < -TRANSITION_HYSTERESIS)

    filled = int(np.count_nonzero(to_fluid))
    emptied = int(np.count_nonzero(to_gas))
    if filled == 0 and emptied == 0:
        return 0, 0

    full_mass = state.rho.astype(np.float64) * state.porous.porosity.astype(np.float64)

    excess = np.zeros(state.shape, dtype=np.float64)
    excess[to_fluid] = state.mass[to_fluid] - full_mass[to_fluid]
    excess[to_gas] = state.mass[to_gas]

    state.mass[to_fluid] = full_mass[to_fluid]
    state.mass[to_gas] = 0.0
    state.flags[to_fluid] = FLAG_FLUID
    state.flags[to_gas] = FLAG_GAS

    newly_filled = to_fluid & (state.fill_time_s < 0)
    state.fill_time_s[newly_filled] = np.float32(time_s)
    state.porous.primed[to_fluid] = True

    # A filled cell's gas neighbours join the interface -- this is how the
    # front advances.
    if filled:
        promote = _dilate(to_fluid) & (state.flags == FLAG_GAS)
        state.flags[promote] = FLAG_INTERFACE
        _initialise_promoted_cells(state, promote)

    # An emptied cell's fluid neighbours must *retreat* to the interface.
    #
    # The scheme's central invariant is that no FLUID cell is ever directly
    # adjacent to a GAS cell: a fluid cell has no free surface, so it never
    # reconstructs its gas-side populations, and one sitting against a void
    # streams into nothing. Handling only the advancing direction lets a
    # receding front tear the surface open -- the interface layer empties, no
    # new one forms, and the run reaches NaN within a few hundred steps.
    # Retreat is not a rare case: it happens wherever the fill accelerates away
    # from a wall, which is most of a real pour.
    if emptied:
        demote = _dilate(to_gas) & (state.flags == FLAG_FLUID)
        if np.any(demote):
            state.flags[demote] = FLAG_INTERFACE

    _redistribute(state, excess)
    return filled, emptied


def _initialise_promoted_cells(state: FillingState, promote: np.ndarray) -> None:
    """
    Give a freshly promoted GAS -> INTERFACE cell something to start from.

    A cell that has just joined the interface holds no populations -- gas cells
    are not simulated. Left alone it has density zero, and the very next
    macroscopic pass divides by it. The Koerner scheme initialises such cells
    from the local equilibrium of their already-wet neighbours, which is both
    the physically sensible value and the one that keeps the front smooth.

    Skipping this is not a subtle accuracy loss: the front advances into cells
    with rho = 0, the velocity solve produces inf, and the run reaches NaN
    within a few hundred steps.
    """
    if not np.any(promote):
        return

    donor = (
        (state.flags == FLAG_FLUID) | (state.flags == FLAG_INTERFACE)
    ) & ~promote
    if not np.any(donor):
        return

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
    if not np.any(has_donor):
        return

    safe = np.where(count > 0, count, 1.0)
    rho_avg = np.where(has_donor, rho_sum / safe, REFERENCE_DENSITY)
    u_avg = np.where(has_donor[None, :, :, :], u_sum / safe[None, :, :, :], 0.0)

    seeded = equilibrium(rho_avg, u_avg, porosity=state.porous.porosity.astype(np.float64))
    state.f[:, has_donor] = seeded[:, has_donor].astype(np.float32)
    state.rho[has_donor] = rho_avg[has_donor].astype(np.float32)


def _redistribute(state: FillingState, excess: np.ndarray) -> None:
    """
    Push leftover mass into neighbouring interface cells.

    Conserves the total exactly when at least one interface neighbour exists.
    When none does -- an isolated cell -- the mass stays put rather than being
    dropped, which keeps the conservation invariant true at the cost of a cell
    that is briefly over- or under-full.
    """
    if not np.any(excess):
        return

    interface = (state.flags == FLAG_INTERFACE).astype(np.float64)
    # Count interface neighbours of each donor cell.
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
        # Send this cell's share to the neighbour in direction i.
        contribution = np.roll(
            share * interface_shifted(interface, c),
            shift=(int(c[0]), int(c[1]), int(c[2])),
            axis=(0, 1, 2),
        )
        state.mass += contribution

    state.mass += stranded


def interface_shifted(interface: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Indicator of "the neighbour in direction ``c`` is an interface cell"."""
    return np.roll(interface, shift=(-int(c[0]), -int(c[1]), -int(c[2])), axis=(0, 1, 2))


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


def gravity_force(
    state: FillingState, units: LatticeUnits, direction: Tuple[float, float, float] = (0.0, 0.0, -1.0)
) -> np.ndarray:
    """
    Body force from gravity, in lattice units, applied only to wet cells.

    Applying gravity to gas cells would accelerate the void, which shows up as
    spurious motion ahead of the front.
    """
    force = np.zeros((3,) + state.shape, dtype=np.float64)
    wet = (state.flags == FLAG_INTERFACE) | (state.flags == FLAG_FLUID)
    magnitude = units.gravity_lattice * state.rho.astype(np.float64)
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
) -> FillingState:
    """
    Advance the solver one timestep.

    Order: macroscopic moments -> collide (TRT + Guo forcing) -> stream ->
    bounce-back on obstacles -> reconstruct gas-side populations -> mass
    exchange -> cell transitions.

    Args:
        state: Solver state, mutated in place and returned.
        units: Lattice unit system.
        gravity_direction: Unit vector for gravity. Default is -Z.
        track_free_surface: When False the free surface is skipped entirely and
            the solver runs single-phase. Used by the closed-box benchmarks
            (hydrostatic rest, Poiseuille), where introducing a free surface
            would confound what is being measured.

    Returns:
        The same state object, advanced.
    """
    force = gravity_force(state, units, gravity_direction)
    rho, u = macroscopic(state, force)
    state.rho = rho.astype(np.float32)
    state.u = u.astype(np.float32)

    porosity = state.porous.porosity.astype(np.float64)
    feq = equilibrium(rho, u, porosity=porosity)

    f_now = state.f.astype(np.float64)
    omega_plus: float | np.ndarray
    omega_minus: float | np.ndarray
    if smagorinsky:
        # Eddy viscosity, evaluated per cell from the local non-equilibrium
        # stress. Suppressed inside filters: the Forchheimer drag is already
        # the closure for pore-scale inertial losses there, and stacking a
        # subgrid model on top double-counts them.
        omega_plus, omega_minus = smagorinsky_relaxation(
            f_now,
            feq,
            rho,
            units.tau,
            suppress=state.porous.drag_c1 > 0.0,
        )
    else:
        omega_plus = units.omega_plus
        omega_minus = units.omega_minus

    source = guo_force_terms(u, force, omega_minus)
    post = collide_trt(f_now, feq, omega_plus, omega_minus, source=source)

    obstacle = state.flags == FLAG_OBSTACLE
    streamed = _stream_with_bounce_back(post, obstacle)

    if track_free_surface:
        # The gas phase is not simulated, and that invariant has to be
        # maintained every step -- not just at construction. Streaming happily
        # pushes populations from the interface into the empty cavity, where
        # they accumulate, acquire a density, and push back on the arriving
        # metal. The front then stalls with the fluid decelerating instead of
        # falling, while mass conservation still looks perfect, because nothing
        # is lost -- it is merely parked in cells that should not exist.
        streamed[:, state.flags == FLAG_GAS] = 0.0

    state.f = streamed.astype(np.float32)

    if track_free_surface:
        reconstructed = reconstruct_gas_populations(state, streamed, post, u)
        state.f = reconstructed.astype(np.float32)
        state.mass = advance_mass(state, post, reconstructed)
        state.step_count += 1
        apply_transitions(state, state.step_count * state.dt_s)
    else:
        state.step_count += 1

    return state


def _stream_with_bounce_back(post: np.ndarray, obstacle: np.ndarray) -> np.ndarray:
    """
    Stream, reflecting any population that would enter a solid cell.

    Halfway bounce-back: a population aimed at an obstacle is written back into
    the originating cell in the opposite direction, which places the wall midway
    between the two nodes.
    """
    post = np.asarray(post, dtype=np.float64)
    out = np.zeros_like(post)
    opp = OPPOSITE.astype(np.intp)

    out[0] = post[0]
    for i in range(1, Q):
        c = LATTICE_VELOCITIES[i]
        forward = (int(c[0]), int(c[1]), int(c[2]))
        backward = (-int(c[0]), -int(c[1]), -int(c[2]))

        # ``blocked[x]`` asks whether the cell at x + c_i is solid, so the
        # decision is made at the *source* cell. Reflecting at the destination
        # instead and then clearing solid cells silently destroys the
        # population -- a mass leak that grows with wall area and is invisible
        # until the conservation assertion catches it.
        blocked = np.roll(obstacle, shift=backward, axis=(0, 1, 2))

        out[i] += np.roll(
            np.where(blocked, 0.0, post[i]), shift=forward, axis=(0, 1, 2)
        )
        out[int(opp[i])] += np.where(blocked, post[i], 0.0)

    # Solid cells never hold populations of their own.
    out[:, obstacle] = 0.0
    return out
