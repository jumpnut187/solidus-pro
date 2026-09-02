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

if NUMBA_AVAILABLE:  # pragma: no cover - import-time branch
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

#: Consecutive gasless steps before a submerged interface cell is healed.
#:
#: :func:`heal_submerged_interface` exists for cells the front has closed over
#: PERMANENTLY -- those stay gasless forever, so waiting costs them 200 steps
#: (tens of milliseconds of pour) out of a run of hundreds of thousands. A cell
#: on an active, sloshing front is gasless only transiently, and healing it on
#: sight ping-pongs with apply_transitions' retreat path: heal -> a neighbour
#: drains -> demote -> refill -> heal, each cycle pulling the deficit out of
#: the advancing front. Measured on Test_Model/Filling_Test before the dwell:
#: 828,368 heal events over 327,200 steps on a 4,346-cell grid (~190 per
#: cell), and 34% less metal delivered through the inlet than the same run
#: without the rule. The dwell keys the rule to persistence, which is the
#: property that actually distinguishes a stranded cell from a busy one.
HEAL_DWELL_STEPS = 200

#: Reference gas density (atmospheric) in lattice units.
GAS_DENSITY = 1.0

#: Reference fluid density in lattice units. LBM is weakly compressible and
#: works in density perturbations about 1.
REFERENCE_DENSITY = 1.0

#: Sentinel for "this cell has never held metal". A cell that has never
#: filled must carry no temperature at all -- seeding it at ambient would
#: silently turn a misrun into a plausible-looking cold casting. NaN rather
#: than e.g. -1 because it poisons arithmetic loudly: any stencil that
#: accidentally reads across into an unfilled cell produces NaN, not a quiet
#: wrong number.
NO_TEMPERATURE = np.float32(np.nan)


@dataclass
class PorousZone:
    """
    Per-cell porous-medium drag, for ceramic filter cells.

    All arrays are dense ``(nx, ny, nz)`` and take *inert* values outside
    filters, so the solver runs one branch-free path. ``drag_c0 = 0.5`` and
    ``drag_c1 = 0.0`` give ``u = v`` exactly, which makes a no-filter run
    bit-identical to a filter run with the drag switched off.

    The four coefficient arrays are stored float32 to match the populations but
    are consumed in float64 everywhere -- the solver used to spell that
    ``.astype(np.float64)`` at a dozen call sites, all inside the per-step loop.
    None of them changes after the zone is built (``core/filling_stage.py``
    stamps the filter cells in while assembling the zone, before the first
    step), so the float64 views below are built once and cached. That is
    9 of the 38 measured per-step ``astype`` calls at a 120^3 grid.

    **The cache assumes immutability.** Mutating ``porosity``, ``drag_c0``,
    ``drag_c1`` or ``entry_offset`` in place after the solver has taken a step
    leaves the float64 view stale, and the two would then disagree silently.
    Build the zone, then hand it over. ``primed`` is the one array that *does*
    change during a run, which is exactly why it is not cached.

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

    # Lazily built float64 views of the four static coefficient arrays. Typed
    # fields rather than plain attribute assignment so mypy sees them.
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

    @property
    def porosity_f64(self) -> np.ndarray:
        """``porosity`` in float64, built once. See the class docstring."""
        cached = self._porosity_f64
        if cached is None or cached.shape != self.porosity.shape:
            cached = self.porosity.astype(np.float64)
            self._porosity_f64 = cached
        return cached

    @property
    def drag_c0_f64(self) -> np.ndarray:
        """``drag_c0`` in float64, built once. See the class docstring."""
        cached = self._drag_c0_f64
        if cached is None or cached.shape != self.drag_c0.shape:
            cached = self.drag_c0.astype(np.float64)
            self._drag_c0_f64 = cached
        return cached

    @property
    def drag_c1_f64(self) -> np.ndarray:
        """``drag_c1`` in float64, built once. See the class docstring."""
        cached = self._drag_c1_f64
        if cached is None or cached.shape != self.drag_c1.shape:
            cached = self.drag_c1.astype(np.float64)
            self._drag_c1_f64 = cached
        return cached

    @property
    def entry_offset_f64(self) -> np.ndarray:
        """``entry_offset`` in float64, built once. See the class docstring."""
        cached = self._entry_offset_f64
        if cached is None or cached.shape != self.entry_offset.shape:
            cached = self.entry_offset.astype(np.float64)
            self._entry_offset_f64 = cached
        return cached


@dataclass
class FillingState:
    """
    Dense solver state on a uniform lattice.

    Populations are ``float32``: LBM is bandwidth-bound and halving the traffic
    is the single biggest constant-factor win. Mass is ``float64`` because it is
    the conserved quantity, and accumulating a global sum over millions of
    cells in float32 loses precisely the digits the conservation assertion is
    checking.

    That claim used to be half true, and the history is worth keeping because
    it is the shape of the mistake. Storing ``f`` float32 while every working
    buffer inside the step stayed float64 bought nothing: the numba path rebuilt
    ``state.f.astype(np.float64)`` every step, so the narrow field cost a
    conversion pass and a 19 x N allocation *on top of* the float64 traffic it
    was supposed to avoid. The first pass fixed the read side (the kernels cast
    per element as they read). This one fixes the write side: the population
    buffers the kernels move between them -- ``_scratch_post`` and
    ``_scratch_out`` -- are float32 too, and the float64 ``_scratch_f_out``
    that ``free_surface_kernel`` used to reconstruct into is gone entirely. At
    120^3 that is 817.4 -> 316.4 MiB of resident scratch, and the populations
    moving between the kernels really are half the bytes now rather than
    nominally. Whole step: 43.2 -> 36.7 ms median, six interleaved A/B rounds,
    won all six. The win was smaller than the buffers suggest because the
    collision kernel turned out to be compute-bound rather than
    bandwidth-bound; removing the duplicate arithmetic that diagnosis exposed
    took the step on to 32.0 ms. See :func:`_collide_stream_numba` for the
    per-kernel measurements of both passes.

    **Storage is float32; arithmetic is float64.** Every kernel reads a
    population through an explicit ``np.float64`` cast and rounds once, at the
    store -- the same rounding ``state.f`` has always received at each step
    boundary, now also applied to the mid-step buffers. Making the kernels
    *compute* in float32 remains deliberately not done: it would change the
    numerics rather than the traffic. Mass is float64 end to end, from
    ``mass_in`` through the exchange accumulator to ``state.mass``, and the
    Koerner exchange's pairwise cancellation stays bit-exact under the narrower
    populations -- see :func:`physics.fluid_flow.free_surface_kernel` for the
    argument and tests/test_filling_parity.py for the measurement.

    The price is that the numba and NumPy step paths are no longer bit-identical:
    they now round populations at different points, so the parity tests carry
    measured tolerances instead of ``assert_array_equal``. The NumPy path stays
    float64 throughout on purpose -- it is the numerics reference, not a
    performance path.

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
        temperature: Degrees Celsius, ``(nx, ny, nz)`` float32. NaN
            (:data:`NO_TEMPERATURE`) wherever the cell has never held metal --
            see F2 in ``docs/filling_implementation.md`` section 4.5. Unlike
            the LBM fields this is not a lattice-unit quantity: Celsius is
            already dx/dt-independent, so no conversion happens at the
            lattice-unit boundary the way it does for velocity or density.
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
    temperature: np.ndarray
    porous: PorousZone
    step_count: int = 0
    inlet_mass: float = 0.0

    # Numba-path scratch buffers, allocated once and reused across steps (see
    # _collide_stream_numba and _free_surface_scratch). Declared here, rather
    # than injected via plain attribute assignment, so they are typed instead
    # of tripping mypy's attr-defined check on every scratch access.
    # float32, like state.f itself -- see the dtype paragraph in the class
    # docstring. The float64 _scratch_f_out that used to sit beside them is
    # gone; free_surface_kernel reconstructs straight into the float32 buffer
    # that becomes state.f.
    _scratch_post: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _scratch_out: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _scratch_mass_out: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    # Float64 moment buffers the collision kernel writes. Held rather than
    # np.empty'd per step: at 120^3 that was 55 MB of allocate-and-free churn
    # every step for arrays whose contents are overwritten in full anyway.
    # Read-only inputs to nothing, so they cannot alias an input array.
    _scratch_rho64: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _scratch_u64: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    # Per-cell count of consecutive steps spent as a gasless interface cell,
    # for heal_submerged_interface's dwell. Lazily allocated, int16, capped at
    # the dwell so it cannot overflow on long runs.
    _submerged_streak: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    # Exposed mold-wall face count per cell, for apply_wall_heat_loss' NumPy
    # fallback. Obstacle cells never change after create_state, so this is
    # static geometry -- see _wall_exposed_faces for the measurement that
    # motivated caching it.
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
    """
    Build a solver state from an occupancy grid.

    Args:
        obstacle: Boolean ``(nx, ny, nz)``; True where the cell is solid.
        dx_m: Cell size in metres.
        dt_s: Timestep in seconds.
        initial_fluid: Optional boolean mask of cells that start full of metal.
            Cells adjacent to the resulting fluid region become INTERFACE.
        porous: Optional filter drag. Defaults to inert.
        initial_temperature_c: Temperature to seed ``initial_fluid`` (and its
            interface skin) at. ``None`` -- the default -- leaves every cell at
            :data:`NO_TEMPERATURE`, which is correct for a synthetic lattice
            with no metal in it at all. Ignored when ``initial_fluid`` is not
            given, since there is then nothing to seed.

    Returns:
        An initialised :class:`FillingState` at rest.
    """
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
        # Full is ``rho * porosity``, which is what fluid_fraction,
        # apply_transitions and heal_submerged_interface all divide by. Seeding
        # REFERENCE_DENSITY flat put a cell that *starts* FLUID inside a filter
        # at eps = 1/porosity -- 1.176 at the default 0.85 -- so the fixture
        # began 18 % over full and the exchange had to work that off. Test 2
        # starts empty and never saw it; a restart or a fixture that pre-fills
        # a filter does.
        #
        # Read through the raw float32 array and widen the selected cells, not
        # through ``zone.porosity_f64``. The float64 view caches on first access
        # and invalidates only on a *shape* change (see PorousZone's docstring),
        # so touching it here would move the cache fill from the first ``step``
        # to construction -- and a caller that builds the state and *then* fills
        # in ``porous.porosity`` would silently solve at porosity 1 while the
        # float32 array the stage's inlet pin reads said 0.85. Widening first is
        # exact: ``porosity_f64`` is ``porosity.astype(np.float64)``, so these
        # are the same numbers.
        por = zone.porosity
        mass[fluid] = REFERENCE_DENSITY * por[fluid].astype(np.float64)
        # Gas cells touching fluid become the initial interface layer.
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
        temperature=temperature,
        porous=zone,
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
    denominator = state.rho.astype(np.float64) * state.porous.porosity_f64
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
            state.porous.drag_c0_f64,
            state.porous.drag_c1_f64,
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
    feq_gas = equilibrium(rho_g, u, porosity=state.porous.porosity_f64)
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
        # Floored at zero, matching free_surface_kernel in physics/fluid_flow.
        # The closure does not bound this below, and a negative population is
        # a negative density contribution that the equilibrium's quadratic-in-u
        # terms then amplify -- see that kernel for the measured cascade this
        # prevents, and for why the resulting small mass gain is the better
        # trade. The two paths must agree or tests/test_filling_parity.py
        # fails, which is exactly the guard that should catch a one-sided fix.
        out[ibar][target] = np.maximum(
            feq_gas[i][target] + feq_gas[ibar][target] - post[i][target],
            0.0,
        )
    return out


def advance_mass(
    state: FillingState,
    f_post_collision: np.ndarray,
    f_streamed: np.ndarray,
) -> np.ndarray:
    """
    Koerner mass exchange over one step.

    ``dm_i = (f_ibar(x + c_i, t) - f_i(x, t)) * w_i``

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
    INTERFACE). The pair weight ``w_i`` follows Koerner et al. (2005), Table 1:
    the fill-fraction average applies to interface-interface pairs *only*, and
    any pair involving a FLUID cell exchanges at weight 1.

    That distinction is load-bearing. At weight 1 the sum over directions
    telescopes -- ``sum_i f_ibar(x + c_i)`` is everything streaming in and
    ``sum_i f_i(x)`` is everything streaming out -- so ``dm`` is exactly the
    cell's own density change over the step. A FLUID cell's mass therefore
    tracks its density identically and cannot drift. Wall bounce-back does not
    spoil it: a direction whose neighbour is an obstacle streams ``f_ibar(x) =
    f_i(x)`` straight back, so its pair contributes zero and skipping obstacle
    neighbours is identical to including them.

    Weighting a FLUID cell's exchanges by ``(eps_x + eps_nb)/2`` instead breaks
    the identity. ``eps = m / (rho * porosity)`` is 1 only to round-off, and the
    weight *scales the exchange amplitude*, so drift in eps amplifies its own
    exchanges -- a random walk with step size proportional to the error.
    Measured with the sprue of ``Test_Model/Filling_Test`` running completely
    full (332 of 332 cells FLUID): per-cell FLUID mass reached +-70,000 and wet
    mass alternated sample-to-sample until the divergence guard fired at step
    251,000. The old docstring claimed "a FLUID cell surrounded by fluid nets
    zero"; under eps weighting that is true only for perfectly uniform eps,
    which is exactly what the drift destroys.

    Interface-interface pairs keep the eps average, clamped to ``[0, 1]``: the
    transition hysteresis lets an interface cell sit slightly outside the band,
    and an out-of-band eps would amplify exchanges the same way.

    Pairs touching a filter are **not** an exception. They used to be, and the
    measurement that removed the exception is in the body's comment.

    Args:
        state: Solver state, before the mass update.
        f_post_collision: Populations after collision, before streaming.
        f_streamed: Populations after streaming.

    Returns:
        The new mass field.
    """
    outgoing = np.asarray(f_post_collision, dtype=np.float64)
    incoming = np.asarray(f_streamed, dtype=np.float64)
    eps = np.clip(fluid_fraction(state), 0.0, 1.0)
    fluid = state.flags == FLAG_FLUID
    wet = (state.flags == FLAG_INTERFACE) | fluid
    opp = OPPOSITE.astype(np.intp)

    # There is no filter special case here, and the reason is measured.
    #
    # Every pair with a porous cell on either side used to be diverted to
    # ``0.5 * (eps_A + eps_B)`` with eps *unclamped*, on the argument that a
    # full porous cell holds ``rho * porosity`` rather than ``rho`` and so must
    # not be handed a weight of 1. The argument is about the right quantity and
    # the branch does the opposite of what it claims. The weight scales the
    # exchange amplitude and is itself a function of the cell's own eps, so any
    # eps error amplifies its own exchanges -- positive feedback, with nothing
    # to arrest it, because every transition and heal scan selects on
    # FLAG_INTERFACE only and a FLUID cell's mass is written by this exchange
    # and by nothing else. Measured on the ``Test 2`` assembly at the filter
    # outlet: the effective weight, which the scheme requires to be 1, reached
    # +10.7 and -13.0, the sign inverted on 7.5 % of outflow samples, and 786
    # of 2,009 FLUID samples of the filter's bottom plane sat over full against
    # 1 of 754 INTERFACE samples of the same cells. eps_max in the porous zone
    # climbed 1.05 -> 7.76 at 2.7e-4/step while the donors fell to eps = -9.5,
    # and the run died on a negative-mass cell ~5,000 steps later. It also
    # never achieved its stated purpose: at eps ~ 1 the eps average *is* 1, so
    # the branch only starts to differ once eps has already drifted. Seeded at
    # exactly eps = 1 on a closed porous box it stays there for 6,000 steps;
    # seeded 5 % high it grows +25.8 %, at a rate independent of the seed.
    #
    # Two variants that look like the fix are worse and are deliberately not
    # here. Weighting the FLUID branch by ``0.5 * (p_A + p_B)`` so mass tracks
    # ``rho * porosity`` measured 6.52e-2 against HEAD's 6.38e-2, because the
    # direction sum only telescopes into ``drho`` when the weight is uniform
    # across a cell's directions -- a porosity-dependent weight takes three
    # values across a filter face and the residual is then set by the
    # individual populations (~1e-2), three orders above the ``(1 - p) * drho``
    # drift it was meant to remove. Any spatially non-uniform weight has that
    # defect. Re-pinning porous FLUID cells to ``rho * porosity`` through
    # ``_redistribute`` is exact where a free surface crosses the filter and
    # does *nothing at all* on a fully submerged one -- the state Test 2's
    # filter is in for most of the run -- because ``_redistribute`` has no
    # INTERFACE recipient and strands the correction back on the donor.
    #
    # What is accepted instead: at weight 1 a porous FLUID cell's mass tracks
    # ``rho``, so it drifts from ``rho * porosity`` by ``(1 - p) * drho`` per
    # step. That increment telescopes, so the whole-fill drift is ``(1 - p)``
    # times the total density excursion -- a few tenths of a percent of a
    # cell's capacity, measured +0.014 % over 6,000 steps -- and it cannot
    # amplify, because with an eps-free weight there is no path from eps back
    # into the exchange amplitude. A bounded 0.1 % offset in place of an
    # exponential divergence is the trade Koerner's scheme makes.

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
    """
    Convert cells whose fill level has crossed the hysteresis band.

    Excess mass above full, and deficit below empty, is redistributed to
    neighbouring interface cells rather than clipped. Clipping is the classic
    free-surface mass leak: each conversion silently discards a sliver, and over
    tens of thousands of steps the fill quietly loses metal.

    Newly filled cells convert their gas neighbours into interface cells, which
    is how the front advances.

    Cost, and why this function has kernels at all
    ---------------------------------------------
    The scan at the top runs on every step of a run and finds nothing on most
    of them: 3.2 ms of a 47 ms step at 120^3, spent proving that no cell moved
    -- now 0.26 ms.
    The rest of the function runs only when something converts, and then it
    costs **240 ms** -- five times the whole rest of the step, now 8.8 ms. See the
    "Cell-transition bookkeeping" comment in ``physics/fluid_flow.py`` for the
    per-phase measurements. Both halves are now kernelled, with the NumPy
    bodies kept as the ``NUMBA_AVAILABLE == False`` fallback.

    The two paths compute ``promote`` and ``demote`` at the same point, before
    either is applied. The NumPy path used to compute ``demote`` further down,
    after ``flags[promote] = FLAG_INTERFACE``; that write only moves cells out
    of GAS and ``demote`` selects on FLUID, so the mask is identical either side
    of it and hoisting it costs nothing but keeps the two branches readable
    side by side.

    Args:
        state: Solver state, mutated in place.
        time_s: Current physical time, recorded as the arrival time.
        use_numba: Take the kernels when they are available. Mirrors
            :func:`step`'s parameter of the same name.

    Returns:
        ``(filled, emptied)`` counts.
    """
    to_fluid: np.ndarray
    to_gas: np.ndarray
    excess: np.ndarray
    if use_numba and NUMBA_AVAILABLE:
        # Every cell of all three is written by the kernel, so np.empty is
        # safe -- and at 120^3 np.empty does not touch the pages, so the
        # kernel's parallel write is also the first touch.
        to_fluid = np.empty(state.shape, dtype=bool)
        to_gas = np.empty(state.shape, dtype=bool)
        excess = np.empty(state.shape, dtype=np.float64)
        # The kernel snaps the converting cells' mass in the same sweep, which
        # is why there is no separate full_mass array on this side.
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
    # An emptied cell holds no metal any more, so it holds no temperature any
    # more either -- the same "no metal, no temperature" rule create_state
    # applies at t=0 applies for the rest of the run.
    state.temperature[to_gas] = NO_TEMPERATURE
    # ...and it holds no populations or density any more either, which makes
    # the "a gas cell holds nothing" invariant true at the moment of
    # transition rather than one stage later.
    #
    # The populations are belt-and-braces: both streaming paths already zero
    # a gas cell's f every step (the numpy branch in step(), the clear_gas
    # branch in fluid_flow.stream_kernel). The *density* is not covered
    # anywhere -- macroscopic() zeroes a gas cell's velocity but computes its
    # density unconditionally from sum(f) -- so without this a retreated cell
    # kept its interface-era rho until something happened to overwrite it.
    # Measured: gas cells carried stale density from step 26 of a plain
    # column fill. Nothing load-bearing reads it (a gas cell is never a
    # seeding donor), so this is an invariant repair, not a bug fix -- it is
    # NOT what caused the divergence at fine resolution.
    state.f[:, to_gas] = 0.0
    state.rho[to_gas] = 0.0

    newly_filled = to_fluid & (state.fill_time_s < 0)
    state.fill_time_s[newly_filled] = np.float32(time_s)
    state.porous.primed[to_fluid] = True

    # Both fronts, read off the same flags: a filled cell's gas neighbours
    # advance to the interface, an emptied cell's fluid neighbours retreat to
    # it. See the docstring for why computing the retreating one here rather
    # than after the advance is equivalent.
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

    # A filled cell's gas neighbours join the interface -- this is how the
    # front advances.
    if promote is not None:
        state.flags[promote] = FLAG_INTERFACE
        _seed_promoted_cells(state, promote, use_numba=use_numba)

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
    if demote is not None and np.any(demote):
        state.flags[demote] = FLAG_INTERFACE

    _redistribute(state, excess, use_numba=use_numba)
    return filled, emptied


def _interface_neighbourhood(state: FillingState) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run :func:`physics.fluid_flow.interface_neighbourhood_kernel` on the
    current flags.

    Returns:
        ``(gasless, detached)`` -- boolean masks of INTERFACE cells with no GAS
        neighbour and with no wet neighbour respectively, over the 18-direction
        lattice neighbourhood. False everywhere else.

    Fresh arrays, not held buffers, and called separately by each caller. The
    flags mutate between :func:`apply_transitions`,
    :func:`heal_submerged_interface` and
    :func:`reset_detached_interface_cells` -- each converts cells the next one
    then has to see -- so a shared result computed once per step would answer
    the question about the wrong lattice. Two bool grids is 3.5 MB at 120^3;
    the 36 full-grid ``np.roll`` calls they replace were 13 ms of a 106 ms step.
    """
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
    """
    Convert a *submerged* interface cell -- one with no gas neighbour -- to FLUID.

    Koerner's scheme reinitialises the flags every step so the interface stays
    exactly one cell thick between metal and gas. This implementation had only
    half of that: :func:`apply_transitions` converts a cell whose *fill level*
    crosses the band, and promotes/demotes its neighbours to keep the layer
    closed. Nothing converted a cell that the front had simply closed over.

    Such a cell can never convert on its own. Under Koerner et al. (2005),
    Table 1 -- see :func:`advance_mass` -- any pair involving a FLUID cell
    exchanges at weight 1, so once every neighbour is wet or solid the
    direction sum telescopes and ``dm`` is exactly the cell's own ``drho``.
    Its mass therefore tracks its density and ``eps = m / (rho * porosity)``
    freezes at whatever the passing front left, forever, a hair either side of
    a threshold it cannot reach. It is not stuck for a physical reason: the
    metal around it is full, and the only thing saying otherwise is a flag.

    Measured before this rule existed:

    * ``_walled_tower(nz=4, width=4)`` ended a head-mode pour with 40 cells
      FLUID and 24 INTERFACE, and **all twenty-four** of those had no gas
      neighbour left, at eps between 0.02 and 0.99. It reported
      ``time_capped`` with 24 misruns in a cavity that was physically full.
    * ``Test_Model/Full_Gating_Test`` stranded 796 such cells -- 8.6 % of the
      metal -- at the 45 s cap.
    * ``Test_Model/Filling_Test`` ended with 1,191 of its 1,322 remaining
      interface cells frozen this way.

    The old eps-pair weighting hid the gap rather than filling it: it let a
    FLUID cell's mass drift, and the drift happened to carry these cells past
    the transition threshold and convert them. That same drift is the +-70,000
    per-cell runaway :class:`~tests.test_filling_stability.FluidMassTracksDensityTests`
    documents. Removing it exposed this; it did not cause it.

    Mass bookkeeping mirrors :func:`apply_transitions`' ``to_fluid`` path
    exactly -- mass snapped to ``rho * porosity``, arrival time stamped if
    unset, filter cell primed -- with the sign reversed: the cell is normally
    *under* full, so the excess handed to :func:`_redistribute` is a deficit,
    and the neighbouring interface cells give the metal up rather than receive
    it. ``_redistribute`` is linear in that argument and needs no sign branch.
    Where a healed cell has no interface neighbour left to draw on -- a whole
    submerged pocket healing at once, which is the common case -- the deficit
    is stranded back onto the cell itself. Nothing is created and nothing is
    lost: total mass is conserved exactly, and the price is a FLUID cell that
    holds less than a full cell's metal. That is the honest outcome; the metal
    genuinely is not there, and inventing it would be a leak dressed up as a
    fill.

    The reverse rule -- an interface cell with no *fluid* neighbour becoming
    GAS -- is deliberately not implemented here. Only the freeze was measured.

    A cell converts only after ``dwell_steps`` CONSECUTIVE gasless steps -- see
    :data:`HEAL_DWELL_STEPS` for the measured ping-pong that forced the dwell.
    A permanently stranded cell is gasless forever and merely waits; a cell on
    an active front loses its streak the moment the surface re-opens beside it,
    so the rule stays out of live flow entirely.

    Args:
        state: Solver state, mutated in place.
        time_s: Current physical time, recorded as the arrival time.
        dwell_steps: Consecutive gasless steps required before conversion.
            Tests of the conversion mechanics pass 1 to fire immediately.
        use_numba: Take the neighbourhood kernel when it is available. Mirrors
            :func:`step`'s parameter of the same name.

    Returns:
        Number of cells healed, for instrumentation. Over a real fill this
        count decays to zero rather than plateauing: a healed cell is FLUID,
        and only a genuinely draining neighbour can send it back to the
        interface, through ``apply_transitions``' existing retreat path.
    """
    interface = state.flags == FLAG_INTERFACE
    streak = state._submerged_streak
    if streak is None or streak.shape != state.shape:
        streak = np.zeros(state.shape, dtype=np.int16)
        state._submerged_streak = streak
    if not np.any(interface):
        streak[...] = 0
        return 0

    # The same neighbourhood apply_transitions advances the front with, so
    # "has a gas neighbour" means the same thing in both directions. The two
    # spellings are held to exact equality by
    # tests/test_filling_parity.py::InterfaceNeighbourhoodParityTests -- the
    # kernel counts x + c_i where the dilate reaches x - c_i, which gives the
    # same answer only because the 18-direction set is closed under negation.
    if use_numba and NUMBA_AVAILABLE:
        submerged, _detached = _interface_neighbourhood(state)
    else:
        submerged = interface & ~_dilate(state.flags == FLAG_GAS)
    # The streak is the dwell's whole state: persistence, not instantaneous
    # topology, is what separates a stranded cell from a sloshing one. Capped
    # at the dwell so an int16 cannot overflow on a 400,000-step run.
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

    # No gas neighbour means no gas *cell* to promote either, so unlike
    # apply_transitions there is no front to advance here -- the layer this
    # cell belonged to has already closed.
    _redistribute(state, deficit, use_numba=use_numba)
    return count


def ensure_interface_skin(
    state: FillingState, halo: np.ndarray, *, use_numba: bool = True
) -> int:
    """
    Promote gas cells in ``halo`` to interface, seeded from their wet neighbours.

    The scheme's central invariant is that no FLUID cell is ever directly
    adjacent to a GAS cell -- see :func:`apply_transitions`, which maintains it
    for every cell *it* converts, in both the advancing and the retreating
    direction. A caller that writes ``FLAG_FLUID`` by hand is outside that
    bookkeeping and has to maintain the invariant for its own cells. The
    filling stage pins its inlet cells FLUID every step, so it is exactly such
    a caller.

    The cost of skipping it, measured on a 2x2x12 column with a pressure inlet:
    the cell above the inlet drained to gas, streaming zeroed it, and the inlet
    stopped receiving every population arriving with ``c_z = -1`` -- one sixth
    of the total weight. Its measured z-momentum jumped from -0.009 (falling)
    to +0.147 (rising) in a single step, and the whole column reversed and ran
    upward at the Mach cap. A boundary that measures its own velocity cannot
    afford to measure it through a hole.

    Args:
        state: Solver state, mutated in place.
        halo: Boolean mask of cells to check. Cells that are not gas are left
            alone, so this is idempotent and safe to call every step.
        use_numba: Passed to the promoted-cell seeding. Mirrors :func:`step`'s
            parameter of the same name.

    Returns:
        Number of cells promoted.
    """
    promote = np.asarray(halo, dtype=bool) & (state.flags == FLAG_GAS)
    count = int(np.count_nonzero(promote))
    if count:
        state.flags[promote] = FLAG_INTERFACE
        _seed_promoted_cells(state, promote, use_numba=use_numba)
    return count


def reset_detached_interface_cells(
    state: FillingState, *, use_numba: bool = True
) -> int:
    """
    Return fully detached interface cells to rest.

    An interface cell with no wet neighbour at all -- no FLUID, no INTERFACE,
    nothing but gas and obstacle in all 18 directions -- is a droplet smaller
    than the grid can represent. Every one of its populations comes from the
    gas-side reconstruction, which at that topology has no real neighbour
    populations to draw on, so its velocity is decided entirely by a closure
    that has nothing to close against.

    Measured on the reference assembly: 5-12 such cells at any moment, 0.1-0.3%
    of the metal, all at the gate/casting junction, each carrying 16-18 of 18
    gas neighbours -- and they were the *only* cells exceeding the Mach guard.
    The bulk was never unstable: FLUID velocities stayed flat at 0.038-0.046
    across 108,000 steps while these few ran to 0.44. A Mach limiter on the
    equilibrium would have damped the healthy bulk and missed this entirely.

    Populations are replaced with the rest equilibrium; mass is deliberately
    left alone, so this removes spurious momentum without inventing or
    destroying metal. Suppresses the runaway completely -- late-run peak
    velocity 0.043 against a 0.05 design target, down from 0.31 -- while
    touching a mean of 3.7 cells per step.

    It does **not** unstick the ~25% fill plateau, which was measured
    separately and has some other cause. This repairs an artifact; it is not a
    fix for the fill.

    Args:
        state: Solver state, mutated in place.
        use_numba: Take the neighbourhood kernel when it is available. Mirrors
            :func:`step`'s parameter of the same name.

    Returns:
        Number of cells reset, for instrumentation.
    """
    interface = state.flags == FLAG_INTERFACE
    if not np.any(interface):
        return 0

    # Only the *count* of wet neighbours matters, never the direction, which is
    # why the kernel's x + c_i and the rolls' x - c_i below give the same mask:
    # the 18-direction set is closed under negation. Held to exact equality by
    # tests/test_filling_parity.py::InterfaceNeighbourhoodParityTests.
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
    """
    Where a boolean mask is True, as three index arrays in C order.

    ``np.nonzero`` for the same thing, but through
    :func:`~physics.fluid_flow.compact_mask_kernel`, which is the same scan run
    in parallel: 2.95 ms against 0.41 ms on a (120, 120, 120) mask holding the
    ~450 cells a real promotion step produces. Read that
    kernel's docstring for why the ordering is guaranteed to match NumPy's,
    which is the property that lets the two be swapped for each other.

    The output arrays are sized from ``np.count_nonzero`` -- a vectorised
    popcount, and cheap -- rather than at the mask's full size, so the caller
    does not carry a 40 MB index buffer to address a few hundred cells. The
    kernel recounts as its first phase; the returned slice is defensive against
    the two ever disagreeing.
    """
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
    """
    Populations *and* temperature for the same freshly promoted cells.

    The two halves are separate functions because they are separately
    meaningful -- and separately load-bearing, each with its own measured
    failure mode if skipped. They are called together from both promotion sites,
    though, and on the kernel path both need the promoted cells' indices, which
    costs a full-grid scan to find. This locates them once and hands the same
    list to both.
    """
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

    That is exactly what used to happen to a *subset* of promoted cells. The
    donor mask excludes cells promoted in the same step, so a cell whose only
    wet neighbours were promoted alongside it has no donor at all. Those cells
    were left untouched at rho = 0 -- while the fallback value they needed was
    already being computed and then discarded. A whole front advancing into
    open space promotes in sheets, so this is routine, not a corner case:
    every cell behind the leading edge of such a sheet has only
    same-step neighbours. They now fall back to reference density at rest,
    which is what an unconstrained cell of fresh metal should start from.

    The NumPy body below is the reference and the ``NUMBA_AVAILABLE == False``
    fallback. It was also, by a wide margin, the most expensive thing in the
    solver on any step where the front advanced: 156.6 ms of a 240 ms
    ``apply_transitions`` at 120^3. Two reasons, and the kernel path removes
    both rather than parallelising them. It ran 90 full-grid ``np.roll`` sums --
    18 directions x five fields -- to produce an average that only the few
    hundred promoted cells ever read; and it then built a full-domain
    ``(19, nx, ny, nz)`` float64 equilibrium, 262 MB and 87 of those 156 ms, to
    take 447 columns out of it. The kernel path loops over the promoted cells'
    indices and runs :func:`~physics.fluid_flow.equilibrium` on the same
    compressed arrays, which is what :func:`reset_detached_interface_cells` has
    always done for its handful of cells.
    """
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
        # equilibrium is elementwise over the trailing shape, so running it on
        # the compressed (n,) arrays gives the same values the full-domain call
        # produced -- it just does not compute the 1.7 million columns that were
        # about to be discarded. Boolean-mask assignment visits the True cells
        # in C order, which is the order np.nonzero returns, so the two
        # spellings of the scatter agree cell for cell.
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

    # Written across every promoted cell, not just the ones with a donor.
    # rho_avg and u_avg already carry the reference-density-at-rest fallback
    # for the rest, so the donorless cells are seeded rather than left at zero.
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
    """
    Give a freshly promoted GAS -> INTERFACE cell a starting temperature.

    Mirrors :func:`_initialise_promoted_cells`: a cell joining the interface
    for the first time has no temperature of its own -- it was gas a moment
    ago -- so it inherits one from whichever already-wet neighbours are
    feeding it, averaged rather than copied from a single donor so the
    thermal front stays as smooth as the population seed does.

    Skipping this leaves a promoted cell at :data:`NO_TEMPERATURE` (NaN) even
    though its flag says FLUID-adjacent metal, which then poisons the next
    advection step's stencil the moment a neighbour reads across into it.

    The NumPy body below is the reference and the ``NUMBA_AVAILABLE == False``
    fallback; 28.4 ms of a 240 ms ``apply_transitions`` at 120^3, mostly 36
    full-grid rolls feeding an average that a few hundred cells read. The kernel
    path keeps the whole-domain fallback mean in NumPy on purpose -- see the
    comment on ``fallback_temp``.
    """
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

    # The whole-domain mean of metal already in the mold, as the fallback for
    # promoted cells with no *local* donor -- the same sheet-promotion case
    # that leaves _initialise_promoted_cells without one. Leaving them at
    # NO_TEMPERATURE is what this function's own docstring warns about: the
    # NaN survives until a neighbour reads across into it, and then spreads
    # through the advection stencil a cell per step.
    #
    # Deliberately np.mean on BOTH paths, not a prange reduction. A parallel
    # reduction sums in whatever order the threads finish in; np.mean sums
    # pairwise. Over a million-cell domain those disagree in the last bits, and
    # this is the one value in the function where that would be visible, so the
    # kernel path takes the reduction it cannot beat for free.
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
    """
    Push leftover mass into neighbouring interface cells.

    Conserves the total exactly when at least one interface neighbour exists.
    When none does -- an isolated cell -- the mass stays put rather than being
    dropped, which keeps the conservation invariant true at the cost of a cell
    that is briefly over- or under-full.

    The NumPy body below is the reference and the ``NUMBA_AVAILABLE == False``
    fallback: 54 full-grid ``np.roll`` calls, 38.0 ms at 120^3, on every step
    where any cell transitions. The kernel path is the same algorithm turned
    receiver-side -- one pass for each donor's share, one for each interface
    cell's gather -- so it needs no atomics and does not depend on thread
    order. See :func:`~physics.fluid_flow.redistribute_gather_kernel` for the
    direction mapping and for the single divergence, which is confined to
    already-diverged lattices.

    Mass conservation is the invariant this function exists to protect, so the
    kernel path was held to the arithmetic rather than merely to a tolerance:
    the gather adds the 18 shares into the receiver's own mass in the same
    direction order the roll-sum does, and adds ``stranded`` last, which is
    where the NumPy path's final ``state.mass += stranded`` sits.
    """
    if not np.any(excess):
        return

    if use_numba and NUMBA_AVAILABLE:
        _C, Ci, _W, _OPP = lattice_tables()
        share_buf = np.empty(state.shape, dtype=np.float64)
        stranded_buf = np.empty(state.shape, dtype=np.float64)
        redistribute_share_kernel(state.flags, excess, Ci, share_buf, stranded_buf)
        # Two launches, not one: the gather reads its neighbours' `share`, so
        # every donor's share has to be final before any receiver looks. Fusing
        # them would make a receiver see some neighbours before and some after
        # their update -- the same hazard free_surface_kernel's mass_in/mass_out
        # split exists to avoid.
        redistribute_gather_kernel(
            state.flags, share_buf, stranded_buf, Ci, state.mass
        )
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
# Thermal coupling (F2)
# ---------------------------------------------------------------------------
#
# Advected as a passive scalar with explicit finite-difference upwind
# advection plus a 6-point Laplacian, not a second (D3Q7) population set. The
# design doc offers either; upwind is stable at these Mach numbers (the LBM
# solver already caps the lattice velocity at 0.05, an order of magnitude
# under the 1/6 diffusion stability limit measured in
# tests/test_filling_thermal.py) and is a fraction of the code, with no extra
# populations to keep conserved.
#
# Both functions work in *lattice steps* (dx = dt = 1), matching how the LBM
# fields themselves are differenced, with one exception: temperature is
# stored in degrees Celsius throughout, never converted to a lattice unit, because
# it is already dx/dt-independent -- unlike velocity or density there is no
# "lattice temperature". ``alpha_lattice`` is the caller's SI thermal
# diffusivity rescaled the same way ``nu_lattice`` is (``alpha_SI * dt_s /
# dx_m**2``), so it drops into the stencil as a plain dimensionless number.


def _neighbour_or_self(
    field: np.ndarray, wet: np.ndarray, axis: int, shift: int
) -> np.ndarray:
    """
    The neighbour's value in direction ``shift`` along ``axis``, or the cell's
    own value where that neighbour is not wet.

    This is what keeps NaN (gas and obstacle cells carry no temperature) out
    of the stencil: a metal cell sitting at the free surface or against a
    mold wall sees a zero-gradient (Neumann) boundary in that direction
    instead of reading across into a cell that has none. Heat still leaves
    through a wall -- that is :func:`apply_wall_heat_loss`'s job, applied as a
    separate lumped term, exactly as the governing equation in
    ``docs/filling_implementation.md`` section 4.5 splits ``alpha * lap(T)``
    from ``q_wall`` into two additive terms rather than one.
    """
    rolled = np.roll(field, shift=shift, axis=axis)
    rolled_wet = np.roll(wet, shift=shift, axis=axis)
    return np.where(rolled_wet, rolled, field)


def advect_temperature(
    state: FillingState, alpha_lattice: float, *, use_numba: bool = True
) -> np.ndarray:
    """
    Advance the temperature field one lattice step: upwind advection plus
    diffusion, over metal cells only.

    ``T_new = T - dt(u . grad T) + alpha * dt * lap(T)``, with ``dt = 1`` and
    axis-aligned first-order upwind differencing for the gradient (the
    stencil direction follows the sign of the local velocity component, so
    information always comes from upstream) and a standard 6-point Laplacian
    for the diffusion term. Both use ``state.u``, which is already in lattice
    units and current as of this step's collision.

    Cells that are neither FLUID nor INTERFACE (no metal) are left exactly as
    they were -- typically :data:`NO_TEMPERATURE` -- rather than being
    advected. A gas cell has no temperature to advect, and computing one
    would silently manufacture a value for a cell the fill has not reached.

    The NumPy body below is the reference implementation and the
    ``NUMBA_AVAILABLE == False`` fallback. It is also, at a 120^3 grid, 20 ms
    of a 106 ms step -- twelve full-grid ``np.roll`` calls on a single core,
    which made it the largest single item in the serial tail that capped
    thread scaling. :func:`physics.fluid_flow.advect_temperature_kernel` is the
    parallel twin; the two are held together by
    ``tests/test_filling_parity.py::ThermalKernelParityTests``.

    Args:
        state: Solver state, mutated in place.
        alpha_lattice: Thermal diffusivity in lattice units
            (``alpha_SI * dt_s / dx_m**2``), the same rescaling
            :func:`physics.fluid_flow.lattice_units` applies to viscosity.
        use_numba: Take the kernel when it is available. Mirrors
            :func:`step`'s parameter of the same name so a caller -- or a
            parity test -- can pin either path.

    Returns:
        The updated ``state.temperature``.
    """
    if use_numba and NUMBA_AVAILABLE:
        # A fresh output array, not a held scratch buffer: every cell reads its
        # neighbours' pre-step values, so the output must never be the input,
        # and the reference path also rebinds a new array here. One (nx, ny, nz)
        # float32 allocation is 7 MB at 120^3 -- noise against the 12 full-grid
        # rolls it replaces.
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
        # roll(shift=-1) brings the value at index+1 to position index -- the
        # same convention physics.filling already uses elsewhere (see
        # interface_shifted) for "the neighbour in the +1 direction".
        t_plus = _neighbour_or_self(T, wet, axis, -1)
        t_minus = _neighbour_or_self(T, wet, axis, 1)

        u_axis = u[axis]
        # Upwind: take the gradient from whichever side the flow is coming
        # from. Using a centred difference here is the textbook way this kind
        # of scheme goes unstable -- upwind is the whole reason this is safe
        # at CFL = 0.05 without a flux limiter.
        gradient = np.where(u_axis >= 0.0, T - t_minus, t_plus - T)
        convective += u_axis * gradient
        laplacian += t_plus + t_minus - 2.0 * T

    T_new = T - convective + alpha_lattice * laplacian
    state.temperature = np.where(wet, T_new, T).astype(np.float32)
    return state.temperature


def _wall_exposed_faces(state: FillingState) -> np.ndarray:
    """
    Number of the six face neighbours that are OBSTACLE, per cell.

    Cached on the state, because this is **static geometry**. The obstacle set
    is written once by :func:`create_state` and nothing in the solver ever
    converts a cell to or from ``FLAG_OBSTACLE`` -- the transitions move cells
    between GAS, INTERFACE and FLUID, and the filling stage's inlet pin writes
    FLAG_FLUID. Rebuilding it cost six full-grid ``np.roll`` calls every step,
    which at a 120^3 grid was most of ``apply_wall_heat_loss``' 10 ms of a
    106 ms step, spent re-deriving a mold that had not moved.

    Six faces, not the 18-direction lattice neighbourhood: a diagonal touch
    shares no area with the mold, so it conducts nothing. This is the same
    distinction :func:`find_entrapped_gas` makes for gas connectivity.

    Only the NumPy fallback needs this. The kernel counts the six neighbours
    per cell as it goes -- six uint8 reads inside a loop that is already
    parallel is cheaper than a cache lookup plus the risk of the cache being
    wrong.

    Returns:
        Float64 ``(nx, ny, nz)``, values 0-6. Float rather than integer because
        it is multiplied straight into the float64 loss coefficient, and
        promoting there instead would change nothing but cost a pass.
    """
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
    """
    Lose heat through faces that border an OBSTACLE cell (the mold wall).

    ``dT = -h (T - T_mold) dt / (rho cp dx)``, applied once per exposed face,
    so a corner cell -- more than one face against the mold -- loses
    proportionally more than a cell with a single flat wall. ``h`` is left to
    the caller rather than looked up here: ``physics/`` cannot import
    ``core/solidification_params.ThermalBoundaryParams``, so
    ``core/filling_stage.py`` resolves ``htc_gating`` vs ``htc_cast_interface``
    per cell and passes the result down as a plain array.

    The NumPy body below is the reference and the ``NUMBA_AVAILABLE == False``
    fallback; :func:`physics.fluid_flow.wall_heat_loss_kernel` is the parallel
    twin. Either way the exposed-face count is no longer rebuilt from scratch
    every step -- see :func:`_wall_exposed_faces`.

    Args:
        state: Solver state, mutated in place. Uses ``state.dx_m``,
            ``state.dt_s`` and ``state.flags``.
        htc_w_m2k: Interface heat-transfer coefficient, W/(m^2 K). Either a
            scalar or a ``(nx, ny, nz)`` array -- gating and casting surfaces
            use different HTCs.
        mold_temperature_c: Ambient mold temperature, degrees Celsius.
        density_kg_m3: Liquid metal density.
        specific_heat_j_kgk: Liquid metal specific heat.
        use_numba: Take the kernel when it is available. Mirrors
            :func:`step`'s parameter of the same name.

    Returns:
        The updated ``state.temperature``.
    """
    # Grouped exactly as the NumPy expression below groups it, so the division
    # rounds the same way on both paths.
    rho_cp_dx = density_kg_m3 * specific_heat_j_kgk * state.dx_m

    if use_numba and NUMBA_AVAILABLE:
        h_arr = np.asarray(htc_w_m2k, dtype=np.float64)
        h_is_field = h_arr.ndim == 3
        if h_is_field:
            # Numba wants a real array, not a broadcast view: a 0-stride
            # dimension is not something the kernel's indexing can rely on.
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
    # Density from the CURRENT populations, not the cached state.rho, which
    # still holds the previous step's value. Using the stale field lags the
    # body force by one step -- small, but it is a real error and it made the
    # NumPy and Numba paths disagree in the z component of every wet cell,
    # which is how it was found.
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
    """
    Advance the solver one timestep.

    Order: macroscopic moments -> collide (TRT + Guo forcing) -> stream ->
    bounce-back on obstacles -> reconstruct gas-side populations -> mass
    exchange -> cell transitions -> flag reinitialisation.

    The two free-surface passes at the end are separate on purpose.
    :func:`apply_transitions` converts a cell whose *fill level* crossed the
    band; :func:`heal_submerged_interface` converts one the front merely closed
    over, which no fill level will ever convert. Both paths below call them in
    the same order, so the numba and numpy branches stay structurally
    identical rather than merely agreeing by accident.

    Args:
        state: Solver state, mutated in place and returned.
        units: Lattice unit system.
        gravity_direction: Unit vector for gravity. Default is -Z.
        track_free_surface: When False the free surface is skipped entirely and
            the solver runs single-phase. Used by the closed-box benchmarks
            (hydrostatic rest, Poiseuille), where introducing a free surface
            would confound what is being measured.
        smagorinsky_cs: Subgrid constant, only consulted when ``smagorinsky``
            is True. Both paths take it, so the two stay comparable; the
            default is the module constant, which is what every benchmark in
            the suite was validated against. Passing 0.0 zeroes the eddy
            viscosity, leaving plain TRT at the molecular tau -- the same
            algebra as ``smagorinsky=False``, not a degenerate case.

    Returns:
        The same state object, advanced.
    """
    if use_numba and NUMBA_AVAILABLE:
        post, streamed, u64 = _collide_stream_numba(
            state, units, gravity_direction, smagorinsky, track_free_surface,
            smagorinsky_cs,
        )
        if track_free_surface:
            mass_out = _free_surface_scratch(state)
            C, Ci, W, OPP = lattice_tables()
            # A fresh float32 array rather than a held buffer, so state.f keeps
            # the "new object every step" identity the NumPy path also has -- a
            # caller that captured the previous state.f still sees the previous
            # step. The kernel writes every direction of every cell of it, and
            # reconstructs into it directly; it is the only population output
            # the free-surface pass has now.
            f32_out = np.empty((Q,) + state.shape, dtype=np.float32)
            free_surface_kernel(
                post,
                streamed,
                state.flags,
                # float32 density on purpose: the NumPy path's fill fraction
                # comes from fluid_fraction, which reads state.rho rounded to
                # float32. The raw float64 would be *more* accurate and less
                # in parity. u, in contrast, is the raw float64 -- that is
                # exactly what reconstruct_gas_populations receives over there.
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
            # Copy back into the array state.mass has always pointed at, so
            # mass_out (the cached scratch buffer) never becomes the same
            # object as mass_in on the next step -- see _free_surface_scratch.
            state.mass[...] = mass_out
            state.step_count += 1
            apply_transitions(state, state.step_count * state.dt_s)
            heal_submerged_interface(state, state.step_count * state.dt_s)
            reset_detached_interface_cells(state)
        else:
            # ``streamed`` is already float32, so this is a copy rather than a
            # conversion -- and the copy is the point. ``streamed`` is the
            # held ``_scratch_out`` buffer; binding it to state.f would make
            # every step overwrite the populations the caller was handed last
            # step, and would let collide_kernel read the array stream_kernel
            # is writing.
            state.f = streamed.astype(np.float32)
            state.step_count += 1
        return state

    force = gravity_force(state, units, gravity_direction)
    rho, u = macroscopic(state, force)
    state.rho = rho.astype(np.float32)
    state.u = u.astype(np.float32)

    # The Guo-Zhao drag, added to the body force so the collision's momentum
    # budget carries it. ``macroscopic`` has already braked ``u`` with it, but
    # the equilibrium is not where a force acts: without this term a filter cell
    # has no terminal velocity and its populations accelerate under gravity for
    # ever, which is what drives them negative. That is the whole reason for the
    # term -- not the frozen antisymmetric non-equilibrium the comment used to
    # cite, whose premise TAU_MINUS_MAX removed. Measured on a jet fixture at
    # the real filter coefficients: removing this kills the run at every inlet
    # velocity including the design velocity, and doubling it cuts peak |u| by
    # 3x. Exactly zero off a filter, so a no-filter run is bit-identical. See
    # physics.fluid_flow.porous_drag_force. Mirrored in collide_kernel.
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
        # Eddy viscosity, evaluated per cell from the local non-equilibrium
        # stress. Suppressed inside filters: the Forchheimer drag is already
        # the closure for pore-scale inertial losses there, and stacking a
        # subgrid model on top double-counts them.
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
    """
    Run the local pass and the streaming pass through the Numba kernels.

    Reuses scratch buffers held on the state. Allocating two
    (19, nx, ny, nz) arrays every step would put most of the win back: at a
    12 in casting that is ~250 MB of churn per step.

    Both population buffers are **float32**, matching ``state.f``. The
    arithmetic inside both kernels is still float64: ``collide_kernel`` reads
    through per-element casts and rounds once at the store, and
    ``stream_kernel`` only copies. See :class:`FillingState`'s dtype paragraph.

    What that bought, and what it did not, measured at 120^3 on 18 threads
    (medians of 12, per kernel, against a machine roofline of 185 GiB/s for a
    parallel 19 x N copy):

    ==================  ========  =======  ==============================
    kernel              float64   float32  effective bandwidth, after
    ==================  ========  =======  ==============================
    ``collide_kernel``   20.3 ms  20.4 ms  18 GiB/s -- 10% of roofline
    ``stream_kernel``     5.6 ms   3.8 ms  65 GiB/s -- 35%
    free surface         11.9 ms   7.0 ms  67 GiB/s -- 36%
    ==================  ========  =======  ==============================

    Streaming and the free-surface pass were bandwidth-bound and halving their
    traffic did what halving traffic does. **The collision was not**: cutting
    its output traffic by 125 MiB a step changed its time by 0.6%, inside the
    noise, because at 18 GiB/s it was nowhere near waiting on memory. It was
    waiting on duplicate arithmetic -- 57 equilibrium evaluations per cell where
    19 would do, its own populations read four times, ``c_i . u`` formed three
    times. Removing the duplication (bit-identically; the values and every
    accumulation order are untouched) took it to **17.3 ms**, and the whole
    step from 36.4 to 32.0 ms. The measurements and the reason LLVM could not
    do it unaided are in :func:`physics.fluid_flow.collide_kernel`'s docstring.

    That is the end of the easy constant factors. What remains in the collision
    is the Smagorinsky stress tensor (3.3 ms of the 17.3, measured by switching
    it off) and 19 scalar float64 equilibria that no longer duplicate anything.
    Going further means either vectorising across directions -- which needs the
    lattice tables transposed to ``Cx[19]``/``Cy[19]``/``Cz[19]`` so the
    per-direction reads stop being stride-3 gathers -- or moving the pass to the
    GPU. Neither is a dtype question, and neither is free.

    Nothing here converts dtypes any more, and that was worth ~26 ms of a
    measured 106 ms step at 120^3 (38 ``astype`` calls per step across the
    loop):

    * ``state.f`` goes to ``collide_kernel`` as float32. The kernel casts each
      population to float64 as it reads it, so the arithmetic -- and the result
      -- is unchanged; what disappears is a 19 x N float64 temporary built and
      thrown away every step.
    * ``porosity`` / ``drag_c0`` / ``drag_c1`` never change after the zone is
      built, so their float64 views are cached on :class:`PorousZone`.
    * ``rho`` and ``u`` are held float64 buffers instead of per-step
      ``np.empty``. The float64 ``u`` is also returned, so the free-surface
      kernel can read it directly rather than round-tripping through float32.

    Returns:
        ``(post, streamed, u64)`` -- the post-collision populations, the
        post-streaming populations, and the float64 velocity. All three are
        buffers owned by the state; the caller must not hold them past the step.
    """
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

    # Fresh float32 moment arrays, filled by the kernel rather than by a
    # follow-up astype. New objects every step, matching what the NumPy path's
    # ``rho.astype(np.float32)`` produced; np.empty does not touch the pages, so
    # the kernel's parallel write is also the first touch.
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
    """
    Cache the ``free_surface_kernel`` mass output buffer on the state.

    This used to hand back a population buffer too: a float64 ``f_out`` the
    kernel reconstructed into and then read back during the mass exchange,
    separate from the float32 copy that became ``state.f``. Now that the
    populations moving through the step are float32 there is nothing for the
    float64 copy to preserve, so it is gone -- 19 x N x 8 B, 250 MB at 120^3 --
    and the kernel reconstructs directly into the float32 buffer ``step()``
    allocates fresh each step. That buffer is *not* cached here, deliberately:
    it becomes ``state.f``, and a held buffer would make ``state.f`` the same
    object every step, so a caller holding the previous step's populations
    would find them silently overwritten.

    ``mass_out`` is float64 like ``state.mass`` itself, so aliasing is a real
    risk: if ``state.mass`` were ever pointed straight at this cached buffer,
    the next step would hand the kernel the same array as both ``mass_in`` and
    ``mass_out``, and neighbouring cells would read half-updated fill
    fractions. ``step()`` avoids that by copying into ``state.mass`` in place
    (``state.mass[...] = mass_out``) rather than rebinding the attribute, so
    this buffer stays a distinct object forever.
    """
    shape = state.shape
    mass_out = state._scratch_mass_out
    if mass_out is None or mass_out.shape != shape:
        mass_out = np.empty(shape, dtype=np.float64)
        state._scratch_mass_out = mass_out
    return mass_out


# ---------------------------------------------------------------------------
# Defect indicators (F3)
# ---------------------------------------------------------------------------


def find_entrapped_gas(flags: np.ndarray, vent_mask: np.ndarray) -> np.ndarray:
    """
    Gas cells that cannot escape to atmosphere as the metal closes around them.

    Labels connected components of GAS cells with **6-connectivity (face
    neighbours only)**, not the 18-neighbour set the D3Q19 stencil streams
    along elsewhere in this module. Two cells that touch only along an edge
    or at a corner share zero boundary area in the real 3-D geometry the
    lattice discretises -- no gas volume can cross a corner. D3Q19 includes
    those directions anyway because they improve the *momentum* stencil's
    isotropy; that is a numerical choice about the fluid solver, not a claim
    about what counts as a connected air pocket. Face connectivity is the
    physically correct choice for "can this gas physically get out".

    A component with no cell in ``vent_mask`` has no route to atmosphere: the
    metal closing in around it cannot displace it, and it freezes in as
    trapped-air porosity.

    Args:
        flags: Cell state, ``(nx, ny, nz)``.
        vent_mask: Boolean ``(nx, ny, nz)``, cells that count as open to
            atmosphere -- see ``core.filling_stage`` for how this is
            assembled (open riser tops, the inlet, the headspace above the
            mold, and gas cells currently on the mold's open top face; the
            sides and floor of the bounding box are mold, not air, and are
            deliberately not vents). An all-``False`` mask makes every gas
            pocket in the domain read as entrapped; this function has no way
            to distinguish "no vent exists" from "no vent has been reached
            yet", so the caller must detect and warn about that condition
            itself (see ``core.filling_stage._entrapment_warning_if_no_vent``).

    Returns:
        Boolean ``(nx, ny, nz)``, True on every GAS cell belonging to a
        component that contains no vent cell.
    """
    from scipy import ndimage

    gas = flags == FLAG_GAS
    if not np.any(gas):
        return np.zeros_like(gas, dtype=bool)

    # generate_binary_structure(3, 1) is the face-only (6-neighbour)
    # structuring element -- connectivity 1 means "shares a face", as
    # opposed to connectivity 2/3 which also admit edge/corner touches.
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
    """
    Cells where two metal fronts met after the metal had already cooled too
    far to fuse.

    Both conditions below are required -- see
    ``docs/filling_implementation.md`` section 6.4/F3. Neither alone is a
    cold shut: two hot fronts meeting just fuses, and cold metal with nothing
    meeting it is ordinary heat loss, not a seam.

    **Front meeting.** This deliberately does not use the literal test from
    the design doc ("a wet neighbour whose arrival time differs by more than
    ``arrival_gap_s``"). That test alone flags the trailing edge of *every*
    moving front: a front's own wake always has an arrival-time gradient
    along the direction of travel, and a slow front accumulates more than
    ``arrival_gap_s`` of that gradient over a long enough run with no second
    front anywhere nearby -- exactly the false positive the design doc's own
    prose warns about two paragraphs later. What actually distinguishes "two
    fronts met here" from "one front passed through here" is *how many
    directions* the earlier metal came from: a single sweeping front --
    however slow, however curved -- has at most one face direction whose
    neighbour arrived substantially earlier (directly behind it); the
    opposite/leading side is either still gas or arrived within the same
    instant. A cell where two fronts converge -- head-on along one axis, or
    from two different axes at a corner -- has *two or more* such directions.
    Requiring at least two is what holds a single front sweep to zero false
    positives in ``tests/test_filling_defects.py`` while still catching both
    head-on and corner meetings.

    **Cold.** Scored against ``liquidus_c + temperature_margin_c`` rather
    than the bare liquidus: fusion quality degrades on approach to the
    liquidus, not only below it, and the margin is what lets a foundry tune
    how conservative the indicator is.

    Args:
        fill_time_s: Arrival time per cell, ``-1`` where never filled.
        temperature: Degrees Celsius, NaN (:data:`NO_TEMPERATURE`) where
            never filled.
        flags: Cell state.
        liquidus_c: Alloy liquidus, degrees Celsius.
        arrival_gap_s: Minimum arrival-time difference across a face to
            count that face as "a different front", seconds.
        temperature_margin_c: Degrees above the liquidus still counted as
            cold enough to risk a seam.

    Returns:
        Boolean ``(nx, ny, nz)``.
    """
    wet = (flags == FLAG_INTERFACE) | (flags == FLAG_FLUID)
    fill_time = np.asarray(fill_time_s, dtype=np.float64)
    temp = np.asarray(temperature, dtype=np.float64)

    # Sentinels never participate: a cell the fill never reached has no
    # arrival time and no temperature to evaluate either condition against,
    # and a corrupted state (wet but still carrying a sentinel) must not
    # silently pass either test.
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
