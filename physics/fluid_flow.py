"""
D3Q19 lattice Boltzmann primitives for the mold-filling solver.

Scope: the lattice itself and the closures that act on a single cell —
equilibrium, TRT collision, streaming, gravity forcing, and the porous-medium
drag a ceramic filter imposes. Free-surface mass tracking, boundary conditions
and the step loop live in :mod:`physics.filling`; the mesh↔grid mapping lives in
``core/filling_stage.py``.

Layering
--------
``physics/`` may not import ``core/`` or ``config/`` — ``tests/
test_architecture_boundaries.py`` enforces it with an AST walk that catches
function-local imports too. So everything here takes plain scalars and arrays.
Nothing in this module knows what a voxel mesh or an alloy is.

Units
-----
Every public function takes **SI** and converts internally. The lattice
conversion is done in exactly one place, :func:`lattice_units`, and asserted in
``tests/test_filling_lattice.py`` — ``tests/test_units.py`` already demonstrates
the failure this discipline prevents (a head pressure computed with inches for
metres is wrong by 39x and looks entirely plausible).

Why TRT and not BGK
-------------------
Molten steel has kinematic viscosity nu = mu/rho = 0.006/7200 = 8.33e-7 m^2/s.
At every realistic (dx, dt) that gives a relaxation time within ~1e-4 of the
tau = 1/2 stability limit, where BGK is unconditionally unstable. Measured for
this codebase's geometries: tau = 0.500007 to 0.500049. A BGK implementation
does not merely lose accuracy, it diverges — and it diverges in a way that
looks like a geometry or boundary-condition bug, which is why this module has
no BGK path to "prototype with".

Two-relaxation-time collision fixes it. Splitting each population into
symmetric and antisymmetric parts with separate rates, tied by the magic
parameter Lambda = (1/omega+ - 1/2)(1/omega- - 1/2) = 3/16, is stable at very
low tau and places the bounce-back wall exactly halfway between nodes. Cost is
about 1.2x BGK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# D3Q19 lattice
# ---------------------------------------------------------------------------

#: Lattice velocities. Index 0 is the rest population; thereafter directions
#: come in opposite pairs (1,2), (3,4), ... so that ``OPPOSITE`` is arithmetic
#: rather than a lookup table, which matters inside a Numba kernel.
LATTICE_VELOCITIES = np.array(
    [
        (0, 0, 0),
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
        (1, 1, 0), (-1, -1, 0),
        (1, -1, 0), (-1, 1, 0),
        (1, 0, 1), (-1, 0, -1),
        (1, 0, -1), (-1, 0, 1),
        (0, 1, 1), (0, -1, -1),
        (0, 1, -1), (0, -1, 1),
    ],
    dtype=np.int8,
)

#: Number of discrete velocities.
Q = 19

#: Equilibrium weights: 1/3 rest, 1/18 face, 1/36 edge.
LATTICE_WEIGHTS = np.array(
    [1.0 / 3.0]
    + [1.0 / 18.0] * 6
    + [1.0 / 36.0] * 12,
    dtype=np.float64,
)

#: ``OPPOSITE[i]`` is the index of -c_i. Used by bounce-back and by the
#: symmetric/antisymmetric split in TRT.
OPPOSITE = np.array(
    [0] + [i + 1 if i % 2 == 1 else i - 1 for i in range(1, Q)],
    dtype=np.int8,
)

#: Lattice speed of sound squared. c_s^2 = 1/3 for D3Q19.
CS2 = 1.0 / 3.0
INV_CS2 = 3.0

#: Magic parameter tying the two TRT relaxation rates.
#:
#: Lambda = 3/16 is the value for which the halfway bounce-back wall sits
#: exactly midway between nodes independently of tau (Ginzburg, d'Humieres &
#: Kuzmin 2011). Any other value makes the effective wall position drift with
#: viscosity, which shows up as a Poiseuille profile fitted to the wrong
#: channel width.
TRT_MAGIC_PARAMETER = 3.0 / 16.0

#: Gravitational acceleration, m/s^2. Mirrors ``physics.units.GRAVITY_M_S2``;
#: duplicated rather than imported only because this module is deliberately
#: dependency-free at import time for the Numba kernels.
GRAVITY_M_S2 = 9.81


# ---------------------------------------------------------------------------
# Lattice unit conversion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatticeUnits:
    """
    The single place SI becomes lattice units.

    Attributes:
        dx_m: Cell size, metres.
        dt_s: Timestep, seconds.
        nu_lattice: Kinematic viscosity in lattice units.
        tau: Relaxation time, ``3 * nu_lattice + 1/2``.
        omega_plus: Symmetric relaxation rate, ``1 / tau``.
        omega_minus: Antisymmetric rate, from the magic parameter.
        gravity_lattice: Gravitational acceleration in lattice units.
        u_max_lattice: Mach guard the timestep was chosen to satisfy.
        u_max_physical: Physical velocity that maps to ``u_max_lattice``.
    """

    dx_m: float
    dt_s: float
    nu_lattice: float
    tau: float
    omega_plus: float
    omega_minus: float
    gravity_lattice: float
    u_max_lattice: float
    u_max_physical: float

    @property
    def velocity_scale(self) -> float:
        """Multiply a lattice velocity by this to get m/s."""
        return self.dx_m / self.dt_s

    def to_physical_velocity(self, u_lattice: np.ndarray) -> np.ndarray:
        """Convert a lattice velocity field to m/s."""
        return np.asarray(u_lattice) * self.velocity_scale

    def to_lattice_velocity(self, u_physical: np.ndarray) -> np.ndarray:
        """Convert a velocity field in m/s to lattice units."""
        return np.asarray(u_physical) / self.velocity_scale


def lattice_units(
    dx_m: float,
    kinematic_viscosity_m2_s: float,
    *,
    u_max_physical_m_s: float,
    u_max_lattice: float = 0.05,
    magic_parameter: float = TRT_MAGIC_PARAMETER,
    gravity_m_s2: float = GRAVITY_M_S2,
) -> LatticeUnits:
    """
    Derive the lattice unit system from SI inputs.

    The timestep follows from the Mach guard rather than being chosen
    independently::

        dt   = u_max_lattice * dx / u_max_physical
        nu_l = nu_SI * dt / dx^2
        tau  = 3 * nu_l + 1/2
        g_l  = g * dt^2 / dx

    Do **not** try to raise tau away from 1/2 by shrinking dt. It scales the
    wrong way: halving dt halves ``nu_lattice`` and pushes tau *closer* to 1/2
    while doubling the step count.

    Args:
        dx_m: Cell size in metres.
        kinematic_viscosity_m2_s: Physical kinematic viscosity.
        u_max_physical_m_s: Fastest physical velocity to be represented.
        u_max_lattice: Lattice velocity that maps to it. Keep well under the
            lattice sound speed (0.577); 0.05 is a conventional safe choice.
        magic_parameter: TRT Lambda. Defaults to 3/16.
        gravity_m_s2: Gravitational acceleration.

    Returns:
        A populated :class:`LatticeUnits`.

    Raises:
        ValueError: If any input is non-positive, or if the resulting tau is
            not above 1/2. A tau at or below 1/2 is not a small inaccuracy --
            the scheme has negative viscosity and diverges -- so it is refused
            here rather than left to blow up thousands of steps later.
    """
    if dx_m <= 0:
        raise ValueError(f"dx_m must be positive, got {dx_m}")
    if kinematic_viscosity_m2_s <= 0:
        raise ValueError(
            f"kinematic_viscosity_m2_s must be positive, got {kinematic_viscosity_m2_s}"
        )
    if u_max_physical_m_s <= 0:
        raise ValueError(f"u_max_physical_m_s must be positive, got {u_max_physical_m_s}")
    if not 0.0 < u_max_lattice < 0.577:
        raise ValueError(
            f"u_max_lattice must be in (0, 0.577) -- the lattice sound speed -- "
            f"got {u_max_lattice}"
        )
    if magic_parameter <= 0:
        raise ValueError(f"magic_parameter must be positive, got {magic_parameter}")

    dt_s = u_max_lattice * dx_m / u_max_physical_m_s
    nu_lattice = kinematic_viscosity_m2_s * dt_s / (dx_m * dx_m)
    tau = 3.0 * nu_lattice + 0.5
    if tau <= 0.5:
        raise ValueError(
            f"tau={tau!r} is not above 1/2; the scheme would have negative "
            f"viscosity and diverge"
        )

    omega_plus = 1.0 / tau
    omega_minus = omega_minus_from_magic(omega_plus, magic_parameter)
    gravity_lattice = gravity_m_s2 * dt_s * dt_s / dx_m

    return LatticeUnits(
        dx_m=float(dx_m),
        dt_s=float(dt_s),
        nu_lattice=float(nu_lattice),
        tau=float(tau),
        omega_plus=float(omega_plus),
        omega_minus=float(omega_minus),
        gravity_lattice=float(gravity_lattice),
        u_max_lattice=float(u_max_lattice),
        u_max_physical=float(u_max_physical_m_s),
    )


def omega_minus_from_magic(
    omega_plus: float, magic_parameter: float = TRT_MAGIC_PARAMETER
) -> float:
    """
    Antisymmetric relaxation rate from the symmetric one.

    Inverts ``Lambda = (1/omega+ - 1/2)(1/omega- - 1/2)``.
    """
    if omega_plus <= 0.0 or omega_plus >= 2.0:
        raise ValueError(f"omega_plus must be in (0, 2), got {omega_plus}")
    half_inverse_plus = 1.0 / omega_plus - 0.5
    if half_inverse_plus <= 0.0:
        raise ValueError(f"omega_plus={omega_plus} implies tau <= 1/2")
    return 1.0 / (magic_parameter / half_inverse_plus + 0.5)


# ---------------------------------------------------------------------------
# Equilibrium and moments
# ---------------------------------------------------------------------------


def equilibrium(rho: np.ndarray, u: np.ndarray, porosity: np.ndarray | float = 1.0) -> np.ndarray:
    """
    D3Q19 equilibrium distribution.

    ``f_i^eq = w_i rho [1 + 3(c_i.u) + 4.5(c_i.u)^2/eps - 1.5 u^2/eps]``

    The porosity division is the Guo-Zhao porous-medium correction; it is
    required for Galilean consistency inside a filter and is exactly the
    standard equilibrium when ``porosity == 1``.

    Args:
        rho: Density, shape ``(...)``.
        u: Velocity, shape ``(3, ...)``.
        porosity: Cell porosity, broadcastable to ``rho``. 1.0 outside filters.

    Returns:
        Array of shape ``(19, ...)``.
    """
    rho = np.asarray(rho, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    eps = np.asarray(porosity, dtype=np.float64)

    usqr = np.einsum("i...,i...->...", u, u)
    feq = np.empty((Q,) + rho.shape, dtype=np.float64)
    for i in range(Q):
        cu = (
            LATTICE_VELOCITIES[i, 0] * u[0]
            + LATTICE_VELOCITIES[i, 1] * u[1]
            + LATTICE_VELOCITIES[i, 2] * u[2]
        )
        feq[i] = LATTICE_WEIGHTS[i] * rho * (
            1.0 + 3.0 * cu + 4.5 * cu * cu / eps - 1.5 * usqr / eps
        )
    return feq


def density(f: np.ndarray) -> np.ndarray:
    """Zeroth moment: ``rho = sum_i f_i``."""
    total: np.ndarray = np.sum(np.asarray(f), axis=0)
    return total


def momentum(f: np.ndarray) -> np.ndarray:
    """
    First moment: ``rho u = sum_i c_i f_i``.

    Returns the *raw* momentum. With a body force present the physical velocity
    additionally needs the ``+ F dt/2`` Guo correction -- see
    :func:`guo_velocity`.
    """
    f = np.asarray(f)
    out = np.zeros((3,) + f.shape[1:], dtype=np.float64)
    for i in range(Q):
        for d in range(3):
            c = LATTICE_VELOCITIES[i, d]
            if c:
                out[d] += c * f[i]
    return out


def guo_velocity(f: np.ndarray, rho: np.ndarray, force: np.ndarray) -> np.ndarray:
    """
    Velocity including the half-force Guo correction.

    ``u = (sum_i c_i f_i + F/2) / rho``

    Omitting the half-force term is the single most common LBM forcing bug: it
    leaves a first-order-in-force error that shows up as a slowly drifting
    velocity in what should be a fluid at rest.
    """
    rho = np.asarray(rho, dtype=np.float64)
    corrected: np.ndarray = (
        momentum(f) + 0.5 * np.asarray(force, dtype=np.float64)
    ) / rho
    return corrected


def guo_force_terms(
    u: np.ndarray, force: np.ndarray, omega_minus: float | np.ndarray
) -> np.ndarray:
    """
    Guo forcing source terms.

    ``F_i = w_i (1 - omega_minus/2) [3(c_i - u) + 9(c_i.u) c_i] . F``

    The prefactor uses the **antisymmetric** relaxation rate, not ``1 - 1/(2 tau)``.
    That distinction is invisible in BGK and fatal in TRT.

    Momentum bookkeeping over one collision::

        sum_i c_i f_i*  =  sum_i c_i f_i  +  omega_minus F / 2  +  prefactor * F

    Momentum is odd in ``c_i``, so only the antisymmetric relaxation touches it,
    and the equilibrium it relaxes toward already carries the ``+F/2`` from
    :func:`guo_velocity`. For the two terms to sum to exactly ``F`` the
    prefactor must therefore be ``1 - omega_minus/2``.

    In BGK ``omega_minus == omega_plus == 1/tau``, which recovers the familiar
    ``1 - 1/(2 tau)``. In TRT the two rates are independent: at the viscosities
    here ``omega_plus`` approaches 2 while ``omega_minus`` falls to ~2.6e-4, so
    using ``tau`` injects roughly 2e-4 of the intended force. The symptom is
    not a crash -- gravity simply does almost nothing, a hydrostatic column
    never develops its pressure gradient, and the fill runs impossibly slowly.

    Args:
        u: Velocity, shape ``(3, ...)``.
        force: Body force per unit volume in lattice units, shape ``(3, ...)``.
        omega_minus: Antisymmetric relaxation rate.

    Returns:
        Array of shape ``(19, ...)``.
    """
    u = np.asarray(u, dtype=np.float64)
    force = np.asarray(force, dtype=np.float64)
    prefactor = 1.0 - 0.5 * omega_minus

    out: np.ndarray = np.empty((Q,) + u.shape[1:], dtype=np.float64)
    for i in range(Q):
        c = LATTICE_VELOCITIES[i].astype(np.float64)
        cu = c[0] * u[0] + c[1] * u[1] + c[2] * u[2]
        total = np.zeros_like(cu)
        for d in range(3):
            total = total + (3.0 * (c[d] - u[d]) + 9.0 * cu * c[d]) * force[d]
        out[i] = LATTICE_WEIGHTS[i] * prefactor * total
    return out


# ---------------------------------------------------------------------------
# Collision and streaming
# ---------------------------------------------------------------------------


def collide_trt(
    f: np.ndarray,
    feq: np.ndarray,
    omega_plus: float | np.ndarray,
    omega_minus: float | np.ndarray,
    source: np.ndarray | None = None,
) -> np.ndarray:
    """
    Two-relaxation-time collision.

    ::

        f_i^+  = (f_i + f_ibar) / 2          f_i^-  = (f_i - f_ibar) / 2
        f_i*   = f_i - omega+ (f_i^+ - f_i^eq+) - omega- (f_i^- - f_i^eq-) + F_i

    Args:
        f: Pre-collision populations, shape ``(19, ...)``.
        feq: Equilibrium populations, same shape.
        omega_plus: Symmetric relaxation rate.
        omega_minus: Antisymmetric relaxation rate.
        source: Optional Guo forcing terms, same shape.

    Returns:
        Post-collision populations.
    """
    f = np.asarray(f, dtype=np.float64)
    feq = np.asarray(feq, dtype=np.float64)
    opp = OPPOSITE.astype(np.intp)

    f_bar = f[opp]
    feq_bar = feq[opp]

    f_sym = 0.5 * (f + f_bar)
    f_asym = 0.5 * (f - f_bar)
    feq_sym = 0.5 * (feq + feq_bar)
    feq_asym = 0.5 * (feq - feq_bar)

    # Broadcast per-cell relaxation rates against the direction axis. Scalars
    # pass through untouched, so the closed-box benchmarks keep their exact
    # single-tau behaviour.
    op = np.asarray(omega_plus, dtype=np.float64)
    om = np.asarray(omega_minus, dtype=np.float64)
    if op.ndim:
        op = op[None, ...]
    if om.ndim:
        om = om[None, ...]

    out: np.ndarray = f - op * (f_sym - feq_sym) - om * (f_asym - feq_asym)
    if source is not None:
        out = out + np.asarray(source, dtype=np.float64)
    return out


def stream(f: np.ndarray) -> np.ndarray:
    """
    Streaming step on a periodic dense grid.

    ``f_i(x + c_i, t+1) = f_i*(x, t)``, implemented as a roll per direction.
    Periodicity is a property of this bare helper only -- real domains are
    closed by the obstacle mask and bounce-back in :mod:`physics.filling`, which
    overwrite anything that wraps.

    Args:
        f: Post-collision populations, shape ``(19, nx, ny, nz)``.

    Returns:
        Streamed populations, same shape.
    """
    f = np.asarray(f)
    out = np.empty_like(f)
    for i in range(Q):
        shift = LATTICE_VELOCITIES[i]
        out[i] = np.roll(f[i], shift=(int(shift[0]), int(shift[1]), int(shift[2])), axis=(0, 1, 2))
    return out


def bounce_back(f: np.ndarray, obstacle: np.ndarray) -> np.ndarray:
    """
    Halfway bounce-back on solid cells.

    Populations sitting on an obstacle cell are reflected into their opposite
    direction. With TRT at Lambda = 3/16 this places the wall exactly halfway
    between the solid and fluid nodes.

    Args:
        f: Populations, shape ``(19, nx, ny, nz)``.
        obstacle: Boolean solid mask, shape ``(nx, ny, nz)``.

    Returns:
        Populations with obstacle cells reflected.
    """
    f = np.asarray(f).copy()
    mask = np.asarray(obstacle, dtype=bool)
    if not np.any(mask):
        return f
    reflected = f[OPPOSITE.astype(np.intp)]
    f[:, mask] = reflected[:, mask]
    return f


# ---------------------------------------------------------------------------
# Ceramic foam filter closure
# ---------------------------------------------------------------------------

#: Ergun prefactor fitted to published alumina-in-water permeabilities.
#:
#: Uncalibrated Ergun over the literature values (k1 = 6.3e-8 / 2.0e-8 / 8e-9
#: m^2 at 30 / 50 / 80 ppi) is wrong by a factor of 2.07 / 2.35 / 2.29 -- i.e.
#: constant to about +/-7 % across a 2.7x span in cell diameter. The d^2 scaling
#: therefore transfers and only the prefactor needs fitting, which is one
#: parameter rather than a curve.
#:
#: **The fit is against a nominal porosity of 0.85 at every pore count**, which
#: is what :data:`NOMINAL_FOAM_POROSITY` exists for. That is not an oversight:
#: measured porosity barely moves with ppi (0.85-0.91 across 10-80 ppi) while
#: permeability falls ~30x, so the ``eps^3/(1-eps)^2`` factor is doing fitting
#: work rather than physics. Feeding *measured* per-ppi porosities into the same
#: prefactor scatters the ratios from 13 % to 78 % and makes 30 ppi wrong by
#: 1.8x. Hold porosity at the nominal value unless you re-fit the prefactor.
ERGUN_PREFACTOR = 2.2

#: Porosity the Ergun prefactor was fitted at. See :data:`ERGUN_PREFACTOR`.
NOMINAL_FOAM_POROSITY = 0.85

#: Ward's inertial coefficient for reticulated foam, ``1/k2 = c_F / sqrt(k1)``.
#:
#: This term carries roughly 95 % of the pressure drop at foundry velocities, so
#: it is the one that must be right. Leaving it at a packed-bed value while
#: calibrating k1 -- the term carrying 1-5 % -- is the trap here.
FORCHHEIMER_WARD_CF = 0.55


def ceramic_foam_permeability(
    porosity: float,
    pore_diameter_m: float,
    *,
    ergun_prefactor: float = ERGUN_PREFACTOR,
) -> float:
    """
    Darcy permeability k1 of a ceramic foam, m^2.

    ``k1 = eps^3 d^2 / (150 C_K (1 - eps)^2)``

    No permeability is published for any Foseco STELEX product, and the
    measured literature values are alumina-in-water with about 2x scatter
    between suppliers at identical nominal ppi. So this is a correlation with a
    fitted prefactor, not a material constant: treat the result as good to
    roughly a factor of two, and see ``tests/test_filter_hydrodynamics.py``
    for the sensitivity bound that makes that acceptable.

    Args:
        porosity: Void fraction. Pass :data:`NOMINAL_FOAM_POROSITY` unless you
            have re-fitted ``ergun_prefactor`` -- the prefactor and the porosity
            are calibrated together, and substituting a measured per-ppi
            porosity into the stock prefactor is wrong by up to 1.8x.
        pore_diameter_m: Mean cell diameter. A 10 ppi foam is ~2.54 mm.
        ergun_prefactor: Fitted C_K.

    Returns:
        Permeability in m^2.

    Raises:
        ValueError: On a porosity outside (0, 1) or a non-positive diameter.
    """
    if not 0.0 < porosity < 1.0:
        raise ValueError(f"porosity must be in (0, 1), got {porosity}")
    if pore_diameter_m <= 0:
        raise ValueError(f"pore_diameter_m must be positive, got {pore_diameter_m}")
    if ergun_prefactor <= 0:
        raise ValueError(f"ergun_prefactor must be positive, got {ergun_prefactor}")
    numerator = porosity ** 3 * pore_diameter_m * pore_diameter_m
    denominator = 150.0 * ergun_prefactor * (1.0 - porosity) ** 2
    return numerator / denominator


def forchheimer_inverse_length(
    k1_m2: float, *, c_f: float = FORCHHEIMER_WARD_CF
) -> float:
    """
    Inertial coefficient ``1/k2`` in 1/m, from Ward's correlation.

    Pairs with :func:`ceramic_foam_permeability` so that

    ``dp/L = (mu / k1) u + rho (1/k2) u^2``

    is internally consistent. Supplying a k1 from one source and a k2 from
    another routinely produces a pair differing by 4-5x from Ward's relation,
    with the two errors partly cancelling and the result looking plausible.
    """
    if k1_m2 <= 0:
        raise ValueError(f"k1_m2 must be positive, got {k1_m2}")
    if c_f <= 0:
        raise ValueError(f"c_f must be positive, got {c_f}")
    return float(c_f / np.sqrt(k1_m2))


def forchheimer_pressure_drop(
    velocity_m_s: float,
    thickness_m: float,
    *,
    k1_m2: float,
    inv_k2_per_m: float,
    dynamic_viscosity_pa_s: float,
    density_kg_m3: float,
) -> float:
    """
    Pressure drop across a porous slab, Pa.

    ``dp = L [ (mu / k1) u + rho (1/k2) u^2 ]``

    Args:
        velocity_m_s: Superficial (empty-tube) velocity.
        thickness_m: Slab thickness along the flow.
        k1_m2: Darcy permeability.
        inv_k2_per_m: Inertial coefficient.
        dynamic_viscosity_pa_s: Fluid viscosity.
        density_kg_m3: Fluid density.

    Returns:
        Pressure drop in Pa.
    """
    viscous = dynamic_viscosity_pa_s / k1_m2 * velocity_m_s
    inertial = density_kg_m3 * inv_k2_per_m * velocity_m_s * velocity_m_s
    return thickness_m * (viscous + inertial)


def porous_drag_coefficients(
    porosity: float,
    k1_m2: float,
    inv_k2_per_m: float,
    *,
    nu_lattice: float,
    dx_m: float,
) -> Tuple[float, float]:
    """
    Guo-Zhao implicit drag coefficients for one porous cell.

    The porous momentum equation

    ``F = eps g - (eps nu / k1) u - eps c_F/sqrt(k1) |u| u``

    is quadratic in ``u``, and Guo & Zhao's closed form solves it exactly::

        v  = (1/rho) sum_i c_i f_i + (dt/2) eps g
        u  = v / (c0 + sqrt(c0^2 + c1 |v|))

    with the coefficients this function returns. No iteration, no timestep
    penalty.

    The drag is moderately stiff at realistic settings -- it equilibrates in
    roughly a dozen steps -- which is exactly why the implicit form is required
    rather than merely convenient. An explicit Guo drag at these coefficients is
    not stable.

    Off-filter the caller passes ``porosity = 1`` and infinite ``k1``, giving
    ``(0.5, 0.0)`` and hence ``u = v / (0.5 + 0.5) = v`` exactly. That keeps one
    branch-free kernel and makes a zero-drag build bit-identical to a build with
    no filter at all.

    Args:
        porosity: Cell void fraction.
        k1_m2: Darcy permeability, ``inf`` for a clear cell.
        inv_k2_per_m: Inertial coefficient, 0 for a clear cell.
        nu_lattice: Kinematic viscosity in lattice units.
        dx_m: Cell size in metres.

    Returns:
        ``(c0, c1)``.
    """
    if not 0.0 < porosity <= 1.0:
        raise ValueError(f"porosity must be in (0, 1], got {porosity}")
    if dx_m <= 0:
        raise ValueError(f"dx_m must be positive, got {dx_m}")

    if not np.isfinite(k1_m2) or k1_m2 <= 0.0:
        c0 = 0.5
    else:
        k1_lattice = k1_m2 / (dx_m * dx_m)
        c0 = 0.5 * (1.0 + 0.5 * porosity * nu_lattice / k1_lattice)
    c1 = 0.5 * porosity * float(inv_k2_per_m) * dx_m
    return float(c0), float(c1)


def resolve_porous_velocity(v: np.ndarray, c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
    """
    Solve the Guo-Zhao quadratic for the true velocity.

    ``u = v / (c0 + sqrt(c0^2 + c1 |v|))``

    Args:
        v: Provisional velocity, shape ``(3, ...)``.
        c0: Linear drag coefficient, broadcastable.
        c1: Quadratic drag coefficient, broadcastable.

    Returns:
        Velocity of the same shape as ``v``. Direction is always preserved and
        the magnitude never grows -- both asserted in the test suite, because a
        drag term that can reverse or amplify a velocity is catastrophic and
        easy to introduce with a sign slip.
    """
    v = np.asarray(v, dtype=np.float64)
    c0 = np.asarray(c0, dtype=np.float64)
    c1 = np.asarray(c1, dtype=np.float64)
    speed = np.sqrt(np.einsum("i...,i...->...", v, v))
    denominator = c0 + np.sqrt(c0 * c0 + c1 * speed)
    resolved: np.ndarray = v / denominator
    return resolved


def capillary_entry_pressure(
    surface_tension_n_m: float,
    contact_angle_deg: float,
    pore_diameter_m: float,
    window_fraction: float,
) -> float:
    """
    Capillary pressure a non-wetting melt must overcome to prime a filter, Pa.

    ``dp = 4 sigma |cos(theta)| / (f_w d_cell)``

    The length scale is the *window* between cells, not the cell itself: using
    the full cell radius gives about half the entry pressure and lands well
    below the 50-75 mm head that foundry practice reports for 10 ppi steel
    filters.

    Steel does not wet zirconia -- contact angles of 110-135 degrees are
    reported -- which is why a filter needs a metallostatic head to prime at
    all, and why a filter that never primes is a real and reportable outcome
    rather than a solver failure.

    Args:
        surface_tension_n_m: Melt surface tension. Steels are 1.7-1.9 N/m.
        contact_angle_deg: Melt-ceramic contact angle, > 90 for non-wetting.
        pore_diameter_m: Mean cell diameter.
        window_fraction: Window-to-cell diameter ratio, typically ~0.45.

    Returns:
        Entry pressure in Pa.
    """
    if surface_tension_n_m <= 0:
        raise ValueError(f"surface_tension_n_m must be positive, got {surface_tension_n_m}")
    if pore_diameter_m <= 0:
        raise ValueError(f"pore_diameter_m must be positive, got {pore_diameter_m}")
    if not 0.0 < window_fraction <= 1.0:
        raise ValueError(f"window_fraction must be in (0, 1], got {window_fraction}")
    cos_theta = abs(np.cos(np.radians(contact_angle_deg)))
    return float(
        4.0 * surface_tension_n_m * cos_theta / (window_fraction * pore_diameter_m)
    )


def capillary_entry_density_offset(
    entry_pressure_pa: float,
    *,
    density_kg_m3: float,
    dt_s: float,
    dx_m: float,
) -> float:
    """
    Entry pressure expressed as a lattice gas-density offset.

    ``d_rho = dp dt^2 / (rho dx^2 c_s^2)``

    Equivalently ``d_rho = (h_prime / dx) * g_lattice / c_s^2`` where
    ``h_prime = dp / (rho g)`` is the priming head. The two forms are
    algebraically identical, and asserting that identity in the tests catches
    every unit error on this path in one assertion -- which is worth having,
    because a capillary term wrong by a factor of dx is invisible in the output
    and merely makes filters prime at the wrong depth.
    """
    if density_kg_m3 <= 0:
        raise ValueError(f"density_kg_m3 must be positive, got {density_kg_m3}")
    if dt_s <= 0 or dx_m <= 0:
        raise ValueError(f"dt_s and dx_m must be positive, got {dt_s}, {dx_m}")
    return entry_pressure_pa * dt_s * dt_s / (density_kg_m3 * dx_m * dx_m * CS2)


def priming_head_m(
    entry_pressure_pa: float, density_kg_m3: float, gravity_m_s2: float = GRAVITY_M_S2
) -> float:
    """Metallostatic head required to prime a filter, metres."""
    return entry_pressure_pa / (density_kg_m3 * gravity_m_s2)


# ---------------------------------------------------------------------------
# Turbulence closure
# ---------------------------------------------------------------------------

#: Smagorinsky constant. Reynolds numbers here are 6,000-720,000, so the flow
#: is strongly turbulent and nowhere near resolved: a subgrid model is correct
#: physics, not a numerical crutch. It is not, however, sufficient on its own to
#: stabilise low tau -- that is what TRT is for.
SMAGORINSKY_CS = 0.16


def smagorinsky_tau(
    tau: float, strain_rate_magnitude: float, smagorinsky_cs: float = SMAGORINSKY_CS
) -> float:
    """
    Relaxation time including the Smagorinsky eddy viscosity.

    ``nu_eff = nu + (C_s dx)^2 |S|``, converted back to a relaxation time. In
    lattice units the filter width is 1, so ``nu_t = (C_s)^2 |S|``.

    Note this must be **suppressed inside filter cells**: the pore Reynolds
    number there is a few hundred, and the Forchheimer term is already the
    closure for those pore-scale inertial losses. Adding eddy viscosity on top
    double-counts them.
    """
    nu = (tau - 0.5) / 3.0
    nu_t = smagorinsky_cs * smagorinsky_cs * abs(strain_rate_magnitude)
    return 3.0 * (nu + nu_t) + 0.5


def strain_rate_magnitude(f: np.ndarray, feq: np.ndarray) -> np.ndarray:
    """
    Magnitude of the non-equilibrium momentum flux, ``|Q|``.

    ``Q_ab = sum_i c_ia c_ib (f_i - f_i^eq)`` and ``|Q| = sqrt(Q_ab Q_ab)``.

    LBM recovers the strain rate locally from the non-equilibrium part of the
    populations, so no finite differences are needed -- which is the whole
    reason a subgrid model is cheap here.

    Args:
        f: Populations, shape ``(19, ...)``.
        feq: Equilibrium populations, same shape.

    Returns:
        ``|Q|`` with the spatial shape of ``f[0]``.
    """
    neq = np.asarray(f, dtype=np.float64) - np.asarray(feq, dtype=np.float64)
    c = LATTICE_VELOCITIES.astype(np.float64)
    total = np.zeros(neq.shape[1:], dtype=np.float64)
    for a in range(3):
        for b in range(3):
            q_ab = np.zeros(neq.shape[1:], dtype=np.float64)
            for i in range(Q):
                weight = c[i, a] * c[i, b]
                if weight:
                    q_ab += weight * neq[i]
            total += q_ab * q_ab
    return np.sqrt(total)


def smagorinsky_relaxation(
    f: np.ndarray,
    feq: np.ndarray,
    rho: np.ndarray,
    tau: float,
    *,
    smagorinsky_cs: float = SMAGORINSKY_CS,
    suppress: np.ndarray | None = None,
    magic_parameter: float = TRT_MAGIC_PARAMETER,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-cell TRT relaxation rates including the Smagorinsky eddy viscosity.

    ``tau_eff = (tau + sqrt(tau^2 + 18 sqrt(2) C_s^2 |Q| / rho)) / 2``

    This is not an optional refinement. Reynolds numbers in a casting are
    6,000-720,000, so the flow is strongly turbulent and nowhere near resolved
    -- the subgrid model is the correct physics. It is also what makes the
    scheme survive: molecular tau sits at 0.50000x, within 1e-5 of the
    stability limit, and the eddy viscosity lifts it by orders of magnitude
    wherever the flow is actually straining. TRT alone keeps a quiescent or
    smooth field stable; a real fill front needs both.

    Args:
        f: Populations, shape ``(19, ...)``.
        feq: Equilibrium populations, same shape.
        rho: Density, spatial shape.
        tau: Molecular relaxation time.
        smagorinsky_cs: Smagorinsky constant.
        suppress: Optional boolean mask of cells where the eddy viscosity must
            NOT be applied. Filter cells belong here: their pore Reynolds
            number is a few hundred and the Forchheimer drag is already the
            closure for those pore-scale losses, so adding eddy viscosity on
            top double-counts them.
        magic_parameter: TRT Lambda.

    Returns:
        ``(omega_plus, omega_minus)`` arrays with the spatial shape of ``rho``.
    """
    q = strain_rate_magnitude(f, feq)
    safe_rho = np.where(np.asarray(rho) > 1e-9, rho, 1.0)
    tau_eff = 0.5 * (
        tau
        + np.sqrt(
            tau * tau
            + 18.0 * np.sqrt(2.0) * smagorinsky_cs * smagorinsky_cs * q / safe_rho
        )
    )
    if suppress is not None:
        tau_eff = np.where(suppress, tau, tau_eff)

    # Guard the far tail: a cell that has already gone bad produces a huge |Q|,
    # and an unbounded tau there would freeze it instead of letting the
    # divergence check see it.
    tau_eff = np.clip(tau_eff, tau, 10.0)

    omega_plus = 1.0 / tau_eff
    half = 1.0 / omega_plus - 0.5
    omega_minus = 1.0 / (magic_parameter / np.maximum(half, 1e-12) + 0.5)
    return omega_plus, omega_minus
