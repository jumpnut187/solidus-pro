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


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------
#
# The NumPy functions above are the reference implementation: readable, easy to
# assert against analytically, and the thing the benchmarks were validated with.
# They are also ~40x too slow, because every one of them allocates a full
# (19, nx, ny, nz) temporary and runs on a single core -- np.roll alone is
# called 90 times per step.
#
# These kernels fuse the whole local half of a step (moments, porous drag,
# equilibrium, Smagorinsky, Guo forcing, TRT collision) into one parallel pass,
# and streaming plus bounce-back into a second. tests/test_filling_parity.py
# holds them to the NumPy path, mirroring how tests/test_parallel_kernel_parity
# guards the existing solidification kernels.

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where numba is absent
    NUMBA_AVAILABLE = False

#: Float views of the lattice tables. Numba wants concrete dtypes, and the int8
#: originals would force a cast in the inner loop.
_C_F64 = LATTICE_VELOCITIES.astype(np.float64)
_C_I64 = LATTICE_VELOCITIES.astype(np.int64)
_W_F64 = LATTICE_WEIGHTS.astype(np.float64)
_OPP_I64 = OPPOSITE.astype(np.int64)

#: Cell flags, duplicated here so the kernels stay free of physics.filling.
#: Kept in sync by tests/test_filling_parity.py.
_FLAG_OBSTACLE = 0
_FLAG_GAS = 1
_FLAG_INTERFACE = 2
_FLAG_FLUID = 3


if NUMBA_AVAILABLE:

    @njit(cache=True, inline="always")
    def _equilibrium_i(i, rho, ux, uy, uz, usqr, inv_eps, C, W):
        """One equilibrium population. Inlined into the kernels."""
        cu = C[i, 0] * ux + C[i, 1] * uy + C[i, 2] * uz
        return W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu * inv_eps - 1.5 * usqr * inv_eps)

    @njit(parallel=True, cache=True)
    def collide_kernel(
        f, flags, porosity, drag_c0, drag_c1,
        C, W, OPP,
        tau, gravity_lattice, gx, gy, gz,
        smagorinsky_cs, use_smagorinsky, magic,
        rho_out, u_out, post_out, rho32_out, u32_out,
    ):
        """
        Fused local pass: moments -> drag -> equilibrium -> Smagorinsky ->
        Guo forcing -> TRT collision.

        Every operation here is cell-local, which is what makes the whole thing
        one parallel loop with no temporaries. Writes ``rho_out``, ``u_out``
        and the post-collision populations ``post_out`` in place.

        ``rho32_out`` and ``u32_out`` receive the same moments rounded to
        float32 -- the dtype the solver state stores them in. Written here
        rather than as ``rho_out.astype(np.float32)`` afterwards for the same
        reason ``free_surface_kernel`` writes its own float32 copy: the cast is
        identical, it just happens on 18 threads instead of one. The float64
        pair stays because the free-surface pass wants the unrounded velocity.

        ``f`` is read through an explicit ``np.float64`` cast at every site so
        the caller can hand it ``state.f`` (float32) directly instead of
        materialising ``state.f.astype(np.float64)`` -- a 19 x N float64
        temporary, ~390 MB of traffic per step at a 120^3 grid, and the single
        largest line in the measured astype churn. The casts are not
        cosmetic: without them ``0.5 * (fi + fb)`` in the TRT split would add
        two float32 values in float32 and the kernel would stop matching the
        NumPy reference bit-for-bit. float32 -> float64 is exact, so with them
        the result is identical to the old pre-converted input.
        """
        nx, ny, nz = flags.shape
        for x in prange(nx):
            for y in range(ny):
                for z in range(nz):
                    flag = flags[x, y, z]
                    if flag == _FLAG_OBSTACLE:
                        rho_out[x, y, z] = 0.0
                        u_out[0, x, y, z] = 0.0
                        u_out[1, x, y, z] = 0.0
                        u_out[2, x, y, z] = 0.0
                        rho32_out[x, y, z] = 0.0
                        u32_out[0, x, y, z] = 0.0
                        u32_out[1, x, y, z] = 0.0
                        u32_out[2, x, y, z] = 0.0
                        for i in range(19):
                            post_out[i, x, y, z] = 0.0
                        continue

                    # --- zeroth and first moments ---
                    rho = 0.0
                    mx = 0.0
                    my = 0.0
                    mz = 0.0
                    for i in range(19):
                        fi = np.float64(f[i, x, y, z])
                        rho += fi
                        mx += C[i, 0] * fi
                        my += C[i, 1] * fi
                        mz += C[i, 2] * fi

                    # --- gravity, wet cells only ---
                    fx = 0.0
                    fy = 0.0
                    fz = 0.0
                    if flag == _FLAG_FLUID or flag == _FLAG_INTERFACE:
                        mag = gravity_lattice * rho
                        fx = mag * gx
                        fy = mag * gy
                        fz = mag * gz

                    # Guo half-force folded into the velocity.
                    mx += 0.5 * fx
                    my += 0.5 * fy
                    mz += 0.5 * fz

                    safe_rho = rho if rho > 1e-12 else 1.0
                    ux = mx / safe_rho
                    uy = my / safe_rho
                    uz = mz / safe_rho

                    # --- Guo-Zhao porous drag (identity off a filter) ---
                    c0 = drag_c0[x, y, z]
                    c1 = drag_c1[x, y, z]
                    speed = np.sqrt(ux * ux + uy * uy + uz * uz)
                    denom = c0 + np.sqrt(c0 * c0 + c1 * speed)
                    ux /= denom
                    uy /= denom
                    uz /= denom

                    if flag == _FLAG_GAS:
                        ux = 0.0
                        uy = 0.0
                        uz = 0.0

                    rho_out[x, y, z] = rho
                    u_out[0, x, y, z] = ux
                    u_out[1, x, y, z] = uy
                    u_out[2, x, y, z] = uz
                    rho32_out[x, y, z] = rho
                    u32_out[0, x, y, z] = ux
                    u32_out[1, x, y, z] = uy
                    u32_out[2, x, y, z] = uz

                    eps = porosity[x, y, z]
                    inv_eps = 1.0 / eps if eps > 0.0 else 1.0
                    usqr = ux * ux + uy * uy + uz * uz

                    # --- Smagorinsky, from the local non-equilibrium stress ---
                    omega_plus = 1.0 / tau
                    if use_smagorinsky and c1 == 0.0:
                        q00 = 0.0; q01 = 0.0; q02 = 0.0
                        q11 = 0.0; q12 = 0.0; q22 = 0.0
                        for i in range(19):
                            neq = np.float64(f[i, x, y, z]) - _equilibrium_i(
                                i, rho, ux, uy, uz, usqr, inv_eps, C, W
                            )
                            cx = C[i, 0]; cy = C[i, 1]; cz = C[i, 2]
                            q00 += cx * cx * neq
                            q01 += cx * cy * neq
                            q02 += cx * cz * neq
                            q11 += cy * cy * neq
                            q12 += cy * cz * neq
                            q22 += cz * cz * neq
                        qmag = np.sqrt(
                            q00 * q00 + q11 * q11 + q22 * q22
                            + 2.0 * (q01 * q01 + q02 * q02 + q12 * q12)
                        )
                        tau_eff = 0.5 * (
                            tau + np.sqrt(
                                tau * tau
                                + 18.0 * 1.4142135623730951
                                * smagorinsky_cs * smagorinsky_cs * qmag / safe_rho
                            )
                        )
                        if tau_eff > 10.0:
                            tau_eff = 10.0
                        if tau_eff < tau:
                            tau_eff = tau
                        omega_plus = 1.0 / tau_eff

                    half = 1.0 / omega_plus - 0.5
                    if half < 1e-12:
                        half = 1e-12
                    omega_minus = 1.0 / (magic / half + 0.5)
                    guo_prefactor = 1.0 - 0.5 * omega_minus

                    # --- TRT collision with Guo source ---
                    for i in range(19):
                        ib = OPP[i]
                        # Cast before the symmetric/antisymmetric split: see
                        # the docstring -- fi + fb in float32 would not match
                        # the NumPy path.
                        fi = np.float64(f[i, x, y, z])
                        fb = np.float64(f[ib, x, y, z])
                        ei = _equilibrium_i(i, rho, ux, uy, uz, usqr, inv_eps, C, W)
                        eb = _equilibrium_i(ib, rho, ux, uy, uz, usqr, inv_eps, C, W)

                        f_sym = 0.5 * (fi + fb)
                        f_asym = 0.5 * (fi - fb)
                        e_sym = 0.5 * (ei + eb)
                        e_asym = 0.5 * (ei - eb)

                        cx = C[i, 0]; cy = C[i, 1]; cz = C[i, 2]
                        cu = cx * ux + cy * uy + cz * uz
                        src = W[i] * guo_prefactor * (
                            (3.0 * (cx - ux) + 9.0 * cu * cx) * fx
                            + (3.0 * (cy - uy) + 9.0 * cu * cy) * fy
                            + (3.0 * (cz - uz) + 9.0 * cu * cz) * fz
                        )

                        post_out[i, x, y, z] = (
                            fi
                            - omega_plus * (f_sym - e_sym)
                            - omega_minus * (f_asym - e_asym)
                            + src
                        )

    @njit(parallel=True, cache=True)
    def stream_kernel(post, flags, C, OPP, clear_gas, out):
        """
        Pull streaming with halfway bounce-back.

        Gathers into ``out`` so each thread owns one destination cell and no
        atomics are needed. Wraps periodically to match the NumPy reference;
        real domains are closed by obstacle cells, so the wrap never fires.

        ``C`` must be the **integer** lattice table: it is used as an array
        index here, not as arithmetic, and the float64 view would make every
        offset a float.
        """
        nx, ny, nz = flags.shape
        for x in prange(nx):
            for y in range(ny):
                for z in range(nz):
                    if flags[x, y, z] == _FLAG_OBSTACLE:
                        for i in range(19):
                            out[i, x, y, z] = 0.0
                        continue
                    if clear_gas and flags[x, y, z] == _FLAG_GAS:
                        for i in range(19):
                            out[i, x, y, z] = 0.0
                        continue

                    out[0, x, y, z] = post[0, x, y, z]
                    for i in range(1, 19):
                        cx = C[i, 0]; cy = C[i, 1]; cz = C[i, 2]

                        # The population arriving at x in direction i comes
                        # from x - c_i. If that neighbour is solid, nothing
                        # streams from it; instead this cell's own population
                        # aimed that way bounces straight back reversed.
                        #
                        # The obstacle test is on x - c_i, NOT x + c_i. Testing
                        # the downstream neighbour reflects the wrong
                        # population and leaves density and velocity looking
                        # correct -- they are computed from the pre-collision
                        # field -- while quietly corrupting the populations and
                        # the mass budget.
                        sx = (x - cx) % nx
                        sy = (y - cy) % ny
                        sz = (z - cz) % nz
                        if flags[sx, sy, sz] == _FLAG_OBSTACLE:
                            out[i, x, y, z] = post[OPP[i], x, y, z]
                        else:
                            out[i, x, y, z] = post[i, sx, sy, sz]


    @njit(parallel=True, cache=True)
    def free_surface_kernel(
        post, streamed, flags, rho, u, porosity, entry_offset, primed,
        C, Ci, W, OPP, gas_density, mass_in, mass_out, f_out, f32_out,
    ):
        """
        Fused free-surface pass: gas-side reconstruction + Koerner mass exchange.

        Both are per-cell with neighbour *reads* only, so they collapse into one
        parallel loop. Doing them in NumPy cost more than the collision did: the
        reconstruction built a full (19, nx, ny, nz) gas equilibrium for every
        cell in the domain when only interface cells need one, and the mass
        exchange ran 18 np.roll calls, each allocating a full grid.

        Reads ``mass_in`` and writes ``mass_out`` so neighbour fill fractions
        are all taken from the same instant; updating in place would let a cell
        see some neighbours before and some after their update.

        Two population outputs, deliberately:

        * ``f_out`` is float64 and is what the mass exchange below reads its own
          reconstructed populations back out of. It has to stay float64 --
          ``physics.filling.advance_mass`` feeds the float64 ``reconstructed``
          array into the exchange too, and rounding it to float32 first would
          make the two paths solve slightly different problems.
        * ``f32_out`` is the float32 copy that becomes ``state.f``. Writing it
          here rather than as ``f_out.astype(np.float32)`` afterwards moves a
          19 x N conversion (~390 MB of traffic per step at 120^3) out of the
          serial NumPy tail and into this already-parallel loop. Every cell of
          every direction is assigned unconditionally at the top, so the buffer
          is fully defined on return.

        ``rho`` is read through an explicit ``np.float64`` cast so the caller
        can pass ``state.rho`` (float32) directly. That is not a shortcut, it is
        the parity requirement: the NumPy path's fill fraction comes from
        ``fluid_fraction``, which reads ``state.rho.astype(np.float64)`` -- the
        float32-rounded density, not the raw float64 the collision produced.
        ``u``, by contrast, is the raw float64 velocity, because that is exactly
        what the NumPy path hands ``reconstruct_gas_populations``.
        """
        nx, ny, nz = flags.shape
        for x in prange(nx):
            for y in range(ny):
                for z in range(nz):
                    flag = flags[x, y, z]
                    for i in range(19):
                        val0 = streamed[i, x, y, z]
                        f_out[i, x, y, z] = val0
                        f32_out[i, x, y, z] = val0
                    mass_out[x, y, z] = mass_in[x, y, z]
                    if flag != _FLAG_INTERFACE and flag != _FLAG_FLUID:
                        continue

                    # --- gas-side reconstruction (interface cells only) ---
                    if flag == _FLAG_INTERFACE:
                        rho_g = gas_density
                        if not primed[x, y, z]:
                            rho_g += entry_offset[x, y, z]
                        ux = u[0, x, y, z]
                        uy = u[1, x, y, z]
                        uz = u[2, x, y, z]
                        usqr = ux * ux + uy * uy + uz * uz
                        eps_p = porosity[x, y, z]
                        inv_eps = 1.0 / eps_p if eps_p > 0.0 else 1.0
                        for i in range(1, 19):
                            bx = (x + Ci[i, 0]) % nx
                            by = (y + Ci[i, 1]) % ny
                            bz = (z + Ci[i, 2]) % nz
                            if flags[bx, by, bz] != _FLAG_GAS:
                                continue
                            ib = OPP[i]
                            ei = _equilibrium_i(i, rho_g, ux, uy, uz, usqr, inv_eps, C, W)
                            eb = _equilibrium_i(ib, rho_g, ux, uy, uz, usqr, inv_eps, C, W)
                            # Floored at zero. Nothing in the Koerner closure
                            # bounds this below: where the interface moves
                            # fast, or a cell has most of its neighbours in
                            # gas, ``post[i]`` can exceed ei + eb and the
                            # reconstruction returns a negative population.
                            # A negative population is a negative contribution
                            # to density, and the equilibrium's quadratic-in-u
                            # terms amplify whatever density it produces.
                            #
                            # Measured on the real assembly: isolated negatives
                            # first appear at step 4,638 and self-heal for
                            # thousands of steps, then cascade -- 1 cell at
                            # step 12,606 to 417 by 12,651, density through
                            # zero at 12,644, the whole field non-finite at
                            # 12,650. See tests/test_filling_stability.py.
                            #
                            # The floor is a stability bound, not a physical
                            # law, and it adds the mass it clips. It fires
                            # rarely enough that total wet mass stayed at 0.69
                            # of the wet-cell count across the run, so the
                            # trade is a small, bounded mass gain in exchange
                            # for the solve surviving at all.
                            val = ei + eb - post[i, x, y, z]
                            if val < 0.0:
                                val = 0.0
                            f_out[ib, x, y, z] = val
                            f32_out[ib, x, y, z] = val

                    # --- Koerner mass exchange ---
                    #
                    # Pair weighting, in exact parity with
                    # physics.filling.advance_mass -- read its docstring for
                    # why. Koerner's Table 1 averages the fill fraction over
                    # interface-interface pairs only; a pair involving a FLUID
                    # cell exchanges at weight 1, which makes a FLUID cell's
                    # mass change equal its own density change and so stops it
                    # drifting. Pairs touching a filter keep the eps average
                    # because a full porous cell holds rho * porosity, not rho.
                    por_here = porosity[x, y, z]
                    denom = np.float64(rho[x, y, z]) * por_here
                    eps_here = mass_in[x, y, z] / denom if denom > 0.0 else 0.0
                    eps_here_c = eps_here
                    if eps_here_c < 0.0:
                        eps_here_c = 0.0
                    elif eps_here_c > 1.0:
                        eps_here_c = 1.0
                    delta = 0.0
                    for i in range(1, 19):
                        sx = (x + Ci[i, 0]) % nx
                        sy = (y + Ci[i, 1]) % ny
                        sz = (z + Ci[i, 2]) % nz
                        nflag = flags[sx, sy, sz]
                        if nflag != _FLAG_INTERFACE and nflag != _FLAG_FLUID:
                            continue
                        por_nb = porosity[sx, sy, sz]
                        if por_here < 1.0 or por_nb < 1.0:
                            nden = np.float64(rho[sx, sy, sz]) * por_nb
                            eps_nb = (
                                mass_in[sx, sy, sz] / nden if nden > 0.0 else 0.0
                            )
                            weight = 0.5 * (eps_here + eps_nb)
                        elif flag == _FLAG_FLUID or nflag == _FLAG_FLUID:
                            weight = 1.0
                        else:
                            nden = np.float64(rho[sx, sy, sz]) * por_nb
                            eps_nb = (
                                mass_in[sx, sy, sz] / nden if nden > 0.0 else 0.0
                            )
                            if eps_nb < 0.0:
                                eps_nb = 0.0
                            elif eps_nb > 1.0:
                                eps_nb = 1.0
                            weight = 0.5 * (eps_here_c + eps_nb)
                        ib = OPP[i]
                        delta += (
                            (f_out[ib, x, y, z] - post[i, x, y, z]) * weight
                        )
                    mass_out[x, y, z] = mass_in[x, y, z] + delta

    # -----------------------------------------------------------------------
    # Thermal kernels
    # -----------------------------------------------------------------------
    #
    # The two thermal passes were the last pure-NumPy work left in the per-step
    # loop, and at a 120^3 grid they cost 20 ms and 10 ms of a 106 ms step --
    # single-threaded, so they were also the Amdahl ceiling on the whole solve
    # (step() alone scales 402 ms -> 81 ms from 1 to 18 threads; the serial
    # tail does not scale at all).
    #
    # Each mirrors a NumPy reference in physics.filling -- advect_temperature
    # and apply_wall_heat_loss -- which stays the NUMBA_AVAILABLE == False
    # fallback. tests/test_filling_parity.py::ThermalKernelParityTests holds
    # the pairs together; a one-sided edit to either half fails there.

    @njit(parallel=True, cache=True)
    def advect_temperature_kernel(temperature, u, flags, alpha, temp_out):
        """
        Upwind advection plus a 6-point Laplacian over wet cells, one step.

        Parity contract with :func:`physics.filling.advect_temperature`, which
        is the readable reference:

        * Non-wet cells (anything but INTERFACE or FLUID) are copied through
          untouched, sentinel NaN included. A gas cell has no temperature to
          advect and inventing one would turn a misrun into a plausible-looking
          cold casting.
        * A neighbour that is not wet contributes the cell's *own* value, which
          is the Neumann (zero-gradient) boundary the NumPy path spells
          ``_neighbour_or_self``. That is what keeps NaN out of the stencil.
        * ``t_plus`` is the neighbour at index **+1** along the axis. The NumPy
          reference reaches it with ``np.roll(shift=-1)``, which brings index+1
          down to index -- the sign flips when a roll becomes index arithmetic,
          the same way it does in :func:`stream_kernel`, and getting it backwards
          silently reverses the upwind direction while leaving the Laplacian
          (which is symmetric) looking correct.
        * Wraps periodically, matching ``np.roll``. Real domains are closed by
          obstacle cells, which are not wet, so the wrap contributes the
          Neumann branch rather than a value from the far side.

        Arithmetic is float64 throughout with an explicit cast on every float32
        read, and the accumulation order matches the reference's axis loop, so
        the two agree bit-for-bit rather than merely closely. ``temp_out`` must
        be a different array from ``temperature``: every cell reads its
        neighbours' *old* values.
        """
        nx, ny, nz = flags.shape
        for x in prange(nx):
            for y in range(ny):
                for z in range(nz):
                    flag = flags[x, y, z]
                    if flag != _FLAG_INTERFACE and flag != _FLAG_FLUID:
                        temp_out[x, y, z] = temperature[x, y, z]
                        continue

                    t_here = np.float64(temperature[x, y, z])
                    convective = 0.0
                    laplacian = 0.0
                    for axis in range(3):
                        ax = 1 if axis == 0 else 0
                        ay = 1 if axis == 1 else 0
                        az = 1 if axis == 2 else 0

                        xp = (x + ax) % nx
                        yp = (y + ay) % ny
                        zp = (z + az) % nz
                        fp = flags[xp, yp, zp]
                        if fp == _FLAG_INTERFACE or fp == _FLAG_FLUID:
                            t_plus = np.float64(temperature[xp, yp, zp])
                        else:
                            t_plus = t_here

                        xm = (x - ax) % nx
                        ym = (y - ay) % ny
                        zm = (z - az) % nz
                        fm = flags[xm, ym, zm]
                        if fm == _FLAG_INTERFACE or fm == _FLAG_FLUID:
                            t_minus = np.float64(temperature[xm, ym, zm])
                        else:
                            t_minus = t_here

                        u_axis = np.float64(u[axis, x, y, z])
                        # Upwind: the gradient comes from whichever side the
                        # flow arrives from. A NaN velocity fails the test and
                        # takes the forward branch, exactly as np.where does.
                        if u_axis >= 0.0:
                            gradient = t_here - t_minus
                        else:
                            gradient = t_plus - t_here
                        convective += u_axis * gradient
                        laplacian += t_plus + t_minus - 2.0 * t_here

                    temp_out[x, y, z] = t_here - convective + alpha * laplacian

    @njit(parallel=True, cache=True)
    def wall_heat_loss_kernel(
        temperature, flags, h_scalar, h_field, h_is_field,
        mold_temperature_c, dt_s, rho_cp_dx, temp_out,
    ):
        """
        Lumped heat loss through faces that border an OBSTACLE cell.

        ``dT = -h * n_faces * (T - T_mold) * dt / (rho cp dx)``, once per
        exposed face, so a corner cell loses proportionally more than a cell
        with one flat wall. Parity contract with
        :func:`physics.filling.apply_wall_heat_loss`.

        The exposed-face count is recomputed per cell from the six *face*
        neighbours -- not the 18-direction lattice neighbourhood; a diagonal
        touch shares no area with the mold. Six uint8 reads per cell inside an
        already-parallel loop is cheaper than the six full-grid ``np.roll``
        calls the NumPy path needs, and it cannot go stale: the NumPy fallback
        caches the count on the state instead (see
        ``physics.filling._wall_exposed_faces``) precisely because rebuilding it
        every step was 10 ms of the measured 106 ms step for geometry that has
        not changed since ``create_state``.

        ``h`` arrives either as a scalar or as a full field -- gating and
        casting surfaces carry different HTCs -- and Numba cannot branch on an
        argument's type, so the caller passes both plus ``h_is_field``. The
        unused one is a 1x1x1 dummy.

        ``rho_cp_dx`` is ``density * specific_heat * dx`` precomputed by the
        caller in that order, so the division rounds identically to the NumPy
        expression.
        """
        nx, ny, nz = flags.shape
        for x in prange(nx):
            for y in range(ny):
                for z in range(nz):
                    flag = flags[x, y, z]
                    if flag != _FLAG_INTERFACE and flag != _FLAG_FLUID:
                        temp_out[x, y, z] = temperature[x, y, z]
                        continue

                    exposed = 0
                    if flags[(x + 1) % nx, y, z] == _FLAG_OBSTACLE:
                        exposed += 1
                    if flags[(x - 1) % nx, y, z] == _FLAG_OBSTACLE:
                        exposed += 1
                    if flags[x, (y + 1) % ny, z] == _FLAG_OBSTACLE:
                        exposed += 1
                    if flags[x, (y - 1) % ny, z] == _FLAG_OBSTACLE:
                        exposed += 1
                    if flags[x, y, (z + 1) % nz] == _FLAG_OBSTACLE:
                        exposed += 1
                    if flags[x, y, (z - 1) % nz] == _FLAG_OBSTACLE:
                        exposed += 1

                    t_here = np.float64(temperature[x, y, z])
                    if exposed == 0:
                        # No exposed face means q_wall is exactly zero here,
                        # not merely small. The NumPy path reaches the same
                        # value through ``T + 0.0``.
                        temp_out[x, y, z] = t_here
                        continue

                    h = h_field[x, y, z] if h_is_field else h_scalar
                    coeff = h * np.float64(exposed) * dt_s / rho_cp_dx
                    temp_out[x, y, z] = t_here + (
                        -coeff * (t_here - mold_temperature_c)
                    )

    # -----------------------------------------------------------------------
    # Free-surface neighbourhood scans
    # -----------------------------------------------------------------------

    @njit(parallel=True, cache=True)
    def interface_neighbourhood_kernel(flags, Ci, gasless_out, detached_out):
        """
        Classify every INTERFACE cell by what its 18 lattice neighbours hold.

        Two boolean outputs, both False everywhere the cell is not INTERFACE:

        * ``gasless_out`` -- no neighbour is GAS. The front has closed over
          this cell; :func:`physics.filling.heal_submerged_interface` is what
          acts on it.
        * ``detached_out`` -- no neighbour is wet (INTERFACE or FLUID). A
          droplet smaller than the grid can represent;
          :func:`physics.filling.reset_detached_interface_cells` acts on it.

        Both replace an 18 x ``np.roll`` full-grid scan -- 10 ms and 3 ms of a
        measured 106 ms step at 120^3, to find a handful of cells each. Both
        results are produced in one sweep because the 18 neighbour flags are
        already loaded, but the two callers **must** call this separately:
        ``apply_transitions`` and ``heal_submerged_interface`` each mutate
        ``flags`` before the next one looks, and precomputing across that
        boundary would answer a question about a lattice that no longer exists.

        The 18 directions are ``LATTICE_VELOCITIES[1:]`` -- D3Q19 minus rest,
        the same neighbourhood the front advances along. The set is closed
        under negation, so counting ``x + c_i`` gives the same totals as the
        NumPy references' ``x - c_i`` rolls; only the count is used, never the
        direction.
        """
        nx, ny, nz = flags.shape
        for x in prange(nx):
            for y in range(ny):
                for z in range(nz):
                    if flags[x, y, z] != _FLAG_INTERFACE:
                        gasless_out[x, y, z] = False
                        detached_out[x, y, z] = False
                        continue
                    gas = 0
                    wet = 0
                    for i in range(1, 19):
                        sx = (x + Ci[i, 0]) % nx
                        sy = (y + Ci[i, 1]) % ny
                        sz = (z + Ci[i, 2]) % nz
                        nflag = flags[sx, sy, sz]
                        if nflag == _FLAG_GAS:
                            gas += 1
                        elif nflag == _FLAG_INTERFACE or nflag == _FLAG_FLUID:
                            wet += 1
                    gasless_out[x, y, z] = gas == 0
                    detached_out[x, y, z] = wet == 0


def lattice_tables():
    """The lattice tables in the dtypes the kernels expect."""
    return _C_F64, _C_I64, _W_F64, _OPP_I64
