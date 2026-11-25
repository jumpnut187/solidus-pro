from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

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

Q = 19

LATTICE_WEIGHTS = np.array(
    [1.0 / 3.0]
    + [1.0 / 18.0] * 6
    + [1.0 / 36.0] * 12,
    dtype=np.float64,
)

OPPOSITE = np.array(
    [0] + [i + 1 if i % 2 == 1 else i - 1 for i in range(1, Q)],
    dtype=np.int8,
)

CS2 = 1.0 / 3.0
INV_CS2 = 3.0

TRT_MAGIC_PARAMETER = 3.0 / 16.0

TAU_MINUS_MAX = 5.0

GRAVITY_M_S2 = 9.81

@dataclass(frozen=True)
class LatticeUnits:

    dx_m: float
    dt_s: float
    nu_lattice: float
    tau: float
    omega_plus: float
    omega_minus: float
    gravity_lattice: float
    u_max_lattice: float
    u_max_physical: float
    tau_minus_max: float = TAU_MINUS_MAX

    @property
    def velocity_scale(self) -> float:
        return self.dx_m / self.dt_s

    def to_physical_velocity(self, u_lattice: np.ndarray) -> np.ndarray:
        return np.asarray(u_lattice) * self.velocity_scale

    def to_lattice_velocity(self, u_physical: "float | np.ndarray") -> np.ndarray:
        return np.asarray(u_physical) / self.velocity_scale

def lattice_units(
    dx_m: float,
    kinematic_viscosity_m2_s: float,
    *,
    u_max_physical_m_s: float,
    u_max_lattice: float = 0.05,
    magic_parameter: float = TRT_MAGIC_PARAMETER,
    tau_minus_max: float = TAU_MINUS_MAX,
    gravity_m_s2: float = GRAVITY_M_S2,
) -> LatticeUnits:
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
    _validate_tau_minus_max(tau_minus_max)

    dt_s = u_max_lattice * dx_m / u_max_physical_m_s
    nu_lattice = kinematic_viscosity_m2_s * dt_s / (dx_m * dx_m)
    tau = 3.0 * nu_lattice + 0.5
    if tau <= 0.5:
        raise ValueError(
            f"tau={tau!r} is not above 1/2; the scheme would have negative "
            f"viscosity and diverge"
        )

    omega_plus = 1.0 / tau
    omega_minus = omega_minus_from_magic(
        omega_plus, magic_parameter, tau_minus_max
    )
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
        tau_minus_max=float(tau_minus_max),
    )

def _validate_tau_minus_max(tau_minus_max: float) -> None:
    if not tau_minus_max > 0.5:
        raise ValueError(
            f"tau_minus_max must be above 1/2 (or inf for no ceiling), got "
            f"{tau_minus_max}: at or below 1/2 it implies omega_minus >= 2, "
            f"outside the TRT stable range, and at 1/2 exactly it zeroes the "
            f"Guo force prefactor"
        )

def omega_minus_from_magic(
    omega_plus: float,
    magic_parameter: float = TRT_MAGIC_PARAMETER,
    tau_minus_max: float = TAU_MINUS_MAX,
) -> float:
    if omega_plus <= 0.0 or omega_plus >= 2.0:
        raise ValueError(f"omega_plus must be in (0, 2), got {omega_plus}")
    _validate_tau_minus_max(tau_minus_max)
    half_inverse_plus = 1.0 / omega_plus - 0.5
    if half_inverse_plus <= 0.0:
        raise ValueError(f"omega_plus={omega_plus} implies tau <= 1/2")
    tau_minus = magic_parameter / half_inverse_plus + 0.5
    if tau_minus > tau_minus_max:
        tau_minus = tau_minus_max
    return 1.0 / tau_minus

def equilibrium(rho: np.ndarray, u: np.ndarray, porosity: np.ndarray | float = 1.0) -> np.ndarray:
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
    total: np.ndarray = np.sum(np.asarray(f), axis=0)
    return total

def momentum(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f)
    out = np.zeros((3,) + f.shape[1:], dtype=np.float64)
    for i in range(Q):
        for d in range(3):
            c = LATTICE_VELOCITIES[i, d]
            if c:
                out[d] += c * f[i]
    return out

def guo_velocity(f: np.ndarray, rho: np.ndarray, force: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float64)
    corrected: np.ndarray = (
        momentum(f) + 0.5 * np.asarray(force, dtype=np.float64)
    ) / rho
    return corrected

def guo_force_terms(
    u: np.ndarray, force: np.ndarray, omega_minus: float | np.ndarray
) -> np.ndarray:
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

def collide_trt(
    f: np.ndarray,
    feq: np.ndarray,
    omega_plus: float | np.ndarray,
    omega_minus: float | np.ndarray,
    source: np.ndarray | None = None,
) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    feq = np.asarray(feq, dtype=np.float64)
    opp = OPPOSITE.astype(np.intp)

    f_bar = f[opp]
    feq_bar = feq[opp]

    f_sym = 0.5 * (f + f_bar)
    f_asym = 0.5 * (f - f_bar)
    feq_sym = 0.5 * (feq + feq_bar)
    feq_asym = 0.5 * (feq - feq_bar)

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
    f = np.asarray(f)
    out = np.empty_like(f)
    for i in range(Q):
        shift = LATTICE_VELOCITIES[i]
        out[i] = np.roll(f[i], shift=(int(shift[0]), int(shift[1]), int(shift[2])), axis=(0, 1, 2))
    return out

def bounce_back(f: np.ndarray, obstacle: np.ndarray) -> np.ndarray:
    f = np.asarray(f).copy()
    mask = np.asarray(obstacle, dtype=bool)
    if not np.any(mask):
        return f
    reflected = f[OPPOSITE.astype(np.intp)]
    f[:, mask] = reflected[:, mask]
    return f

ERGUN_PREFACTOR = 2.2

NOMINAL_FOAM_POROSITY = 0.85

FORCHHEIMER_WARD_CF = 0.55

def ceramic_foam_permeability(
    porosity: float,
    pore_diameter_m: float,
    *,
    ergun_prefactor: float = ERGUN_PREFACTOR,
) -> float:
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
    v = np.asarray(v, dtype=np.float64)
    c0 = np.asarray(c0, dtype=np.float64)
    c1 = np.asarray(c1, dtype=np.float64)
    speed = np.sqrt(np.einsum("i...,i...->...", v, v))
    denominator = c0 + np.sqrt(c0 * c0 + c1 * speed)
    resolved: np.ndarray = v / denominator
    return resolved

def porous_drag_force(
    rho: np.ndarray, u: np.ndarray, c0: np.ndarray, c1: np.ndarray
) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    c0 = np.asarray(c0, dtype=np.float64)
    c1 = np.asarray(c1, dtype=np.float64)
    speed = np.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
    coefficient = 2.0 * rho * ((2.0 * c0 - 1.0) + c1 * speed)
    drag: np.ndarray = -coefficient * u
    return drag

def capillary_entry_pressure(
    surface_tension_n_m: float,
    contact_angle_deg: float,
    pore_diameter_m: float,
    window_fraction: float,
) -> float:
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
    if density_kg_m3 <= 0:
        raise ValueError(f"density_kg_m3 must be positive, got {density_kg_m3}")
    if dt_s <= 0 or dx_m <= 0:
        raise ValueError(f"dt_s and dx_m must be positive, got {dt_s}, {dx_m}")
    return entry_pressure_pa * dt_s * dt_s / (density_kg_m3 * dx_m * dx_m * CS2)

def priming_head_m(
    entry_pressure_pa: float, density_kg_m3: float, gravity_m_s2: float = GRAVITY_M_S2
) -> float:
    return entry_pressure_pa / (density_kg_m3 * gravity_m_s2)

SMAGORINSKY_CS = 0.16

def smagorinsky_tau(
    tau: float, strain_rate_magnitude: float, smagorinsky_cs: float = SMAGORINSKY_CS
) -> float:
    nu = (tau - 0.5) / 3.0
    nu_t = smagorinsky_cs * smagorinsky_cs * abs(strain_rate_magnitude)
    return 3.0 * (nu + nu_t) + 0.5

def strain_rate_magnitude(f: np.ndarray, feq: np.ndarray) -> np.ndarray:
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
    tau_minus_max: float = TAU_MINUS_MAX,
) -> Tuple[np.ndarray, np.ndarray]:
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

    tau_eff = np.clip(tau_eff, tau, 10.0)

    omega_plus = 1.0 / tau_eff
    half = 1.0 / omega_plus - 0.5
    tau_minus = magic_parameter / np.maximum(half, 1e-12) + 0.5
    omega_minus = 1.0 / np.minimum(tau_minus, tau_minus_max)
    return omega_plus, omega_minus

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

_C_F64 = LATTICE_VELOCITIES.astype(np.float64)
_C_I64 = LATTICE_VELOCITIES.astype(np.int64)
_W_F64 = LATTICE_WEIGHTS.astype(np.float64)
_OPP_I64 = OPPOSITE.astype(np.int64)

_FLAG_OBSTACLE = 0
_FLAG_GAS = 1
_FLAG_INTERFACE = 2
_FLAG_FLUID = 3

if NUMBA_AVAILABLE:

    @njit(cache=True, inline="always")
    def _equilibrium_i(i, rho, ux, uy, uz, usqr, inv_eps, C, W):
        cu = C[i, 0] * ux + C[i, 1] * uy + C[i, 2] * uz
        return W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu * inv_eps - 1.5 * usqr * inv_eps)

    @njit(parallel=True, cache=True)
    def collide_kernel(
        f, flags, porosity, drag_c0, drag_c1,
        C, W, OPP,
        tau, gravity_lattice, gx, gy, gz,
        smagorinsky_cs, use_smagorinsky, magic, tau_minus_max,
        rho_out, u_out, post_out, rho32_out, u32_out,
    ):
        nx, ny, nz = flags.shape
        for x in prange(nx):
            eq = np.empty(19, dtype=np.float64)
            fl = np.empty(19, dtype=np.float64)
            cul = np.empty(19, dtype=np.float64)
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

                    rho = 0.0
                    mx = 0.0
                    my = 0.0
                    mz = 0.0
                    for i in range(19):
                        fi = np.float64(f[i, x, y, z])
                        fl[i] = fi
                        rho += fi
                        mx += C[i, 0] * fi
                        my += C[i, 1] * fi
                        mz += C[i, 2] * fi

                    fx = 0.0
                    fy = 0.0
                    fz = 0.0
                    if flag == _FLAG_FLUID or flag == _FLAG_INTERFACE:
                        mag = gravity_lattice * rho
                        fx = mag * gx
                        fy = mag * gy
                        fz = mag * gz

                    mx += 0.5 * fx
                    my += 0.5 * fy
                    mz += 0.5 * fz

                    safe_rho = rho if rho > 1e-12 else 1.0
                    ux = mx / safe_rho
                    uy = my / safe_rho
                    uz = mz / safe_rho

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

                    speed_u = np.sqrt(ux * ux + uy * uy + uz * uz)
                    drag = 2.0 * rho * ((2.0 * c0 - 1.0) + c1 * speed_u)
                    fx -= drag * ux
                    fy -= drag * uy
                    fz -= drag * uz

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

                    for i in range(19):
                        cul[i] = C[i, 0] * ux + C[i, 1] * uy + C[i, 2] * uz
                        eq[i] = _equilibrium_i(
                            i, rho, ux, uy, uz, usqr, inv_eps, C, W
                        )

                    omega_plus = 1.0 / tau
                    if use_smagorinsky and c1 == 0.0:
                        q00 = 0.0; q01 = 0.0; q02 = 0.0
                        q11 = 0.0; q12 = 0.0; q22 = 0.0
                        for i in range(19):
                            neq = fl[i] - eq[i]
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
                    tau_minus = magic / half + 0.5
                    if tau_minus > tau_minus_max:
                        tau_minus = tau_minus_max
                    omega_minus = 1.0 / tau_minus
                    guo_prefactor = 1.0 - 0.5 * omega_minus

                    for i in range(19):
                        ib = OPP[i]
                        fi = fl[i]
                        fb = fl[ib]
                        ei = eq[i]
                        eb = eq[ib]

                        f_sym = 0.5 * (fi + fb)
                        f_asym = 0.5 * (fi - fb)
                        e_sym = 0.5 * (ei + eb)
                        e_asym = 0.5 * (ei - eb)

                        cx = C[i, 0]; cy = C[i, 1]; cz = C[i, 2]
                        cu = cul[i]
                        src = W[i] * guo_prefactor * (
                            (3.0 * (cx - ux) + 9.0 * cu * cx) * fx
                            + (3.0 * (cy - uy) + 9.0 * cu * cy) * fy
                            + (3.0 * (cz - uz) + 9.0 * cu * cz) * fz
                        )

                        post_out[i, x, y, z] = np.float32(
                            fi
                            - omega_plus * (f_sym - e_sym)
                            - omega_minus * (f_asym - e_asym)
                            + src
                        )

    @njit(parallel=True, cache=True)
    def stream_kernel(post, flags, C, OPP, clear_gas, out):
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
        C, Ci, W, OPP, gas_density, mass_in, mass_out, f_out,
    ):
        nx, ny, nz = flags.shape
        for x in prange(nx):
            for y in range(ny):
                for z in range(nz):
                    flag = flags[x, y, z]
                    for i in range(19):
                        f_out[i, x, y, z] = streamed[i, x, y, z]
                    mass_out[x, y, z] = mass_in[x, y, z]
                    if flag != _FLAG_INTERFACE and flag != _FLAG_FLUID:
                        continue

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
                            val = ei + eb - np.float64(post[i, x, y, z])
                            if val < 0.0:
                                val = 0.0
                            f_out[ib, x, y, z] = np.float32(val)

                    por_here = porosity[x, y, z]
                    denom = np.float64(rho[x, y, z]) * por_here
                    eps_here_c = mass_in[x, y, z] / denom if denom > 0.0 else 0.0
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
                        if flag == _FLAG_FLUID or nflag == _FLAG_FLUID:
                            weight = 1.0
                        else:
                            nden = (
                                np.float64(rho[sx, sy, sz]) * porosity[sx, sy, sz]
                            )
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
                            np.float64(f_out[ib, x, y, z])
                            - np.float64(post[i, x, y, z])
                        ) * weight
                    mass_out[x, y, z] = mass_in[x, y, z] + delta

    @njit(parallel=True, cache=True)
    def advect_temperature_kernel(temperature, u, flags, alpha, temp_out):
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
                        temp_out[x, y, z] = t_here
                        continue

                    h = h_field[x, y, z] if h_is_field else h_scalar
                    coeff = h * np.float64(exposed) * dt_s / rho_cp_dx
                    temp_out[x, y, z] = t_here + (
                        -coeff * (t_here - mold_temperature_c)
                    )

    @njit(parallel=True, cache=True)
    def interface_neighbourhood_kernel(flags, Ci, gasless_out, detached_out):
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

    @njit(parallel=True, cache=True)
    def transition_scan_kernel(
        mass, rho, porosity, flags, fluid_threshold, gas_threshold,
        to_fluid_out, to_gas_out, excess_out,
    ):
        nx, ny, nz = flags.shape
        filled = 0
        emptied = 0
        for x in prange(nx):
            for y in range(ny):
                for z in range(nz):
                    excess_out[x, y, z] = 0.0
                    to_fluid_out[x, y, z] = False
                    to_gas_out[x, y, z] = False
                    if flags[x, y, z] != _FLAG_INTERFACE:
                        continue

                    full = np.float64(rho[x, y, z]) * porosity[x, y, z]
                    m = mass[x, y, z]
                    eps = m / full if full > 0.0 else 0.0
                    if eps > fluid_threshold:
                        to_fluid_out[x, y, z] = True
                        excess_out[x, y, z] = m - full
                        mass[x, y, z] = full
                        filled += 1
                    elif eps < gas_threshold:
                        to_gas_out[x, y, z] = True
                        excess_out[x, y, z] = m
                        mass[x, y, z] = 0.0
                        emptied += 1
        return filled, emptied

    @njit(parallel=True, cache=True)
    def transition_front_kernel(flags, to_fluid, to_gas, Ci, promote_out, demote_out):
        nx, ny, nz = flags.shape
        for x in prange(nx):
            for y in range(ny):
                for z in range(nz):
                    promote_out[x, y, z] = False
                    demote_out[x, y, z] = False

            for dx in range(-1, 2):
                sx = (x + dx) % nx
                for sy in range(ny):
                    for sz in range(nz):
                        fills = to_fluid[sx, sy, sz]
                        empties = to_gas[sx, sy, sz]
                        if not fills and not empties:
                            continue
                        for i in range(1, 19):
                            tx = (sx + Ci[i, 0]) % nx
                            if tx != x:
                                continue
                            ty = (sy + Ci[i, 1]) % ny
                            tz = (sz + Ci[i, 2]) % nz
                            flag = flags[tx, ty, tz]
                            if fills and flag == _FLAG_GAS:
                                promote_out[tx, ty, tz] = True
                            if empties and flag == _FLAG_FLUID:
                                demote_out[tx, ty, tz] = True

    @njit(parallel=True, cache=True)
    def redistribute_share_kernel(flags, excess, Ci, share_out, stranded_out):
        nx, ny, nz = flags.shape
        for x in prange(nx):
            for y in range(ny):
                for z in range(nz):
                    e = excess[x, y, z]
                    if e == 0.0:
                        share_out[x, y, z] = 0.0
                        stranded_out[x, y, z] = 0.0
                        continue
                    count = 0.0
                    for i in range(1, 19):
                        sx = (x + Ci[i, 0]) % nx
                        sy = (y + Ci[i, 1]) % ny
                        sz = (z + Ci[i, 2]) % nz
                        if flags[sx, sy, sz] == _FLAG_INTERFACE:
                            count += 1.0
                    if count > 0.0:
                        share_out[x, y, z] = e / count
                        stranded_out[x, y, z] = 0.0
                    else:
                        share_out[x, y, z] = 0.0
                        stranded_out[x, y, z] = e

    @njit(parallel=True, cache=True)
    def redistribute_gather_kernel(flags, share, stranded, Ci, mass):
        nx, ny, nz = flags.shape
        for x in prange(nx):
            for y in range(ny):
                for z in range(nz):
                    m = mass[x, y, z]
                    if flags[x, y, z] == _FLAG_INTERFACE:
                        for i in range(1, 19):
                            sx = (x - Ci[i, 0]) % nx
                            sy = (y - Ci[i, 1]) % ny
                            sz = (z - Ci[i, 2]) % nz
                            m += share[sx, sy, sz]
                    mass[x, y, z] = m + stranded[x, y, z]

    @njit(parallel=True, cache=True)
    def compact_mask_kernel(mask, counts, ix_out, iy_out, iz_out):
        nx, ny, nz = mask.shape
        for x in prange(nx):
            n = 0
            for y in range(ny):
                for z in range(nz):
                    if mask[x, y, z]:
                        n += 1
            counts[x + 1] = n

        counts[0] = 0
        total = 0
        for x in range(nx):
            total += counts[x + 1]
            counts[x + 1] = total

        for x in prange(nx):
            at = counts[x]
            for y in range(ny):
                for z in range(nz):
                    if mask[x, y, z]:
                        ix_out[at] = x
                        iy_out[at] = y
                        iz_out[at] = z
                        at += 1
        return total

    @njit(parallel=True, cache=True)
    def promoted_donor_average_kernel(
        px, py, pz, flags, promote, rho, u, Ci,
        reference_density, rho_avg_out, u_avg_out,
    ):
        nx, ny, nz = flags.shape
        n = px.shape[0]
        for k in prange(n):
            x = px[k]
            y = py[k]
            z = pz[k]
            count = 0.0
            rho_sum = 0.0
            ux_sum = 0.0
            uy_sum = 0.0
            uz_sum = 0.0
            for i in range(1, 19):
                sx = (x + Ci[i, 0]) % nx
                sy = (y + Ci[i, 1]) % ny
                sz = (z + Ci[i, 2]) % nz
                nflag = flags[sx, sy, sz]
                w = 0.0
                if (
                    nflag == _FLAG_FLUID or nflag == _FLAG_INTERFACE
                ) and not promote[sx, sy, sz]:
                    w = 1.0
                count += w
                rho_sum += w * np.float64(rho[sx, sy, sz])
                ux_sum += w * np.float64(u[0, sx, sy, sz])
                uy_sum += w * np.float64(u[1, sx, sy, sz])
                uz_sum += w * np.float64(u[2, sx, sy, sz])

            if count > 0.0:
                rho_avg_out[k] = rho_sum / count
                u_avg_out[0, k] = ux_sum / count
                u_avg_out[1, k] = uy_sum / count
                u_avg_out[2, k] = uz_sum / count
            else:
                rho_avg_out[k] = reference_density
                u_avg_out[0, k] = 0.0
                u_avg_out[1, k] = 0.0
                u_avg_out[2, k] = 0.0

    @njit(parallel=True, cache=True)
    def promoted_donor_temperature_mask_kernel(
        flags, promote, temperature, donor_out
    ):
        nx, ny, nz = flags.shape
        for x in prange(nx):
            for y in range(ny):
                for z in range(nz):
                    ok = False
                    nflag = flags[x, y, z]
                    if (
                        nflag == _FLAG_FLUID or nflag == _FLAG_INTERFACE
                    ) and not promote[x, y, z]:
                        ok = np.isfinite(temperature[x, y, z])
                    donor_out[x, y, z] = ok

    @njit(parallel=True, cache=True)
    def promoted_temperature_average_kernel(
        px, py, pz, donor_has_temp, temperature, Ci, fallback, temp_out
    ):
        nx, ny, nz = donor_has_temp.shape
        n = px.shape[0]
        for k in prange(n):
            x = px[k]
            y = py[k]
            z = pz[k]
            count = 0.0
            temp_sum = 0.0
            for i in range(1, 19):
                sx = (x + Ci[i, 0]) % nx
                sy = (y + Ci[i, 1]) % ny
                sz = (z + Ci[i, 2]) % nz
                if donor_has_temp[sx, sy, sz]:
                    count += 1.0
                    temp_sum += np.float64(temperature[sx, sy, sz])
            if count > 0.0:
                temp_out[k] = temp_sum / count
            else:
                temp_out[k] = fallback

def lattice_tables():
    return _C_F64, _C_I64, _W_F64, _OPP_I64
