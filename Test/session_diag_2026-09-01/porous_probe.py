"""Does a porous cell's MOMENTUM actually feel the drag, or only its velocity?

Guo-Zhao brakes the flow with a body force F = eps*g - (eps nu/k1) u
- (eps c_F/sqrt(k1)) |u| u.  The solver resolves u from that force implicitly
(resolve_porous_velocity), but the force itself never reaches the collision's
Guo source.  If that is the bug, then a fully periodic block of filter under
gravity has no terminal velocity: sum_i c_i f_i grows linearly for ever while
u -- the number the equilibrium is built from -- creeps up only as sqrt.  The
gap between the two is the non-equilibrium the TRT never relaxes (omega_minus
~ 2e-4), and once |f_asym| passes f_sym the upstream population goes negative.

Runs both the numba and numpy paths so a fix has to land in both.
"""
import os, sys
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np

from physics.filling import create_state, step, PorousZone
from physics.fluid_flow import (
    lattice_units, porous_drag_coefficients, ceramic_foam_permeability,
    forchheimer_inverse_length, LATTICE_VELOCITIES, Q,
)

DX = 0.0043
NU = 8.5e-7
UMAXP = float(np.sqrt(2.0 * 9.81 * 0.40))
POROSITY = 0.85
PORE_D = 2.54e-3


def make(porous_everywhere=True, n=6):
    units = lattice_units(DX, NU, u_max_physical_m_s=UMAXP, u_max_lattice=0.05)
    k1 = ceramic_foam_permeability(POROSITY, PORE_D)
    ik2 = forchheimer_inverse_length(k1)
    c0, c1 = porous_drag_coefficients(
        POROSITY, k1, ik2, nu_lattice=units.nu_lattice, dx_m=DX)
    shape = (n, n, n)
    zone = PorousZone.inert(shape)
    if porous_everywhere:
        zone.porosity[...] = np.float32(POROSITY)
        zone.drag_c0[...] = np.float32(c0)
        zone.drag_c1[...] = np.float32(c1)
    obstacle = np.zeros(shape, dtype=bool)
    st = create_state(obstacle, dx_m=DX, dt_s=units.dt_s,
                      initial_fluid=np.ones(shape, dtype=bool), porous=zone)
    return st, units, (k1, ik2, c0, c1)


def momentum_z(st):
    f = np.asarray(st.f, dtype=np.float64)
    m = np.zeros(st.shape)
    for i in range(Q):
        if LATTICE_VELOCITIES[i, 2]:
            m += LATTICE_VELOCITIES[i, 2] * f[i]
    return m


def terminal_u(units, c0, c1):
    """Solve 2*c0*u + c1*u^2 = g_lattice for the physical terminal velocity."""
    g = units.gravity_lattice
    # v = 2 c0 u + c1 |u| u, and at terminal the per-step gravity increment is
    # exactly balanced: g = (2c0 - 1) u + c1 u^2  (the -1 is the un-dragged part)
    a, b, c = c1, 2.0 * c0 - 1.0, -g
    return (-b + np.sqrt(b * b + 4 * a * g)) / (2 * a)


def run(label, use_numba, steps=400, every=50):
    st, units, (k1, ik2, c0, c1) = make()
    print(f"\n=== {label} ===")
    print(f"  dt={units.dt_s:.3e}s tau={units.tau:.6f} omega+={units.omega_plus:.4f} "
          f"omega-={units.omega_minus:.3e} g_l={units.gravity_lattice:.3e}")
    print(f"  k1={k1:.3e} 1/k2={ik2:.3e} c0={c0:.4f} c1={c1:.4f}")
    print(f"  algebraic terminal u_lattice = {terminal_u(units, c0, c1):.5f}")
    for s in range(1, steps + 1):
        step(st, units, track_free_surface=False, use_numba=use_numba)
        if s % every == 0 or s == 1:
            f = np.asarray(st.f, dtype=np.float64)
            mz = momentum_z(st)
            print(f"  step {s:5d}: rho {f.sum(0).mean():.6f} "
                  f"mom_z {mz.mean():+.6e}  u_z {np.asarray(st.u)[2].mean():+.6e} "
                  f"min f {f.min():+.6e}  negs {(f < 0).sum()}")
            if not np.isfinite(f).all():
                print("  NON-FINITE"); break


if __name__ == "__main__":
    run("numba", True)
    run("numpy", False)
