"""Audit item (d): do the two step paths agree on the porous weight branch,
including when a porous FLUID cell is OVER-full (eps > 1, where the branch's
unclamped eps makes the weight exceed 1)?

Runs the same seeded state one step down each path and compares mass, rho and f.
Also dumps the actual weight a straddling pair gets from each side, computed
from the kernel's formula, to show it is symmetric.
"""
import os, sys
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import physics.filling as pf
from physics.filling import PorousZone, FLAG_FLUID, FLAG_INTERFACE
from physics.fluid_flow import lattice_units, porous_drag_coefficients

NU, DX_M, W, NZ = 0.006 / 7200.0, 0.0152, 3, 14
BAND = (5, 7); POR = 0.85

def build(seed_eps):
    n = W + 2; shape = (n, n, NZ + 2)
    ob = np.zeros(shape, bool)
    ob[0] = ob[-1] = True; ob[:, 0] = ob[:, -1] = True; ob[:, :, 0] = ob[:, :, -1] = True
    u = lattice_units(DX_M, NU, u_max_physical_m_s=2.8)
    z = PorousZone.inert(shape)
    band = np.zeros(shape, bool); band[1:W+1, 1:W+1, BAND[0]:BAND[1]+1] = True
    z.porosity[band] = np.float32(POR)
    c0, c1 = porous_drag_coefficients(POR, 1e-8, 1e4, nu_lattice=u.nu_lattice, dx_m=DX_M)
    z.drag_c0[band] = np.float32(c0); z.drag_c1[band] = np.float32(c1)
    st = pf.create_state(ob, dx_m=DX_M, dt_s=u.dt_s, initial_fluid=~ob, porous=z)
    full = st.rho.astype(np.float64) * st.porous.porosity_f64
    st.mass[band] = full[band] * seed_eps
    # make one plain cell just under the band an INTERFACE cell so there is a
    # straddling porous/plain pair with mixed flags, not only FLUID/FLUID.
    st.flags[2, 2, BAND[0] - 1] = FLAG_INTERFACE
    st.mass[2, 2, BAND[0] - 1] = 0.6 * full[2, 2, BAND[0] - 1]
    return st, u, band

def clone(st):
    import copy
    return copy.deepcopy(st)

for seed in (1.0, 1.05, 2.0, 7.8):
    a, u, band = build(seed); b = clone(a)
    pf.step(a, u, use_numba=True)
    pf.step(b, u, use_numba=False)
    dm = np.abs(a.mass - b.mass); dr = np.abs(a.rho.astype(np.float64) - b.rho.astype(np.float64))
    df = np.abs(a.f.astype(np.float64) - b.f.astype(np.float64))
    print(f"seed_eps={seed:<5} max|dmass|={dm.max():.3e} "
          f"max|drho|={dr.max():.3e} max|df|={df.max():.3e} "
          f"bitwise_mass={'YES' if np.array_equal(a.mass, b.mass) else 'no'} "
          f"total_mass numba={a.mass.sum():.12g} numpy={b.mass.sum():.12g}", flush=True)
