"""Closed box, metal on top of a ceramic slab, gravity: does it get through
without losing mass or producing a negative population? Sizing probe."""
import os, sys
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np

from physics.filling import (
    create_state, step, total_mass, PorousZone, FLAG_GAS, FLAG_OBSTACLE,
)
from physics.fluid_flow import (
    lattice_units, porous_drag_coefficients, ceramic_foam_permeability,
    forchheimer_inverse_length,
)

DX = 0.0043
NU = 0.006 / 7200.0


def slab(n=16, band=(4, 6)):
    units = lattice_units(DX, NU, u_max_physical_m_s=2.8)
    obstacle = np.zeros((n, n, n), dtype=bool)
    obstacle[0] = obstacle[-1] = True
    obstacle[:, 0] = obstacle[:, -1] = True
    obstacle[:, :, 0] = obstacle[:, :, -1] = True
    k1 = ceramic_foam_permeability(0.85, 2.54e-3)
    ik2 = forchheimer_inverse_length(k1)
    c0, c1 = porous_drag_coefficients(0.85, k1, ik2,
                                      nu_lattice=units.nu_lattice, dx_m=DX)
    zone = PorousZone.inert((n, n, n))
    sl = slice(band[0], band[1] + 1)
    zone.porosity[1:-1, 1:-1, sl] = np.float32(0.85)
    zone.drag_c0[1:-1, 1:-1, sl] = np.float32(c0)
    zone.drag_c1[1:-1, 1:-1, sl] = np.float32(c1)
    fluid = np.zeros((n, n, n), dtype=bool)
    fluid[1:-1, 1:-1, band[1] + 2:n - 2] = True
    st = create_state(obstacle, dx_m=DX, dt_s=units.dt_s,
                      initial_fluid=fluid, porous=zone)
    st.porous.primed[...] = True
    return st, units, (c0, c1)


def run(use_numba, steps=4000, every=500):
    st, units, (c0, c1) = slab()
    por = np.asarray(st.porous.porosity) < 1.0
    below = np.zeros(st.shape, dtype=bool)
    below[1:-1, 1:-1, 1:4] = True
    m0 = total_mass(st)
    print(f"  c0={c0:.4f} c1={c1:.4f} m0={m0:.6f} porous cells={int(por.sum())}")
    worst_neg_por = worst_neg = 0
    for s in range(1, steps + 1):
        step(st, units, use_numba=use_numba)
        f = np.asarray(st.f, dtype=np.float64)
        wet = (st.flags != FLAG_GAS) & (st.flags != FLAG_OBSTACLE)
        negc = np.any(f < 0.0, axis=0) & wet
        worst_neg = max(worst_neg, int(negc.sum()))
        worst_neg_por = max(worst_neg_por, int((negc & por).sum()))
        if s % every == 0:
            m = total_mass(st)
            print(f"  step {s:5d} mass {m:.6f} drift {(m-m0)/m0:+.3e} "
                  f"below-slab mass {st.mass[below].sum():.4f} "
                  f"wet-porous {int((por & wet).sum())} "
                  f"neg {int(negc.sum())} negpor {int((negc & por).sum())} "
                  f"|u|max {np.abs(np.asarray(st.u)).max():.4f}")
    print(f"  worst neg {worst_neg} worst neg_porous {worst_neg_por}")


print("numba"); run(True)
print("numpy"); run(False, steps=1500, every=500)
