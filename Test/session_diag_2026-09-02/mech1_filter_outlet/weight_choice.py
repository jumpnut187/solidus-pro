"""Which weight for a porous pair involving a FLUID cell: 1.0 (plain Koerner
Table 1) or the porosity average (0.5*(p_A+p_B))?  Both are eps-free, so both
remove the feedback; they differ in which invariant they preserve.

One fixture, five variants, 6000 steps, fully submerged filter seeded 5% over
full (the case where _redistribute has no INTERFACE recipient and so cannot
repair anything).  Reports max|eps-1| in the band and, separately, whether
mass tracks rho (weight-1's invariant) or rho*porosity (the physical one).
"""
import os, sys
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import physics.filling as pf
from physics.filling import PorousZone, FLAG_FLUID, FLAG_INTERFACE
from physics.fluid_flow import lattice_units, LATTICE_VELOCITIES, Q, OPPOSITE
NU, DX_M = 0.006 / 7200.0, 0.0152


def mk_advance(mode):
    def am(state, fp, fs):
        outg = np.asarray(fp, np.float64); inc = np.asarray(fs, np.float64)
        eps_raw = pf.fluid_fraction(state); eps = np.clip(eps_raw, 0.0, 1.0)
        fluid = state.flags == FLAG_FLUID
        wet = (state.flags == FLAG_INTERFACE) | fluid
        por = state.porous.porosity_f64
        opp = OPPOSITE.astype(np.intp); axes = (0, 1, 2)
        delta = np.zeros(state.shape, np.float64)
        for i in range(1, Q):
            c = LATTICE_VELOCITIES[i]
            sh = (-int(c[0]), -int(c[1]), -int(c[2]))
            nwet = np.roll(wet, shift=sh, axis=axes); ibar = int(opp[i])
            fpair = fluid | np.roll(fluid, shift=sh, axis=axes)
            ie = 0.5 * (eps + np.roll(eps, shift=sh, axis=axes))
            if mode == "head":
                w = np.where(fpair, 1.0, ie)
                w = np.where((por < 1.0) | np.roll(por < 1.0, shift=sh, axis=axes),
                             0.5 * (eps_raw + np.roll(eps_raw, shift=sh, axis=axes)), w)
            elif mode == "w1":
                w = np.where(fpair, 1.0, ie)
            elif mode == "wpor":
                w = np.where(fpair, 0.5 * (por + np.roll(por, shift=sh, axis=axes)), ie)
            delta += np.where(wet & nwet, (inc[ibar] - outg[i]) * w, 0.0)
        return state.mass + delta
    return am


orig_trans = pf.apply_transitions


def repin(state, time_s, **kw):
    por = state.porous.porosity_f64
    m = (state.flags == FLAG_FLUID) & (por < 1.0)
    if m.any():
        full = state.rho.astype(np.float64) * por
        ex = np.zeros(state.shape, np.float64); ex[m] = state.mass[m] - full[m]
        state.mass[m] = full[m]
        pf._redistribute(state, ex, use_numba=False)
    return orig_trans(state, time_s, **kw)


def build():
    n = 5; shape = (n, n, 16)
    ob = np.zeros(shape, bool)
    ob[0] = ob[-1] = True; ob[:, 0] = ob[:, -1] = True; ob[:, :, 0] = ob[:, :, -1] = True
    u = lattice_units(DX_M, NU, u_max_physical_m_s=2.8)
    z = PorousZone.inert(shape)
    b = np.zeros(shape, bool); b[1:4, 1:4, 6:9] = True
    z.porosity[b] = np.float32(0.85)
    st = pf.create_state(ob, dx_m=DX_M, dt_s=u.dt_s, initial_fluid=~ob, porous=z)
    assert int((st.flags == FLAG_INTERFACE).sum()) == 0
    full = st.rho.astype(np.float64) * st.porous.porosity_f64
    st.mass[b] = (full * 1.05)[b]
    return st, u, b


print(f"{'variant':<22} {'max|eps-1|':>11} {'|m-rho*p|max':>13} {'|m-rho|max':>11} "
      f"{'rho band':>18} {'dM/M0':>10}")
for label, mode, pin in (("HEAD", "head", False),
                         ("weight 1", "w1", False),
                         ("weight porosity", "wpor", False),
                         ("weight 1 + repin", "w1", True),
                         ("weight porosity+repin", "wpor", True)):
    st, u, b = build()
    o1, o2 = pf.advance_mass, pf.apply_transitions
    pf.advance_mass = mk_advance(mode)
    if pin: pf.apply_transitions = repin
    try:
        m0 = st.mass.sum()
        for s in range(6000):
            pf.step(st, u, use_numba=False)
        r = st.rho.astype(np.float64); p = st.porous.porosity_f64
        fu = r * p
        with np.errstate(invalid="ignore", divide="ignore"):
            e = np.where(fu > 0, st.mass / fu, 1.0)
        print(f"{label:<22} {np.abs(e[b]-1).max():>11.4e} "
              f"{np.abs(st.mass[b]-fu[b]).max():>13.4e} "
              f"{np.abs(st.mass[b]-r[b]).max():>11.4e} "
              f"{r[b].min():.5f}-{r[b].max():.5f} "
              f"{(st.mass.sum()-m0)/abs(m0):>10.1e}")
    finally:
        pf.advance_mass, pf.apply_transitions = o1, o2
