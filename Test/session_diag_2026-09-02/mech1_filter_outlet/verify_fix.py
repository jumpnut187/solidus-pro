import os, sys, copy
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import physics.filling as pf
from physics.filling import PorousZone, FLAG_FLUID, FLAG_INTERFACE, FLAG_GAS
from physics.fluid_flow import lattice_units, LATTICE_VELOCITIES, Q, OPPOSITE

NU, DX_M = 0.006 / 7200.0, 0.0152

def advance_mass_fixed(state, f_post_collision, f_streamed):
    outgoing = np.asarray(f_post_collision, dtype=np.float64)
    incoming = np.asarray(f_streamed, dtype=np.float64)
    eps = np.clip(pf.fluid_fraction(state), 0.0, 1.0)
    fluid = state.flags == FLAG_FLUID
    wet = (state.flags == FLAG_INTERFACE) | fluid
    opp = OPPOSITE.astype(np.intp)
    por = state.porous.porosity_f64
    axes = (0, 1, 2)
    delta = np.zeros(state.shape, dtype=np.float64)
    for i in range(1, Q):
        c = LATTICE_VELOCITIES[i]
        shift = (-int(c[0]), -int(c[1]), -int(c[2]))
        neighbour_wet = np.roll(wet, shift=shift, axis=axes)
        ibar = int(opp[i])
        weight = np.where(
            fluid | np.roll(fluid, shift=shift, axis=axes),
            0.5 * (por + np.roll(por, shift=shift, axis=axes)),
            0.5 * (eps + np.roll(eps, shift=shift, axis=axes)),
        )
        delta += np.where(wet & neighbour_wet,
                          (incoming[ibar] - outgoing[i]) * weight, 0.0)
    return state.mass + delta

def make(nz=16, w=3, porous=False, half_full=True):
    n = w + 2; shape = (n, n, nz + 2)
    ob = np.zeros(shape, bool)
    ob[0] = ob[-1] = True; ob[:, 0] = ob[:, -1] = True; ob[:, :, 0] = ob[:, :, -1] = True
    u = lattice_units(DX_M, NU, u_max_physical_m_s=2.8)
    z = PorousZone.inert(shape)
    if porous:
        b = np.zeros(shape, bool); b[1:w+1, 1:w+1, 5:8] = True
        z.porosity[b] = np.float32(0.85)
    init = ~ob
    if half_full:
        init = init & (np.arange(shape[2])[None, None, :] <= nz // 3)
    st = pf.create_state(ob, dx_m=DX_M, dt_s=u.dt_s, initial_fluid=init, porous=z)
    return st, u

print("=== A. porosity == 1.0 everywhere: HEAD vs Fix 1, bit-for-bit ===")
a, u = make(porous=False); b = copy.deepcopy(a)
print(f"    lattice: {int((a.flags==FLAG_FLUID).sum())} FLUID, "
      f"{int((a.flags==FLAG_INTERFACE).sum())} INTERFACE, "
      f"{int((a.flags==FLAG_GAS).sum())} GAS  (real free surface)")
worst = 0
for s in range(1, 401):
    pf.step(a, u, use_numba=False)
    orig = pf.advance_mass; pf.advance_mass = advance_mass_fixed
    try:
        pf.step(b, u, use_numba=False)
    finally:
        pf.advance_mass = orig
    if not np.array_equal(a.mass, b.mass):
        worst = s; break
if worst:
    d = np.abs(a.mass - b.mass)
    print(f"    DIVERGED at step {worst}, max|dmass| = {d.max():.3e}")
else:
    print(f"    400/400 steps BIT-IDENTICAL  (np.array_equal on state.mass)")
    print(f"    flags identical: {np.array_equal(a.flags, b.flags)}   "
          f"f identical: {np.array_equal(a.f, b.f)}   "
          f"total mass {a.mass.sum():.12g}")

print("\n=== B. porosity 0.85 band: does Fix 1 remove the amplification? ===")
for label, patch in (("HEAD  ", False), ("Fix 1 ", True)):
    st, u = make(porous=True)
    band = st.porous.porosity_f64 < 1.0
    full = st.rho.astype(np.float64) * st.porous.porosity_f64
    st.mass[band & (st.flags == FLAG_FLUID)] = (full * 1.05)[band & (st.flags == FLAG_FLUID)]
    orig = pf.advance_mass
    if patch: pf.advance_mass = advance_mass_fixed
    try:
        m0 = st.mass.sum(); out = []
        for s in range(1, 4001):
            pf.step(st, u, use_numba=False)
            if s % 1000 == 0:
                fl = st.flags == FLAG_FLUID
                fu = st.rho.astype(np.float64) * st.porous.porosity_f64
                e = np.where(fu > 0, st.mass / fu, 1.0)
                m = fl & band
                out.append(f"step {s}: max|eps-1| in band = "
                           f"{(np.abs(e[m]-1).max() if m.any() else 0):.4e}  "
                           f"dM/M0 = {(st.mass.sum()-m0)/abs(m0):+.2e}")
        print(f"  {label}: " + "\n           ".join(out))
    finally:
        pf.advance_mass = orig

print("\n=== C. Fix 1 + Fix 2 (re-pin porous FLUID cells through _redistribute) ===")
orig_trans_global = pf.apply_transitions

def repin_then_transitions(state, time_s, **kw):
    por = state.porous.porosity_f64
    m = (state.flags == FLAG_FLUID) & (por < 1.0)
    if m.any():
        full = state.rho.astype(np.float64) * por
        excess = np.zeros(state.shape, dtype=np.float64)
        excess[m] = state.mass[m] - full[m]
        state.mass[m] = full[m]
        pf._redistribute(state, excess, use_numba=False)
    return orig_trans_global(state, time_s, **kw)

for label, (pa, pt) in (("Fix1+Fix2", (True, True)), ("Fix2 only", (False, True))):
    st, u = make(porous=True)
    band = st.porous.porosity_f64 < 1.0
    full = st.rho.astype(np.float64) * st.porous.porosity_f64
    sel = band & (st.flags == FLAG_FLUID)
    st.mass[sel] = (full * 1.05)[sel]
    o1, o2 = pf.advance_mass, pf.apply_transitions
    if pa: pf.advance_mass = advance_mass_fixed
    if pt: pf.apply_transitions = repin_then_transitions
    try:
        m0 = st.mass.sum(); out = []
        for s in range(1, 4001):
            pf.step(st, u, use_numba=False)
            if s % 1000 == 0:
                fl = st.flags == FLAG_FLUID
                fu = st.rho.astype(np.float64) * st.porous.porosity_f64
                with np.errstate(invalid="ignore", divide="ignore"):
                    e = np.where(fu > 0, st.mass / fu, 1.0)
                mm = fl & band
                out.append(f"step {s}: max|eps-1| in band = "
                           f"{(np.abs(e[mm]-1).max() if mm.any() else 0):.4e}  "
                           f"dM/M0 = {(st.mass.sum()-m0)/abs(m0):+.2e}")
        print(f"  {label}: " + "\n             ".join(out))
    finally:
        pf.advance_mass, pf.apply_transitions = o1, o2

print("\n=== D. Fix 1 + Fix 2 on a NO-filter lattice: still bit-identical? ===")
a, u = make(porous=False); b = copy.deepcopy(a)
bad = 0
for s in range(1, 401):
    pf.step(a, u, use_numba=False)
    o1, o2 = pf.advance_mass, pf.apply_transitions
    pf.advance_mass = advance_mass_fixed; pf.apply_transitions = repin_then_transitions
    try:
        pf.step(b, u, use_numba=False)
    finally:
        pf.advance_mass, pf.apply_transitions = o1, o2
    if not np.array_equal(a.mass, b.mass):
        bad = s; break
print(f"    {'DIVERGED at step %d' % bad if bad else '400/400 steps BIT-IDENTICAL'}")

print("\n=== E. FULLY SUBMERGED filter (no INTERFACE cell anywhere): can Fix 2 repair? ===")
print("    _redistribute can only donate to INTERFACE neighbours; with none it")
print("    strands the excess back on the donor, so the re-pin cannot act and")
print("    the weight rule is on its own. This is the case that decides whether")
print("    Fix 1 is still needed once Fix 2 is in.")
for label, (pa, pt) in (("HEAD     ", (False, False)),
                        ("Fix2 only", (False, True)),
                        ("Fix1 only", (True, False)),
                        ("Fix1+Fix2", (True, True))):
    n = 5; shape = (n, n, 16)
    ob = np.zeros(shape, bool)
    ob[0] = ob[-1] = True; ob[:, 0] = ob[:, -1] = True; ob[:, :, 0] = ob[:, :, -1] = True
    u = lattice_units(DX_M, NU, u_max_physical_m_s=2.8)
    z = PorousZone.inert(shape)
    bnd = np.zeros(shape, bool); bnd[1:4, 1:4, 6:9] = True
    z.porosity[bnd] = np.float32(0.85)
    st = pf.create_state(ob, dx_m=DX_M, dt_s=u.dt_s, initial_fluid=~ob, porous=z)
    assert int((st.flags == FLAG_INTERFACE).sum()) == 0, "must be fully submerged"
    full = st.rho.astype(np.float64) * st.porous.porosity_f64
    st.mass[bnd] = (full * 1.05)[bnd]
    o1, o2 = pf.advance_mass, pf.apply_transitions
    if pa: pf.advance_mass = advance_mass_fixed
    if pt: pf.apply_transitions = repin_then_transitions
    try:
        m0 = st.mass.sum(); e_last = None
        for s in range(1, 6001):
            pf.step(st, u, use_numba=False)
        fu = st.rho.astype(np.float64) * st.porous.porosity_f64
        with np.errstate(invalid="ignore", divide="ignore"):
            e = np.where(fu > 0, st.mass / fu, 1.0)
        print(f"  {label}: after 6000 steps max|eps-1| in band = "
              f"{np.abs(e[bnd]-1).max():.4e}   (seeded 5.0000e-02)   "
              f"dM/M0 = {(st.mass.sum()-m0)/abs(m0):+.2e}   "
              f"interface cells = {int((st.flags==FLAG_INTERFACE).sum())}")
    finally:
        pf.advance_mass, pf.apply_transitions = o1, o2
