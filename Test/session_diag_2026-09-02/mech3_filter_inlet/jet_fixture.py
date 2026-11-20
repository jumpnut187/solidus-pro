from __future__ import annotations

import os
import sys

sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

import physics.filling as pf
from physics.filling import PorousZone, FLAG_FLUID, FLAG_INTERFACE, FLAG_GAS
from physics.fluid_flow import lattice_units, equilibrium

DX_M = 0.0043
NU = 0.006 / 7200.0
C0 = 0.500022828579
C1 = 1.381721139
POR = 0.85

NZ = int(os.environ.get("NZ", "44"))
W = int(os.environ.get("W", "5"))
BAND = (22, 27)
STEPS = int(os.environ.get("STEPS", "2000"))
TAUM_MAX = float(os.environ.get("TAUM_MAX", "5.0"))

def units():
    return lattice_units(DX_M, NU, u_max_physical_m_s=2.8)

def build(c1_scale=1.0, porous=True, geom="full"):
    n = W + 2
    shape = (n, n, NZ)
    ob = np.zeros(shape, dtype=bool)
    ob[0] = ob[-1] = True
    ob[:, 0] = ob[:, -1] = True
    ob[:, :, 0] = True
    ob[:, :, -1] = True
    u = units()
    z = PorousZone.inert(shape)
    band = np.zeros(shape, dtype=bool)
    ys = slice(1, W + 1)
    xs = slice(1, W + 1)
    if geom in ("lateral", "corner"):
        ys = slice(1, 1 + (W + 1) // 2)
    if geom == "corner":
        xs = slice(1, 1 + (W + 1) // 2)
    band[xs, ys, BAND[0]:BAND[1] + 1] = True
    if porous:
        z.porosity[band] = np.float32(POR)
        z.drag_c0[band] = np.float32(C0)
        if c1_scale:
            z.drag_c1[band] = np.float32(C1 * c1_scale)
    st = pf.create_state(ob, dx_m=DX_M, dt_s=u.dt_s,
                         initial_fluid=~ob, porous=z)
    assert int((st.flags == FLAG_INTERFACE).sum()) == 0
    assert int((st.flags == FLAG_GAS).sum()) == 0
    return st, u, band

_orig_smag = pf.smagorinsky_relaxation

def smag_no_suppress(f, feq, rho, tau, **kw):
    kw.pop("suppress", None)
    return _orig_smag(f, feq, rho, tau, **kw)

def smag_clamp_taum(f, feq, rho, tau, **kw):
    op, om = _orig_smag(f, feq, rho, tau, **kw)
    om = np.asarray(om, dtype=np.float64)
    floor = 1.0 / TAUM_MAX
    return op, np.maximum(om, floor)

def smag_plus_only(f, feq, rho, tau, **kw):
    kw2 = dict(kw); kw2.pop("suppress", None)
    op_free, _om_free = _orig_smag(f, feq, rho, tau, **kw2)
    _op_sup, om_sup = _orig_smag(f, feq, rho, tau, **kw)
    return op_free, om_sup

def smag_minus_only(f, feq, rho, tau, **kw):
    kw2 = dict(kw); kw2.pop("suppress", None)
    _op_free, om_free = _orig_smag(f, feq, rho, tau, **kw2)
    op_sup, _om_sup = _orig_smag(f, feq, rho, tau, **kw)
    return op_sup, om_free

VARIANTS = {
    "stock":      dict(c1=1.0, porous=True,  smag=None),
    "lat_stock":  dict(c1=1.0, porous=True,  smag=None,     geom="lateral"),
    "lat_smag":   dict(c1=1.0, porous=True,  smag="nosup",  geom="lateral"),
    "lat_taum":   dict(c1=1.0, porous=True,  smag="clamp",  geom="lateral"),
    "lat_dragx2": dict(c1=2.0, porous=True,  smag=None,     geom="lateral"),
    "lat_smagP":  dict(c1=1.0, porous=True,  smag="plus",   geom="lateral"),
    "lat_smagM":  dict(c1=1.0, porous=True,  smag="minus",  geom="lateral"),
    "full_supOnly": dict(c1=1.0, porous=False, smag=None, geom="full",
                         keep_suppress=True),
    "full_porSup":  dict(c1=0.0, porous=True,  smag=None, geom="full",
                         keep_suppress=True),
    "lat_supOnly": dict(c1=1.0, porous=False, smag=None, geom="lateral",
                        keep_suppress=True),
    "lat_porOnly": dict(c1=0.0, porous=True,  smag="nosup", geom="lateral"),
    "lat_porSup":  dict(c1=0.0, porous=True,  smag=None, geom="lateral",
                        keep_suppress=True),
    "cor_stock":  dict(c1=1.0, porous=True,  smag=None,     geom="corner"),
    "cor_taum":   dict(c1=1.0, porous=True,  smag="clamp",  geom="corner"),
    "drag_off":   dict(c1=0.0, porous=True,  smag=None, keep_suppress=True),
    "drag_half":  dict(c1=0.5, porous=True,  smag=None),
    "drag_x2":    dict(c1=2.0, porous=True,  smag=None),
    "smag_on":    dict(c1=1.0, porous=True,  smag="nosup"),
    "taum_clamp": dict(c1=1.0, porous=True,  smag="clamp"),
    "no_filter":  dict(c1=1.0, porous=False, smag=None),
}

def run(name, U, cfg):
    st, un, band = build(cfg["c1"], cfg["porous"], cfg.get("geom", "full"))
    orig = pf.smagorinsky_relaxation
    if cfg.get("keep_suppress"):
        def keep(f, feq, rho, tau, **kw):
            kw["suppress"] = band
            return _orig_smag(f, feq, rho, tau, **kw)
        pf.smagorinsky_relaxation = keep
    elif cfg["smag"] == "nosup":
        pf.smagorinsky_relaxation = smag_no_suppress
    elif cfg["smag"] == "clamp":
        pf.smagorinsky_relaxation = smag_clamp_taum
    elif cfg["smag"] == "plus":
        pf.smagorinsky_relaxation = smag_plus_only
    elif cfg["smag"] == "minus":
        pf.smagorinsky_relaxation = smag_minus_only

    inlet = (slice(1, W + 1), slice(1, W + 1), slice(1, 3))
    jet = np.zeros((3, W, W, 2), dtype=np.float64)
    jet[2] = U
    feq_in = equilibrium(np.ones((W, W, 2)), jet)
    try:
        umax_hist = []
        bad = None
        for s in range(1, STEPS + 1):
            st.f[:, inlet[0], inlet[1], inlet[2]] = feq_in.astype(np.float32)
            pf.step(st, un, use_numba=False)
            with np.errstate(over="ignore", invalid="ignore"):
                sp = np.sqrt(np.sum(np.asarray(st.u, np.float64) ** 2, axis=0))
                m = float(np.nanmax(sp)) if np.any(np.isfinite(sp)) else np.inf
            umax_hist.append(m)
            if not (np.all(np.isfinite(st.f)) and np.all(np.isfinite(st.mass))
                    and np.isfinite(m)):
                bad = s
                break
        return bad, umax_hist
    finally:
        pf.smagorinsky_relaxation = orig

def main():
    only = os.environ.get("ONLY", "")
    Us = [float(x) for x in os.environ.get(
        "US", "0.05,0.10,0.15,0.20,0.30,0.40").split(",")]
    un = units()
    print(f"lattice: dx={DX_M} dt={un.dt_s:.6g} nu_l={un.nu_lattice:.6g} "
          f"tau={un.tau:.6f} omega_plus={un.omega_plus:.6f} "
          f"omega_minus={un.omega_minus:.6g} tau_minus={1/un.omega_minus:.1f}")
    print(f"filter: porosity={POR} c0={C0} c1={C1}   band z{BAND[0]}-{BAND[1]}, "
          f"channel {W}x{W}x{NZ}, {STEPS} steps, NumPy path\n")
    hdr = f"{'variant':<12}" + "".join(f"{('U='+str(u)):>13}" for u in Us)
    print(hdr)
    print("-" * len(hdr))
    for name, cfg in VARIANTS.items():
        if only and name not in only.split(","):
            continue
        cells = []
        for U in Us:
            bad, hist = run(name, U, cfg)
            if bad:
                cells.append(f"{'DIED@'+str(bad):>13}")
            else:
                cells.append(f"{max(hist):>13.4f}")
        print(f"{name:<12}" + "".join(cells), flush=True)
    print("\ncell = max|u| reached over the run (survived), or DIED@step "
          "(non-finite f/mass/u).")

if __name__ == "__main__":
    main()
