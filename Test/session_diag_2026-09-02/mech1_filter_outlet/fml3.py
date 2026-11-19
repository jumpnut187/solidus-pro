"""Mechanism-1 probe v2: per-operation mass ledger at the filter's bottom plane.

Derived from Test/session_diag_2026-09-01/filter_mass_ledger.py (never ran; it
died on NUMBA_NUM_THREADS).  Changes made here:

  * the weight rule below was re-verified line by line against the CURRENT
    free_surface_kernel (physics/fluid_flow.py 1600-1637 at f1daf97) -- it was
    already correct, the porous branch uses UNCLAMPED eps on both sides
  * the global row is written even when the watch set is empty, so the onset
    can be seen before any cell is over-full
  * the watch set is (re)armed lazily: the first time any wet cell that is
    ITSELF porous (por < 1, i.e. a real FILTER cell) exceeds EPS_WATCH.  fml2's
    first pass armed on a transient RUNNER front cell instead, which drained
    again; this pass follows the persistent filter-bottom population.
  * added a whole-grid mass ledger per operation: exchange / transitions /
    heal / external(stage), summed over all non-obstacle cells, so "is mass
    created" is answered globally and not only on the watch set
  * added a whole-grid straddling-pair antisymmetry audit restricted to the
    porous boundary (every wet porous cell x every wet neighbour), which is
    cheap enough to run every AUDIT_EVERY steps
  * added w1 (weight-1) and wpor (weight-porosity) counterfactual direction
    sums per watched cell, so amplification is separable from transport

  CASTSIM_THREAD_BUDGET=8 START=5000 STOP=13000 ./venv/bin/python fml2.py
"""
from __future__ import annotations

import collections
import os
import sys
import time

sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402

START = int(os.environ.get("START", "5000"))
STOP = int(os.environ.get("STOP", "13000"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "50"))
AUDIT_EVERY = int(os.environ.get("AUDIT_EVERY", "250"))
EPS_WATCH = float(os.environ.get("EPS_WATCH", "1.05"))
OUT = os.environ.get("OUT", "fml2")

S = {}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------------------
# weight rule, replicated from free_surface_kernel (fluid_flow.py 1600-1637)
#
#   por_here = porosity[x]; denom = rho[x]*por_here
#   eps_here = mass_in[x]/denom          (UNCLAMPED)
#   eps_here_c = clip(eps_here, 0, 1)
#   if por_here < 1 or por_nb < 1:  w = 0.5*(eps_here   + eps_nb)     UNCLAMPED
#   elif flag==FLUID or nflag==FLUID: w = 1.0
#   else:                           w = 0.5*(eps_here_c + eps_nb_c)   CLAMPED
# ---------------------------------------------------------------------------
def _eps_raw(mass, rho32, por, idx):
    den = np.float64(rho32[idx]) * por[idx]
    return mass[idx] / den if den > 0.0 else 0.0


def _clamp01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def pair_weight(flag_a, flag_b, por_a, por_b, eps_a, eps_b, FLUID):
    """Weight cell A uses for its pair with B."""
    if por_a < 1.0 or por_b < 1.0:
        return 0.5 * (eps_a + eps_b), "porous"
    if flag_a == FLUID or flag_b == FLUID:
        return 1.0, "fluid"
    return 0.5 * (_clamp01(eps_a) + _clamp01(eps_b)), "iface"


def install(fs):
    import physics.filling as pf
    from physics.filling import (
        FLAG_FLUID, FLAG_INTERFACE, FLAG_OBSTACLE, fluid_fraction,
    )
    from physics.fluid_flow import lattice_tables
    from core.part_manager import PART_TYPE_TO_CODE

    _C, Ci, _W, OPP = lattice_tables()
    Ci = np.asarray(Ci, dtype=np.int64)
    OPP = np.asarray(OPP, dtype=np.int64)
    S["names"] = {c: p.value for p, c in PART_TYPE_TO_CODE.items()}
    IF = int(FLAG_FLUID)
    II = int(FLAG_INTERFACE)

    orig_zone = fs.build_porous_zone

    def zone(dense_types, specs, **kw):
        S["types"] = np.asarray(dense_types)
        log(f"types captured {S['types'].shape}")
        return orig_zone(dense_types, specs, **kw)

    fs.build_porous_zone = zone

    orig_fsk = pf.free_surface_kernel
    orig_trans = pf.apply_transitions
    orig_heal = pf.heal_submerged_interface
    orig_reset = pf.reset_detached_interface_cells
    orig_step = pf.step

    n = {"i": 0}
    cell_fh = open(f"{OUT}_cells.csv", "w", buffering=1)
    cell_fh.write(
        "step,x,y,z,type,flag,por,rho,mass_pre,eps_pre,"
        "dm_exchange,dm_w1,dm_wpor,dm_transitions,dm_heal,dm_reset,dm_external,"
        "mass_post,eps_post\n"
    )
    glob_fh = open(f"{OUT}_global.csv", "w", buffering=1)
    glob_fh.write(
        "step,exch_sum,exch_abs,exch_sum_porous,exch_sum_plain,"
        "g_trans,g_heal,g_reset,g_ext,wet_mass,tot_mass,"
        "eps_max,eps_max_por,eps_min,"
        "n_gt1_fluid,n_gt1_iface,n_gt1_por,"
        "pair_resid_max,pair_n,pair_resid_sum,pair_resid_max_straddle\n"
    )

    def porous_near(por_mask):
        adj = np.zeros_like(por_mask)
        for i in range(1, 19):
            adj |= np.roll(por_mask, tuple(Ci[i]), axis=(0, 1, 2))
        return por_mask | adj

    def watch_indices(state):
        por = state.porous.porosity_f64 < 1.0
        wet = (state.flags == FLAG_FLUID) | (state.flags == FLAG_INTERFACE)
        eps = fluid_fraction(state)
        # ITSELF porous, not merely porous-adjacent.
        m = wet & por & (eps > EPS_WATCH)
        idx = np.argwhere(m)
        if idx.shape[0] > 90:
            # keep the LOWEST cells: the filter's bottom plane, which is where
            # the reported population sits (z 14-15).
            order = np.lexsort((-eps[m], np.argwhere(m)[:, 2]))[:90]
            idx = idx[order]
        return idx

    # ------------------------------------------------------------------
    def fsk(post, streamed, flags, rho, u, porosity, entry_offset, primed,
            C, Cii, W, Op, gas_density, mass_in, mass_out, f_out):
        mass_before = mass_in.copy()
        orig_fsk(post, streamed, flags, rho, u, porosity, entry_offset,
                 primed, C, Cii, W, Op, gas_density, mass_in, mass_out, f_out)
        i = n["i"] + 1
        if i < START or i > STOP:
            return
        d = mass_out - mass_before
        wet = (flags == II) | (flags == IF)
        por = porosity < 1.0
        S["exch_sum"] = float(np.sum(d))
        S["exch_abs"] = float(np.sum(np.abs(d)))
        near = porous_near(por)
        S["exch_por"] = float(np.sum(d[near & wet]))
        S["exch_plain"] = float(np.sum(d[(~near) & wet]))

        do_audit = (i % AUDIT_EVERY == 0)
        nx, ny, nz = flags.shape

        # -------- whole-grid antisymmetry audit over the porous boundary ----
        if do_audit:
            resid_max = 0.0
            resid_max_str = 0.0
            resid_sum = 0.0
            npairs = 0
            xs, ys, zs = np.where(near & wet)
            for k in range(xs.shape[0]):
                x, y, z = int(xs[k]), int(ys[k]), int(zs[k])
                fl = int(flags[x, y, z])
                por_a = porosity[x, y, z]
                eps_a = _eps_raw(mass_before, rho, porosity, (x, y, z))
                for ii in range(1, 19):
                    sx = (x + Ci[ii, 0]) % nx
                    sy = (y + Ci[ii, 1]) % ny
                    sz = (z + Ci[ii, 2]) % nz
                    nf = int(flags[sx, sy, sz])
                    if nf != IF and nf != II:
                        continue
                    ib = int(OPP[ii])
                    por_b = porosity[sx, sy, sz]
                    eps_b = _eps_raw(mass_before, rho, porosity, (sx, sy, sz))
                    w_a, _ = pair_weight(fl, nf, por_a, por_b, eps_a, eps_b, IF)
                    w_b, _ = pair_weight(nf, fl, por_b, por_a, eps_b, eps_a, IF)
                    term_a = (np.float64(f_out[ib, x, y, z])
                              - np.float64(post[ii, x, y, z]))
                    term_b = (np.float64(f_out[ii, sx, sy, sz])
                              - np.float64(post[ib, sx, sy, sz]))
                    resid = term_a * w_a + term_b * w_b
                    resid_sum += resid
                    scale = max(abs(term_a * w_a), 1e-300)
                    rel = abs(resid) / scale
                    if rel > resid_max:
                        resid_max = rel
                    if (por_a < 1.0) != (por_b < 1.0) and rel > resid_max_str:
                        resid_max_str = rel
                    npairs += 1
            S["pair_resid"] = resid_max
            S["pair_resid_str"] = resid_max_str
            S["pair_resid_sum"] = resid_sum
            S["pair_n"] = npairs

        idx = S.get("watch")
        if idx is None or idx.shape[0] == 0:
            return
        wx, wy, wz = idx[:, 0], idx[:, 1], idx[:, 2]
        S["dm_exchange"] = d[wx, wy, wz].copy()
        S["mass_pre"] = mass_before[wx, wy, wz].copy()

        dm_w1 = np.zeros(idx.shape[0])
        dm_wpor = np.zeros(idx.shape[0])
        for k in range(idx.shape[0]):
            x, y, z = int(wx[k]), int(wy[k]), int(wz[k])
            fl = int(flags[x, y, z])
            if fl != IF and fl != II:
                continue
            pa = porosity[x, y, z]
            tot1 = 0.0
            for ii in range(1, 19):
                sx = (x + Ci[ii, 0]) % nx
                sy = (y + Ci[ii, 1]) % ny
                sz = (z + Ci[ii, 2]) % nz
                nf = int(flags[sx, sy, sz])
                if nf != IF and nf != II:
                    continue
                ib = int(OPP[ii])
                tot1 += (np.float64(f_out[ib, x, y, z])
                         - np.float64(post[ii, x, y, z]))
            dm_w1[k] = tot1
            dm_wpor[k] = tot1 * pa
        S["dm_w1"] = dm_w1
        S["dm_wpor"] = dm_wpor

    # ------------------------------------------------------------------
    def _wrap(orig, key):
        def inner(state, *a, **kw):
            i = n["i"] + 1
            if not (START <= i <= STOP):
                return orig(state, *a, **kw)
            idx = S.get("watch")
            nonob = state.flags != FLAG_OBSTACLE
            gpre = float(state.mass[nonob].sum())
            pre = (state.mass[idx[:, 0], idx[:, 1], idx[:, 2]].copy()
                   if idx is not None and idx.shape[0] else None)
            out = orig(state, *a, **kw)
            nonob2 = state.flags != FLAG_OBSTACLE
            S["g_" + key] = float(state.mass[nonob2].sum()) - gpre
            if pre is not None:
                S["dm_" + key] = state.mass[idx[:, 0], idx[:, 1], idx[:, 2]] - pre
            return out
        return inner

    trans = _wrap(orig_trans, "trans")
    heal = _wrap(orig_heal, "heal")
    reset = _wrap(orig_reset, "reset")

    # ------------------------------------------------------------------
    def step(state, units, **kw):
        i = n["i"] + 1
        idx = S.get("watch")
        nonob = state.flags != FLAG_OBSTACLE
        entry_g = float(state.mass[nonob].sum())
        entry = (state.mass[idx[:, 0], idx[:, 1], idx[:, 2]].copy()
                 if idx is not None and idx.shape[0] else None)
        prevg = S.get("prev_g")
        S["g_ext"] = (entry_g - prevg) if prevg is not None else 0.0
        prev = S.get("prev_post")
        S["dm_ext"] = ((entry - prev) if (entry is not None and prev is not None
                                         and prev.shape == entry.shape)
                       else None)

        out = orig_step(state, units, **kw)
        n["i"] = i

        if i % 2000 == 0:
            log(f"heartbeat {i} finite={bool(np.isfinite(state.mass).all())} "
                f"wet={int(((state.flags==1)|(state.flags==2)|(state.flags==3)).sum())}")

        if not (START <= i <= STOP):
            S["prev_g"] = float(state.mass[state.flags != FLAG_OBSTACLE].sum())
            return out

        if not np.isfinite(state.mass).all():
            log(f"NON-FINITE at step {i}")
            raise SystemExit(0)

        # arm the watch lazily, once
        if S.get("watch") is None or S["watch"].shape[0] == 0:
            w = watch_indices(state)
            if w.shape[0]:
                S["watch"] = w
                t = S.get("types")
                zs = w[:, 2]
                cnt = (collections.Counter(
                    S["names"].get(int(c), int(c))
                    for c in t[w[:, 0], w[:, 1], w[:, 2]]) if t is not None
                    else "?")
                log(f"WATCH ARMED at step {i}: {w.shape[0]} cells, "
                    f"z {int(zs.min())}-{int(zs.max())}, types {cnt}")
            else:
                S["watch"] = w

        idx = S.get("watch")

        if i % LOG_EVERY == 0:
            eps = fluid_fraction(state)
            wet = (state.flags == FLAG_FLUID) | (state.flags == FLAG_INTERFACE)
            por = state.porous.porosity_f64 < 1.0
            near = porous_near(por)
            hot = wet & (eps > 1.0 + 1e-3)
            nonob = state.flags != FLAG_OBSTACLE
            glob_fh.write(
                f"{i},{S.get('exch_sum', float('nan')):.10e},"
                f"{S.get('exch_abs', float('nan')):.6e},"
                f"{S.get('exch_por', float('nan')):.6e},"
                f"{S.get('exch_plain', float('nan')):.6e},"
                f"{S.get('g_trans', 0.0):.6e},{S.get('g_heal', 0.0):.6e},"
                f"{S.get('g_reset', 0.0):.6e},{S.get('g_ext', 0.0):.6e},"
                f"{float(state.mass[wet].sum()):.10g},"
                f"{float(state.mass[nonob].sum()):.10g},"
                f"{float(eps[wet].max()) if wet.any() else 0:.6g},"
                f"{float(eps[wet & near].max()) if (wet & near).any() else 0:.6g},"
                f"{float(eps[wet].min()) if wet.any() else 0:.6g},"
                f"{int((hot & (state.flags == FLAG_FLUID)).sum())},"
                f"{int((hot & (state.flags == FLAG_INTERFACE)).sum())},"
                f"{int((hot & near).sum())},"
                f"{S.get('pair_resid', float('nan')):.3e},{S.get('pair_n', 0)},"
                f"{S.get('pair_resid_sum', float('nan')):.6e},"
                f"{S.get('pair_resid_str', float('nan')):.3e}\n"
            )
            if idx is not None and idx.shape[0]:
                wx, wy, wz = idx[:, 0], idx[:, 1], idx[:, 2]
                t = S.get("types")
                post_m = state.mass[wx, wy, wz]
                rhov = np.asarray(state.rho, dtype=np.float64)[wx, wy, wz]
                porv = state.porous.porosity_f64[wx, wy, wz]
                z0 = np.zeros(len(wx))
                dm_ex = S.get("dm_exchange", z0)
                dm_w1 = S.get("dm_w1", z0)
                dm_wp = S.get("dm_wpor", z0)
                dm_tr = S.get("dm_trans", z0)
                dm_he = S.get("dm_heal", z0)
                dm_rs = S.get("dm_reset", z0)
                dm_ext = S.get("dm_ext")
                if dm_ext is None:
                    dm_ext = z0
                mass_pre = S.get("mass_pre", post_m)
                for k in range(len(wx)):
                    den = rhov[k] * porv[k]
                    cell_fh.write(
                        f"{i},{wx[k]},{wy[k]},{wz[k]},"
                        f"{int(t[wx[k],wy[k],wz[k]]) if t is not None else -1},"
                        f"{int(state.flags[wx[k],wy[k],wz[k]])},{porv[k]:.4g},"
                        f"{rhov[k]:.8g},{mass_pre[k]:.8g},"
                        f"{mass_pre[k]/den if den>0 else 0:.6g},"
                        f"{dm_ex[k]:.6e},{dm_w1[k]:.6e},{dm_wp[k]:.6e},"
                        f"{dm_tr[k]:.6e},{dm_he[k]:.6e},{dm_rs[k]:.6e},"
                        f"{dm_ext[k]:.6e},"
                        f"{post_m[k]:.8g},{post_m[k]/den if den>0 else 0:.6g}\n"
                    )
        for kk in ("dm_trans", "dm_heal", "dm_reset"):
            S.pop(kk, None)
        for kk in ("g_trans", "g_heal", "g_reset"):
            S[kk] = 0.0

        if idx is not None and idx.shape[0]:
            S["prev_post"] = state.mass[idx[:, 0], idx[:, 1], idx[:, 2]].copy()
        S["prev_g"] = float(state.mass[state.flags != FLAG_OBSTACLE].sum())

        if i >= STOP:
            log("probe reached STOP")
            raise SystemExit(0)
        return out

    pf.free_surface_kernel = fsk
    pf.apply_transitions = trans
    pf.heal_submerged_interface = heal
    pf.reset_detached_interface_cells = reset
    pf.step = step
    import core.filling_stage as cfs
    cfs.step = step
    log(f"instrumentation installed: START={START} STOP={STOP} "
        f"LOG_EVERY={LOG_EVERY} AUDIT_EVERY={AUDIT_EVERY}")


def main():
    import scipy.spatial as ss
    o = ss.cKDTree

    class T(o):
        def __init__(self, d, *a, **kw):
            kw.setdefault("balanced_tree", False)
            super().__init__(d, *a, **kw)

    ss.cKDTree = T
    ss.KDTree = T
    import scipy.spatial._ckdtree as ck
    ck.cKDTree = T
    from core.part_manager import PartManager, PartType
    from core.pour_location import PourLocation
    from core.materials import get_alloy
    from core import simulation_pipeline as sp
    from core import filling_stage as fs
    from pathlib import Path

    pm = PartManager()
    d = Path("/Users/brendonjarett/Downloads/Test 2")
    for f in sorted(p for p in d.iterdir()
                    if p.suffix.lower() in (".stl", ".step", ".stp")):
        try:
            pm.load_assembly_parts(str(f))
        except Exception as e:  # noqa: BLE001
            log(f"skip {f.name}: {e}")
    sprue = next(p for p in pm.get_all_parts() if p.part_type is PartType.SPRUE)
    b = sprue.get_bounds()
    loc = PourLocation(
        point_in=(float((b[0][0] + b[0][1]) / 2),
                  float((b[1][0] + b[1][1]) / 2), float(b[2][1])),
        normal=(0., 0., 1.), part_id=sprue.id)
    inputs = sp.PipelineInputs(alloy_name="WCB", voxel_size=0.17,
                               refinement_level=4,
                               analysis_options={"filling": True},
                               pour_location=loc.to_dict())
    (_a, _b, _c, _d, mesh, _e) = sp.prepare_mesh_and_geometry(
        inputs, part_manager=pm, stl_loader=None)
    log(f"mesh {len(mesh.voxel_centers):,} voxels")
    options = sp.create_analysis_options({"filling": True}, use_gpu=False,
                                         use_incremental=False)
    options.filling = True
    install(fs)
    try:
        fs.maybe_run_filling(mesh, get_alloy("WCB"), options, inputs,
                             progress_message_callback=lambda m: log(f"FILL: {m}"))
    except SystemExit:
        log("probe stop")


if __name__ == "__main__":
    import threading
    threading.stack_size(512 * 1024 * 1024)
    t = threading.Thread(target=main)
    t.start()
    t.join()
