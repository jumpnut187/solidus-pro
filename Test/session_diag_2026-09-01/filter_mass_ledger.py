from __future__ import annotations

import collections
import os
import sys
import time

sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

START = int(os.environ.get("START", "8000"))
STOP = int(os.environ.get("STOP", "12500"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "50"))
AUDIT_EVERY = int(os.environ.get("AUDIT_EVERY", "250"))
EPS_WATCH = float(os.environ.get("EPS_WATCH", "1.05"))
OUT = os.environ.get("OUT", "fml")

S = {}

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def _eps_raw(mass, rho32, por, idx):
    den = np.float64(rho32[idx]) * por[idx]
    return mass[idx] / den if den > 0.0 else 0.0

def _clamp01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

def pair_weight(flag_a, flag_b, por_a, por_b, eps_a, eps_b, FLUID):
    if por_a < 1.0 or por_b < 1.0:
        return 0.5 * (eps_a + eps_b), "porous"
    if flag_a == FLUID or flag_b == FLUID:
        return 1.0, "fluid"
    return 0.5 * (_clamp01(eps_a) + _clamp01(eps_b)), "iface"

def install(fs):
    import physics.filling as pf
    from physics.filling import (
        FLAG_FLUID, FLAG_INTERFACE, FLAG_GAS, FLAG_OBSTACLE, fluid_fraction,
    )
    from physics.fluid_flow import lattice_tables
    from core.part_manager import PART_TYPE_TO_CODE

    _C, Ci, _W, OPP = lattice_tables()
    Ci = np.asarray(Ci, dtype=np.int64)
    OPP = np.asarray(OPP, dtype=np.int64)
    S["names"] = {c: p.value for p, c in PART_TYPE_TO_CODE.items()}

    orig_zone = fs.build_porous_zone

    def zone(dense_types, specs, **kw):
        S["types"] = np.asarray(dense_types)
        log(f"types captured {S['types'].shape}")
        return orig_zone(dense_types, specs, **kw)

    fs.build_porous_zone = zone

    orig_fsk = pf.free_surface_kernel
    orig_trans = pf.apply_transitions
    orig_heal = pf.heal_submerged_interface
    orig_step = pf.step

    n = {"i": 0}
    cell_fh = open(f"{OUT}_cells.csv", "w", buffering=1)
    cell_fh.write(
        "step,x,y,z,type,flag,por,rho,mass_pre,eps_pre,"
        "dm_exchange,dm_w1,dm_transitions,dm_heal,dm_external,"
        "mass_post,eps_post\n"
    )
    glob_fh = open(f"{OUT}_global.csv", "w", buffering=1)
    glob_fh.write(
        "step,exch_sum_delta,exch_sum_abs,wet_mass,eps_max,eps_max_porousadj,"
        "n_eps_gt1_fluid,n_eps_gt1_iface,pair_max_resid,pair_n\n"
    )

    def watch_indices(state):
        por = np.asarray(state.porous.porosity) < 1.0
        adj = np.zeros_like(por)
        for i in range(1, 19):
            adj |= np.roll(por, tuple(Ci[i]), axis=(0, 1, 2))
        near = por | adj
        wet = (state.flags == FLAG_FLUID) | (state.flags == FLAG_INTERFACE)
        eps = fluid_fraction(state)
        m = wet & near & (eps > EPS_WATCH)
        idx = np.argwhere(m)
        if idx.shape[0] > 90:
            order = np.argsort(-eps[m])[:90]
            idx = idx[order]
        return idx

    def fsk(post, streamed, flags, rho, u, porosity, entry_offset, primed,
            C, Cii, W, Op, gas_density, mass_in, mass_out, f_out):
        orig_fsk(post, streamed, flags, rho, u, porosity, entry_offset,
                 primed, C, Cii, W, Op, gas_density, mass_in, mass_out, f_out)
        i = n["i"] + 1
        if i < START or i > STOP:
            return
        d = mass_out - mass_in
        S["exch_sum"] = float(np.sum(d))
        S["exch_abs"] = float(np.sum(np.abs(d)))
        idx = S.get("watch")
        if idx is None or idx.shape[0] == 0:
            return
        wx, wy, wz = idx[:, 0], idx[:, 1], idx[:, 2]
        S["dm_exchange"] = d[wx, wy, wz].copy()
        S["mass_pre"] = mass_in[wx, wy, wz].copy()

        do_audit = (i % AUDIT_EVERY == 0)
        nx, ny, nz = flags.shape
        dm_w1 = np.zeros(idx.shape[0])
        resid_max = 0.0
        npairs = 0
        for k in range(idx.shape[0]):
            x, y, z = int(wx[k]), int(wy[k]), int(wz[k])
            fl = int(flags[x, y, z])
            if fl not in (int(FLAG_FLUID), int(FLAG_INTERFACE)):
                continue
            por_a = porosity[x, y, z]
            eps_a = _eps_raw(mass_in, rho, porosity, (x, y, z))
            tot1 = 0.0
            for ii in range(1, 19):
                sx = (x + Ci[ii, 0]) % nx
                sy = (y + Ci[ii, 1]) % ny
                sz = (z + Ci[ii, 2]) % nz
                nf = int(flags[sx, sy, sz])
                if nf not in (int(FLAG_FLUID), int(FLAG_INTERFACE)):
                    continue
                ib = int(OPP[ii])
                term = (np.float64(f_out[ib, x, y, z])
                        - np.float64(post[ii, x, y, z]))
                tot1 += term
                if not do_audit:
                    continue
                por_b = porosity[sx, sy, sz]
                eps_b = _eps_raw(mass_in, rho, porosity, (sx, sy, sz))
                w_a, _ = pair_weight(fl, nf, por_a, por_b, eps_a, eps_b,
                                     int(FLAG_FLUID))
                w_b, _ = pair_weight(nf, fl, por_b, por_a, eps_b, eps_a,
                                     int(FLAG_FLUID))
                term_b = (np.float64(f_out[ii, sx, sy, sz])
                          - np.float64(post[ib, sx, sy, sz]))
                resid = term * w_a + term_b * w_b
                scale = max(abs(term * w_a), 1e-300)
                resid_max = max(resid_max, abs(resid) / scale)
                npairs += 1
            dm_w1[k] = tot1
        S["dm_w1"] = dm_w1
        S["pair_resid"] = resid_max
        S["pair_n"] = npairs

    def trans(state, time_s, **kw):
        i = n["i"] + 1
        idx = S.get("watch")
        if idx is None or not (START <= i <= STOP):
            return orig_trans(state, time_s, **kw)
        wx, wy, wz = idx[:, 0], idx[:, 1], idx[:, 2]
        pre = state.mass[wx, wy, wz].copy()
        out = orig_trans(state, time_s, **kw)
        S["dm_trans"] = state.mass[wx, wy, wz] - pre
        return out

    def heal(state, time_s, *a, **kw):
        i = n["i"] + 1
        idx = S.get("watch")
        if idx is None or not (START <= i <= STOP):
            return orig_heal(state, time_s, *a, **kw)
        wx, wy, wz = idx[:, 0], idx[:, 1], idx[:, 2]
        pre = state.mass[wx, wy, wz].copy()
        out = orig_heal(state, time_s, *a, **kw)
        S["dm_heal"] = state.mass[wx, wy, wz] - pre
        return out

    def step(state, units, **kw):
        i = n["i"] + 1
        idx = S.get("watch")
        entry = None
        if idx is not None and START <= i <= STOP:
            entry = state.mass[idx[:, 0], idx[:, 1], idx[:, 2]].copy()
        out = orig_step(state, units, **kw)
        n["i"] = i

        if i == START:
            S["watch"] = watch_indices(state)
            t = S.get("types")
            zs = S["watch"][:, 2]
            log(f"watch set at step {i}: {S['watch'].shape[0]} cells, "
                f"z {zs.min()}-{zs.max()}, types "
                f"{collections.Counter(S['names'].get(int(c), c) for c in t[S['watch'][:,0], S['watch'][:,1], S['watch'][:,2]]) if t is not None else '?'}")

        if i % 2000 == 0:
            log(f"heartbeat {i} finite={bool(np.isfinite(state.mass).all())}")

        if not (START <= i <= STOP):
            return out
        if not np.isfinite(state.mass).all():
            log(f"NON-FINITE at step {i}")
            raise SystemExit(0)

        if i % LOG_EVERY == 0 and idx is not None and idx.shape[0]:
            eps = fluid_fraction(state)
            wet = (state.flags == FLAG_FLUID) | (state.flags == FLAG_INTERFACE)
            por = np.asarray(state.porous.porosity) < 1.0
            adj = np.zeros_like(por)
            for k in range(1, 19):
                adj |= np.roll(por, tuple(Ci[k]), axis=(0, 1, 2))
            near = por | adj
            hot = wet & (eps > 1.0 + 1e-3)
            glob_fh.write(
                f"{i},{S.get('exch_sum', float('nan')):.10e},"
                f"{S.get('exch_abs', float('nan')):.6e},"
                f"{float(np.sum(state.mass[wet])):.8g},"
                f"{float(eps[wet].max()):.6g},"
                f"{float(eps[wet & near].max()) if (wet & near).any() else 0.0:.6g},"
                f"{int((hot & (state.flags == FLAG_FLUID)).sum())},"
                f"{int((hot & (state.flags == FLAG_INTERFACE)).sum())},"
                f"{S.get('pair_resid', float('nan')):.3e},{S.get('pair_n', 0)}\n"
            )
            wx, wy, wz = idx[:, 0], idx[:, 1], idx[:, 2]
            t = S.get("types")
            post_m = state.mass[wx, wy, wz]
            rho = np.asarray(state.rho, dtype=np.float64)[wx, wy, wz]
            porv = np.asarray(state.porous.porosity, dtype=np.float64)[wx, wy, wz]
            dm_ex = S.get("dm_exchange", np.zeros(len(wx)))
            dm_w1 = S.get("dm_w1", np.zeros(len(wx)))
            dm_tr = S.get("dm_trans", np.zeros(len(wx)))
            dm_he = S.get("dm_heal", np.zeros(len(wx)))
            mass_pre = S.get("mass_pre", post_m)
            prev = S.get("prev_post")
            dm_ext = (entry - prev) if (entry is not None and prev is not None
                                        and prev.shape == entry.shape) \
                else np.zeros(len(wx))
            for k in range(len(wx)):
                den = rho[k] * porv[k]
                cell_fh.write(
                    f"{i},{wx[k]},{wy[k]},{wz[k]},"
                    f"{int(t[wx[k],wy[k],wz[k]]) if t is not None else -1},"
                    f"{int(state.flags[wx[k],wy[k],wz[k]])},{porv[k]:.4g},"
                    f"{rho[k]:.8g},{mass_pre[k]:.8g},"
                    f"{mass_pre[k]/den if den>0 else 0:.6g},"
                    f"{dm_ex[k]:.6e},{dm_w1[k]:.6e},{dm_tr[k]:.6e},"
                    f"{dm_he[k]:.6e},{dm_ext[k]:.6e},"
                    f"{post_m[k]:.8g},{post_m[k]/den if den>0 else 0:.6g}\n"
                )
            S["dm_trans"] = np.zeros(len(wx))
            S["dm_heal"] = np.zeros(len(wx))

        if idx is not None and idx.shape[0]:
            S["prev_post"] = state.mass[idx[:, 0], idx[:, 1], idx[:, 2]].copy()

        if i >= STOP:
            log("probe reached STOP")
            raise SystemExit(0)
        return out

    pf.free_surface_kernel = fsk
    pf.apply_transitions = trans
    pf.heal_submerged_interface = heal
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
        except Exception as e:
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
