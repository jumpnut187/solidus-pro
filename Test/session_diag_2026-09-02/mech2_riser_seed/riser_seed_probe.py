"""MECHANISM 2: find the FIRST spurious write into the upper riser on Test 2.

Watch set = RISER cells with z >= 60 (the front is below z 40 for the whole
window of interest, so ANY wet flag / nonzero mass / nonzero population there
before the front arrives is spurious). Sub-operation attribution by wrapping
every writer physics.filling.step calls, plus the gap between steps (which is
core.filling_stage's inlet pin / top-up / skin).

  CASTSIM_THREAD_BUDGET=8 ./venv/bin/python riser_seed_probe.py
"""
import sys, os, time, collections, json
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np

TYPES_NPY = ("/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim/"
             "Test/session_diag_2026-09-01/r3_t2_ledger.csv.types.npy")
ZMIN = int(os.environ.get("ZMIN", "60"))
STOP = int(os.environ.get("STOP", "17000"))
MAX_EVENTS = int(os.environ.get("MAX_EVENTS", "4"))
S = {}

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def install(fs):
    import physics.filling as pf
    from physics.filling import (FLAG_GAS, FLAG_OBSTACLE, FLAG_INTERFACE,
                                 FLAG_FLUID)
    from physics.fluid_flow import LATTICE_VELOCITIES
    from core.part_manager import PART_TYPE_TO_CODE
    S["names"] = {c: p.value for p, c in PART_TYPE_TO_CODE.items()}
    S["events"] = 0
    orig = {n: getattr(pf, n) for n in
            ("_collide_stream_numba", "free_surface_kernel", "apply_transitions",
             "heal_submerged_interface", "reset_detached_interface_cells", "step")}
    orig_zone = fs.build_porous_zone
    n = {"i": 0}
    C = LATTICE_VELOCITIES

    def zone(dense_types, specs, **kw):
        S["types"] = np.asarray(dense_types).copy()
        log(f"types captured {S['types'].shape}")
        return orig_zone(dense_types, specs, **kw)

    def prep(state):
        t = S.get("types")
        if t is None:
            t = S["types"] = np.load(TYPES_NPY)
        assert t.shape == state.shape, (t.shape, state.shape)
        z = np.arange(t.shape[2])[None, None, :]
        w = (t == 1) & (z >= ZMIN)
        S["wmask"] = w
        S["widx"] = np.nonzero(w)
        S["nw"] = int(w.sum())
        obst = state.flags == FLAG_OBSTACLE
        log(f"watch: RISER z>={ZMIN} -> {S['nw']} cells; "
            f"obstacle among them {int((w & obst).sum())}")
        # front tracker
        S["metal"] = t >= 0
        return w

    def snap(state):
        ix = S["widx"]
        return (state.flags[ix].copy(), state.mass[ix].copy())

    def viol(sn):
        from physics.filling import FLAG_GAS, FLAG_OBSTACLE
        fl, ms = sn
        bad = ((fl != FLAG_GAS) & (fl != FLAG_OBSTACLE)) | (ms != 0.0)
        return bad

    def report(state, op, bad, extra=""):
        from physics.filling import (FLAG_GAS, FLAG_OBSTACLE, FLAG_INTERFACE,
                                     FLAG_FLUID)
        FN = {0: "OBST", 1: "GAS", 2: "IFACE", 3: "FLUID"}
        ix = S["widx"]
        k = np.nonzero(bad)[0]
        xs, ys, zs = ix[0][k], ix[1][k], ix[2][k]
        t = S["types"]; nx, ny, nz = state.shape
        f = np.asarray(state.f, dtype=np.float64)
        wet = (state.flags == FLAG_FLUID) | (state.flags == FLAG_INTERFACE)
        wz = np.nonzero(wet.any(axis=(0, 1)))[0]
        log("=" * 78)
        log(f"*** FIRST SPURIOUS WRITE  step={n['i']}  op={op}  "
            f"cells={len(k)}  {extra}")
        log(f"    global wet-cell z extent: {wz.min()}-{wz.max()};  "
            f"wet total {int(wet.sum())};  "
            f"wet cells with z<60: {int(wet[:, :, :60].sum())}, "
            f"z>=60: {int(wet[:, :, 60:].sum())}")
        log(f"    watched cells z histogram: "
            f"{dict(collections.Counter(zs.tolist()).most_common(8))}")
        for j in range(min(len(k), 6)):
            x, y, z = int(xs[j]), int(ys[j]), int(zs[j])
            log(f"  --- cell ({x},{y},{z}) part={S['names'].get(int(t[x,y,z]))} "
                f"flag={FN.get(int(state.flags[x,y,z]))} "
                f"mass={float(state.mass[x,y,z]):.6g} "
                f"rho={float(state.rho[x,y,z]):.6g} "
                f"|f|sum={float(np.abs(f[:,x,y,z]).sum()):.6g} "
                f"fmin={float(f[:,x,y,z].min()):.6g}")
            for i in range(1, 19):
                cx, cy, cz = int(C[i][0]), int(C[i][1]), int(C[i][2])
                sx, sy, sz = (x + cx) % nx, (y + cy) % ny, (z + cz) % nz
                wrapped = (x + cx != sx) or (y + cy != sy) or (z + cz != sz)
                fl = int(state.flags[sx, sy, sz])
                if not wrapped and fl in (0, 1) and state.mass[sx,sy,sz] == 0:
                    continue
                log(f"        nb i={i:2d} c=({cx:2d},{cy:2d},{cz:2d}) -> "
                    f"({sx},{sy},{sz}) {'WRAPPED' if wrapped else '       '} "
                    f"part={S['names'].get(int(t[sx,sy,sz]),'mold')} "
                    f"flag={FN.get(fl)} mass={float(state.mass[sx,sy,sz]):.6g} "
                    f"|f|={float(np.abs(f[:,sx,sy,sz]).sum()):.6g}")
        log("=" * 78)
        S["events"] += 1
        if S["events"] >= MAX_EVENTS:
            raise SystemExit(0)

    def check(state, op):
        if "widx" not in S:
            return
        sn = snap(state)
        bad = viol(sn)
        if bad.any():
            report(state, op, bad)

    # --- wrapped sub-operations -------------------------------------------
    def mk(name):
        f0 = orig[name]
        def wrapped(*a, **kw):
            r = f0(*a, **kw)
            st = a[0] if a and hasattr(a[0], "flags") else None
            if st is not None:
                check(st, name)
            return r
        return wrapped

    for nm in ("apply_transitions", "heal_submerged_interface",
               "reset_detached_interface_cells", "_collide_stream_numba"):
        setattr(pf, nm, mk(nm))

    fsk0 = orig["free_surface_kernel"]
    def fsk(post, streamed, flags, rho, u, por, eo, primed, C_, Ci, W, OPP,
            gd, mass_in, mass_out, f_out):
        fsk0(post, streamed, flags, rho, u, por, eo, primed, C_, Ci, W, OPP,
             gd, mass_in, mass_out, f_out)
        if "widx" in S:
            ix = S["widx"]
            bad = (mass_out[ix] != 0.0) & (flags[ix] == 1)
            if bad.any():
                k = np.nonzero(bad)[0]
                log(f"[fsk] step {n['i']}: free_surface_kernel wrote mass into "
                    f"{len(k)} GAS watched cells, first "
                    f"({ix[0][k[0]]},{ix[1][k[0]]},{ix[2][k[0]]}) "
                    f"m={mass_out[ix][k[0]]:.6g}")
    pf.free_surface_kernel = fsk

    def step(state, units, **kw):
        n["i"] += 1
        i = n["i"]
        if "widx" not in S:
            prep(state)
        else:
            # gap since previous step returned == the stage's own writes
            prev = S.get("exit")
            if prev is not None:
                cur = snap(state)
                if (cur[0] != prev[0]).any() or (cur[1] != prev[1]).any():
                    bad = viol(cur) & ((cur[0] != prev[0]) | (cur[1] != prev[1]))
                    if bad.any():
                        report(state, "STAGE-between-steps", bad,
                               f"(changed since step {i-1} returned)")
        out = orig["step"](state, units, **kw)
        S["exit"] = snap(state)
        if i % 500 == 0:
            from physics.filling import FLAG_FLUID, FLAG_INTERFACE
            wet = (state.flags == FLAG_FLUID) | (state.flags == FLAG_INTERFACE)
            wz = np.nonzero(wet.any(axis=(0, 1)))[0]
            log(f"hb step {i} wet={int(wet.sum())} zrange={wz.min()}-{wz.max()} "
                f"watched_wet={int(wet[S['wmask']].sum())} "
                f"finite={bool(np.isfinite(state.mass).all())}")
        if i >= STOP:
            raise SystemExit(0)
        return out

    pf.step = step
    fs.build_porous_zone = zone
    import core.filling_stage as cfs
    cfs.step = step
    cfs.build_porous_zone = zone


def main():
    import scipy.spatial as ss
    o = ss.cKDTree
    class T(o):
        def __init__(self, d, *a, **kw):
            kw.setdefault("balanced_tree", False); super().__init__(d, *a, **kw)
    ss.cKDTree = T; ss.KDTree = T
    import scipy.spatial._ckdtree as ck; ck.cKDTree = T
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
    loc = PourLocation(point_in=(float((b[0][0]+b[0][1])/2),
                                 float((b[1][0]+b[1][1])/2), float(b[2][1])),
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
    t = threading.Thread(target=main); t.start(); t.join()
