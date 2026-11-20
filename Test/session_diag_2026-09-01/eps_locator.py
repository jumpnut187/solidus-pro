import sys, os, time, collections
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np

FROM_STEP = int(os.environ.get("FROM_STEP", "10000"))
EVERY = int(os.environ.get("EVERY", "250"))
STOP = int(os.environ.get("STOP", "16600"))
EPS_HIGH = float(os.environ.get("EPS_HIGH", "1.5"))
TYPES_NPY = os.environ.get(
    "TYPES_NPY",
    "/private/tmp/claude-501/-Users-brendonjarett-Desktop-Active-Codebase-Python-Cast-Sim/"
    "ae9fb16b-4a3f-4912-aa13-a696388671c4/scratchpad/r3_t2_ledger.csv.types.npy")
STATE = {}

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def census(mask, types, names, z):
    hist = collections.Counter(
        names.get(int(c), f"code{int(c)}") for c in types[mask])
    zz = z[mask]
    return dict(hist), (int(zz.min()), int(zz.max()),
                        collections.Counter(zz.tolist()).most_common(3))

def install(fs):
    import physics.filling as pf
    from physics.filling import (
        FLAG_GAS, FLAG_OBSTACLE, FLAG_INTERFACE, FLAG_FLUID, fluid_fraction)
    from core.part_manager import PART_TYPE_TO_CODE
    orig_step = pf.step
    orig_zone = fs.build_porous_zone
    n = {"i": 0}

    STATE["names"] = {c: p.value for p, c in PART_TYPE_TO_CODE.items()}

    def zone(dense_types, specs, **kw):
        STATE["types"] = np.asarray(dense_types)
        log(f"types captured {STATE['types'].shape}")
        return orig_zone(dense_types, specs, **kw)

    def step(state, units, **kw):
        out = orig_step(state, units, **kw)
        n["i"] += 1
        i = n["i"]
        finite = bool(np.isfinite(state.mass).all())
        if i % 2000 == 0:
            log(f"heartbeat step {i} finite={finite} "
                f"wet={int(((state.flags == FLAG_FLUID) | (state.flags == FLAG_INTERFACE)).sum())}")
        if "types" not in STATE:
            STATE["types"] = np.load(TYPES_NPY)
        if i >= FROM_STEP and (i % EVERY == 0 or not finite):
            f = np.asarray(state.f, dtype=np.float64)
            finite = finite and bool(np.isfinite(f).all())
            t = STATE["types"]; names = STATE["names"]
            _, _, zgrid = np.meshgrid(*[np.arange(s) for s in state.shape],
                                      indexing="ij")
            wet = (state.flags == FLAG_FLUID) | (state.flags == FLAG_INTERFACE)
            eps = fluid_fraction(state)
            por = np.asarray(state.porous.porosity) < 1.0
            hot = wet & (eps > EPS_HIGH)
            neg = np.any(f < 0.0, axis=0) & wet
            parts = []
            if hot.any():
                h, zr = census(hot, t, names, zgrid)
                iface = int((hot & (state.flags == FLAG_INTERFACE)).sum())
                parts.append(
                    f"eps>{EPS_HIGH}: {int(hot.sum())} cells {h} "
                    f"iface {iface} z {zr[0]}-{zr[1]} mode {zr[2]} "
                    f"epsmax {eps[wet].max():.2f}")
            if neg.any():
                h, zr = census(neg, t, names, zgrid)
                parts.append(f"neg: {int(neg.sum())} cells {h} "
                             f"porous {int((neg & por).sum())} "
                             f"z {zr[0]}-{zr[1]} mode {zr[2]}")
            riser_wet = int((wet & (t == 1)).sum())
            log(f"step {i}: riser wet {riser_wet} | "
                + (" | ".join(parts) if parts else "clean")
                + ("" if finite else "  <<< NON-FINITE"))
            if not finite:
                raise SystemExit(0)
        if i >= STOP:
            raise SystemExit(0)
        return out

    pf.step = step
    import core.filling_stage as cfs
    cfs.step = step

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
