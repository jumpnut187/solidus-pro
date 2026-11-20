import sys, os, time, collections
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np

FROM_STEP = int(os.environ.get("FROM_STEP", "5000"))
EVERY = int(os.environ.get("EVERY", "1000"))
STOP = int(os.environ.get("STOP", "12000"))
STATE = {}

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def install(fs):
    import physics.filling as pf
    from physics.filling import FLAG_GAS, FLAG_OBSTACLE, FLAG_INTERFACE, FLAG_FLUID, fluid_fraction
    from core.part_manager import PART_TYPE_TO_CODE
    orig_step = pf.step
    orig_zone = fs.build_porous_zone
    n = {"i": 0}

    def zone(dense_types, specs, **kw):
        STATE["types"] = np.asarray(dense_types)
        STATE["names"] = {c: p.value for p, c in PART_TYPE_TO_CODE.items()}
        return orig_zone(dense_types, specs, **kw)

    def step(state, units, **kw):
        out = orig_step(state, units, **kw)
        n["i"] += 1
        i = n["i"]
        if i >= FROM_STEP and i % EVERY == 0:
            f = np.asarray(state.f)
            wet = (state.flags == FLAG_FLUID) | (state.flags == FLAG_INTERFACE)
            bad = np.any(f < 0.0, axis=0) & wet
            k = int(bad.sum())
            if k:
                t = STATE["types"]; names = STATE["names"]
                bx, by, bz = np.nonzero(bad)
                hist = collections.Counter(
                    names.get(int(t[x, y, z]), f"code{int(t[x,y,z])}")
                    for x, y, z in zip(bx, by, bz))
                gas = state.flags == FLAG_GAS
                obs = state.flags == FLAG_OBSTACLE
                def near(mask):
                    d = np.zeros_like(mask)
                    for ax in (0, 1, 2):
                        for sh in (-1, 1):
                            d |= np.roll(mask, sh, axis=ax)
                    return int((d & bad).sum())
                eps = fluid_fraction(state)
                iface = int((bad & (state.flags == FLAG_INTERFACE)).sum())
                log(f"step {i}: {k} neg cells | types {dict(hist)} | "
                    f"interface {iface} ({iface/k:.0%}) | next-to-gas {near(gas)} "
                    f"| next-to-obstacle {near(obs)} | z {bz.min()}-{bz.max()} "
                    f"(mode {collections.Counter(bz.tolist()).most_common(3)}) | "
                    f"eps {eps[bad].min():.3f}..{eps[bad].max():.3f} | "
                    f"|u| max {float(np.abs(np.asarray(state.u)[:, bad]).max()):.3f}")
            else:
                log(f"step {i}: 0 neg cells")
        if i >= STOP:
            raise SystemExit(0)
        return out

    pf.step = step
    fs.build_porous_zone = zone
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
    for f in sorted(p for p in d.iterdir() if p.suffix.lower() in (".stl", ".step", ".stp")):
        try: pm.load_assembly_parts(str(f))
        except Exception as e: log(f"skip {f.name}: {e}")
    sprue = next(p for p in pm.get_all_parts() if p.part_type is PartType.SPRUE)
    b = sprue.get_bounds()
    loc = PourLocation(point_in=(float((b[0][0]+b[0][1])/2), float((b[1][0]+b[1][1])/2),
                                float(b[2][1])), normal=(0., 0., 1.), part_id=sprue.id)
    inputs = sp.PipelineInputs(alloy_name="WCB", voxel_size=0.17, refinement_level=4,
                               analysis_options={"filling": True}, pour_location=loc.to_dict())
    (_a, _b, _c, _d, mesh, _e) = sp.prepare_mesh_and_geometry(inputs, part_manager=pm, stl_loader=None)
    log(f"mesh {len(mesh.voxel_centers):,} voxels")
    options = sp.create_analysis_options({"filling": True}, use_gpu=False, use_incremental=False)
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
