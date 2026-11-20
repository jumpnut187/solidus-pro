import faulthandler, os, sys, time
from pathlib import Path
REPO = Path("/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
sys.path.insert(0, str(REPO))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
faulthandler.enable()
import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "/Users/brendonjarett/Downloads/Test 2"
voxel = float(sys.argv[2]) if len(sys.argv) > 2 else 0.17
refine = int(sys.argv[3]) if len(sys.argv) > 3 else 4
out = sys.argv[4] if len(sys.argv) > 4 else None

from core.part_manager import PartManager
from core import simulation_pipeline as sp
from core import filling_stage as fs

def main():

    def log(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    pm = PartManager()
    files = sorted(p for p in Path(d).iterdir() if p.suffix.lower() in (".stl", ".step", ".stp"))
    log(f"loading {len(files)} files")
    for f in files:
        ids = pm.load_assembly_parts(str(f))
        for pid in ids:
            p = pm.get_part(pid)
            log(f"  {p.name:40s} -> {p.part_type.value}")

    inputs = sp.PipelineInputs(alloy_name="WCB", voxel_size=voxel, refinement_level=refine)
    t0 = time.time()
    _m, _b, _v, _p, vm, _s = sp.prepare_mesh_and_geometry(
        inputs, part_manager=pm, stl_loader=None,
        progress_message_callback=lambda m: log(f"mesh: {m}"))
    log(f"voxel mesh: {len(vm.voxel_centers):,} voxels in {time.time()-t0:.1f}s")
    import collections
    log("part type histogram: " + str(collections.Counter(np.asarray(vm.voxel_part_types).tolist())))

    if out:
        centers = np.asarray(vm.voxel_centers, dtype=np.float64)
        dx_in = float(np.asarray(vm.voxel_sizes).ravel()[0])
        mapping = fs.build_grid_mapping(centers, dx_in)
        fine = fs.dense_part_types(mapping, np.asarray(vm.voxel_part_types))
        for c in (1, 2):
            np.save(f"{out}_types_c{c}.npy", fs.coarsen_part_types(fine, c))
        log(f"saved {out}_types_c1.npy / _c2.npy  (dx_in={dx_in})")
    log("DONE")

if __name__ == "__main__":
    main()
