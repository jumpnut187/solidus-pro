"""Does Full_Gating_Test's endgame stall come from the new boundary, or just
from pouring faster? Same build, same mesh, target rate swept."""
import sys, os, time
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np


def main():
    import scipy.spatial as ss
    orig = ss.cKDTree
    class T(orig):
        def __init__(self, data, *a, **kw):
            kw.setdefault("balanced_tree", False); super().__init__(data, *a, **kw)
    ss.cKDTree = T; ss.KDTree = T
    import scipy.spatial._ckdtree as ck; ck.cKDTree = T

    from core.part_manager import PartManager, PartType
    from core.pour_location import PourLocation
    from core.materials import get_alloy
    from core import simulation_pipeline as sp
    from core import filling_stage as fs
    from pathlib import Path

    pm = PartManager()
    d = Path("/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim/Test_Model/Full_Gating_Test")
    for f in sorted(p for p in d.iterdir() if p.suffix.lower() in (".stl", ".step", ".stp")):
        pm.load_assembly_parts(str(f))
    parts = pm.get_all_parts()
    sprue = next(p for p in parts if p.part_type is PartType.SPRUE)
    b = sprue.get_bounds()
    loc = PourLocation(point_in=(float((b[0][0]+b[0][1])/2), float((b[1][0]+b[1][1])/2),
                                 float(b[2][1])), normal=(0., 0., 1.), part_id=sprue.id)
    inputs = sp.PipelineInputs(alloy_name="WCB", voxel_size=0.20, refinement_level=4,
                               analysis_options={"filling": True}, pour_location=loc.to_dict())
    (_a, _b2, _c, _d, mesh, _e) = sp.prepare_mesh_and_geometry(
        inputs, part_manager=pm, stl_loader=None)
    alloy = get_alloy("WCB")
    for rate in (float(x) for x in sys.argv[1:]):
        options = sp.create_analysis_options({"filling": True}, use_gpu=False,
                                             use_incremental=False)
        options.filling = True
        options.solidification_params.filling.pour_rate_lb_s = rate
        t0 = time.time(); msgs = []
        r = fs.maybe_run_filling(mesh, alloy, options, inputs,
                                 progress_message_callback=msgs.append)
        peak = max((float(m.split("delivering ")[1].split(" of")[0])
                    for m in msgs if "delivering " in m), default=0.0)
        print(f"RESULT rate={rate:5.1f} status={r.status:12s} steps={r.steps:>7,} "
              f"filled={r.filled_fraction:.3f} poured={r.delivered_lb:6.2f} lb "
              f"peak_delivery={peak:5.2f} lb/s wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    import threading
    threading.stack_size(512 * 1024 * 1024)
    t = threading.Thread(target=main); t.start(); t.join()
