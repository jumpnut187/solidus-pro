import sys, numpy as np, faulthandler
faulthandler.enable()
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
from core.part_manager import PartManager
from core.voxel_mesh import _robust_contains_check_impl
pm = PartManager()
ids = pm.load_assembly_parts("/Users/brendonjarett/Downloads/Test 2/Part Studio 1 - Sprue & Well.step")
part = pm.get_part(ids[0])
m = part.get_transformed_mesh()
print("faces", len(m.faces), "verts", len(m.vertices), flush=True)
lo, hi = m.bounds
rng = np.random.default_rng(0)
pts = rng.uniform(lo - 0.1, hi + 0.1, size=(int(sys.argv[1]) if len(sys.argv) > 1 else 11720, 3))
out = _robust_contains_check_impl(m, pts, 0.085)
print("OK inside", int(out.sum()), flush=True)
