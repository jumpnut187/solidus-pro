import sys, os, numpy as np, hashlib
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
if os.environ.get("PRE"):
    __import__(os.environ["PRE"])
from core.part_manager import PartManager
pm = PartManager()
ids = pm.load_assembly_parts("/Users/brendonjarett/Downloads/Test 2/Part Studio 1 - Sprue & Well.step")
m = pm.get_part(ids[0]).get_transformed_mesh()
v = m.vertices[m.referenced_vertices]
a = np.asarray(v, dtype=np.float64)
print("shape", a.shape, "type", type(v).__name__, "finite", bool(np.isfinite(a).all()),
      "md5", hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest(),
      "contig", np.ascontiguousarray(a) is a, a.flags['C_CONTIGUOUS'], flush=True)
print("min", a.min(0), "max", a.max(0), flush=True)
