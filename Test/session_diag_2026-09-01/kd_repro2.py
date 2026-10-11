import sys, numpy as np, faulthandler
faulthandler.enable()
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
from core.part_manager import PartManager
pm = PartManager()
ids = pm.load_assembly_parts("/Users/brendonjarett/Downloads/Test 2/Part Studio 1 - Sprue & Well.step")
part = pm.get_part(ids[0])
m = part.get_transformed_mesh() if hasattr(part, "get_transformed_mesh") else part.trimesh_obj
v = np.asarray(m.vertices[m.referenced_vertices], dtype=np.float64)
print("faces", len(m.faces), "verts", len(m.vertices), "tree pts", v.shape,
      "unique", len(np.unique(v, axis=0)), flush=True)
mode = sys.argv[1]
from scipy.spatial import cKDTree
if mode == "merged":
    m.merge_vertices(merge_norm=True, merge_tex=True)
    v = np.asarray(m.vertices[m.referenced_vertices], dtype=np.float64)
    print("after merge", v.shape, "unique", len(np.unique(v, axis=0)), flush=True)
    t = cKDTree(v)
elif mode == "mid":
    t = cKDTree(v, balanced_tree=False)
else:
    t = cKDTree(v)
print("BUILT", t.n, flush=True)
