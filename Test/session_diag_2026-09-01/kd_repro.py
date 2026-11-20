import sys, numpy as np, trimesh
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
from core.part_manager import PartManager

pm = PartManager()
f = "/Users/brendonjarett/Downloads/Test 2/Part Studio 1 - Sprue & Well.step"
ids = pm.load_assembly_parts(f, voxel_size_in=0.17)
part = pm.get_part(ids[0])
m = part.trimesh_obj
print("verts", len(m.vertices), "unique", len(np.unique(m.vertices, axis=0)),
      "faces", len(m.faces), flush=True)
mode = sys.argv[1]
pts = np.asarray(m.vertices[m.referenced_vertices], dtype=np.float64)
print("tree points", pts.shape, "unique", len(np.unique(pts, axis=0)), flush=True)
from scipy.spatial import cKDTree
if mode == "default":
    t = cKDTree(pts)
elif mode == "mid":
    t = cKDTree(pts, balanced_tree=False)
elif mode == "merged":
    m.merge_vertices()
    pts = np.asarray(m.vertices[m.referenced_vertices], dtype=np.float64)
    print("after merge", pts.shape, "unique", len(np.unique(pts, axis=0)), flush=True)
    t = cKDTree(pts)
print("built ok", t.n, flush=True)
q = np.random.default_rng(0).random((100, 3)) * 5
print("query ok", t.query(q)[0][:3], flush=True)
