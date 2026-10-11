import sys, numpy as np, faulthandler
faulthandler.enable()
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
import os
if os.environ.get("PRE"):
    for mod in os.environ["PRE"].split(","):
        __import__(mod)
        print("imported", mod, flush=True)
from core.part_manager import PartManager
pm = PartManager()
ids = pm.load_assembly_parts("/Users/brendonjarett/Downloads/Test 2/Part Studio 1 - Sprue & Well.step")
m = pm.get_part(ids[0]).get_transformed_mesh()
lo, hi = m.bounds
pts = np.random.default_rng(0).uniform(lo, hi, size=(100, 3))
steps = sys.argv[1].split(",")
from scipy.spatial import cKDTree
for s in steps:
    if s == "contains":
        print("contains ->", int(m.contains(pts).sum()), flush=True)
    elif s == "rtree":
        t = m.triangles_tree; print("rtree ->", t, flush=True)
    elif s == "kd":
        v = m.vertices[m.referenced_vertices]
        print("kd input", np.asarray(v).shape, np.asarray(v).dtype, flush=True)
        k = cKDTree(v); print("kd ->", k.n, flush=True)
    elif s == "kdmid":
        v = m.vertices[m.referenced_vertices]
        k = cKDTree(v, balanced_tree=False); print("kdmid ->", k.n, flush=True)
    elif s == "kdcopy":
        v = np.array(m.vertices[m.referenced_vertices], dtype=np.float64)
        k = cKDTree(v); print("kdcopy ->", k.n, flush=True)
    elif s == "onsurf":
        r = m.nearest.on_surface(pts); print("onsurf ->", r[1][:2], flush=True)
    elif s == "nearby":
        from trimesh.proximity import nearby_faces
        r = nearby_faces(m, pts); print("nearby ->", len(r), flush=True)
    elif s == "triangles":
        print("triangles ->", m.triangles.shape, flush=True)
    elif s == "watertight":
        print("watertight", m.is_watertight, "winding", m.is_winding_consistent, flush=True)
print("ALL OK", flush=True)
