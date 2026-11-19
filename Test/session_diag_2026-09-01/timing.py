import sys, time, numpy as np
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
import trimesh
from core.voxel_mesh import _robust_contains_check_impl
def soup(sub):
    box = trimesh.creation.box(extents=(2.0,1.0,3.0))
    for _ in range(sub): box = box.subdivide()
    ex = np.asarray(box.vertices)[box.faces].reshape(-1,3)
    return trimesh.Trimesh(vertices=ex, faces=np.arange(len(ex)).reshape(-1,3), process=False)
for sub in (1,2,3):
    m = soup(sub)
    lo,hi = m.bounds
    pts = np.random.default_rng(17).uniform(lo-0.2, hi+0.2, size=(50,3))
    t0=time.time(); c = m.contains(pts); t1=time.time()
    print(f"sub={sub} faces={len(m.faces)} verts={len(m.vertices)} contains={t1-t0:.2f}s inside={c.sum()}", flush=True)
    t0=time.time(); r=_robust_contains_check_impl(m, pts, 0.05); t1=time.time()
    print(f"   robust={t1-t0:.2f}s inside={r.sum()}", flush=True)
