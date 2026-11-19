import sys, numpy as np, threading
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
from core.part_manager import PartManager
from scipy.spatial import cKDTree
pm = PartManager()
import glob
def depth(tree):
    # iterative DFS over the Python node wrapper
    stack = [(tree.tree, 1)]
    best = 0
    n = 0
    while stack:
        node, d = stack.pop()
        n += 1
        best = max(best, d)
        for child in ("lesser", "greater"):
            c = getattr(node, child, None)
            if c is not None:
                stack.append((c, d + 1))
    return best, n
for f in sorted(glob.glob("/Users/brendonjarett/Downloads/Test 2/*.step")):
    ids = pm.load_assembly_parts(f)
    for pid in ids:
        p = pm.get_part(pid)
        m = p.get_transformed_mesh()
        a = np.ascontiguousarray(np.asarray(m.vertices[m.referenced_vertices], dtype=np.float64))
        u = np.unique(a, axis=0)
        res = {}
        for label, arr, kw in (("median", a, {}), ("midpoint", a, {"balanced_tree": False}),
                               ("median-uniq", u, {})):
            t = cKDTree(arr, **kw)
            res[label] = depth(t)
        print(f"{p.name[:34]:34s} n={len(a):6d} uniq={len(u):6d} "
              + "  ".join(f"{k}: depth={v[0]:6d} nodes={v[1]:6d}" for k, v in res.items()), flush=True)
