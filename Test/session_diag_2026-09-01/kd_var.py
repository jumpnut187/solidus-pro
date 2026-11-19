import sys, os, numpy as np, faulthandler, ctypes
faulthandler.enable()
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
libc = ctypes.CDLL(None)
libc.pthread_self.restype = ctypes.c_void_p
libc.pthread_get_stackaddr_np.restype = ctypes.c_void_p
libc.pthread_get_stackaddr_np.argtypes = [ctypes.c_void_p]
if os.environ.get("PRE"):
    __import__(os.environ["PRE"])
from core.part_manager import PartManager
from scipy.spatial import cKDTree
mode = sys.argv[1]
if mode.startswith("syn"):
    rng = np.random.default_rng(0)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    a = np.repeat(rng.random((n, 3)) * 3.0, 2, axis=0)
else:
    pm = PartManager()
    ids = pm.load_assembly_parts("/Users/brendonjarett/Downloads/Test 2/Part Studio 1 - Sprue & Well.step")
    m = pm.get_part(ids[0]).get_transformed_mesh()
    a = np.ascontiguousarray(np.asarray(m.vertices[m.referenced_vertices], dtype=np.float64))
    if mode == "uniq":
        a = np.unique(a, axis=0)
print("n", len(a), "uniq", len(np.unique(a, axis=0)), flush=True)
t = cKDTree(a)
print("BUILT", t.n, flush=True)
