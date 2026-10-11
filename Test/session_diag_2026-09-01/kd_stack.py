import sys, os, threading, numpy as np, faulthandler
faulthandler.enable()
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
if os.environ.get("PRE"):
    __import__(os.environ["PRE"])
from core.part_manager import PartManager
from scipy.spatial import cKDTree
pm = PartManager()
ids = pm.load_assembly_parts("/Users/brendonjarett/Downloads/Test 2/Part Studio 1 - Sprue & Well.step")
m = pm.get_part(ids[0]).get_transformed_mesh()
a = np.ascontiguousarray(np.asarray(m.vertices[m.referenced_vertices], dtype=np.float64))
mb = int(sys.argv[1])
threading.stack_size(mb * 1024 * 1024)
def run():
    t = cKDTree(a)
    print(f"BUILT {mb}MB n={t.n}", flush=True)
th = threading.Thread(target=run); th.start(); th.join()
