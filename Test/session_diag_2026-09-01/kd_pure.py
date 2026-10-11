import sys, numpy as np, faulthandler
faulthandler.enable()
from scipy.spatial import cKDTree
a = np.load(sys.argv[1])
if len(sys.argv) > 2 and sys.argv[2] == "uniq":
    a = np.unique(a, axis=0)
print("n", len(a), flush=True)
t = cKDTree(a)
print("BUILT", t.n, flush=True)
