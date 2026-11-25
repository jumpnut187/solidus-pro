import numpy as np, collections, sys
p = "Test/session_diag_2026-09-01/r3_t2_ledger.csv.types.npy"
t = np.load(p)
print("types shape", t.shape, "dtype", t.dtype)
vals, cnt = np.unique(t, return_counts=True)
NAME = {0:"CAST",1:"RISER",2:"CHILL",4:"CORE",6:"SPRUE",7:"RUNNER",8:"FILTER",9:"GATE",-1:"MOLD/outside"}
print("value histogram:")
for v,c in zip(vals,cnt):
    print(f"  {v:4d} {NAME.get(int(v),'?'):14s} {c}")
nx,ny,nz = t.shape
metal = t >= 0
print("total metal cells:", int(metal.sum()))
faces = [("x=0", t[0,:,:]), (f"x={nx-1}", t[nx-1,:,:]),
         ("y=0", t[:,0,:]), (f"y={ny-1}", t[:,ny-1,:]),
         ("z=0", t[:,:,0]), (f"z={nz-1}", t[:,:,nz-1])]
print("\n--- FACE AUDIT (metal/part cells lying ON each array face) ---")
for name, sl in faces:
    m = sl >= 0
    n = int(m.sum())
    if n == 0:
        print(f"{name}: 0 part cells (all mold/outside)")
        continue
    c = collections.Counter(NAME.get(int(v),str(v)) for v in sl[m].ravel())
    print(f"{name}: {n} part cells -> {dict(c)}")
top = t[:,:,nz-1]
if (top>=0).any():
    xs,ys = np.nonzero(top>=0)
    print(f"  top plane z={nz-1} part cells bbox x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]")
    for v in np.unique(top[top>=0]):
        mm = top==v
        xs,ys = np.nonzero(mm)
        print(f"    {NAME.get(int(v),v)}: {int(mm.sum())} cells x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]")
print("\n--- per-part z extent ---")
for v in vals:
    if v < 0: continue
    m = t==v
    zs = np.nonzero(m.any(axis=(0,1)))[0]
    xs = np.nonzero(m.any(axis=(1,2)))[0]
    ys = np.nonzero(m.any(axis=(0,2)))[0]
    print(f"  {NAME.get(int(v),v):8s} z[{zs.min()},{zs.max()}] x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}] n={int(m.sum())}")
