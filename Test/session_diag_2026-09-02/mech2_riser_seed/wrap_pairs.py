import numpy as np, collections
t = np.load("Test/session_diag_2026-09-01/r3_t2_ledger.csv.types.npy")
NAME = {0:"CAST",1:"RISER",2:"CHILL",4:"CORE",6:"SPRUE",7:"RUNNER",8:"FILTER",9:"GATE",-1:"MOLD"}
nx,ny,nz = t.shape
def nm(a): return NAME.get(int(a),str(a))
pairs = [("z", t[:,:,0], t[:,:,nz-1], "z=0", f"z={nz-1}"),
         ("x", t[0,:,:], t[nx-1,:,:], "x=0", f"x={nx-1}"),
         ("y", t[:,0,:], t[:,ny-1,:], "y=0", f"y={ny-1}")]
for ax, lo, hi, ln, hn in pairs:
    both = (lo>=0) & (hi>=0)
    print(f"\n=== axis {ax}: cells that are PART on BOTH {ln} and {hn} (direct wrap metal->metal): {int(both.sum())}")
    if both.sum():
        c = collections.Counter((nm(lo[i,j]), nm(hi[i,j])) for i,j in zip(*np.nonzero(both)))
        for k,v in sorted(c.items(), key=lambda kv:-kv[1]):
            print(f"    {ln}:{k[0]:7s} <-> {hn}:{k[1]:7s}   {v} cells")
        ii,jj = np.nonzero(both)
        print(f"    coords bbox: [{ii.min()},{ii.max()}] x [{jj.min()},{jj.max()}]")
    only_hi = (hi>=0) & (lo<0); only_lo=(lo>=0)&(hi<0)
    print(f"    part only on {hn}: {int(only_hi.sum())}, part only on {ln}: {int(only_lo.sum())}")

riser_top = t[:,:,nz-1]==1
cast_bot  = t[:,:,0]==0
ov = riser_top & cast_bot
print(f"\n*** RISER cells on z=93 whose (x,y) column has CAST on z=0: {int(ov.sum())}")
if ov.sum():
    ii,jj=np.nonzero(ov); print("    bbox x[%d,%d] y[%d,%d]"%(ii.min(),ii.max(),jj.min(),jj.max()))
    print("    sample:", list(zip(ii[:10].tolist(), jj[:10].tolist())))
print("\n*** y=0 <-> y=134 detail")
lo=t[:,0,:]; hi=t[:,ny-1,:]
b=(lo>=0)&(hi>=0)
print("   both:", int(b.sum()))
for i,k in zip(*np.nonzero(b)):
    pass
ii,kk=np.nonzero(b)
if len(ii): print("   x range",ii.min(),ii.max()," z range",kk.min(),kk.max())
