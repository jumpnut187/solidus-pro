import sys, numpy as np, collections
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
import os; os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
from physics.filling import _dilate
NAME={0:"CAST",1:"RISER",6:"SPRUE",7:"RUNNER",8:"FILTER",9:"GATE",-1:"mold"}
t = np.load("Test/session_diag_2026-09-01/r3_t2_ledger.csv.types.npy")
nx,ny,nz=t.shape
obstacle = t < 0
inlet = np.zeros(t.shape, bool)
inlet[:,:,nz-1] = (t[:,:,nz-1]==6)
print("inlet (sprue mouth at z=93):", int(inlet.sum()), "cells")
iface = _dilate(inlet) & ~inlet & ~obstacle
print("initial interface layer from that seed:", int(iface.sum()))
zs = np.nonzero(iface.any(axis=(0,1)))[0]
print("  its z extent:", zs.min(), "-", zs.max(), "  <-- gap of", zs.max()-zs.min()-1, "planes")
for z in zs:
    m = iface[:,:,z]
    c = collections.Counter(NAME.get(int(v),str(v)) for v in t[:,:,z][m])
    xs,ys = np.nonzero(m)
    print(f"  z={z}: {int(m.sum())} cells {dict(c)} x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]")
print()
print("--- and the inlet halo the stage pins (core/filling_stage.py:2497-2503) ---")
halo = _dilate(inlet) & ~inlet & ~obstacle
hz = np.nonzero(halo.any(axis=(0,1)))[0]
print("  halo z planes:", hz.tolist())
print()
print("--- how far the riser seed is from the front when it fires ---")
cast_bot = (t[:,:,0]==0); riser_top=(t[:,:,nz-1]==1)
ov = cast_bot & riser_top
print(f"  {int(ov.sum())} (x,y) columns pair CAST@z=0 with RISER@z=93.")
print( "  A CAST cell at z=0 turning FLUID promotes the RISER cell at z=93 in")
print( "  its own or a diagonally adjacent column -- 93 empty planes away.")
bot = np.zeros(t.shape,bool); bot[:,:,0]=cast_bot
reach = _dilate(bot) & (t==1) & (np.arange(nz)[None,None,:]>=60)
print(f"  riser cells (z>=60) directly promotable by the z=0 wrap: {int(reach.sum())}")
zz=np.nonzero(reach.any(axis=(0,1)))[0]; print("  at z planes:", zz.tolist())
