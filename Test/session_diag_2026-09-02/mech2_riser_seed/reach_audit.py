"""Are the 298 CAST@z=0 wrap columns actually reachable by the metal?"""
import sys, numpy as np
sys.path.insert(0,"/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
import os; os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
from scipy import ndimage
from core.filling_stage import _obstacle_mask
t=np.load("Test/session_diag_2026-09-01/r3_t2_ledger.csv.types.npy")
nx,ny,nz=t.shape
obst=_obstacle_mask(t); openc=~obst
lab,n=ndimage.label(openc, ndimage.generate_binary_structure(3,1))
print("open (non-obstacle) components:", n)
inlet = (t[:,:,nz-1]==6)
ilabs = np.unique(lab[:,:,nz-1][inlet]); ilabs=ilabs[ilabs>0]
print("inlet component label(s):", ilabs.tolist(), "size", int(np.isin(lab,ilabs).sum()))
main = np.isin(lab, ilabs)
bot = np.zeros(t.shape,bool); bot[:,:,0]=(t[:,:,0]==0)
riser_top = np.zeros(t.shape,bool); riser_top[:,:,nz-1]=(t[:,:,nz-1]==1)
# the wrap columns
ov = (t[:,:,0]==0)&(t[:,:,nz-1]==1)
wrapcol = np.zeros(t.shape,bool); wrapcol[:,:,0]=ov
print(f"\n298-column CAST cells at z=0: {int(wrapcol.sum())}, "
      f"of which in the inlet's component (i.e. metal CAN reach them): "
      f"{int((wrapcol&main).sum())}")
print(f"all CAST cells at z=0: {int(bot.sum())}, reachable: {int((bot&main).sum())}")
print(f"riser cells at z=93: {int(riser_top.sum())}, in main component: {int((riser_top&main).sum())}")
# how deep below the gates is z=0
print("\ngate z range 32-35; the cast's own z extent 0-93.")
for z in range(0,10):
    print(f"   cast cells at z={z}: {int((t[:,:,z]==0).sum())}  reachable {int((main[:,:,z] & (t[:,:,z]==0)).sum())}")
