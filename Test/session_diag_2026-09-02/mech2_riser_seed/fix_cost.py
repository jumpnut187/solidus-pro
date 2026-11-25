import sys, numpy as np, collections
sys.path.insert(0,"/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
import os; os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
from core.filling_stage import _obstacle_mask, _metal_mask, _domain_boundary_mask, _static_vent_mask, EMPTY_CELL
NAME={0:"CAST",1:"RISER",6:"SPRUE",7:"RUNNER",8:"FILTER",9:"GATE"}
t=np.load("Test/session_diag_2026-09-01/r3_t2_ledger.csv.types.npy")
metal=_metal_mask(t); obst=_obstacle_mask(t)
print("metal cells:", int(metal.sum()), " obstacle:", int(obst.sum()))
b=_domain_boundary_mask(t.shape)
lost = metal & b
print(f"\nSTOPGAP  obstacle |= _domain_boundary_mask : sacrifices {int(lost.sum())} metal cells "
      f"= {100*lost.sum()/metal.sum():.2f}% of the metal")
print("   by part:", {NAME.get(int(k),k):v for k,v in collections.Counter(t[lost].tolist()).items()})
print("   riser cells lost:", int((lost&(t==1)).sum()), "of", int((t==1).sum()))

print("\nPAD  np.pad(solve_types, 1, constant_values=EMPTY_CELL):")
tp = np.pad(t, 1, mode="constant", constant_values=EMPTY_CELL)
mp=_metal_mask(tp); op=_obstacle_mask(tp)
print("   shape", t.shape, "->", tp.shape, " cells", t.size, "->", tp.size,
      f"(+{100*(tp.size-t.size)/t.size:.1f}%)")
print("   metal cells preserved:", int(mp.sum()), "==", int(metal.sum()), mp.sum()==metal.sum())
for nmf, sl in [("x=0",op[0]),("x=-1",op[-1]),("y=0",op[:,0]),("y=-1",op[:,-1]),
                ("z=0",op[:,:,0]),("z=-1",op[:,:,-1])]:
    print(f"   face {nmf:5s} all obstacle: {bool(sl.all())}")
inlet = np.array(np.nonzero((t[:,:,t.shape[2]-1]==6))).T
inlet = np.column_stack((inlet, np.full(len(inlet), t.shape[2]-1)))
v0=_static_vent_mask(t, inlet, True)
inlet_p = inlet + 1
vp=_static_vent_mask(tp, inlet_p, True)
print(f"\n   _static_vent_mask riser-top vents: unpadded {int((v0&(t==1)).sum())}, "
      f"padded {int((vp&(tp==1)).sum())}  -> preserved: "
      f"{int((v0&(t==1)).sum())==int((vp&(tp==1)).sum())}")
bp=_domain_boundary_mask(tp.shape)
print(f"   BUT _domain_boundary_mask on the padded grid selects only obstacle: "
      f"{bool(op[bp].all())}  -> the 'boundary GAS = atmosphere' vent term becomes EMPTY "
      f"({int((bp & ~op).sum())} non-obstacle cells on it). This needs re-pointing.")
