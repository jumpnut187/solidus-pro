import sys
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
import numpy as np
from core import filling_stage as fs
SP = "/private/tmp/claude-501/-Users-brendonjarett-Desktop-Active-Codebase-Python-Cast-Sim/ae9fb16b-4a3f-4912-aa13-a696388671c4/scratchpad"
IN = 0.0254
rows = [("Test 2, dx 4.32 mm (coarsening 1)", "t2_types_c1.npy", 0.17 * IN),
        ("Test 2, dx 8.64 mm (coarsening 2, solved)", "t2_types_c2.npy", 0.34 * IN),
        ("Full_Gating_Test, dx 0.20 in (defaults)", "fgt_types_c1.npy", 0.20 * IN),
        ("Filling_Test, dx 0.20 in (documented)", "ft_types_c1.npy", 0.20 * IN)]
for label, f, dx_m in rows:
    st = np.load(f"{SP}/{f}")
    by = fs.measure_choke_thickness_by_part(st)
    code = fs.narrowest_choke_part_code(by)
    print(f"\n=== {label}")
    print("    old scalar (per-part max) would be: see before-table")
    print("    per part: " + ", ".join(
        f"{fs._METAL_PATH_LABELS[c]} {by[c]:.1f}" for c in fs._METAL_PATH_CODES if c in by))
    print(f"    narrowest: {fs._METAL_PATH_LABELS[code]}   scalar {min(by.values()):.2f}")
    print(f"    area solved {fs.measure_choke_area_cm2(st, dx_m):.2f} cm2")
    for c in fs._METAL_PATH_CODES:
        t = by.get(c)
        if t is None:
            continue
        if t <= fs.MIN_CHOKE_THICKNESS_CELLS:
            print(f"    HARD  ({fs._METAL_PATH_LABELS[c]}, {t:.1f})")
        elif t <= fs.MARGINAL_CHOKE_THICKNESS_CELLS:
            print(f"    NOTE  ({fs._METAL_PATH_LABELS[c]}, {t:.1f})")
    if all(t > fs.MARGINAL_CHOKE_THICKNESS_CELLS for t in by.values()):
        print("    SILENT")
