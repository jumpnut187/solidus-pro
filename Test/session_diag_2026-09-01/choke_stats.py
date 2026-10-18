"""Candidate robust-thickness statistics on a saved part-type grid."""
import sys
import numpy as np
from scipy import ndimage
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
from core.part_manager import (PART_TYPE_SPRUE, PART_TYPE_RUNNER,
                               PART_TYPE_GATE, PART_TYPE_FILTER)
NAMES = {PART_TYPE_SPRUE: "sprue", PART_TYPE_RUNNER: "runner",
         PART_TYPE_GATE: "gate", PART_TYPE_FILTER: "filter"}

def stats(st, label, include_filter=True):
    st = np.asarray(st)
    codes = (PART_TYPE_SPRUE, PART_TYPE_RUNNER, PART_TYPE_GATE) + (
        (PART_TYPE_FILTER,) if include_filter else ())
    flow = np.isin(st, codes)
    if not flow.any():
        print(f"{label}: no flow"); return
    d = ndimage.distance_transform_edt(np.pad(flow, 1))[1:-1, 1:-1, 1:-1]
    print(f"--- {label}  (region={'incl filter' if include_filter else 'sprue/runner/gate'})")
    for code in codes:
        m = st == code
        if not m.any():
            continue
        v = d[m]
        def t(x): return 2.0 * x - 1.0
        print(f"   {NAMES[code]:7s} n={int(m.sum()):6d}  clearance max={v.max():.2f} "
              f"p90={np.percentile(v,90):.2f} med={np.median(v):.2f} "
              f"p25={np.percentile(v,25):.2f} min={v.min():.2f}  || "
              f"thick max={t(v.max()):.2f} med={t(np.median(v)):.2f} "
              f"p25={t(np.percentile(v,25)):.2f} p75={t(np.percentile(v,75)):.2f}")

if __name__ == "__main__":
    for path in sys.argv[1:]:
        st = np.load(path)
        stats(st, path.split("/")[-1], include_filter=True)
        stats(st, path.split("/")[-1], include_filter=False)
