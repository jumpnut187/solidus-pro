import sys, time
import numpy as np
from scipy import ndimage
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
from core.part_manager import (PART_TYPE_SPRUE, PART_TYPE_RUNNER,
                               PART_TYPE_GATE, PART_TYPE_FILTER)
NAMES = {PART_TYPE_SPRUE: "sprue", PART_TYPE_RUNNER: "runner",
         PART_TYPE_GATE: "gate", PART_TYPE_FILTER: "filter"}
CODES = (PART_TYPE_SPRUE, PART_TYPE_RUNNER, PART_TYPE_FILTER, PART_TYPE_GATE)

def local_thickness(region):
    region = np.asarray(region, dtype=bool)
    if not region.any():
        return np.zeros(region.shape, dtype=np.float64)
    idx = np.nonzero(region)
    sl = tuple(slice(max(i.min() - 1, 0), i.max() + 2) for i in idx)
    sub = region[sl]
    padded = np.pad(sub, 1)
    d = ndimage.distance_transform_edt(padded)
    best = np.zeros_like(d)
    r_max = float(d.max())
    r = np.floor(r_max * 2.0) / 2.0
    while r >= 0.5:
        seeds = d >= r
        if seeds.any():
            reach = ndimage.distance_transform_edt(~seeds)
            np.maximum(best, np.where(reach <= r, r, 0.0), out=best)
        r -= 0.5
    best = best[1:-1, 1:-1, 1:-1]
    out = np.zeros(region.shape, dtype=np.float64)
    out[sl] = np.where(sub, 2.0 * best - 1.0, 0.0)
    return out

def report(path, label):
    st = np.load(path)
    region = np.isin(st, CODES)
    t0 = time.time()
    lt = local_thickness(region)
    dt = time.time() - t0
    print(f"--- {label}  shape={st.shape}  lt in {dt:.2f}s")
    for code in CODES:
        m = st == code
        if not m.any():
            continue
        v = lt[m]
        print(f"    {NAMES[code]:7s} n={int(m.sum()):6d} "
              f"min={v.min():5.2f} p10={np.percentile(v,10):5.2f} "
              f"p25={np.percentile(v,25):5.2f} med={np.median(v):5.2f} "
              f"p75={np.percentile(v,75):5.2f} max={v.max():5.2f}")

if __name__ == "__main__":
    for a in sys.argv[1:]:
        report(a, a.split("/")[-1])
