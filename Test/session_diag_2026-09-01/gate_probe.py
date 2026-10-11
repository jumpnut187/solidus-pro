"""Per-gate / per-filter resolution probe on the saved part-type grids."""
import sys
from pathlib import Path
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
import numpy as np
from scipy import ndimage
from core import filling_stage as fs
from core.part_manager import (
    PART_TYPE_SPRUE, PART_TYPE_RUNNER, PART_TYPE_GATE, PART_TYPE_FILTER,
    PART_TYPE_CAST,
)

SP = Path("/private/tmp/claude-501/-Users-brendonjarett-Desktop-Active-Codebase-Python-Cast-Sim/ae9fb16b-4a3f-4912-aa13-a696388671c4/scratchpad")
DX_IN = 0.17

for c in (1, 2):
    st = np.load(SP / f"t2_types_c{c}.npy")
    dx_m = DX_IN * 0.0254 * c
    print(f"\n===== coarsening {c}  dx = {dx_m*1000:.2f} mm  shape {st.shape}")
    print(f"measure_choke_thickness_cells = {fs.measure_choke_thickness_cells(st):.2f}")
    print(f"measure_choke_area_cm2        = {fs.measure_choke_area_cm2(st, dx_m):.2f} cm^2"
          f"  (_min_flow_slice_cells = {fs._min_flow_slice_cells(st)})")

    for name, code in (("GATE", PART_TYPE_GATE), ("FILTER", PART_TYPE_FILTER)):
        m = st == code
        lab, n = ndimage.label(m, structure=np.ones((3, 3, 3), bool))
        print(f"\n  {name}: {int(m.sum())} cells in {n} bodies")
        for i in range(1, n + 1):
            b = lab == i
            idx = np.nonzero(b)
            ext = [int(x.max() - x.min() + 1) for x in idx]
            # narrowest slab: min over axes of the min per-slice cell count
            per_axis = []
            for ax in range(3):
                counts = b.sum(axis=tuple(j for j in range(3) if j != ax))
                nzc = counts[counts > 0]
                per_axis.append((int(nzc.min()), int(len(nzc))))
            print(f"    body {i}: {int(b.sum())} cells, bbox extents {ext} cells "
                  f"({[round(e*dx_m*39.3701, 2) for e in ext]} in)")
            print(f"      slices along x/y/z: n_slices={[p[1] for p in per_axis]}, "
                  f"min cells per slice={[p[0] for p in per_axis]}")

    # connectivity questions
    def comp(mask, label):
        l, n = ndimage.label(mask, structure=np.ones((3, 3, 3), bool))
        sizes = sorted(np.bincount(l.ravel())[1:], reverse=True)[:5]
        print(f"  {label}: {n} components, sizes {sizes}")

    comp(np.isin(st, (PART_TYPE_RUNNER, PART_TYPE_GATE, PART_TYPE_CAST)),
         "runner|gate|cast")
    comp(np.isin(st, (PART_TYPE_RUNNER, PART_TYPE_FILTER)), "runner|filter")
    comp(np.isin(st, (PART_TYPE_SPRUE, PART_TYPE_RUNNER, PART_TYPE_FILTER,
                      PART_TYPE_GATE, PART_TYPE_CAST)), "whole flow + cast")

    # Does the gate have any INTERIOR after halfway bounce-back?
    # A cell is "flowable" if it is not adjacent to an obstacle along the
    # thin direction. Approximate: count gate cells all of whose 6 face
    # neighbours are non-obstacle.
    obst = fs._obstacle_mask(st)
    open_ = ~obst
    interior = open_.copy()
    for ax in range(3):
        for s in (1, -1):
            interior &= np.roll(open_, s, axis=ax)
    g = st == PART_TYPE_GATE
    print(f"  gate cells with all 6 face-neighbours open: "
          f"{int((g & interior).sum())} / {int(g.sum())}")
    f_ = st == PART_TYPE_FILTER
    print(f"  filter cells with all 6 face-neighbours open: "
          f"{int((f_ & interior).sum())} / {int(f_.sum())}")
