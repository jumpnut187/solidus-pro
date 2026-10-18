"""A walled tower with a ceramic slab across it: does the metal cross, and
does any porous cell hold a negative population while it does?

Sizing probe for the regression test that goes into
tests/test_filling_stability.py.
"""
import os, sys
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np

from tests.test_filling_stability import _walled_tower, _run, DX
from core.part_manager import PART_TYPE_FILTER, PART_TYPE_CAST


def filtered_tower(nz=14, width=3, band=(5, 7)):
    mesh = _walled_tower(nz=nz, width=width)
    idx = np.rint(np.asarray(mesh.voxel_centers) / DX).astype(int)
    inband = (
        (mesh.voxel_part_types == PART_TYPE_CAST)
        & (idx[:, 2] >= band[0]) & (idx[:, 2] <= band[1])
    )
    print("filter cells:", int(inband.sum()))
    mesh.voxel_part_types[inband] = PART_TYPE_FILTER
    return mesh


def main():
    import physics.filling as pf
    from physics.filling import FLAG_GAS, FLAG_OBSTACLE
    orig = pf.step
    samples = []

    def traced(state, units, **kw):
        out = orig(state, units, **kw)
        if state.step_count % 200 == 0:
            f = np.asarray(state.f)
            por = np.asarray(state.porous.porosity) < 1.0
            wet = (state.flags != FLAG_GAS) & (state.flags != FLAG_OBSTACLE)
            neg = np.any(f < 0.0, axis=0) & wet
            samples.append((state.step_count, int((neg & por).sum()),
                            int((neg & ~por).sum()),
                            int((por & wet).sum()),
                            float(np.abs(np.asarray(state.u)).max())))
        return out

    pf.step = traced
    import core.filling_stage as cfs
    cfs.step = traced
    try:
        msgs = []
        res = _run(mesh=filtered_tower(), max_fill_time_s=1.0, messages=msgs)
    finally:
        pf.step = orig
        cfs.step = orig
    for s in samples[::max(1, len(samples) // 25)]:
        print("  step %6d  neg_porous %4d  neg_other %4d  wet_porous %4d  |u|max %.4f" % s)
    print("status", res.status, "fill", res.filled_fraction, "steps", res.steps)
    print("warnings:", res.warnings)
    print("worst neg_porous:", max(s[1] for s in samples),
          "worst neg_other:", max(s[2] for s in samples),
          "max wet_porous:", max(s[3] for s in samples))
    for m in msgs[-6:]:
        print("  msg:", m)


if __name__ == "__main__":
    main()
