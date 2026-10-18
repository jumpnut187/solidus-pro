"""Geometry-only probe: coarse part-type grid, choke thickness, filter layout."""
import os, sys, time
from pathlib import Path
REPO = Path("/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
sys.path.insert(0, str(REPO))
import numpy as np

# trimesh's proximity.nearby_faces builds cKDTree with the default (median)
# split; the STEP tessellations here have ~50% duplicate vertices and that
# recurses until the C stack dies. Force midpoint splitting, as the repo's
# own core.voxel_mesh._build_kdtree already does for its own trees.
import scipy.spatial as _ss
_orig_kd = _ss.cKDTree
class _KD(_orig_kd):
    def __init__(self, data, *a, **kw):
        kw.setdefault("balanced_tree", False)
        super().__init__(data, *a, **kw)
_ss.cKDTree = _KD
_ss.KDTree = _KD
import scipy.spatial._ckdtree as _ck
_ck.cKDTree = _KD

d = sys.argv[1]
voxel = float(sys.argv[2])
refine = int(sys.argv[3])
out = sys.argv[4]

from core.part_manager import (
    PartManager, PartType, PART_TYPE_TO_CODE, PART_TYPE_FILTER, PART_TYPE_GATE,
    PART_TYPE_RUNNER, PART_TYPE_SPRUE,
)
from core import simulation_pipeline as sp
from core import filling_stage as fs

pm = PartManager()
for f in sorted(p for p in Path(d).iterdir() if p.suffix.lower() in (".stl", ".step", ".stp")):
    pm.load_assembly_parts(str(f))

inputs = sp.PipelineInputs(alloy_name="WCB", voxel_size=voxel, refinement_level=refine)
_m, _b, _v, _p, vm, _s = sp.prepare_mesh_and_geometry(inputs, part_manager=pm, stl_loader=None)

centers = np.asarray(vm.voxel_centers, dtype=np.float64)
dx_in = float(np.asarray(vm.voxel_sizes).ravel()[0])
mapping = fs.build_grid_mapping(centers, dx_in)
fine = fs.dense_part_types(mapping, np.asarray(vm.voxel_part_types))
for c in (1, 2):
    st = fs.coarsen_part_types(fine, c)
    np.save(f"{out}_types_c{c}.npy", st)
    import collections
    hist = collections.Counter(st.ravel().tolist())
    print(f"--- coarsening {c}: shape {st.shape} dx={dx_in*c*25.4:.2f} mm")
    for code, n in sorted(hist.items()):
        print(f"    code {code}: {n}")
    print(f"    choke thickness (sprue/runner/gate) = "
          f"{fs.measure_choke_thickness_cells(st):.2f} cells")
    # per-part thickness via distance transform
    from scipy import ndimage
    for name, code in (("sprue", PART_TYPE_SPRUE), ("runner", PART_TYPE_RUNNER),
                       ("gate", PART_TYPE_GATE), ("filter", PART_TYPE_FILTER)):
        m = st == code
        if not m.any():
            print(f"    {name}: 0 cells")
            continue
        dt = ndimage.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1, 1:-1]
        print(f"    {name}: {int(m.sum())} cells, max inscribed radius "
              f"{dt.max():.2f} -> thickness {2*dt.max()-1:.2f} cells, "
              f"bbox {[int(x.max()-x.min()+1) for x in np.nonzero(m)]}")
    # connectivity of the flow path incl. filter
    flowmask = np.isin(st, (PART_TYPE_SPRUE, PART_TYPE_RUNNER, PART_TYPE_GATE, PART_TYPE_FILTER))
    lab, n = ndimage.label(flowmask, structure=np.ones((3,3,3), bool))
    print(f"    flow path (sprue|runner|gate|filter) components: {n}, "
          f"sizes {sorted(np.bincount(lab.ravel())[1:], reverse=True)[:6]}")
    # is the metal domain (obstacle-free) one piece?
    obst = fs._obstacle_mask(st)
    lab2, n2 = ndimage.label(~obst, structure=np.ones((3,3,3), bool))
    print(f"    open (non-obstacle) components: {n2}, "
          f"sizes {sorted(np.bincount(lab2.ravel())[1:], reverse=True)[:6]}")
print("done")


