from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path("/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
sys.path.insert(0, str(REPO))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import faulthandler
import traceback

import numpy as np

faulthandler.enable()

def _trace_kdtree():
    import scipy.spatial as ss
    orig = ss.cKDTree

    class TracedKDTree(orig):
        def __init__(self, data, *a, **kw):
            arr = np.asarray(data, dtype=np.float64)
            nonfinite = int(np.count_nonzero(~np.isfinite(arr)))
            dup = ""
            if arr.ndim == 2 and arr.shape[0] and arr.shape[0] < 2_000_000:
                dup = f" unique={len(np.unique(arr, axis=0))}"
            site = "".join(traceback.format_stack()[-3:-1]).strip().replace("\n", " | ")
            log(
                f"cKDTree: n={arr.shape} nonfinite={nonfinite}{dup} "
                f"kw={kw} <- {site[:400]}"
            )
            kw.setdefault("balanced_tree", False)
            super().__init__(arr, *a, **kw)

    ss.cKDTree = TracedKDTree
    ss.KDTree = TracedKDTree
    import scipy.spatial._ckdtree as ck
    ck.cKDTree = TracedKDTree

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--voxel", type=float, default=0.17)
    ap.add_argument("--refine", type=int, default=4)
    ap.add_argument("--coarsening", type=int, default=None)
    ap.add_argument("--max-fill-time", type=float, default=None)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--ledger-every", type=int, default=250)
    ap.add_argument("--max-steps-override", type=int, default=None)
    args = ap.parse_args()

    _trace_kdtree()
    from core.part_manager import PartManager
    from core.pour_location import PourLocation
    from core.materials import get_alloy
    from core import simulation_pipeline as sp
    from core import filling_stage as fs

    pm = PartManager()
    files = sorted(
        p for p in Path(args.dir).iterdir()
        if p.suffix.lower() in (".stl", ".step", ".stp")
    )
    log(f"loading {len(files)} files from {args.dir}")
    for f in files:
        try:
            ids = pm.load_assembly_parts(str(f))
        except Exception as exc:
            log(f"  FAILED {f.name}: {exc}")
            continue
        for pid in ids:
            part = pm.get_part(pid)
            log(f"  {part.name:35s} -> {part.part_type.value}")

    parts = pm.get_all_parts()
    log(f"{len(parts)} parts loaded")

    from core.part_manager import PartType
    sprue = next((p for p in parts if p.part_type is PartType.SPRUE), None)
    if sprue is None:
        raise SystemExit("no sprue part found")
    b = sprue.get_bounds()
    top = (
        float((b[0][0] + b[0][1]) / 2.0),
        float((b[1][0] + b[1][1]) / 2.0),
        float(b[2][1]),
    )
    log(f"pour point (top of '{sprue.name}'): {top}")
    loc = PourLocation(point_in=top, normal=(0.0, 0.0, 1.0), part_id=sprue.id)

    inputs = sp.PipelineInputs(
        alloy_name="WCB",
        voxel_size=args.voxel,
        refinement_level=args.refine,
        analysis_options={"filling": True},
        pour_location=loc.to_dict(),
    )

    t0 = time.time()
    (_mesh, _bounds, _vol, _parts, voxel_mesh, _mem) = sp.prepare_mesh_and_geometry(
        inputs, part_manager=pm, stl_loader=None,
        progress_message_callback=lambda m: log(f"mesh: {m}"),
    )
    log(f"voxel mesh: {len(voxel_mesh.voxel_centers):,} voxels in {time.time()-t0:.1f}s")

    alloy = get_alloy(inputs.alloy_name)
    options = sp.create_analysis_options(
        inputs.analysis_options, use_gpu=False, use_incremental=False
    )
    options.filling = True
    fp = options.solidification_params.filling
    log(
        f"filling params: coarsening={fp.fill_grid_coarsening} "
        f"head_in={fp.initial_head_in} max_fill_time_s={fp.max_fill_time_s} "
        f"inlet_mode={fp.inlet_mode_code} "
        f"filter.porosity={fp.filter.porosity}"
    )
    if args.coarsening is not None:
        fp.fill_grid_coarsening = args.coarsening
    if args.max_fill_time is not None:
        fp.max_fill_time_s = args.max_fill_time

    if args.ledger:
        LEDGER_STATE["types_out"] = args.ledger + ".types.npy"
        install_ledger(fs, args.ledger, args.ledger_every)
    capture_types(fs)

    t0 = time.time()
    result = fs.maybe_run_filling(
        voxel_mesh, alloy, options, inputs,
        progress_message_callback=lambda m: log(f"FILL: {m}"),
    )
    log(f"filling finished in {time.time()-t0:.1f}s")
    if result is not None:
        log(f"status={getattr(result, 'status', '?')} steps={getattr(result, 'steps', '?')}")
        for w in getattr(result, "warnings", []) or []:
            log(f"  warn: {w}")

def capture_types(fs):
    from core.part_manager import PART_TYPE_TO_CODE
    orig = fs.build_porous_zone

    def wrapped(dense_types, specs, **kw):
        try:
            st = np.asarray(dense_types)
            LEDGER_STATE["types"] = st
            LEDGER_STATE["type_codes"] = {
                pt.value: code for pt, code in PART_TYPE_TO_CODE.items()
            }
            import collections
            c = collections.Counter(st.ravel().tolist())
            log("coarse type histogram: " + ", ".join(
                f"{code}:{n}" for code, n in sorted(c.items())))
            if LEDGER_STATE.get("types_out"):
                np.save(LEDGER_STATE["types_out"], st)
            geometry_report(st, fs)
        except Exception:
            import traceback
            log("capture_types FAILED:\n" + traceback.format_exc())
        return orig(dense_types, specs, **kw)

    fs.build_porous_zone = wrapped
    log("capture_types installed on build_porous_zone")

def geometry_report(st, fs):
    from scipy import ndimage
    from core.part_manager import (
        PART_TYPE_SPRUE, PART_TYPE_RUNNER, PART_TYPE_GATE, PART_TYPE_FILTER,
    )
    log(f"solve grid shape {st.shape}")
    log(f"choke thickness (sprue/runner/gate) = "
        f"{fs.measure_choke_thickness_cells(st):.2f} cells")
    for name, code in (("sprue", PART_TYPE_SPRUE), ("runner", PART_TYPE_RUNNER),
                       ("gate", PART_TYPE_GATE), ("filter", PART_TYPE_FILTER)):
        m = st == code
        if not m.any():
            log(f"  {name}: 0 cells")
            continue
        dt = ndimage.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1, 1:-1]
        bb = [int(x.max() - x.min() + 1) for x in np.nonzero(m)]
        log(f"  {name}: {int(m.sum())} cells, thickness {2*dt.max()-1:.2f} "
            f"cells, bbox {bb}")
    flow = np.isin(st, (PART_TYPE_SPRUE, PART_TYPE_RUNNER, PART_TYPE_GATE,
                        PART_TYPE_FILTER))
    lab, n = ndimage.label(flow, structure=np.ones((3, 3, 3), bool))
    log(f"  flow path components: {n} sizes "
        f"{sorted(np.bincount(lab.ravel())[1:], reverse=True)[:6]}")
    obst = fs._obstacle_mask(st)
    lab2, n2 = ndimage.label(~obst, structure=np.ones((3, 3, 3), bool))
    log(f"  open (non-obstacle) components: {n2} sizes "
        f"{sorted(np.bincount(lab2.ravel())[1:], reverse=True)[:6]}")

LEDGER_STATE = {}

def install_ledger(fs, path, every):
    import physics.filling as pf
    from physics.filling import (
        FLAG_FLUID, FLAG_INTERFACE, FLAG_GAS, FLAG_OBSTACLE, fluid_fraction,
    )

    fh = open(path, "w", buffering=1)
    fh.write(
        "step,wet_cells,fluid_cells,iface_cells,wet_mass,ceiling,"
        "max_abs_u,rho_min,rho_max,eps_min,eps_max,neg_f,neg_cells,"
        "neg_porous_cells,min_temp,nonfinite,mass_by_type\n"
    )
    orig_step = pf.step
    ctr = {"n": 0, "dumped": False, "state": None}

    def wrapped(state, units, **kw):
        out = orig_step(state, units, **kw)
        ctr["n"] += 1
        n = ctr["n"]
        LEDGER_STATE["state"] = state
        types = LEDGER_STATE.get("types")
        nonfinite = (
            (not np.all(np.isfinite(state.mass)))
            or (not np.all(np.isfinite(np.asarray(state.f, dtype=np.float64))))
        )
        if n % every == 0 or (nonfinite and not ctr["dumped"]):
            flags = state.flags
            fluid = flags == FLAG_FLUID
            iface = flags == FLAG_INTERFACE
            wet = fluid | iface
            nwet = int(np.count_nonzero(wet))
            mass = np.asarray(state.mass, dtype=np.float64)
            f64 = np.asarray(state.f, dtype=np.float64)
            u = np.asarray(state.u, dtype=np.float64)
            rho = np.asarray(state.rho, dtype=np.float64)
            eps = fluid_fraction(state)
            speed = np.sqrt(np.sum(u * u, axis=0))
            negf = int(np.count_nonzero(f64[:, wet] < 0.0)) if nwet else 0
            negcell_mask = np.any(f64 < 0.0, axis=0) & wet
            negcells = int(np.count_nonzero(negcell_mask))
            porous_mask = np.asarray(state.porous.porosity) < 1.0
            negporous = int(np.count_nonzero(negcell_mask & porous_mask))
            temps = np.asarray(state.temperature, dtype=np.float64)[wet] if nwet else None
            mintemp = (
                float(np.nanmin(temps)) if temps is not None and temps.size
                and np.any(np.isfinite(temps)) else float("nan")
            )
            by_type = ""
            if types is not None:
                parts = []
                for name, code in LEDGER_STATE.get("type_codes", {}).items():
                    m = (types == code) & wet
                    parts.append(f"{name}={float(np.sum(mass[m])):.4g}/{int(m.sum())}")
                by_type = " ".join(parts)
                zsum = mass.sum(axis=(0, 1))
                nz_ = np.nonzero(zsum > 1e-9)[0]
                if nz_.size:
                    by_type += " | z[" + ",".join(
                        f"{int(z)}:{zsum[z]:.1f}" for z in nz_
                    ) + "]"
            fh.write(
                f"{n},{nwet},{int(fluid.sum())},{int(iface.sum())},"
                f"{float(np.sum(mass[wet])) if nwet else 0.0:.6g},"
                f"{1.5*max(nwet,1):.6g},"
                f"{float(np.max(speed[wet])) if nwet else 0.0:.6g},"
                f"{float(np.min(rho[wet])) if nwet else 0.0:.6g},"
                f"{float(np.max(rho[wet])) if nwet else 0.0:.6g},"
                f"{float(np.min(eps[wet])) if nwet else 0.0:.6g},"
                f"{float(np.max(eps[wet])) if nwet else 0.0:.6g},"
                f"{negf},{negcells},{negporous},{mintemp:.5g},"
                f"{int(nonfinite)},{by_type}\n"
            )
        if nonfinite and not ctr["dumped"]:
            ctr["dumped"] = True
            dump_nonfinite(fh, state, n)
        return out

    pf.step = wrapped
    import core.filling_stage as cfs
    cfs.step = wrapped
    log(f"ledger -> {path} every {every} steps")

def dump_nonfinite(fh, state, n):
    from physics.filling import fluid_fraction
    mass = np.asarray(state.mass, dtype=np.float64)
    f64 = np.asarray(state.f, dtype=np.float64)
    bad = ~np.isfinite(mass)
    if not bad.any():
        bad = ~np.all(np.isfinite(f64), axis=0)
    idx = np.argwhere(bad)
    fh.write(f"# NONFINITE at step {n}: {len(idx)} cells\n")
    types = LEDGER_STATE.get("types")
    eps = fluid_fraction(state)
    for (x, y, z) in idx[:25]:
        t = int(types[x, y, z]) if types is not None else -1
        fh.write(
            f"#  cell ({x},{y},{z}) type={t} flag={int(state.flags[x,y,z])} "
            f"rho={float(state.rho[x,y,z]):.6g} mass={mass[x,y,z]:.6g} "
            f"eps={eps[x,y,z]:.6g} por={float(state.porous.porosity[x,y,z]):.4g} "
            f"u={np.asarray(state.u)[:,x,y,z]}\n"
        )
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nx_, ny_, nz_ = x + dx, y + dy, z + dz
                    if not (0 <= nx_ < mass.shape[0] and 0 <= ny_ < mass.shape[1]
                            and 0 <= nz_ < mass.shape[2]):
                        continue
                    if dx == dy == dz == 0:
                        continue
                    tt = int(types[nx_, ny_, nz_]) if types is not None else -1
                    fh.write(
                        f"#    nb ({nx_},{ny_},{nz_}) type={tt} "
                        f"flag={int(state.flags[nx_,ny_,nz_])} "
                        f"mass={mass[nx_,ny_,nz_]:.5g} "
                        f"por={float(state.porous.porosity[nx_,ny_,nz_]):.3g}\n"
                    )
    fh.flush()

if __name__ == "__main__":
    import threading
    threading.stack_size(512 * 1024 * 1024)
    t = threading.Thread(target=main)
    t.start()
    t.join()
