"""Headless mold-filling driver.

Usage:
  python fill_driver.py --dir <assembly dir> --voxel 0.17 --refine 4 [--ledger out.csv]

Loads every STL/STEP in a directory through PartManager (same auto part typing
the GUI uses), builds the adaptive voxel mesh, picks the pour point at the top
centre of the sprue, and runs core.filling_stage.run_filling_stage.

No tracked file is modified; instrumentation is installed by monkeypatching
physics.filling.step at import time.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path("/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
sys.path.insert(0, str(REPO))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import faulthandler  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

faulthandler.enable()


def _trace_kdtree():
    """Log every cKDTree build: size, finiteness, and the Python call site.

    Two runs died with 512 MB of C stack consumed inside ckdtree build() --
    runaway recursion, which is what a non-finite coordinate does to a
    midpoint split.
    """
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
        except Exception as exc:  # noqa: BLE001
            log(f"  FAILED {f.name}: {exc}")
            continue
        for pid in ids:
            part = pm.get_part(pid)
            log(f"  {part.name:35s} -> {part.part_type.value}")

    parts = pm.get_all_parts()
    log(f"{len(parts)} parts loaded")

    # Pour point: top centre of the sprue (matches Test/fill_verification/e2e_gui.py)
    from core.part_manager import PartType
    sprue = next((p for p in parts if p.part_type is PartType.SPRUE), None)
    if sprue is None:
        raise SystemExit("no sprue part found")
    b = sprue.get_bounds()  # (3, 2): rows X, Y, Z; cols min, max
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
    # --- MECHANISM 3 fix validation ------------------------------------
    # physics/filling.py:2076-2091 passes float(TRT_MAGIC_PARAMETER) into
    # collide_kernel, which forms
    #     half = 1/omega_plus - 1/2 ;  omega_minus = 1/(magic/half + 1/2)
    # (physics/fluid_flow.py:1337-1339). At tau_plus = 0.50001, half = 1.04e-5
    # and the stock magic = 3/16 gives tau_minus = 18,060 -- the odd,
    # momentum-carrying moments have an 18,000-step relaxation time.
    # Setting magic = 4.5 * half emulates the proposed "clamp tau_minus <= 5"
    # exactly at the molecular tau, which is where all but a handful of cells
    # sit, and over-relaxes omega_minus (toward 2) only in the few cells where
    # Smagorinsky has already raised tau_eff -- i.e. conservative in the
    # stabilising direction. Monkeypatch only; the repo is untouched.
    _magic = os.environ.get("MAGIC")
    if _magic:
        import physics.filling as _pf
        log(f"MAGIC OVERRIDE: TRT_MAGIC_PARAMETER "
            f"{_pf.TRT_MAGIC_PARAMETER} -> {float(_magic)}")
        _pf.TRT_MAGIC_PARAMETER = float(_magic)

    capture_types(fs)
    install_pocket(fs,
                   os.environ.get("POCKET", "pocket"),
                   int(os.environ.get("POCKET_FROM", "18800")),
                   int(os.environ.get("POCKET_EVERY", "20")),
                   int(os.environ.get("POCKET_PERSTEP", "20080")))

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
    """Grab the coarse part-type grid the solver actually runs on.

    Hooked on build_porous_zone rather than coarsen_part_types: it is called
    with solve_types a few lines later and cannot be missed.
    """
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
        zone = orig(dense_types, specs, **kw)
        try:
            for sp in specs or []:
                log(f"SPEC zone={sp.zone_id} cells={sp.cell_count} "
                    f"porosity={sp.porosity} pore_d={sp.pore_diameter_m:.6g} "
                    f"k1={sp.k1_m2:.6g} inv_k2={sp.inv_k2_per_m:.6g} "
                    f"along={sp.cells_along_flow:.4g} corr={sp.thickness_correction:.6g} "
                    f"thick_m={sp.thickness_m:.6g} area={sp.front_face_area_m2:.6g}")
            c0 = np.asarray(zone.drag_c0, dtype=np.float64)
            c1 = np.asarray(zone.drag_c1, dtype=np.float64)
            por = np.asarray(zone.porosity, dtype=np.float64)
            m = por < 1.0
            log(f"DRAG COEFFS: porous cells={int(m.sum())} "
                f"c0 unique={sorted(set(np.round(c0[m],12).tolist()))} "
                f"c1 unique={sorted(set(np.round(c1[m],9).tolist()))} "
                f"c0 off-filter={sorted(set(np.round(c0[~m],12).tolist()))[:3]} "
                f"c1 off-filter={sorted(set(np.round(c1[~m],9).tolist()))[:3]}")
            LEDGER_STATE["zone"] = zone
            # the stability map: fractional momentum removed per step,
            # R = 2[(2c0-1)+c1|u|] / (2c0 + c1|u|)
            cc0 = float(c0[m].max()) if m.any() else 0.5
            cc1 = float(c1[m].max()) if m.any() else 0.0
            for uu in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0):
                a = cc1 * uu
                R = 2.0 * ((2*cc0 - 1.0) + a) / (2*cc0 + a)
                log(f"  R(|u|={uu:<5}) = {R:.6f}   gain|1-R| = {abs(1.0-R):.6f}"
                    f"   c1|u| = {a:.4g}")
        except Exception:
            import traceback
            log("SPEC log FAILED:\n" + traceback.format_exc())
        return zone

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
    """Wrap physics.filling.step to record a per-N-step ledger."""
    import physics.filling as pf
    from physics.filling import (
        FLAG_FLUID, FLAG_INTERFACE, FLAG_GAS, FLAG_OBSTACLE, fluid_fraction,
    )

    fh = open(path, "w", buffering=1)
    fh.write(
        "step,wet_cells,fluid_cells,iface_cells,wet_mass,ceiling,"
        "max_abs_u,rho_min,rho_max,eps_min,eps_max,neg_f,neg_cells,"
        "neg_porous_cells,min_temp,nonfinite,"
        "eps_fluid_min,eps_fluid_max,eps_filter_max,eps_filternb_max,"
        "u_p99,u_med,wetz_by_type,mass_by_type\n"
    )
    orig_step = pf.step
    ctr = {"n": 0, "dumped": False, "state": None}

    def wrapped(state, units, **kw):
        out = orig_step(state, units, **kw)
        ctr["n"] += 1
        n = ctr["n"]
        LEDGER_STATE["state"] = state
        types = LEDGER_STATE.get("types")
        # mass goes non-finite in the same step f does, and widening f to
        # float64 every step dominates a long run -- so check mass per step and
        # f only on the ledger cadence.
        nonfinite = not np.all(np.isfinite(state.mass))
        due = (n % every == 0)
        if due or (nonfinite and not ctr["dumped"]):
            flags = state.flags
            fluid = flags == FLAG_FLUID
            iface = flags == FLAG_INTERFACE
            wet = fluid | iface
            nwet = int(np.count_nonzero(wet))
            mass = np.asarray(state.mass, dtype=np.float64)
            f64 = np.asarray(state.f, dtype=np.float64)
            if not np.all(np.isfinite(f64)):
                nonfinite = True
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
            # --- FLUID-only eps band, and the filter zone + its neighbours ---
            eps_fl_min = eps_fl_max = float("nan")
            if fluid.any():
                ef = eps[fluid]
                eps_fl_min, eps_fl_max = float(ef.min()), float(ef.max())
            eps_filt = eps_filtnb = float("nan")
            fmask = LEDGER_STATE.get("filter_mask")
            if fmask is None and types is not None:
                from core.part_manager import PART_TYPE_FILTER
                fmask = (types == PART_TYPE_FILTER)
                LEDGER_STATE["filter_mask"] = fmask
                if fmask.any():
                    from scipy import ndimage
                    LEDGER_STATE["filter_nb_mask"] = ndimage.binary_dilation(
                        fmask, structure=np.ones((3, 3, 3), bool)
                    )
                else:
                    LEDGER_STATE["filter_nb_mask"] = fmask
            if fmask is not None and fmask.any():
                sel = fmask & wet
                if sel.any():
                    eps_filt = float(eps[sel].max())
                selnb = LEDGER_STATE["filter_nb_mask"] & wet
                if selnb.any():
                    eps_filtnb = float(eps[selnb].max())
            up99 = float(np.percentile(speed[wet], 99)) if nwet else 0.0
            umed = float(np.median(speed[wet])) if nwet else 0.0

            # --- per-part WET CELL counts by z ---
            wetz = ""
            if types is not None:
                chunks = []
                for name, code in LEDGER_STATE.get("type_codes", {}).items():
                    m = (types == code) & wet
                    if not m.any():
                        continue
                    prof = m.sum(axis=(0, 1))
                    nzz = np.nonzero(prof)[0]
                    chunks.append(
                        name + "[" + ";".join(
                            f"{int(z)}:{int(prof[z])}" for z in nzz) + "]"
                    )
                wetz = " ".join(chunks)

            by_type = ""
            if types is not None:
                parts = []
                for name, code in LEDGER_STATE.get("type_codes", {}).items():
                    m = (types == code) & wet
                    parts.append(f"{name}={float(np.sum(mass[m])):.4g}/{int(m.sum())}")
                by_type = " ".join(parts)
                # z-profile of wet mass, to see where the metal actually is
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
                f"{int(nonfinite)},"
                f"{eps_fl_min:.6g},{eps_fl_max:.6g},"
                f"{eps_filt:.6g},{eps_filtnb:.6g},"
                f"{up99:.6g},{umed:.6g},{wetz},{by_type}\n"
            )
        if nonfinite and not ctr["dumped"]:
            ctr["dumped"] = True
            dump_nonfinite(fh, state, n)
        return out

    pf.step = wrapped
    # filling_stage imported `step` by name
    import core.filling_stage as cfs
    cfs.step = wrapped
    log(f"ledger -> {path} every {every} steps")


# ---------------------------------------------------------------------------
# MECHANISM 3 probe: the filter-inlet pocket, and the drag actually applied
# ---------------------------------------------------------------------------
#
# The fatal pocket is ORIGINAL x[115,116] y[48,51] z[20,26] == PADDED
# x[116,117] y[49,52] z[21,27]. Watch a margin around it.
POCKET = (113, 121, 46, 56, 18, 30)      # padded, inclusive-exclusive slices


def install_pocket(fs, out, from_step, every, perstep_from):
    """Second wrapper on physics.filling.step.

    Records, without touching the repo:
      * per-step pocket + global summary from ``perstep_from``
      * per-cell pocket dumps every ``every`` steps from ``from_step``
      * the Guo-Zhao drag ACTUALLY applied, recomputed from the kernel's own
        closed form on the recorded (rho, u, c0, c1) -- collide_kernel computes
        it inline (physics/fluid_flow.py:1223-1257) so there is no function to
        hook, but the formula is exact and the inputs are all in ``state``.
    """
    import physics.filling as pf
    from physics.filling import FLAG_FLUID, FLAG_INTERFACE

    x0, x1, y0, y1, z0, z1 = POCKET
    sl = (slice(x0, x1), slice(y0, y1), slice(z0, z1))

    gh = open(f"{out}_summary.csv", "w", buffering=1)
    gh.write(
        "step,gmax_u,gmax_x,gmax_y,gmax_z,gmax_type,gmax_por,gmax_flag,"
        "gmax_rho,pk_max_u,pk_x,pk_y,pk_z,pk_type,pk_rho,"
        "Rmax_porous,Rmax_x,Rmax_y,Rmax_z,Rmax_u,"
        "n_por_R_gt1,n_por_R_gt1p5,n_por_R_gt1p9,n_por_wet,"
        "gmax_u_porous,gmax_u_por_x,gmax_u_por_y,gmax_u_por_z,"
        "fmax_abs,fmin,rho_min,rho_max\n"
    )
    ch = open(f"{out}_cells.csv", "w", buffering=1)
    ch.write(
        "step,x,y,z,type,flag,por,c0,c1,rho,mass,eps,ux,uy,uz,speed,"
        "denom,R,drag_mag,dragx,dragy,dragz,fgrav_z,fmin_cell,fmax_cell\n"
    )

    orig_step = pf.step
    ctr = {"n": 0}

    def wrapped(state, units, **kw):
        out_st = orig_step(state, units, **kw)
        n = ctr["n"] = ctr["n"] + 1
        if n < from_step:
            return out_st
        due_cells = (n % every == 0)
        due_sum = due_cells or (n >= perstep_from)
        if not due_sum:
            return out_st

        types = LEDGER_STATE.get("types")
        zone = state.porous
        c0 = zone.drag_c0_f64 if hasattr(zone, "drag_c0_f64") else \
            np.asarray(zone.drag_c0, np.float64)
        c1 = zone.drag_c1_f64 if hasattr(zone, "drag_c1_f64") else \
            np.asarray(zone.drag_c1, np.float64)
        por = np.asarray(zone.porosity, np.float64)
        u = np.asarray(state.u, np.float64)
        rho = np.asarray(state.rho, np.float64)
        mass = np.asarray(state.mass, np.float64)
        flags = state.flags
        wet = (flags == FLAG_FLUID) | (flags == FLAG_INTERFACE)
        with np.errstate(over="ignore", invalid="ignore"):
            speed = np.sqrt(np.sum(u * u, axis=0))
            # the kernel's own quantities
            a = c1 * speed
            denom = 2.0 * c0 + a                       # == c0+sqrt(c0^2+c1|v|)
            R = np.where(denom > 0, 2.0 * ((2.0 * c0 - 1.0) + a) / denom, 0.0)

        pm = (por < 1.0) & wet
        sp_w = np.where(wet, speed, -1.0)
        gi = np.unravel_index(int(np.nanargmax(sp_w)), speed.shape)
        Rw = np.where(pm, R, -1.0)
        ri = np.unravel_index(int(np.nanargmax(Rw)), R.shape)
        spp = np.where(pm, speed, -1.0)
        pi = np.unravel_index(int(np.nanargmax(spp)), speed.shape)
        pk_sp = np.where(wet[sl], speed[sl], -1.0)
        pj = np.unravel_index(int(np.nanargmax(pk_sp)), pk_sp.shape)
        pj = (pj[0] + x0, pj[1] + y0, pj[2] + z0)
        tt = (lambda ix: int(types[ix]) if types is not None else -1)
        f64 = np.asarray(state.f, np.float64)
        gh.write(
            f"{n},{speed[gi]:.6g},{gi[0]},{gi[1]},{gi[2]},{tt(gi)},"
            f"{por[gi]:.3g},{int(flags[gi])},{rho[gi]:.6g},"
            f"{speed[pj]:.6g},{pj[0]},{pj[1]},{pj[2]},{tt(pj)},{rho[pj]:.6g},"
            f"{R[ri]:.6g},{ri[0]},{ri[1]},{ri[2]},{speed[ri]:.6g},"
            f"{int(np.count_nonzero(pm & (R > 1.0)))},"
            f"{int(np.count_nonzero(pm & (R > 1.5)))},"
            f"{int(np.count_nonzero(pm & (R > 1.9)))},"
            f"{int(np.count_nonzero(pm))},"
            f"{speed[pi]:.6g},{pi[0]},{pi[1]},{pi[2]},"
            f"{float(np.nanmax(np.abs(f64))):.6g},{float(np.nanmin(f64)):.6g},"
            f"{float(np.nanmin(rho[wet])) if wet.any() else 0:.6g},"
            f"{float(np.nanmax(rho[wet])) if wet.any() else 0:.6g}\n"
        )
        if due_cells:
            grav = LEDGER_STATE.get("gz", float("nan"))
            for x in range(x0, x1):
                for y in range(y0, y1):
                    for z in range(z0, z1):
                        if not wet[x, y, z]:
                            continue
                        dm = 2.0 * rho[x, y, z] * (
                            (2.0 * c0[x, y, z] - 1.0)
                            + c1[x, y, z] * speed[x, y, z])
                        den = rho[x, y, z] * por[x, y, z]
                        ch.write(
                            f"{n},{x},{y},{z},{tt((x,y,z))},"
                            f"{int(flags[x,y,z])},{por[x,y,z]:.4g},"
                            f"{c0[x,y,z]:.8g},{c1[x,y,z]:.8g},"
                            f"{rho[x,y,z]:.8g},{mass[x,y,z]:.8g},"
                            f"{mass[x,y,z]/den if den>0 else 0:.6g},"
                            f"{u[0,x,y,z]:.6g},{u[1,x,y,z]:.6g},"
                            f"{u[2,x,y,z]:.6g},{speed[x,y,z]:.6g},"
                            f"{denom[x,y,z]:.6g},{R[x,y,z]:.6g},"
                            f"{dm:.6g},{-dm*u[0,x,y,z]:.6g},"
                            f"{-dm*u[1,x,y,z]:.6g},{-dm*u[2,x,y,z]:.6g},"
                            f"{grav},"
                            f"{float(np.nanmin(f64[:,x,y,z])):.6g},"
                            f"{float(np.nanmax(f64[:,x,y,z])):.6g}\n"
                        )
        return out_st

    pf.step = wrapped
    import core.filling_stage as cfs
    cfs.step = wrapped
    log(f"POCKET probe installed: {out}_summary.csv / {out}_cells.csv "
        f"pocket(padded)={POCKET} from={from_step} every={every} "
        f"perstep_from={perstep_from}")


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
        # neighbourhood
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
    # scipy's cKDTree (built inside trimesh proximity, outside this repo's
    # own balanced_tree=False fix) recurses past the 8 MB macOS main-thread
    # stack on this assembly and takes the process down with SIGSEGV.
    # Run everything on a 512 MB thread stack instead.
    import threading
    threading.stack_size(512 * 1024 * 1024)
    t = threading.Thread(target=main)
    t.start()
    t.join()
