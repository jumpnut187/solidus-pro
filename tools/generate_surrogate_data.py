#!/usr/bin/env python3
"""
Generate surrogate training records from synthetic geometries with the full solver.

Sweeps K seeded geometries (``core.surrogate.geometries``) x alloys x
Chvorinov mold constants, captures each run as a ``.npz`` record and tracks
progress in ``<out-dir>/generation_plan.json`` so an interrupted batch can be
resumed (``capture_run`` names files from a time hash, so the sidecar is what
identifies a finished combination).

Voxel size per geometry (``voxel_size_for``): ``s = max(max_dim / grid,
cbrt(volume / target_voxels))``, then ``s <= min_thickness / 3`` and
``s >= cbrt(volume / 50000)``, rounded to 0.005 in.  ``--voxel-size`` pins a
fixed value instead.

Examples:

    # Show the plan without solving
    python tools/generate_surrogate_data.py --count 40 --dry-run

    # 40 geometries x all alloys x one mold constant into the default dataset
    python tools/generate_surrogate_data.py --count 40

    # Smoke batch into a temp dir
    python tools/generate_surrogate_data.py --count 3 --alloys WCB,CF8 \
        --out-dir /tmp/surrogate_smoke

    # 120 rigged assemblies (riser, chill, sand core), 3 alloys each
    python tools/generate_surrogate_data.py --assemblies --count 120 --alloys-per-geometry 3

    # 160 large multi-feature assemblies (12-36 in, several risers/cores/chills)
    python tools/generate_surrogate_data.py --complex --count 160 --alloys-per-geometry 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.materials import get_all_alloy_names
from core.surrogate.dataset import DEFAULT_DATASET_DIR, last_load_skips, load_dataset
from core.surrogate.assemblies import ASSEMBLY_FAMILIES, COMPLEX_FAMILIES, iter_assemblies
from core.surrogate.geometries import FAMILIES, iter_geometries, named_catalog, voxel_size_for
from core.surrogate.training import (
    _load_plan,
    _plan_record_exists,
    _run_on_large_stack,
    alloys_for_geometry,
    generate_dataset,
    generation_plan_key,
)

PLAN_FILENAME = "generation_plan.json"


def _parse_floats(text: str):
    return [float(x) for x in text.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate surrogate training records from synthetic geometries.")
    parser.add_argument("--count", type=int, default=20, help="Number of seeded geometries (default 20).")
    parser.add_argument("--seed", type=int, default=0, help="Geometry RNG seed (default 0).")
    parser.add_argument("--families", default="",
                        help=f"Comma-separated subset of {FAMILIES}, or of {ASSEMBLY_FAMILIES} "
                             "with --assemblies (default all).")
    parser.add_argument("--assemblies", action="store_true",
                        help="Rigged assemblies (casting + riser, optional chill, sand cores) from "
                             "core.surrogate.assemblies instead of bare castings.")
    parser.add_argument("--complex", action="store_true",
                        help=f"Large multi-feature assemblies ({', '.join(COMPLEX_FAMILIES)}); "
                             "raises the voxel-size rule defaults to 250000 target / 400000 cap so "
                             "the voxel size follows the encoding cell instead of the voxel count.")
    parser.add_argument("--alloys-per-geometry", type=int, default=None,
                        help="Run each geometry with this many of the alloys (seeded by name) "
                             "instead of all of them.")
    parser.add_argument("--catalog", action="store_true", help="Use the fixed named catalog instead of seeded geometries.")
    parser.add_argument("--alloys", default="", help="Comma-separated alloy names (default: all catalog alloys).")
    parser.add_argument("--mold-constants", default="280", help="Comma-separated chvorinov_mold_constant values (default 280).")
    parser.add_argument("--out-dir", default=DEFAULT_DATASET_DIR, help="Record directory.")
    parser.add_argument("--grid", type=int, default=64, help="Encoding grid resolution (default 64).")
    parser.add_argument("--voxel-size", type=float, default=None, help="Fixed voxel size in inches (default: per-geometry rule).")
    parser.add_argument("--target-voxels", type=int, default=None,
                        help="Voxel-size rule target (default 20000; 250000 with --complex).")
    parser.add_argument("--max-voxels", type=int, default=None,
                        help="Voxel-size rule hard cap (default 50000; 400000 with --complex).")
    parser.add_argument("--dense", action="store_true",
                        help="Force voxel size <= encoding cell (max_dim / grid) so the 64^3 occupancy "
                             "is dense like UI captures; raises voxel counts for small parts.")
    parser.add_argument("--no-retry-failed", action="store_true", help="Do not re-run combinations marked failed in the plan.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit without solving.")
    args = parser.parse_args()
    if args.target_voxels is None:
        args.target_voxels = 250_000 if args.complex else 20_000
    if args.max_voxels is None:
        args.max_voxels = 400_000 if args.complex else 50_000

    alloys = [a.strip() for a in args.alloys.split(",") if a.strip()] or get_all_alloy_names()
    mold_constants = _parse_floats(args.mold_constants) or [280.0]
    default_families = COMPLEX_FAMILIES if args.complex else ASSEMBLY_FAMILIES if args.assemblies else FAMILIES
    families = [f.strip() for f in args.families.split(",") if f.strip()] or list(default_families)
    if args.assemblies or args.complex:
        # Built on the solver's large-stack thread: rigging rasterises and
        # ray-casts every design, and the same native recursion that needs it
        # in the solver segfaults on the 8 MB macOS main-thread stack.
        specs = _run_on_large_stack(lambda: list(iter_assemblies(args.count, seed=args.seed, families=families)))
    elif args.catalog:
        specs = list(named_catalog().values())
    else:
        specs = list(iter_geometries(args.count, seed=args.seed, families=families))
    plan_path = os.path.join(args.out_dir, PLAN_FILENAME)
    plan = _load_plan(plan_path)

    total = sum(len(alloys_for_geometry(s.name, alloys, args.alloys_per_geometry)) for s in specs) * len(mold_constants)
    per = f"{args.alloys_per_geometry} of {len(alloys)}" if args.alloys_per_geometry else f"{len(alloys)}"
    print(f"Plan: {len(specs)} geometries x {per} alloys x {len(mold_constants)} mold constants = {total} runs")
    print(f"  out-dir: {args.out_dir}  grid: {args.grid}  plan: {plan_path}")
    pending = 0
    for spec in specs:
        vs = float(args.voxel_size) if args.voxel_size is not None else voxel_size_for(
            spec, grid=args.grid, target_voxels=args.target_voxels,
            max_voxels=args.max_voxels, dense=args.dense,
        )
        est = spec.mesh.volume / vs ** 3
        cell = spec.max_dim / args.grid
        statuses = []
        for alloy in alloys_for_geometry(spec.name, alloys, args.alloys_per_geometry):
            for mc in mold_constants:
                entry = plan.get(generation_plan_key(spec.name, alloy, mc, vs, args.grid))
                status = entry.get("status") if entry else "pending"
                if status == "done" and not _plan_record_exists(entry, args.out_dir):
                    status = "pending"
                if status == "pending" or (status == "failed" and not args.no_retry_failed):
                    pending += 1
                statuses.append(status)
        summary = ", ".join(f"{s}={statuses.count(s)}" for s in sorted(set(statuses)))
        print(
            f"  {spec.name:34s} max_dim={spec.max_dim:5.2f}in t_min={spec.min_thickness:.3f}in "
            f"vs={vs:.3f}in (~{est:,.0f} voxels, {vs / cell:.2f}x grid cell)  [{summary}]"
        )
    print(f"  pending runs: {pending}")
    if args.dry_run:
        return 0

    started = time.time()
    generated = generate_dataset(
        alloys,
        args.out_dir,
        grid=args.grid,
        progress_cb=lambda msg: print(msg, flush=True),
        geometries=specs,
        voxel_size=args.voxel_size,
        target_voxels=args.target_voxels,
        max_voxels=args.max_voxels,
        dense=args.dense,
        mold_constants=mold_constants,
        plan_path=plan_path,
        retry_failed=not args.no_retry_failed,
        alloys_per_geometry=args.alloys_per_geometry,
    )
    elapsed = time.time() - started
    plan = _load_plan(plan_path)
    walls = [e["wall_s"] for e in plan.values() if e.get("status") == "done" and "wall_s" in e]
    failed = [(k, e.get("error")) for k, e in plan.items() if e.get("status") in ("failed", "invalid")]
    print(f"\nGenerated {generated} records in {elapsed:.0f}s"
          + (f"; mean solver wall {sum(walls) / len(walls):.1f}s/run (n={len(walls)})" if walls else ""))
    for key, error in failed:
        print(f"  {key}: {error}")

    records = load_dataset(args.out_dir)
    skips = last_load_skips()
    print(f"load_dataset: {len(records)} usable records, {len(skips)} skipped")
    for path, reason in skips:
        print(f"  skipped {os.path.basename(path)}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
