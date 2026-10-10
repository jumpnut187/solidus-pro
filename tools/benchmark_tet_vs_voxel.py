#!/usr/bin/env python3
"""Wall time and total solidification time, voxel vs tet, through ``run_pipeline``.

Each run uses the desktop default analyses (shrinkage, hot spots, fraction
liquid; no filling). The voxel size passed to the pipeline is ``--base`` (0.1
in); tet runs set ``settings.mesh.tet_edge_length_factor`` so the edge length
is ``base * factor``. Total solidification time is the largest finite
solidification time over metal points. ``field err`` is the mean absolute
difference of the solidification-time field from the reference, linearly
interpolated at up to 4,000 reference metal nodes, as a percentage of the
reference total.

The reference row is a fine tet run (``--reference-edge``, 0.05 in) used as
the h-converged value. A factor is acceptable when its total is within
``--tolerance`` (2 %) of the reference; the recommendation is the acceptable
factor with the smallest wall time.

Usage::

    venv/bin/python tools/benchmark_tet_vs_voxel.py
    venv/bin/python tools/benchmark_tet_vs_voxel.py --parts thick_thin --factors 1.0 1.5 2.0

fTetWild is multithreaded and not bit-deterministic, so tet totals vary by a
few tenths of a percent between runs.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.settings import get_settings  # noqa: E402
from core import simulation_pipeline as sp  # noqa: E402
from core.part_manager import is_metal  # noqa: E402
from core.stl_loader import STLLoader  # noqa: E402

PARTS = {
    "thick_thin": "sample_stl_files/thick_thin_test.stl",
    "l_bracket": "sample_stl_files/l_bracket.stl",
}
DESKTOP_ANALYSES = {
    "shrinkage": True,
    "hotspot": True,
    "niyama": False,
    "fraction_liquid": True,
    "filling": False,
}


@dataclass
class Run:
    part: str
    mesh_type: str
    size_in: float  # voxel size or tet edge length
    factor: Optional[float]  # tet edge factor, None for voxel
    points: int
    wall_s: float
    total_s: float
    metal_xyz: np.ndarray  # (M,3) in, metal points
    metal_times: np.ndarray  # (M,) s, solidification time at metal points


def _load(path: str) -> STLLoader:
    loader = STLLoader()
    with contextlib.redirect_stdout(io.StringIO()):
        if not loader.load_stl(os.path.join(REPO_ROOT, path)):
            raise RuntimeError(f"could not load {path}")
    return loader


def run_case(part: str, mesh_type: str, base_in: float, factor: float = 1.0) -> Run:
    """One ``run_pipeline`` call. ``factor`` scales the tet edge length."""
    mesh_settings = get_settings().mesh
    saved_factor = mesh_settings.tet_edge_length_factor
    mesh_settings.tet_edge_length_factor = factor
    try:
        loader = _load(PARTS[part])
        inputs = sp.PipelineInputs(
            "WCB", base_in, 0, analysis_options=dict(DESKTOP_ANALYSES), use_gpu=True, mesh_type=mesh_type
        )
        started = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            result = sp.run_pipeline(inputs, stl_loader=loader)
        wall_s = time.perf_counter() - started
    finally:
        mesh_settings.tet_edge_length_factor = saved_factor
    mesh = result.voxel_mesh
    times = np.asarray(result.engine.solidification_times, dtype=np.float64)
    part_types = getattr(mesh, "voxel_part_types", None)
    # A single-STL voxel run has no part assignment: every point is cast metal.
    metal = np.ones(len(times), dtype=bool) if part_types is None else is_metal(
        np.asarray(part_types, dtype=np.int32)
    )
    return Run(
        part=part,
        mesh_type=mesh_type,
        size_in=base_in * factor if mesh_type == "tet" else base_in,
        factor=factor if mesh_type == "tet" else None,
        points=int(mesh.get_voxel_count()),
        wall_s=wall_s,
        total_s=float(np.nanmax(times[metal])),
        metal_xyz=np.asarray(mesh.voxel_centers, dtype=np.float64)[metal],
        metal_times=times[metal],
    )


def field_error(run: Run, reference: Run, max_probes: int = 4000) -> float:
    """Mean |t_run - t_ref| at reference metal nodes, % of the reference total."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    rng = np.random.default_rng(0)
    finite = np.isfinite(reference.metal_times)
    probes = np.flatnonzero(finite)
    if probes.size > max_probes:
        probes = rng.choice(probes, max_probes, replace=False)
    xyz = reference.metal_xyz[probes]
    keep = np.isfinite(run.metal_times)
    values = LinearNDInterpolator(run.metal_xyz[keep], run.metal_times[keep])(xyz)
    outside = ~np.isfinite(values)
    if np.any(outside):
        values[outside] = NearestNDInterpolator(run.metal_xyz[keep], run.metal_times[keep])(xyz[outside])
    return float(100.0 * np.mean(np.abs(values - reference.metal_times[probes])) / reference.total_s)


def recommend_factor(runs: Sequence[Run], reference: Dict[str, float], tolerance: float) -> Optional[float]:
    """Smallest summed wall time among factors within ``tolerance`` of the reference on every part."""
    by_factor: Dict[float, List[Run]] = {}
    for run in runs:
        if run.mesh_type == "tet" and run.factor is not None:
            by_factor.setdefault(run.factor, []).append(run)
    best: Optional[float] = None
    best_wall = float("inf")
    for factor, factor_runs in sorted(by_factor.items()):
        if {r.part for r in factor_runs} != set(reference):
            continue
        if all(abs(r.total_s / reference[r.part] - 1.0) <= tolerance for r in factor_runs):
            wall = sum(r.wall_s for r in factor_runs)
            if wall < best_wall:
                best, best_wall = factor, wall
    return best


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--parts", nargs="+", default=list(PARTS), choices=list(PARTS))
    parser.add_argument("--base", type=float, default=0.1, help="base voxel size, in")
    parser.add_argument("--voxel-sizes", nargs="+", type=float, default=[0.1, 0.05])
    parser.add_argument("--factors", nargs="+", type=float, default=[1.0, 1.5, 2.0])
    parser.add_argument("--reference-edge", type=float, default=0.05, help="tet edge of the h-converged run, in")
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args(argv)

    # Warm-up: numba compilation and backend start-up stay out of the table.
    run_case("l_bracket", "tet", 0.2)
    run_case("l_bracket", "voxel", 0.2)

    runs: List[Run] = []
    reference: Dict[str, float] = {}
    reference_runs: Dict[str, Run] = {}
    for part in args.parts:
        for size in args.voxel_sizes:
            runs.append(run_case(part, "voxel", size))
        for factor in args.factors:
            runs.append(run_case(part, "tet", args.base, factor))
        ref = run_case(part, "tet", args.base, args.reference_edge / args.base)
        reference[part] = ref.total_s
        reference_runs[part] = ref
        print(f"{part}: reference tet edge {ref.size_in:g} in, {ref.points} nodes, "
              f"wall {ref.wall_s:.1f} s, total {ref.total_s:.2f} s", flush=True)

    print()
    print("| part | mesh | size in | factor | points | wall s | total s | vs tet ref | field err |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        factor = f"{r.factor:g}" if r.factor is not None else "-"
        delta = 100.0 * (r.total_s / reference[r.part] - 1.0)
        err = field_error(r, reference_runs[r.part])
        print(f"| {r.part} | {r.mesh_type} | {r.size_in:g} | {factor} | {r.points} | "
              f"{r.wall_s:.1f} | {r.total_s:.2f} | {delta:+.1f} % | {err:.1f} % |")
    best = recommend_factor(runs, reference, args.tolerance)
    print()
    if best is None:
        print(f"No tested factor is within {100 * args.tolerance:g} % of the reference on every part.")
    else:
        print(f"Recommended tet_edge_length_factor: {best:g} "
              f"(within {100 * args.tolerance:g} % of the reference on every part, least wall time).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
