"""Graded sand mold against the uniform lattice mold (MoldParams.mold_grading_levels).

Standalone verification script (not a unittest suite):

    venv/bin/python tests/verify_graded_mold.py            # the case table
    venv/bin/python tests/verify_graded_mold.py --sweep    # band x levels sweep
    venv/bin/python tests/verify_graded_mold.py --only cube_2in_wall4_0.1

Every case runs the full pipeline twice or more on the explicit auto mold:
once on the uniform lattice (``mold_grading_levels = 0``) and once per
grading. Per run it prints the mold sand cells (lattice and graded), the
wall time of the thermal solve (every ``CoupledMoldSolver.step``, casting
and mold) and of the mold model alone (``flush``), the total freeze time, and
the per-voxel solidification end-time difference from the uniform run (max
and mean of |dt|, in seconds and in % of the uniform total). Voxels that
froze in one run and not in the other (a sleeved riser still liquid at the
horizon) are counted apart.

Cases use only tests/ci_models (CI-safe geometry).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.helpers.ci_models import ci_model  # noqa: E402
from core.part_manager import build_part_manager  # noqa: E402
from core.simulation_pipeline import PipelineInputs, run_pipeline  # noqa: E402
from physics.mold_coupling import CoupledMoldSolver  # noqa: E402
from tests.test_simulation_scenarios import _memory_monitor_check, _ok_validation  # noqa: E402


@dataclass
class Case:
    name: str
    parts: Sequence[Tuple[str, str]]  # (ci model, part type)
    voxel_in: float
    wall_in: float  # 0: automatic


CASES: List[Case] = [
    Case("cube_2in_wall2_0.2", [("block_2inch.stl", "cast")], 0.2, 2.0),
    Case("cube_2in_wall4_0.2", [("block_2inch.stl", "cast")], 0.2, 4.0),
    Case("cube_2in_wall2_0.1", [("block_2inch.stl", "cast")], 0.1, 2.0),
    Case("cube_2in_wall4_0.1", [("block_2inch.stl", "cast")], 0.1, 4.0),
    Case("plate_riser_auto_0.2", [("cast.stl", "cast"), ("riser.stl", "riser")], 0.2, 0.0),
    Case("l_bracket_auto_0.1", [("l_bracket.stl", "cast")], 0.1, 0.0),
    Case("thick_thin_auto_0.1", [("thick_thin_test.stl", "cast")], 0.1, 0.0),
]

#: (levels, band) runs per case in the default table.
DEFAULT_GRADINGS = [(3, 4)]
#: The sweep (run on SWEEP_CASES).
SWEEP_GRADINGS = [(levels, band) for band in (2, 4, 8) for levels in (1, 2, 3, 4)]
SWEEP_CASES = ["plate_riser_auto_0.2", "l_bracket_auto_0.1"]


@dataclass
class RunResult:
    end: np.ndarray
    lattice_cells: int
    mold_cells: int
    max_span: int
    solve_s: float
    mold_s: float
    total_s: float


def run(case: Case, levels: int, band: int) -> RunResult:
    with contextlib.redirect_stdout(io.StringIO()):
        manager = build_part_manager([{"filepath": ci_model(m), "part_type": t} for m, t in case.parts])
    mold = {"auto_mold": 1, "wall_thickness_in": case.wall_in,
            "mold_grading_levels": levels, "mold_fine_band_cells": band}
    inputs = PipelineInputs(
        alloy_name="WCB",
        voxel_size=case.voxel_in,
        refinement_level=0,
        memory_mode="conservative",
        analysis_options={"shrinkage": True, "hotspot": True},
        max_voxels_for_visualization=5_000,
        solidification_params={"mold": mold},
    )
    clock = {"solve": 0.0, "mold": 0.0}
    step, flush = CoupledMoldSolver.step, CoupledMoldSolver.flush

    def timed_step(self, dt):
        t0 = time.perf_counter()
        step(self, dt)
        clock["solve"] += time.perf_counter() - t0

    def timed_flush(self):
        t0 = time.perf_counter()
        flush(self)
        clock["mold"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch("core.voxel_mesh.auto_adjust_voxel_size", return_value=(case.voxel_in, False, "")))
        stack.enter_context(patch("core.voxel_mesh.validate_mesh_parameters", side_effect=_ok_validation))
        stack.enter_context(patch("core.voxel_mesh.MemoryMonitor.start", lambda self: None))
        stack.enter_context(patch("core.voxel_mesh.MemoryMonitor.check", _memory_monitor_check))
        stack.enter_context(patch.object(CoupledMoldSolver, "step", timed_step))
        stack.enter_context(patch.object(CoupledMoldSolver, "flush", timed_flush))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        engine = run_pipeline(inputs, part_manager=manager).engine
    total = time.perf_counter() - t0
    ext = engine._mold_extension
    return RunResult(
        end=np.asarray(engine.solidification_end_times, dtype=np.float64),
        lattice_cells=int(ext.lattice_mold_cells),
        mold_cells=int(ext.n_mold),
        max_span=int(ext.max_span),
        solve_s=clock["solve"],
        mold_s=clock["mold"],
        total_s=total,
    )


def compare(reference: RunResult, other: RunResult) -> Dict[str, float]:
    a, b = reference.end, other.end
    both = np.isfinite(a) & np.isfinite(b)
    total = float(np.nanmax(a))
    diff = np.abs(a[both] - b[both])
    return {
        "max_s": float(diff.max()) if diff.size else 0.0,
        "mean_s": float(diff.mean()) if diff.size else 0.0,
        "max_pct": 100.0 * float(diff.max()) / total if diff.size else 0.0,
        "mean_pct": 100.0 * float(diff.mean()) / total if diff.size else 0.0,
        "unmatched": int(np.count_nonzero(np.isfinite(a) != np.isfinite(b))),
    }


HEADER = (
    f"{'case':24s} {'L':>2s} {'band':>4s} {'lattice':>10s} {'graded':>10s} {'cut':>6s} {'span':>4s} "
    f"{'solve s':>8s} {'mold s':>7s} {'run s':>7s} {'freeze s':>9s} {'max|dt| s':>9s} {'%':>6s} "
    f"{'mean s':>7s} {'%':>6s} {'unm':>4s}"
)


def row(name: str, levels: int, band: int, res: RunResult, cmp: Optional[Dict[str, float]]) -> str:
    cut = res.lattice_cells / max(res.mold_cells, 1)
    text = (
        f"{name:24s} {levels:2d} {band:4d} {res.lattice_cells:10,d} {res.mold_cells:10,d} {cut:6.2f} {res.max_span:4d} "
        f"{res.solve_s:8.1f} {res.mold_s:7.1f} {res.total_s:7.1f} {float(np.nanmax(res.end)):9.1f}"
    )
    if cmp is None:
        return text
    return (
        text + f" {cmp['max_s']:9.2f} {cmp['max_pct']:6.2f} {cmp['mean_s']:7.3f} {cmp['mean_pct']:6.3f} "
        f"{cmp['unmatched']:4d}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep", action="store_true", help="band x levels sweep on SWEEP_CASES")
    parser.add_argument("--only", nargs="*", help="case names to run")
    parser.add_argument("--transition", type=int, default=None,
                        help="override core.mold_setup.GRADING_TRANSITION_CELLS for the graded runs")
    parser.add_argument("--grading", nargs="*", default=None,
                        help="levels:band pairs for the table (default 3:4)")
    args = parser.parse_args(argv)

    if args.transition is not None:
        import core.mold_setup as mold_setup

        mold_setup.GRADING_TRANSITION_CELLS = int(args.transition)
        print(f"GRADING_TRANSITION_CELLS = {args.transition}")
    cases = {c.name: c for c in CASES}
    names = args.only or (SWEEP_CASES if args.sweep else list(cases))
    gradings = SWEEP_GRADINGS if args.sweep else DEFAULT_GRADINGS
    if args.grading:
        gradings = [tuple(int(v) for v in g.split(":")) for g in args.grading]
    print(HEADER, flush=True)
    for name in names:
        case = cases[name]
        uniform = run(case, 0, 4)
        print(row(name, 0, 0, uniform, None), flush=True)
        for levels, band in gradings:
            graded = run(case, levels, band)
            print(row(name, levels, band, graded, compare(uniform, graded)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
