"""Bit-identity check for solver changes that must not move results.

A performance change to the thermal solver, the mesh build or the time loop
should give the same numbers, not close ones. This script runs a fixed set of
cases through ``run_pipeline`` and saves every numeric array the engine, the
mesh and the heat model end up with. Run it before and after a change and
compare the two folders:

    python tests/verify_solver_snapshot.py save /tmp/before
    ... change the code ...
    python tests/verify_solver_snapshot.py save /tmp/warmup    # see below
    python tests/verify_solver_snapshot.py save /tmp/after
    python tests/verify_solver_snapshot.py compare /tmp/before /tmp/after

``compare`` exits 0 when every array is equal bit for bit (NaN equals NaN).
With ``--rtol`` it accepts float arrays that agree to that relative tolerance
(float32 arrays to no better than 1e-6, a few of their last bits), still
requires integer and boolean arrays to be equal, and prints the largest
relative gap of each case. Use it for a change that is meant to move the last
digits, such as the fused kernels (``CASTSIM_FUSED_KERNELS=0`` is the
bit-for-bit path they are compared with).

Run the save twice after editing a file that holds Numba kernels. Numba
recompiles the kernels of an edited file, and a kernel compiled in the running
process does not always give the last bit a kernel loaded from the cache gives
(seen on ``plate_riser_015``: final temperatures 1.6e-11 degC apart, every
other array equal, with no code change at all). The first save after an edit
refills the cache; compare the second.

The cases cover: graded and uniform explicit mold, the boundary path with no
mold, cut cells on and off, a chill, a sand core, and a riser. They are the
CI models, so the script runs on a clean checkout. ``plate_riser_015`` has
100k voxels and takes about half a minute; the rest take seconds.

Standalone script, not part of the unittest suite. Method and results: implementation.md.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

ANALYSES = {
    "shrinkage": True,
    "hotspot": True,
    "fraction_liquid": True,
    "niyama": True,
    "solver_mode": "thermal_coupled",
}
NO_MOLD = {"mold": {"auto_mold": 0, "auto_cores": 0}}
UNIFORM_MOLD = {"mold": {"mold_grading_levels": 0}}

#: A part is (CI model file, part type) or (file, type, (x, y, z) position in inches).
Part = Tuple[Any, ...]

#: name -> (parts, voxel size in inches, solidification params, cut cells or None for the default)
CASES: Dict[str, Tuple[Sequence[Part], float, Dict[str, Any], Optional[bool]]] = {
    "plate_riser_030": ([("cast.stl", "cast"), ("riser.stl", "riser")], 0.30, {}, None),
    "plate_riser_015": ([("cast.stl", "cast"), ("riser.stl", "riser")], 0.15, {}, None),
    "plate_riser_030_boundary": ([("cast.stl", "cast"), ("riser.stl", "riser")], 0.30, NO_MOLD, None),
    "plate_riser_030_nocut": ([("cast.stl", "cast"), ("riser.stl", "riser")], 0.30, {}, False),
    "lbracket_020_uniform_mold": ([("l_bracket.stl", "cast")], 0.20, UNIFORM_MOLD, None),
    "thickthin_020_boundary_nocut": ([("thick_thin_test.stl", "cast")], 0.20, NO_MOLD, False),
    # 2 in block, chill on the +x face, sand core on the -x face, riser on top.
    "block_chill_core_riser_010": (
        [
            ("block_2inch.stl", "cast"),
            ("small_block.stl", "chill", (1.5, 0.0, 0.5)),
            ("small_block.stl", "core", (-1.5, 0.0, 0.5)),
            ("cylinder.stl", "riser", (0.0, 0.0, 1.0)),
        ],
        0.10,
        {},
        None,
    ),
    "block_chill_core_010_boundary": (
        [
            ("block_2inch.stl", "cast"),
            ("small_block.stl", "chill", (1.5, 0.0, 0.5)),
            ("small_block.stl", "core", (-1.5, 0.0, 0.5)),
        ],
        0.10,
        NO_MOLD,
        None,
    ),
    "block_chill_010_boundary_nocut": (
        [("block_2inch.stl", "cast"), ("small_block.stl", "chill", (1.5, 0.0, 0.5))],
        0.10,
        NO_MOLD,
        False,
    ),
    "block_chill_riser_015_uniform_mold_nocut": (
        [
            ("block_2inch.stl", "cast"),
            ("small_block.stl", "chill", (1.5, 0.0, 0.5)),
            ("cylinder.stl", "riser", (0.0, 0.0, 1.0)),
        ],
        0.15,
        UNIFORM_MOLD,
        False,
    ),
}


def _numeric_arrays(obj: Any, prefix: str) -> Dict[str, np.ndarray]:
    """Every non-empty numeric array attribute of ``obj``."""
    found: Dict[str, np.ndarray] = {}
    for name, value in vars(obj).items():
        if not (hasattr(value, "dtype") and hasattr(value, "shape")):
            continue
        array = np.asarray(value)
        if array.dtype.kind in "fiub" and array.size > 0:
            found[f"{prefix}.{name}"] = np.array(array)
    return found


def save(out_dir: str, names: Sequence[str]) -> int:
    from core.part_manager import build_part_manager
    from core.simulation_pipeline import PipelineInputs, run_pipeline
    from tests.helpers.ci_models import ci_model

    unknown = [name for name in names if name not in CASES]
    if unknown:
        print(f"Unknown case(s): {', '.join(unknown)}. Known: {', '.join(CASES)}")
        return 2
    os.makedirs(out_dir, exist_ok=True)
    for name, (parts, voxel_size, params, cut_cells) in CASES.items():
        if names and name not in names:
            continue
        specs = []
        for part in parts:
            spec = {"filepath": ci_model(part[0]), "part_type": part[1]}
            if len(part) > 2:
                spec.update(position_x=part[2][0], position_y=part[2][1], position_z=part[2][2])
            specs.append(spec)
        inputs = PipelineInputs(
            alloy_name="WCB",
            voxel_size=voxel_size,
            refinement_level=0,
            solidification_params=dict(params),
            analysis_options=dict(ANALYSES),
            show_progress=False,
            voxel_cut_cells=cut_cells,
        )
        started = time.perf_counter()
        result = run_pipeline(inputs, part_manager=build_part_manager(specs))
        wall = time.perf_counter() - started
        engine, mesh = result.engine, result.voxel_mesh

        arrays = _numeric_arrays(engine, "engine")
        arrays.update(_numeric_arrays(mesh, "mesh"))
        heat_model = getattr(engine, "_thermal_heat_model", None)
        if heat_model is not None:
            arrays["heat.temperatures"] = np.array(heat_model.temperatures)
            if hasattr(heat_model, "mold_temperatures"):
                arrays["heat.mold_temperatures"] = np.array(heat_model.mold_temperatures)
        history = getattr(engine, "fraction_liquid_history", None)
        if history:
            arrays["engine.fraction_liquid_history_stack"] = np.stack([np.asarray(frame) for frame in history])
            arrays["engine.fraction_liquid_times_arr"] = np.asarray(engine.fraction_liquid_times, dtype=np.float64)
        for key, value in (result.result_data or {}).items():
            if isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(value, bool):
                arrays[f"result.{key}"] = np.asarray(float(value))

        np.savez(os.path.join(out_dir, name + ".npz"), **arrays)
        print(
            f"SNAPSHOT {name}: voxels={len(mesh.voxel_states)} arrays={len(arrays)} "
            f"simulated={engine.time:.3f}s wall={wall:.2f}s",
            flush=True,
        )
    return 0


def _relative_gap(x: np.ndarray, y: np.ndarray) -> float:
    """Largest |x - y| / max(|x|, |y|) over the entries that are finite in both."""
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    finite = np.isfinite(x64) & np.isfinite(y64)
    if not np.any(finite):
        return 0.0
    scale = np.maximum(np.maximum(np.abs(x64[finite]), np.abs(y64[finite])), 1e-300)
    return float(np.max(np.abs(x64[finite] - y64[finite]) / scale))


def compare(dir_a: str, dir_b: str, rtol: float = 0.0, atol: float = 0.0) -> int:
    files_a = {f for f in os.listdir(dir_a) if f.endswith(".npz")}
    files_b = {f for f in os.listdir(dir_b) if f.endswith(".npz")}
    differences = 0
    for missing in sorted(files_a ^ files_b):
        print(f"{missing}: only in {dir_a if missing in files_a else dir_b}")
        differences += 1
    if not (files_a & files_b):
        print("No case is in both folders.")
        return 1
    for filename in sorted(files_a & files_b):
        a, b = np.load(os.path.join(dir_a, filename)), np.load(os.path.join(dir_b, filename))
        keys_a, keys_b = set(a.files), set(b.files)
        found = []
        worst_gap, worst_key = 0.0, ""
        for key in sorted(keys_a | keys_b):
            if key not in keys_a or key not in keys_b:
                found.append(f"{key}: only in {dir_a if key in keys_a else dir_b}")
                continue
            x, y = a[key], b[key]
            if x.shape != y.shape or x.dtype != y.dtype:
                found.append(f"{key}: {x.dtype}{x.shape} against {y.dtype}{y.shape}")
            elif not np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
                if x.dtype.kind == "f" and rtol > 0.0:
                    gap = _relative_gap(x, y)
                    if gap > worst_gap:
                        worst_gap, worst_key = gap, key
                    allowed = max(rtol, 1e-6) if x.dtype == np.float32 else rtol
                    # Same NaN and inf pattern, and the finite entries within tolerance.
                    if np.array_equal(np.isfinite(x), np.isfinite(y)) and np.allclose(
                        x.astype(np.float64), y.astype(np.float64), rtol=allowed, atol=atol, equal_nan=True
                    ):
                        continue
                if x.dtype.kind == "f":
                    unequal = ~((x == y) | (np.isnan(x) & np.isnan(y)))
                    gap = np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))
                    found.append(f"{key}: {int(unequal.sum())} of {x.size} differ, largest gap {gap:.3e}")
                else:
                    found.append(f"{key}: {int(np.sum(x != y))} of {x.size} differ")
        differences += len(found)
        note = f", largest relative gap {worst_gap:.2e} ({worst_key})" if rtol > 0.0 and worst_key else ""
        print(f"{filename}: {len(keys_a & keys_b)} arrays compared, {len(found)} differ{note}")
        for line in found[:25]:
            print("    " + line)
    passed = "WITHIN TOLERANCE" if rtol > 0.0 else "IDENTICAL"
    print(f"RESULT: {passed}" if differences == 0 else f"RESULT: {differences} DIFFERENCES")
    return 0 if differences == 0 else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    save_parser = commands.add_parser("save", help="run the cases and save their arrays")
    save_parser.add_argument("out_dir")
    save_parser.add_argument("cases", nargs="*", help="case names (default: all)")
    compare_parser = commands.add_parser("compare", help="compare two saved folders bit for bit")
    compare_parser.add_argument("dir_a")
    compare_parser.add_argument("dir_b")
    compare_parser.add_argument("--rtol", type=float, default=0.0, help="accept float arrays within this relative tolerance")
    compare_parser.add_argument("--atol", type=float, default=0.0, help="absolute tolerance added to --rtol")
    args = parser.parse_args(argv)
    if args.command == "save":
        return save(args.out_dir, args.cases)
    return compare(args.dir_a, args.dir_b, rtol=args.rtol, atol=args.atol)


if __name__ == "__main__":
    sys.exit(main())
