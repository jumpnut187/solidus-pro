#!/usr/bin/env python3
"""Voxel with and without cut cells against a fine tet reference, through ``run_pipeline``.

Cases: ``sample_stl_files/thick_thin_test.stl``, ``sample_stl_files/l_bracket.stl``
and the box + riser + chill assembly of ``tests/test_tet_end_to_end.py``.
Each case runs voxel at ``--sizes`` with ``settings.mesh.voxel_cut_cells``
off and on, and tet at ``--reference-edge`` (0.05 in) as the reference.
Columns: wall time, points, metal volume vs CAD, total solidification time
(largest finite time over metal points) vs the reference, field error
(``tools/benchmark_tet_vs_voxel.field_error``: mean |t - t_ref| at reference
metal nodes, % of the reference total) and the cast hot spot (metal hot spot
for single-STL cases) with its distance from the reference hot spot.

``--c05813 DIR`` runs the owner's 17-part model instead (parts typed with
``infer_part_type_from_filename``), voxel only, default analyses, and
reports peak RSS and metal volume per part code.

Tet references are cached as ``.npz`` in ``--cache-dir``.

Usage::

    venv/bin/python tools/benchmark_cut_cells.py
    venv/bin/python tools/benchmark_cut_cells.py --cases thick_thin --sizes 0.1
    venv/bin/python tools/benchmark_cut_cells.py --c05813 "/path/C05813 Rigging" --sizes 0.2 0.1 --cut-cells on
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os
import resource
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import trimesh  # noqa: E402

from config.settings import get_settings  # noqa: E402
from core import simulation_pipeline as sp  # noqa: E402
from core.part_manager import (  # noqa: E402
    PART_TYPE_CAST,
    PART_TYPE_TO_CODE,
    Part,
    PartManager,
    PartType,
    infer_part_type_from_filename,
    is_metal,
)
from core.stl_loader import STLLoader  # noqa: E402
from tools.benchmark_tet_vs_voxel import DESKTOP_ANALYSES, Run, field_error  # noqa: E402

STL_CASES = {
    "thick_thin": "sample_stl_files/thick_thin_test.stl",
    "l_bracket": "sample_stl_files/l_bracket.stl",
}
CASES = (*STL_CASES, "assembly")


@dataclass
class CaseResult:
    run: Run
    cut_cells: Optional[bool]
    metal_volume: float
    hot_spot: np.ndarray
    code_volumes: Dict[int, float]
    peak_rss_gb: float = 0.0


def _add_part(manager: PartManager, name: str, mesh: trimesh.Trimesh, part_type: PartType, path: str = "") -> None:
    part = Part(id=name, name=name, filepath=path or f"{name}.stl", trimesh_obj=mesh)
    part.set_part_type(part_type)
    manager.parts[name] = part
    manager._part_order.append(name)


def assembly_manager() -> PartManager:
    """2 x 2 x 1 in casting, 1.2 in dia riser on top, 1 x 1 x 0.4 in chill (tests/test_tet_end_to_end.py)."""
    manager = PartManager()
    cast = trimesh.creation.box(extents=(2.0, 2.0, 1.0))
    cast.apply_translation((0.0, 0.0, 0.5))
    riser = trimesh.creation.cylinder(radius=0.6, height=1.2, sections=48)
    riser.apply_translation((0.0, 0.0, 1.6))
    chill = trimesh.creation.box(extents=(1.0, 1.0, 0.4))
    chill.apply_translation((0.5, 0.0, -0.2))
    _add_part(manager, "cast", cast, PartType.CAST)
    _add_part(manager, "riser", riser, PartType.RISER)
    _add_part(manager, "chill", chill, PartType.CHILL)
    return manager


def model_manager(directory: str) -> Tuple[PartManager, Dict[int, float]]:
    """All STLs in ``directory`` typed by file name; returns CAD volume per code."""
    manager = PartManager()
    cad: Dict[int, float] = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.stl"))):
        mesh = trimesh.load(path)
        name = os.path.basename(path)
        part_type = infer_part_type_from_filename(name)
        _add_part(manager, name, mesh, part_type, path)
        code = int(PART_TYPE_TO_CODE[part_type])
        cad[code] = cad.get(code, 0.0) + float(mesh.volume)
    return manager, cad


def case_sources(case: str) -> dict:
    if case == "assembly":
        return {"part_manager": assembly_manager()}
    loader = STLLoader()
    with contextlib.redirect_stdout(io.StringIO()):
        if not loader.load_stl(os.path.join(REPO_ROOT, STL_CASES[case])):
            raise RuntimeError(f"could not load {STL_CASES[case]}")
    return {"stl_loader": loader}


def cad_metal_volume(case: str) -> float:
    if case == "assembly":
        manager = assembly_manager()
        return float(sum(abs(p.trimesh_obj.volume) for p in manager.parts.values() if p.part_type != PartType.CHILL))
    return float(trimesh.load(os.path.join(REPO_ROOT, STL_CASES[case])).volume)


def run_one(
    label: str,
    sources: dict,
    mesh_type: str,
    size_in: float,
    *,
    cut_cells: Optional[bool],
    edge_factor: float = 1.0,
    analyses: Optional[dict] = None,
    quiet: bool = True,
    chunked: bool = False,
) -> CaseResult:
    mesh_settings = get_settings().mesh
    saved = (
        mesh_settings.tet_edge_length_factor,
        getattr(mesh_settings, "voxel_cut_cells", False),
        getattr(mesh_settings, "tet_auto_fallback", False),
    )
    mesh_settings.tet_edge_length_factor = edge_factor
    mesh_settings.voxel_cut_cells = bool(cut_cells)
    # A reference must be a tet run: no automatic fallback to voxel.
    mesh_settings.tet_auto_fallback = False
    try:
        inputs = sp.PipelineInputs(
            "WCB", size_in, 0, analysis_options=None if analyses is None else dict(analyses),
            use_gpu=True, mesh_type=mesh_type, use_chunked_voxelization=chunked,
        )
        started = time.perf_counter()
        sink = io.StringIO() if quiet else sys.stdout
        with contextlib.redirect_stdout(sink):
            result = sp.run_pipeline(inputs, **sources)
        wall_s = time.perf_counter() - started
    finally:
        (
            mesh_settings.tet_edge_length_factor,
            mesh_settings.voxel_cut_cells,
            mesh_settings.tet_auto_fallback,
        ) = saved
    mesh = result.voxel_mesh
    times = np.asarray(result.engine.solidification_times, dtype=np.float64)
    part_types = getattr(mesh, "voxel_part_types", None)
    codes = np.zeros(len(times), dtype=np.int32) if part_types is None else np.asarray(part_types, dtype=np.int32)
    metal = is_metal(codes)
    centers = np.asarray(mesh.voxel_centers, dtype=np.float64)
    if mesh_type == "tet":
        volumes = np.zeros(len(times))
        tet_codes = np.asarray(mesh.tet_part_types)
        code_volumes = {int(c): float(np.sum(mesh.tet_volumes[tet_codes == c])) for c in np.unique(tet_codes)}
        metal_volume = float(sum(v for c, v in code_volumes.items() if is_metal(np.array([c]))[0]))
    else:
        volumes = np.asarray(mesh.voxel_volumes, dtype=np.float64)
        code_volumes = {int(c): float(np.sum(volumes[codes == c])) for c in np.unique(codes)}
        metal_volume = float(np.sum(volumes[metal]))
    cast = np.flatnonzero(codes == PART_TYPE_CAST)
    pool = cast if len(cast) else np.flatnonzero(metal)
    hot = centers[pool[np.nanargmax(times[pool])]]
    run = Run(
        part=label,
        mesh_type=mesh_type,
        size_in=size_in * edge_factor if mesh_type == "tet" else size_in,
        factor=edge_factor if mesh_type == "tet" else None,
        points=int(mesh.get_voxel_count()),
        wall_s=wall_s,
        total_s=float(np.nanmax(times[metal])),
        metal_xyz=centers[metal],
        metal_times=times[metal],
    )
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
    return CaseResult(run, cut_cells, metal_volume, hot, code_volumes, rss_gb)


def reference(
    case: str, edge_in: float, cache_dir: str, max_dt_s: Optional[float] = None, max_dtemp_c: Optional[float] = None
) -> CaseResult:
    """Tet run at ``edge_in``, cached. ``max_dt_s``/``max_dtemp_c`` override the tet step limits."""
    suffix = "" if max_dt_s is None else f"_dt{max_dt_s:g}_{max_dtemp_c:g}"
    path = os.path.join(cache_dir, f"tet_{case}_{edge_in:g}{suffix}.npz")
    if os.path.exists(path):
        data = np.load(path)
        run = Run(case, "tet", edge_in, 1.0, int(data["points"]), float(data["wall_s"]), float(data["total_s"]),
                  data["metal_xyz"], data["metal_times"])
        return CaseResult(run, None, float(data["metal_volume"]), data["hot_spot"], {})
    mesh_settings = get_settings().mesh
    saved = (mesh_settings.tet_max_time_step_s, mesh_settings.tet_max_delta_t_c)
    if max_dt_s is not None:
        mesh_settings.tet_max_time_step_s = max_dt_s
        mesh_settings.tet_max_delta_t_c = max_dtemp_c
    try:
        result = run_one(case, case_sources(case), "tet", edge_in, cut_cells=None, analyses=DESKTOP_ANALYSES)
    finally:
        mesh_settings.tet_max_time_step_s, mesh_settings.tet_max_delta_t_c = saved
    os.makedirs(cache_dir, exist_ok=True)
    r = result.run
    np.savez(path, points=r.points, wall_s=r.wall_s, total_s=r.total_s, metal_xyz=r.metal_xyz,
             metal_times=r.metal_times, metal_volume=result.metal_volume, hot_spot=result.hot_spot)
    return result


def _fmt_xyz(xyz: np.ndarray) -> str:
    return "(" + ", ".join(f"{v:.2f}" for v in xyz) + ")"


def benchmark_cases(args: argparse.Namespace) -> int:
    run_one("warmup", case_sources("l_bracket"), "voxel", 0.2, cut_cells=True, analyses=DESKTOP_ANALYSES)
    run_one("warmup", case_sources("l_bracket"), "voxel", 0.2, cut_cells=False, analyses=DESKTOP_ANALYSES)
    print("| case | mesh | size in | cut cells | points | wall s | metal in3 | vs CAD | total s | vs ref | field err | hot spot in | hot spot dist in |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for case in args.cases:
        ref = reference(case, args.reference_edge, args.cache_dir, args.reference_max_dt, args.reference_max_dtemp)
        cad = cad_metal_volume(case)
        rows: List[CaseResult] = [ref]
        for size in args.sizes:
            for flag in args.cut_cells:
                rows.append(run_one(case, case_sources(case), "voxel", size, cut_cells=flag,
                                    analyses=DESKTOP_ANALYSES, chunked=args.chunked))
        for res in rows:
            r = res.run
            flag = "-" if res.cut_cells is None else ("on" if res.cut_cells else "off")
            err = 0.0 if res is ref else field_error(r, ref.run)
            dist = float(np.linalg.norm(res.hot_spot - ref.hot_spot))
            print(f"| {case} | {r.mesh_type} | {r.size_in:g} | {flag} | {r.points} | {r.wall_s:.1f} | "
                  f"{res.metal_volume:.3f} | {100 * (res.metal_volume / cad - 1):+.1f} % | {r.total_s:.1f} | "
                  f"{100 * (r.total_s / ref.run.total_s - 1):+.1f} % | {err:.2f} % | {_fmt_xyz(res.hot_spot)} | {dist:.2f} |",
                  flush=True)
    return 0


def benchmark_model(args: argparse.Namespace) -> int:
    manager, cad = model_manager(args.c05813)
    for size in args.sizes:
        for flag in args.cut_cells:
            res = run_one("C05813", {"part_manager": manager}, "voxel", size, cut_cells=flag,
                          quiet=not args.verbose, chunked=args.chunked)
            r = res.run
            record = {
                "size_in": size,
                "cut_cells": flag,
                "points": r.points,
                "wall_s": round(r.wall_s, 1),
                "peak_rss_gb": round(res.peak_rss_gb, 2),
                "total_s": round(r.total_s, 1),
                "vs_tet_0.2": f"{100 * (r.total_s / args.tet_total - 1):+.1f} %",
                "metal_in3": round(res.metal_volume, 2),
                "code_volumes_in3": {c: round(v, 2) for c, v in sorted(res.code_volumes.items())},
                "cad_in3": {c: round(v, 2) for c, v in sorted(cad.items())},
                "cast_hot_spot_in": [round(float(v), 2) for v in res.hot_spot],
            }
            print("MODEL " + json.dumps(record), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    parser.add_argument("--sizes", nargs="+", type=float, default=[0.2, 0.1, 0.05])
    parser.add_argument("--cut-cells", nargs="+", choices=["off", "on"], default=["off", "on"])
    parser.add_argument("--reference-edge", type=float, default=0.05, help="tet edge of the reference, in")
    parser.add_argument("--reference-max-dt", type=float, default=0.25,
                        help="tet max time step for the reference, s (default steps 2 s / 15 C are not dt-converged)")
    parser.add_argument("--reference-max-dtemp", type=float, default=2.0, help="tet max |dT| per step for the reference, C")
    parser.add_argument("--cache-dir", default=os.path.join(REPO_ROOT, "logs", "cut_cell_refs"))
    parser.add_argument("--c05813", default=None, help="directory of the owner's STL parts")
    parser.add_argument("--tet-total", type=float, default=748.6, help="tet total solidification time for --c05813, s")
    parser.add_argument("--chunked", action="store_true", help="use the chunked centre-containment voxelizer")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    args.cut_cells = [flag == "on" for flag in args.cut_cells]
    if args.c05813:
        return benchmark_model(args)
    return benchmark_cases(args)


if __name__ == "__main__":
    raise SystemExit(main())
