"""Post-solidification analysis strategy execution."""

from __future__ import annotations

from typing import Any

import numpy as np
import time


def run_post_analyses(engine: Any, *, show_progress: bool) -> None:
    """
    Execute analysis stages in dependency order and update engine stats.
    """
    stage_timings = getattr(engine, "stage_timings", None)

    def _record(stage_name: str, started_at: float) -> None:
        if isinstance(stage_timings, dict):
            stage_timings[stage_name] = stage_timings.get(stage_name, 0.0) + (time.perf_counter() - started_at)

    if engine.analysis_options.feeding_path:
        if show_progress:
            print("  Analyzing feeding paths...")
        stage_start = time.perf_counter()
        try:
            if engine._feeding_path_available:
                engine._calculate_feeding_paths()
            elif show_progress:
                print("  Warning: Feeding path analysis module not available")
        except Exception as exc:
            engine._logger.log_simulation_error(
                stage="feeding_path_analysis",
                error=exc,
                mesh_info={"n_voxels": len(engine.mesh.voxel_centers)},
            )
            if show_progress:
                print(f"  Warning: Feeding path analysis failed: {exc}")
        finally:
            _record("analysis_feeding_path", stage_start)

    if engine.analysis_options.hotspot:
        if show_progress:
            print("  Calculating hot spots...")
        stage_start = time.perf_counter()
        engine._calculate_hotspots()
        _record("analysis_hotspots", stage_start)

    if engine.analysis_options.niyama or engine.analysis_options.hot_tear or engine.analysis_options.shrinkage:
        if show_progress:
            print("  Calculating Niyama criterion...")
        stage_start = time.perf_counter()
        engine._calculate_niyama()
        _record("analysis_niyama", stage_start)

    if engine.analysis_options.shrinkage:
        if show_progress:
            print("  Calculating shrinkage...")
        stage_start = time.perf_counter()
        engine._calculate_shrinkage()
        _record("analysis_shrinkage", stage_start)
        if show_progress:
            print("  Calculating porosity...")
        stage_start = time.perf_counter()
        engine._calculate_porosity()
        _record("analysis_porosity", stage_start)
        apply_shrink_scalar = getattr(engine, "_apply_shrink_scalar", None)
        if callable(apply_shrink_scalar):
            stage_start = time.perf_counter()
            apply_shrink_scalar()
            _record("analysis_shrink_scalar", stage_start)

    if engine.analysis_options.hot_tear:
        if show_progress:
            print("  Calculating hot tear risk zones...")
        stage_start = time.perf_counter()
        engine._calculate_hot_tear_risk()
        _record("analysis_hot_tear", stage_start)

    if engine.analysis_options.sand_burn_in:
        if show_progress:
            print("  Calculating sand burn-in risk...")
        stage_start = time.perf_counter()
        try:
            engine._calculate_sand_burn_in()
        except Exception as exc:
            engine._logger.log_simulation_error(
                stage="sand_burn_in_analysis",
                error=exc,
                mesh_info={
                    "n_voxels": len(engine.mesh.voxel_centers),
                    "alloy": engine.alloy.name,
                    "analysis_options": {"sand_burn_in": engine.analysis_options.sand_burn_in},
                },
            )
            if show_progress:
                print(f"  Warning: Sand burn-in analysis failed: {exc}")
            engine.sand_burn_in_risk = np.zeros(len(engine.mesh.voxel_centers), dtype=np.float32)
            engine.sand_burn_in_zones = np.zeros(len(engine.mesh.voxel_centers), dtype=np.bool_)
            engine.sand_burn_in_count = 0
        finally:
            _record("analysis_sand_burn_in", stage_start)

    if engine.analysis_options.stress_analysis:
        if show_progress:
            print("  Calculating thermal stress and distortion...")
        stage_start = time.perf_counter()
        try:
            if engine._stress_analysis_available:
                engine._calculate_stress_analysis()
            elif show_progress:
                print("  Warning: Stress analysis module not available")
        except Exception as exc:
            engine._logger.log_simulation_error(
                stage="stress_analysis",
                error=exc,
                mesh_info={"n_voxels": len(engine.mesh.voxel_centers)},
            )
            if show_progress:
                print(f"  Warning: Stress analysis failed: {exc}")
        finally:
            _record("analysis_stress", stage_start)

    if engine.analysis_options.microstructure:
        if show_progress:
            print("  Predicting microstructure (SDAS)...")
        stage_start = time.perf_counter()
        try:
            if engine._microstructure_available:
                engine._calculate_microstructure()
            elif show_progress:
                print("  Warning: Microstructure prediction module not available")
        except Exception as exc:
            engine._logger.log_simulation_error(
                stage="microstructure_prediction",
                error=exc,
                mesh_info={"n_voxels": len(engine.mesh.voxel_centers)},
            )
            if show_progress:
                print(f"  Warning: Microstructure prediction failed: {exc}")
        finally:
            _record("analysis_microstructure", stage_start)

    stage_start = time.perf_counter()
    engine._calculate_chvorinov_times()
    _record("analysis_chvorinov_times", stage_start)
    stage_start = time.perf_counter()
    engine._generate_statistics()
    _record("analysis_statistics", stage_start)

    if show_progress:
        print("\n" + "=" * 60)
        print("Simulation Results")
        print("=" * 60)
        print(f"Total simulation time: {engine.time:.1f}s")
        if engine.analysis_options.shrinkage:
            print(
                f"Voxels with shrinkage risk: {engine.affected_voxel_count:,} "
                f"({engine.affected_voxel_count / len(engine.mesh.voxel_centers) * 100:.1f}%)"
            )
            print(f"Estimated shrinkage volume: {engine.total_shrinkage_volume:.4f} in³")
        if engine.analysis_options.hotspot:
            print(f"Hot spots detected: {engine.hotspot_count:,}")
        if engine.analysis_options.niyama:
            threshold = engine._niyama_thresholds.get(engine.alloy.name.split()[0], 1.0)
            print(f"Niyama defect risk voxels (< {threshold}): {engine.niyama_defect_count:,}")
        if engine.analysis_options.hot_tear:
            print(f"Hot tear risk zones: {engine.hot_tear_count:,}")
        if engine.analysis_options.sand_burn_in:
            print(f"Sand burn-in risk zones: {engine.sand_burn_in_count:,}")
        print("=" * 60 + "\n")
