"""Surrogate simulation pipeline.

This pipeline replaces the traditional time-integration loop with a single
neural/inference forward pass.  It reuses the exact geometry preparation and
result packaging from ``core.simulation_pipeline`` and drives the *same*
post-analysis engine, so the surrogate produces a fully compatible
:class:`PipelineResult` with all core defect analyses (shrinkage, porosity,
hotspot, Niyama, sand burn-in, gas porosity, fraction-liquid playback).

The surrogate only predicts the solidification *schedule* fields
(start/end/local times); the final ``voxel_states`` and fraction-liquid
history are synthesized with the existing schedule machinery, and
``run_post_analyses`` derives every downstream field from them.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from core.analysis_strategies import run_post_analyses
from core.materials import AlloyProperties, get_alloy
from core.filling_stage import maybe_run_filling
from core.simulation_pipeline import (
    PipelineInputs,
    PipelineResult,
    _check_cancel,
    _emit_message,
    build_visualization_payload,
    create_analysis_options,
    prepare_mesh_and_geometry,
)
from core.solidification import SOLVER_MODE_SCHEDULE_LEGACY, SolidificationEngine
from core.part_manager import is_chill
from core.solidification_loop import build_schedule_fraction_history, compute_schedule_fraction_solid
from core.surrogate.conditioning import Conditioning, build_conditioning_vector
from core.surrogate.fields import encode_channels, unproject_grid
from core.surrogate.model import SurrogateModel
from config.exceptions import SimulationError

UNSUPPORTED_ANALYSES: Tuple[str, ...] = ("hot_tear", "feeding_path", "stress_analysis", "microstructure")


def _build_config(inputs: PipelineInputs, bounds_to_use: Any, analysis_options: Any) -> Dict[str, float]:
    """Extract the conditioning config scalars from pipeline inputs."""
    bounds = np.asarray(bounds_to_use, dtype=np.float64)
    if bounds.ndim == 2 and bounds.shape[1] == 2:
        max_dim = float(np.max(bounds[:, 1] - bounds[:, 0]))
    else:
        max_dim = 10.0

    params = analysis_options.solidification_params
    timing = getattr(params, "timing", None)
    feeding = getattr(params, "feeding", None)

    def _g(obj, name, default):
        if obj is None:
            return float(default)
        return float(getattr(obj, name, default) or default)

    return {
        "max_dim": max_dim,
        "voxel_size": float(inputs.voxel_size),
        "feeding_zone_vertical_bias": _g(feeding, "feeding_zone_vertical_bias", 0.0),
        "chvorinov_mold_constant": _g(timing, "chvorinov_mold_constant", 280.0),
        "safety_factor_no_risers": _g(timing, "safety_factor_no_risers", 1.5),
        "safety_factor_with_risers": _g(timing, "safety_factor_with_risers", 1.8),
        "min_simulation_time": _g(timing, "min_simulation_time", 120.0),
        "max_simulation_time": _g(timing, "max_simulation_time", 14400.0),
    }


def _apply_schedule_to_engine(
    engine: SolidificationEngine,
    start: np.ndarray,
    local: np.ndarray,
) -> None:
    """Write the predicted schedule into the engine and synthesize states."""
    n_voxels = len(start)
    part_types = getattr(engine.mesh, "voxel_part_types", None)
    if part_types is None:
        part_types = np.zeros(n_voxels, dtype=np.int32)
    part_types = np.asarray(part_types, dtype=np.int32)
    chill_mask = is_chill(part_types)

    active = (local > 0.0) & ~chill_mask

    start = np.asarray(start, dtype=np.float32).copy()
    local = np.asarray(local, dtype=np.float32).copy()

    start[chill_mask] = 0.0
    local[chill_mask] = 0.0
    engine.mesh.voxel_states[chill_mask] = 1.0

    if np.any(active):
        # The timeline is defined by the schedule itself (start + local),
        # matching how every fraction-solid computation and the traditional
        # pipeline derive completion.  Trusting the predicted end field
        # directly lets a stray end head inflate the fraction-liquid window
        # (and solidification_times) far past the true completion time.
        completion = start + local
        final_time = float(np.max(completion[active]))
    else:
        final_time = float(engine.max_time or 0.0)
    engine.time = final_time

    engine.solidification_start_times[:] = start
    engine.solidification_end_times[:] = completion
    engine.solidification_times[:] = completion
    engine.local_solidification_time[:] = local

    pre_solid = np.asarray(engine.mesh.voxel_states, dtype=np.float32) >= 1.0
    engine.mesh.voxel_states[:] = compute_schedule_fraction_solid(
        start, local, final_time, pre_solid
    )

    critical_fs = float(getattr(engine.alloy, "critical_fraction_solid", 0.5))
    feeding = engine.feeding_fraction_solid
    if feeding is not None:
        feeding[:] = np.nan
        feeding[active] = critical_fs

    if engine._should_capture_fraction_history():
        history, times = build_schedule_fraction_history(
            start,
            local,
            pre_solid,
            final_time,
            curve_exponent=float(getattr(engine, "_schedule_curve_exponent", 2.0)),
        )
        engine.fraction_liquid_history = history
        engine.fraction_liquid_times = times


def _disable_unsupported_analyses(analysis_options: Any) -> list:
    """Force unsupported analyses off and report which were disabled."""
    disabled = []
    for name in UNSUPPORTED_ANALYSES:
        if getattr(analysis_options, name, False):
            setattr(analysis_options, name, False)
            disabled.append(name)
    return disabled


def run_surrogate_pipeline(
    inputs: PipelineInputs,
    *,
    model: Optional[SurrogateModel] = None,
    part_manager: Any = None,
    stl_loader: Any = None,
    progress_message_callback=None,
    detailed_progress_callback=None,
    cancel_callback=None,
) -> PipelineResult:
    """
    Execute the surrogate solidification pipeline.

    Args:
        inputs: Same :class:`PipelineInputs` used by the traditional pipeline.
        model: A loaded :class:`SurrogateModel` (required).

    Returns:
        A fully compatible :class:`PipelineResult`.
    """
    if model is None:
        raise SimulationError(
            "No surrogate model loaded",
            phase="surrogate",
            suggestion="Load a surrogate model from the Surrogate menu first.",
        )

    _check_cancel(cancel_callback)
    _emit_message(progress_message_callback, f"Loading material properties: {inputs.alloy_name}")
    alloy = get_alloy(inputs.alloy_name)

    mesh_to_use, bounds_to_use, total_volume, parts_with_types, voxel_mesh, mem_stats = prepare_mesh_and_geometry(
        inputs,
        part_manager=part_manager,
        stl_loader=stl_loader,
        progress_message_callback=progress_message_callback,
        detailed_progress_callback=detailed_progress_callback,
        cancel_callback=cancel_callback,
    )

    _emit_message(
        progress_message_callback,
        f"Preparing surrogate model: {model.manifest.name} ({model.manifest.backend})",
    )

    analysis_options = create_analysis_options(
        inputs.analysis_options,
        use_gpu=False,
        use_incremental=False,
        solidification_params=inputs.solidification_params,
    )
    analysis_options.solver_mode = SOLVER_MODE_SCHEDULE_LEGACY
    disabled_analyses = _disable_unsupported_analyses(analysis_options)


    filling_result = maybe_run_filling(
        voxel_mesh,
        alloy,
        analysis_options,
        inputs,
        cancel_callback=cancel_callback,
        progress_message_callback=progress_message_callback,
    )

    engine = SolidificationEngine(voxel_mesh, alloy, analysis_options)
    engine.initialize()
    if inputs.thread_budget_provider is not None:
        engine._thread_budget_provider = inputs.thread_budget_provider
    if filling_result is not None:
        engine.set_initial_temperature_field(
            filling_result.temperature_c,
            fill_arrival_times=filling_result.fill_time_s,
        )

    # Geometry-derived input channels (no physics loop required).
    engine._calculate_feeding_distances()
    chill_boost = engine._calculate_chill_influence()
    riser_slowdown = engine._calculate_riser_influence()
    engine._chill_boost = chill_boost
    engine._air_gap_slowdown = np.ones(len(voxel_mesh.voxel_centers), dtype=np.float32)

    config = _build_config(inputs, bounds_to_use, analysis_options)
    conditioning = build_conditioning_vector(alloy, config)
    resolution = int(getattr(model.manifest, "grid_resolution", 64) or 64)

    _emit_message(
        progress_message_callback,
        f"Running surrogate inference ({model.manifest.backend} backend)...",
    )
    channels, meta = encode_channels(engine, chill_boost, riser_slowdown, resolution=resolution)
    prediction = model.predict(channels, conditioning)

    centers = np.asarray(voxel_mesh.voxel_centers, dtype=np.float64)
    start = unproject_grid(prediction[0], centers, meta)
    local = unproject_grid(prediction[2], centers, meta)

    _emit_message(progress_message_callback, "Running post analyses...")
    _apply_schedule_to_engine(engine, start, local)
    engine.stage_timings = {}
    run_post_analyses(engine, show_progress=inputs.show_progress)
    _check_cancel(cancel_callback)

    result_data, visualization_data = build_visualization_payload(
        engine,
        voxel_mesh,
        total_volume=total_volume,
        bounds=np.asarray(bounds_to_use),
        max_voxels=inputs.max_voxels_for_visualization,
    )

    engine.surrogate_run = True
    engine.surrogate_model_name = model.manifest.name
    engine.surrogate_backend = model.manifest.backend
    engine.surrogate_confidence = float(model.manifest.accuracy.get("confidence", 0.0) or 0.0)
    engine.surrogate_disabled_analyses = disabled_analyses

    return PipelineResult(
        part_manager=part_manager,
        voxel_mesh=voxel_mesh,
        engine=engine,
        result_data=result_data,
        visualization_data=visualization_data,
    )