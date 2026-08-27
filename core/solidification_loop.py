"""Time-integration loop extracted from SolidificationEngine."""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional

import numpy as np
from config.hardware_config import hardware
from core.thread_budget import resolve_thread_budget

try:
    from numba import get_num_threads, jit, prange, set_num_threads

    NUMBA_THREADING_AVAILABLE = True
except ImportError:
    NUMBA_THREADING_AVAILABLE = False
    prange = range  # type: ignore[assignment]

try:
    from physics.solidification_physics import apply_enthalpy_correction_parallel

    ENTHALPY_KERNEL_AVAILABLE = True
except ImportError:
    ENTHALPY_KERNEL_AVAILABLE = False


def _jit(fn):  # type: ignore[no-redef]
    """JIT decorator that gracefully no-ops when numba is missing."""
    if NUMBA_THREADING_AVAILABLE:
        return jit(nopython=True, parallel=True, fastmath=True, cache=True)(fn)
    return fn


ProgressCallback = Optional[Callable[[int, int, str], None]]
DEFAULT_SCHEDULE_CURVE_EXPONENT = 2.0
THERMAL_COUPLED_SOLVER_MODE = "thermal_coupled"
NIYAMA_EVAL_FRACTION_SOLID = 0.90
THREAD_BUDGET_CHECK_INTERVAL = 2.0


def _resolve_solidification_thread_target() -> int:
    """
    Resolve Numba thread target for solidification kernels.

    Environment overrides:
    - CASTSIM_SOLIDIFICATION_THREADS=all|physical|logical|<int>
    - CASTSIM_NUMBA_THREADS=all|physical|logical|<int>
    """
    logical_cores = int(getattr(hardware, "logical_cores", 0) or (os.cpu_count() or 1))
    physical_cores = int(getattr(hardware, "physical_cores", 0) or logical_cores)
    max_threads = max(logical_cores, 1)

    raw_value = (
        os.getenv("CASTSIM_SOLIDIFICATION_THREADS")
        or os.getenv("CASTSIM_NUMBA_THREADS")
        or "all"
    ).strip().lower()

    if raw_value in ("all", "max", "logical"):
        target = logical_cores
    elif raw_value in ("physical", "phy"):
        target = physical_cores
    else:
        try:
            target = int(raw_value)
        except ValueError:
            target = logical_cores

    budget = resolve_thread_budget()
    if budget is not None:
        target = min(target, max(1, budget))

    return max(1, min(int(target), max_threads))


def _configure_numba_threads_for_solidification(engine: Any, *, show_progress: bool) -> None:
    """Set Numba thread count so solidification can use full CPU resources."""
    if not NUMBA_THREADING_AVAILABLE:
        return

    target_threads = _resolve_solidification_thread_target()
    set_num_threads(target_threads)
    active_threads = int(get_num_threads())
    setattr(engine, "_solidification_numba_threads", active_threads)
    if callable(getattr(engine, "_thread_budget_provider", None)):
        setattr(engine, "_active_thread_budget", resolve_thread_budget())

    logical_cores = int(getattr(hardware, "logical_cores", 0) or (os.cpu_count() or active_threads))
    physical_cores = int(getattr(hardware, "physical_cores", 0) or logical_cores)
    print(
        "  CPU threading: "
        f"Numba thread pool configured {active_threads}/{logical_cores} "
        f"(physical cores: {physical_cores})"
    )


def _apply_live_thread_budget(engine: Any) -> None:
    """
    Re-apply the fair-share thread budget if it changed since the last check.

    The budget is read through ``engine._thread_budget_provider`` (attached
    by the web pipeline). The provider updates the process-local budget and
    hardware cap; only the Numba thread count is re-applied here, from the
    loop thread, and only when the budget actually changed. Numba applies
    the new pool size at the next parallel region.
    """
    if not NUMBA_THREADING_AVAILABLE:
        return
    provider = getattr(engine, "_thread_budget_provider", None)
    if not callable(provider):
        return
    budget = provider()
    if budget is None:
        return
    if getattr(engine, "_active_thread_budget", None) == budget:
        return
    target_threads = _resolve_solidification_thread_target()
    try:
        set_num_threads(target_threads)
        active_threads = int(get_num_threads())
        setattr(engine, "_active_thread_budget", budget)
        setattr(engine, "_solidification_numba_threads", active_threads)
    except Exception:
        # A thread-pool resize hiccup must never abort a running simulation.
        pass


def _least_squares_gradients_c_per_mm(
    *,
    centers_in: np.ndarray,
    temperatures: np.ndarray,
    neighbors: np.ndarray,
    target_indices: np.ndarray,
    cast_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return local temperature-gradient magnitudes and quality flags.

    ``cast_mask`` selects which voxels are *evaluated*; neighbour temperatures
    are sampled from every in-mesh voxel regardless of part type. Chill, riser
    and core voxels are solved cells carrying valid temperatures, and at a chill
    face they hold nearly the whole local delta-T - dropping them would collapse
    the gradient exactly where it is steepest.
    """
    gradients = np.zeros(len(target_indices), dtype=np.float32)
    quality = np.zeros(len(target_indices), dtype=np.int8)

    for out_i, voxel_i in enumerate(target_indices):
        if not cast_mask[voxel_i]:
            continue

        neighbor_indices = neighbors[voxel_i]
        in_bounds = (neighbor_indices >= 0) & (neighbor_indices < len(centers_in))
        valid_neighbors = neighbor_indices[in_bounds]
        if len(valid_neighbors) == 0:
            continue

        offsets_mm = (centers_in[valid_neighbors] - centers_in[voxel_i]) * 25.4
        valid = np.linalg.norm(offsets_mm, axis=1) > 1e-9
        offsets_mm = offsets_mm[valid]
        valid_neighbors = valid_neighbors[valid]
        if len(valid_neighbors) == 0:
            continue

        delta_t = temperatures[valid_neighbors] - temperatures[voxel_i]
        try:
            grad_vec, _, rank, _ = np.linalg.lstsq(offsets_mm, delta_t, rcond=None)
            grad_mag = float(np.linalg.norm(grad_vec))
            if rank < 3:
                # The minimum-norm solution under-reports |grad T| along the
                # unsampled directions; fall back to the steepest sampled slope.
                slopes = np.abs(delta_t) / np.maximum(np.linalg.norm(offsets_mm, axis=1), 1e-9)
                grad_mag = max(grad_mag, float(np.max(slopes)))
            gradients[out_i] = grad_mag
            quality[out_i] = 2 if rank >= 3 and len(valid_neighbors) >= 4 else 1
        except np.linalg.LinAlgError:
            slopes = np.abs(delta_t) / np.maximum(np.linalg.norm(offsets_mm, axis=1), 1e-9)
            gradients[out_i] = float(np.max(slopes)) if len(slopes) else 0.0
            quality[out_i] = 1

    return gradients, quality


@_jit
def _feeding_zone_mask_kernel(
    states: np.ndarray,
    neighbors: np.ndarray,
    part_types: np.ndarray,
    out_mask: np.ndarray,
) -> None:
    """Fill ``out_mask`` with True for voxels inside the riser feeding zone.

    A voxel is in the feeding zone when it is itself a riser or any of its
    liquid neighbors is a riser. The legacy
    ``acceleration.update_solidification_with_feeding_connectivity`` rule also
    checks ``riser_slowdown < 0.6`` for propagated connectivity; the
    thermal-coupled path does not carry per-voxel riser_slowdown arrays, so
    only the part-type rule is reproduced here.
    """
    RISER = np.int32(1)
    n_voxels = states.shape[0]
    for i in prange(n_voxels):
        in_zone = False
        if part_types[i] == RISER:
            in_zone = True
        else:
            for j in range(neighbors.shape[1]):
                n_idx = neighbors[i, j]
                if n_idx < 0:
                    continue
                if (
                    states[n_idx] < 1.0
                    and part_types[n_idx] == RISER
                ):
                    in_zone = True
                    break
        out_mask[i] = in_zone


@_jit
def _directional_solidification_delta_kernel(
    states: np.ndarray,
    heights: np.ndarray,
    feeding_zone_mask: np.ndarray,
    feeding_bias: float,
    baseline: float,
    base_rate: float,
    dt: float,
    out_delta: np.ndarray,
) -> None:
    """Compute per-voxel directional-solidification delta in parallel.

    Writes to ``out_delta``:

        extra[i] = min(1 - states[i],
                       (1 - heights[i]) * factor * base_rate * dt)

    where ``factor = max(feeding_bias, baseline)`` inside the feeding zone
    and ``factor = baseline`` elsewhere. ``out_delta[i] = 0`` once the voxel
    is fully solid.
    """
    n_voxels = states.shape[0]
    fz_strength = max(feeding_bias, baseline)
    for i in prange(n_voxels):
        if states[i] >= 1.0:
            out_delta[i] = 0.0
            continue
        factor = fz_strength if feeding_zone_mask[i] else baseline
        raw = (1.0 - heights[i]) * factor * base_rate * dt
        cap = 1.0 - states[i]
        if raw < 0.0:
            raw = 0.0
        elif raw > cap:
            raw = cap
        out_delta[i] = raw


# Public helpers exposed for tests; thin Python wrappers around the JIT
# kernels above to keep test code ergonomic.
def _compute_feeding_zone_mask(
    voxel_states: np.ndarray,
    voxel_neighbors: np.ndarray,
    part_types: np.ndarray,
) -> np.ndarray:
    """Return a boolean mask of voxels inside the riser feeding zone.

    JIT-compiled kernel; see ``_feeding_zone_mask_kernel`` for semantics.
    """
    out_mask = np.zeros(voxel_states.shape[0], dtype=np.bool_)
    states_f32 = np.ascontiguousarray(voxel_states, dtype=np.float32)
    neighbors_i32 = np.ascontiguousarray(voxel_neighbors, dtype=np.int32)
    part_types_i32 = np.ascontiguousarray(part_types, dtype=np.int32)
    _feeding_zone_mask_kernel(
        states_f32, neighbors_i32, part_types_i32, out_mask
    )
    return out_mask


# Baseline directional-solidification strength applied to EVERY voxel so the
# casting solidifies from bottom to top by default. Matches the
# `vertical_factor = 1 + (1 - h) * 0.3` constant in the legacy front-based
# solver at ``core.acceleration.update_solidification_with_feeding_connectivity``.
FEEDING_ZONE_BASELINE_VERTICAL_BIAS = 0.3
DVR_RATE = 0.02


def _resolve_feeding_zone_bias(engine: Any) -> float:
    """Read ``feeding_zone_vertical_bias`` defensively from ``engine``."""
    params = getattr(getattr(engine, "analysis_options", None), "solidification_params", None)
    if params is None:
        return 0.0
    feeding = getattr(params, "feeding", None)
    if feeding is None:
        return 0.0
    try:
        return float(feeding.feeding_zone_vertical_bias)
    except (TypeError, ValueError, AttributeError):
        return 0.0


def _apply_feeding_zone_bias(
    engine: Any,
    solidification_model: Any,
    feeding_bias: float,
) -> bool:
    """Apply directional-solidification bias in the thermal-coupled path.

    Always applies a baseline vertical bias
    (``FEEDING_ZONE_BASELINE_VERTICAL_BIAS``) to every liquid voxel so the
    casting still solidifies bottom-up when no risers are present. In
    addition, voxels inside the riser feeding zone receive an extra
    directional bias controlled by the user-facing
    ``feeding_zone_vertical_bias`` slider. The effective per-voxel
    extra fraction-solid delta matches the legacy front-based formulation:

        extra = (1 - normalized_height) * factor * DVR_RATE * dt

    where ``factor`` is ``FEEDING_ZONE_BASELINE_VERTICAL_BIAS`` outside the
    feeding zone and ``max(feeding_bias, FEEDING_ZONE_BASELINE_VERTICAL_BIAS)``
    inside it.

    Mutates ``solidification_model._fraction_solid`` and
    ``engine.mesh.voxel_states`` so the bias compounds across iterations
    rather than being overwritten by the next temperature-driven
    ``SolidificationModel.update`` call.

    Returns:
        ``True`` when the bias produced a state change; ``False`` otherwise.
    """
    baseline = float(FEEDING_ZONE_BASELINE_VERTICAL_BIAS)
    states = np.asarray(solidification_model._fraction_solid, dtype=np.float32)
    n_voxels = states.shape[0]
    if n_voxels == 0:
        return False
    if baseline <= 0.0 and float(feeding_bias) <= 0.0:
        return False
    heights_fn = getattr(engine, "_get_normalized_heights", None)
    if not callable(heights_fn):
        return False
    # Cached heights array avoids the per-call O(n) z-min/z-max scan when the
    # integration loop calls us every time step.
    heights_cache = getattr(engine, "_directional_bias_heights", None)
    if heights_cache is None or heights_cache.shape[0] != n_voxels:
        heights_cache = np.ascontiguousarray(heights_fn(), dtype=np.float32)
        engine._directional_bias_heights = heights_cache
    heights = heights_cache

    feeding_zone_mask = None
    part_types_raw = getattr(engine.mesh, "voxel_part_types", None)
    if part_types_raw is not None and float(feeding_bias) > 0.0:
        part_types = np.ascontiguousarray(part_types_raw, dtype=np.int32)
        if int(np.sum(part_types == 1)) > 0:
            neighbors_raw = getattr(engine.mesh, "voxel_neighbors", None)
            if neighbors_raw is not None:
                neighbors = np.ascontiguousarray(neighbors_raw, dtype=np.int32)
                feeding_zone_mask = np.zeros(n_voxels, dtype=np.bool_)
                _feeding_zone_mask_kernel(states, neighbors, part_types, feeding_zone_mask)

    if feeding_zone_mask is None:
        feeding_zone_mask = np.zeros(n_voxels, dtype=np.bool_)

    extra = np.zeros(n_voxels, dtype=np.float32)
    _directional_solidification_delta_kernel(
        states.astype(np.float32, copy=False),
        heights,
        feeding_zone_mask,
        float(feeding_bias),
        float(baseline),
        float(DVR_RATE),
        float(engine.time_step),
        extra,
    )
    if not bool(np.any(extra > 0.0)):
        return False
    new_states = np.minimum(
        states.astype(np.float32, copy=False) + extra.astype(np.float32, copy=False),
        1.0,
    ).astype(np.float64)
    solidification_model._fraction_solid[:] = new_states
    prev = np.asarray(solidification_model._fraction_solid_prev, dtype=np.float64)
    solidification_model._fraction_solid_prev[:] = np.minimum(prev, new_states)
    current_time = float(engine.time)
    newly_complete = (
        np.asarray(solidification_model._fraction_solid_prev, dtype=np.float64) < 0.99
    ) & (new_states >= 0.99)
    if newly_complete.any():
        end_arr = np.asarray(solidification_model._solidification_end, dtype=np.float64)
        end_arr[newly_complete] = current_time
        solidification_model._solidification_end[:] = end_arr
    newly_started = (
        np.asarray(solidification_model._fraction_solid_prev, dtype=np.float64) < 0.01
    ) & (new_states >= 0.01)
    if newly_started.any():
        start_arr = np.asarray(solidification_model._solidification_start, dtype=np.float64)
        start_arr[newly_started] = current_time
        solidification_model._solidification_start[:] = start_arr
    engine.mesh.voxel_states[:] = np.asarray(new_states, dtype=np.float32)
    return True


def _capture_niyama_history_step(
    engine: Any,
    *,
    previous_states: np.ndarray,
    current_states: np.ndarray,
    previous_temperatures: np.ndarray,
    temperatures: np.ndarray,
    cooling_rate_step: np.ndarray,
) -> None:
    """Sample raw Niyama inputs once when each cast voxel reaches late solidification."""
    sampled = getattr(engine, "_thermal_niyama_sampled", None)
    if sampled is None:
        return

    eval_fs = float(getattr(engine, "niyama_eval_fraction_solid", NIYAMA_EVAL_FRACTION_SOLID))
    crossed = (~sampled) & (previous_states < eval_fs) & (current_states >= eval_fs)
    part_types = getattr(engine.mesh, "voxel_part_types", None)
    if part_types is not None and len(part_types) == len(crossed):
        cast_mask = np.asarray(part_types, dtype=np.int32) == 0
        crossed &= cast_mask
    else:
        cast_mask = np.ones(len(crossed), dtype=bool)

    target_indices = np.flatnonzero(crossed).astype(np.int64)
    if len(target_indices) == 0:
        return

    denom = np.maximum(current_states[target_indices] - previous_states[target_indices], 1e-9)
    step_fraction = np.clip((eval_fs - previous_states[target_indices]) / denom, 0.0, 1.0)
    eval_temperatures = previous_temperatures[target_indices] + (
        temperatures[target_indices] - previous_temperatures[target_indices]
    ) * step_fraction
    eval_times = (engine.time - engine.time_step) + engine.time_step * step_fraction

    gradients, quality = _least_squares_gradients_c_per_mm(
        centers_in=np.asarray(engine.mesh.voxel_centers, dtype=np.float64),
        temperatures=np.asarray(temperatures, dtype=np.float64),
        neighbors=np.asarray(engine.mesh.voxel_neighbors, dtype=np.int32),
        target_indices=target_indices,
        cast_mask=cast_mask,
    )
    rates = np.maximum(np.asarray(cooling_rate_step[target_indices], dtype=np.float32), 1e-6)
    niyama = gradients / np.sqrt(rates)

    engine._thermal_niyama_gradient_at_eval[target_indices] = gradients
    engine._thermal_niyama_cooling_rate_at_eval[target_indices] = rates
    engine._thermal_niyama_temperature_at_eval[target_indices] = eval_temperatures.astype(np.float32)
    engine._thermal_niyama_time_at_eval[target_indices] = eval_times.astype(np.float32)
    engine._thermal_niyama_values_at_eval[target_indices] = niyama.astype(np.float32)
    engine._thermal_niyama_quality[target_indices] = quality
    sampled[target_indices] = True


def _run_front_based_integration(
    engine: Any,
    chill_boost: np.ndarray,
    riser_slowdown: np.ndarray,
    *,
    show_progress: bool,
    progress_callback: ProgressCallback,
) -> int:
    """Legacy front-based time loop (fallback path)."""
    wcb_diffusivity = 45 / (7200 * 600)
    thermal_rate_factor = engine.alloy.thermal_diffusivity / wcb_diffusivity

    freezing_range = engine.alloy.liquidus_temp - engine.alloy.solidus_temp
    wcb_freezing_range = 61
    freezing_range_factor = wcb_freezing_range / max(freezing_range, 1)
    engine.material_solidification_factor = thermal_rate_factor * freezing_range_factor

    if show_progress:
        print("  Simulating solidification progression...")
        print(f"    Material: {engine.alloy.name}")
        print(
            f"    Thermal diffusivity: {engine.alloy.thermal_diffusivity:.2e} m²/s "
            f"(factor: {thermal_rate_factor:.2f}x vs WCB)"
        )
        print(f"    Freezing range: {freezing_range:.0f}°C (factor: {freezing_range_factor:.2f}x vs WCB)")
        print(f"    Combined solidification rate factor: {engine.material_solidification_factor:.2f}x")
        if np.any(chill_boost > 1.0):
            print(f"    Chill effect: up to {np.max(chill_boost):.1f}x faster cooling")
        if np.any(riser_slowdown < 1.0):
            print(f"    Riser exothermic effect: {(1.0 / np.min(riser_slowdown)):.1f}x slower cooling")

        perf_opts = []
        if engine._use_gpu:
            perf_opts.append(f"GPU ({engine._gpu_device_name})")
        if engine._use_incremental:
            perf_opts.append("Incremental front tracking")
        if perf_opts:
            print(f"    Performance: {', '.join(perf_opts)}")

    if engine._use_incremental:
        engine._incremental_engine = engine._incremental_engine_cls(len(engine.mesh.voxel_centers))
        engine._incremental_engine.initialize_front(
            np.asarray(engine.mesh.voxel_states),
            np.asarray(engine.mesh.voxel_neighbors),
        )

    iteration = 0
    max_iterations = int(engine.max_time / engine.time_step)
    target_snapshots = 100
    last_snapshot_progress = -1.0
    progress_interval = 100.0 / target_snapshots
    last_progress_report = 0

    history_capture_fn = getattr(engine, "_should_capture_fraction_history", None)
    if callable(history_capture_fn):
        capture_fraction_history = bool(history_capture_fn())
    else:
        capture_fraction_history = (
            engine.analysis_options.fraction_liquid
            or engine.analysis_options.hot_tear
            or engine.analysis_options.shrinkage
            or engine.analysis_options.feeding_path
        )

    if capture_fraction_history:
        engine.fraction_liquid_history.append(engine.mesh.voxel_states.copy())
        engine.fraction_liquid_times.append(0.0)

    while iteration < max_iterations:
        if engine._use_incremental and engine._incremental_engine is not None:
            if engine._incremental_engine.is_complete(np.asarray(engine.mesh.voxel_states)):
                break
        elif np.all(engine.mesh.voxel_states >= 1.0):
            break

        effective_slowdown = riser_slowdown * engine._air_gap_slowdown
        previous_states = np.asarray(engine.mesh.voxel_states, dtype=np.float32).copy()

        feeding_bias = _resolve_feeding_zone_bias(engine)

        if engine._use_incremental and engine._incremental_engine is not None:
            engine.mesh.voxel_states[:] = engine._incremental_engine.update(
                np.asarray(engine.mesh.voxel_states, dtype=np.float32),
                np.asarray(engine.mesh.voxel_neighbors, dtype=np.int32),
                np.asarray(engine.mesh.voxel_moduli, dtype=np.float32),
                np.asarray(engine._get_normalized_heights(), dtype=np.float32),
                np.asarray(chill_boost, dtype=np.float32),
                np.asarray(effective_slowdown, dtype=np.float32),
                engine.time_step,
                engine.material_solidification_factor,
            )
        elif engine._use_gpu:
            part_types = getattr(engine.mesh, "voxel_part_types", None)
            if part_types is None:
                part_types = np.zeros(len(engine.mesh.voxel_states), dtype=np.int32)
            engine.mesh.voxel_states[:] = engine._gpu_update_fn(
                np.asarray(engine.mesh.voxel_states, dtype=np.float32),
                np.asarray(engine.mesh.voxel_neighbors, dtype=np.int32),
                np.asarray(engine.mesh.voxel_moduli, dtype=np.float32),
                np.asarray(engine._get_normalized_heights(), dtype=np.float32),
                np.asarray(chill_boost, dtype=np.float32),
                np.asarray(effective_slowdown, dtype=np.float32),
                np.asarray(part_types, dtype=np.int32),
                engine.time_step,
                engine.material_solidification_factor,
                feeding_bias,
            )
        else:
            engine.mesh.voxel_states = engine._update_solidification_with_part_types(
                chill_boost,
                effective_slowdown,
                engine.material_solidification_factor,
            )

        feeding_fraction_solid = getattr(engine, "feeding_fraction_solid", None)
        if feeding_fraction_solid is not None:
            critical_fs = float(getattr(engine.alloy, "critical_fraction_solid", 1.0))
            crossed_feeding_limit = (
                (previous_states < critical_fs)
                & (engine.mesh.voxel_states >= critical_fs)
                & np.isnan(feeding_fraction_solid)
            )
            feeding_fraction_solid[crossed_feeding_limit] = engine.mesh.voxel_states[crossed_feeding_limit]

        newly_solid = (engine.mesh.voxel_states >= 1.0) & (engine.solidification_times == 0)
        engine.solidification_times[newly_solid] = engine.time

        engine.time += engine.time_step
        iteration += 1

        if engine._chill_centers is not None and iteration % int(10.0 / engine.time_step) == 0:
            engine._update_dynamic_chill_boost()
        if iteration % int(20.0 / engine.time_step) == 0:
            engine._update_mold_air_gap_effect()
        if iteration % int(5.0 / engine.time_step) == 0:
            engine._update_riser_exothermic_effect(riser_slowdown)

        current_progress_pct = np.mean(engine.mesh.voxel_states) * 100
        if capture_fraction_history and current_progress_pct - last_snapshot_progress >= progress_interval:
            engine.fraction_liquid_history.append(engine.mesh.voxel_states.copy())
            engine.fraction_liquid_times.append(engine.time)
            last_snapshot_progress = current_progress_pct

        if progress_callback and int(current_progress_pct) > last_progress_report:
            progress_callback(int(current_progress_pct), 100, "Solidification")
            last_progress_report = int(current_progress_pct)

        if show_progress and iteration % max(1, max_iterations // 10) == 0:
            print(f"    Progress: {current_progress_pct:.1f}% solidified (t={engine.time:.1f}s)")

    if capture_fraction_history:
        engine.fraction_liquid_history.append(engine.mesh.voxel_states.copy())
        engine.fraction_liquid_times.append(engine.time)
        final_progress = np.mean(engine.mesh.voxel_states) * 100
        fully_solid = bool(np.all(engine.mesh.voxel_states >= 1.0))
        if not fully_solid:
            solid_state = np.ones_like(engine.mesh.voxel_states)
            engine.fraction_liquid_history.append(solid_state)
            engine.fraction_liquid_times.append(engine.time + 0.1)
            if show_progress:
                print(
                    f"    Note: Simulation ended at {final_progress:.1f}% solid "
                    f"with residual liquid - added 100% solid snapshot"
                )
        if show_progress and engine.analysis_options.fraction_liquid:
            print(f"  ✓ Fraction liquid: {len(engine.fraction_liquid_times)} snapshots captured")
            print(f"    Time range: 0.0s to {engine.fraction_liquid_times[-1]:.1f}s")

    feeding_fraction_solid = getattr(engine, "feeding_fraction_solid", None)
    if feeding_fraction_solid is not None:
        unresolved = np.isnan(feeding_fraction_solid)
        feeding_fraction_solid[unresolved] = np.asarray(engine.mesh.voxel_states, dtype=np.float32)[unresolved]

    if progress_callback:
        progress_callback(100, 100, "Solidification")

    engine._logger.info(
        "Solidification simulation completed",
        context={
            "total_time_seconds": engine.time,
            "iterations": iteration,
            "alloy": engine.alloy.name,
            "voxel_count": len(engine.mesh.voxel_states),
        },
    )

    if show_progress:
        print(f"  ✓ Solidification complete at t={engine.time:.1f}s")
        if engine.analysis_options.fraction_liquid:
            print(f"    Fraction liquid snapshots: {len(engine.fraction_liquid_times)}")
        if engine._use_incremental and engine._incremental_engine is not None:
            stats = engine._incremental_engine.get_stats()
            print(f"  Incremental stats: {stats['estimated_speedup']:.1f}x estimated speedup")
            print(f"    Front updates avoided: {stats['full_array_updates_avoided']:,} voxel-iterations")

    return iteration


def compute_schedule_fraction_solid(
    start_times: np.ndarray,
    local_times: np.ndarray,
    eval_time: float,
    pre_solid_mask: np.ndarray,
    *,
    curve_exponent: float = DEFAULT_SCHEDULE_CURVE_EXPONENT,
) -> np.ndarray:
    """Compute fraction solid from precomputed solidification schedules."""
    fraction_solid = np.zeros_like(start_times, dtype=np.float32)
    active = local_times > 0.0
    if np.any(active):
        progress = np.clip(
            (eval_time - start_times[active]) / local_times[active],
            0.0,
            1.0,
        )
        if abs(curve_exponent - 1.0) > 1e-6:
            progress = np.power(progress, curve_exponent)
        fraction_solid[active] = progress.astype(np.float32, copy=False)
    fraction_solid[pre_solid_mask] = 1.0
    return fraction_solid


def build_schedule_fraction_history(
    start_times: np.ndarray,
    local_times: np.ndarray,
    pre_solid_mask: np.ndarray,
    final_time: float,
    *,
    curve_exponent: float = DEFAULT_SCHEDULE_CURVE_EXPONENT,
    sample_count: int = 101,
) -> tuple[list[np.ndarray], list[float]]:
    """Sample the schedule on a uniform time grid for playback and history analyses."""
    if final_time <= 0.0:
        sample_times = np.array([0.0], dtype=np.float32)
    else:
        sample_times = np.linspace(
            0.0,
            final_time,
            num=max(sample_count, 2),
            dtype=np.float32,
        )

    history = [
        compute_schedule_fraction_solid(
            start_times,
            local_times,
            float(sample_time),
            pre_solid_mask,
            curve_exponent=curve_exponent,
        )
        for sample_time in sample_times
    ]
    history_times = [float(sample_time) for sample_time in sample_times]

    # Only append a synthetic all-solid snapshot if the last sample genuinely
    # failed to reach solid fraction due to floating-point rounding.  When the
    # schedule is uncapped, the true final time equals max(end_times), so the
    # last voxel is exactly at its end_time and should be solid.
    if not np.all(np.isclose(history[-1], 1.0, atol=1e-6, rtol=0)):
        history.append(np.ones_like(start_times, dtype=np.float32))
        history_times.append(float(final_time) + 0.1)

    return history, history_times


def _run_schedule_based_integration(
    engine: Any,
    *,
    show_progress: bool,
    progress_callback: ProgressCallback,
) -> int:
    """Schedule-driven fraction-solid loop used by the authoritative engine."""
    start_times = np.asarray(engine.solidification_start_times, dtype=np.float32)
    end_times = np.asarray(engine.solidification_end_times, dtype=np.float32)
    local_times = np.asarray(engine.local_solidification_time, dtype=np.float32)
    pre_solid_mask = np.asarray(engine.mesh.voxel_states, dtype=np.float32) >= 1.0
    curve_exponent = float(
        getattr(engine, "_schedule_curve_exponent", DEFAULT_SCHEDULE_CURVE_EXPONENT)
    )

    last_progress_report = 0

    history_capture_fn = getattr(engine, "_should_capture_fraction_history", None)
    if callable(history_capture_fn):
        capture_fraction_history = bool(history_capture_fn())
    else:
        capture_fraction_history = (
            engine.analysis_options.fraction_liquid
            or engine.analysis_options.hot_tear
            or engine.analysis_options.shrinkage
            or engine.analysis_options.feeding_path
        )

    final_schedule_time = float(
        np.max(end_times[local_times > 0.0]) if np.any(local_times > 0.0) else 0.0
    )
    max_time = float(engine.max_time)

    # Compute final states directly from schedule; no need to iterate through 0.1s
    # steps when the uncapped horizon may be 30+ ks (hours of wall-clock time).
    final_states = compute_schedule_fraction_solid(
        start_times,
        local_times,
        final_schedule_time,
        pre_solid_mask,
        curve_exponent=curve_exponent,
    )
    engine.mesh.voxel_states[:] = final_states
    engine.time = final_schedule_time

    # Set solidification_times from the schedule directly.
    engine.solidification_times[:] = end_times

    # feeding_fraction_solid: record the fraction solid when each voxel
    # first crosses the coherency point.  Keep consistent with the
    # front_based / thermal_coupled convention so that downstream
    # feeding-path and Darcy analyses see physically meaningful values.
    critical_fs = float(getattr(engine.alloy, "critical_fraction_solid", 1.0))
    feeding_fraction_solid = getattr(engine, "feeding_fraction_solid", None)
    if feeding_fraction_solid is not None:
        mask = np.isnan(feeding_fraction_solid) & (local_times > 0)
        feeding_fraction_solid[mask] = critical_fs

    # Report progress at discretized percentages so the callback still fires.
    if progress_callback or show_progress:
        n_reports = 10
        for pct in range(10, 101, 10):
            t_at_pct = final_schedule_time * pct / 100.0
            states_at_pct = compute_schedule_fraction_solid(
                start_times, local_times, t_at_pct, pre_solid_mask, curve_exponent=curve_exponent
            )
            current_progress_pct = float(np.mean(states_at_pct)) * 100
            if progress_callback and int(current_progress_pct) > last_progress_report:
                progress_callback(int(current_progress_pct), 100, "Solidification")
                last_progress_report = int(current_progress_pct)
            if show_progress:
                print(f"    Progress: {current_progress_pct:.1f}% solidified (t={t_at_pct:.1f}s)")

    if capture_fraction_history:
        history, history_times = build_schedule_fraction_history(
            start_times,
            local_times,
            pre_solid_mask,
            final_schedule_time,
            curve_exponent=curve_exponent,
        )
        engine.fraction_liquid_history = history
        engine.fraction_liquid_times = history_times
        if show_progress and engine.analysis_options.fraction_liquid:
            print(f"  ✓ Fraction liquid: {len(engine.fraction_liquid_times)} snapshots captured")
            print(f"    Time range: 0.0s to {engine.fraction_liquid_times[-1]:.1f}s")

    if progress_callback:
        progress_callback(100, 100, "Solidification")

    engine._logger.info(
        "Solidification simulation completed",
        context={
            "total_time_seconds": engine.time,
            "iterations": 0,
            "alloy": engine.alloy.name,
            "voxel_count": len(engine.mesh.voxel_states),
            "mode": "schedule",
        },
    )

    if show_progress:
        print(f"  ✓ Solidification complete at t={engine.time:.1f}s")
        if engine.analysis_options.fraction_liquid:
            print(f"    Fraction liquid snapshots: {len(engine.fraction_liquid_times)}")

    return 0


def _apply_enthalpy_correction_numpy(
    T_post: np.ndarray,
    fs_prev: np.ndarray,
    T_final: np.ndarray,
    T_liq: float,
    T_sol: float,
    dT: float,
    beta: float,
) -> None:
    """
    NumPy reference for the implicit latent-heat correction.

    Fallback for when the Numba kernel is unavailable, and the parity
    reference the kernel is tested against. Writes into ``T_final``; see
    ``_run_thermal_coupled_integration`` for the derivation.
    """
    T_final[:] = T_post
    below_liq = T_post < T_liq
    if not np.any(below_liq):
        return

    liq_idx = np.flatnonzero(below_liq)
    fs_max = np.clip((T_liq - T_post[liq_idx]) / dT, 0.0, 1.0)  # fs(T_post)
    active = (fs_max > fs_prev[liq_idx]) & (T_post[liq_idx] > T_sol)
    if np.any(active):
        sel = liq_idx[active]
        T_mush = (T_post[sel] + beta * (T_liq / dT - fs_prev[sel])) / (1.0 + beta / dT)
        # Branch: implicit solve leaves the mushy zone upward (would
        # remelt -> no latent, keep T_post) or downward (all latent
        # released at solidus -> recalesce by the remaining fraction).
        remelt = T_mush >= T_liq
        all_solid = T_mush <= T_sol
        T_final[sel] = np.where(
            remelt,
            T_post[sel],
            np.where(
                all_solid,
                T_post[sel] + beta * (1.0 - fs_prev[sel]),
                T_mush,
            ),
        )
    below_sol = T_post[liq_idx] <= T_sol
    if np.any(below_sol):
        # Already below solidus: release all remaining latent.
        sel2 = liq_idx[below_sol]
        remaining = 1.0 - fs_prev[sel2]
        # Recalescence must not push a completed voxel back above
        # solidus. On a coarse step a voxel can cross below solidus
        # with a large remaining latent fraction (fine steps cross
        # with remaining ~ 0), and without this cap the temperature
        # jump beta*remaining (tens of K) re-heats completed voxels
        # into the mushy zone as permanent hot lumps, stalling their
        # neighbors (observed: a dt=2s run frozen at ~98% while the
        # dt=0.1s run finished at the same wall time). Capping at
        # T_sol makes coarse-step crossings consistent with the
        # fine-step limit.
        T_final[sel2] = np.minimum(
            T_post[sel2] + beta * np.clip(remaining, 0.0, 1.0),
            T_sol,
        )


def _run_thermal_coupled_integration(
    engine: Any,
    *,
    show_progress: bool,
    progress_callback: ProgressCallback,
) -> int:
    """Thermal-field integration loop for the coupled heat/solidification solver."""
    heat_model = getattr(engine, "_thermal_heat_model", None)
    solidification_model = getattr(engine, "_thermal_solidification_model", None)
    if heat_model is None or solidification_model is None:
        raise RuntimeError("Thermal-coupled mode requires initialized thermal models.")

    iteration = 0
    max_iterations = int(engine.max_time / engine.time_step)
    last_progress_report = 0
    last_budget_check = time.monotonic()
    last_snapshot_progress = -1.0
    progress_interval = 1.0
    history_capture_fn = getattr(engine, "_should_capture_fraction_history", None)
    if callable(history_capture_fn):
        capture_fraction_history = bool(history_capture_fn())
    else:
        capture_fraction_history = (
            engine.analysis_options.fraction_liquid
            or engine.analysis_options.hot_tear
            or engine.analysis_options.shrinkage
            or engine.analysis_options.feeding_path
        )

    if capture_fraction_history:
        engine.fraction_liquid_history = [np.asarray(engine.mesh.voxel_states, dtype=np.float32).copy()]
        engine.fraction_liquid_times = [0.0]

    # Thermal gradients feed shrinkage/feeding-effectiveness, hot-tear, and
    # Niyama post-analysis; the Niyama history sampler feeds Niyama. When none
    # of those analyses are enabled there is no consumer for either per-
    # iteration product (hotspot, sand_burn_in, and fraction-liquid history do
    # not read them), so gate both behind one flag computed before the loop
    # and skip the full-voxel gradient kernel plus per-voxel lstsq sampling.
    # Missing attributes fall back to the AnalysisOptions defaults so
    # SimpleNamespace test doubles behave like the real engine.
    needs_thermal_history = bool(
        getattr(engine.analysis_options, "niyama", False)
        or getattr(engine.analysis_options, "hot_tear", False)
        or getattr(engine.analysis_options, "shrinkage", True)
        or getattr(engine.analysis_options, "feeding_path", False)
    )

    # Per-iteration scratch, allocated once: the enthalpy correction's output
    # field and the pre-step fraction-solid snapshot it reads.
    n_voxels = int(np.asarray(heat_model.temperatures).size)
    T_final = np.empty(n_voxels, dtype=np.float64)
    fs_prev = np.empty(n_voxels, dtype=np.float64)

    while iteration < max_iterations:
        previous_states = np.asarray(engine.mesh.voxel_states, dtype=np.float32).copy()
        if np.all(previous_states >= 0.99):
            break

        previous_temperatures = np.asarray(heat_model.temperatures, dtype=np.float64).copy()
        heat_model.step(engine.time_step)  # pure diffusion only

        # Implicit (enthalpy) latent-heat release on the SAME step.
        #
        # The naive explicit correction  T += L_vol*dfs/rho_cp  overshoots:
        # the full-step latent of a single temperature drop is ~8x larger
        # than the step's diffusion cooling, so the field alternates between
        # a latent pulse and pure diffusion at ~1/8 duty cycle, stalling the
        # mushy plateau ~8x longer than the energy balance allows. Instead
        # solve the enthalpy equation implicitly for T_final in the mushy
        # zone:
        #   T_final = T_post + beta * (fs(T_final) - fs_prev),
        #   beta = L_vol/rho_cp,  fs(T) = (T_liq - T)/dT
        # which has the closed-form solution
        #   T_final = (T_post + beta*(T_liq/dT - fs_prev)) / (1 + beta/dT)
        # with branch handling when T_final leaves the mushy zone. This is
        # the classic apparent-heat-capacity formulation: oscillation-free,
        # energy-conserving, and the plateau proceeds at the physical rate.
        np.copyto(fs_prev, np.asarray(solidification_model.fraction_solid, dtype=np.float64))
        T_post = np.asarray(heat_model.temperatures, dtype=np.float64)
        T_liq = float(engine.alloy.liquidus_temp)
        T_sol = float(engine.alloy.solidus_temp)
        dT = max(T_liq - T_sol, 1e-6)
        rho_cp = float(engine.alloy.density_liquid * engine.alloy.specific_heat)
        l_vol = float(
            engine.alloy.latent_heat
            * (engine.alloy.density_liquid + engine.alloy.density_solid)
            / 2.0
        )
        beta = l_vol / rho_cp  # K of recalescence per unit fraction solid

        if ENTHALPY_KERNEL_AVAILABLE:
            apply_enthalpy_correction_parallel(
                T_post, fs_prev, T_final, T_liq, T_sol, dT, beta
            )
        else:
            _apply_enthalpy_correction_numpy(
                T_post, fs_prev, T_final, T_liq, T_sol, dT, beta
            )
        heat_model.temperatures[:] = T_final

        # Fraction solid and solidification-time tracking from the corrected
        # field; update() keeps fs monotone (irreversible).
        latent_heat_rate = solidification_model.update(T_final, engine.time_step)

        # Re-read AFTER the correction so cooling rate, Niyama capture,
        # gradients, and progress all use the corrected field.
        temperatures = np.asarray(heat_model.temperatures, dtype=np.float64)
        cooling_rate_step = np.maximum(heat_model.get_cooling_rate(previous_temperatures, engine.time_step), 0.0)

        current_states = np.asarray(solidification_model.fraction_solid, dtype=np.float32)
        engine.mesh.voxel_states[:] = current_states
        engine.time = heat_model.time
        iteration += 1

        # thermal-coupled mode is purely temperature-driven; no heuristic vertical solidification bias

        start_times = np.asarray(solidification_model.solidification_time, dtype=np.float64)
        end_times = np.asarray(solidification_model._solidification_end, dtype=np.float64)

        finite_start = np.isfinite(start_times)
        finite_end = np.isfinite(end_times)
        # Same value as the model's local_solidification_time property, built
        # from the arrays already in hand instead of re-deriving them. The
        # masked subtract leaves unfinished voxels at inf without evaluating
        # inf - inf for them.
        finite_local = finite_start & finite_end
        local_times = np.full_like(start_times, np.inf)
        np.subtract(end_times, start_times, out=local_times, where=finite_local)

        engine.solidification_start_times[:] = np.where(finite_start, start_times, 0.0).astype(np.float32)
        # Unfinished voxels get NaN (not 0.0) so they cannot collide with
        # chill/pre-solid semantics downstream.
        engine.solidification_end_times[:] = np.where(finite_end, end_times, np.nan).astype(np.float32)
        engine.local_solidification_time[:] = np.where(
            finite_local,
            local_times,
            np.where(finite_start, np.maximum(engine.time - start_times, engine.time_step), np.nan),
        ).astype(np.float32)
        engine.solidification_times[:] = engine.solidification_end_times

        if getattr(engine, "feeding_fraction_solid", None) is not None:
            critical_fs = float(getattr(engine.alloy, "critical_fraction_solid", 1.0))
            crossed_feeding_limit = (
                (previous_states < critical_fs)
                & (current_states >= critical_fs)
                & np.isnan(engine.feeding_fraction_solid)
            )
            # With same-step latent heat most voxels cross from liquid to fully
            # solid within one step; recording current_states (=1.0) for those
            # makes the feeding-path tracer treat them as unreachable and
            # collapses feeding paths. The meaningful value at the coherency
            # cutoff is the critical fraction solid itself; record that for
            # degenerate single-step crossings and the observed overshoot
            # otherwise (matches legacy multi-step behavior).
            full_solid_crossing = crossed_feeding_limit & (current_states >= 0.99)
            engine.feeding_fraction_solid[full_solid_crossing] = critical_fs
            engine.feeding_fraction_solid[crossed_feeding_limit & ~full_solid_crossing] = (
                current_states[crossed_feeding_limit & ~full_solid_crossing]
            )

        if getattr(engine, "_thermal_peak_cooling_rates", None) is not None:
            engine._thermal_peak_cooling_rates[:] = np.maximum(
                np.asarray(engine._thermal_peak_cooling_rates, dtype=np.float32),
                cooling_rate_step.astype(np.float32),
            )
            engine.cooling_rates = np.asarray(engine._thermal_peak_cooling_rates, dtype=np.float32).copy()

        if needs_thermal_history:
            _capture_niyama_history_step(
                engine,
                previous_states=previous_states,
                current_states=current_states,
                previous_temperatures=previous_temperatures,
                temperatures=temperatures,
                cooling_rate_step=cooling_rate_step,
            )

        engine._thermal_latest_temperatures = temperatures.astype(np.float32, copy=True)
        if (
            needs_thermal_history
            and getattr(engine, "_thermal_heat_model", None) is not None
            and getattr(engine, "_thermal_voxel_size_m", None)
        ):
            neighbor_map = np.asarray(engine.mesh.voxel_neighbors, dtype=np.int32)
            if hasattr(heat_model, "get_gradient_magnitude"):
                gradients_m = heat_model.get_gradient_magnitude(
                    neighbor_map,
                    float(engine._thermal_voxel_size_m),
                )
            else:
                gradients_m = heat_model.temperature_field.get_gradient_magnitude(
                    neighbor_map,
                    float(engine._thermal_voxel_size_m),
                )
            engine._thermal_latest_gradients = (gradients_m / 1000.0).astype(np.float32)
            engine.thermal_gradients = engine._thermal_latest_gradients.copy()

        current_progress_pct = float(np.mean(current_states) * 100.0)
        if capture_fraction_history and current_progress_pct - last_snapshot_progress >= progress_interval:
            engine.fraction_liquid_history.append(current_states.copy())
            engine.fraction_liquid_times.append(engine.time)
            last_snapshot_progress = current_progress_pct

        if progress_callback and int(current_progress_pct) > last_progress_report:
            progress_callback(int(current_progress_pct), 100, "Solidification")
            last_progress_report = int(current_progress_pct)

        if show_progress and iteration % max(1, max_iterations // 10) == 0:
            print(
                f"    Progress: {current_progress_pct:.1f}% solidified "
                f"(t={engine.time:.1f}s, Tmin={np.min(temperatures):.1f}C, Tmax={np.max(temperatures):.1f}C)"
            )

        if time.monotonic() - last_budget_check >= THREAD_BUDGET_CHECK_INTERVAL:
            last_budget_check = time.monotonic()
            _apply_live_thread_budget(engine)

    if capture_fraction_history:
        engine.fraction_liquid_history.append(np.asarray(engine.mesh.voxel_states, dtype=np.float32).copy())
        engine.fraction_liquid_times.append(engine.time)

    # Voxels still liquid at the cap never finished: write NaN (not 0.0) so
    # they cannot collide with chill/pre-solid semantics downstream. Chill
    # voxels (pre-solid, states == 1.0 with times 0.0) keep 0.0.
    unfinished = np.asarray(engine.mesh.voxel_states, dtype=np.float32) < 0.99
    engine.solidification_end_times[unfinished] = np.nan
    engine.solidification_times[unfinished] = np.nan

    if getattr(engine, "feeding_fraction_solid", None) is not None:
        unresolved = np.isnan(engine.feeding_fraction_solid)
        engine.feeding_fraction_solid[unresolved] = np.asarray(engine.mesh.voxel_states, dtype=np.float32)[unresolved]

    if progress_callback:
        progress_callback(100, 100, "Solidification")

    engine._logger.info(
        "Solidification simulation completed",
        context={
            "total_time_seconds": engine.time,
            "iterations": iteration,
            "alloy": engine.alloy.name,
            "voxel_count": len(engine.mesh.voxel_states),
            "mode": THERMAL_COUPLED_SOLVER_MODE,
        },
    )

    if show_progress:
        print(f"  ✓ Solidification complete at t={engine.time:.1f}s")
        if engine.analysis_options.fraction_liquid:
            print(f"    Fraction liquid snapshots: {len(engine.fraction_liquid_times)}")

    return iteration


def run_time_integration(
    engine: Any,
    chill_boost: np.ndarray,
    riser_slowdown: np.ndarray,
    *,
    show_progress: bool,
    progress_callback: ProgressCallback,
) -> int:
    """
    Execute the solidification time loop and update engine state in place.
    """
    _configure_numba_threads_for_solidification(engine, show_progress=show_progress)

    solver_mode = getattr(getattr(engine, "analysis_options", None), "solver_mode", None)
    if solver_mode == THERMAL_COUPLED_SOLVER_MODE:
        return _run_thermal_coupled_integration(
            engine,
            show_progress=show_progress,
            progress_callback=progress_callback,
        )

    has_schedule = (
        getattr(engine, "solidification_start_times", None) is not None
        and getattr(engine, "solidification_end_times", None) is not None
        and getattr(engine, "local_solidification_time", None) is not None
    )
    if has_schedule:
        return _run_schedule_based_integration(
            engine,
            show_progress=show_progress,
            progress_callback=progress_callback,
        )
    return _run_front_based_integration(
        engine,
        chill_boost,
        riser_slowdown,
        show_progress=show_progress,
        progress_callback=progress_callback,
    )
