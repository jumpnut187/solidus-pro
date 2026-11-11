"""
Shared simulation pipeline used by desktop UI and web API.

This module centralizes part loading, voxel mesh generation, simulation
execution, and result packaging to avoid duplicated orchestration logic.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import os

import numpy as np

from core.filling_stage import maybe_run_filling

try:
    from core.error_logger import logger as _error_logger
except ImportError:  # pragma: no cover - headless/dummy fallback
    from core.error_logger_dummy import logger as _error_logger  # type: ignore
from core.materials import get_alloy
from core.part_manager import (
    PartManager,
    PartType,
    build_part_manager,
    infer_part_type_from_filename,
)
from core.solidification import (
    AnalysisOptions,
    SOLVER_MODE_THERMAL_COUPLED,
    VALID_SOLVER_MODES,
    SolidificationEngine,
)
from core.voxel_mesh import AdaptiveVoxelMesh
from config.exceptions import (
    MeshError,
    SimulationError,
    ValidationError,
)
from config.hardware_config import hardware
from config.settings import MeshSettings, get_settings
from core.thread_budget import set_process_thread_budget

ProgressMessageCallback = Callable[[str], None]
DetailedProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]
ThreadBudgetProvider = Callable[[], Optional[int]]


def _get_default_solver_mode_override() -> Optional[str]:
    """Return validated solver-mode override from environment, if present."""
    raw_mode: Optional[str] = os.getenv("CASTSIM_DEFAULT_SOLVER_MODE")
    if raw_mode is None:
        return None

    solver_mode: str = raw_mode.strip().lower()
    if solver_mode not in VALID_SOLVER_MODES:
        return None

    return solver_mode


class SimulationCancelledError(RuntimeError):
    """Raised when an external caller requests cancellation."""


class PipelineInputs:
    """Normalized simulation input parameters."""

    def __init__(
        self,
        alloy_name: str,
        voxel_size: float,
        refinement_level: int,
        memory_mode: str = "auto",
        analysis_options: Optional[Dict[str, Any]] = None,
        use_gpu: bool = False,
        use_incremental: bool = False,
        use_chunked_voxelization: bool = False,
        part_assignment_mode: str = "auto",
        part_assignment_workers: int = 0,
        part_assignment_batch_size: int = 2500,
        max_voxels_for_visualization: int = 50_000,
        show_progress: bool = False,
        solidification_params: Optional[Dict[str, Any]] = None,
        thread_budget_provider: Optional[ThreadBudgetProvider] = None,
        pour_location: Optional[Dict[str, Any]] = None,
        voxel_cut_cells: Optional[bool] = None,
    ) -> None:
        self.alloy_name = alloy_name
        self.voxel_size = voxel_size
        self.refinement_level = refinement_level
        self.memory_mode = memory_mode
        self.analysis_options = analysis_options
        self.use_gpu = use_gpu
        self.use_incremental = use_incremental
        self.use_chunked_voxelization = use_chunked_voxelization
        self.part_assignment_mode = part_assignment_mode
        self.part_assignment_workers = part_assignment_workers
        self.part_assignment_batch_size = part_assignment_batch_size
        self.max_voxels_for_visualization = max_voxels_for_visualization
        self.show_progress = show_progress
        self.solidification_params = solidification_params
        self.thread_budget_provider = thread_budget_provider
        # Serialized PourLocation, or None. Filling runs only when set.
        self.pour_location = pour_location
        # Cut cells on the voxel mesh (core/cut_cells.py); None uses
        # settings.mesh.voxel_cut_cells.
        self.voxel_cut_cells = voxel_cut_cells


class PipelineResult:
    """Simulation pipeline outputs."""

    def __init__(
        self,
        part_manager: Optional[PartManager],
        voxel_mesh: AdaptiveVoxelMesh,
        engine: SolidificationEngine,
        result_data: Dict[str, Any],
        visualization_data: Dict[str, Any],
    ) -> None:
        self.part_manager = part_manager
        self.voxel_mesh = voxel_mesh
        self.engine = engine
        self.result_data = result_data
        self.visualization_data = visualization_data


def resolve_voxel_cut_cells(inputs: PipelineInputs, mesh_settings: Optional[MeshSettings] = None) -> bool:
    """Cut cells for a voxel mesh of this run: ``inputs.voxel_cut_cells`` when set, else the setting.

    A settings object without the field (written before it existed) means off.
    """
    requested = getattr(inputs, "voxel_cut_cells", None)
    if requested is not None:
        return bool(requested)
    settings_mesh = mesh_settings if mesh_settings is not None else get_settings().mesh
    return bool(getattr(settings_mesh, "voxel_cut_cells", False))


def _emit_message(callback: Optional[ProgressMessageCallback], message: str) -> None:
    if callback is not None:
        callback(message)


def _check_cancel(callback: Optional[CancelCallback]) -> None:
    if callback is not None and callback():
        raise SimulationCancelledError("Cancelled")


def _apply_thread_budget(provider: Optional[ThreadBudgetProvider]) -> None:
    """
    Apply the current fair-share thread budget to this process.

    Called at pipeline stage boundaries so every thread-count consumer in
    the process (Numba pools, voxelization/part-assignment workers, and the
    env var inherited by spawned worker pools) honors the same budget.
    """
    if provider is None:
        return
    budget = provider()
    set_process_thread_budget(budget)
    hardware.set_thread_cap(budget)
    if budget is None:
        os.environ.pop("CASTSIM_THREAD_BUDGET", None)
    else:
        os.environ["CASTSIM_THREAD_BUDGET"] = str(budget)


def build_part_manager_from_parts(parts_data: Sequence[Dict[str, Any]]) -> PartManager:
    """Build a ``PartManager`` from serialized web API part payloads.

    Thin wrapper around :func:`core.part_manager.build_part_manager` that
    preserves the historical "preserve STL coordinates" default for the
    desktop pipeline.
    """
    return build_part_manager(parts_data, preserve_origin=True)


def create_analysis_options(
    options: Optional[Dict[str, Any]],
    *,
    use_gpu: bool,
    use_incremental: bool,
    solidification_params: Optional[Dict[str, Any]] = None,
) -> AnalysisOptions:
    """
    Convert dictionary options from UI/web into AnalysisOptions.
    """
    from core.solidification_params import SolidificationParameters

    option_map = options or {}
    analysis_options = AnalysisOptions()
    analysis_options.shrinkage = bool(option_map.get("shrinkage", True))
    analysis_options.hotspot = bool(option_map.get("hotspot", True))
    analysis_options.niyama = bool(option_map.get("niyama", False))
    analysis_options.hot_tear = bool(option_map.get("hot_tear", False))
    analysis_options.sand_burn_in = bool(option_map.get("sand_burn_in", False))
    analysis_options.fraction_liquid = bool(option_map.get("fraction_liquid", False))
    analysis_options.feeding_path = bool(option_map.get("feeding_path", False))
    analysis_options.stress_analysis = bool(option_map.get("stress_analysis", False))
    analysis_options.microstructure = bool(option_map.get("microstructure", False))
    analysis_options.gas_porosity = bool(option_map.get("gas_porosity", False))
    analysis_options.filling = bool(option_map.get("filling", False))
    try:
        sand_restraint_value = float(option_map.get("sand_restraint_index", 0.5))
    except (TypeError, ValueError):
        sand_restraint_value = 0.5
    analysis_options.sand_restraint_index = float(np.clip(sand_restraint_value, 0.0, 1.0))
    analysis_options.use_gpu = use_gpu
    analysis_options.use_incremental = use_incremental
    solver_mode = str(option_map.get("solver_mode", SOLVER_MODE_THERMAL_COUPLED))
    if solver_mode not in VALID_SOLVER_MODES:
        solver_mode = SOLVER_MODE_THERMAL_COUPLED
    analysis_options.solver_mode = solver_mode
    if solidification_params is not None:
        analysis_options.solidification_params = SolidificationParameters.from_dict(
            solidification_params
        )
    return analysis_options


def build_visualization_payload(
    engine: SolidificationEngine,
    voxel_mesh: AdaptiveVoxelMesh,
    *,
    total_volume: float,
    bounds: np.ndarray,
    max_voxels: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Build summary and downsampled visualization payload for web clients.
    """
    voxel_centers = voxel_mesh.voxel_centers
    if voxel_centers is None:
        raise RuntimeError("Voxel mesh centers are not available for visualization payload.")
    voxel_sizes = voxel_mesh.voxel_sizes
    if voxel_sizes is None:
        voxel_sizes = np.full(len(voxel_centers), 0.1, dtype=np.float32)
    n_voxels = len(voxel_centers)

    result_data: Dict[str, Any] = {
        "voxel_count": n_voxels,
        "total_volume": float(total_volume),
        "bounds": {
            "x": [float(bounds[0][0]), float(bounds[0][1])],
            "y": [float(bounds[1][0]), float(bounds[1][1])],
            "z": [float(bounds[2][0]), float(bounds[2][1])],
        },
    }

    if n_voxels > max_voxels:
        step = max(1, n_voxels // max_voxels)
        indices = np.arange(0, n_voxels, step)
    else:
        indices = np.arange(n_voxels)

    visualization_data: Dict[str, Any] = {
        "centers": voxel_centers[indices].tolist(),
        "sizes": voxel_sizes[indices].tolist() if voxel_sizes is not None else [0.1] * len(indices),
    }
    if voxel_mesh.surface_voxels is not None:
        surface_voxels = np.asarray(voxel_mesh.surface_voxels, dtype=bool)
        if len(surface_voxels) == n_voxels:
            visualization_data["surface_voxels"] = surface_voxels[indices].tolist()

    if engine.shrinkage_probability is not None:
        # The Shrink result: same array the desktop shows (chills at 0, no porosity).
        display_fn = getattr(engine, "get_shrink_display_intensity", None)
        if callable(display_fn):
            shrinkage = np.asarray(display_fn(), dtype=np.float32)
        else:
            shrinkage = np.asarray(engine.shrinkage_probability, dtype=np.float32)
        if len(shrinkage) != n_voxels:
            shrinkage = np.asarray(engine.shrinkage_probability, dtype=np.float32)
        visualization_data["shrinkage"] = shrinkage[indices].tolist()
        result_data["shrink_scalar"] = float(getattr(engine, "shrink_scalar", 1.0))
        result_data["shrink_scalar_formula"] = str(
            getattr(engine, "shrink_scalar_formula", "adjusted_risk = clamp(raw_risk / shrink_scalar, 0, 1)")
        )
        result_data["shrinkage_stats"] = {
            "max": float(np.max(shrinkage)),
            "mean": float(np.mean(shrinkage)),
            "affected_count": int(np.sum(shrinkage > 0.05)),
        }

    # Expose feeding-blocked and feeding-path fields for the shrinkage/porosity overlay.
    feeding_blocked = getattr(engine, "feeding_blocked", None)
    if feeding_blocked is not None and len(feeding_blocked) == n_voxels:
        visualization_data["feeding_blocked"] = np.asarray(feeding_blocked, dtype=bool)[indices].tolist()
    feeding_path_lengths = getattr(engine, "feeding_path_lengths", None)
    if feeding_path_lengths is not None and len(feeding_path_lengths) == n_voxels:
        visualization_data["feeding_path_lengths"] = np.asarray(feeding_path_lengths, dtype=np.float32)[indices].tolist()
    feeding_open_fraction = getattr(engine, "feeding_open_fraction", None)
    if feeding_open_fraction is not None and len(feeding_open_fraction) == n_voxels:
        visualization_data["feeding_open_fraction"] = np.asarray(feeding_open_fraction, dtype=np.float32)[indices].tolist()
    # The Microporosity result: same array the desktop shows (chills at 0).
    porosity_standalone = getattr(engine, "porosity_probability", None)
    microporosity_fn = getattr(engine, "get_microporosity_display_intensity", None)
    if porosity_standalone is not None and callable(microporosity_fn):
        porosity_standalone = microporosity_fn()
    if porosity_standalone is not None and len(porosity_standalone) == n_voxels:
        visualization_data["porosity"] = np.asarray(porosity_standalone, dtype=np.float32)[indices].tolist()

    local_modulus = getattr(engine, "local_modulus", None)
    if local_modulus is None:
        local_modulus = getattr(voxel_mesh, "voxel_moduli", None)
    if local_modulus is not None:
        feed_modulus = np.asarray(local_modulus, dtype=np.float32)
        if len(feed_modulus) == n_voxels:
            feed_modulus_in = np.where(np.isfinite(feed_modulus), feed_modulus, 0.0).astype(np.float32)
            # Feed modulus is only meaningful for cast metal; exclude chills and cores.
            part_types = getattr(voxel_mesh, "voxel_part_types", None)
            if part_types is not None and len(part_types) == n_voxels:
                chill_mask = np.isin(np.asarray(part_types, dtype=np.int32), (2, 4))
                if np.any(chill_mask):
                    feed_modulus_in = feed_modulus_in.copy()
                    feed_modulus_in[chill_mask] = 0.0
            visualization_data["feed_modulus"] = feed_modulus_in[indices].tolist()
            non_zero = feed_modulus_in[feed_modulus_in > 0.0]
            if non_zero.size > 0:
                result_data["feed_modulus_stats"] = {
                    "min_in": float(np.min(non_zero)),
                    "max_in": float(np.max(non_zero)),
                    "mean_in": float(np.mean(non_zero)),
                }
            else:
                result_data["feed_modulus_stats"] = {
                    "min_in": 0.0,
                    "max_in": 0.0,
                    "mean_in": 0.0,
                }

    if engine.hotspot_intensity is not None:
        hotspot = np.asarray(engine.hotspot_intensity)
        visualization_data["hotspot"] = hotspot[indices].tolist()
        result_data["hotspot_stats"] = {
            "max": float(np.max(hotspot)),
            "mean": float(np.mean(hotspot)),
            "count": int(np.sum(hotspot > 0.5)),
        }

    if engine.niyama_values is not None:
        niyama_values = np.asarray(engine.niyama_values, dtype=np.float32)
        if len(niyama_values) == n_voxels:
            niyama_raw = np.where(np.isfinite(niyama_values), niyama_values, 0.0).astype(np.float32)
            visualization_data["niyama_raw"] = niyama_raw[indices].tolist()
    if getattr(engine, "niyama_risk", None) is not None:
        niyama_risk = np.asarray(engine.niyama_risk, dtype=np.float32)
        if len(niyama_risk) == n_voxels:
            visualization_data["niyama"] = niyama_risk[indices].tolist()
            visualization_data["niyama_risk"] = niyama_risk[indices].tolist()
            result_data["niyama_stats"] = {
                "max_risk": float(np.max(niyama_risk)),
                "mean_risk": float(np.mean(niyama_risk)),
                "affected_count": int(np.sum(niyama_risk > 0.0)),
            }
    if getattr(engine, "combined_feed_risk", None) is not None:
        combined_feed_risk = np.asarray(engine.combined_feed_risk, dtype=np.float32)
        if len(combined_feed_risk) == n_voxels:
            visualization_data["combined_feed_risk"] = combined_feed_risk[indices].tolist()

    # Chills and phantom gap-fill voxels must never render as liquid metal;
    # route liquid fields through the engine's mask when available.
    mask_liquid = getattr(engine, "_mask_non_cast_fraction_liquid", None)
    if not callable(mask_liquid):
        mask_liquid = lambda arr: arr  # noqa: E731

    if voxel_mesh.voxel_states is not None:
        fraction_liquid = 1.0 - np.asarray(voxel_mesh.voxel_states).astype(float)
        fraction_liquid = np.asarray(mask_liquid(fraction_liquid), dtype=float)
        visualization_data["fraction_liquid"] = fraction_liquid[indices].tolist()
    if engine.solidification_times is not None:
        solidification_times = np.asarray(engine.solidification_times).astype(float)
        visualization_data["solidification_times"] = solidification_times[indices].tolist()

    if engine.fraction_liquid_history:
        fl_history = []
        for snapshot in engine.fraction_liquid_history:
            fl_frame = 1.0 - np.asarray(snapshot)
            fl_frame = np.asarray(mask_liquid(fl_frame), dtype=float)
            fl_history.append(fl_frame[indices].tolist())
        visualization_data["fraction_liquid_history"] = fl_history
        visualization_data["fraction_liquid_times"] = engine.fraction_liquid_times or []
        result_data["fraction_liquid_snapshots"] = len(fl_history)

    if getattr(engine, "gas_porosity_risk", None) is not None:
        gas_risk = np.asarray(engine.gas_porosity_risk, dtype=np.float32)
        if len(gas_risk) == n_voxels:
            visualization_data["gas_porosity_risk"] = gas_risk[indices].tolist()
            result_data["gas_porosity_stats"] = {
                "max_risk": float(np.max(gas_risk)),
                "mean_risk": float(np.mean(gas_risk)),
                "hotspot_count": int(getattr(engine, "gas_porosity_count", 0)),
            }
    if getattr(engine, "gas_generation", None) is not None:
        gas_gen = np.asarray(engine.gas_generation, dtype=np.float32)
        if len(gas_gen) == n_voxels:
            visualization_data["gas_generation"] = gas_gen[indices].tolist()
    if getattr(engine, "gas_entrapment", None) is not None:
        gas_ent = np.asarray(engine.gas_entrapment, dtype=np.float32)
        if len(gas_ent) == n_voxels:
            visualization_data["gas_entrapment"] = gas_ent[indices].tolist()
    if getattr(engine, "gas_porosity_hotspots", None) is not None:
        hotspots = np.asarray(engine.gas_porosity_hotspots, dtype=bool)
        if len(hotspots) == n_voxels:
            visualization_data["gas_porosity_hotspots"] = hotspots[indices].tolist()

    # Mold-filling arrival time, in seconds, per voxel; -1 marks metal the
    # fill never reached (a misrun) and NaN marks a voxel that is not metal at
    # all, for which the question does not apply. Only present when a filling
    # stage ran, so a run without filling does not gain an empty "fill_time"
    # mode.
    fill_arrival_times = getattr(engine, "_fill_arrival_times", None)
    if fill_arrival_times is not None:
        fill_times = np.asarray(fill_arrival_times, dtype=np.float64)
        if len(fill_times) == n_voxels:
            visualization_data["fill_time"] = fill_times[indices].tolist()
            filled_mask = fill_times >= 0.0
            filled_count = int(np.sum(filled_mask))
            # Non-metal voxels are NaN, not misruns (see core.filling_stage);
            # subtracting from n_voxels would bill the entire mold as unfilled.
            unfilled_count = int(np.sum(np.isfinite(fill_times) & ~filled_mask))
            max_fill_time_s = float(np.max(fill_times[filled_mask])) if filled_count > 0 else 0.0
            result_data["fill_time_stats"] = {
                "max_fill_time_s": max_fill_time_s,
                "filled_voxels": filled_count,
                "unfilled_voxels": unfilled_count,
            }

    # Playback frames from the pour, gathered onto voxels by the filling stage
    # (``core.filling_stage.FillFrame``). ``fill_time`` alone gives a static
    # arrival-time map that a slider can sweep; these are the actual free
    # surface at each instant, including the partial cells that are the
    # advancing front, which is what makes a fill animation read as metal
    # flowing rather than as cells switching on. The engine holds them only
    # when a filling stage ran and was trustworthy.
    fill_frames = getattr(engine, "_fill_frames", None)
    if fill_frames:
        fill_history: list[list[float]] = []
        fill_frame_times: list[float] = []
        for frame in fill_frames:
            fraction = getattr(frame, "fill_fraction", None)
            if fraction is None:
                continue
            fraction_arr = np.asarray(fraction, dtype=np.float32)
            if len(fraction_arr) != n_voxels:
                continue
            fill_history.append(np.clip(fraction_arr[indices], 0.0, 1.0).tolist())
            fill_frame_times.append(float(getattr(frame, "time_s", 0.0)))
        if fill_history:
            visualization_data["fill_animation_history"] = fill_history
            visualization_data["fill_animation_times"] = fill_frame_times
            result_data["fill_animation_frames"] = len(fill_history)

    _add_summary_statistics(result_data, engine)

    return result_data, visualization_data


def _add_summary_statistics(result_data: Dict[str, Any], engine: SolidificationEngine) -> None:
    """
    Add the run's scalar statistics under flat, final names.

    The engine's own ``export_results`` carries roughly twenty-five of these,
    but it never reached the web layer: the artifact summary was assembled
    separately and only picked up a handful of nested ``*_stats`` dicts. The
    browser now shows a statistics panel, so the numbers have to arrive, and
    they have to arrive under names that do not change shape from run to run.

    Naming rules, applied here so the panel never has to guess:
      * flat scalars only, no nesting -- ``GridMeta.stats`` is a flat map
      * a unit suffix wherever the unit is not implied: ``_s``, ``_in3``,
        ``_mpa``, ``_um``, ``_percentage``
      * ``_count`` for whole voxels, ``_fraction`` for 0..1
    A statistic the run did not produce is absent rather than zero: zero
    hot tears and "hot tearing was not analysed" are different answers.
    """

    def put(name: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (int, float)) and not np.isfinite(float(value)):
            return
        result_data[name] = value

    put("alloy", getattr(getattr(engine, "alloy", None), "name", None))
    put("simulation_time_s", _optional_float(getattr(engine, "time", None)))

    # Chvorinov is a per-voxel field; the panel wants the scalars off it.
    chvorinov = getattr(engine, "chvorinov_time", None)
    if chvorinov is not None:
        values = np.asarray(chvorinov, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size:
            put("chvorinov_time_max_s", float(np.max(finite)))
            put("chvorinov_time_mean_s", float(np.mean(finite)))

    # The defect volume the desktop's "Calculate Defect Volume" reports, at
    # its own default threshold, so the panel has a number without the user
    # having to pick one first.
    estimate = getattr(engine, "get_defect_volume_estimate", None)
    if callable(estimate):
        try:
            defect = estimate()
        except Exception:  # pragma: no cover - a statistic must never fail a run
            defect = None
        if isinstance(defect, dict):
            put("defect_volume_in3", _optional_float(defect.get("total_volume_in3")))
            put("defect_volume_max_in3", _optional_float(defect.get("max_volume_in3")))
            put("defect_volume_percentage", _optional_float(defect.get("defect_percentage")))
            put("defect_voxel_count", _optional_int(defect.get("defect_count")))
            put("defect_threshold", _optional_float(defect.get("threshold_used")))
            put("casting_volume_in3", _optional_float(defect.get("casting_volume_in3")))

    put("shrinkage_volume_in3", _optional_float(getattr(engine, "total_shrinkage_volume", None)))
    affected = _optional_int(getattr(engine, "affected_voxel_count", None))
    if affected is not None:
        put("shrinkage_voxel_count", affected)
        voxel_count = _optional_int(result_data.get("voxel_count"))
        if voxel_count:
            put("shrinkage_percentage", 100.0 * affected / voxel_count)

    put("hotspot_voxel_count", _optional_int(getattr(engine, "hotspot_count", None)))
    put("niyama_defect_count", _optional_int(getattr(engine, "niyama_defect_count", None)))
    put("hot_tear_voxel_count", _optional_int(getattr(engine, "hot_tear_count", None)))
    put("sand_burn_in_voxel_count", _optional_int(getattr(engine, "sand_burn_in_count", None)))
    put("gas_porosity_voxel_count", _optional_int(getattr(engine, "gas_porosity_count", None)))
    put("feeding_blocked_count", _optional_int(getattr(engine, "feeding_blocked_count", None)))

    feeding = getattr(engine, "feeding_path_result", None)
    put("fed_fraction", _optional_float(getattr(feeding, "fed_fraction", None)))

    # Primary pipe from the riser volume balance. Pipe volume outside the
    # riser means a feeder froze before the section it feeds.
    pipe_report = getattr(engine, "riser_pipe_report", None)
    if isinstance(pipe_report, list) and pipe_report:
        put("riser_pipe_volume_in3", float(sum(r["pipe_in3"] for r in pipe_report)))
        put("riser_pipe_in_casting_in3", float(sum(r["pipe_in_casting_in3"] for r in pipe_report)))
        clearances = [
            r["pipe_bottom_above_riser_base_in"]
            for r in pipe_report
            if np.isfinite(r["pipe_bottom_above_riser_base_in"])
        ]
        if clearances:
            put("riser_pipe_min_clearance_in", float(min(clearances)))

    # Casting metal no riser reaches drains by gravity too: its own pools, kept
    # apart from the riser numbers above.
    casting_pipe = getattr(engine, "casting_pipe_report", None)
    if isinstance(casting_pipe, list) and casting_pipe:
        put("casting_pipe_volume_in3", float(sum(r["pipe_in3"] for r in casting_pipe)))
        put("casting_pipe_pool_count", len(casting_pipe))

    put("max_stress_mpa", _optional_float(getattr(engine, "max_stress_mpa", None)))
    put("avg_sdas_um", _optional_float(getattr(engine, "avg_sdas_um", None)))

    # Mold filling, promoted out of the nested dict so the names read the same
    # way as everything else here.
    fill_stats = result_data.get("fill_time_stats")
    if isinstance(fill_stats, dict):
        put("fill_max_time_s", _optional_float(fill_stats.get("max_fill_time_s")))
        put("fill_filled_voxel_count", _optional_int(fill_stats.get("filled_voxels")))
        put("fill_unfilled_voxel_count", _optional_int(fill_stats.get("unfilled_voxels")))


def _optional_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _optional_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _prepare_mold_stage(
    inputs: PipelineInputs,
    voxel_mesh: AdaptiveVoxelMesh,
    part_manager: Optional[PartManager],
    mesh_to_use: Any,
    progress_message_callback: Optional[ProgressMessageCallback],
) -> None:
    """Sand mold and automatic cores (core/mold_setup.py), before filling and solidification."""
    from core.mold_setup import MoldSetupError, auto_cores_apply, mold_requested, prepare_mold
    from core.part_manager import PART_TYPE_CAST
    from core.solidification_params import SolidificationParameters

    mold_params = SolidificationParameters.from_dict(inputs.solidification_params).mold
    if part_manager is not None and part_manager.get_part_count() > 0:
        setup_parts = part_manager.get_mold_setup_parts()
        mold_meshes = part_manager.get_mold_meshes()
    else:
        setup_parts = [(mesh_to_use, PART_TYPE_CAST, False)]
        mold_meshes = []
    voxel_mesh.mold_layout = None
    if not (mold_requested(mold_params, mold_meshes) or auto_cores_apply(mold_params, setup_parts)):
        return
    _emit_message(progress_message_callback, "Building the sand mold and cores...")
    try:
        layout = prepare_mold(voxel_mesh, setup_parts, mold_meshes, mold_params)
    except MoldSetupError as exc:
        raise ValidationError(str(exc), parameter="mold", expected="A mold that fits the voxel budget") from exc
    if layout is not None:
        sand = (
            layout.grading.describe(layout.mold_cell_count)
            if layout.grading is not None
            else f"{layout.mold_cell_count:,} mold sand cells"
        )
        _emit_message(
            progress_message_callback,
            f"Mold: {sand}, {layout.added_core_voxels:,} auto-core voxels",
        )


def _prepare_riser_sleeves(voxel_mesh: AdaptiveVoxelMesh, part_manager: Optional[PartManager]) -> None:
    """Mark the voxels of risers set to no sleeve (``Part.riser_sleeve``), for the thermal setup."""
    from core.riser_sleeve import bare_riser_mask

    voxel_mesh.voxel_riser_bare = None
    if part_manager is None or part_manager.get_part_count() == 0:
        return
    if voxel_mesh.voxel_part_types is None or voxel_mesh.voxel_neighbors is None:
        return
    voxel_mesh.voxel_riser_bare = bare_riser_mask(
        np.asarray(voxel_mesh.voxel_centers),
        np.asarray(voxel_mesh.voxel_part_types),
        np.asarray(voxel_mesh.voxel_neighbors),
        part_manager.get_riser_sleeve_parts(),
    )


def prepare_mesh_and_geometry(
    inputs: PipelineInputs,
    *,
    part_manager: Optional[PartManager],
    stl_loader: Any,
    progress_message_callback: Optional[ProgressMessageCallback] = None,
    detailed_progress_callback: Optional[DetailedProgressCallback] = None,
    cancel_callback: Optional[CancelCallback] = None,
) -> Tuple[Any, Any, float, Sequence[Any], AdaptiveVoxelMesh, Dict[str, Any]]:
    """
    Resolve the geometry source and generate the simulation mesh.

    Shared by the traditional pipeline (:func:`run_pipeline`) and the
    surrogate pipeline (``core.surrogate.pipeline``) so both paths perform
    identical geometry work.

    Returns:
        ``(mesh_to_use, bounds_to_use, total_volume, parts_with_types,
        voxel_mesh, mem_stats)``. ``parts_with_types`` is empty when the
        geometry came from a bare STL.
    """
    mesh_settings = get_settings().mesh
    _check_cancel(cancel_callback)

    mesh_to_use = None
    bounds_to_use = None
    total_volume = 0.0
    parts_with_types = []

    if part_manager is not None and part_manager.get_part_count() > 0:
        combined_mesh = part_manager.get_combined_mesh()
        combined_bounds = part_manager.get_combined_bounds()
        if combined_mesh is None or combined_bounds is None:
            raise MeshError(
                "Failed to build combined mesh from parts",
                suggestion="Inspect each STL for malformed geometry and reload",
            )
        mesh_to_use = combined_mesh
        bounds_to_use = combined_bounds
        total_volume = float(part_manager.get_total_volume())
        parts_with_types = part_manager.get_parts_with_types()
    elif stl_loader is not None and getattr(stl_loader, "trimesh_obj", None) is not None:
        mesh_to_use = stl_loader.trimesh_obj
        bounds_to_use = stl_loader.get_bounds_array()
        total_volume = float(getattr(stl_loader, "volume", 0.0) or 0.0)
    else:
        raise ValidationError(
            "No valid geometry source provided to simulation pipeline",
            parameter="part_manager|stl_loader",
            expected="A non-empty PartManager or an initialized STLLoader",
        )

    _emit_message(
        progress_message_callback,
        f"Generating adaptive voxel mesh (Memory: {inputs.memory_mode.upper()})...",
    )

    voxel_mesh = AdaptiveVoxelMesh(
        base_voxel_size=inputs.voxel_size,
        max_refinement_level=inputs.refinement_level,
        memory_mode=inputs.memory_mode,
        use_chunked_voxelization=inputs.use_chunked_voxelization,
        part_assignment_mode=inputs.part_assignment_mode,
        part_assignment_workers=inputs.part_assignment_workers,
        part_assignment_batch_size=inputs.part_assignment_batch_size,
        voxel_cut_cells=resolve_voxel_cut_cells(inputs, mesh_settings),
    )

    def on_mesh_progress(current: int, total: int, stage: str) -> None:
        _check_cancel(cancel_callback)
        if detailed_progress_callback is not None:
            detailed_progress_callback(current, total, stage)

    _apply_thread_budget(inputs.thread_budget_provider)
    assert bounds_to_use is not None
    success = voxel_mesh.generate(
        mesh_to_use,
        bounds_to_use,
        show_progress=False,
        progress_callback=on_mesh_progress,
        # The parts themselves, so a pocket they seal off between them is not
        # filled as metal along with their interiors.
        part_meshes=[mesh for mesh, _ in parts_with_types] or None,
    )
    if not success:
        raise SimulationError(
            "Voxel mesh generation failed",
            phase="mesh_generation",
            suggestion="Increase voxel size or reduce model complexity",
        )

    _check_cancel(cancel_callback)
    if parts_with_types:
        _emit_message(progress_message_callback, "Assigning part types (Cast/Riser/Chill)...")
        _apply_thread_budget(inputs.thread_budget_provider)
        voxel_mesh.assign_part_types(parts_with_types)

    _check_cancel(cancel_callback)
    _prepare_mold_stage(inputs, voxel_mesh, part_manager, mesh_to_use, progress_message_callback)
    _prepare_riser_sleeves(voxel_mesh, part_manager)

    mem_stats = voxel_mesh.get_memory_stats()
    _emit_message(
        progress_message_callback,
        (
            f"Memory: RAM={mem_stats['ram_mb']:.1f}MB, "
            f"Disk={mem_stats['disk_mb']:.1f}MB, "
            f"Available={mem_stats['available_gb']:.1f}GB"
        ),
    )
    return mesh_to_use, bounds_to_use, total_volume, parts_with_types, voxel_mesh, mem_stats


def run_pipeline(
    inputs: PipelineInputs,
    *,
    part_manager: Optional[PartManager] = None,
    stl_loader: Any = None,
    progress_message_callback: Optional[ProgressMessageCallback] = None,
    detailed_progress_callback: Optional[DetailedProgressCallback] = None,
    cancel_callback: Optional[CancelCallback] = None,
) -> PipelineResult:
    """
    Execute the full simulation pipeline.
    """
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

    analysis_options = create_analysis_options(
        inputs.analysis_options,
        use_gpu=inputs.use_gpu,
        use_incremental=inputs.use_incremental,
        solidification_params=inputs.solidification_params,
    )
    default_solver_mode = _get_default_solver_mode_override()
    if default_solver_mode is not None:
        analysis_options.solver_mode = default_solver_mode
    # LEGACY FALLBACK DISABLED — always use thermal_coupled unless explicitly overridden.
    # This ensures the full Numba-prange time-integration loop runs on all hardware.
    # To re-enable schedule_legacy as a fallback, set CASTSIM_DEFAULT_SOLVER_MODE=schedule_legacy
    # or pass solver_mode="schedule_legacy" explicitly in analysis_options.


    # Filling messages go to the log as well as the progress callback. The
    # callback lands in a status-bar label that the next message overwrites
    # seconds later, so without this the entire record of a fill -- including
    # "the solver diverged and the results were discarded" -- existed only for
    # as long as it took solidification to emit its first line. A run that
    # threw its fill away looked identical to one where filling was never
    # ticked.
    def _filling_message(message: str) -> None:
        # Console as well as the log and the progress callback. The callback
        # reaches a Qt status label and nothing else, so on the terminal the
        # mesh stage chatters through every part and then goes completely
        # silent for the minutes the fill takes -- which reads as a hang, not
        # as work. Every other long stage in this pipeline prints; this one
        # was the exception.
        print(f"  {message}", flush=True)
        _error_logger.info(message)
        _emit_message(progress_message_callback, message)

    filling_result = maybe_run_filling(
        voxel_mesh,
        alloy,
        analysis_options,
        inputs,
        cancel_callback=cancel_callback,
        progress_message_callback=_filling_message,
        detailed_progress_callback=detailed_progress_callback,
    )

    _emit_message(progress_message_callback, "Running solidification simulation...")
    _apply_thread_budget(inputs.thread_budget_provider)
    engine = SolidificationEngine(voxel_mesh, alloy, analysis_options)
    engine.initialize()
    if inputs.thread_budget_provider is not None:
        engine._thread_budget_provider = inputs.thread_budget_provider
    if filling_result is not None:
        # Attached whether or not the fill can be used. The Fill Time and Fill
        # Animation views disappear from the results list when a fill is
        # discarded, and a view that is simply absent is indistinguishable
        # from an analysis that was never enabled. This is what lets the UI
        # say which of the two happened, and why.
        engine._filling_summary = {
            "status": filling_result.status,
            "trustworthy": filling_result.trustworthy,
            "diverged_at_step": filling_result.diverged_at_step,
            "filled_voxel_count": filling_result.filled_voxel_count,
            "misrun_voxel_count": filling_result.misrun_voxel_count,
            "filled_fraction": filling_result.filled_fraction,
            "fill_seconds": filling_result.fill_seconds,
            "choke_thickness_cells": filling_result.choke_thickness_cells,
            # Which body is the narrowest, not just how narrow. One scalar
            # for "the gating system" cannot tell an operator whether to
            # re-mesh the gate or the sprue.
            "choke_thickness_by_part": dict(filling_result.choke_thickness_by_part),
            # Per body, since a part code covers every body of that type and
            # the scalar above is the thinnest of them.
            "choke_thickness_bodies_by_part": {
                name: list(values)
                for name, values in
                filling_result.choke_thickness_bodies_by_part.items()
            },
            # Bodies too small to judge the grid by; reported, not acted on.
            "choke_body_fragments_by_part": {
                name: list(sizes)
                for name, sizes in
                filling_result.choke_body_fragments_by_part.items()
            },
            "narrowest_choke_part": filling_result.narrowest_choke_part,
            "warnings": tuple(filling_result.warnings),
        }

    if filling_result is not None and not filling_result.trustworthy:
        # A diverged fill still returns arrays of the right shape, so seeding
        # from them would run a complete, plausible-looking solidification on
        # numerical debris. Fall back to the two-constant placeholder and say
        # so, rather than silently dressing up a failed solve as a result.
        _filling_message(
            f"Filling: DISCARDED -- the solver diverged at step "
            f"{filling_result.diverged_at_step:,}. Solidification will use the "
            f"uniform pour-temperature placeholder instead of the fill's "
            f"thermal field, and no Fill Time result will be available."
        )
    elif filling_result is not None:
        engine.set_initial_temperature_field(
            filling_result.temperature_c,
            fill_arrival_times=filling_result.fill_time_s,
        )
        engine._fill_frames = filling_result.frames
        # Report the handover explicitly. Without this the only sign a fill ran
        # is its progress chatter, so a run where filling was silently off is
        # indistinguishable from one where it ran and found nothing wrong.
        _emit_message(
            progress_message_callback,
            f"Filling: seeded initial temperature from the fill "
            f"({filling_result.filled_voxel_count:,} voxels filled, "
            f"{filling_result.misrun_voxel_count:,} misrun, "
            f"{filling_result.filled_fraction:.0%} of the metal) in "
            f"{filling_result.fill_seconds:.2f} s of pour",
        )

    def on_sim_progress(current: int, total: int, stage: str) -> None:
        _check_cancel(cancel_callback)
        if detailed_progress_callback is not None:
            detailed_progress_callback(current, total, stage)

    engine.run_simulation(
        show_progress=inputs.show_progress,
        progress_callback=on_sim_progress,
    )
    _check_cancel(cancel_callback)

    engine.export_results()
    result_data, visualization_data = build_visualization_payload(
        engine,
        voxel_mesh,
        total_volume=total_volume,
        bounds=np.asarray(bounds_to_use),
        max_voxels=inputs.max_voxels_for_visualization,
    )

    return PipelineResult(
        part_manager=part_manager,
        voxel_mesh=voxel_mesh,
        engine=engine,
        result_data=result_data,
        visualization_data=visualization_data,
    )
