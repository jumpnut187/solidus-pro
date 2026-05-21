"""
Shared simulation pipeline used by desktop UI and web API.

This module centralizes part loading, voxel mesh generation, simulation
execution, and result packaging to avoid duplicated orchestration logic.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, cast
import os

import numpy as np

from core.materials import get_alloy
from core.part_manager import PartManager, PartType, infer_part_type_from_filename
from core.solidification import (
    AnalysisOptions,
    SOLVER_MODE_SCHEDULE_LEGACY,
    SOLVER_MODE_THERMAL_COUPLED,
    VALID_SOLVER_MODES,
    SolidificationEngine,
)
from core.voxel_mesh import AdaptiveVoxelMesh


ProgressMessageCallback = Callable[[str], None]
DetailedProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


def _should_use_legacy_solver_default(options: Optional[Dict[str, Any]]) -> bool:
    """Return True when the caller did not explicitly request a solver mode."""
    if options is None:
        return True

    return "solver_mode" not in options


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


def _emit_message(callback: Optional[ProgressMessageCallback], message: str) -> None:
    if callback is not None:
        callback(message)


def _check_cancel(callback: Optional[CancelCallback]) -> None:
    if callback is not None and callback():
        raise SimulationCancelledError("Cancelled")


def build_part_manager_from_parts(parts_data: Sequence[Dict[str, Any]]) -> PartManager:
    """
    Build a PartManager instance from serialized web API part payloads.
    """
    part_manager = PartManager()
    type_map = {
        "cast": PartType.CAST,
        "riser": PartType.RISER,
        "riser_post": PartType.RISER_POST,
        "chill": PartType.CHILL,
        "core": PartType.CORE,
    }

    for part_data in parts_data:
        filepath = str(part_data.get("filepath", ""))
        if not filepath or not os.path.exists(filepath):
            raise FileNotFoundError(f"Part file not found: {filepath}")

        part_id = part_manager.load_part(filepath, name=part_data.get("name"))
        if not part_id:
            raise RuntimeError(f"Failed to load part: {filepath}")

        part = part_manager.get_part(part_id)
        if part is None:
            raise RuntimeError(f"Loaded part missing from manager: {part_id}")

        part.set_position(
            float(part_data.get("position_x", 0.0)),
            float(part_data.get("position_y", 0.0)),
            float(part_data.get("position_z", 0.0)),
        )
        part.set_rotation(
            float(part_data.get("rotation_x", 0.0)),
            float(part_data.get("rotation_y", 0.0)),
            float(part_data.get("rotation_z", 0.0)),
        )
        part.set_scale(float(part_data.get("scale", 1.0)))
        part_type_key = str(part_data.get("part_type", "cast"))
        part_type = cast(PartType, type_map.get(part_type_key, PartType.CAST))
        inferred_type = infer_part_type_from_filename(filepath)
        part_name = str(part_data.get("name", "")).lower()
        if inferred_type != PartType.RISER_POST:
            if any(keyword in part_name for keyword in ("riser post", "riser_post", "post", "neck")):
                inferred_type = PartType.RISER_POST
        if part_type == PartType.RISER and inferred_type == PartType.RISER_POST:
            part_type = PartType.RISER_POST
        part.set_part_type(part_type)

    return part_manager


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
        shrinkage = np.asarray(engine.shrinkage_probability)
        visualization_data["shrinkage"] = shrinkage[indices].tolist()
        result_data["shrinkage_stats"] = {
            "max": float(np.max(shrinkage)),
            "mean": float(np.mean(shrinkage)),
            "affected_count": int(np.sum(shrinkage > 0.05)),
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

    if voxel_mesh.voxel_states is not None:
        fraction_liquid = 1.0 - np.asarray(voxel_mesh.voxel_states).astype(float)
        visualization_data["fraction_liquid"] = fraction_liquid[indices].tolist()
    if engine.solidification_times is not None:
        solidification_times = np.asarray(engine.solidification_times).astype(float)
        visualization_data["solidification_times"] = solidification_times[indices].tolist()

    if engine.fraction_liquid_history:
        fl_history = []
        for snapshot in engine.fraction_liquid_history:
            fl_frame = 1.0 - np.asarray(snapshot)
            fl_history.append(fl_frame[indices].tolist())
        visualization_data["fraction_liquid_history"] = fl_history
        visualization_data["fraction_liquid_times"] = engine.fraction_liquid_times or []
        result_data["fraction_liquid_snapshots"] = len(fl_history)

    return result_data, visualization_data


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

    mesh_to_use = None
    bounds_to_use = None
    total_volume = 0.0
    parts_with_types = []

    if part_manager is not None and part_manager.get_part_count() > 0:
        combined_mesh = part_manager.get_combined_mesh()
        combined_bounds = part_manager.get_combined_bounds()
        if combined_mesh is None or combined_bounds is None:
            raise RuntimeError("Failed to build combined mesh from parts")
        mesh_to_use = combined_mesh
        bounds_to_use = combined_bounds
        total_volume = float(part_manager.get_total_volume())
        parts_with_types = part_manager.get_parts_with_types()
    elif stl_loader is not None and getattr(stl_loader, "trimesh_obj", None) is not None:
        mesh_to_use = stl_loader.trimesh_obj
        bounds_to_use = stl_loader.get_bounds_array()
        total_volume = float(getattr(stl_loader, "volume", 0.0) or 0.0)
    else:
        raise ValueError("No valid geometry source provided to simulation pipeline")

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
    )

    def on_mesh_progress(current: int, total: int, stage: str) -> None:
        _check_cancel(cancel_callback)
        if detailed_progress_callback is not None:
            detailed_progress_callback(current, total, stage)

    success = voxel_mesh.generate(
        mesh_to_use,
        bounds_to_use,
        show_progress=False,
        progress_callback=on_mesh_progress,
    )
    if not success:
        raise RuntimeError("Voxel mesh generation failed")

    _check_cancel(cancel_callback)
    if parts_with_types:
        _emit_message(progress_message_callback, "Assigning part types (Cast/Riser/Chill)...")
        voxel_mesh.assign_part_types(parts_with_types)

    mem_stats = voxel_mesh.get_memory_stats()
    _emit_message(
        progress_message_callback,
        (
            f"Memory: RAM={mem_stats['ram_mb']:.1f}MB, "
            f"Disk={mem_stats['disk_mb']:.1f}MB, "
            f"Available={mem_stats['available_gb']:.1f}GB"
        ),
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
    elif (
        analysis_options.solver_mode == SOLVER_MODE_THERMAL_COUPLED
        and _should_use_legacy_solver_default(inputs.analysis_options)
        and not analysis_options.niyama
        and (
            analysis_options.fraction_liquid
            or analysis_options.hot_tear
            or analysis_options.shrinkage
            or analysis_options.feeding_path
        )
    ):
        # Preserve historical timing/history behavior for legacy callers that do not
        # explicitly pin the new thermal solver.
        analysis_options.solver_mode = SOLVER_MODE_SCHEDULE_LEGACY

    _emit_message(progress_message_callback, "Running solidification simulation...")
    engine = SolidificationEngine(voxel_mesh, alloy, analysis_options)
    engine.initialize()

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
