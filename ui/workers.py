"""Background workers for simulation and ML recommendations."""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

from PyQt5.QtCore import QThread, pyqtSignal

from core.mesh_ml import log_mesh_training_example, recommend_mesh_settings
from core.simulation_pipeline import PipelineInputs, SimulationCancelledError, run_pipeline


class SimulationWorker(QThread):
    """Background thread for running simulation."""

    progress = pyqtSignal(str)
    detailed_progress = pyqtSignal(int, int, str)  # current, total, stage
    finished = pyqtSignal(bool, str)

    def __init__(
        self,
        stl_loader: Any,
        alloy_name: str,
        voxel_size: float,
        refinement_level: int,
        memory_mode: str = "auto",
        settings_origin: str = "manual",
        part_manager: Any = None,
        analysis_options: Optional[dict] = None,
        performance_options: Optional[dict] = None,
        solidification_params: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.stl_loader = stl_loader
        self.alloy_name = alloy_name
        self.voxel_size = voxel_size
        self.refinement_level = refinement_level
        self.memory_mode = memory_mode
        self.settings_origin = settings_origin
        self.part_manager = part_manager
        self.analysis_options = analysis_options or {}
        self.performance_options = performance_options or {}
        self.solidification_params = solidification_params or {}
        self.voxel_mesh: Any = None
        self.simulation: Any = None
        self._cancel_requested = threading.Event()

    def _progress_callback(self, current: int, total: int, stage: str) -> None:
        self.detailed_progress.emit(current, total, stage)

    def cancel(self) -> None:
        """Request cancellation for the active simulation."""
        self._cancel_requested.set()

    def run(self) -> None:
        mesh_generation_time = None
        simulation_time = None
        try:
            use_gpu = bool(self.performance_options.get("use_gpu", False))
            use_chunked = bool(self.performance_options.get("chunked_voxelization", False))
            use_incremental = bool(self.performance_options.get("incremental_solidification", False))
            part_assignment_mode = str(self.performance_options.get("part_assignment_mode", "auto"))
            part_assignment_workers = int(self.performance_options.get("part_assignment_workers", 0) or 0)
            part_assignment_batch_size = int(self.performance_options.get("part_assignment_batch_size", 2500) or 2500)

            perf_flags = []
            if use_gpu:
                perf_flags.append("GPU")
            if use_chunked:
                perf_flags.append("Chunked")
            if use_incremental:
                perf_flags.append("Incremental")
            perf_text = f" [{', '.join(perf_flags)}]" if perf_flags else ""

            self.progress.emit(f"Generating adaptive voxel mesh (Memory: {self.memory_mode.upper()}){perf_text}...")

            inputs = PipelineInputs(
                alloy_name=self.alloy_name,
                voxel_size=self.voxel_size,
                refinement_level=self.refinement_level,
                memory_mode=self.memory_mode,
                analysis_options=self.analysis_options,
                use_gpu=use_gpu,
                use_incremental=use_incremental,
                use_chunked_voxelization=use_chunked,
                part_assignment_mode=part_assignment_mode,
                part_assignment_workers=part_assignment_workers,
                part_assignment_batch_size=part_assignment_batch_size,
                show_progress=True,
                solidification_params=self.solidification_params,
            )

            pipeline_start = time.perf_counter()
            result = run_pipeline(
                inputs,
                part_manager=self.part_manager,
                stl_loader=self.stl_loader,
                progress_message_callback=self.progress.emit,
                detailed_progress_callback=self._progress_callback,
                cancel_callback=self._cancel_requested.is_set,
            )
            pipeline_time = time.perf_counter() - pipeline_start

            self.voxel_mesh = result.voxel_mesh
            self.simulation = result.engine
            mesh_generation_time = pipeline_time
            simulation_time = pipeline_time

            self.progress.emit("✓ Simulation complete!")
            self.finished.emit(True, "completed")

            voxel_count = self.voxel_mesh.get_voxel_count() if self.voxel_mesh is not None else 0

            log_mesh_training_example(
                self.stl_loader,
                {
                    "base_voxel_size": self.voxel_size,
                    "refinement_level": self.refinement_level,
                },
                {
                    "success": True,
                    "voxel_count": voxel_count,
                    "mesh_generation_time_s": mesh_generation_time,
                    "simulation_time_s": simulation_time,
                    "applied_ml": self.settings_origin == "ml",
                },
                origin=self.settings_origin,
            )
        except SimulationCancelledError:
            self.progress.emit("Simulation cancelled")
            self.finished.emit(False, "cancelled")
        except Exception as exc:
            try:
                from core.error_logger import logger as error_logger

                error_logger.log_simulation_error(
                    stage="simulation",
                    error=exc,
                    mesh_info={
                        "voxel_size": self.voxel_size,
                        "refinement_level": self.refinement_level,
                        "voxel_count": self.voxel_mesh.get_voxel_count() if self.voxel_mesh else 0,
                        "alloy": self.alloy_name,
                    },
                )
            except Exception:
                pass

            self.progress.emit(f"Error: {str(exc)}")
            voxel_count = self.voxel_mesh.get_voxel_count() if self.voxel_mesh is not None else 0
            log_mesh_training_example(
                self.stl_loader,
                {
                    "base_voxel_size": self.voxel_size,
                    "refinement_level": self.refinement_level,
                },
                {
                    "success": False,
                    "voxel_count": voxel_count,
                    "mesh_generation_time_s": mesh_generation_time,
                    "simulation_time_s": simulation_time,
                    "applied_ml": self.settings_origin == "ml",
                    "exception": str(exc),
                },
                origin=self.settings_origin,
            )
            self.finished.emit(False, "failed")


class MeshSuggestionWorker(QThread):
    """Background worker that generates ML mesh recommendations."""

    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, stl_loader: Any, target_accuracy: Optional[float] = None) -> None:
        super().__init__()
        self.stl_loader = stl_loader
        self.target_accuracy = target_accuracy

    def run(self) -> None:
        try:
            recommendation = recommend_mesh_settings(self.stl_loader, self.target_accuracy)
            self.result_ready.emit(recommendation)
        except Exception as exc:
            self.error.emit(str(exc))
