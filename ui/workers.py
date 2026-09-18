"""Background workers for simulation and ML recommendations."""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

from ui.qt import (
    QThread,
    Signal,
)

from core.mesh_ml import log_mesh_training_example, recommend_mesh_settings
from core.simulation_pipeline import PipelineInputs, SimulationCancelledError, run_pipeline


class SimulationWorker(QThread):
    """Background thread for running simulation."""

    progress = Signal(str)
    detailed_progress = Signal(int, int, str)  # current, total, stage
    finished = Signal(bool, str)

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
        pour_location: Optional[dict] = None,
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
        self.pour_location = pour_location
        self.voxel_mesh: Any = None
        self.simulation: Any = None
        self._cancel_requested = threading.Event()

    def _progress_callback(self, current: int, total: int, stage: str) -> None:
        self.detailed_progress.emit(current, total, stage)

    def cancel(self) -> None:
        """Request cancellation for the active simulation."""
        self._cancel_requested.set()

    def _maybe_capture_surrogate_training_data(self) -> None:
        """Opt-in capture of this run as surrogate training data."""
        try:
            from config.settings import get_settings

            settings = get_settings()
            if not settings.surrogate.auto_capture:
                return
            if self.simulation is None:
                return
            from core.surrogate.dataset import DEFAULT_DATASET_DIR, capture_run

            captured = capture_run(
                self.simulation,
                voxel_size=self.voxel_size,
                resolution=int(getattr(settings.surrogate, "grid_resolution", 64) or 64),
                out_dir=DEFAULT_DATASET_DIR,
            )
            if captured:
                self.progress.emit("✓ Surrogate training data captured")
        except Exception:
            pass

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
                pour_location=self.pour_location,
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

            self._maybe_capture_surrogate_training_data()

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

    result_ready = Signal(object)
    error = Signal(str)

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


class SurrogateTrainWorker(QThread):
    """Background worker that trains a surrogate model from captured data."""

    progress = Signal(str)
    detailed_progress = Signal(int, int, str)  # current, total, stage
    finished = Signal(bool, str, object)  # success, status, summary

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        grid_resolution: int = 64,
        generate_first: bool = False,
        model_name: str = "Trained Linear Surrogate",
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.grid_resolution = int(grid_resolution)
        self.generate_first = bool(generate_first)
        self.model_name = model_name
        self._cancel_requested = threading.Event()

    def cancel(self) -> None:
        """Request cancellation for the active training run."""
        self._cancel_requested.set()

    def run(self) -> None:
        try:
            from core.surrogate.training import (
                TrainingCancelledError,
                retrain_from_dataset,
                synthetic_geometries,
            )

            def _detailed(current: int, total: int) -> None:
                self.detailed_progress.emit(current, total, "Training")

            summary = retrain_from_dataset(
                data_dir=self.data_dir,
                output_dir=self.output_dir,
                grid=self.grid_resolution,
                name=self.model_name,
                generate_first=self.generate_first,
                progress_cb=self.progress.emit,
                detailed_cb=_detailed,
                cancel_cb=self._cancel_requested.is_set,
            )
            self.progress.emit("✓ Surrogate model trained!")
            self.finished.emit(True, "completed", summary)
        except TrainingCancelledError:
            self.progress.emit("Training cancelled")
            self.finished.emit(False, "cancelled", None)
        except Exception as exc:
            self.progress.emit(f"Error: {str(exc)}")
            self.finished.emit(False, "failed", None)


class SurrogateWorker(QThread):
    """Background worker that runs a surrogate (NPU/ML) simulation."""

    progress = Signal(str)
    detailed_progress = Signal(int, int, str)  # current, total, stage
    finished = Signal(bool, str)

    def __init__(
        self,
        stl_loader: Any,
        model: Any,
        alloy_name: str,
        voxel_size: float,
        refinement_level: int,
        memory_mode: str = "auto",
        part_manager: Any = None,
        analysis_options: Optional[dict] = None,
        solidification_params: Optional[dict] = None,
        pour_location: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.stl_loader = stl_loader
        self.model = model
        self.alloy_name = alloy_name
        self.voxel_size = voxel_size
        self.refinement_level = refinement_level
        self.memory_mode = memory_mode
        self.part_manager = part_manager
        self.analysis_options = analysis_options or {}
        self.solidification_params = solidification_params or {}
        self.pour_location = pour_location
        self.voxel_mesh: Any = None
        self.simulation: Any = None
        self._cancel_requested = threading.Event()

    def cancel(self) -> None:
        """Request cancellation for the active surrogate run."""
        self._cancel_requested.set()

    def _progress_callback(self, current: int, total: int, stage: str) -> None:
        self.detailed_progress.emit(current, total, stage)

    def run(self) -> None:
        try:
            from core.surrogate import run_surrogate_pipeline

            inputs = PipelineInputs(
                alloy_name=self.alloy_name,
                voxel_size=self.voxel_size,
                refinement_level=self.refinement_level,
                memory_mode=self.memory_mode,
                analysis_options=self.analysis_options,
                use_gpu=False,
                use_incremental=False,
                show_progress=True,
                solidification_params=self.solidification_params,
                pour_location=self.pour_location,
            )
            result = run_surrogate_pipeline(
                inputs,
                model=self.model,
                part_manager=self.part_manager,
                stl_loader=self.stl_loader,
                progress_message_callback=self.progress.emit,
                detailed_progress_callback=self._progress_callback,
                cancel_callback=self._cancel_requested.is_set,
            )
            self.voxel_mesh = result.voxel_mesh
            self.simulation = result.engine
            self.progress.emit("✓ Surrogate simulation complete!")
            self.finished.emit(True, "completed")
        except SimulationCancelledError:
            self.progress.emit("Surrogate simulation cancelled")
            self.finished.emit(False, "cancelled")
        except Exception as exc:
            self.progress.emit(f"Error: {str(exc)}")
            self.finished.emit(False, "failed")


class DeepTrainWorker(QThread):
    """Background worker that trains the deep 3D U-Net surrogate."""

    progress = Signal(str)
    detailed_progress = Signal(int, int, str)  # current, total, stage
    finished = Signal(bool, str, object)  # success, status, summary

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        grid_resolution: int = 64,
        epochs: int = 100,
        batch_size: int = 4,
        learning_rate: float = 1e-3,
        min_records: int = 4,
        model_name: str = "Trained Deep Surrogate (3D U-Net)",
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.grid_resolution = int(grid_resolution)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.min_records = int(min_records)
        self.model_name = model_name
        self._cancel_requested = threading.Event()

    def cancel(self) -> None:
        """Request cancellation for the active training run."""
        self._cancel_requested.set()

    def run(self) -> None:
        try:
            from core.deep_model.training import (
                DeepTrainingCancelledError,
                train_deep_model,
            )
            from core.deep_model.autotrain import (
                AutoTrainState,
                dataset_signature,
                save_state,
            )

            def _detailed(current: int, total: int) -> None:
                self.detailed_progress.emit(current, total, "Training")

            summary = train_deep_model(
                data_dir=self.data_dir,
                output_dir=self.output_dir,
                grid=self.grid_resolution,
                epochs=self.epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                name=self.model_name,
                min_records=self.min_records,
                progress_cb=self.progress.emit,
                detailed_cb=_detailed,
                cancel_cb=self._cancel_requested.is_set,
            )
            # Refresh the auto-train state so cooldown applies after a manual train.
            signature = dataset_signature(self.data_dir)
            state = AutoTrainState(
                signature_count=signature[0] if signature else 0,
                signature_mtimes=signature[1] if signature else (),
                last_completed_at=time.time(),
                model_exists=True,
            )
            save_state(self.output_dir, state)
            self.progress.emit("✓ Deep model trained!")
            self.finished.emit(True, "completed", summary)
        except DeepTrainingCancelledError:
            self.progress.emit("Training cancelled")
            self.finished.emit(False, "cancelled", None)
        except Exception as exc:
            self.progress.emit(f"Error: {str(exc)}")
            self.finished.emit(False, "failed", None)


class DeepExportWorker(QThread):
    """Background worker that exports the deep model to an NPU target."""

    progress = Signal(str)
    finished = Signal(bool, str, object)  # success, status, result

    def __init__(self, model_dir: str, target: str, out_dir: str) -> None:
        super().__init__()
        self.model_dir = model_dir
        self.target = target
        self.out_dir = out_dir
        self._cancel_requested = threading.Event()

    def cancel(self) -> None:
        """Request cancellation for the active export run."""
        self._cancel_requested.set()

    def run(self) -> None:
        try:
            from core.deep_model.export import export_for_target

            result = export_for_target(
                self.model_dir,
                self.target,
                out_dir=self.out_dir,
                progress_cb=self.progress.emit,
                cancel_cb=self._cancel_requested.is_set,
            )
            self.progress.emit(f"✓ {self.target.upper()} export complete")
            self.finished.emit(True, "completed", result)
        except Exception as exc:
            self.progress.emit(f"Error: {str(exc)}")
            self.finished.emit(False, "failed", None)


class DeepAutoTrainWorker(QThread):
    """Background worker that evaluates the retrain policy, then trains.

    Watcher-driven (the main window owns the ``QFileSystemWatcher`` on the
    dataset directory): this worker applies the policy (min records, cooldown,
    dataset-changed, torch present, no other training/export) and either skips
    (emitting a reason) or retrains on the entire accumulated dataset.
    """

    progress = Signal(str)
    detailed_progress = Signal(int, int, str)  # current, total, stage
    finished = Signal(bool, str, object)  # success, status, summary
    skipped = Signal(str)

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        grid_resolution: int = 64,
        epochs: int = 100,
        batch_size: int = 4,
        learning_rate: float = 1e-3,
        min_records: int = 4,
        cooldown_seconds: int = 300,
        state: Any = None,
        export_in_progress: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.grid_resolution = int(grid_resolution)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.min_records = int(min_records)
        self.cooldown_seconds = int(cooldown_seconds)
        self.state = state  # AutoTrainState; updated + persisted on completion
        self.export_in_progress = bool(export_in_progress)
        self._cancel_requested = threading.Event()

    def cancel(self) -> None:
        """Request cancellation for the active auto-train run."""
        self._cancel_requested.set()

    def run(self) -> None:
        try:
            from core.deep_model.autotrain import (
                AutoTrainState,
                dataset_signature,
                save_state,
                should_retrain,
            )
            from core.deep_model.training import (
                DeepTrainingCancelledError,
                train_deep_model,
            )

            torch_available = True
            try:
                import torch  # noqa: F401
            except ImportError:
                torch_available = False

            signature = dataset_signature(self.data_dir)
            should_run, reason = should_retrain(
                enabled=True,
                torch_available=torch_available,
                training_active=False,
                export_in_progress=self.export_in_progress,
                current_signature=signature,
                state=self.state,
                min_records=self.min_records,
                cooldown_seconds=self.cooldown_seconds,
            )
            if not should_run:
                self.skipped.emit(reason)
                self.finished.emit(False, "skipped", None)
                return

            self.progress.emit(reason)
            summary = train_deep_model(
                data_dir=self.data_dir,
                output_dir=self.output_dir,
                grid=self.grid_resolution,
                epochs=self.epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                name="Trained Deep Surrogate (3D U-Net)",
                min_records=self.min_records,
                progress_cb=self.progress.emit,
                detailed_cb=lambda c, t: self.detailed_progress.emit(c, t, "Auto-Train"),
                cancel_cb=self._cancel_requested.is_set,
            )
            state = AutoTrainState(
                signature_count=signature[0] if signature else 0,
                signature_mtimes=signature[1] if signature else (),
                last_completed_at=time.time(),
                model_exists=True,
            )
            save_state(self.output_dir, state)
            self.progress.emit("✓ Auto-train complete")
            self.finished.emit(True, "completed", summary)
        except DeepTrainingCancelledError:
            self.progress.emit("Auto-train cancelled")
            self.finished.emit(False, "cancelled", None)
        except Exception as exc:
            self.progress.emit(f"Error: {str(exc)}")
            self.finished.emit(False, "failed", None)
