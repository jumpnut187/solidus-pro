from __future__ import annotations

import multiprocessing
from multiprocessing.managers import BaseManager
import os
import psutil
import sys
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, cast

from config.hardware_config import hardware

JOB_LOG_LINES = 500
from web.services.fair_share import (
    can_start_simulation,
    compute_thread_share,
    make_thread_budget_provider,
    resolve_usable_cores,
)

if sys.platform == 'darwin':
    multiprocessing.set_start_method('spawn', force=True)

def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default

class QueueConfig:
    def __init__(
        self,
        max_concurrent_simulations: int = 1,
        max_queue_size: int = 6,
        max_simulations_per_user: int = 1,
        reserved_cores: int = 1,
        reserve_memory_mb: int = 2048,
        min_sim_memory_mb: int = 2048,
    ) -> None:
        self.max_concurrent_simulations = max_concurrent_simulations
        self.max_queue_size = max_queue_size
        self.max_simulations_per_user = max_simulations_per_user
        self.reserved_cores = reserved_cores
        self.reserve_memory_mb = reserve_memory_mb
        self.min_sim_memory_mb = min_sim_memory_mb

def default_queue_config() -> QueueConfig:
    physical_cores = int(getattr(hardware, "physical_cores", 0) or (os.cpu_count() or 4))
    auto_concurrency = max(1, min(4, physical_cores // 4))

    try:
        total_mb = psutil.virtual_memory().total / (1024 * 1024)
    except Exception:
        total_mb = 16384.0
    auto_reserve_mb = int(max(2048.0, total_mb * 0.10))

    return QueueConfig(
        max_concurrent_simulations=_env_int(
            "CASTSIM_MAX_CONCURRENT_SIMULATIONS", auto_concurrency
        ),
        reserved_cores=_env_int("CASTSIM_WEB_RESERVED_CORES", 1),
        reserve_memory_mb=_env_int("CASTSIM_WEB_RESERVE_MEMORY_MB", auto_reserve_mb),
        min_sim_memory_mb=_env_int("CASTSIM_WEB_MIN_SIM_MEMORY_MB", 2048),
    )

def build_pipeline_inputs(
    sim_data: Dict[str, Any],
    *,
    thread_budget_provider: Any = None,
) -> Any:
    from core.simulation_pipeline import PipelineInputs

    performance = sim_data.get("performance_options") or {}
    analysis = sim_data.get("analysis_options") or {}
    return PipelineInputs(
        alloy_name=str(sim_data.get("alloy", "WCB")),
        voxel_size=float(sim_data.get("voxel_size", 0.1)),
        refinement_level=int(sim_data.get("refinement_level", 3)),
        memory_mode=str(sim_data.get("memory_mode", "auto")),
        analysis_options=analysis,
        use_gpu=bool(performance.get("use_gpu", False)),
        use_incremental=bool(performance.get("use_incremental", False)),
        use_chunked_voxelization=bool(performance.get("use_chunked_voxelization", False)),
        part_assignment_mode=str(performance.get("part_assignment_mode", "auto")),
        part_assignment_workers=int(performance.get("part_assignment_workers", 0) or 0),
        part_assignment_batch_size=int(performance.get("part_assignment_batch_size", 2500) or 2500),
        max_voxels_for_visualization=int(performance.get("max_voxels_for_visualization", 80_000) or 80_000),
        solidification_params=sim_data.get("solidification_params") or {},
        thread_budget_provider=thread_budget_provider,
        pour_location=sim_data.get("pour_location"),
        voxel_cut_cells=(
            None if performance.get("voxel_cut_cells") is None else bool(performance.get("voxel_cut_cells"))
        ),
    )

def _simulation_process_worker(
    sim_id: str,
    sim_data: Dict[str, Any],
    parts_data: list[Dict[str, Any]],
    user_ip: str,
    db_path: str,
    base_dir: str,
    active_simulations: Dict[str, Any],
    simulation_results: Dict[str, Any],
    stop_flags: Dict[str, bool],
    running_count,
    user_simulation_counts: Dict[str, int],
    queue_lock,
    queued_simulations: Dict[str, Any],
    max_simulations_per_user: int,
    thread_budget,
    sim_memory_mb: Dict[str, Any],
    usable_cores: int,
    max_concurrent_simulations: int,
    reserve_memory_mb: int,
    min_sim_memory_mb: int,
    simulation_logs: Dict[str, Any],
) -> None:
    def _check_cancelled() -> bool:
        return stop_flags.get(sim_id, False)

    def _log(message: str) -> None:
        try:
            with queue_lock:
                entry = dict(simulation_logs.get(sim_id) or {"lines": [], "first": 0})
                lines = list(entry.get("lines", []))
                if lines and lines[-1].endswith(message):
                    return
                lines.append(f"{time.strftime('%H:%M:%S')}  {message}")
                first = int(entry.get("first", 0))
                if len(lines) > JOB_LOG_LINES:
                    dropped = len(lines) - JOB_LOG_LINES
                    lines = lines[dropped:]
                    first += dropped
                simulation_logs[sim_id] = {"lines": lines, "first": first}
        except Exception:
            pass

    try:
        with queue_lock:
            simulation_logs[sim_id] = {"lines": [], "first": 0}
    except Exception:
        pass

    def _update_active(**kwargs: Any) -> None:
        with queue_lock:
            if sim_id in active_simulations:
                entry = dict(active_simulations[sim_id])
                entry.update(kwargs)
                active_simulations[sim_id] = entry

    def _set_progress(progress: int, message: str) -> None:
        _log(message)
        with queue_lock:
            if sim_id in active_simulations:
                entry = dict(active_simulations[sim_id])
                entry["progress"] = progress
                entry["message"] = message
                active_simulations[sim_id] = entry

    def _chain_memory_ok() -> bool:
        try:
            available_mb = psutil.virtual_memory().available / (1024 * 1024)
        except Exception:
            available_mb = float(reserve_memory_mb + min_sim_memory_mb)
        return can_start_simulation(
            running_count=max(0, int(running_count.value)),
            max_concurrent=max_concurrent_simulations,
            available_mb=available_mb,
            active_memory_mb=float(sum(sim_memory_mb.values())),
            reserve_mb=float(reserve_memory_mb),
            min_sim_memory_mb=float(min_sim_memory_mb),
        )

    try:
        _set_progress(4, "Loading assembly...")

        with queue_lock:
            sim_memory_mb[sim_id] = float(min_sim_memory_mb)

        from web.payloads import build_part_manager
        from core.simulation_pipeline import SimulationCancelledError, run_pipeline

        part_manager = build_part_manager(parts_data)
        if _check_cancelled():
            raise SimulationCancelledError("Cancelled")

        inputs = build_pipeline_inputs(
            sim_data,
            thread_budget_provider=make_thread_budget_provider(thread_budget),
        )

        def progress_message(message: str) -> None:
            _log(message)
            if message.startswith("Generating adaptive voxel mesh"):
                _set_progress(15, "Generating adaptive voxel mesh...")
            elif message.startswith("Assigning part types"):
                _set_progress(42, "Assigning cast, riser, chill, and riser-post voxels...")
            elif message.startswith("Running solidification simulation"):
                _set_progress(55, "Running solidification physics...")
            else:
                _set_progress(82, message)

        def detailed_progress(current: int, total: int, stage: str) -> None:
            stage_l = stage.lower()
            if stage_l.startswith("solid"):
                progress = 55 + int((current / max(total, 1)) * 32)
                _set_progress(progress, f"Solidifying: {current}%")
            else:
                progress = 15 + int((current / max(total, 1)) * 25)
                _set_progress(progress, f"Voxelizing: {stage} ({current}/{total})")

        result = run_pipeline(
            inputs,
            part_manager=part_manager,
            progress_message_callback=progress_message,
            detailed_progress_callback=detailed_progress,
            cancel_callback=_check_cancelled,
        )

        try:
            mem_stats = result.voxel_mesh.get_memory_stats()
            actual_mb = float(mem_stats.get("ram_mb", 0.0) or 0.0)
            if actual_mb > 0.0:
                with queue_lock:
                    sim_memory_mb[sim_id] = actual_mb
        except Exception:
            pass

        from pathlib import Path
        from web.payloads import build_fork_artifact, write_artifact
        from web import repository as repo

        from web.payloads import simulated_parts_snapshot

        artifact = build_fork_artifact(result)
        artifact_dir = Path(base_dir) / "web" / "static" / "temp" / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{sim_id}.json"
        write_artifact(artifact_path, artifact)
        summary = {**artifact["summary"], "simulated_parts": simulated_parts_snapshot(parts_data)}
        repo.mark_project_completed(db_path, sim_id, summary, str(artifact_path))

        with queue_lock:
            simulation_results.pop(sim_id, None)

        _log("Simulation complete")
        _update_active(status="completed", message="Simulation complete", progress=100)

    except Exception as exc:
        from core.simulation_pipeline import SimulationCancelledError
        from web import repository as repo

        if isinstance(exc, SimulationCancelledError):
            _log("Simulation cancelled")
            _update_active(status="cancelled", message="Simulation cancelled", progress=0)
            repo.mark_project_status(db_path, sim_id, "cancelled")
        else:
            import traceback
            traceback.print_exc()
            for line in traceback.format_exc().splitlines():
                _log(line)
            _update_active(status="error", message=str(exc), progress=0)
            repo.mark_project_status(db_path, sim_id, "error")

    finally:
        with queue_lock:
            running_count.value = max(0, running_count.value - 1)
            counts = dict(user_simulation_counts)
            if user_ip in counts:
                user_simulation_counts[user_ip] = max(0, counts[user_ip] - 1)
                if counts[user_ip] <= 1:
                    user_simulation_counts.pop(user_ip, None)
            stop_flags.pop(sim_id, None)
            active_simulations.pop(sim_id, None)
            sim_memory_mb.pop(sim_id, None)

            next_sim_id = None
            next_sim_info = None
            if queued_simulations:
                next_sim_id, next_sim_info = next(iter(queued_simulations.items()))
                del queued_simulations[next_sim_id]
                if _chain_memory_ok():
                    running_count.value += 1
                else:
                    queued_simulations[next_sim_id] = next_sim_info
                    next_sim_id = None
                    next_sim_info = None

            thread_budget.value = compute_thread_share(
                usable_cores, max(1, running_count.value)
            )

        if next_sim_id and next_sim_info:
            _simulation_process_worker(
                sim_id=next_sim_id,
                sim_data=next_sim_info["sim_data"],
                parts_data=next_sim_info["parts_data"],
                user_ip=next_sim_info["user_ip"],
                db_path=db_path,
                base_dir=base_dir,
                active_simulations=active_simulations,
                simulation_results=simulation_results,
                stop_flags=stop_flags,
                running_count=running_count,
                user_simulation_counts=user_simulation_counts,
                queue_lock=queue_lock,
                queued_simulations=queued_simulations,
                max_simulations_per_user=max_simulations_per_user,
                thread_budget=thread_budget,
                sim_memory_mb=sim_memory_mb,
                usable_cores=usable_cores,
                max_concurrent_simulations=max_concurrent_simulations,
                reserve_memory_mb=reserve_memory_mb,
                min_sim_memory_mb=min_sim_memory_mb,
                simulation_logs=simulation_logs,
            )

class SimulationQueueManager:

    def __init__(self, config: Optional[QueueConfig] = None) -> None:
        self.config = config or QueueConfig()
        self._manager: Optional[BaseManager] = None
        self.queue_lock: Any = None

        self.active_simulations: Any = None
        self.simulation_results: Any = None
        self.simulation_processes: Any = None
        self.simulation_stop_flags: Any = None
        self.queued_simulations: Any = None
        self.user_simulation_counts: Any = None
        self.running_count: Any = None
        self.thread_budget: Any = None
        self.sim_memory_mb: Any = None
        self._db_path: Optional[str] = None
        self._base_dir: Optional[str] = None

    def _init_manager(self) -> None:
        if self._manager is not None:
            return
        self._manager = multiprocessing.Manager()
        self.queue_lock = self._manager.Lock()
        self.active_simulations = self._manager.dict()
        self.simulation_results = self._manager.dict()
        self.simulation_processes = {}
        self.simulation_stop_flags = self._manager.dict()
        self.queued_simulations = self._manager.dict()
        self.user_simulation_counts = self._manager.dict()
        self.running_count = self._manager.Value('i', 0)
        self.thread_budget = self._manager.Value('i', 0)
        self.sim_memory_mb = self._manager.dict()
        self.simulation_logs = self._manager.dict()
    def _usable_cores(self) -> int:
        logical = int(getattr(hardware, "logical_cores", 0) or (os.cpu_count() or 1))
        return resolve_usable_cores(logical, self.config.reserved_cores)

    def _memory_ok_to_start(self) -> bool:
        try:
            available_mb = psutil.virtual_memory().available / (1024 * 1024)
        except Exception:
            available_mb = float(self.config.reserve_memory_mb + self.config.min_sim_memory_mb)
        return can_start_simulation(
            running_count=int(self.running_count.value),
            max_concurrent=self.config.max_concurrent_simulations,
            available_mb=available_mb,
            active_memory_mb=float(sum((self.sim_memory_mb or {}).values())),
            reserve_mb=float(self.config.reserve_memory_mb),
            min_sim_memory_mb=float(self.config.min_sim_memory_mb),
        )

    def set_paths(self, db_path: str, base_dir: str) -> None:
        self._db_path = db_path
        self._base_dir = base_dir

    def can_user_submit(self, user_ip: str) -> bool:
        self._init_manager()
        with self.queue_lock:
            return bool(
                self.user_simulation_counts.get(user_ip, 0)
                < self.config.max_simulations_per_user
            )

    def increment_user_count(self, user_ip: str) -> None:
        self._init_manager()
        with self.queue_lock:
            self.user_simulation_counts[user_ip] = self.user_simulation_counts.get(user_ip, 0) + 1

    def decrement_user_count(self, user_ip: str) -> None:
        self._init_manager()
        with self.queue_lock:
            counts = dict(self.user_simulation_counts)
            if user_ip in counts:
                self.user_simulation_counts[user_ip] = max(0, counts[user_ip] - 1)
                if counts[user_ip] <= 1:
                    self.user_simulation_counts.pop(user_ip, None)

    def get_queue_position(self, sim_id: str) -> int:
        self._init_manager()
        with self.queue_lock:
            queued = self.queued_simulations
            if queued is None or sim_id not in queued:
                return 0
            return list(queued.keys()).index(sim_id) + 1

    def get_queue_status(self, user_ip: str) -> Dict[str, int]:
        self._init_manager()
        with self.queue_lock:
            try:
                available_mb = psutil.virtual_memory().available / (1024 * 1024)
            except Exception:
                available_mb = 0.0
            return {
                "running": self.running_count.value,
                "max_concurrent": self.config.max_concurrent_simulations,
                "queued": len(self.queued_simulations or {}),
                "max_queue": self.config.max_queue_size,
                "your_simulations": self.user_simulation_counts.get(user_ip, 0),
                "max_per_user": self.config.max_simulations_per_user,
                "thread_share": self.thread_budget.value,
                "reserved_cores": self.config.reserved_cores,
                "memory_available_mb": int(available_mb),
                "active_memory_mb": int(sum((self.sim_memory_mb or {}).values())),
            }

    def submit(
        self,
        sim_id: str,
        sim_data: Dict[str, Any],
        parts_data: list[Dict[str, Any]],
        user_ip: str,
    ) -> Dict[str, Any]:
        self._init_manager()

        if not self.can_user_submit(user_ip):
            return {
                "status": "rate_limited",
                "error": (
                    f"You have reached the maximum of {self.config.max_simulations_per_user} simulations. "
                    "Please wait for one to complete."
                ),
            }

        self.increment_user_count(user_ip)
        should_rollback_user_count = False
        with self.queue_lock:
            can_run_now = (
                self.running_count.value < self.config.max_concurrent_simulations
                and self._memory_ok_to_start()
            )
            if can_run_now:
                self.running_count.value += 1
                self.thread_budget.value = compute_thread_share(
                    self._usable_cores(), max(1, self.running_count.value)
                )
            else:
                if len(self.queued_simulations or {}) >= self.config.max_queue_size:
                    should_rollback_user_count = True
                    can_run_now = False
                else:
                    queued = self.queued_simulations
                    if queued is None:
                        queued = {}
                        self.queued_simulations = queued
                    queued[sim_id] = {
                        "user_ip": user_ip,
                        "sim_data": sim_data,
                        "parts_data": parts_data,
                        "queued_at": time.time(),
                    }
                    can_run_now = False
        if should_rollback_user_count:
            self.decrement_user_count(user_ip)
            return {
                "status": "queue_full",
                "error": "Server is busy. Queue is full. Please try again later.",
            }

        if can_run_now:
            self._start_running_simulation(
                sim_id=sim_id,
                sim_data=sim_data,
                parts_data=parts_data,
                user_ip=user_ip,
                message="Initializing...",
            )
            return {"status": "started", "sim_id": sim_id}

        position = self.get_queue_position(sim_id)
        with self.queue_lock:
            active = self.active_simulations
            if active is None:
                active = {}
                self.active_simulations = active
            active[sim_id] = {
                "status": "queued",
                "progress": 0,
                "message": f"Queued (position {position})",
                "queue_position": position,
                "queued_at": time.time(),
                "user_ip": user_ip,
            }
            queue_size = len(self.queued_simulations or {})

        return {
            "status": "queued",
            "sim_id": sim_id,
            "queue_position": position,
            "queue_size": queue_size,
        }

    def _start_running_simulation(
        self,
        *,
        sim_id: str,
        sim_data: Dict[str, Any],
        parts_data: list[Dict[str, Any]],
        user_ip: str,
        message: str,
    ) -> None:
        self._init_manager()
        db_path = self._db_path or ""
        base_dir = self._base_dir or ""

        with self.queue_lock:
            active = self.active_simulations
            if active is None:
                active = {}
                self.active_simulations = active
            active[sim_id] = {
                "status": "running",
                "progress": 0,
                "message": message,
                "started_at": time.time(),
                "user_ip": user_ip,
            }
            stop_flags = self.simulation_stop_flags
            if stop_flags is None:
                stop_flags = {}
                self.simulation_stop_flags = stop_flags
            stop_flags[sim_id] = False

        proc = multiprocessing.Process(
            target=_simulation_process_worker,
            args=(
                sim_id,
                sim_data,
                parts_data,
                user_ip,
                db_path,
                base_dir,
                self.active_simulations,
                self.simulation_results,
                self.simulation_stop_flags,
                self.running_count,
                self.user_simulation_counts,
                self.queue_lock,
                self.queued_simulations,
                self.config.max_simulations_per_user,
                self.thread_budget,
                self.sim_memory_mb,
                self._usable_cores(),
                self.config.max_concurrent_simulations,
                self.config.reserve_memory_mb,
                self.config.min_sim_memory_mb,
                self.simulation_logs,
            ),
            daemon=False,
        )
        proc.start()
        with self.queue_lock:
            self.simulation_processes[sim_id] = proc

    def should_cancel(self, sim_id: str) -> bool:
        self._init_manager()
        with self.queue_lock:
            stop_flags = self.simulation_stop_flags
            if stop_flags is None:
                return False
            return bool(stop_flags.get(sim_id, False))

    def set_progress(self, sim_id: str, progress: int, message: str) -> None:
        self._init_manager()
        with self.queue_lock:
            active = self.active_simulations
            if active is not None and sim_id in active:
                entry = dict(active[sim_id])
                entry["progress"] = progress
                entry["message"] = message
                active[sim_id] = entry

    def set_status(self, sim_id: str, status: str, message: str, progress: int) -> None:
        self._init_manager()
        with self.queue_lock:
            active = self.active_simulations
            if active is not None and sim_id in active:
                entry = dict(active[sim_id])
                entry["status"] = status
                entry["message"] = message
                entry["progress"] = progress
                active[sim_id] = entry

    def save_result(self, sim_id: str, result: Dict[str, Any]) -> None:
        self._init_manager()
        with self.queue_lock:
            results = self.simulation_results
            if results is None:
                results = {}
                self.simulation_results = results
            results[sim_id] = result

    def get_result(self, sim_id: str) -> Optional[Dict[str, Any]]:
        self._init_manager()
        with self.queue_lock:
            results = self.simulation_results
            if results is None:
                return None
            return cast("Optional[Dict[str, Any]]", results.get(sim_id))

    def get_status(self, sim_id: str) -> Optional[Dict[str, Any]]:
        self._init_manager()
        with self.queue_lock:
            active = self.active_simulations
            if active is None:
                return None
            status = active.get(sim_id)
            if status is None:
                return None
            return dict(status)

    def cancel(self, sim_id: str) -> Dict[str, Any]:
        self._init_manager()
        decrement_ip: Optional[str] = None
        with self.queue_lock:
            queued = self.queued_simulations
            if queued and sim_id in queued:
                sim_info = queued[sim_id]
                del queued[sim_id]
                decrement_ip = str(sim_info["user_ip"])
                active = self.active_simulations
                if active and sim_id in active:
                    entry = dict(active[sim_id])
                    entry["status"] = "cancelled"
                    entry["message"] = "Removed from queue"
                    active[sim_id] = entry
                response = {"status": "cancelled", "was_queued": True}
            else:
                stop_flags = self.simulation_stop_flags
                if stop_flags and sim_id in stop_flags:
                    stop_flags[sim_id] = True
                    active = self.active_simulations
                    if active and sim_id in active:
                        entry = dict(active[sim_id])
                        entry["status"] = "cancelled"
                        entry["message"] = "Cancelled by user"
                        active[sim_id] = entry
                    response = {"status": "cancelled"}
                else:
                    response = {"status": "not_found"}

        if decrement_ip:
            self.decrement_user_count(decrement_ip)
        return response

    def cleanup_user(self, user_ip: str) -> list[str]:
        self._init_manager()
        cancelled: list[str] = []
        queued_cancelled = 0

        with self.queue_lock:
            queued = self.queued_simulations
            if queued:
                for sim_id in list(queued.keys()):
                    if queued[sim_id]["user_ip"] == user_ip:
                        del queued[sim_id]
                        cancelled.append(sim_id)
                        queued_cancelled += 1
                        active = self.active_simulations
                        if active and sim_id in active:
                            entry = dict(active[sim_id])
                            entry["status"] = "cancelled"
                            entry["message"] = "Browser disconnected"
                            active[sim_id] = entry

            stop_flags = self.simulation_stop_flags
            if stop_flags:
                for sim_id in list(stop_flags.keys()):
                    active = self.active_simulations
                    if active and active.get(sim_id, {}).get("user_ip") == user_ip:
                        stop_flags[sim_id] = True
                        cancelled.append(sim_id)
                        if active and sim_id in active:
                            entry = dict(active[sim_id])
                            entry["status"] = "cancelled"
                            entry["message"] = "Browser disconnected"
                            active[sim_id] = entry

        for _ in range(queued_cancelled):
            self.decrement_user_count(user_ip)

        return cancelled

    def get_log(self, sim_id: str, since: int = 0) -> Optional[Dict[str, Any]]:
        self._init_manager()
        with self.queue_lock:
            logs = self.simulation_logs
            if logs is None:
                return None
            entry = logs.get(sim_id)
            if entry is None:
                return None
            lines = list(entry.get("lines", []))
            first = int(entry.get("first", 0))

        cursor = max(0, int(since))
        dropped = max(0, first - cursor)
        start = max(0, cursor - first)
        return {
            "lines": lines[start:],
            "next": first + len(lines),
            "dropped": dropped,
        }

    def clear_tracking(self, sim_id: str) -> None:
        self._init_manager()
        with self.queue_lock:
            stop_flags = self.simulation_stop_flags
            if stop_flags:
                stop_flags.pop(sim_id, None)
            procs = self.simulation_processes
            if procs:
                procs.pop(sim_id, None)
            logs = self.simulation_logs
            if logs:
                logs.pop(sim_id, None)
