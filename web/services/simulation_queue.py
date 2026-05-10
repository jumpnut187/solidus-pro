"""Queue and state management for web simulation execution."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, Optional
import threading
import time


TaskWorker = Callable[[str, Dict[str, Any], list[Dict[str, Any]]], None]


class QueueConfig:
    def __init__(
        self,
        max_concurrent_simulations: int = 2,
        max_queue_size: int = 10,
        max_simulations_per_user: int = 2,
    ) -> None:
        self.max_concurrent_simulations = max_concurrent_simulations
        self.max_queue_size = max_queue_size
        self.max_simulations_per_user = max_simulations_per_user


class SimulationQueueManager:
    """Thread-safe queue manager for background simulations."""

    def __init__(self, config: Optional[QueueConfig] = None) -> None:
        self.config = config or QueueConfig()
        self.queue_lock = threading.Lock()

        self.active_simulations: Dict[str, Dict[str, Any]] = {}
        self.simulation_results: Dict[str, Dict[str, Any]] = {}
        self.simulation_threads: Dict[str, threading.Thread] = {}
        self.simulation_stop_flags: Dict[str, bool] = {}
        self.queued_simulations: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.user_simulation_counts: Dict[str, int] = {}
        self.running_count = 0
        self._worker: Optional[TaskWorker] = None

    def set_worker(self, worker: TaskWorker) -> None:
        self._worker = worker

    def can_user_submit(self, user_ip: str) -> bool:
        with self.queue_lock:
            return self.user_simulation_counts.get(user_ip, 0) < self.config.max_simulations_per_user

    def increment_user_count(self, user_ip: str) -> None:
        with self.queue_lock:
            self.user_simulation_counts[user_ip] = self.user_simulation_counts.get(user_ip, 0) + 1

    def decrement_user_count(self, user_ip: str) -> None:
        with self.queue_lock:
            if user_ip in self.user_simulation_counts:
                self.user_simulation_counts[user_ip] = max(0, self.user_simulation_counts[user_ip] - 1)
                if self.user_simulation_counts[user_ip] == 0:
                    del self.user_simulation_counts[user_ip]

    def get_queue_position(self, sim_id: str) -> int:
        with self.queue_lock:
            if sim_id not in self.queued_simulations:
                return 0
            for idx, queued_id in enumerate(self.queued_simulations.keys(), start=1):
                if queued_id == sim_id:
                    return idx
        return 0

    def get_queue_status(self, user_ip: str) -> Dict[str, int]:
        with self.queue_lock:
            return {
                "running": self.running_count,
                "max_concurrent": self.config.max_concurrent_simulations,
                "queued": len(self.queued_simulations),
                "max_queue": self.config.max_queue_size,
                "your_simulations": self.user_simulation_counts.get(user_ip, 0),
                "max_per_user": self.config.max_simulations_per_user,
            }

    def submit(
        self,
        sim_id: str,
        sim_data: Dict[str, Any],
        parts_data: list[Dict[str, Any]],
        user_ip: str,
    ) -> Dict[str, Any]:
        if self._worker is None:
            raise RuntimeError("Simulation queue worker is not configured")

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
            if self.running_count < self.config.max_concurrent_simulations:
                self.running_count += 1
                can_run_now = True
            else:
                if len(self.queued_simulations) >= self.config.max_queue_size:
                    should_rollback_user_count = True
                    can_run_now = False
                else:
                    self.queued_simulations[sim_id] = {
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
            self.active_simulations[sim_id] = {
                "status": "queued",
                "progress": 0,
                "message": f"Queued (position {position} of {len(self.queued_simulations)})",
                "queue_position": position,
                "queued_at": time.time(),
                "user_ip": user_ip,
            }
            queue_size = len(self.queued_simulations)

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
        with self.queue_lock:
            self.active_simulations[sim_id] = {
                "status": "running",
                "progress": 0,
                "message": message,
                "started_at": time.time(),
                "user_ip": user_ip,
            }
            self.simulation_stop_flags[sim_id] = False

        thread = threading.Thread(
            target=self._run_task_wrapper,
            args=(sim_id, sim_data, parts_data, user_ip),
            daemon=True,
        )
        thread.start()
        with self.queue_lock:
            self.simulation_threads[sim_id] = thread

    def _run_task_wrapper(
        self,
        sim_id: str,
        sim_data: Dict[str, Any],
        parts_data: list[Dict[str, Any]],
        user_ip: str,
    ) -> None:
        try:
            if self._worker is None:
                raise RuntimeError("Simulation queue worker is not configured")
            self._worker(sim_id, sim_data, parts_data)
        finally:
            with self.queue_lock:
                self.running_count = max(0, self.running_count - 1)
            self.decrement_user_count(user_ip)
            self.clear_tracking(sim_id)
            self.process_queue()

    def process_queue(self) -> None:
        with self.queue_lock:
            if self.running_count >= self.config.max_concurrent_simulations:
                return
            if not self.queued_simulations:
                return
            sim_id, sim_info = next(iter(self.queued_simulations.items()))
            del self.queued_simulations[sim_id]
            self.running_count += 1

        self._start_running_simulation(
            sim_id=sim_id,
            sim_data=sim_info["sim_data"],
            parts_data=sim_info["parts_data"],
            user_ip=sim_info["user_ip"],
            message="Starting...",
        )

    def should_cancel(self, sim_id: str) -> bool:
        with self.queue_lock:
            return self.simulation_stop_flags.get(sim_id, False)

    def set_progress(self, sim_id: str, progress: int, message: str) -> None:
        with self.queue_lock:
            if sim_id in self.active_simulations:
                self.active_simulations[sim_id]["progress"] = progress
                self.active_simulations[sim_id]["message"] = message

    def set_status(self, sim_id: str, status: str, message: str, progress: int) -> None:
        with self.queue_lock:
            if sim_id in self.active_simulations:
                self.active_simulations[sim_id]["status"] = status
                self.active_simulations[sim_id]["message"] = message
                self.active_simulations[sim_id]["progress"] = progress

    def save_result(self, sim_id: str, result: Dict[str, Any]) -> None:
        with self.queue_lock:
            self.simulation_results[sim_id] = result

    def get_result(self, sim_id: str) -> Optional[Dict[str, Any]]:
        with self.queue_lock:
            return self.simulation_results.get(sim_id)

    def get_status(self, sim_id: str) -> Optional[Dict[str, Any]]:
        with self.queue_lock:
            status = self.active_simulations.get(sim_id)
            if status is None:
                return None
            return dict(status)

    def cancel(self, sim_id: str) -> Dict[str, Any]:
        decrement_ip: Optional[str] = None
        with self.queue_lock:
            if sim_id in self.queued_simulations:
                sim_info = self.queued_simulations[sim_id]
                del self.queued_simulations[sim_id]
                decrement_ip = str(sim_info["user_ip"])
                if sim_id in self.active_simulations:
                    self.active_simulations[sim_id]["status"] = "cancelled"
                    self.active_simulations[sim_id]["message"] = "Removed from queue"
                response = {"status": "cancelled", "was_queued": True}
            elif sim_id in self.simulation_stop_flags:
                self.simulation_stop_flags[sim_id] = True
                if sim_id in self.active_simulations:
                    self.active_simulations[sim_id]["status"] = "cancelled"
                    self.active_simulations[sim_id]["message"] = "Cancelled by user"
                response = {"status": "cancelled"}
            else:
                response = {"status": "not_found"}

        if decrement_ip:
            self.decrement_user_count(decrement_ip)
        return response

    def cleanup_user(self, user_ip: str) -> list[str]:
        cancelled: list[str] = []
        queued_cancelled = 0

        with self.queue_lock:
            for sim_id in list(self.queued_simulations.keys()):
                if self.queued_simulations[sim_id]["user_ip"] == user_ip:
                    del self.queued_simulations[sim_id]
                    cancelled.append(sim_id)
                    queued_cancelled += 1
                    if sim_id in self.active_simulations:
                        self.active_simulations[sim_id]["status"] = "cancelled"
                        self.active_simulations[sim_id]["message"] = "Browser disconnected"

            for sim_id in list(self.simulation_stop_flags.keys()):
                if sim_id in self.active_simulations and self.active_simulations[sim_id].get("user_ip") == user_ip:
                    self.simulation_stop_flags[sim_id] = True
                    cancelled.append(sim_id)
                    self.active_simulations[sim_id]["status"] = "cancelled"
                    self.active_simulations[sim_id]["message"] = "Browser disconnected"

        for _ in range(queued_cancelled):
            self.decrement_user_count(user_ip)

        return cancelled

    def clear_tracking(self, sim_id: str) -> None:
        with self.queue_lock:
            self.simulation_stop_flags.pop(sim_id, None)
            self.simulation_threads.pop(sim_id, None)
