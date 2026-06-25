"""Regression tests for web simulation queue using multiprocessing.Process."""

from __future__ import annotations

import multiprocessing
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from web.services.simulation_queue import (
    QueueConfig,
    SimulationQueueManager,
    _simulation_process_worker,
)


class SimulationQueueProcessTests(unittest.TestCase):
    """Verify simulation queue uses multiprocessing.Process (not threading.Thread)."""

    def test_manager_uses_multiprocessing_manager(self) -> None:
        """Manager should use multiprocessing.Manager for cross-process shared state."""
        manager = SimulationQueueManager(QueueConfig(max_concurrent_simulations=1))
        manager._init_manager()

        self.assertIsNotNone(manager._manager)
        self.assertIsNotNone(manager.queue_lock)
        self.assertIsNotNone(manager.active_simulations)
        self.assertIsNotNone(manager.simulation_results)
        self.assertIsNotNone(manager.simulation_stop_flags)
        self.assertIsNotNone(manager.queued_simulations)
        self.assertIsNotNone(manager.running_count)
        self.assertIsNotNone(manager.user_simulation_counts)

    def test_running_count_is_manager_value(self) -> None:
        """running_count should be a Manager Value proxy, not a plain int."""
        manager = SimulationQueueManager(QueueConfig(max_concurrent_simulations=1))
        manager._init_manager()

        self.assertEqual(manager.running_count.value, 0)
        manager.running_count.value += 1
        self.assertEqual(manager.running_count.value, 1)

    def test_submit_starts_multiprocessing_process(self) -> None:
        """submit() should create a multiprocessing.Process, not a threading.Thread."""
        manager = SimulationQueueManager(QueueConfig(max_concurrent_simulations=1))
        manager._init_manager()
        manager.set_paths("/tmp/test.db", "/tmp")
        manager.set_worker(lambda *a, **k: None)

        sim_id = "test-process-sim"
        sim_data = {"alloy": "WCB", "voxel_size": 0.1, "refinement_level": 1}
        parts_data = []

        response = manager.submit(sim_id, sim_data, parts_data, "127.0.0.1")

        self.assertEqual(response["status"], "started")
        self.assertIn(sim_id, manager.simulation_processes)
        proc = manager.simulation_processes[sim_id]
        self.assertIsInstance(proc, multiprocessing.Process)

        proc.terminate()
        manager.clear_tracking(sim_id)

    def test_simulation_process_is_not_daemon(self) -> None:
        """Worker process must be non-daemon so it can spawn child pools for part assignment.

        ``multiprocessing.ProcessPoolExecutor`` (used by part assignment) calls
        ``multiprocessing.Process`` under the hood. Python refuses to spawn
        children from daemon parents, which would force the parallel
        containment checks into a slow sequential fallback.
        """
        manager = SimulationQueueManager(QueueConfig(max_concurrent_simulations=1))
        manager._init_manager()
        manager.set_paths("/tmp/test.db", "/tmp")
        manager.set_worker(lambda *a, **k: None)

        sim_id = "test-daemon-sim"
        manager.submit(sim_id, {}, [], "127.0.0.1")

        proc = manager.simulation_processes[sim_id]
        self.assertFalse(
            proc.daemon,
            "Simulation worker must be non-daemon so it can spawn child processes "
            "(e.g. ProcessPoolExecutor for parallel part assignment).",
        )

        proc.terminate()
        manager.clear_tracking(sim_id)

    def test_process_worker_receives_proxy_objects(self) -> None:
        """_simulation_process_worker should receive Manager proxy objects as args."""
        import inspect
        sig = inspect.signature(_simulation_process_worker)
        param_names = list(sig.parameters.keys())

        required = [
            "active_simulations",
            "simulation_results",
            "stop_flags",
            "running_count",
            "user_simulation_counts",
            "queue_lock",
            "queued_simulations",
        ]
        for req in required:
            self.assertIn(req, param_names)

    def test_cancel_sets_stop_flag(self) -> None:
        """cancel() should set the stop flag so the subprocess can check it."""
        manager = SimulationQueueManager(QueueConfig(max_concurrent_simulations=1))
        manager._init_manager()
        manager.set_paths("/tmp/test.db", "/tmp")
        manager.set_worker(lambda *a, **k: None)

        sim_id = "test-cancel-sim"

        response = manager.submit(sim_id, {}, [], "127.0.0.1")
        self.assertEqual(response["status"], "started")

        self.assertFalse(manager.should_cancel(sim_id))

        manager.cancel(sim_id)

        self.assertTrue(manager.should_cancel(sim_id))

        manager.clear_tracking(sim_id)

    def test_get_status_returns_dict(self) -> None:
        """get_status() should return a plain dict (not a proxy)."""
        manager = SimulationQueueManager(QueueConfig(max_concurrent_simulations=1))
        manager._init_manager()
        manager.set_paths("/tmp/test.db", "/tmp")
        manager.set_worker(lambda *a, **k: None)

        sim_id = "test-status-sim"

        manager.submit(sim_id, {"alloy": "WCB"}, [], "127.0.0.1")

        status = manager.get_status(sim_id)
        self.assertIsInstance(status, dict)
        self.assertEqual(status["status"], "running")

        result = manager.get_status("nonexistent")
        self.assertIsNone(result)

        manager.clear_tracking(sim_id)

    def test_queue_config_defaults_match_plan(self) -> None:
        """QueueConfig defaults should match the plan: 1 concurrent, 6 queue, 1 per user."""
        config = QueueConfig()
        self.assertEqual(config.max_concurrent_simulations, 1)
        self.assertEqual(config.max_queue_size, 6)
        self.assertEqual(config.max_simulations_per_user, 1)


class SimulationQueueCleanupTests(unittest.TestCase):
    """Verify cleanup and cancellation work across process boundaries."""

    def test_clear_tracking_removes_process(self) -> None:
        """clear_tracking should remove the process from tracking dict."""
        manager = SimulationQueueManager(QueueConfig(max_concurrent_simulations=1))
        manager._init_manager()
        manager.set_paths("/tmp/test.db", "/tmp")
        manager.set_worker(lambda *a, **k: None)

        sim_id = "test-cleanup-sim"

        manager.submit(sim_id, {}, [], "127.0.0.1")

        self.assertIn(sim_id, manager.simulation_processes)

        manager.clear_tracking(sim_id)

        self.assertNotIn(sim_id, manager.simulation_processes)

    def test_cleanup_user_cancels_all_user_sims(self) -> None:
        """cleanup_user() should cancel all running/queued sims for that user."""
        manager = SimulationQueueManager(QueueConfig(max_concurrent_simulations=1))
        manager._init_manager()
        manager.set_paths("/tmp/test.db", "/tmp")
        manager.set_worker(lambda *a, **k: None)

        user_ip = "192.168.1.100"

        manager.submit("sim1", {}, [], user_ip)
        manager.submit("sim2", {}, [], user_ip)

        cancelled = manager.cleanup_user(user_ip)

        self.assertGreaterEqual(len(cancelled), 1)

        manager.clear_tracking("sim1")
        manager.clear_tracking("sim2")

    def test_get_queue_status_reflects_running_count(self) -> None:
        """get_queue_status should show the current running count."""
        manager = SimulationQueueManager(QueueConfig(max_concurrent_simulations=1))
        manager._init_manager()
        manager.set_paths("/tmp/test.db", "/tmp")

        status = manager.get_queue_status("127.0.0.1")
        self.assertEqual(status["running"], 0)
        self.assertEqual(status["max_concurrent"], 1)


if __name__ == "__main__":
    if sys.platform == 'darwin':
        multiprocessing.set_start_method('spawn', force=True)
    unittest.main()
