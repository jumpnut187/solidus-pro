"""Regression tests for web simulation queue using multiprocessing.Process."""

from __future__ import annotations

import multiprocessing
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from web.services.fair_share import compute_thread_share
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
            "thread_budget",
            "sim_memory_mb",
            "usable_cores",
            "max_concurrent_simulations",
            "reserve_memory_mb",
            "min_sim_memory_mb",
        ]
        for req in required:
            self.assertIn(req, param_names)

    def test_cancel_sets_stop_flag(self) -> None:
        """cancel() should set the stop flag so the subprocess can check it."""
        manager = SimulationQueueManager(QueueConfig(max_concurrent_simulations=1))
        manager._init_manager()
        manager.set_paths("/tmp/test.db", "/tmp")

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


class SimulationWorkerInputsTests(unittest.TestCase):
    """Verify _simulation_process_worker plumbs performance options into PipelineInputs."""

    def _run_worker(self, sim_data: dict) -> "PipelineInputs":
        """Invoke _simulation_process_worker with mocked dependencies and return the
        PipelineInputs instance passed to run_pipeline."""
        from core.simulation_pipeline import PipelineInputs

        captured = {}

        def fake_run_pipeline(inputs, **kwargs):
            captured["inputs"] = inputs
            fake_result = mock.MagicMock()
            fake_result.voxel_mesh.get_memory_stats.return_value = {"ram_mb": 0.0}
            return fake_result

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "web.payloads.build_part_manager", return_value=mock.MagicMock()
            ):
                with mock.patch(
                    "core.simulation_pipeline.run_pipeline", side_effect=fake_run_pipeline
                ):
                    with mock.patch(
                        "web.payloads.build_fork_artifact",
                        return_value={"summary": {}, "data": {}},
                    ):
                        with mock.patch("web.payloads.write_artifact"):
                            with mock.patch("web.repository.mark_project_completed"):
                                _simulation_process_worker(
                                    sim_id="test-gpu-sim",
                                    sim_data=sim_data,
                                    parts_data=[],
                                    user_ip="127.0.0.1",
                                    db_path=str(Path(tmpdir) / "web.db"),
                                    base_dir=tmpdir,
                                    active_simulations={},
                                    simulation_results={},
                                    stop_flags={},
                                    running_count=SimpleNamespace(value=1),
                                    user_simulation_counts={},
                                    queue_lock=mock.MagicMock(),
                                    queued_simulations={},
                                    max_simulations_per_user=1,
                                    thread_budget=SimpleNamespace(value=0),
                                    sim_memory_mb={},
                                    usable_cores=1,
                                    max_concurrent_simulations=1,
                                    reserve_memory_mb=0,
                                    min_sim_memory_mb=0,
                                )

        self.assertIn("inputs", captured)
        self.assertIsInstance(captured["inputs"], PipelineInputs)
        return captured["inputs"]

    def test_use_gpu_option_reaches_pipeline_inputs(self) -> None:
        """performance_options={'use_gpu': True} must reach PipelineInputs.use_gpu=True."""
        inputs = self._run_worker(
            {"performance_options": {"use_gpu": True}, "alloy": "WCB"}
        )
        self.assertTrue(inputs.use_gpu)

    def test_use_gpu_defaults_to_false_when_option_missing(self) -> None:
        """Callers that omit use_gpu must keep the existing default of False."""
        inputs = self._run_worker({"alloy": "WCB"})
        self.assertFalse(inputs.use_gpu)

    def test_pour_location_reaches_pipeline_inputs_through_worker(self) -> None:
        """A stored project's pour_location must survive the subprocess worker path.

        This covers the same mapping as
        ``BuildPipelineInputsTests.test_pour_location_reaches_pipeline_inputs`` in
        tests/test_web_api.py, but through ``_simulation_process_worker`` -- the
        entry point the live queued-simulation path actually calls.
        """
        pour_location = {
            "schema_version": 1,
            "point_in": [0.5, 1.5, 2.5],
            "normal": [0.0, 1.0, 0.0],
            "part_id": "gate-1",
        }
        inputs = self._run_worker({"alloy": "WCB", "pour_location": pour_location})
        self.assertEqual(inputs.pour_location, pour_location)


class SimulationQueueCleanupTests(unittest.TestCase):
    """Verify cleanup and cancellation work across process boundaries."""

    def test_clear_tracking_removes_process(self) -> None:
        """clear_tracking should remove the process from tracking dict."""
        manager = SimulationQueueManager(QueueConfig(max_concurrent_simulations=1))
        manager._init_manager()
        manager.set_paths("/tmp/test.db", "/tmp")

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


class SimulationQueueFairShareTests(unittest.TestCase):
    """Fair-share thread budgeting and memory gating in the queue manager."""

    def setUp(self) -> None:
        self.manager = SimulationQueueManager(
            QueueConfig(
                max_concurrent_simulations=2,
                max_simulations_per_user=1,
                reserved_cores=1,
                reserve_memory_mb=0,
                min_sim_memory_mb=0,
            )
        )
        self.manager._init_manager()
        self.manager.set_paths("/tmp/test.db", "/tmp")

    def tearDown(self) -> None:
        self.manager._manager.shutdown() if self.manager._manager else None

    def test_submit_updates_thread_budget_fair_share(self) -> None:
        """Starting sims halves the thread share; finishing returns it."""
        usable = self.manager._usable_cores()

        with mock.patch.object(self.manager, "_start_running_simulation"):
            first = self.manager.submit("sim-a", {}, [], "10.0.0.1")
            self.assertEqual(first["status"], "started")
            self.assertEqual(self.manager.thread_budget.value, compute_thread_share(usable, 1))

            second = self.manager.submit("sim-b", {}, [], "10.0.0.2")
            self.assertEqual(second["status"], "started")
            self.assertEqual(self.manager.thread_budget.value, compute_thread_share(usable, 2))

        self.manager.clear_tracking("sim-a")
        self.manager.clear_tracking("sim-b")

    def test_memory_gate_queues_when_headroom_low(self) -> None:
        """A new sim is queued, not started, when RAM headroom is insufficient."""
        self.manager.config.min_sim_memory_mb = 10**9  # effectively infinite floor

        with mock.patch.object(self.manager, "_start_running_simulation"):
            first = self.manager.submit("sim-mem-a", {}, [], "10.0.0.3")
            self.assertEqual(first["status"], "started")
            self.assertEqual(self.manager.running_count.value, 1)

            second = self.manager.submit("sim-mem-b", {}, [], "10.0.0.4")
            self.assertEqual(second["status"], "queued")
            self.assertEqual(self.manager.running_count.value, 1)
            self.assertEqual(second["queue_size"], 1)

        self.manager.cancel("sim-mem-b")
        self.manager.clear_tracking("sim-mem-a")

    def test_queue_status_includes_fair_share_fields(self) -> None:
        """Queue status exposes thread share and memory info."""
        status = self.manager.get_queue_status("127.0.0.1")
        for field in (
            "thread_share",
            "reserved_cores",
            "memory_available_mb",
            "active_memory_mb",
        ):
            self.assertIn(field, status)
        self.assertEqual(status["reserved_cores"], 1)
        self.assertEqual(status["active_memory_mb"], 0)


if __name__ == "__main__":
    if sys.platform == 'darwin':
        multiprocessing.set_start_method('spawn', force=True)
    unittest.main()
