import unittest
from unittest.mock import MagicMock, patch

from PyQt5.QtCore import QCoreApplication

from core.simulation_pipeline import SimulationCancelledError
from ui.workers import SimulationWorker


class SimulationWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def test_cancelled_worker_emits_cancelled_status(self) -> None:
        worker = SimulationWorker(MagicMock(), "WCB", 0.1, 3)
        finished_events = []
        progress_messages = []
        worker.finished.connect(lambda success, status: finished_events.append((success, status)))
        worker.progress.connect(progress_messages.append)

        def fake_run_pipeline(*args, **kwargs):
            self.assertTrue(kwargs["cancel_callback"]())
            raise SimulationCancelledError("Cancelled")

        worker.cancel()

        with patch("ui.workers.run_pipeline", side_effect=fake_run_pipeline):
            worker.run()

        self.assertEqual(finished_events, [(False, "cancelled")])
        self.assertIn("Simulation cancelled", progress_messages)

    def test_use_gpu_option_reaches_pipeline_inputs(self) -> None:
        """performance_options={'use_gpu': True} must reach PipelineInputs.use_gpu=True."""
        worker = SimulationWorker(
            MagicMock(), "WCB", 0.1, 3, performance_options={"use_gpu": True}
        )
        captured = {}

        def fake_run_pipeline(*args, **kwargs):
            captured["inputs"] = args[0]
            raise SimulationCancelledError("Cancelled")

        with patch("ui.workers.run_pipeline", side_effect=fake_run_pipeline):
            worker.run()

        self.assertIsNotNone(captured.get("inputs"))
        self.assertTrue(captured["inputs"].use_gpu)

    def test_use_gpu_defaults_to_false_when_option_missing(self) -> None:
        """Callers that omit use_gpu must keep the existing default of False."""
        worker = SimulationWorker(MagicMock(), "WCB", 0.1, 3)
        captured = {}

        def fake_run_pipeline(*args, **kwargs):
            captured["inputs"] = args[0]
            raise SimulationCancelledError("Cancelled")

        with patch("ui.workers.run_pipeline", side_effect=fake_run_pipeline):
            worker.run()

        self.assertIsNotNone(captured.get("inputs"))
        self.assertFalse(captured["inputs"].use_gpu)


if __name__ == "__main__":
    unittest.main()
