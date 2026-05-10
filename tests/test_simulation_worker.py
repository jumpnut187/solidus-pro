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


if __name__ == "__main__":
    unittest.main()
