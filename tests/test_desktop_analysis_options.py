from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from ui.main_window import MainWindow

class DesktopAnalysisOptionsTests(unittest.TestCase):
    def test_fraction_liquid_does_not_force_thermal_solver(self) -> None:
        window = MainWindow.__new__(MainWindow)
        window._analysis_options = {
            "shrinkage": True,
            "hotspot": True,
            "fraction_liquid": True,
        }

        options = window._get_effective_analysis_options()

        self.assertTrue(options["fraction_liquid"])
        self.assertNotIn("solver_mode", options)

    def test_saved_solver_mode_is_ignored_without_desktop_solver_selector(self) -> None:
        window = MainWindow.__new__(MainWindow)
        window._analysis_options = {
            "shrinkage": True,
            "fraction_liquid": True,
            "solver_mode": "thermal_coupled",
        }

        options = window._get_effective_analysis_options()

        self.assertNotIn("solver_mode", options)

    def test_purge_previous_simulation_memory_preserves_part_manager(self) -> None:
        window = MainWindow.__new__(MainWindow)
        part_manager = object()
        worker = MagicMock()
        worker.isRunning.return_value = False
        worker.simulation = object()
        worker.voxel_mesh = object()

        window.part_manager = part_manager
        window.current_results = {"old": "results"}
        window.result_type_combo = MagicMock()
        window.visualizer = MagicMock()
        window.simulation_worker = worker

        with (
            patch("ui.main_window.reset_memory_manager") as reset_memory_manager,
            patch("ui.main_window.gc.collect") as collect,
        ):
            window._purge_previous_simulation_memory()

        self.assertIsNone(window.current_results)
        self.assertIsNone(window.simulation_worker)
        self.assertIsNone(worker.simulation)
        self.assertIsNone(worker.voxel_mesh)
        self.assertIs(window.part_manager, part_manager)
        window.result_type_combo.setEnabled.assert_called_once_with(False)
        window.visualizer.clear_visualization.assert_called_once_with()
        reset_memory_manager.assert_called_once_with()
        collect.assert_called_once_with()

if __name__ == "__main__":
    unittest.main()
