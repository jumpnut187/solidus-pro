from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
