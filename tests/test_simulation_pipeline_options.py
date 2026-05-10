from __future__ import annotations

import unittest

from core.solidification import SOLVER_MODE_SCHEDULE_LEGACY, SOLVER_MODE_THERMAL_COUPLED
from core.simulation_pipeline import create_analysis_options


class SimulationPipelineOptionsTests(unittest.TestCase):
    def test_sand_restraint_defaults_to_neutral(self) -> None:
        options = create_analysis_options({}, use_gpu=False, use_incremental=False)
        self.assertAlmostEqual(options.sand_restraint_index, 0.5, places=6)

    def test_sand_restraint_clamps_to_unit_interval(self) -> None:
        high = create_analysis_options(
            {"sand_restraint_index": 1.7},
            use_gpu=False,
            use_incremental=False,
        )
        low = create_analysis_options(
            {"sand_restraint_index": -0.3},
            use_gpu=False,
            use_incremental=False,
        )
        self.assertEqual(high.sand_restraint_index, 1.0)
        self.assertEqual(low.sand_restraint_index, 0.0)

    def test_sand_restraint_invalid_value_falls_back_to_neutral(self) -> None:
        options = create_analysis_options(
            {"sand_restraint_index": "invalid"},
            use_gpu=False,
            use_incremental=False,
        )
        self.assertAlmostEqual(options.sand_restraint_index, 0.5, places=6)

    def test_solver_mode_defaults_to_thermal_coupled(self) -> None:
        options = create_analysis_options({}, use_gpu=False, use_incremental=False)
        self.assertEqual(options.solver_mode, SOLVER_MODE_THERMAL_COUPLED)

    def test_solver_mode_accepts_legacy_fallback(self) -> None:
        options = create_analysis_options(
            {"solver_mode": SOLVER_MODE_SCHEDULE_LEGACY},
            use_gpu=False,
            use_incremental=False,
        )
        self.assertEqual(options.solver_mode, SOLVER_MODE_SCHEDULE_LEGACY)

    def test_solver_mode_invalid_value_falls_back_to_thermal(self) -> None:
        options = create_analysis_options(
            {"solver_mode": "not-a-mode"},
            use_gpu=False,
            use_incremental=False,
        )
        self.assertEqual(options.solver_mode, SOLVER_MODE_THERMAL_COUPLED)


if __name__ == "__main__":
    unittest.main()
