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

    def test_filling_defaults_to_false(self) -> None:
        options = create_analysis_options({}, use_gpu=False, use_incremental=False)
        self.assertFalse(options.filling)

    def test_filling_is_set_from_option_map(self) -> None:
        options = create_analysis_options(
            {"filling": True},
            use_gpu=False,
            use_incremental=False,
        )
        self.assertTrue(options.filling)

        options_off = create_analysis_options(
            {"filling": False},
            use_gpu=False,
            use_incremental=False,
        )
        self.assertFalse(options_off.filling)


if __name__ == "__main__":
    unittest.main()


class AnalysisHeaderTests(unittest.TestCase):
    """
    The run header lists which analyses are active. It is a hand-maintained
    table, and `filling` was missing from it at first -- so a run with the fill
    switched off printed exactly what a successful fill would have, and there
    was no way to tell from the log which had happened.
    """

    #: Boolean fields on AnalysisOptions that are NOT analyses.
    _NOT_ANALYSES = frozenset({"use_gpu", "use_incremental"})

    def test_every_analysis_flag_is_reported(self) -> None:
        """
        The guard that matters: a new analysis flag must be added to
        ANALYSIS_LABELS or this fails. Catches the omission class, not just
        the one instance of it.
        """
        import dataclasses

        from core.solidification import ANALYSIS_LABELS, AnalysisOptions

        labelled = {attribute for attribute, _ in ANALYSIS_LABELS}
        for f in dataclasses.fields(AnalysisOptions):
            if f.type not in ("bool", bool) or f.name in self._NOT_ANALYSES:
                continue
            with self.subTest(flag=f.name):
                self.assertIn(
                    f.name, labelled,
                    f"AnalysisOptions.{f.name} is an analysis toggle but is not "
                    f"in ANALYSIS_LABELS, so runs will not report it",
                )

    def test_only_real_toggles_are_listed(self) -> None:
        """
        The inverse guard. sand_restraint_index is a float defaulting to 0.5,
        so listing it would report "Sand Restraint" on every single run.
        """
        import dataclasses

        from core.solidification import ANALYSIS_LABELS, AnalysisOptions

        by_name = {f.name: f for f in dataclasses.fields(AnalysisOptions)}
        for attribute, _ in ANALYSIS_LABELS:
            with self.subTest(flag=attribute):
                self.assertIn(attribute, by_name)
                self.assertIn(by_name[attribute].type, ("bool", bool))

    def test_filling_is_reported_only_when_enabled(self) -> None:
        from core.solidification import AnalysisOptions, describe_analyses

        options = AnalysisOptions()
        self.assertNotIn("Mold Filling (CFD)", describe_analyses(options))
        options.filling = True
        self.assertIn("Mold Filling (CFD)", describe_analyses(options))

    def test_disabled_analyses_are_omitted(self) -> None:
        from core.solidification import AnalysisOptions, describe_analyses

        options = AnalysisOptions(shrinkage=False, hotspot=False)
        self.assertEqual(describe_analyses(options), [])

    def test_niyama_carries_its_threshold(self) -> None:
        from core.materials import get_alloy
        from core.solidification import AnalysisOptions, describe_analyses

        options = AnalysisOptions(shrinkage=False, hotspot=False, niyama=True)
        labels = describe_analyses(options, get_alloy("WCB"))
        self.assertEqual(len(labels), 1)
        self.assertTrue(labels[0].startswith("Niyama (threshold="))

    def test_niyama_without_an_alloy_still_reports(self) -> None:
        from core.solidification import AnalysisOptions, describe_analyses

        options = AnalysisOptions(shrinkage=False, hotspot=False, niyama=True)
        self.assertEqual(describe_analyses(options), ["Niyama"])
