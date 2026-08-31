"""Tests for how a filling run's outcome is reported to the user.

These exist because of a specific failure: the pipeline correctly refuses to
seed solidification from a diverged fill, which leaves ``_fill_arrival_times``
unset, which removes the Fill Time and Fill Animation entries from the results
list. The result was that a run whose solver blew up looked *identical* to one
where filling was never enabled -- no view, no error, nothing. The explanation
went to a status-bar label that the next progress message overwrote, and never
reached the log file at all.

So the thing under test is not the physics. It is that each of the four
outcomes says something different, that a discarded fill says so, and that the
warning about solidification falling back to a placeholder actually appears --
because that is what tells an operator how far to trust the rest of the run.
"""

from __future__ import annotations

import unittest

from core.filling_stage import (
    STATUS_COMPLETE,
    STATUS_DIVERGED,
    STATUS_STALLED,
    STATUS_TIME_CAPPED,
)


def _summary(status, **overrides):
    base = {
        "status": status,
        "trustworthy": status != STATUS_DIVERGED,
        "diverged_at_step": 12800 if status == STATUS_DIVERGED else 0,
        "filled_voxel_count": 23_915,
        "misrun_voxel_count": 6_392,
        "filled_fraction": 0.789,
        "fill_seconds": 1.87,
        "choke_thickness_cells": 3.0,
        "warnings": ("6,392 metal voxels never filled (misrun): 21.1% of the metal",),
    }
    base.update(overrides)
    return base


def _notice(summary):
    from ui.main_window import format_filling_notice

    return format_filling_notice(summary)


class NoticeLevelTests(unittest.TestCase):
    def test_no_filling_run_says_nothing(self) -> None:
        """Filling that was never enabled must not raise a notice at all."""
        notice = _notice(None)
        self.assertEqual(notice["level"], "none")
        self.assertEqual(notice["status_line"], "")
        self.assertEqual(notice["tag"], "")

    def test_a_clean_fill_does_not_interrupt(self) -> None:
        notice = _notice(_summary(STATUS_COMPLETE, filled_fraction=1.0))
        self.assertEqual(notice["level"], "info")
        self.assertEqual(notice["tag"], "OK")

    def test_a_diverged_fill_is_critical(self) -> None:
        self.assertEqual(_notice(_summary(STATUS_DIVERGED))["level"], "critical")

    def test_stalled_and_capped_warn_without_being_critical(self) -> None:
        for status in (STATUS_STALLED, STATUS_TIME_CAPPED):
            with self.subTest(status=status):
                self.assertEqual(_notice(_summary(status))["level"], "warning")

    def test_every_outcome_is_distinguishable(self) -> None:
        """
        The whole point. Four outcomes, four different things said -- a fill
        that blew up must not read like one that merely ran out of time.
        """
        lines = {
            status: _notice(_summary(status))["status_line"]
            for status in (
                STATUS_COMPLETE,
                STATUS_DIVERGED,
                STATUS_STALLED,
                STATUS_TIME_CAPPED,
            )
        }
        self.assertEqual(len(set(lines.values())), 4, lines)
        self.assertNotIn("", lines.values())


class DivergedNoticeTests(unittest.TestCase):
    """A discarded fill has to say what it cost the rest of the run."""

    def setUp(self) -> None:
        self.notice = _notice(_summary(STATUS_DIVERGED))
        self.body = self.notice["body"].lower()

    def test_it_says_the_results_were_discarded(self) -> None:
        self.assertIn("discard", self.body)

    def test_it_names_the_views_that_are_missing(self) -> None:
        """Otherwise their absence is the unexplained symptom all over again."""
        self.assertIn("fill time", self.body)
        self.assertIn("fill animation", self.body)

    def test_it_warns_that_solidification_used_a_placeholder(self) -> None:
        """
        The load-bearing sentence: it tells the operator the solidification
        numbers on screen do not reflect the pour.
        """
        self.assertIn("placeholder", self.body)

    def test_it_reports_the_step_with_a_thousands_separator(self) -> None:
        """Matches the solver's own warning text sitting right beside it."""
        self.assertIn("12,800", self.notice["body"])

    def test_it_carries_the_solver_warnings_through(self) -> None:
        warning = "6,392 metal voxels never filled (misrun): 21.1% of the metal"
        self.assertIn(warning, self.notice["body"])

    def test_it_has_a_dialog_title(self) -> None:
        self.assertTrue(self.notice["title"].strip())


class StalledNoticeTests(unittest.TestCase):
    def test_it_distinguishes_a_stall_from_a_time_limit(self) -> None:
        """
        These are different findings. A stall is a misrun or an under-resolved
        section; a time cap just means the clock ran out. Reading one as the
        other sends the user to change the wrong setting.
        """
        body = _notice(_summary(STATUS_STALLED))["body"].lower()
        self.assertIn("time", body)
        self.assertTrue(
            "misrun" in body or "under-resolved" in body or "resolved" in body,
            body,
        )

    def test_a_time_cap_says_the_misrun_count_is_not_a_defect(self) -> None:
        body = _notice(_summary(STATUS_TIME_CAPPED))["body"].lower()
        self.assertIn("max_fill_time_s", body)


class RobustnessTests(unittest.TestCase):
    """The summary crosses a process boundary; it must not be trusted blindly."""

    def test_an_empty_summary_is_treated_as_no_run(self) -> None:
        self.assertEqual(_notice({})["level"], "none")

    def test_missing_keys_do_not_raise(self) -> None:
        for status in (
            STATUS_COMPLETE,
            STATUS_DIVERGED,
            STATUS_STALLED,
            STATUS_TIME_CAPPED,
        ):
            with self.subTest(status=status):
                notice = _notice({"status": status})
                self.assertIn(notice["level"], {"info", "warning", "critical"})

    def test_an_unknown_status_does_not_raise(self) -> None:
        notice = _notice(_summary("something_new"))
        self.assertIn("level", notice)

    def test_no_warnings_is_fine(self) -> None:
        notice = _notice(_summary(STATUS_DIVERGED, warnings=()))
        self.assertTrue(notice["body"].strip())


class SummaryPlumbingTests(unittest.TestCase):
    """The engine must expose the summary even when the fill was unusable."""

    def test_engine_defaults_to_no_summary(self) -> None:
        from core.solidification import SolidificationEngine

        engine = SolidificationEngine.__new__(SolidificationEngine)
        self.assertIsNone(engine.get_filling_summary())

    def test_a_diverged_summary_survives_the_handover(self) -> None:
        """
        The diverged branch skips set_initial_temperature_field entirely, so
        the summary has to be attached outside it or it would be lost in
        exactly the case it is needed.
        """
        from core.solidification import SolidificationEngine

        engine = SolidificationEngine.__new__(SolidificationEngine)
        engine._filling_summary = _summary(STATUS_DIVERGED)
        summary = engine.get_filling_summary()
        self.assertIsNotNone(summary)
        self.assertEqual(summary["status"], STATUS_DIVERGED)
        self.assertFalse(summary["trustworthy"])


if __name__ == "__main__":
    unittest.main()
