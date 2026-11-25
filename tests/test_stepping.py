from __future__ import annotations

import unittest

from core.stepping import SnapshotSchedule, should_capture_fraction_history

class _Options:
    def __init__(self, **flags) -> None:
        self.fraction_liquid = flags.get("fraction_liquid", False)
        self.hot_tear = flags.get("hot_tear", False)
        self.shrinkage = flags.get("shrinkage", False)
        self.feeding_path = flags.get("feeding_path", False)

class _Engine:
    def __init__(self, options, capture_fn=None) -> None:
        self.analysis_options = options
        if capture_fn is not None:
            self._should_capture_fraction_history = capture_fn

class CaptureDecisionTests(unittest.TestCase):
    def test_engine_method_takes_precedence(self) -> None:
        engine = _Engine(_Options(), capture_fn=lambda: True)
        self.assertTrue(should_capture_fraction_history(engine))

        engine = _Engine(_Options(shrinkage=True), capture_fn=lambda: False)
        self.assertFalse(should_capture_fraction_history(engine))

    def test_fallback_is_true_for_any_history_dependent_analysis(self) -> None:
        for flag in ("fraction_liquid", "hot_tear", "shrinkage", "feeding_path"):
            with self.subTest(flag=flag):
                engine = _Engine(_Options(**{flag: True}))
                self.assertTrue(should_capture_fraction_history(engine))

    def test_fallback_is_false_when_nothing_needs_history(self) -> None:
        self.assertFalse(should_capture_fraction_history(_Engine(_Options())))

    def test_non_callable_attribute_falls_back(self) -> None:
        engine = _Engine(_Options(shrinkage=True))
        engine._should_capture_fraction_history = "not callable"
        self.assertTrue(should_capture_fraction_history(engine))

class SnapshotScheduleTests(unittest.TestCase):
    def test_rejects_non_positive_frame_counts(self) -> None:
        for bad in (0, -1):
            with self.subTest(frames=bad):
                with self.assertRaises(ValueError):
                    SnapshotSchedule(frames=bad)

    def test_first_call_always_captures(self) -> None:
        self.assertTrue(SnapshotSchedule(frames=100).should_capture(0.0))

    def test_captures_once_per_interval(self) -> None:
        schedule = SnapshotSchedule(frames=100)
        self.assertTrue(schedule.should_capture(0.0))
        self.assertFalse(schedule.should_capture(0.5))
        self.assertTrue(schedule.should_capture(1.0))
        self.assertFalse(schedule.should_capture(1.9))
        self.assertTrue(schedule.should_capture(2.0))

    def test_frame_count_over_a_full_run(self) -> None:
        schedule = SnapshotSchedule(frames=100)
        captured = sum(
            1 for i in range(1001) if schedule.should_capture(i * 0.1)
        )
        self.assertEqual(captured, 101)

    def test_coarser_schedule_captures_fewer(self) -> None:
        schedule = SnapshotSchedule(frames=10)
        captured = sum(1 for i in range(1001) if schedule.should_capture(i * 0.1))
        self.assertEqual(captured, 11)

    def test_a_jump_in_progress_still_captures_once(self) -> None:
        schedule = SnapshotSchedule(frames=100)
        self.assertTrue(schedule.should_capture(0.0))
        self.assertTrue(schedule.should_capture(40.0))
        self.assertFalse(schedule.should_capture(40.5))

    def test_reset_restores_initial_state(self) -> None:
        schedule = SnapshotSchedule(frames=100)
        schedule.should_capture(50.0)
        self.assertFalse(schedule.should_capture(50.5))
        schedule.reset()
        self.assertTrue(schedule.should_capture(50.5))

if __name__ == "__main__":
    unittest.main()
