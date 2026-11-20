from __future__ import annotations

import unittest

import numpy as np

from ui.main_window import MainWindow
from ui.main_window import FRACTION_LIQUID_TIME_SLIDER_MAX
from ui.main_window import FRACTION_LIQUID_PLAYBACK_SPEEDS

class FractionLiquidStateLabelTests(unittest.TestCase):

    def _label(self, array: np.ndarray) -> str:
        return MainWindow._get_fraction_liquid_state_label(array)

    def test_all_solid_returns_fully_solid(self) -> None:
        arr = np.zeros(1000, dtype=np.float32)
        self.assertEqual("Fully Solid", self._label(arr))

    def test_all_zeros_max_below_threshold_returns_fully_solid(self) -> None:
        arr = np.full(500, 0.005, dtype=np.float32)
        self.assertEqual("Fully Solid", self._label(arr))

    def test_none_returns_fully_solid(self) -> None:
        self.assertEqual("Fully Solid", self._label(None))

    def test_empty_array_returns_fully_solid(self) -> None:
        self.assertEqual("Fully Solid", self._label(np.array([], dtype=np.float32)))

    def test_low_residual_liquid_returns_trace_liquid(self) -> None:
        arr = np.zeros(1000, dtype=np.float32)
        arr[:10] = 0.02
        self.assertEqual("Trace Liquid", self._label(arr))

    def test_localized_high_liquid_returns_risers_hot_spots(self) -> None:
        arr = np.zeros(1000, dtype=np.float32)
        arr[:50] = 0.9
        self.assertEqual("Risers/Hot Spots Liquid", self._label(arr))

    def test_broad_high_liquid_returns_mostly_liquid(self) -> None:
        arr = np.full(1000, 0.85, dtype=np.float32)
        self.assertEqual("Mostly Liquid", self._label(arr))

    def test_mid_liquid_returns_solidifying(self) -> None:
        arr = np.full(1000, 0.50, dtype=np.float32)
        self.assertEqual("Solidifying", self._label(arr))

    def test_low_mean_but_not_trace_returns_mostly_solid(self) -> None:
        arr = np.full(1000, 0.15, dtype=np.float32)
        self.assertEqual("Mostly Solid", self._label(arr))

    def test_just_poured_all_liquid_returns_mostly_liquid(self) -> None:
        arr = np.ones(1000, dtype=np.float32)
        self.assertEqual("Mostly Liquid", self._label(arr))

    def test_risers_with_mixed_liquid_returns_risers_hot_spots(self) -> None:
        arr = np.zeros(1000, dtype=np.float32)
        arr[:100] = 0.7
        arr[100:140] = 0.3
        self.assertEqual("Risers/Hot Spots Liquid", self._label(arr))

    def test_slider_time_mapping_preserves_exact_right_endpoint(self) -> None:
        mapped = MainWindow._map_fraction_liquid_slider_time(
            FRACTION_LIQUID_TIME_SLIDER_MAX,
            12.5,
            987.25,
        )

        self.assertEqual(987.25, mapped)

    def test_slider_time_mapping_preserves_exact_left_endpoint(self) -> None:
        mapped = MainWindow._map_fraction_liquid_slider_time(0, 12.5, 987.25)

        self.assertEqual(12.5, mapped)

    def test_slider_time_mapping_interpolates_middle_values(self) -> None:
        mapped = MainWindow._map_fraction_liquid_slider_time(
            FRACTION_LIQUID_TIME_SLIDER_MAX // 2,
            10.0,
            30.0,
        )

        self.assertEqual(20.0, mapped)

class FractionLiquidPlaybackTests(unittest.TestCase):

    def test_advance_normal_advance(self) -> None:
        result = MainWindow._advance_fraction_liquid_slider_value(50, 10, 1000)
        self.assertEqual(60, result)

    def test_advance_at_zero_wraps(self) -> None:
        result = MainWindow._advance_fraction_liquid_slider_value(0, 10, 1000)
        self.assertEqual(10, result)

    def test_advance_reaches_max_then_wraps_next_tick(self) -> None:
        result = MainWindow._advance_fraction_liquid_slider_value(990, 10, 1000)
        self.assertEqual(1000, result)
        next_result = MainWindow._advance_fraction_liquid_slider_value(1000, 10, 1000)
        self.assertEqual(0, next_result)

    def test_advance_exceeds_max_wraps_immediately(self) -> None:
        result = MainWindow._advance_fraction_liquid_slider_value(995, 10, 1000)
        self.assertEqual(0, result)

    def test_advance_at_max_wraps_next_tick(self) -> None:
        result = MainWindow._advance_fraction_liquid_slider_value(1000, 10, 1000)
        self.assertEqual(0, result)

    def test_playback_speeds_constant_contains_expected_values(self) -> None:
        self.assertIn(0.5, FRACTION_LIQUID_PLAYBACK_SPEEDS)
        self.assertIn(1.0, FRACTION_LIQUID_PLAYBACK_SPEEDS)
        self.assertIn(2.0, FRACTION_LIQUID_PLAYBACK_SPEEDS)
        self.assertIn(4.0, FRACTION_LIQUID_PLAYBACK_SPEEDS)

    def test_speed_handler_invalid_data_sets_1(self) -> None:
        mw = MainWindow.__new__(MainWindow)
        mw._fraction_liquid_playback_speed = 2.0
        mw._is_playing_fraction_liquid = False
        mw.fraction_liquid_speed_combo = _FakeCombo(None)
        mw._stop_fraction_liquid_playback = _nop
        mw._start_fraction_liquid_playback = _nop
        mw._on_fraction_liquid_speed_changed(0)
        self.assertEqual(1.0, mw._fraction_liquid_playback_speed)

    def test_speed_handler_valid_data_updates_speed(self) -> None:
        mw = MainWindow.__new__(MainWindow)
        mw._fraction_liquid_playback_speed = 1.0
        mw._is_playing_fraction_liquid = False
        mw.fraction_liquid_speed_combo = _FakeCombo(2.0)
        mw._stop_fraction_liquid_playback = _nop
        mw._start_fraction_liquid_playback = _nop
        mw._on_fraction_liquid_speed_changed(0)
        self.assertEqual(2.0, mw._fraction_liquid_playback_speed)

    def test_speed_handler_while_playing_triggers_stop_start(self) -> None:
        mw = MainWindow.__new__(MainWindow)
        mw._fraction_liquid_playback_speed = 1.0
        mw._is_playing_fraction_liquid = True
        mw.fraction_liquid_speed_combo = _FakeCombo(4.0)
        stop_called = False
        start_called = False

        def stub_stop():
            nonlocal stop_called
            stop_called = True

        def stub_start():
            nonlocal start_called
            start_called = True

        mw._stop_fraction_liquid_playback = stub_stop
        mw._start_fraction_liquid_playback = stub_start
        mw._on_fraction_liquid_speed_changed(0)

        self.assertTrue(stop_called)
        self.assertTrue(start_called)

class _FakeCombo:
    def __init__(self, data):
        self._data = data

    def currentData(self):
        return self._data

def _nop():
    pass

if __name__ == "__main__":
    unittest.main()
