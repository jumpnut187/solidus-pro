from __future__ import annotations

import unittest

import numpy as np

from ui.main_window import MainWindow
from ui.main_window import FRACTION_LIQUID_TIME_SLIDER_MAX


class FractionLiquidStateLabelTests(unittest.TestCase):
    """Tests for MainWindow._get_fraction_liquid_state_label using synthetic arrays."""

    def _label(self, array: np.ndarray) -> str:
        return MainWindow._get_fraction_liquid_state_label(array)

    def test_all_solid_returns_fully_solid(self) -> None:
        arr = np.zeros(1000, dtype=np.float32)
        self.assertEqual("Fully Solid", self._label(arr))

    def test_all_zeros_max_below_threshold_returns_fully_solid(self) -> None:
        arr = np.full(500, 0.005, dtype=np.float32)
        self.assertEqual("Fully Solid", self._label(arr))

    def test_none_returns_fully_solid(self) -> None:
        self.assertEqual("Fully Solid", self._label(None))  # type: ignore[arg-type]

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


if __name__ == "__main__":
    unittest.main()
