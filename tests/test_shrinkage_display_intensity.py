from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from core.solidification import SolidificationEngine
from ui.main_window import MainWindow


class ShrinkageDisplayIntensityTests(unittest.TestCase):
    def test_part_type_aware_display_intensity(self) -> None:
        engine = SolidificationEngine.__new__(SolidificationEngine)
        engine.mesh = SimpleNamespace(
            voxel_part_types=np.array([0, 1, 3, 2], dtype=np.int32)
        )
        engine.shrinkage_probability = np.array([0.9, 0.2, 0.3, 0.7], dtype=np.float32)
        engine.porosity_probability = np.array([0.4, 0.8, 0.1, 0.6], dtype=np.float32)

        display = engine.get_shrinkage_display_intensity()

        # All non-CHILL voxels use combined max risk.
        self.assertAlmostEqual(float(display[0]), 0.9, places=6)
        self.assertAlmostEqual(float(display[1]), 0.8, places=6)
        self.assertAlmostEqual(float(display[2]), 0.3, places=6)
        # CHILL voxels are hidden.
        self.assertAlmostEqual(float(display[3]), 0.0, places=6)

    def test_cast_shrinkage_above_threshold_with_low_porosity(self) -> None:
        engine = SolidificationEngine.__new__(SolidificationEngine)
        engine.mesh = SimpleNamespace(
            voxel_part_types=np.array([0], dtype=np.int32)
        )
        engine.shrinkage_probability = np.array([0.85], dtype=np.float32)
        engine.porosity_probability = np.array([0.1], dtype=np.float32)

        display = engine.get_shrinkage_display_intensity()

        self.assertGreater(float(display[0]), 0.5)

    def test_porosity_connected_cast_is_damped(self) -> None:
        engine = SolidificationEngine.__new__(SolidificationEngine)
        engine.mesh = SimpleNamespace(
            voxel_part_types=np.array([0, 0, 1, 3], dtype=np.int32)
        )
        engine.shrinkage_risk = np.array([0.8, 0.8, 0.8, 0.8], dtype=np.float32)
        engine.shrinkage_probability = engine.shrinkage_risk.copy()
        engine.niyama_severity = np.zeros(4, dtype=np.float32)
        engine.feeding_quality = np.ones(4, dtype=np.float32)
        engine.feeding_open_fraction = np.ones(4, dtype=np.float32)
        engine.feeding_blocked = np.zeros(4, dtype=bool)
        engine.feeding_darcy_margin = np.ones(4, dtype=np.float32)
        engine._get_connected_feeder_mask = lambda: np.array([True, False, False, True], dtype=bool)

        engine._calculate_porosity()

        porosity = np.asarray(engine.porosity_probability, dtype=np.float32)
        self.assertLess(float(porosity[0]), float(porosity[1]))

    def test_shrinkage_threshold_auto_lowers_to_show_real_low_risk(self) -> None:
        intensities = np.array([0.0, 0.02, 0.12, 0.05], dtype=np.float32)

        threshold, lowered = MainWindow._get_effective_shrinkage_threshold(
            intensities,
            requested_threshold=0.30,
        )

        self.assertTrue(lowered)
        self.assertAlmostEqual(threshold, 0.12, places=6)

    def test_shrinkage_threshold_does_not_lower_for_numerical_noise(self) -> None:
        intensities = np.array([0.0, 0.002, 0.009], dtype=np.float32)

        threshold, lowered = MainWindow._get_effective_shrinkage_threshold(
            intensities,
            requested_threshold=0.30,
        )

        self.assertFalse(lowered)
        self.assertAlmostEqual(threshold, 0.30, places=6)

    def test_shrinkage_threshold_keeps_requested_when_values_are_visible(self) -> None:
        intensities = np.array([0.0, 0.31, 0.12], dtype=np.float32)

        threshold, lowered = MainWindow._get_effective_shrinkage_threshold(
            intensities,
            requested_threshold=0.30,
        )

        self.assertFalse(lowered)
        self.assertAlmostEqual(threshold, 0.30, places=6)


if __name__ == "__main__":
    unittest.main()
