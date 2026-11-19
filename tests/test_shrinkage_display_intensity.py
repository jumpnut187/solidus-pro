from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from core.solidification_params import SolidificationParameters
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

    def test_shrink_scalar_above_one_lowers_public_probabilities(self) -> None:
        engine = SolidificationEngine.__new__(SolidificationEngine)
        raw_shrinkage = np.array([0.2, 0.8], dtype=np.float32)
        raw_porosity = np.array([0.4, 0.9], dtype=np.float32)
        engine.shrinkage_risk = raw_shrinkage.copy()
        engine.shrinkage_probability = engine.shrinkage_risk
        engine.porosity_risk = raw_porosity.copy()
        engine.porosity_probability = engine.porosity_risk
        engine.analysis_options = SimpleNamespace(
            solidification_params=SolidificationParameters.from_dict(
                {"feeding": {"shrink_scalar": 2.0}}
            )
        )

        engine._apply_shrink_scalar()

        np.testing.assert_allclose(engine.shrinkage_risk, raw_shrinkage)
        np.testing.assert_allclose(engine.porosity_risk, raw_porosity)
        np.testing.assert_allclose(engine.shrinkage_probability, [0.1, 0.4])
        np.testing.assert_allclose(engine.porosity_probability, [0.2, 0.45])
        self.assertEqual(
            engine.shrink_scalar_formula,
            "adjusted_risk = clamp(raw_risk / shrink_scalar, 0, 1)",
        )

    def test_shrink_scalar_below_one_raises_public_probabilities_with_clipping(self) -> None:
        engine = SolidificationEngine.__new__(SolidificationEngine)
        engine.shrinkage_risk = np.array([0.2, 0.8], dtype=np.float32)
        engine.shrinkage_probability = engine.shrinkage_risk
        engine.porosity_risk = np.array([0.4, 0.9], dtype=np.float32)
        engine.porosity_probability = engine.porosity_risk
        engine.analysis_options = SimpleNamespace(
            solidification_params=SolidificationParameters.from_dict(
                {"feeding": {"shrink_scalar": 0.5}}
            )
        )

        engine._apply_shrink_scalar()

        np.testing.assert_allclose(engine.shrinkage_probability, [0.4, 1.0])
        np.testing.assert_allclose(engine.porosity_probability, [0.8, 1.0])

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

    def test_isolation_factor_uses_feeding_blocked_when_hotspot_missing(self) -> None:
        """Isolation factor is non-zero for feeding-blocked voxels even without hotspot."""
        engine = SolidificationEngine.__new__(SolidificationEngine)
        engine.mesh = SimpleNamespace(
            voxel_part_types=np.array([0, 0], dtype=np.int32),
            voxel_moduli=np.array([0.5, 0.5], dtype=np.float32),
        )
        engine.solidification_times = np.array([100.0, 100.0], dtype=np.float32)
        engine.hotspot_intensity = None
        engine.feeding_blocked = np.array([True, False], dtype=bool)
        engine.feeding_open_fraction = np.array([0.1, 1.0], dtype=np.float32)

        valid_indices = np.array([0, 1], dtype=np.int32)
        normalized_heights = np.array([0.5, 0.5], dtype=np.float32)
        local_section_size = np.array([1.0, 1.0], dtype=np.float32)
        riser_mask = np.array([False, False], dtype=bool)
        riser_post_mask = np.array([False, False], dtype=bool)

        (base_risk, _, _, _, _, _, _, _, _, _, _) = engine._compute_shrinkage_base_risk(
            valid_indices,
            normalized_heights,
            local_section_size,
            riser_mask,
            riser_post_mask,
        )

        # Blocked voxel (index 0) should have higher base risk due to
        # feeding_blocked contributing 0.85 to isolation_factor (40% weight).
        # Open voxel (index 1) has no isolation contribution.
        self.assertGreater(
            float(base_risk[0]),
            float(base_risk[1]),
            "Feeding-blocked voxel should have higher base_risk",
        )

    def test_shrinkage_keeps_isolated_pocket_visible(self) -> None:
        """Feeding-blocked voxels return high feed_loss so they remain visible.

        The low-evidence cap (Item 3) would otherwise suppress these voxels
        since they have low thermal evidence.  The blocked_floor override ensures
        they stay above the 0.30 UI threshold.
        """
        dummy = SimpleNamespace(
            mesh=SimpleNamespace(
                voxel_centers=np.zeros((4, 3), dtype=np.float32),
            ),
            shrinkage_probability=np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
            solidification_times=np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float32),
            niyama_severity=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            feeding_quality=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
            feeding_open_fraction=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
            feeding_blocked=np.array([False, False, True, True], dtype=bool),
            feeding_darcy_margin=np.array([1.0, 1.0, -0.8, -0.8], dtype=np.float32),
        )
        dummy._get_connected_feeder_mask = lambda: np.array([True, True, True, True], dtype=bool)

        evidence, feed_loss, healthy = SolidificationEngine._calculate_shrinkage_evidence(
            dummy,
            np.array([1.0, 1.0, 0.1, 0.1], dtype=np.float32),
        )

        # Voxels 0 and 1 are well-fed; their feed_loss should be low.
        self.assertLess(float(feed_loss[0]), 0.1)
        self.assertLess(float(feed_loss[1]), 0.1)
        # Voxels 2 and 3 are feeding-blocked; feed_loss must be high so that
        # the blocked_floor in _calculate_shrinkage keeps shrinkage visible.
        self.assertGreater(
            float(feed_loss[2]),
            0.9,
            "Blocked voxel 2 must have high feed_loss",
        )
        self.assertGreater(
            float(feed_loss[3]),
            0.9,
            "Blocked voxel 3 must have high feed_loss",
        )


if __name__ == "__main__":
    unittest.main()
