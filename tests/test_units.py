from __future__ import annotations

import unittest

import numpy as np

from physics.units import (
    GRAVITY_M_S2,
    IN_TO_M,
    IN_TO_MM,
    M_TO_IN,
    MM_TO_IN,
    inches_to_meters,
    inches_to_mm,
    meters_to_inches,
    mm_to_inches,
)

class UnitConstantTests(unittest.TestCase):
    def test_inch_is_exactly_25_4_mm(self) -> None:
        self.assertEqual(IN_TO_MM, 25.4)
        self.assertEqual(IN_TO_M, 0.0254)

    def test_reciprocals_are_consistent(self) -> None:
        self.assertAlmostEqual(IN_TO_M * M_TO_IN, 1.0, places=15)
        self.assertAlmostEqual(IN_TO_MM * MM_TO_IN, 1.0, places=15)

    def test_metre_and_millimetre_factors_agree(self) -> None:
        self.assertAlmostEqual(IN_TO_M * 1000.0, IN_TO_MM, places=12)

class ScalarConversionTests(unittest.TestCase):
    def test_known_values(self) -> None:
        self.assertAlmostEqual(inches_to_meters(1.0), 0.0254, places=12)
        self.assertAlmostEqual(inches_to_mm(1.0), 25.4, places=12)
        self.assertAlmostEqual(meters_to_inches(0.0254), 1.0, places=12)
        self.assertAlmostEqual(mm_to_inches(25.4), 1.0, places=12)

    def test_round_trip_metres(self) -> None:
        for value in (0.0, 1e-6, 0.1, 2.0, 137.5):
            self.assertAlmostEqual(meters_to_inches(inches_to_meters(value)), value, places=12)

    def test_round_trip_millimetres(self) -> None:
        for value in (0.0, 1e-6, 0.1, 2.0, 137.5):
            self.assertAlmostEqual(mm_to_inches(inches_to_mm(value)), value, places=12)

class ArrayConversionTests(unittest.TestCase):
    def test_arrays_convert_elementwise(self) -> None:
        inches = np.array([0.0, 0.5, 1.0, 4.0], dtype=np.float64)
        np.testing.assert_allclose(inches_to_meters(inches), inches * 0.0254, rtol=0, atol=0)
        np.testing.assert_allclose(inches_to_mm(inches), inches * 25.4, rtol=0, atol=0)

    def test_array_round_trip(self) -> None:
        inches = np.array([0.0, 0.125, 1.0, 12.0, 250.0], dtype=np.float64)
        np.testing.assert_allclose(meters_to_inches(inches_to_meters(inches)), inches, rtol=1e-12)

    def test_conversion_preserves_shape(self) -> None:
        centers = np.zeros((7, 3), dtype=np.float64)
        self.assertEqual(inches_to_meters(centers).shape, (7, 3))

class GravityPairingTests(unittest.TestCase):

    def test_gravity_is_si(self) -> None:
        self.assertAlmostEqual(GRAVITY_M_S2, 9.81, places=6)

    def test_head_pressure_requires_metre_height(self) -> None:
        density = 7200.0
        height_inches = 10.0

        correct = density * GRAVITY_M_S2 * inches_to_meters(height_inches)
        unconverted = density * GRAVITY_M_S2 * height_inches

        self.assertAlmostEqual(correct, 7200.0 * 9.81 * 0.254, places=6)
        self.assertAlmostEqual(unconverted / correct, M_TO_IN, places=9)

    def test_cfl_length_must_match_gravity_units(self) -> None:
        spacing_inches = 0.25
        velocity_m_s = 0.5

        dt_correct = inches_to_meters(spacing_inches) / velocity_m_s
        dt_wrong = spacing_inches / velocity_m_s

        self.assertAlmostEqual(dt_correct, 0.00635 / 0.5, places=9)
        self.assertGreater(dt_wrong / dt_correct, 39.0)

if __name__ == "__main__":
    unittest.main()
