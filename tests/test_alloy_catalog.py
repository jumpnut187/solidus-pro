from __future__ import annotations

import unittest

from core.materials import get_all_alloy_names, get_alloy
from core.solidification import NIYAMA_THRESHOLDS


class AlloyCatalogTests(unittest.TestCase):
    added_alloys = ("C964", "MONEL400", "IN625", "CA6NM")

    def test_added_alloys_are_registered(self) -> None:
        names = set(get_all_alloy_names())
        for code in self.added_alloys:
            self.assertIn(code, names)

    def test_added_alloys_have_reasonable_metal_properties(self) -> None:
        for code in self.added_alloys:
            alloy = get_alloy(code)
            self.assertLess(alloy.solidus_temp, alloy.liquidus_temp)
            self.assertLess(alloy.liquidus_temp, alloy.pouring_temp)
            self.assertLess(0, alloy.density_change_ratio)
            self.assertLess(alloy.density_liquid, alloy.density_solid)
            self.assertGreater(alloy.thermal_conductivity, 0)
            self.assertGreater(alloy.specific_heat, 0)
            self.assertGreater(alloy.latent_heat, 0)
            self.assertGreater(alloy.volumetric_shrinkage, 0)
            self.assertLess(alloy.volumetric_shrinkage, 15)
            self.assertGreaterEqual(alloy.feeding_distance_multiplier, 0)
            self.assertLess(alloy.btr_fraction_solid_start, alloy.btr_fraction_solid_end)
            self.assertGreater(alloy.btr_fraction_solid_end, 0)
            self.assertLess(alloy.btr_fraction_solid_end, 1)
            self.assertGreater(alloy.critical_fraction_solid, 0)
            self.assertLess(alloy.critical_fraction_solid, 1)
            self.assertGreaterEqual(alloy.hot_tear_susceptibility, 0)
            self.assertLessEqual(alloy.hot_tear_susceptibility, 1)

    def test_volumetric_shrinkage_is_density_derived(self) -> None:
        for code in get_all_alloy_names():
            alloy = get_alloy(code)
            expected = (1.0 - (alloy.density_liquid / alloy.density_solid)) * 100.0
            self.assertAlmostEqual(alloy.volumetric_shrinkage, expected, places=6)

    def test_niyama_thresholds_cover_added_alloys(self) -> None:
        for code in self.added_alloys:
            self.assertIn(code, NIYAMA_THRESHOLDS)

    def test_surface_tension_is_positive_for_all_alloys(self) -> None:
        for code in get_all_alloy_names():
            with self.subTest(alloy=code):
                alloy = get_alloy(code)
                self.assertGreater(alloy.surface_tension, 0)

    def test_steel_surface_tension_in_expected_range(self) -> None:
        steels = ("WCB", "CF8", "CA6NM")
        for code in steels:
            with self.subTest(alloy=code):
                alloy = get_alloy(code)
                self.assertGreaterEqual(alloy.surface_tension, 1.5)
                self.assertLessEqual(alloy.surface_tension, 2.1)

    def test_copper_alloys_have_lower_surface_tension_than_steels(self) -> None:
        steels = ("WCB", "CF8", "CA6NM")
        copper_alloys = ("C958", "C964")
        min_steel_surface_tension = min(
            get_alloy(code).surface_tension for code in steels
        )
        for code in copper_alloys:
            with self.subTest(alloy=code):
                alloy = get_alloy(code)
                self.assertLess(alloy.surface_tension, min_steel_surface_tension)


if __name__ == "__main__":
    unittest.main()
