from __future__ import annotations

import collections
import os
import statistics
import tempfile
import unittest

from core.filter_database import (
    ALLOY_STEEL_CLASS,
    CAPACITY_KG_PER_CM2,
    CAPACITY_VALIDATED_THICKNESS_MM,
    DEOX_CASI_AL,
    DEOX_ZR_TI,
    DEOXIDATION_PRACTICES,
    FLOW_RATE_KG_S_PER_CM2,
    LIQUID_STEEL_DENSITY_KG_M3,
    MIN_FILTER_TO_CHOKE_AREA_RATIO,
    POUR_MODE_BOTTOM,
    POUR_MODE_LIP,
    POUR_MODES,
    SHAPE_ROUND,
    SHAPE_SQUARE,
    SHAPES,
    STEEL_CARBON_LOW_ALLOY,
    STEEL_CLASSES,
    STEEL_HIGH_ALLOY_STAINLESS,
    THIN_FILTER_CAPACITY_FACTOR,
    capacity_kg_for_area,
    flow_rate_kg_s_for_area,
    load_filter_database,
    match_filter_by_area,
    select_filter,
    steel_class_for_alloy,
)
from core.materials import get_all_alloy_names

_HEADER = (
    "brand,product_line,material,nominal_size,shape,width_mm,height_mm,diameter_mm,"
    "thickness_mm,front_face_area_cm2,steel_class,pour_mode,deoxidation,"
    "flow_rate_kg_s,capacity_kg,max_pour_temp_c,source\n"
)
_GOOD_ROW = (
    "Foseco,STELEX,zirconia,75x75x25,square,75.0,75.0,,25.0,56.25,"
    "carbon_low_alloy,lip,zr_ti,5.6,85.0,1680,handbook_t18_2\n"
)

class CatalogShapeTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.db = load_filter_database()

    def test_all_104_published_cells_are_present(self) -> None:
        self.assertEqual(len(self.db), 104)
        self.assertEqual(len({spec.nominal_size for spec in self.db}), 13)

    def test_both_shapes_are_carried(self) -> None:
        by_shape = collections.Counter(spec.shape for spec in self.db)
        self.assertEqual(by_shape[SHAPE_SQUARE], 6 * 8)
        self.assertEqual(by_shape[SHAPE_ROUND], 7 * 8)

    def test_every_key_combination_appears_exactly_once_per_size(self) -> None:
        seen = collections.Counter(
            (s.nominal_size, s.steel_class, s.pour_mode, s.deoxidation) for s in self.db
        )
        for key, count in seen.items():
            with self.subTest(key=key):
                self.assertEqual(count, 1)

    def test_every_row_is_zirconia_stelex_for_steel(self) -> None:
        for spec in self.db:
            with self.subTest(size=spec.nominal_size, cls=spec.steel_class):
                self.assertEqual(spec.material, "zirconia")
                self.assertEqual(spec.brand, "Foseco")
                self.assertIn(spec.steel_class, STEEL_CLASSES)
                self.assertIn(spec.pour_mode, POUR_MODES)
                self.assertIn(spec.deoxidation, DEOXIDATION_PRACTICES)
                self.assertIn(spec.shape, SHAPES)

    def test_face_area_agrees_with_stated_dimensions(self) -> None:
        import math

        for spec in self.db:
            with self.subTest(size=spec.nominal_size):
                if spec.shape == SHAPE_SQUARE:
                    assert spec.width_mm is not None and spec.height_mm is not None
                    derived = spec.width_mm * spec.height_mm / 100.0
                    self.assertIsNone(spec.diameter_mm)
                else:
                    assert spec.diameter_mm is not None
                    derived = math.pi * spec.diameter_mm ** 2 / 4.0 / 100.0
                    self.assertIsNone(spec.width_mm)
                self.assertAlmostEqual(spec.front_face_area_cm2, derived, places=3)

    def test_the_six_square_catalog_sizes_are_present(self) -> None:
        square = {s.nominal_size for s in self.db if s.shape == SHAPE_SQUARE}
        self.assertEqual(
            square,
            {"50x50x20", "55x55x25", "75x75x25", "100x100x25", "125x125x30", "150x150x30"},
        )

    def test_the_seven_round_catalog_sizes_are_present(self) -> None:
        round_sizes = {s.nominal_size for s in self.db if s.shape == SHAPE_ROUND}
        self.assertEqual(
            round_sizes,
            {
                "50dia x 25",
                "60dia x 25",
                "70dia x 25",
                "90dia x 25",
                "100dia x 25",
                "125dia x 30",
                "150dia x 30",
            },
        )

    def test_missing_database_returns_empty_not_raises(self) -> None:
        self.assertEqual(load_filter_database("/nonexistent/filter_database.csv"), [])

    def test_malformed_rows_are_skipped_not_fatal(self) -> None:
        bad_number = _GOOD_ROW.replace(",56.25,", ",NOT_A_NUMBER,")
        bad_enum = _GOOD_ROW.replace(",carbon_low_alloy,", ",cast_iron,")
        bad_deox = _GOOD_ROW.replace(",zr_ti,", ",mystery,")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "filters.csv")
            with open(path, "w") as handle:
                handle.write(_HEADER + _GOOD_ROW + bad_number + bad_enum + bad_deox)
            specs = load_filter_database(path)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].nominal_size, "75x75x25")

class FlowRateIsAnExactAreaLawTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.db = load_filter_database()

    def test_square_flow_cells_match_the_area_law_to_publication_rounding(self) -> None:
        for spec in self.db:
            if spec.shape != SHAPE_SQUARE:
                continue
            with self.subTest(
                size=spec.nominal_size, cls=spec.steel_class, mode=spec.pour_mode
            ):
                predicted = flow_rate_kg_s_for_area(
                    spec.front_face_area_cm2, spec.steel_class, spec.pour_mode
                )
                self.assertLess(
                    abs(predicted - spec.flow_rate_kg_s) / spec.flow_rate_kg_s,
                    0.01,
                    f"{spec.nominal_size} {spec.steel_class}/{spec.pour_mode}: "
                    f"area law gives {predicted:.3f}, table says {spec.flow_rate_kg_s}",
                )

    def test_round_flow_cells_follow_the_law_only_approximately(self) -> None:
        worst = 0.0
        for spec in self.db:
            if spec.shape != SHAPE_ROUND:
                continue
            predicted = flow_rate_kg_s_for_area(
                spec.front_face_area_cm2, spec.steel_class, spec.pour_mode
            )
            relative = abs(predicted - spec.flow_rate_kg_s) / spec.flow_rate_kg_s
            with self.subTest(size=spec.nominal_size, cls=spec.steel_class, mode=spec.pour_mode):
                self.assertLess(relative, 0.04)
            worst = max(worst, relative)
        self.assertGreater(
            worst, 0.01, "round drift vanished - the law is tighter than documented"
        )

    def test_the_law_is_exact_at_100x100(self) -> None:
        exact = {
            (STEEL_CARBON_LOW_ALLOY, POUR_MODE_LIP): 10.0,
            (STEEL_CARBON_LOW_ALLOY, POUR_MODE_BOTTOM): 12.5,
            (STEEL_HIGH_ALLOY_STAINLESS, POUR_MODE_LIP): 14.0,
            (STEEL_HIGH_ALLOY_STAINLESS, POUR_MODE_BOTTOM): 17.5,
        }
        for spec in self.db:
            if spec.nominal_size != "100x100x25":
                continue
            with self.subTest(cls=spec.steel_class, mode=spec.pour_mode):
                self.assertEqual(
                    spec.flow_rate_kg_s, exact[(spec.steel_class, spec.pour_mode)]
                )

    def test_flow_is_independent_of_deoxidation_practice(self) -> None:
        by_key = collections.defaultdict(dict)
        for spec in self.db:
            key = (spec.nominal_size, spec.steel_class, spec.pour_mode)
            by_key[key][spec.deoxidation] = spec.flow_rate_kg_s
        self.assertEqual(len(by_key), 13 * 2 * 2)
        for key, flows in by_key.items():
            with self.subTest(key=key):
                self.assertEqual(flows[DEOX_ZR_TI], flows[DEOX_CASI_AL])

    def test_flow_scales_linearly_with_area(self) -> None:
        base = flow_rate_kg_s_for_area(25.0, STEEL_CARBON_LOW_ALLOY, POUR_MODE_LIP)
        quad = flow_rate_kg_s_for_area(100.0, STEEL_CARBON_LOW_ALLOY, POUR_MODE_LIP)
        self.assertAlmostEqual(quad, 4.0 * base, places=9)

    def test_bottom_pour_exceeds_lip_pour(self) -> None:
        for steel_class in STEEL_CLASSES:
            with self.subTest(cls=steel_class):
                self.assertGreater(
                    flow_rate_kg_s_for_area(100.0, steel_class, POUR_MODE_BOTTOM),
                    flow_rate_kg_s_for_area(100.0, steel_class, POUR_MODE_LIP),
                )

    def test_high_alloy_exceeds_carbon(self) -> None:
        for mode in POUR_MODES:
            with self.subTest(mode=mode):
                self.assertGreater(
                    flow_rate_kg_s_for_area(100.0, STEEL_HIGH_ALLOY_STAINLESS, mode),
                    flow_rate_kg_s_for_area(100.0, STEEL_CARBON_LOW_ALLOY, mode),
                )

class CapacityIsNotAnAreaLawTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.db = load_filter_database()

    def test_law_holds_for_square_filters_at_or_above_25mm(self) -> None:
        for spec in self.db:
            if spec.shape != SHAPE_SQUARE or spec.thickness_mm < CAPACITY_VALIDATED_THICKNESS_MM:
                continue
            with self.subTest(
                size=spec.nominal_size, cls=spec.steel_class,
                mode=spec.pour_mode, deox=spec.deoxidation,
            ):
                predicted = capacity_kg_for_area(
                    spec.front_face_area_cm2, spec.steel_class,
                    spec.pour_mode, spec.deoxidation, spec.thickness_mm,
                )
                self.assertLess(
                    abs(predicted - spec.capacity_kg) / spec.capacity_kg, 0.02
                )

    def test_the_20mm_filter_really_does_deviate(self) -> None:
        ratios = []
        for spec in self.db:
            if spec.nominal_size != "50x50x20":
                continue
            key = (spec.steel_class, spec.pour_mode, spec.deoxidation)
            uncorrected = CAPACITY_KG_PER_CM2[key] * spec.front_face_area_cm2
            ratios.append(spec.capacity_kg / uncorrected)
        self.assertEqual(len(ratios), 8)
        self.assertGreater(min(ratios), 0.92)
        self.assertLess(max(ratios), 0.95)
        self.assertAlmostEqual(statistics.mean(ratios), THIN_FILTER_CAPACITY_FACTOR, places=2)

    def test_thin_filter_correction_is_applied_and_lowers_capacity(self) -> None:
        thick = capacity_kg_for_area(25.0, STEEL_CARBON_LOW_ALLOY, POUR_MODE_LIP, DEOX_ZR_TI, 25.0)
        thin = capacity_kg_for_area(25.0, STEEL_CARBON_LOW_ALLOY, POUR_MODE_LIP, DEOX_ZR_TI, 20.0)
        self.assertLess(thin, thick)
        self.assertAlmostEqual(thin, thick * THIN_FILTER_CAPACITY_FACTOR, places=9)

    def test_corrected_law_matches_the_published_20mm_values(self) -> None:
        for spec in self.db:
            if spec.nominal_size != "50x50x20":
                continue
            with self.subTest(cls=spec.steel_class, mode=spec.pour_mode, deox=spec.deoxidation):
                predicted = capacity_kg_for_area(
                    spec.front_face_area_cm2, spec.steel_class,
                    spec.pour_mode, spec.deoxidation, spec.thickness_mm,
                )
                self.assertLess(abs(predicted - spec.capacity_kg) / spec.capacity_kg, 0.02)

    def test_casi_al_multiplier_is_not_uniform(self) -> None:
        ratios = {}
        for spec in self.db:
            key = (spec.steel_class, spec.pour_mode)
            ratios.setdefault(key, {})[spec.deoxidation] = ratios.get(key, {}).get(
                spec.deoxidation, 0.0
            ) + spec.capacity_kg
        computed = {k: v[DEOX_CASI_AL] / v[DEOX_ZR_TI] for k, v in ratios.items()}
        outlier = computed[(STEEL_HIGH_ALLOY_STAINLESS, POUR_MODE_BOTTOM)]
        others = [v for k, v in computed.items() if k != (STEEL_HIGH_ALLOY_STAINLESS, POUR_MODE_BOTTOM)]
        for value in others:
            self.assertAlmostEqual(value, 1.50, delta=0.02)
        self.assertGreater(outlier, 1.60)

    def test_round_capacities_are_not_reproduced_by_the_law(self) -> None:
        worst = 0.0
        for spec in self.db:
            if spec.shape != SHAPE_ROUND:
                continue
            predicted = capacity_kg_for_area(
                spec.front_face_area_cm2, spec.steel_class,
                spec.pour_mode, spec.deoxidation, spec.thickness_mm,
            )
            worst = max(worst, abs(predicted - spec.capacity_kg) / spec.capacity_kg)
        self.assertGreater(worst, 0.03, "round scatter vanished - re-check the data")

class RatingGuardTests(unittest.TestCase):

    def test_unknown_steel_class_raises(self) -> None:
        with self.assertRaises(ValueError):
            flow_rate_kg_s_for_area(100.0, "cast_iron", POUR_MODE_LIP)

    def test_unknown_pour_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            flow_rate_kg_s_for_area(100.0, STEEL_CARBON_LOW_ALLOY, "tilt")

    def test_unknown_deoxidation_raises(self) -> None:
        with self.assertRaises(ValueError):
            capacity_kg_for_area(100.0, STEEL_CARBON_LOW_ALLOY, POUR_MODE_LIP, "mystery")

    def test_non_positive_area_raises(self) -> None:
        for area in (0.0, -5.0):
            with self.subTest(area=area):
                with self.assertRaises(ValueError):
                    flow_rate_kg_s_for_area(area, STEEL_CARBON_LOW_ALLOY, POUR_MODE_LIP)
                with self.assertRaises(ValueError):
                    capacity_kg_for_area(area, STEEL_CARBON_LOW_ALLOY, POUR_MODE_LIP)

    def test_every_class_mode_pair_has_a_flow_constant(self) -> None:
        for steel_class in STEEL_CLASSES:
            for mode in POUR_MODES:
                with self.subTest(cls=steel_class, mode=mode):
                    self.assertIn((steel_class, mode), FLOW_RATE_KG_S_PER_CM2)

    def test_every_triple_has_a_capacity_constant(self) -> None:
        for steel_class in STEEL_CLASSES:
            for mode in POUR_MODES:
                for deox in DEOXIDATION_PRACTICES:
                    with self.subTest(cls=steel_class, mode=mode, deox=deox):
                        self.assertIn((steel_class, mode, deox), CAPACITY_KG_PER_CM2)

class AlloyMappingTests(unittest.TestCase):
    def test_mapped_alloys_exist_in_the_catalog(self) -> None:
        names = set(get_all_alloy_names())
        for code in ALLOY_STEEL_CLASS:
            with self.subTest(alloy=code):
                self.assertIn(code, names)

    def test_steels_map_and_non_ferrous_do_not(self) -> None:
        self.assertEqual(steel_class_for_alloy("WCB"), STEEL_CARBON_LOW_ALLOY)
        self.assertEqual(steel_class_for_alloy("CF8"), STEEL_HIGH_ALLOY_STAINLESS)
        self.assertEqual(steel_class_for_alloy("CA6NM"), STEEL_HIGH_ALLOY_STAINLESS)
        for code in ("C958", "C964", "MONEL400", "IN625"):
            with self.subTest(alloy=code):
                self.assertIsNone(
                    steel_class_for_alloy(code),
                    "the Foseco steel tables must not be applied to a non-ferrous alloy",
                )

    def test_lookup_is_case_and_whitespace_insensitive(self) -> None:
        self.assertEqual(steel_class_for_alloy("  wcb "), STEEL_CARBON_LOW_ALLOY)

    def test_every_catalog_alloy_is_classified_or_deliberately_excluded(self) -> None:
        known_non_ferrous = {"C958", "C964", "MONEL400", "IN625"}
        for name in get_all_alloy_names():
            with self.subTest(alloy=name):
                self.assertTrue(
                    name in ALLOY_STEEL_CLASS or name in known_non_ferrous,
                    f"alloy {name} is neither mapped to a Foseco steel class nor "
                    f"listed as non-ferrous - classify it in ALLOY_STEEL_CLASS",
                )

class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db = load_filter_database()

    def test_selects_smallest_adequate_square_filter(self) -> None:
        spec = select_filter(6.0, self.db, STEEL_CARBON_LOW_ALLOY, POUR_MODE_LIP)
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.nominal_size, "100x100x25")

    def test_exact_requirement_is_satisfied_by_that_size(self) -> None:
        spec = select_filter(5.6, self.db, STEEL_CARBON_LOW_ALLOY, POUR_MODE_LIP)
        assert spec is not None
        self.assertEqual(spec.nominal_size, "75x75x25")

    def test_returns_none_when_no_single_filter_suffices(self) -> None:
        self.assertIsNone(select_filter(1e6, self.db, STEEL_CARBON_LOW_ALLOY))

    def test_selection_respects_steel_class(self) -> None:
        carbon = select_filter(7.0, self.db, STEEL_CARBON_LOW_ALLOY, POUR_MODE_LIP)
        high = select_filter(7.0, self.db, STEEL_HIGH_ALLOY_STAINLESS, POUR_MODE_LIP)
        assert carbon is not None and high is not None
        self.assertLess(high.front_face_area_cm2, carbon.front_face_area_cm2)

    def test_round_filters_are_selectable(self) -> None:
        spec = select_filter(
            6.0, self.db, STEEL_CARBON_LOW_ALLOY, POUR_MODE_LIP, shape=SHAPE_ROUND
        )
        assert spec is not None
        self.assertEqual(spec.shape, SHAPE_ROUND)

    def test_shape_none_considers_both(self) -> None:
        spec = select_filter(
            6.0, self.db, STEEL_CARBON_LOW_ALLOY, POUR_MODE_LIP, shape=None
        )
        assert spec is not None
        self.assertEqual(spec.shape, SHAPE_ROUND)

class AreaMatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db = load_filter_database()

    def test_matches_a_modelled_body_to_its_catalog_size(self) -> None:
        spec = match_filter_by_area(
            55.06, self.db, STEEL_CARBON_LOW_ALLOY, POUR_MODE_LIP, shape=SHAPE_SQUARE
        )
        assert spec is not None
        self.assertEqual(spec.nominal_size, "75x75x25")

    def test_rejects_a_body_that_is_no_catalog_size(self) -> None:
        self.assertIsNone(match_filter_by_area(5000.0, self.db, STEEL_CARBON_LOW_ALLOY))

    def test_non_positive_area_returns_none(self) -> None:
        self.assertIsNone(match_filter_by_area(0.0, self.db, STEEL_CARBON_LOW_ALLOY))

class SolverUnitTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.db = load_filter_database()

    def test_si_conversions(self) -> None:
        spec = next(s for s in self.db if s.nominal_size == "100x100x25")
        self.assertAlmostEqual(spec.thickness_m, 0.025, places=9)
        self.assertAlmostEqual(spec.front_face_area_m2, 0.01, places=9)

    def test_rated_velocity_honours_the_alloy_density(self) -> None:
        spec = next(s for s in self.db if s.nominal_size == "100x100x25")
        generic = spec.rated_superficial_velocity_m_s()
        wcb = spec.rated_superficial_velocity_m_s(7200.0)
        self.assertNotAlmostEqual(generic, wcb, places=4)
        self.assertAlmostEqual(wcb, generic * (7000.0 / 7200.0), places=12)

    def test_rated_velocity_matches_its_definition(self) -> None:
        spec = next(s for s in self.db if s.nominal_size == "100x100x25")
        expected = spec.flow_rate_kg_s / (
            LIQUID_STEEL_DENSITY_KG_M3 * spec.front_face_area_m2
        )
        self.assertAlmostEqual(spec.rated_superficial_velocity_m_s(), expected, places=12)

    def test_rated_velocity_sits_in_the_practice_band(self) -> None:
        for spec in self.db:
            with self.subTest(size=spec.nominal_size, cls=spec.steel_class, mode=spec.pour_mode):
                u_rated = spec.rated_superficial_velocity_m_s(7200.0)
                self.assertGreater(u_rated, 0.10)
                self.assertLess(u_rated, 0.26)

    def test_rating_brackets_campbells_critical_ingate_velocity(self) -> None:
        implied = {
            (s.steel_class, s.pour_mode): s.rated_superficial_velocity_m_s(7200.0)
            * MIN_FILTER_TO_CHOKE_AREA_RATIO
            for s in self.db
            if s.shape == SHAPE_SQUARE
        }
        self.assertLess(implied[(STEEL_CARBON_LOW_ALLOY, POUR_MODE_LIP)], 0.5)
        self.assertGreater(implied[(STEEL_HIGH_ALLOY_STAINLESS, POUR_MODE_BOTTOM)], 0.5)
        for key, velocity in implied.items():
            with self.subTest(column=key):
                self.assertGreater(velocity, 0.5 / 1.5)
                self.assertLess(velocity, 0.5 * 1.5)

    def test_face_velocity_is_size_independent_within_a_column(self) -> None:
        for steel_class in STEEL_CLASSES:
            for mode in POUR_MODES:
                velocities = [
                    s.rated_superficial_velocity_m_s()
                    for s in self.db
                    if s.steel_class == steel_class
                    and s.pour_mode == mode
                    and s.shape == SHAPE_SQUARE
                ]
                with self.subTest(cls=steel_class, mode=mode):
                    spread = (max(velocities) - min(velocities)) / min(velocities)
                    self.assertLess(spread, 0.02, f"face velocities vary: {velocities}")

if __name__ == "__main__":
    unittest.main()
