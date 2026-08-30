"""Tests for voxel part-type codes and the semantic grouping predicates.

``AdaptiveVoxelMesh.voxel_part_types`` is an int8 array. Before these existed,
the codes appeared as bare integers at ~45 comparison sites across core/, which
meant adding a part type required auditing every one of them.

The mold-filling stage added POUR_CUP, SPRUE, RUNNER, FILTER and GATE. The four
metal ones are metal-filled (so they solidify) but are not the casting (so they
must not be scored for shrinkage defects) -- that split is METAL_PART_CODES vs
SCORED_METAL_PART_CODES. FILTER is a refractory ceramic, so it groups with the
mold, but it restricts flow rather than blocking it, which is why
FLOW_RESTRICTOR_PART_CODES exists separately from MOLD_PART_CODES.

ALL_CODES below is derived from PART_TYPE_TO_CODE rather than listed by hand, so
a new part type cannot be added without these assertions being confronted.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.part_manager import (
    CODE_TO_PART_TYPE,
    FEEDER_PART_CODES,
    FLOW_PATH_PART_CODES,
    FLOW_RESTRICTOR_PART_CODES,
    GATING_METAL_PART_CODES,
    METAL_PART_CODES,
    MOLD_PART_CODES,
    PART_TYPE_CAST,
    PART_TYPE_CHILL,
    PART_TYPE_CORE,
    PART_TYPE_FILTER,
    PART_TYPE_GATE,
    PART_TYPE_POUR_CUP,
    PART_TYPE_RISER,
    PART_TYPE_RISER_POST,
    PART_TYPE_RUNNER,
    PART_TYPE_SPRUE,
    PART_TYPE_TO_CODE,
    SCORED_METAL_PART_CODES,
    PartType,
    is_cast,
    is_chill,
    is_feeder,
    is_filter,
    is_flow_path,
    is_flow_restrictor,
    is_gating_metal,
    is_metal,
    is_mold,
    is_scored_metal,
)


#: Every code, ascending. Derived so a new part type is automatically covered.
ALL_CODES = np.array(sorted(PART_TYPE_TO_CODE.values()), dtype=np.int32)


def _expected(*codes: int) -> np.ndarray:
    """Boolean expectation over ALL_CODES for the given member codes."""
    members = set(codes)
    return np.array([code in members for code in ALL_CODES], dtype=bool)


class PartTypeCodeTests(unittest.TestCase):
    def test_codes_match_the_documented_convention(self) -> None:
        """0=CAST 1=RISER 2=CHILL 3=RISER_POST 4=CORE, then the flow path.

        These integers are persisted in saved projects and encoded into ML
        training records, so they are not free to renumber. The gating codes
        run 5..9 in flow order: pour cup, sprue, runner, filter, gate.
        """
        self.assertEqual(PART_TYPE_CAST, 0)
        self.assertEqual(PART_TYPE_RISER, 1)
        self.assertEqual(PART_TYPE_CHILL, 2)
        self.assertEqual(PART_TYPE_RISER_POST, 3)
        self.assertEqual(PART_TYPE_CORE, 4)
        self.assertEqual(PART_TYPE_POUR_CUP, 5)
        self.assertEqual(PART_TYPE_SPRUE, 6)
        self.assertEqual(PART_TYPE_RUNNER, 7)
        self.assertEqual(PART_TYPE_FILTER, 8)
        self.assertEqual(PART_TYPE_GATE, 9)

    def test_mapping_covers_every_part_type(self) -> None:
        self.assertEqual(set(PART_TYPE_TO_CODE), set(PartType))

    def test_mapping_is_a_bijection(self) -> None:
        self.assertEqual(len(set(PART_TYPE_TO_CODE.values())), len(PART_TYPE_TO_CODE))
        for part_type, code in PART_TYPE_TO_CODE.items():
            self.assertIs(CODE_TO_PART_TYPE[code], part_type)


class GroupingTests(unittest.TestCase):
    def test_metal_and_mold_are_disjoint(self) -> None:
        self.assertEqual(set(METAL_PART_CODES) & set(MOLD_PART_CODES), set())

    def test_every_code_is_classified(self) -> None:
        """No part type may be neither metal nor mold."""
        classified = set(METAL_PART_CODES) | set(MOLD_PART_CODES)
        self.assertEqual(
            classified,
            set(PART_TYPE_TO_CODE.values()),
            msg="a part type belongs to neither METAL_PART_CODES nor MOLD_PART_CODES",
        )

    def test_feeders_are_metal(self) -> None:
        self.assertTrue(set(FEEDER_PART_CODES).issubset(set(METAL_PART_CODES)))

    def test_cast_is_metal_but_not_a_feeder(self) -> None:
        self.assertIn(PART_TYPE_CAST, METAL_PART_CODES)
        self.assertNotIn(PART_TYPE_CAST, FEEDER_PART_CODES)

    def test_scored_metal_is_a_strict_subset_of_metal(self) -> None:
        """Gating metal solidifies but must never reach a defect score."""
        self.assertTrue(set(SCORED_METAL_PART_CODES).issubset(set(METAL_PART_CODES)))
        self.assertTrue(set(SCORED_METAL_PART_CODES) < set(METAL_PART_CODES))

    def test_gating_metal_is_metal_and_is_never_scored(self) -> None:
        self.assertTrue(set(GATING_METAL_PART_CODES).issubset(set(METAL_PART_CODES)))
        self.assertEqual(
            set(GATING_METAL_PART_CODES) & set(SCORED_METAL_PART_CODES), set()
        )

    def test_scored_metal_and_gating_metal_exactly_partition_metal(self) -> None:
        self.assertEqual(
            set(SCORED_METAL_PART_CODES) | set(GATING_METAL_PART_CODES),
            set(METAL_PART_CODES),
        )

    def test_flow_path_is_gating_metal_plus_the_restrictors(self) -> None:
        self.assertEqual(
            set(GATING_METAL_PART_CODES) | set(FLOW_RESTRICTOR_PART_CODES),
            set(FLOW_PATH_PART_CODES),
        )

    def test_flow_restrictors_are_mold_not_metal(self) -> None:
        """A filter must not solidify, and must not be an impermeable wall."""
        self.assertTrue(set(FLOW_RESTRICTOR_PART_CODES).issubset(set(MOLD_PART_CODES)))
        self.assertEqual(set(FLOW_RESTRICTOR_PART_CODES) & set(METAL_PART_CODES), set())
        self.assertTrue(set(FLOW_RESTRICTOR_PART_CODES) < set(MOLD_PART_CODES))


class PredicateTests(unittest.TestCase):
    def test_is_metal_selects_casting_metal_and_gating_metal(self) -> None:
        np.testing.assert_array_equal(
            is_metal(ALL_CODES),
            _expected(
                PART_TYPE_CAST,
                PART_TYPE_RISER,
                PART_TYPE_RISER_POST,
                PART_TYPE_POUR_CUP,
                PART_TYPE_SPRUE,
                PART_TYPE_RUNNER,
                PART_TYPE_GATE,
            ),
        )

    def test_is_mold_selects_chill_core_and_filter(self) -> None:
        np.testing.assert_array_equal(
            is_mold(ALL_CODES),
            _expected(PART_TYPE_CHILL, PART_TYPE_CORE, PART_TYPE_FILTER),
        )

    def test_is_feeder_selects_riser_and_post(self) -> None:
        np.testing.assert_array_equal(
            is_feeder(ALL_CODES), _expected(PART_TYPE_RISER, PART_TYPE_RISER_POST)
        )

    def test_is_scored_metal_excludes_the_gating_system(self) -> None:
        """The whole point of the split: a sprue solidifies but is not scored."""
        np.testing.assert_array_equal(
            is_scored_metal(ALL_CODES),
            _expected(PART_TYPE_CAST, PART_TYPE_RISER, PART_TYPE_RISER_POST),
        )

    def test_is_flow_path_covers_the_gating_system_including_the_filter(self) -> None:
        np.testing.assert_array_equal(
            is_flow_path(ALL_CODES),
            _expected(
                PART_TYPE_POUR_CUP,
                PART_TYPE_SPRUE,
                PART_TYPE_RUNNER,
                PART_TYPE_FILTER,
                PART_TYPE_GATE,
            ),
        )

    def test_is_gating_metal_excludes_the_filter(self) -> None:
        """The filter is on the flow path but is ceramic, not metal."""
        np.testing.assert_array_equal(
            is_gating_metal(ALL_CODES),
            _expected(
                PART_TYPE_POUR_CUP, PART_TYPE_SPRUE, PART_TYPE_RUNNER, PART_TYPE_GATE
            ),
        )

    def test_is_flow_restrictor_and_is_filter_select_the_filter(self) -> None:
        np.testing.assert_array_equal(
            is_flow_restrictor(ALL_CODES), _expected(PART_TYPE_FILTER)
        )
        np.testing.assert_array_equal(is_filter(ALL_CODES), _expected(PART_TYPE_FILTER))

    def test_is_cast_and_is_chill_are_single_type(self) -> None:
        np.testing.assert_array_equal(is_cast(ALL_CODES), _expected(PART_TYPE_CAST))
        np.testing.assert_array_equal(is_chill(ALL_CODES), _expected(PART_TYPE_CHILL))

    def test_metal_and_mold_partition_the_array(self) -> None:
        metal = is_metal(ALL_CODES)
        mold = is_mold(ALL_CODES)
        np.testing.assert_array_equal(metal | mold, np.ones(len(ALL_CODES), dtype=bool))
        np.testing.assert_array_equal(metal & mold, np.zeros(len(ALL_CODES), dtype=bool))

    def test_predicates_accept_lists_and_preserve_shape(self) -> None:
        np.testing.assert_array_equal(is_cast([0, 1, 0]), np.array([True, False, True]))
        grid = np.array([[0, 2], [1, 4]], dtype=np.int32)
        self.assertEqual(is_metal(grid).shape, (2, 2))

    def test_empty_input_gives_empty_mask(self) -> None:
        empty = np.array([], dtype=np.int32)
        self.assertEqual(is_metal(empty).shape, (0,))
        self.assertEqual(is_metal(empty).dtype, bool)


if __name__ == "__main__":
    unittest.main()
