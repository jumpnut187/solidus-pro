"""Tests for voxel part-type codes and the semantic grouping predicates.

``AdaptiveVoxelMesh.voxel_part_types`` is an int32 array. Before these existed,
the codes appeared as bare integers at ~45 comparison sites across core/, which
meant adding a part type required auditing every one of them.

A mold-filling stage will add SPRUE, RUNNER, GATE and POUR_CUP. Those are
metal-filled (so they solidify) but are not the casting (so they must not be
scored for shrinkage defects). The point of the groupings is that this
distinction gets encoded once, here.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.part_manager import (
    CODE_TO_PART_TYPE,
    FEEDER_PART_CODES,
    METAL_PART_CODES,
    MOLD_PART_CODES,
    PART_TYPE_CAST,
    PART_TYPE_CHILL,
    PART_TYPE_CORE,
    PART_TYPE_RISER,
    PART_TYPE_RISER_POST,
    PART_TYPE_TO_CODE,
    PartType,
    is_cast,
    is_chill,
    is_feeder,
    is_metal,
    is_mold,
)


ALL_CODES = np.array(
    [
        PART_TYPE_CAST,
        PART_TYPE_RISER,
        PART_TYPE_CHILL,
        PART_TYPE_RISER_POST,
        PART_TYPE_CORE,
    ],
    dtype=np.int32,
)


class PartTypeCodeTests(unittest.TestCase):
    def test_codes_match_the_documented_convention(self) -> None:
        """0=CAST, 1=RISER, 2=CHILL, 3=RISER_POST, 4=CORE.

        These integers are persisted in saved projects and encoded into ML
        training records, so they are not free to renumber.
        """
        self.assertEqual(PART_TYPE_CAST, 0)
        self.assertEqual(PART_TYPE_RISER, 1)
        self.assertEqual(PART_TYPE_CHILL, 2)
        self.assertEqual(PART_TYPE_RISER_POST, 3)
        self.assertEqual(PART_TYPE_CORE, 4)

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


class PredicateTests(unittest.TestCase):
    def test_is_metal_selects_cast_riser_and_post(self) -> None:
        np.testing.assert_array_equal(
            is_metal(ALL_CODES),
            np.array([True, True, False, True, False]),
        )

    def test_is_mold_selects_chill_and_core(self) -> None:
        np.testing.assert_array_equal(
            is_mold(ALL_CODES),
            np.array([False, False, True, False, True]),
        )

    def test_is_feeder_selects_riser_and_post(self) -> None:
        np.testing.assert_array_equal(
            is_feeder(ALL_CODES),
            np.array([False, True, False, True, False]),
        )

    def test_is_cast_and_is_chill_are_single_type(self) -> None:
        np.testing.assert_array_equal(
            is_cast(ALL_CODES), np.array([True, False, False, False, False])
        )
        np.testing.assert_array_equal(
            is_chill(ALL_CODES), np.array([False, False, True, False, False])
        )

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
