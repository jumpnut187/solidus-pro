from __future__ import annotations

import unittest

import numpy as np

from core import part_manager
from ui.main_window import MainWindow

class VoxelMaskFromPartsTests(unittest.TestCase):

    def _window_with(self, parts, hidden):
        mw = MainWindow.__new__(MainWindow)

        class _PM:
            def __init__(self, items):
                self.parts = items

            def get_part(self, part_id):
                return self.parts.get(part_id)

        class _Viz:
            def hidden_part_ids(self):
                return set(hidden)

        mw.part_manager = _PM(parts)
        mw.visualizer = _Viz()
        return mw

    def _fake_part(self, part_type, lo, hi):
        class _Mesh:
            bounds = np.array([lo, hi], dtype=np.float64)

        class _Part:
            name = "p"

        part = _Part()
        part.part_type = part_type
        part.get_transformed_mesh = lambda: _Mesh()
        return part

    def test_mask_drops_only_matching_type_inside_hidden_box(self) -> None:
        PartType = part_manager.PartType
        code = part_manager.PART_TYPE_TO_CODE
        parts = {
            "cast": self._fake_part(PartType.CAST, [0, 0, 0], [10, 10, 10]),
            "riser": self._fake_part(PartType.RISER, [4, 4, 10], [6, 6, 14]),
        }
        mw = self._window_with(parts, hidden={"riser"})

        class _Mesh:
            base_voxel_size = 1.0
            voxel_centers = np.array([
                [5, 5, 5],
                [5, 5, 12],
                [5, 5, 12.5],
                [9, 9, 9],
            ], dtype=np.float64)
            voxel_part_types = np.array([
                code[PartType.CAST], code[PartType.RISER], code[PartType.CAST], code[PartType.CAST]
            ], dtype=np.int8)

        mask = mw._build_voxel_display_mask(_Mesh())
        np.testing.assert_array_equal(mask, np.array([True, False, True, True]))

    def test_no_hidden_parts_means_no_mask(self) -> None:
        mw = self._window_with({}, hidden=set())

        class _Mesh:
            voxel_centers = np.zeros((3, 3))

        self.assertIsNone(mw._build_voxel_display_mask(_Mesh()))

    def test_box_padding_uses_half_voxel(self) -> None:
        PartType = part_manager.PartType
        parts = {"chill": self._fake_part(PartType.CHILL, [0, 0, 0], [2, 2, 2])}
        mw = self._window_with(parts, hidden={"chill"})

        class _Mesh:
            base_voxel_size = 1.0
            voxel_centers = np.array([[2.4, 1, 1], [2.6, 1, 1]], dtype=np.float64)
            voxel_part_types = None

        mask = mw._build_voxel_display_mask(_Mesh())
        np.testing.assert_array_equal(mask, np.array([False, True]))

if __name__ == "__main__":
    unittest.main()
