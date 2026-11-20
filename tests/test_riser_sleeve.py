"""Which riser voxels are bare (core/riser_sleeve.py) and the Part setting behind it."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import trimesh

from core.part_manager import (
    PART_TYPE_CAST,
    PART_TYPE_RISER,
    RISER_SLEEVE_ADIABATIC,
    RISER_SLEEVE_INSULATING,
    RISER_SLEEVE_NONE,
    build_part_manager,
    riser_sleeve_from_value,
)
from core.riser_sleeve import bare_riser_mask, riser_sleeve_masks


def _grid(nx: int):
    """A row of nx unit voxels along x, face neighbors only (slots -x, +x)."""
    centers = np.column_stack([np.arange(nx) + 0.5, np.full(nx, 0.5), np.full(nx, 0.5)])
    nbr = np.full((nx, 2), -1, dtype=np.int32)
    nbr[1:, 0] = np.arange(nx - 1)
    nbr[:-1, 1] = np.arange(1, nx)
    return centers, nbr


def _box(x0: float, x1: float) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=(x1 - x0, 1.0, 1.0))
    box.apply_translation(((x0 + x1) / 2.0, 0.5, 0.5))
    return box


class RiserSleeveValueTests(unittest.TestCase):
    def test_unknown_or_missing_is_insulating(self) -> None:
        self.assertEqual(riser_sleeve_from_value(None), RISER_SLEEVE_INSULATING)
        self.assertEqual(riser_sleeve_from_value("exothermic"), RISER_SLEEVE_INSULATING)
        self.assertEqual(riser_sleeve_from_value(" None "), RISER_SLEEVE_NONE)
        self.assertEqual(riser_sleeve_from_value("Adiabatic"), RISER_SLEEVE_ADIABATIC)

    def test_part_default_payload_override_and_riser_only_listing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "block.stl")
            _box(0, 1).export(path)
            manager = build_part_manager(
                [
                    {"filepath": path, "name": "a", "part_type": "riser"},
                    {"filepath": path, "name": "b", "part_type": "riser", "riser_sleeve": "none"},
                    {"filepath": path, "name": "c", "part_type": "cast", "riser_sleeve": "none"},
                ]
            )
        a, b, c = manager.get_all_parts()
        self.assertEqual(a.riser_sleeve, RISER_SLEEVE_INSULATING)
        self.assertEqual(b.riser_sleeve, RISER_SLEEVE_NONE)
        self.assertEqual([sleeve for _, sleeve in manager.get_riser_sleeve_parts()], ["insulating", "none"])
        copy = manager.get_part(manager.duplicate_part(b.id))
        self.assertEqual(copy.riser_sleeve, RISER_SLEEVE_NONE)


class BareRiserMaskTests(unittest.TestCase):
    def setUp(self) -> None:
        # Voxels 0-1 cast, 2-3 one riser, 4 cast, 5-6 a second riser.
        self.centers, self.neighbors = _grid(7)
        self.types = np.array([PART_TYPE_CAST] * 2 + [PART_TYPE_RISER] * 2 + [PART_TYPE_CAST] + [PART_TYPE_RISER] * 2)

    def test_none_when_every_riser_is_insulating(self) -> None:
        parts = [(_box(2, 4), RISER_SLEEVE_INSULATING), (_box(5, 7), RISER_SLEEVE_INSULATING)]
        self.assertIsNone(bare_riser_mask(self.centers, self.types, self.neighbors, parts))

    def test_only_the_bare_riser_is_marked(self) -> None:
        parts = [(_box(2, 4), RISER_SLEEVE_NONE), (_box(5, 7), RISER_SLEEVE_INSULATING)]
        mask = bare_riser_mask(self.centers, self.types, self.neighbors, parts)
        self.assertEqual(mask.tolist(), [False, False, True, True, False, False, False])

    def test_component_spanning_parts_follows_the_majority_and_ties_are_insulating(self) -> None:
        centers, neighbors = _grid(4)
        types = np.full(4, PART_TYPE_RISER)
        # Three voxels in the bare part, one in the sleeved: the component is bare.
        parts = [(_box(0, 3), RISER_SLEEVE_NONE), (_box(3, 4), RISER_SLEEVE_INSULATING)]
        self.assertTrue(bare_riser_mask(centers, types, neighbors, parts).all())
        # Two and two: a tie is insulating, so there is nothing to mark.
        parts = [(_box(0, 2), RISER_SLEEVE_NONE), (_box(2, 4), RISER_SLEEVE_INSULATING)]
        self.assertIsNone(bare_riser_mask(centers, types, neighbors, parts))


class NoHeatLossRiserMaskTests(unittest.TestCase):
    def setUp(self) -> None:
        # Voxels 0-1 cast, 2-3 one riser, 4 cast, 5-6 a second riser.
        self.centers, self.neighbors = _grid(7)
        self.types = np.array([PART_TYPE_CAST] * 2 + [PART_TYPE_RISER] * 2 + [PART_TYPE_CAST] + [PART_TYPE_RISER] * 2)

    def test_nothing_is_marked_when_every_riser_is_insulating(self) -> None:
        parts = [(_box(2, 4), RISER_SLEEVE_INSULATING), (_box(5, 7), RISER_SLEEVE_INSULATING)]
        self.assertEqual(riser_sleeve_masks(self.centers, self.types, self.neighbors, parts), (None, None))

    def test_each_riser_takes_its_own_sleeve(self) -> None:
        parts = [(_box(2, 4), RISER_SLEEVE_ADIABATIC), (_box(5, 7), RISER_SLEEVE_NONE)]
        bare, adiabatic = riser_sleeve_masks(self.centers, self.types, self.neighbors, parts)
        self.assertEqual(adiabatic.tolist(), [False, False, True, True, False, False, False])
        self.assertEqual(bare.tolist(), [False, False, False, False, False, True, True])
        # One no-heat-loss riser beside an insulating one: no bare mask at all.
        parts = [(_box(2, 4), RISER_SLEEVE_ADIABATIC), (_box(5, 7), RISER_SLEEVE_INSULATING)]
        bare, adiabatic = riser_sleeve_masks(self.centers, self.types, self.neighbors, parts)
        self.assertIsNone(bare)
        self.assertEqual(adiabatic.tolist(), [False, False, True, True, False, False, False])

    def test_component_spanning_parts_takes_the_most_voxels_and_ties_are_insulating(self) -> None:
        centers, neighbors = _grid(5)
        types = np.full(5, PART_TYPE_RISER)
        parts = [(_box(0, 3), RISER_SLEEVE_ADIABATIC), (_box(3, 4), RISER_SLEEVE_NONE), (_box(4, 5), RISER_SLEEVE_INSULATING)]
        bare, adiabatic = riser_sleeve_masks(centers, types, neighbors, parts)
        self.assertIsNone(bare)
        self.assertTrue(adiabatic.all())
        # Two no-heat-loss, two bare, one insulating: nobody has the most, so the component is insulating.
        parts = [(_box(0, 2), RISER_SLEEVE_ADIABATIC), (_box(2, 4), RISER_SLEEVE_NONE), (_box(4, 5), RISER_SLEEVE_INSULATING)]
        self.assertEqual(riser_sleeve_masks(centers, types, neighbors, parts), (None, None))


try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
except ImportError:  # pragma: no cover - no Qt in this environment
    QApplication = None


@unittest.skipIf(QApplication is None, "PySide6 is not installed")
class PartsDialogSleeveTests(unittest.TestCase):
    def test_sleeve_column_follows_the_part_and_the_type(self) -> None:
        from ui.dialogs.parts_dialog import PartsDialog

        app = QApplication.instance() or QApplication([])
        self.assertIsNotNone(app)
        dialog = PartsDialog()
        changes = []
        dialog.riser_sleeve_changed.connect(lambda part_id, value: changes.append((part_id, value)))
        dialog.populate_parts(
            [("a", "Riser", "riser", True, "none"), ("b", "Casting", "cast", False), ("c", "Old", "riser")]
        )
        table = dialog.parts_table
        self.assertEqual(table.cellWidget(0, 3).currentData(), "none")
        self.assertEqual(table.cellWidget(2, 3).currentData(), "insulating")  # a part tuple without one
        self.assertFalse(table.cellWidget(1, 3).isEnabled())
        self.assertEqual(changes, [])  # populating the table announces nothing

        combo = table.cellWidget(2, 3)
        self.assertEqual([combo.itemData(i) for i in range(combo.count())], ["insulating", "adiabatic", "none"])
        combo.setCurrentIndex(combo.findData("none"))
        self.assertEqual(changes, [("c", "none")])
        combo.setCurrentIndex(combo.findData("adiabatic"))
        self.assertEqual(changes[-1], ("c", "adiabatic"))
        # Retyping a riser away from Riser puts its sleeve back to the default.
        table.cellWidget(2, 1).setCurrentText("Cast")
        self.assertEqual(changes[-1], ("c", "insulating"))
        self.assertFalse(table.cellWidget(2, 3).isEnabled())


if __name__ == "__main__":
    unittest.main()
