from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import trimesh
from PySide6.QtWidgets import QApplication

from core import orientation
from core.part_manager import Part, PartManager, PartType
from ui.dialogs.orientation_dialog import (
    PICK_FACE_DOWN,
    PICK_PARTING_HEIGHT,
    OrientationDialog,
)
from ui.orientation_tool import (
    OrientationController,
    PartingLine,
    default_seat_side,
    outward_face_normal,
    part_world_bounds,
)
from ui.project_state import build_project_state, load_project_state, normalize_parting_line

def _add_box(manager: PartManager, name: str, extents, part_type=PartType.CAST, position=(0.0, 0.0, 0.0)) -> Part:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(np.asarray(extents, dtype=float) / 2.0)
    part = Part(id=name, name=name, filepath=f"{name}.stl", trimesh_obj=mesh)
    part.set_part_type(part_type)
    part.set_position(*(float(v) for v in position))
    manager.parts[name] = part
    manager._part_order.append(name)
    return part

def _flange_casting(manager: PartManager) -> Part:
    flange = trimesh.creation.box(extents=(6.0, 6.0, 1.0))
    flange.apply_translation((3.0, 3.0, 0.5))
    boss = trimesh.creation.box(extents=(2.0, 2.0, 3.0))
    boss.apply_translation((3.0, 3.0, 2.5))
    mesh = trimesh.util.concatenate([flange, boss])
    part = Part(id="casting", name="casting", filepath="casting.stl", trimesh_obj=mesh)
    part.set_part_type(PartType.CAST)
    manager.parts["casting"] = part
    manager._part_order.append("casting")
    return part

class SelectionStub:
    def __init__(self) -> None:
        self.part_id = None

    def __call__(self):
        return self.part_id

class OrientationControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = PartManager()
        self.selection = SelectionStub()
        self.controller = OrientationController(self.manager, self.selection)

    def test_target_is_selected_part_else_casting(self) -> None:
        casting = _add_box(self.manager, "casting", (4, 2, 1))
        riser = _add_box(self.manager, "riser", (1, 1, 2), PartType.RISER)
        self.assertIs(self.controller.target(), casting)
        self.selection.part_id = "riser"
        self.assertIs(self.controller.target(), riser)
        self.selection.part_id = "deleted"
        self.assertIs(self.controller.target(), casting)

    def test_is_single_casting_project(self) -> None:
        self.assertFalse(self.controller.is_single_casting_project())
        _add_box(self.manager, "casting", (4, 2, 1))
        self.assertTrue(self.controller.is_single_casting_project())
        _add_box(self.manager, "riser", (1, 1, 2), PartType.RISER)
        self.assertFalse(self.controller.is_single_casting_project())

    def test_rotations_compose_in_world_axes(self) -> None:
        part = _add_box(self.manager, "casting", (4, 2, 1))
        self.controller.rotate("y", 90)
        self.controller.rotate("x", 90)
        expected = orientation.axis_rotation_matrix("x", 90) @ orientation.axis_rotation_matrix("y", 90)
        np.testing.assert_allclose(orientation.euler_to_matrix(part.rotation), expected, atol=1e-9)
        self.assertFalse(
            np.allclose(orientation.euler_to_matrix((90.0, 90.0, 0.0)), expected, atol=1e-9)
        )

    def test_rotate_keeps_center_xy_and_floor(self) -> None:
        part = _add_box(self.manager, "casting", (4, 2, 1), position=(1.0, 1.0, 0.5))
        before = part_world_bounds(part)
        self.controller.rotate("x", 90)
        after = part_world_bounds(part)
        np.testing.assert_allclose(after.mean(axis=0)[:2], before.mean(axis=0)[:2], atol=1e-9)
        self.assertAlmostEqual(after[0, 2], before[0, 2])
        np.testing.assert_allclose(after[1] - after[0], (4.0, 1.0, 2.0), atol=1e-9)

    def test_rotate_without_keep_on_floor_pivots_about_center(self) -> None:
        part = _add_box(self.manager, "casting", (4, 2, 1))
        self.controller.keep_on_floor = False
        before = part_world_bounds(part)
        self.controller.rotate("x", 90)
        after = part_world_bounds(part)
        np.testing.assert_allclose(after.mean(axis=0), before.mean(axis=0), atol=1e-9)
        self.assertAlmostEqual(after[0, 2], -0.5)

    def test_face_down_and_up(self) -> None:
        part = _add_box(self.manager, "casting", (4, 2, 1))
        self.assertTrue(self.controller.face_to((1.0, 0.0, 0.0), "down"))
        bounds = part_world_bounds(part)
        self.assertAlmostEqual(bounds[1, 2] - bounds[0, 2], 4.0)
        self.assertAlmostEqual(bounds[0, 2], 0.0)
        np.testing.assert_allclose(
            orientation.euler_to_matrix(part.rotation) @ (1.0, 0.0, 0.0), (0.0, 0.0, -1.0), atol=1e-9
        )
        self.controller.face_to((0.0, 0.0, -1.0), "up")
        np.testing.assert_allclose(
            orientation.euler_to_matrix(part.rotation) @ (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), atol=1e-9
        )

    def test_reset_and_exact_rotation(self) -> None:
        part = _add_box(self.manager, "casting", (4, 2, 1))
        self.controller.rotate("z", 45)
        self.controller.rotate("x", 15)
        self.assertTrue(self.controller.reset_orientation())
        np.testing.assert_allclose(part.rotation, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(part_world_bounds(part)[0, 2], 0.0)
        self.assertFalse(self.controller.reset_orientation())
        self.assertTrue(self.controller.set_rotation(90.0, 0.0, 0.0))
        np.testing.assert_allclose(part.rotation, (90.0, 0.0, 0.0))
        self.assertAlmostEqual(part_world_bounds(part)[0, 2], 0.0)

    def test_size_readout(self) -> None:
        _add_box(self.manager, "casting", (4, 2, 1))
        self.controller.rotate("x", 90)
        np.testing.assert_allclose(self.controller.size(), (4.0, 1.0, 2.0), atol=1e-9)

    def test_height_moves_and_locked_parting_follows_casting(self) -> None:
        part = _add_box(self.manager, "casting", (4, 2, 1))
        self.controller.set_parting_z(0.5)
        self.assertTrue(self.controller.move_z(0.25))
        self.assertAlmostEqual(self.controller.bottom_z(), 0.25)
        self.assertAlmostEqual(self.controller.parting.z, 0.75)
        self.controller.set_bottom_z(2.0)
        self.assertAlmostEqual(part_world_bounds(part)[0, 2], 2.0)
        self.assertAlmostEqual(self.controller.parting.z, 2.5)
        self.controller.drop_to_floor()
        self.assertAlmostEqual(part_world_bounds(part)[0, 2], 0.0)
        self.assertAlmostEqual(self.controller.parting.z, 0.5)
        self.controller.set_parting_locked(False)
        self.controller.move_z(1.0)
        self.assertAlmostEqual(self.controller.parting.z, 0.5)

    def test_moving_a_riser_does_not_move_locked_parting(self) -> None:
        _add_box(self.manager, "casting", (4, 2, 1))
        _add_box(self.manager, "riser", (1, 1, 2), PartType.RISER)
        self.controller.set_parting_z(0.5)
        self.selection.part_id = "riser"
        self.controller.move_z(1.0)
        self.assertAlmostEqual(self.controller.parting.z, 0.5)

    def test_parting_presets_and_widest_default(self) -> None:
        _flange_casting(self.manager)
        self.assertIsNone(self.controller.parting)
        self.assertTrue(self.controller.set_parting_visible(True))
        self.assertAlmostEqual(self.controller.parting.z, 1.0)
        self.assertTrue(self.controller.parting.visible)
        self.controller.apply_parting_preset("top")
        self.assertAlmostEqual(self.controller.parting.z, 4.0)
        self.controller.apply_parting_preset("bottom")
        self.assertAlmostEqual(self.controller.parting.z, 0.0)
        self.controller.apply_parting_preset("center")
        self.assertAlmostEqual(self.controller.parting.z, 2.0)
        self.controller.nudge_parting(-0.25)
        self.assertAlmostEqual(self.controller.parting.z, 1.75)
        self.controller.set_parting_visible(False)
        self.assertFalse(self.controller.parting.visible)
        self.controller.apply_parting_preset("widest")
        self.assertTrue(self.controller.parting.visible)
        self.assertAlmostEqual(self.controller.parting.z, 1.0)

    def test_cope_drag_readout(self) -> None:
        _add_box(self.manager, "casting", (2, 2, 4))
        self.controller.set_parting_z(1.0)
        split = self.controller.parting_split()
        self.assertAlmostEqual(split["cope_depth"], 3.0)
        self.assertAlmostEqual(split["drag_depth"], 1.0)
        self.assertAlmostEqual(split["cope_fraction"], 0.75, places=2)

    def test_seat_and_hang(self) -> None:
        _add_box(self.manager, "casting", (4, 2, 1))
        riser = _add_box(self.manager, "riser", (1, 1, 2), PartType.RISER, position=(0, 0, 5))
        runner = _add_box(self.manager, "runner", (3, 0.5, 0.5), PartType.RUNNER)
        self.selection.part_id = "riser"
        self.assertFalse(self.controller.seat_target("above"))
        self.controller.set_parting_z(1.0)
        self.assertEqual(self.controller.suggested_seat_side(), "above")
        self.assertTrue(self.controller.seat_target("above"))
        self.assertAlmostEqual(part_world_bounds(riser)[0, 2], 1.0)
        self.selection.part_id = "runner"
        self.assertEqual(self.controller.suggested_seat_side(), "below")
        self.assertTrue(self.controller.seat_target("below"))
        self.assertAlmostEqual(part_world_bounds(runner)[1, 2], 1.0)
        self.selection.part_id = None
        self.assertFalse(self.controller.seat_target("above"))

    def test_default_seat_side(self) -> None:
        for part_type in (PartType.RISER, PartType.SPRUE, PartType.POUR_CUP, PartType.RISER_POST):
            self.assertEqual(default_seat_side(part_type), "above")
        for part_type in (PartType.RUNNER, PartType.GATE):
            self.assertEqual(default_seat_side(part_type), "below")

    def test_section_segments_and_plane_extent(self) -> None:
        _add_box(self.manager, "casting", (4, 2, 1))
        self.assertEqual(self.controller.section_segments().shape, (0, 2, 3))
        self.controller.set_parting_z(0.5)
        segments = self.controller.section_segments()
        self.assertGreater(len(segments), 0)
        np.testing.assert_allclose(segments[:, :, 2], 0.5)
        xmin, xmax, ymin, ymax = self.controller.plane_extent()
        self.assertLess(xmin, 0.0)
        self.assertGreater(xmax, 4.0)
        self.assertLess(ymin, 0.0)
        self.assertGreater(ymax, 2.0)

    def test_snapshot_restore_reverts_transforms_and_parting(self) -> None:
        part = _add_box(self.manager, "casting", (4, 2, 1))
        snap = self.controller.snapshot()
        self.controller.rotate("x", 90)
        self.controller.move_z(2.0)
        self.controller.set_parting_z(3.0)
        self.assertTrue(self.controller.restore(snap))
        np.testing.assert_allclose(part.rotation, (0.0, 0.0, 0.0))
        np.testing.assert_allclose(part.position, (0.0, 0.0, 0.0))
        self.assertIsNone(self.controller.parting)
        self.assertFalse(self.controller.restore(snap))

    def test_outward_face_normal_faces_viewer(self) -> None:
        tri = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        np.testing.assert_allclose(outward_face_normal(tri, (0, 0, 5)), (0, 0, 1))
        np.testing.assert_allclose(outward_face_normal(tri, (0, 0, -5)), (0, 0, -1))
        self.assertIsNone(outward_face_normal([(0, 0, 0), (1, 0, 0), (2, 0, 0)], (0, 0, 1)))

class PartingLinePersistenceTests(unittest.TestCase):
    def _payload(self, ui_state):
        return build_project_state(
            PartManager(),
            alloy_name="WCB",
            voxel_size=0.1,
            refinement_level=3,
            memory_mode="auto",
            analysis_options={},
            performance_options={},
            mesh_setting_origin="manual",
            current_vis_mode="shrinkage",
            ui_state=ui_state,
        ).to_dict()

    def test_round_trip(self) -> None:
        parting = PartingLine(z=1.25, visible=False, locked_to_casting=False)
        loaded = load_project_state(self._payload({"parting_line": parting.to_dict()})).to_dict()
        self.assertEqual(
            loaded["ui_state"]["parting_line"], {"z": 1.25, "visible": False, "locked_to_casting": False}
        )
        self.assertEqual(PartingLine.from_dict(loaded["ui_state"]["parting_line"]), parting)

    def test_missing_key_means_not_set(self) -> None:
        loaded = load_project_state(self._payload({"threshold": 30})).to_dict()
        self.assertNotIn("parting_line", loaded["ui_state"])
        self.assertIsNone(PartingLine.from_dict(loaded["ui_state"].get("parting_line")))

    def test_malformed_parting_line_is_dropped(self) -> None:
        for bad in ({"z": "abc"}, {"z": float("nan")}, {"visible": True}, "1.0", None):
            loaded = load_project_state(self._payload({"parting_line": bad})).to_dict()
            self.assertNotIn("parting_line", loaded["ui_state"], bad)

    def test_defaults_fill_missing_flags(self) -> None:
        self.assertEqual(
            normalize_parting_line({"z": 2}), {"z": 2.0, "visible": True, "locked_to_casting": True}
        )

class OrientationDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.manager = PartManager()
        self.casting = _flange_casting(self.manager)
        self.controller = OrientationController(self.manager)
        self.dialog = OrientationDialog(self.controller)
        self.geometry_events = []
        self.parting_events = []
        self.picks = []
        self.dialog.geometry_changed.connect(lambda: self.geometry_events.append(1))
        self.dialog.parting_changed.connect(lambda: self.parting_events.append(1))
        self.dialog.pick_requested.connect(lambda mode, ids: self.picks.append((mode, ids)))

    def tearDown(self) -> None:
        self.dialog.close()
        self.dialog.deleteLater()

    def test_every_control_has_an_accessible_name(self) -> None:
        from PySide6.QtWidgets import QAbstractButton, QAbstractSpinBox, QComboBox

        controls = self.dialog.findChildren(QAbstractButton) + self.dialog.findChildren(
            QAbstractSpinBox
        ) + self.dialog.findChildren(QComboBox)
        self.assertGreater(len(controls), 20)
        for control in controls:
            self.assertTrue(control.accessibleName() or control.text(), control)

    def test_rotate_button_and_step(self) -> None:
        self.dialog.rotate_buttons["Rotate X +"].click()
        np.testing.assert_allclose(self.casting.rotation, (90.0, 0.0, 0.0))
        self.assertEqual(len(self.geometry_events), 1)
        self.assertIn("6.00 x 4.00 x 6.00", self.dialog.size_label.text())
        self.dialog.rotation_step_combo.setCurrentIndex(1)
        self.dialog.rotate_buttons["Rotate X -"].click()
        np.testing.assert_allclose(self.casting.rotation, (45.0, 0.0, 0.0))
        self.assertIn("45", self.dialog.rotate_buttons["Rotate Z +"].toolTip())

    def test_widest_section_and_readout(self) -> None:
        self.dialog.preset_buttons["widest"].click()
        self.assertAlmostEqual(self.controller.parting.z, 1.0)
        self.assertTrue(self.dialog.show_parting_check.isChecked())
        self.assertAlmostEqual(self.dialog.parting_z_spin.value(), 1.0)
        self.assertIn("Cope depth: 3.00 in", self.dialog.split_label.text())
        self.assertEqual(len(self.parting_events), 1)
        self.dialog.raise_parting_btn.click()
        self.assertAlmostEqual(self.controller.parting.z, 1.25)
        self.dialog.parting_z_spin.setValue(2.0)
        self.assertAlmostEqual(self.controller.parting.z, 2.0)

    def test_height_controls_move_locked_parting(self) -> None:
        self.dialog.preset_buttons["widest"].click()
        self.dialog.raise_part_btn.click()
        self.assertAlmostEqual(self.controller.bottom_z(), 0.25)
        self.assertAlmostEqual(self.controller.parting.z, 1.25)
        self.dialog.bottom_z_spin.setValue(1.0)
        self.assertAlmostEqual(self.controller.bottom_z(), 1.0)
        self.dialog.drop_to_floor_btn.click()
        self.assertAlmostEqual(self.controller.bottom_z(), 0.0)
        self.assertAlmostEqual(self.controller.parting.z, 1.0)

    def test_pick_modes(self) -> None:
        self.dialog.face_down_btn.click()
        self.assertEqual(self.picks, [(PICK_FACE_DOWN, ["casting"])])
        self.assertTrue(self.dialog.face_down_btn.isChecked())
        self.assertFalse(self.dialog.pick_row_widget.isHidden())
        self.dialog.handle_pick(
            {
                "point": (6.0, 3.0, 0.5),
                "triangle": [(6, 0, 0), (6, 6, 0), (6, 0, 1)],
                "toward_viewer": (1.0, 0.0, 0.0),
                "part_id": "casting",
            }
        )
        self.assertIsNone(self.dialog.pick_mode)
        self.assertFalse(self.dialog.face_down_btn.isChecked())
        np.testing.assert_allclose(
            orientation.euler_to_matrix(self.casting.rotation) @ (1.0, 0.0, 0.0), (0.0, 0.0, -1.0), atol=1e-9
        )
        self.dialog.pick_height_btn.click()
        self.assertEqual(self.picks[-1], (PICK_PARTING_HEIGHT, None))
        self.dialog.handle_pick({"point": (0.0, 0.0, 2.5), "triangle": [], "toward_viewer": (0, 0, 1)})
        self.assertAlmostEqual(self.controller.parting.z, 2.5)

    def test_cancel_pick(self) -> None:
        cancelled = []
        self.dialog.pick_cancelled.connect(lambda: cancelled.append(1))
        self.dialog.face_up_btn.click()
        self.dialog.cancel_pick_btn.click()
        self.assertIsNone(self.dialog.pick_mode)
        self.assertFalse(self.dialog.face_up_btn.isChecked())
        self.assertEqual(cancelled, [1])

    def test_revert_changes(self) -> None:
        self.dialog.show()
        self.dialog.rotate_buttons["Rotate Y +"].click()
        self.dialog.preset_buttons["top"].click()
        self.dialog.restore_original_values()
        np.testing.assert_allclose(self.casting.rotation, (0.0, 0.0, 0.0))
        self.assertIsNone(self.controller.parting)

if __name__ == "__main__":
    unittest.main()
