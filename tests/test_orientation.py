import unittest

import numpy as np
import trimesh

from core import orientation as ori
from core.part_manager import Part, PartType

def _box(extents=(4.0, 2.0, 1.0)):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(np.array(extents) / 2.0)
    return mesh

class EulerRoundTrip(unittest.TestCase):
    def test_round_trip(self):
        for rot in [(0, 0, 0), (90, 0, 0), (30, -45, 120), (0, 90, 0), (-90, 0, 180)]:
            m = ori.euler_to_matrix(rot)
            back = ori.matrix_to_euler(m)
            np.testing.assert_allclose(ori.euler_to_matrix(back), m, atol=1e-9)

    def test_matches_part_manager(self):
        mesh = _box()
        part = Part(id="p", name="p", filepath="", part_type=PartType.CAST, trimesh_obj=mesh)
        part.set_rotation(30, -20, 75)
        part.set_position(1, 2, 3)
        expected = part.get_transformed_mesh().bounds
        got = ori.world_bounds(mesh, (30, -20, 75), (1, 2, 3))
        np.testing.assert_allclose(got, expected, atol=1e-9)

class WorldAxisRotation(unittest.TestCase):
    def test_compound_turns_compose_in_world(self):
        rot, pos = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        rot, pos = ori.rotate_about_world_axis(rot, pos, "x", 90, (0, 0, 0))
        rot, pos = ori.rotate_about_world_axis(rot, pos, "z", 90, (0, 0, 0))
        expected = ori.axis_rotation_matrix("z", 90) @ ori.axis_rotation_matrix("x", 90)
        np.testing.assert_allclose(ori.euler_to_matrix(rot), expected, atol=1e-9)

    def test_turn_about_center_keeps_center(self):
        mesh = _box()
        rot, pos = (0.0, 0.0, 0.0), (5.0, 5.0, 0.0)
        b0 = ori.world_bounds(mesh, rot, pos)
        center = b0.mean(axis=0)
        rot, pos = ori.rotate_about_world_axis(rot, pos, "y", 90, center)
        b1 = ori.world_bounds(mesh, rot, pos)
        np.testing.assert_allclose(b1.mean(axis=0), center, atol=1e-9)
        np.testing.assert_allclose(b1[1] - b1[0], [1, 2, 4], atol=1e-9)

    def test_angles_are_clean(self):
        rot, _ = ori.rotate_about_world_axis((0, 0, 0), (0, 0, 0), "x", 90, (0, 0, 0))
        self.assertEqual(rot, (90.0, 0.0, 0.0))

class FaceDown(unittest.TestCase):
    def test_face_down(self):
        mesh = _box()
        rot, pos = ori.face_to_direction((0, 0, 0), (0, 0, 0), (1, 0, 0), (2, 1, 0.5))
        b = ori.world_bounds(mesh, rot, pos)
        np.testing.assert_allclose(b[1] - b[0], [1, 2, 4], atol=1e-9)

    def test_opposite_normal(self):
        m = ori.align_vector_matrix((0, 0, 1), (0, 0, -1))
        np.testing.assert_allclose(m @ [0, 0, 1], [0, 0, -1], atol=1e-9)

class PartingLine(unittest.TestCase):
    def test_snap(self):
        b = np.array([[0, 0, 2.0], [1, 1, 5.0]])
        self.assertAlmostEqual(ori.snap_to_parting_dz(b, 3.0, "above"), 1.0)
        self.assertAlmostEqual(ori.snap_to_parting_dz(b, 3.0, "below"), -2.0)
        self.assertAlmostEqual(ori.drop_to_floor_dz(b), -2.0)

    def test_suggest_on_flanged_part(self):
        flange = _box((6, 6, 1))
        boss = _box((2, 2, 3))
        boss.apply_translation([2, 2, 1])
        mesh = trimesh.util.concatenate([flange, boss])
        s = ori.suggest_parting_z(mesh, (0, 0, 0), (0, 0, 0))
        self.assertAlmostEqual(s["max_section"], 1.0, places=6)
        self.assertAlmostEqual(s["bottom"], 0.0)
        self.assertAlmostEqual(s["top"], 4.0)

    def test_cope_drag_split(self):
        mesh = _box((2, 2, 4))
        split = ori.cope_drag_split(mesh, (0, 0, 0), (0, 0, 0), 1.0)
        self.assertAlmostEqual(split["cope_depth"], 3.0)
        self.assertAlmostEqual(split["drag_depth"], 1.0)
        self.assertAlmostEqual(split["cope_fraction"], 0.75, places=1)

    def test_snap_angle(self):
        self.assertEqual(ori.snap_angle(44, 15), 45.0)
        self.assertEqual(ori.snap_angle(181, 90), 180.0)

if __name__ == "__main__":
    unittest.main()
