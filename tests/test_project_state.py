import os
import tempfile
import unittest

import trimesh

from core.part_manager import PartManager, PartType
from ui.project_state import build_project_state, load_project_state


class ProjectStateTests(unittest.TestCase):
    def test_round_trip_preserves_part_transforms_and_settings(self) -> None:
        manager = PartManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            mesh_path = os.path.join(temp_dir, "casting.stl")
            trimesh.creation.box(extents=(1.0, 2.0, 3.0)).export(mesh_path)

            part_id = manager.load_part(mesh_path, name="Casting")
            self.assertIsNotNone(part_id)

            part = manager.get_part(part_id)
            self.assertIsNotNone(part)
            assert part is not None

            part.set_part_type(PartType.RISER)
            part.set_position(1.25, -2.5, 3.75)
            part.set_rotation(15.0, 25.0, 35.0)
            part.set_scale(1.1)
            part.visible = False
            part.is_open_top = True
            part.color = (0.1, 0.2, 0.3)

            payload = build_project_state(
                manager,
                alloy_name="WCB",
                voxel_size=0.15,
                refinement_level=4,
                memory_mode="disk",
                analysis_options={"shrinkage": True, "microstructure": True, "sand_restraint_index": 0.75},
                performance_options={"use_gpu": False, "chunked_voxelization": True},
                mesh_setting_origin="ml",
                current_vis_mode="hotspot",
                ui_state={"result_type": "Hot Spots", "threshold": 12},
            ).to_dict()

            loaded = load_project_state(payload).to_dict()
            self.assertEqual(loaded["simulation"]["alloy_name"], "WCB")
            self.assertEqual(loaded["simulation"]["memory_mode"], "disk")
            self.assertEqual(loaded["simulation"]["mesh_setting_origin"], "ml")
            self.assertEqual(loaded["simulation"]["current_vis_mode"], "hotspot")
            self.assertEqual(loaded["ui_state"]["result_type"], "Hot Spots")
            self.assertEqual(loaded["simulation"]["analysis_options"]["sand_restraint_index"], 0.75)

            self.assertEqual(len(loaded["parts"]), 1)
            loaded_part = loaded["parts"][0]
            self.assertEqual(loaded_part["name"], "Casting")
            self.assertEqual(loaded_part["part_type"], "riser")
            self.assertEqual(loaded_part["position"], [1.25, -2.5, 3.75])
            self.assertEqual(loaded_part["rotation"], [15.0, 25.0, 35.0])
            self.assertEqual(loaded_part["scale"], 1.1)
            self.assertEqual(loaded_part["color"], [0.1, 0.2, 0.3])
            self.assertFalse(loaded_part["visible"])
            self.assertTrue(loaded_part["is_open_top"])

    def test_load_normalizes_partial_vectors(self) -> None:
        loaded = load_project_state(
            {
                "schema_version": 1,
                "simulation": {},
                "parts": [
                    {
                        "filepath": "part.stl",
                        "position": [1.0],
                        "rotation": [2.0, 3.0],
                        "color": [0.8],
                    }
                ],
            }
        ).to_dict()

        loaded_part = loaded["parts"][0]
        self.assertEqual(loaded_part["position"], [1.0, 0.0, 0.0])
        self.assertEqual(loaded_part["rotation"], [2.0, 3.0, 0.0])
        self.assertEqual(loaded_part["color"], [0.8, 0.7, 0.7])


if __name__ == "__main__":
    unittest.main()
