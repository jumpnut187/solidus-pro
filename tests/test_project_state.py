import os
import tempfile
import unittest

import trimesh

from core.part_manager import PartManager, PartType
from core.pour_location import PourLocation
from ui.project_state import (
    PROJECT_SCHEMA_VERSION,
    build_project_state,
    load_project_state,
)


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
                performance_options={"chunked_voxelization": True},
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

    def test_round_trip_preserves_pour_location(self) -> None:
        manager = PartManager()
        location = PourLocation(point_in=(1.5, -2.5, 3.0), normal=(0.0, 0.0, 1.0), part_id="cast-1")

        payload = build_project_state(
            manager,
            alloy_name="WCB",
            voxel_size=0.1,
            refinement_level=3,
            memory_mode="auto",
            analysis_options={"filling": True},
            performance_options={},
            pour_location=location.to_dict(),
            mesh_setting_origin="manual",
            current_vis_mode="shrinkage",
            ui_state={},
        ).to_dict()

        self.assertEqual(payload["simulation"]["pour_location"], location.to_dict())

        loaded = load_project_state(payload).to_dict()
        restored = PourLocation.from_dict(loaded["simulation"]["pour_location"])
        self.assertEqual(restored.point_in, location.point_in)
        self.assertEqual(restored.normal, location.normal)
        self.assertEqual(restored.part_id, location.part_id)

    def test_round_trip_with_no_pour_location_stores_none(self) -> None:
        manager = PartManager()

        payload = build_project_state(
            manager,
            alloy_name="WCB",
            voxel_size=0.1,
            refinement_level=3,
            memory_mode="auto",
            analysis_options={},
            performance_options={},
            mesh_setting_origin="manual",
            current_vis_mode="shrinkage",
            ui_state={},
        ).to_dict()

        self.assertIsNone(payload["simulation"]["pour_location"])

        loaded = load_project_state(payload).to_dict()
        self.assertIsNone(loaded["simulation"]["pour_location"])

    def test_load_defaults_pour_location_to_none_for_older_project_files(self) -> None:
        """A project saved before pour locations existed has no such key."""
        loaded = load_project_state(
            {
                "schema_version": 1,
                "simulation": {},
                "parts": [],
            }
        ).to_dict()

        self.assertIsNone(loaded["simulation"]["pour_location"])

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


class LegacyPourRateMigrationTests(unittest.TestCase):
    """
    A stored pour rate of exactly 5 lb/s is the old fixed default.

    It is right only near a 25 lb casting -- the rate follows sqrt(W) -- so an
    untouched project carrying it pours at a rate that was chosen for no
    casting in particular. 0 means "derive from the square-root rule", which
    is what those projects should have had all along.

    The migration is gated on the schema version rather than run on every
    load, because 5.0 typed deliberately looks identical to 5.0 inherited.
    Saving stamps the current version, so a re-set 5.0 survives; that is the
    half of this behaviour the version bump exists for.
    """

    def _loaded_rate(self, rate, *, version=1):
        payload = {
            "schema_version": version,
            "simulation": {
                "solidification_params": {"filling": {"pour_rate_lb_s": rate}}
            },
            "parts": [],
        }
        loaded = load_project_state(payload).to_dict()
        return loaded["simulation"]["solidification_params"]["filling"][
            "pour_rate_lb_s"
        ]

    def test_the_old_default_becomes_derive(self) -> None:
        self.assertEqual(self._loaded_rate(5.0), 0.0)

    def test_a_rate_just_below_is_left_alone(self) -> None:
        """Only the exact old default is suspect; 4.9 was typed by someone."""
        self.assertEqual(self._loaded_rate(4.9), 4.9)

    def test_a_rate_just_above_is_left_alone(self) -> None:
        self.assertEqual(self._loaded_rate(5.1), 5.1)

    def test_derive_stays_derive(self) -> None:
        self.assertEqual(self._loaded_rate(0.0), 0.0)

    def test_a_current_project_keeps_a_deliberate_five(self) -> None:
        """The whole point of the version gate: 5 lb/s must be settable."""
        self.assertEqual(self._loaded_rate(5.0, version=PROJECT_SCHEMA_VERSION), 5.0)

    def test_a_project_with_no_filling_block_loads(self) -> None:
        loaded = load_project_state(
            {"schema_version": 1, "simulation": {}, "parts": []}
        ).to_dict()
        self.assertEqual(loaded["simulation"]["solidification_params"], {})

    def test_the_stored_payload_is_not_mutated(self) -> None:
        """Migrating must not reach back into the caller's parsed file."""
        payload = {
            "schema_version": 1,
            "simulation": {
                "solidification_params": {"filling": {"pour_rate_lb_s": 5.0}}
            },
            "parts": [],
        }
        load_project_state(payload)
        self.assertEqual(
            payload["simulation"]["solidification_params"]["filling"][
                "pour_rate_lb_s"
            ],
            5.0,
        )

    def test_an_older_project_is_reported_at_the_current_version(self) -> None:
        """It has been migrated forward, so it must not still claim to be v1."""
        loaded = load_project_state(
            {"schema_version": 1, "simulation": {}, "parts": []}
        ).to_dict()
        self.assertEqual(loaded["schema_version"], PROJECT_SCHEMA_VERSION)

    def test_an_unknown_version_is_still_refused(self) -> None:
        with self.assertRaises(ValueError):
            load_project_state(
                {"schema_version": 99, "simulation": {}, "parts": []}
            )


if __name__ == "__main__":
    unittest.main()
