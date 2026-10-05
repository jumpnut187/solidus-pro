"""Tests for the routes that store bodies modeled in the browser."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import trimesh
from flask import Flask

from web import repository as repo
from web.routes import api_bp
from web.routes_modeling import modeling_bp
from web.routes_tools import tools_bp
from web.services.simulation_queue import SimulationQueueManager


def _box_mesh(size: float = 2.0, offset: float = 0.0) -> dict:
    box = trimesh.creation.box(extents=(size, size, size))
    box.apply_translation([offset, 0.0, 0.0])
    return {
        "vertices": box.vertices.reshape(-1).tolist(),
        "faces": box.faces.reshape(-1).tolist(),
    }


def _feature(**overrides) -> dict:
    feature = {
        "version": 1,
        "kind": "extrude",
        "seq": 1,
        "sketch": {
            "plane": {"kind": "standard", "id": "top", "offset": 0.0},
            "entities": [{"id": "e1", "type": "circle", "cx": 0.0, "cy": 0.0, "d": 2.0}],
        },
        "endCondition": "blind",
        "depth": 2.0,
        "reverse": False,
        "draftDeg": 0.0,
        "draftOutward": False,
        "trimOverlap": True,
    }
    feature.update(overrides)
    return feature


class WebModelingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.database_path = root / "test.db"
        self.upload_folder = root / "uploads"
        self.upload_folder.mkdir()
        repo.init_db(self.database_path)

        app = Flask(__name__)
        app.config["DATABASE"] = self.database_path
        app.config["UPLOAD_FOLDER"] = self.upload_folder
        app.config["TEMP_FOLDER"] = root / "temp"
        app.extensions["simulation_queue_manager"] = SimulationQueueManager()
        app.register_blueprint(api_bp)
        app.register_blueprint(tools_bp)
        app.register_blueprint(modeling_bp)
        self.client = app.test_client()
        response = self.client.post("/api/projects", json={"name": "Modeling"})
        self.project_id = str(response.get_json()["id"])

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create(self, **overrides) -> dict:
        payload = {
            "name": "Riser 1",
            "part_type": "riser",
            "is_open_top": True,
            "feature": _feature(),
            "mesh": _box_mesh(),
        }
        payload.update(overrides)
        return self.client.post(f"/api/projects/{self.project_id}/features", json=payload)

    def test_create_stores_stl_feature_and_body_fields(self) -> None:
        response = self._create()
        self.assertEqual(response.status_code, 201, response.get_json())
        part = response.get_json()["part"]
        self.assertEqual(part["part_type"], "riser")
        self.assertTrue(part["is_open_top"])
        self.assertEqual(part["feature"]["endCondition"], "blind")
        self.assertNotIn("feature_json", part)
        self.assertTrue(part["watertight"])
        self.assertAlmostEqual(part["mesh_data"]["volume"], 8.0, places=4)

        stl = Path(part["filepath"])
        self.assertTrue(stl.is_file())
        self.assertEqual(stl.parent.resolve(), self.upload_folder.resolve())
        self.assertAlmostEqual(trimesh.load(str(stl), force="mesh").volume, 8.0, places=4)

        listed = self.client.get(f"/api/projects/{self.project_id}").get_json()["parts"]
        self.assertEqual(listed[0]["feature"]["sketch"]["entities"][0]["type"], "circle")

    def test_open_top_is_dropped_for_a_chill(self) -> None:
        part = self._create(part_type="chill").get_json()["part"]
        self.assertFalse(part["is_open_top"])

    def test_update_replaces_the_file_and_the_feature(self) -> None:
        created = self._create().get_json()
        part_id = created["id"]
        old_path = Path(created["part"]["filepath"])
        response = self.client.put(
            f"/api/projects/{self.project_id}/features/{part_id}",
            json={
                "name": "Chill 1",
                "part_type": "chill",
                "feature": _feature(depth=4.0),
                "mesh": _box_mesh(size=3.0),
            },
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        part = response.get_json()["part"]
        self.assertEqual(part["name"], "Chill 1")
        self.assertEqual(part["part_type"], "chill")
        self.assertEqual(part["feature"]["depth"], 4.0)
        self.assertNotEqual(Path(part["filepath"]), old_path)
        self.assertFalse(old_path.exists())
        self.assertAlmostEqual(part["mesh_data"]["volume"], 27.0, places=4)

    def test_update_keeps_a_file_another_part_still_uses(self) -> None:
        created = self._create().get_json()
        old_path = created["part"]["filepath"]
        # The browser's Duplicate reuses the source part's file.
        self.client.post(
            f"/api/projects/{self.project_id}/parts",
            json={"name": "copy", "part_type": "riser", "filepath": old_path},
        )
        self.client.put(
            f"/api/projects/{self.project_id}/features/{created['id']}",
            json={"name": "Riser 1", "part_type": "riser", "feature": _feature(), "mesh": _box_mesh(3.0)},
        )
        self.assertTrue(Path(old_path).exists())

    def test_generic_part_write_cannot_change_the_feature(self) -> None:
        created = self._create().get_json()
        part = dict(created["part"])
        part["feature"] = {"kind": "extrude", "depth": 99}
        part.pop("mesh_data", None)
        self.client.put(f"/api/projects/{self.project_id}/parts/{created['id']}", json=part)
        stored = repo.get_part(self.database_path, self.project_id, created["id"])
        assert stored is not None
        self.assertEqual(stored["feature"]["depth"], 2.0)

    def test_rejects_bad_input(self) -> None:
        bad_cases = [
            {"feature": None},
            {"feature": {"kind": "loft"}},
            {"part_type": "unobtainium"},
            {"mesh": {"vertices": [0, 0, 0, 1, 0, 0, 0, 1, 0], "faces": [0, 1, 5]}},
            {"mesh": {"vertices": [0, 0, float("nan")], "faces": [0, 0, 0]}},
            {"mesh": {"vertices": [0, 0, 0, 1], "faces": [0, 1, 2]}},
            {"mesh": {"vertices": [], "faces": []}},
        ]
        for overrides in bad_cases:
            with self.subTest(overrides=overrides):
                response = self._create(**overrides)
                self.assertEqual(response.status_code, 400)
        self.assertEqual(list(self.upload_folder.iterdir()), [])

    def test_unknown_project_and_part(self) -> None:
        response = self.client.post("/api/projects/nope/features", json={})
        self.assertEqual(response.status_code, 404)
        response = self.client.put(f"/api/projects/{self.project_id}/features/nope", json={})
        self.assertEqual(response.status_code, 404)

    def test_duplicate_project_remaps_the_target_body(self) -> None:
        cast_id = self._create(name="Base", part_type="cast", feature=_feature(seq=1)).get_json()["id"]
        self._create(
            name="Riser",
            feature=_feature(seq=2, endCondition="upToBody", targetPartId=cast_id),
        )
        response = self.client.post(f"/api/projects/{self.project_id}/duplicate", json={"name": "Copy"})
        self.assertEqual(response.status_code, 201)
        copy_id = response.get_json()["id"]
        copied = {part["name"]: part for part in repo.get_parts(self.database_path, copy_id)}
        self.assertEqual(copied["Riser"]["feature"]["targetPartId"], copied["Base"]["id"])
        self.assertNotEqual(copied["Base"]["id"], cast_id)

    def test_export_carries_the_feature(self) -> None:
        self._create()
        response = self.client.get(f"/api/projects/{self.project_id}/export")
        self.assertEqual(response.status_code, 200)
        parts = response.get_json()["parts"]
        self.assertEqual(parts[0]["feature"]["kind"], "extrude")

    def test_reference_planes_round_trip_and_survive_a_settings_save(self) -> None:
        planes = [
            {
                "id": "p1",
                "name": "Plane1",
                "source": {
                    "kind": "offset",
                    "base": {"kind": "standard", "id": "top", "offset": 0},
                    "distance": 2.5,
                    "flip": False,
                },
            }
        ]
        url = f"/api/projects/{self.project_id}/reference-planes"
        response = self.client.put(url, json={"planes": planes})
        self.assertEqual(response.status_code, 200)
        # The shell's autosave sends a ui_state without the planes.
        self.client.put(f"/api/projects/{self.project_id}", json={"name": "Renamed", "ui_state": {}})
        project = self.client.get(f"/api/projects/{self.project_id}").get_json()["project"]
        self.assertEqual(project["ui_state"]["reference_planes"], planes)
        self.assertEqual(project["name"], "Renamed")

    def test_reference_planes_reject_bad_input(self) -> None:
        url = f"/api/projects/{self.project_id}/reference-planes"
        self.assertEqual(self.client.put(url, json={"planes": "x"}).status_code, 400)
        self.assertEqual(self.client.put(url, json={"planes": [{"id": 1}]}).status_code, 400)
        self.assertEqual(self.client.put("/api/projects/nope/reference-planes", json={"planes": []}).status_code, 404)

    def test_old_database_gains_the_feature_column(self) -> None:
        legacy = Path(self.temp_dir.name) / "legacy.db"
        with sqlite3.connect(legacy) as db:
            db.execute(
                "CREATE TABLE parts (id TEXT PRIMARY KEY, project_id TEXT NOT NULL, name TEXT NOT NULL, "
                "part_type TEXT NOT NULL, filepath TEXT NOT NULL, position_x REAL DEFAULT 0, "
                "position_y REAL DEFAULT 0, position_z REAL DEFAULT 0, rotation_x REAL DEFAULT 0, "
                "rotation_y REAL DEFAULT 0, rotation_z REAL DEFAULT 0, scale REAL DEFAULT 1, "
                "visible INTEGER DEFAULT 1, is_open_top INTEGER DEFAULT 0, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            db.execute(
                "INSERT INTO parts (id, project_id, name, part_type, filepath) VALUES ('p1', 'x', 'a', 'cast', 'f')"
            )
        repo.init_db(legacy)
        part = repo.get_part(legacy, "x", "p1")
        assert part is not None
        self.assertIsNone(part["feature"])


if __name__ == "__main__":
    unittest.main()
