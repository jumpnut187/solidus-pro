from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import trimesh
from flask import Flask

from web.services.simulation_queue import SimulationQueueManager
from web import repository as repo
from web.payloads import build_fork_surface_payload, build_result_fields_payload, build_volume_payload
from web.routes import api_bp


def _synthetic_visualization() -> dict:
    centers = []
    shrinkage = []
    hot_tear = []
    sizes = []
    for x in range(4):
        for y in range(4):
            for z in range(4):
                centers.append([float(x), float(y), float(z)])
                value = 1.0 if (x, y, z) in {(1, 1, 1), (2, 2, 2)} else 0.0
                shrinkage.append(value)
                hot_tear.append(value)
                sizes.append(1.0)
    return {
        "centers": centers,
        "sizes": sizes,
        "shrinkage": shrinkage,
        "hotspot": shrinkage,
        "hot_tear": hot_tear,
        "solidification_times": [float(x + y + z) for x in range(4) for y in range(4) for z in range(4)],
        "fraction_liquid_history": [shrinkage, [v * 0.5 for v in shrinkage]],
        "fraction_liquid_times": [0.0, 10.0],
    }


class WebApiPayloadTests(unittest.TestCase):
    def test_advanced_surface_mode_builds_babylon_geometry(self) -> None:
        payload = build_fork_surface_payload(_synthetic_visualization(), "hot_tear", 0.5, 0)

        self.assertFalse(payload["empty"])
        self.assertEqual(payload["active_count"], 2)
        self.assertGreater(payload["vertex_count"], 0)
        self.assertEqual(len(payload["positions"]) % 3, 0)
        self.assertEqual(len(payload["indices"]) % 3, 0)

    def test_volume_payload_samples_fraction_liquid_history(self) -> None:
        visualization = _synthetic_visualization()
        visualization.pop("solidification_times")
        payload = build_volume_payload(visualization, "fraction_liquid", 100)

        self.assertFalse(payload["empty"])
        self.assertEqual(payload["sample_count"], 64)
        self.assertAlmostEqual(max(payload["values"]), 0.5)

    def test_rejects_unknown_surface_mode(self) -> None:
        with self.assertRaises(ValueError):
            build_fork_surface_payload(_synthetic_visualization(), "unknown", 0.5, 0)

    def test_result_fields_payload_preloads_modes_and_solidification(self) -> None:
        payload = build_result_fields_payload(_synthetic_visualization(), max_points=1000)

        self.assertFalse(payload["empty"])
        self.assertEqual(payload["sample_count"], 64)
        self.assertIn("hotspot", payload["fields"])
        self.assertIn("hot_tear", payload["fields"])
        self.assertIn("fraction_liquid", payload["modes"])
        self.assertEqual(len(payload["centers"]), 64 * 3)
        self.assertEqual(len(payload["solidification"]["normalized_times"]), 64)
        self.assertEqual(len(payload["fraction_liquid"]["frames"]), 2)
        self.assertEqual(len(payload["fraction_liquid"]["frames"][0]), 64)


class WebApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.database_path = root / "web.db"
        self.upload_folder = root / "uploads"
        self.temp_folder = root / "temp"
        self.upload_folder.mkdir()
        self.temp_folder.mkdir()
        repo.init_db(self.database_path)

        app = Flask(__name__)
        app.config["DATABASE"] = self.database_path
        app.config["UPLOAD_FOLDER"] = self.upload_folder
        app.config["TEMP_FOLDER"] = self.temp_folder
        app.extensions["simulation_queue_manager"] = SimulationQueueManager()
        app.register_blueprint(api_bp)
        self.client = app.test_client()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_project_and_part_crud_preserves_riser_open_top(self) -> None:
        create_response = self.client.post(
            "/api/projects",
            json={"name": "Web Test", "alloy": "WCB", "voxel_size": 0.2, "refinement_level": 2},
        )
        self.assertEqual(create_response.status_code, 201)
        project_id = create_response.get_json()["id"]

        part_response = self.client.post(
            f"/api/projects/{project_id}/parts",
            json={
                "name": "Riser",
                "part_type": "riser",
                "filepath": "riser.stl",
                "is_open_top": True,
                "visible": True,
            },
        )
        self.assertEqual(part_response.status_code, 201)
        part_payload = part_response.get_json()["part"]
        self.assertTrue(part_payload["is_open_top"])
        self.assertEqual(part_payload["part_type"], "riser")

    def test_upload_endpoint_returns_mesh_data(self) -> None:
        mesh_path = Path(self.temp_dir.name) / "casting.stl"
        trimesh.creation.box(extents=(1.0, 2.0, 3.0)).export(mesh_path)

        with mesh_path.open("rb") as handle:
            response = self.client.post(
                "/api/uploads",
                data={"files": (io.BytesIO(handle.read()), "casting.stl")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(len(payload["files"]), 1)
        uploaded = payload["files"][0]
        self.assertEqual(uploaded["part_type"], "cast")
        self.assertGreater(uploaded["mesh_data"]["face_count"], 0)

    def test_results_and_surface_read_saved_artifact(self) -> None:
        project_id = repo.create_project(self.database_path, {"name": "Artifact Test"})
        artifact_path = self.temp_folder / "artifacts" / f"{project_id}.json"
        artifact_path.parent.mkdir(parents=True)
        artifact = {
            "summary": {"voxel_count": 64, "total_volume": 64.0, "available_modes": ["hot_tear"]},
            "visualization": _synthetic_visualization(),
        }
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        repo.mark_project_completed(self.database_path, project_id, artifact["summary"], str(artifact_path))

        result_response = self.client.get(f"/api/projects/{project_id}/results")
        self.assertEqual(result_response.status_code, 200)
        self.assertEqual(result_response.get_json()["data"]["voxel_count"], 64)

        summary_response = self.client.get(f"/api/projects/{project_id}/results?summary_only=1")
        self.assertEqual(summary_response.status_code, 200)
        self.assertNotIn("visualization", summary_response.get_json())

        surface_response = self.client.get(f"/api/projects/{project_id}/surface?mode=hot_tear&threshold=0.5")
        self.assertEqual(surface_response.status_code, 200)
        self.assertFalse(surface_response.get_json()["empty"])

        fields_response = self.client.get(f"/api/projects/{project_id}/result-fields")
        self.assertEqual(fields_response.status_code, 200)
        fields_payload = fields_response.get_json()
        self.assertEqual(fields_payload["sample_count"], 64)
        self.assertIn("hot_tear", fields_payload["fields"])


if __name__ == "__main__":
    unittest.main()