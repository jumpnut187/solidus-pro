from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from flask import Flask

from core.surface_extraction import build_surface_payload, normalize_surface_request
from web import repository as repo
from web.routes import api_bp
from web.services.simulation_queue import SimulationQueueManager


def _synthetic_visualization() -> dict:
    centers = []
    shrinkage = []
    fraction_history = []
    frame_a = []
    frame_b = []

    for x in range(7):
        for y in range(7):
            for z in range(7):
                centers.append([float(x), float(y), float(z)])
                distance = np.linalg.norm(np.array([x, y, z], dtype=np.float32) - 3.0)
                value = max(0.0, 1.0 - float(distance) / 3.3)
                shrinkage.append(value)
                frame_a.append(value)
                frame_b.append(value * 0.5)

    fraction_history.extend([frame_a, frame_b])
    return {
        "centers": centers,
        "sizes": [1.0] * len(centers),
        "shrinkage": shrinkage,
        "hotspot": shrinkage,
        "niyama": shrinkage,
        "fraction_liquid_history": fraction_history,
        "fraction_liquid_times": [0.0, 10.0],
    }


def _shell_visualization(*, include_surface_mask: bool = True, interior_value: float = 1.0) -> dict:
    centers = []
    shrinkage = []
    fraction = []
    surface_voxels = []

    for x in range(5):
        for y in range(5):
            for z in range(5):
                is_surface = x in (0, 4) or y in (0, 4) or z in (0, 4)
                centers.append([float(x), float(y), float(z)])
                surface_voxels.append(is_surface)
                shrinkage.append(1.0)
                fraction.append(1.0 if is_surface else interior_value)

    payload = {
        "centers": centers,
        "sizes": [1.0] * len(centers),
        "shrinkage": shrinkage,
        "fraction_liquid_history": [fraction],
        "fraction_liquid_times": [0.0],
    }
    if include_surface_mask:
        payload["surface_voxels"] = surface_voxels
    return payload


class SurfaceExtractionTests(unittest.TestCase):
    def test_extracts_non_empty_surface_from_scalar_field(self) -> None:
        request = normalize_surface_request("shrinkage", 0.35, 0)
        payload = build_surface_payload(_synthetic_visualization(), request)

        self.assertFalse(payload["empty"])
        self.assertGreater(payload["vertex_count"], 0)
        self.assertGreater(payload["triangle_count"], 0)
        self.assertEqual(len(payload["positions"]) % 3, 0)
        self.assertEqual(len(payload["indices"]) % 3, 0)

    def test_fraction_liquid_history_time_changes_active_count(self) -> None:
        early = build_surface_payload(
            _synthetic_visualization(),
            normalize_surface_request("fraction_liquid", 0.35, 0),
        )
        late = build_surface_payload(
            _synthetic_visualization(),
            normalize_surface_request("fraction_liquid", 0.35, 100),
        )

        self.assertGreater(early["active_count"], late["active_count"])

    def test_fraction_liquid_excludes_explicit_surface_shell_voxels(self) -> None:
        payload = build_surface_payload(
            _shell_visualization(),
            normalize_surface_request("fraction_liquid", 0.5, 0),
        )

        self.assertFalse(payload["empty"])
        self.assertEqual(payload["active_count"], 1)
        self.assertEqual(payload["source_count"], 1)

    def test_fraction_liquid_returns_empty_when_only_shell_is_liquid(self) -> None:
        payload = build_surface_payload(
            _shell_visualization(interior_value=0.0),
            normalize_surface_request("fraction_liquid", 0.5, 0),
        )

        self.assertTrue(payload["empty"])
        self.assertEqual(payload["active_count"], 0)

    def test_fraction_liquid_falls_back_to_one_layer_interior_for_late_core(self) -> None:
        vis = _shell_visualization(interior_value=0.0)
        centers = vis["centers"]
        fraction = [0.0] * len(centers)
        for index, center in enumerate(centers):
            if center == [2.0, 2.0, 2.0]:
                fraction[index] = 1.0
        vis["fraction_liquid_history"] = [fraction]

        payload = build_surface_payload(
            vis,
            normalize_surface_request("fraction_liquid", 0.5, 0),
        )

        self.assertFalse(payload["empty"])
        self.assertEqual(payload["active_count"], 1)

    def test_fraction_liquid_infers_boundary_shell_for_old_artifacts(self) -> None:
        payload = build_surface_payload(
            _shell_visualization(include_surface_mask=False),
            normalize_surface_request("fraction_liquid", 0.5, 0),
        )

        self.assertFalse(payload["empty"])
        self.assertEqual(payload["active_count"], 1)
        self.assertEqual(payload["source_count"], 1)

    def test_shrinkage_keeps_surface_shell_voxels(self) -> None:
        payload = build_surface_payload(
            _shell_visualization(),
            normalize_surface_request("shrinkage", 0.5, 0),
        )

        self.assertFalse(payload["empty"])
        self.assertEqual(payload["active_count"], 125)


class SurfaceApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.database_path = root / "simulations.db"
        self.temp_folder = root / "temp"
        self.upload_folder = root / "uploads"
        self.temp_folder.mkdir()
        self.upload_folder.mkdir()
        repo.init_db(self.database_path)

        app = Flask(__name__)
        app.config["DATABASE"] = self.database_path
        app.config["TEMP_FOLDER"] = self.temp_folder
        app.config["UPLOAD_FOLDER"] = self.upload_folder
        app.extensions["simulation_queue_manager"] = SimulationQueueManager()
        app.register_blueprint(api_bp)
        self.client = app.test_client()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_completed_project(self, name: str = "Surface API") -> str:
        """Create a project with a saved visualization artifact and return its id.

        The surface endpoint loads the artifact before validating query
        parameters, so any test that needs to reach parameter validation must
        go through a project that actually has results.
        """
        sim_id = repo.create_project(
            self.database_path,
            {
                "name": name,
                "alloy": "WCB",
                "voxel_size": 1.0,
                "refinement_level": 1,
            },
        )
        summary = {"voxel_count": 125, "total_volume": 125.0}
        artifact_path = Path(self.temp_dir.name) / f"surface-artifact-{sim_id}.json"
        artifact_path.write_text(
            json.dumps({"data": summary, "visualization": _synthetic_visualization()}),
            encoding="utf-8",
        )
        repo.mark_project_completed(
            self.database_path, sim_id, summary, str(artifact_path)
        )
        return sim_id

    def test_surface_endpoint_reads_saved_visualization_artifact(self) -> None:
        sim_id = self._create_completed_project()

        response = self.client.get(f"/api/projects/{sim_id}/surface?mode=shrinkage&threshold=0.35")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIsNotNone(payload)
        self.assertFalse(payload["empty"])

    def test_surface_endpoint_rejects_invalid_mode(self) -> None:
        sim_id = self._create_completed_project("Invalid mode")

        response = self.client.get(f"/api/projects/{sim_id}/surface?mode=unknown")

        self.assertEqual(response.status_code, 400)

    def test_surface_endpoint_returns_404_for_unknown_project(self) -> None:
        response = self.client.get("/api/projects/example/surface?mode=shrinkage")

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
