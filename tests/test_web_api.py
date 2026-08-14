from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import trimesh
from flask import Flask

from core.simulation_pipeline import build_visualization_payload
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

    def test_synthetic_fraction_liquid_only_when_history_empty(self) -> None:
        """Synthetic fraction_liquid front is only returned when no history exists.

        When fraction_liquid_history is present, the actual interpolated frame
        from history is returned instead of the synthetic smooth front.
        """
        from web.payloads import _solidification_path_values

        # Case 1: Both history and solidification_times present - synthetic NOT returned.
        vis_with_history = {
            "centers": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            "sizes": [1.0, 1.0, 1.0],
            "solidification_times": [0.0, 50.0, 100.0],
            "fraction_liquid_history": [
                [1.0, 1.0, 1.0],  # fully liquid at t=0
                [0.5, 0.5, 0.5],  # half solid at t=mid
                [0.0, 0.0, 0.0],  # fully solid at t=end
            ],
        }
        result_with_history = _solidification_path_values(vis_with_history, 50.0)
        # Should return None because history is non-empty.
        self.assertIsNone(
            result_with_history,
            "Synthetic field should be None when history is present",
        )

        # Case 2: Only solidification_times (no history) - synthetic IS returned.
        vis_no_history = {
            "centers": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            "sizes": [1.0, 1.0, 1.0],
            "solidification_times": [0.0, 50.0, 100.0],
            "fraction_liquid_history": [],
        }
        result_no_history = _solidification_path_values(vis_no_history, 50.0)
        self.assertIsNotNone(
            result_no_history,
            "Synthetic field should be returned when history is empty",
        )
        # At 50% progress, the front is at the middle (normalized=0.5).
        # The synthetic front is a smooth gradient, so values should vary.
        self.assertEqual(len(result_no_history), 3)

    def test_shrinkage_mode_uses_combined_adjusted_risk(self) -> None:
        shrinkage = np.array([0.1, 0.8, 0.0], dtype=np.float32)
        porosity = np.array([0.7, 0.2, 0.4], dtype=np.float32)
        engine = SimpleNamespace(
            shrinkage_probability=shrinkage,
            porosity_probability=porosity,
            shrink_scalar=2.0,
            shrink_scalar_formula="adjusted_risk = clamp(raw_risk / shrink_scalar, 0, 1)",
            hotspot_intensity=None,
            niyama_values=None,
            niyama_risk=None,
            combined_feed_risk=None,
            solidification_times=None,
            fraction_liquid_history=[],
        )
        engine.get_shrinkage_display_intensity = lambda: np.maximum(shrinkage, porosity)
        voxel_mesh = SimpleNamespace(
            voxel_centers=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            voxel_sizes=np.ones(3, dtype=np.float32),
            surface_voxels=None,
            voxel_states=None,
        )

        summary, visualization = build_visualization_payload(
            engine,
            voxel_mesh,
            total_volume=3.0,
            bounds=np.array([[0.0, 2.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            max_voxels=10,
        )
        fields = build_result_fields_payload(visualization, max_points=10)

        np.testing.assert_allclose(visualization["shrinkage"], [0.7, 0.8, 0.4])
        np.testing.assert_allclose(fields["fields"]["shrinkage"]["values"], [0.7, 0.8, 0.4])
        self.assertEqual(summary["shrink_scalar"], 2.0)
        self.assertEqual(summary["shrinkage_stats"]["affected_count"], 3)

    def test_porosity_mode_exposed_separately(self) -> None:
        """Feeding-blocked, path lengths, open fraction, and porosity are exposed."""
        shrinkage = np.array([0.1, 0.8, 0.3], dtype=np.float32)
        porosity = np.array([0.7, 0.2, 0.4], dtype=np.float32)
        feeding_blocked = np.array([False, True, True], dtype=bool)
        feeding_path_lengths = np.array([-1.0, 2.5, 3.0], dtype=np.float32)
        feeding_open_fraction = np.array([1.0, 0.1, 0.0], dtype=np.float32)
        engine = SimpleNamespace(
            shrinkage_probability=shrinkage,
            porosity_probability=porosity,
            shrink_scalar=1.0,
            shrink_scalar_formula="adjusted_risk = clamp(raw_risk / shrink_scalar, 0, 1)",
            hotspot_intensity=None,
            niyama_values=None,
            niyama_risk=None,
            combined_feed_risk=None,
            solidification_times=None,
            fraction_liquid_history=[],
            feeding_blocked=feeding_blocked,
            feeding_path_lengths=feeding_path_lengths,
            feeding_open_fraction=feeding_open_fraction,
        )
        engine.get_shrinkage_display_intensity = lambda: np.maximum(shrinkage, porosity)
        voxel_mesh = SimpleNamespace(
            voxel_centers=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            voxel_sizes=np.ones(3, dtype=np.float32),
            surface_voxels=None,
            voxel_states=None,
        )

        summary, visualization = build_visualization_payload(
            engine,
            voxel_mesh,
            total_volume=3.0,
            bounds=np.array([[0.0, 2.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            max_voxels=10,
        )

        # Check that all new fields are present and have correct length.
        self.assertIn("porosity", visualization)
        self.assertEqual(len(visualization["porosity"]), 3)
        np.testing.assert_allclose(visualization["porosity"], [0.7, 0.2, 0.4])

        self.assertIn("feeding_blocked", visualization)
        self.assertEqual(len(visualization["feeding_blocked"]), 3)
        self.assertEqual(list(visualization["feeding_blocked"]), [False, True, True])

        self.assertIn("feeding_path_lengths", visualization)
        self.assertEqual(len(visualization["feeding_path_lengths"]), 3)
        np.testing.assert_allclose(visualization["feeding_path_lengths"], [-1.0, 2.5, 3.0])

        self.assertIn("feeding_open_fraction", visualization)
        self.assertEqual(len(visualization["feeding_open_fraction"]), 3)
        np.testing.assert_allclose(visualization["feeding_open_fraction"], [1.0, 0.1, 0.0])

    def test_feed_modulus_mode_uses_local_modulus_in_inches(self) -> None:
        engine = SimpleNamespace(
            shrinkage_probability=None,
            hotspot_intensity=None,
            niyama_values=None,
            niyama_risk=None,
            combined_feed_risk=None,
            solidification_times=None,
            fraction_liquid_history=[],
            local_modulus=np.array([0.25, 0.5, 1.0], dtype=np.float32),
        )
        voxel_mesh = SimpleNamespace(
            voxel_centers=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            voxel_sizes=np.ones(3, dtype=np.float32),
            surface_voxels=None,
            voxel_states=None,
            voxel_moduli=np.array([0.1, 0.1, 0.1], dtype=np.float32),
        )

        summary, visualization = build_visualization_payload(
            engine,
            voxel_mesh,
            total_volume=3.0,
            bounds=np.array([[0.0, 2.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            max_voxels=10,
        )
        fields = build_result_fields_payload(visualization, max_points=10)

        np.testing.assert_allclose(visualization["feed_modulus"], [0.25, 0.5, 1.0])
        np.testing.assert_allclose(fields["fields"]["feed_modulus"]["values"], [0.25, 0.5, 1.0])
        self.assertIn("feed_modulus", fields["modes"])
        self.assertAlmostEqual(summary["feed_modulus_stats"]["max_in"], 1.0, places=5)

    def test_feed_modulus_excludes_chill_voxels(self) -> None:
        engine = SimpleNamespace(
            shrinkage_probability=None,
            hotspot_intensity=None,
            niyama_values=None,
            niyama_risk=None,
            combined_feed_risk=None,
            solidification_times=None,
            fraction_liquid_history=[],
            local_modulus=np.array([0.25, 0.5, 1.0], dtype=np.float32),
        )
        voxel_mesh = SimpleNamespace(
            voxel_centers=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            voxel_sizes=np.ones(3, dtype=np.float32),
            surface_voxels=None,
            voxel_states=None,
            voxel_part_types=np.array([0, 2, 0], dtype=np.int32),  # 0=CAST, 2=CHILL
            voxel_moduli=np.array([0.1, 0.1, 0.1], dtype=np.float32),
        )

        summary, visualization = build_visualization_payload(
            engine,
            voxel_mesh,
            total_volume=3.0,
            bounds=np.array([[0.0, 2.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            max_voxels=10,
        )
        fields = build_result_fields_payload(visualization, max_points=10)

        # Chill voxel (index 1) must report 0.0 feed modulus.
        np.testing.assert_allclose(visualization["feed_modulus"], [0.25, 0.0, 1.0])
        np.testing.assert_allclose(fields["fields"]["feed_modulus"]["values"], [0.25, 0.0, 1.0])
        # Stats should only reflect non-chill voxels.
        self.assertAlmostEqual(summary["feed_modulus_stats"]["min_in"], 0.25, places=5)
        self.assertAlmostEqual(summary["feed_modulus_stats"]["max_in"], 1.0, places=5)


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

    def test_ml_mesh_requires_parts_then_recommends_settings(self) -> None:
        create_response = self.client.post(
            "/api/projects",
            json={"name": "ML Mesh Test", "alloy": "WCB", "voxel_size": 0.2, "refinement_level": 2},
        )
        self.assertEqual(create_response.status_code, 201)
        project_id = create_response.get_json()["id"]

        empty_response = self.client.post(f"/api/projects/{project_id}/ml-mesh", json={})
        self.assertEqual(empty_response.status_code, 400)

        mesh_path = Path(self.temp_dir.name) / "ml_casting.stl"
        trimesh.creation.box(extents=(2.0, 2.0, 2.0)).export(mesh_path)
        with mesh_path.open("rb") as handle:
            upload = self.client.post(
                "/api/uploads",
                data={"files": (io.BytesIO(handle.read()), "ml_casting.stl")},
                content_type="multipart/form-data",
            )
        file_entry = upload.get_json()["files"][0]
        self.client.post(
            f"/api/projects/{project_id}/parts",
            json={"name": "Cast", "part_type": "cast", "filepath": file_entry["filepath"], "visible": True},
        )

        response = self.client.post(f"/api/projects/{project_id}/ml-mesh", json={})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertGreater(payload["base_voxel_size"], 0)
        self.assertIn(payload["refinement_level"], {1, 2, 3, 4})
        self.assertIn("reason", payload)
        self.assertIsInstance(payload["used_model"], bool)

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


class ExecuteProjectTaskInputsTests(unittest.TestCase):
    """Verify execute_project_task plumbs performance options into PipelineInputs."""

    def _run_task(self, project: dict):
        from core.simulation_pipeline import PipelineInputs
        from web.routes import execute_project_task

        captured = {}

        def fake_run_pipeline(inputs, **kwargs):
            captured["inputs"] = inputs
            return MagicMock()

        manager = MagicMock()
        manager.should_cancel.return_value = False

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = Flask(__name__)
            app.config["DATABASE"] = str(root / "web.db")
            app.config["UPLOAD_FOLDER"] = str(root / "uploads")
            app.config["TEMP_FOLDER"] = str(root / "temp")
            with app.app_context():
                with patch("web.routes.run_pipeline", side_effect=fake_run_pipeline):
                    with patch("web.routes.build_part_manager", return_value=MagicMock()):
                        with patch(
                            "web.routes.build_fork_artifact",
                            return_value={"summary": {}, "data": {}},
                        ):
                            with patch("web.routes.write_artifact"):
                                with patch("web.routes.repo"):
                                    execute_project_task(
                                        "proj-1",
                                        project,
                                        [],
                                        manager,
                                        root / "web.db",
                                    )

        self.assertIn("inputs", captured)
        self.assertIsInstance(captured["inputs"], PipelineInputs)
        return captured["inputs"]

    def test_use_gpu_option_reaches_pipeline_inputs(self) -> None:
        """performance_options={'use_gpu': True} must reach PipelineInputs.use_gpu=True."""
        inputs = self._run_task({"performance_options": {"use_gpu": True}, "alloy": "WCB"})
        self.assertTrue(inputs.use_gpu)

    def test_use_gpu_defaults_to_false_when_option_missing(self) -> None:
        """Callers that omit use_gpu must keep the existing default of False."""
        inputs = self._run_task({"alloy": "WCB"})
        self.assertFalse(inputs.use_gpu)


if __name__ == "__main__":
    unittest.main()
