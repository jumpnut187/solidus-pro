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
    def test_shrinkage_mode_is_shrink_only(self) -> None:
        """The web 'shrinkage' (Shrink) payload is shrink risk, not max(shrink, porosity)."""
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
        engine.get_shrink_display_intensity = lambda: shrinkage
        engine.get_microporosity_display_intensity = lambda: porosity
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

        np.testing.assert_allclose(visualization["shrinkage"], [0.1, 0.8, 0.0])
        np.testing.assert_allclose(visualization["porosity"], [0.7, 0.2, 0.4])
        self.assertEqual(summary["shrink_scalar"], 2.0)
        self.assertEqual(summary["shrinkage_stats"]["affected_count"], 2)

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

        np.testing.assert_allclose(visualization["feed_modulus"], [0.25, 0.5, 1.0])
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

        # Chill voxel (index 1) must report 0.0 feed modulus.
        np.testing.assert_allclose(visualization["feed_modulus"], [0.25, 0.0, 1.0])
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
        # A part saved without a sleeve choice is insulating; the choice persists.
        self.assertEqual(part_payload["riser_sleeve"], "insulating")
        update_response = self.client.put(
            f"/api/projects/{project_id}/parts/{part_payload['id']}",
            json={"riser_sleeve": "none"},
        )
        self.assertEqual(update_response.status_code, 200)
        self.assertEqual(update_response.get_json()["part"]["riser_sleeve"], "none")
        stored = self.client.get(f"/api/projects/{project_id}").get_json()["parts"]
        self.assertEqual(stored[0]["riser_sleeve"], "none")
        self.assertTrue(stored[0]["is_open_top"])
        # The no-heat-loss sleeve is a stored choice too; an unknown one falls back to insulating.
        for posted, kept in (("adiabatic", "adiabatic"), ("exothermic", "insulating")):
            response = self.client.put(
                f"/api/projects/{project_id}/parts/{part_payload['id']}",
                json={"riser_sleeve": posted},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["part"]["riser_sleeve"], kept)
            stored = self.client.get(f"/api/projects/{project_id}").get_json()["parts"]
            self.assertEqual(stored[0]["riser_sleeve"], kept)

    def test_pour_location_round_trips_through_project_api(self) -> None:
        """A pour location set via PUT persists and reads back through GET.

        The value is stored inside the existing ``ui_state`` JSON blob (see
        ``web.repository._merge_pour_location``) rather than a new column,
        but the API exposes it as a plain top-level ``pour_location`` field.
        """
        create_response = self.client.post(
            "/api/projects",
            json={"name": "Pour Location Test", "alloy": "WCB"},
        )
        self.assertEqual(create_response.status_code, 201)
        project_id = create_response.get_json()["id"]
        self.assertIsNone(create_response.get_json()["project"]["pour_location"])

        pour_location = {
            "schema_version": 1,
            "point_in": [1.5, -2.25, 3.0],
            "normal": [0.0, 0.0, 1.0],
            "part_id": "abc123",
        }
        update_response = self.client.put(
            f"/api/projects/{project_id}",
            json={"name": "Pour Location Test", "alloy": "WCB", "pour_location": pour_location},
        )
        self.assertEqual(update_response.status_code, 200)
        self.assertEqual(update_response.get_json()["project"]["pour_location"], pour_location)

        get_response = self.client.get(f"/api/projects/{project_id}")
        self.assertEqual(get_response.status_code, 200)
        project = get_response.get_json()["project"]
        self.assertEqual(project["pour_location"], pour_location)
        # It must also be readable back out of the ui_state blob it is
        # actually stored in, since that is the persistence mechanism.
        self.assertEqual(project["ui_state"]["pour_location"], pour_location)

        clear_response = self.client.put(
            f"/api/projects/{project_id}",
            json={"name": "Pour Location Test", "alloy": "WCB", "pour_location": None},
        )
        self.assertEqual(clear_response.status_code, 200)
        self.assertIsNone(clear_response.get_json()["project"]["pour_location"])

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

    def test_results_reads_saved_artifact(self) -> None:
        project_id = repo.create_project(self.database_path, {"name": "Artifact Test"})
        artifact_path = self.temp_folder / "artifacts" / f"{project_id}.json"
        artifact_path.parent.mkdir(parents=True)
        artifact = {
            "summary": {"voxel_count": 64, "total_volume": 64.0},
            "visualization": _synthetic_visualization(),
        }
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        repo.mark_project_completed(self.database_path, project_id, artifact["summary"], str(artifact_path))

        result_response = self.client.get(f"/api/projects/{project_id}/results")
        self.assertEqual(result_response.status_code, 200)
        self.assertEqual(result_response.get_json()["data"]["voxel_count"], 64)
        self.assertIn("hot_tear", result_response.get_json()["visualization"])

        summary_response = self.client.get(f"/api/projects/{project_id}/results?summary_only=1")
        self.assertEqual(summary_response.status_code, 200)
        self.assertNotIn("visualization", summary_response.get_json())

    def test_retired_json_result_routes_are_gone(self) -> None:
        """The sampled-field, surface and volume routes no longer exist.

        Everything the viewport draws now comes from the dense grid endpoints
        in ``web/routes_fields.py``. A 404 here is the whole point: a stale
        client asking for the old shapes must fail loudly rather than get a
        half-maintained payload.
        """
        project_id = repo.create_project(self.database_path, {"name": "Retired Routes"})
        artifact_path = self.temp_folder / "artifacts" / f"{project_id}.json"
        artifact_path.parent.mkdir(parents=True)
        summary = {"voxel_count": 64, "total_volume": 64.0}
        artifact_path.write_text(
            json.dumps({"summary": summary, "visualization": _synthetic_visualization()}),
            encoding="utf-8",
        )
        repo.mark_project_completed(self.database_path, project_id, summary, str(artifact_path))

        for path in ("result-fields", "surface?mode=hot_tear&threshold=0.5", "volume"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(f"/api/projects/{project_id}/{path}").status_code, 404)


class BuildPipelineInputsTests(unittest.TestCase):
    """Verify the web layer plumbs stored options into PipelineInputs.

    Targets web.services.simulation_queue.build_pipeline_inputs, which is the
    mapping the live queued-simulation path actually uses. These assertions
    previously ran against web.routes.execute_project_task -- a dead duplicate
    of the same mapping that nothing but this test called, so drift between the
    two copies could not have been caught.
    """

    def _build(self, project: dict):
        from web.services.simulation_queue import build_pipeline_inputs

        return build_pipeline_inputs(project)

    def test_use_gpu_option_reaches_pipeline_inputs(self) -> None:
        """performance_options={'use_gpu': True} must reach PipelineInputs.use_gpu=True."""
        inputs = self._build({"performance_options": {"use_gpu": True}, "alloy": "WCB"})
        self.assertTrue(inputs.use_gpu)

    def test_use_gpu_defaults_to_false_when_option_missing(self) -> None:
        """Callers that omit use_gpu must keep the existing default of False."""
        inputs = self._build({"alloy": "WCB"})
        self.assertFalse(inputs.use_gpu)

    def test_performance_options_are_plumbed_through(self) -> None:
        """Every performance key the queue reads must land on PipelineInputs."""
        inputs = self._build(
            {
                "alloy": "CF8",
                "voxel_size": 0.25,
                "refinement_level": 2,
                "memory_mode": "conservative",
                "performance_options": {
                    "use_incremental": True,
                    "use_chunked_voxelization": True,
                    "part_assignment_mode": "parallel",
                    "part_assignment_workers": 4,
                    "part_assignment_batch_size": 1000,
                    "max_voxels_for_visualization": 12345,
                },
            }
        )

        self.assertEqual(inputs.alloy_name, "CF8")
        self.assertEqual(inputs.voxel_size, 0.25)
        self.assertEqual(inputs.refinement_level, 2)
        self.assertEqual(inputs.memory_mode, "conservative")
        self.assertTrue(inputs.use_incremental)
        self.assertTrue(inputs.use_chunked_voxelization)
        self.assertEqual(inputs.part_assignment_mode, "parallel")
        self.assertEqual(inputs.part_assignment_workers, 4)
        self.assertEqual(inputs.part_assignment_batch_size, 1000)
        self.assertEqual(inputs.max_voxels_for_visualization, 12345)

    def test_pour_location_reaches_pipeline_inputs(self) -> None:
        """A project's pour_location must reach PipelineInputs.pour_location.

        Mold filling (core.filling_stage.maybe_run_filling) gates entirely on
        this attribute being set, so the web layer must plumb it through.
        """
        pour_location = {
            "schema_version": 1,
            "point_in": [1.0, 2.0, 3.0],
            "normal": None,
            "part_id": None,
        }
        inputs = self._build({"alloy": "WCB", "pour_location": pour_location})
        self.assertEqual(inputs.pour_location, pour_location)

    def test_pour_location_defaults_to_none_when_missing(self) -> None:
        """Projects with no pour location must leave PipelineInputs.pour_location unset."""
        inputs = self._build({"alloy": "WCB"})
        self.assertIsNone(inputs.pour_location)

    def test_analysis_and_solidification_params_pass_through(self) -> None:
        """Nested option dicts travel opaquely, so new keys need no queue change."""
        inputs = self._build(
            {
                "analysis_options": {"hot_tear": True},
                "solidification_params": {"feeding": {"base_feeding_distance": 8.0}},
            }
        )

        self.assertEqual(inputs.analysis_options, {"hot_tear": True})
        self.assertEqual(
            inputs.solidification_params,
            {"feeding": {"base_feeding_distance": 8.0}},
        )


if __name__ == "__main__":
    unittest.main()
