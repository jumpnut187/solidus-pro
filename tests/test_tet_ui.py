"""Mesh-type toggle plumbing for desktop and web (tet migration wave 3, docs/tet_migration).

1. ``ui/project_state.py`` round-trips ``performance_options["mesh_type"]``.
2. ``web/services/simulation_queue.build_pipeline_inputs`` and the desktop
   workers put it on ``PipelineInputs.mesh_type``.
3. ``/mesh-preview`` and ``/ml-mesh`` return a tet node estimate on tet and
   the unchanged voxel response otherwise.
4. ``MeshDialog`` combo default, label, store/restore; the riser-design tet
   mesh; the verbatim ``MeshGenerationError`` dialog.
"""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from flask import Flask  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from config.exceptions import MeshGenerationError  # noqa: E402
from config.settings import get_settings  # noqa: E402
from core.part_manager import PartManager  # noqa: E402
from core.simulation_pipeline import SimulationCancelledError  # noqa: E402
from core.tet_mesh import TetVolumeMesh  # noqa: E402
from ui import main_window as mw  # noqa: E402
from ui.project_state import build_project_state, load_project_state  # noqa: E402
from ui.workers import SimulationWorker, SurrogateWorker  # noqa: E402
from web import repository as repo  # noqa: E402
from web.routes import api_bp  # noqa: E402
from web.services.simulation_queue import SimulationQueueManager, build_pipeline_inputs  # noqa: E402

VOXEL_PREVIEW_KEYS = {
    "estimated_voxels",
    "estimated_memory_mb",
    "can_proceed",
    "message",
    "recommended_voxel_size",
}


def _project_payload(performance_options: dict) -> dict:
    return build_project_state(
        PartManager(),
        alloy_name="WCB",
        voxel_size=0.2,
        refinement_level=2,
        memory_mode="auto",
        analysis_options={},
        performance_options=performance_options,
        mesh_setting_origin="manual",
        current_vis_mode="shrinkage",
        ui_state={},
    ).to_dict()


def _box_parts() -> list:
    """The 2 x 2 x 1 in box the preview tests upload, as pipeline parts."""
    from core.part_manager import PART_TYPE_CAST

    return [(trimesh.creation.box(extents=(2.0, 2.0, 1.0)), PART_TYPE_CAST)]


def _box_node_estimate(edge_length_in: float) -> int:
    """``estimate_tet_node_count`` of the preview box."""
    from core.simulation_pipeline import estimate_tet_node_count

    return estimate_tet_node_count(_box_parts(), edge_length_in)


def _box_edge_length_for_node_limit(max_nodes: int) -> float:
    """``tet_edge_length_for_node_limit`` of the preview box."""
    from core.simulation_pipeline import tet_edge_length_for_node_limit

    return tet_edge_length_for_node_limit(_box_parts(), max_nodes)


class ProjectStateMeshTypeTests(unittest.TestCase):
    def test_mesh_type_round_trips_in_performance_options(self) -> None:
        for mesh_type in ("tet", "voxel"):
            with self.subTest(mesh_type=mesh_type):
                payload = _project_payload({"use_gpu": True, "mesh_type": mesh_type})
                loaded = load_project_state(payload).to_dict()
                self.assertEqual(loaded["simulation"]["performance_options"]["mesh_type"], mesh_type)
                self.assertTrue(loaded["simulation"]["performance_options"]["use_gpu"])

    def test_mesh_type_is_normalized_and_unknown_values_dropped(self) -> None:
        loaded = load_project_state(_project_payload({"mesh_type": " TET "})).to_dict()
        self.assertEqual(loaded["simulation"]["performance_options"]["mesh_type"], "tet")
        loaded = load_project_state(_project_payload({"mesh_type": "hex", "use_gpu": False})).to_dict()
        self.assertNotIn("mesh_type", loaded["simulation"]["performance_options"])
        self.assertFalse(loaded["simulation"]["performance_options"]["use_gpu"])

    def test_project_without_mesh_type_loads_without_one(self) -> None:
        loaded = load_project_state(_project_payload({"use_gpu": True})).to_dict()
        self.assertNotIn("mesh_type", loaded["simulation"]["performance_options"])


class PipelineInputsMeshTypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_build_pipeline_inputs_maps_mesh_type(self) -> None:
        inputs = build_pipeline_inputs({"alloy": "WCB", "performance_options": {"mesh_type": "tet"}})
        self.assertEqual(inputs.mesh_type, "tet")

    def test_build_pipeline_inputs_leaves_mesh_type_none_when_absent(self) -> None:
        self.assertIsNone(build_pipeline_inputs({"alloy": "WCB"}).mesh_type)
        self.assertIsNone(build_pipeline_inputs({"alloy": "WCB", "performance_options": {}}).mesh_type)

    def _captured_inputs(self, worker) -> object:
        captured = {}

        def fake_pipeline(*args, **kwargs):
            captured["inputs"] = args[0]
            raise SimulationCancelledError("Cancelled")

        with patch("ui.workers.run_pipeline", side_effect=fake_pipeline), patch(
            "core.surrogate.run_surrogate_pipeline", side_effect=fake_pipeline
        ):
            worker.run()
        return captured["inputs"]

    def test_simulation_worker_passes_mesh_type(self) -> None:
        worker = SimulationWorker(MagicMock(), "WCB", 0.2, 3, performance_options={"mesh_type": "tet"})
        self.assertEqual(self._captured_inputs(worker).mesh_type, "tet")
        worker = SimulationWorker(MagicMock(), "WCB", 0.2, 3)
        self.assertIsNone(self._captured_inputs(worker).mesh_type)

    def test_surrogate_worker_passes_mesh_type(self) -> None:
        worker = SurrogateWorker(MagicMock(), MagicMock(), "WCB", 0.2, 3, mesh_type="tet")
        self.assertEqual(self._captured_inputs(worker).mesh_type, "tet")

    def test_mesh_generation_error_text_is_kept_for_the_dialog(self) -> None:
        worker = SimulationWorker(MagicMock(), "WCB", 0.2, 3, performance_options={"mesh_type": "tet"})
        error = MeshGenerationError("tet mesh too big")
        with patch("ui.workers.run_pipeline", side_effect=error), patch("ui.workers.log_mesh_training_example"):
            worker.run()
        self.assertEqual(worker.mesh_error_message, str(error))

        window = mw.MainWindow.__new__(mw.MainWindow)
        window.progress_bar = MagicMock()
        window.status_label = MagicMock()
        window._set_simulation_running = MagicMock()
        window._current_alloy = "WCB"
        window.simulation_worker = worker
        with patch.object(mw, "QMessageBox") as box:
            window._on_simulation_finished(False, "failed")
        box.critical.assert_called_once_with(window, "Mesh Generation Error", str(error))


class TetRunLoggingAndProgressTests(unittest.TestCase):
    """review_integration.md P1-1 and the owner's GUI findings (a), (b)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _run(self, worker, *, mesh_type_of_result=None, error=None):
        result = MagicMock()
        result.voxel_mesh = MagicMock(mesh_type=mesh_type_of_result)
        result.voxel_mesh.get_voxel_count.return_value = 100
        pipeline = patch("ui.workers.run_pipeline", side_effect=error) if error else patch(
            "ui.workers.run_pipeline", return_value=result
        )
        messages = []
        worker.progress.connect(messages.append)
        with pipeline, patch("ui.workers.log_mesh_training_example") as log, patch.object(
            SimulationWorker, "_maybe_capture_surrogate_training_data"
        ):
            worker.run()
        return log, messages

    def test_tet_runs_never_feed_the_voxel_size_recommender(self) -> None:
        for error in (None, RuntimeError("solver blew up")):
            with self.subTest(error=error):
                worker = SimulationWorker(MagicMock(), "WCB", 0.2, 3, performance_options={"mesh_type": "tet"})
                log, _ = self._run(worker, mesh_type_of_result="tet", error=error)
                log.assert_not_called()
                worker = SimulationWorker(MagicMock(), "WCB", 0.2, 3, performance_options={"mesh_type": "voxel"})
                log, _ = self._run(worker, mesh_type_of_result="voxel", error=error)
                log.assert_called_once()

    def test_settings_default_tet_is_a_tet_run(self) -> None:
        with patch.object(get_settings().mesh, "mesh_type", "tet"):
            worker = SimulationWorker(MagicMock(), "WCB", 0.2, 3)
            self.assertEqual(worker.run_mesh_type(), "tet")
            log, messages = self._run(worker, error=RuntimeError("no mesh"))
        log.assert_not_called()
        self.assertTrue(messages[0].startswith("Generating tetrahedral mesh"), messages[0])
        with patch.object(get_settings().mesh, "mesh_type", "voxel"):
            worker = SimulationWorker(MagicMock(), "WCB", 0.2, 3)
            log, messages = self._run(worker, error=RuntimeError("no mesh"))
        log.assert_called_once()
        self.assertTrue(messages[0].startswith("Generating adaptive voxel mesh"), messages[0])
        with patch.object(get_settings().mesh, "mesh_type", "tet"):
            self.assertEqual(SurrogateWorker(MagicMock(), MagicMock(), "WCB", 0.2, 3).run_mesh_type(), "tet")

    def _progress_window(self):
        window = mw.MainWindow.__new__(mw.MainWindow)
        window.progress_bar = MagicMock()
        window.status_label = MagicMock()
        window.slice_start_times = []
        window.last_slice_index = 0
        return window

    def test_tet_progress_shows_stage_percent_without_z_slices(self) -> None:
        window = self._progress_window()
        tet = SimulationWorker(MagicMock(), "WCB", 0.2, 3, performance_options={"mesh_type": "tet"})
        with patch.object(window, "sender", return_value=tet):
            window._on_detailed_progress(40, 100, "Tetrahedralizing parts")
        window.status_label.setText.assert_called_with("Tetrahedralizing parts 40%")
        window = self._progress_window()
        voxel = SimulationWorker(MagicMock(), "WCB", 0.2, 3, performance_options={"mesh_type": "voxel"})
        with patch.object(window, "sender", return_value=voxel):
            window._on_detailed_progress(40, 100, "Voxelizing")
        window.status_label.setText.assert_called_with("Processing Voxelizing: Z-slice 40/100")

    def test_completion_rate_counts_nodes_on_tet_runs(self) -> None:
        from core.error_logger import logger as error_logger

        with patch.object(error_logger, "info") as info:
            error_logger.log_performance("Complete simulation", 2.0, 1000, count_unit="nodes")
            error_logger.log_performance("Complete simulation", 2.0, 1000)
        self.assertIn("(500 nodes/sec)", info.call_args_list[0].args[0])
        self.assertIn("(500 voxels/sec)", info.call_args_list[1].args[0])

        class _Stop(Exception):
            pass

        for mesh_type, unit in (("tet", "nodes"), ("voxel", "voxels")):
            window = mw.MainWindow.__new__(mw.MainWindow)
            window.progress_bar = MagicMock()
            window._set_simulation_running = MagicMock()
            window._current_alloy = "WCB"
            mesh = MagicMock(mesh_type=mesh_type)
            mesh.get_voxel_count.return_value = 10
            window.simulation_worker = MagicMock(voxel_mesh=mesh)
            window.simulation_worker.simulation.export_results.side_effect = _Stop
            with self.subTest(mesh_type=mesh_type), patch.object(mw, "logger") as logger:
                with self.assertRaises(_Stop):
                    window._on_simulation_finished(True, "completed")
                self.assertEqual(logger.log_performance.call_args.kwargs["count_unit"], unit)


class WebMeshPreviewTetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        database_path = root / "web.db"
        upload_folder = root / "uploads"
        temp_folder = root / "temp"
        upload_folder.mkdir()
        temp_folder.mkdir()
        repo.init_db(database_path)

        app = Flask(__name__)
        app.config["DATABASE"] = database_path
        app.config["UPLOAD_FOLDER"] = upload_folder
        app.config["TEMP_FOLDER"] = temp_folder
        app.extensions["simulation_queue_manager"] = SimulationQueueManager()
        app.register_blueprint(api_bp)
        self.client = app.test_client()
        self.project_id = self._project_with_box(root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _project_with_box(self, root: Path) -> str:
        project_id = self.client.post("/api/projects", json={"name": "Tet", "alloy": "WCB"}).get_json()["id"]
        mesh_path = root / "box.stl"
        trimesh.creation.box(extents=(2.0, 2.0, 1.0)).export(mesh_path)
        with mesh_path.open("rb") as handle:
            upload = self.client.post(
                "/api/uploads",
                data={"files": (io.BytesIO(handle.read()), "box.stl")},
                content_type="multipart/form-data",
            )
        file_entry = upload.get_json()["files"][0]
        self.client.post(
            f"/api/projects/{project_id}/parts",
            json={"name": "Cast", "part_type": "cast", "filepath": file_entry["filepath"], "visible": True},
        )
        return project_id

    def _preview(self, body: dict):
        return self.client.post(f"/api/projects/{self.project_id}/mesh-preview", json=body)

    def test_tet_preview_reports_node_estimate_and_limit(self) -> None:
        mesh_settings = get_settings().mesh
        response = self._preview({"voxel_size": 0.2, "mesh_type": "tet"})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["mesh_type"], "tet")
        edge = 0.2 * float(mesh_settings.tet_edge_length_factor)
        self.assertAlmostEqual(payload["edge_length_in"], edge)
        self.assertEqual(payload["estimated_nodes"], _box_node_estimate(edge))
        self.assertEqual(payload["max_nodes"], int(mesh_settings.tet_max_nodes))
        self.assertTrue(payload["can_proceed"])
        self.assertIsNone(payload["recommended_voxel_size"])
        self.assertNotIn("estimated_voxels", payload)

    def test_tet_preview_above_the_limit_suggests_a_size(self) -> None:
        with patch.object(get_settings().mesh, "tet_max_nodes", 100):
            payload = self._preview({"voxel_size": 0.1, "mesh_type": "tet"}).get_json()
        self.assertFalse(payload["can_proceed"])
        self.assertIn("tet_max_nodes", payload["message"])
        suggested = payload["recommended_voxel_size"]
        self.assertGreater(suggested, 0.1)
        factor = float(get_settings().mesh.tet_edge_length_factor)
        self.assertLessEqual(_box_node_estimate(suggested * factor), 100)
        # The route rounds the helper's edge length up to 0.001 in of voxel size.
        self.assertAlmostEqual(suggested, np.ceil(1000.0 * _box_edge_length_for_node_limit(100) / factor) / 1000.0)

    def test_voxel_preview_is_unchanged(self) -> None:
        for body in ({"voxel_size": 0.2, "mesh_type": "voxel"}, {"voxel_size": 0.2}):
            with self.subTest(body=body), patch.object(get_settings().mesh, "mesh_type", "voxel"):
                payload = self._preview(body).get_json()
                self.assertEqual(set(payload), VOXEL_PREVIEW_KEYS)

    def test_absent_mesh_type_uses_the_settings_default(self) -> None:
        with patch.object(get_settings().mesh, "mesh_type", "tet"):
            payload = self._preview({"voxel_size": 0.2}).get_json()
        self.assertEqual(payload["mesh_type"], "tet")

    def test_unknown_mesh_type_is_rejected(self) -> None:
        self.assertEqual(self._preview({"voxel_size": 0.2, "mesh_type": "hex"}).status_code, 400)

    def test_ml_mesh_adds_the_tet_estimate_only_on_tet(self) -> None:
        url = f"/api/projects/{self.project_id}/ml-mesh"
        voxel = self.client.post(url, json={"mesh_type": "voxel"}).get_json()
        self.assertNotIn("mesh_type", voxel)
        self.assertNotIn("estimated_nodes", voxel)

        tet = self.client.post(url, json={"mesh_type": "tet"}).get_json()
        self.assertEqual(tet["mesh_type"], "tet")
        self.assertEqual(tet["base_voxel_size"], voxel["base_voxel_size"])
        edge = tet["base_voxel_size"] * float(get_settings().mesh.tet_edge_length_factor)
        self.assertAlmostEqual(tet["edge_length_in"], edge)
        self.assertEqual(tet["estimated_nodes"], _box_node_estimate(edge))
        self.assertEqual(tet["max_nodes"], int(get_settings().mesh.tet_max_nodes))
        self.assertIsInstance(tet["can_proceed"], bool)


class MeshDialogMeshTypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_default_follows_settings_and_label_follows_type(self) -> None:
        for default, label in (("tet", mw.MeshDialog.ELEMENT_SIZE_LABEL), ("voxel", mw.MeshDialog.VOXEL_SIZE_LABEL)):
            with self.subTest(default=default), patch.object(get_settings().mesh, "mesh_type", default):
                dialog = mw.MeshDialog()
                self.assertEqual(dialog.get_mesh_type(), default)
                self.assertEqual(dialog.voxel_size_label.text(), label)
                dialog.deleteLater()

    def test_store_restore_and_set_values(self) -> None:
        with patch.object(get_settings().mesh, "mesh_type", "voxel"):
            dialog = mw.MeshDialog()
        dialog.store_original_values()
        dialog.set_mesh_type("tet")
        self.assertEqual(dialog.voxel_size_label.text(), "Mesh size (in):")
        dialog.restore_original_values()
        self.assertEqual(dialog.get_mesh_type(), "voxel")
        self.assertEqual(dialog.voxel_size_label.text(), mw.MeshDialog.VOXEL_SIZE_LABEL)

        dialog.set_values(0.25, 2, "ram", mesh_type="tet")
        self.assertEqual(dialog.get_mesh_type(), "tet")
        dialog.set_values(0.25, 2, "ram")  # mesh_type None leaves it alone
        self.assertEqual(dialog.get_mesh_type(), "tet")
        dialog.set_mesh_type("voxel")
        dialog.set_mesh_type("hex")  # unknown falls back to the tet default
        self.assertEqual(dialog.get_mesh_type(), "tet")
        dialog.deleteLater()

    def test_tet_shows_the_effective_edge_length(self) -> None:
        with patch.object(get_settings().mesh, "tet_edge_length_factor", 2.0):
            dialog = mw.MeshDialog()
            dialog.set_mesh_type("tet")
            dialog.voxel_size_spin.setValue(0.1)
            self.assertEqual(dialog.tet_edge_label.text(), "Tet edge \u2248 0.200 in")
            dialog.voxel_size_spin.setValue(0.15)
            self.assertEqual(dialog.tet_edge_label.text(), "Tet edge \u2248 0.300 in")
            dialog.set_mesh_type("voxel")
            self.assertEqual(dialog.tet_edge_label.text(), "")
            self.assertTrue(dialog.tet_edge_label.isHidden())
        dialog.deleteLater()

    def test_sidebar_summary_shows_the_effective_edge_length(self) -> None:
        from PySide6.QtWidgets import QLabel

        window = MagicMock()
        window.mesh_summary_label = QLabel()
        window._current_voxel_size = 0.1
        window._current_refinement = 3
        with patch.object(get_settings().mesh, "tet_edge_length_factor", 2.0):
            window._performance_options = {"mesh_type": "tet"}
            mw.MainWindow._update_mesh_summary(window)
            self.assertEqual(window.mesh_summary_label.text(), 'Tet: 0.100", edge \u2248 0.200"')
            window._performance_options = {"mesh_type": "voxel"}
            mw.MainWindow._update_mesh_summary(window)
            self.assertEqual(window.mesh_summary_label.text(), 'Voxel: 0.100", Ref: 3')


class RiserDesignTetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        cast = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        cls.mesh = mw.build_riser_design_tet_mesh(cast, 0.25)

    def test_tet_mesh_of_the_cast_part(self) -> None:
        self.assertIsInstance(self.mesh, TetVolumeMesh)
        self.assertEqual(self.mesh.mesh_type, "tet")
        self.assertAlmostEqual(float(np.sum(self.mesh.voxel_volumes)), 1.0, delta=0.01)

    def test_node_limit_raises_mesh_generation_error(self) -> None:
        with patch.object(get_settings().mesh, "tet_max_nodes", 10):
            with self.assertRaises(MeshGenerationError) as ctx:
                mw.build_riser_design_tet_mesh(trimesh.creation.box(extents=(1.0, 1.0, 1.0)), 0.25)
        self.assertIn("tet_max_nodes", str(ctx.exception))

    def test_designer_gets_node_volumes_and_mesh_type(self) -> None:
        dialog = mw.RiserDesignDialog(None)
        dialog.set_mesh_data(self.mesh, MagicMock())
        n = self.mesh.get_voxel_count()
        designer = MagicMock()
        designer.calculate_local_modulus.return_value = np.ones(n)
        designer.design_risers.return_value = []
        designer.hotspots = []
        designer.analyze_coverage.return_value = {"covered_fraction": 1.0, "total_riser_volume": 0.0}
        designer.generate_report.return_value = ""
        with patch("core.riser_design.RiserDesigner", return_value=designer) as cls_mock, patch.object(
            mw, "QMessageBox"
        ) as box:
            dialog._run_analysis()
        box.critical.assert_not_called()
        kwargs = cls_mock.call_args.kwargs
        self.assertEqual(kwargs["mesh_type"], "tet")
        self.assertIs(kwargs["voxel_volumes"], self.mesh.voxel_volumes)
        self.assertIs(kwargs["neighbors"], self.mesh.voxel_neighbors)
        self.assertIs(kwargs["surface_mask"], self.mesh.surface_voxels)
        dialog.deleteLater()


if __name__ == "__main__":
    unittest.main()
