"""STEP handling in the web upload path (web.payloads.upload_cad_file).

Covers the tessellation tolerance a STEP upload is loaded at, the optional
``voxel_size_in`` upload form field, removal of the original STEP after
fan-out, the inch scale of the Test_Model files, and single-body naming.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import numpy as np
import trimesh
from flask import Flask
from werkzeug.datastructures import FileStorage

from core.part_manager import STEP_SUPPORTED, tessellation_tolerance_mm
from web import repository as repo
from web.payloads import upload_cad_file, upload_voxel_size_in
from web.routes import api_bp
from web.services.simulation_queue import SimulationQueueManager


FILLING_TEST_DIR = Path(__file__).resolve().parent.parent / "Test_Model" / "Filling_Test"
RISER_STEP = FILLING_TEST_DIR / "Feeding Distance Test - Riser.step"
CASTING_STEP = FILLING_TEST_DIR / "Feeding Distance Test - Casting.step"

DEFAULT_TOL_MM = tessellation_tolerance_mm(repo.DEFAULT_VOXEL_SIZE_IN)

# Any bytes do for a STEP whose load is mocked; the unit check only logs.
_FAKE_STEP_BYTES = b"ISO-10303-21;\nEND-ISO-10303-21;\n"


def _stl_bytes() -> bytes:
    return bytes(trimesh.creation.box(extents=(1.0, 2.0, 3.0)).export(file_type="stl"))


def _fake_step_scene(*_args: Any, **_kwargs: Any) -> trimesh.Scene:
    """What trimesh returns for a one-body STEP: a Scene in meters."""
    return trimesh.Scene(geometry={"Body": trimesh.creation.box(extents=(0.0254, 0.0254, 0.0254))})


def _upload(folder: Path, data: bytes, filename: str, **kwargs: Any) -> List[Dict[str, Any]]:
    return upload_cad_file(FileStorage(stream=io.BytesIO(data), filename=filename), folder, **kwargs)


def _extents(entry: Dict[str, Any]) -> np.ndarray:
    bounds = entry["mesh_data"]["bounds"]
    return np.array([bounds[axis][1] - bounds[axis][0] for axis in ("x", "y", "z")])


class UploadVoxelSizeTests(unittest.TestCase):
    def test_positive_finite_values_are_used(self) -> None:
        self.assertEqual(upload_voxel_size_in("0.2"), 0.2)
        self.assertEqual(upload_voxel_size_in(0.05), 0.05)

    def test_missing_or_invalid_values_fall_back_to_the_project_default(self) -> None:
        for value in (None, "", "abc", "0", "-0.1", "nan", "inf", "-inf"):
            with self.subTest(value=value):
                self.assertEqual(upload_voxel_size_in(value), repo.DEFAULT_VOXEL_SIZE_IN)


@patch("web.payloads.STEP_SUPPORTED", True)
class UploadLoadArgumentsTests(unittest.TestCase):
    """The arguments upload_cad_file hands trimesh.load, with the load mocked."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_step_is_loaded_at_the_default_voxel_tolerance(self) -> None:
        with patch("web.payloads.trimesh.load", side_effect=_fake_step_scene) as load:
            _upload(self.folder, _FAKE_STEP_BYTES, "part.step")
        self.assertEqual(load.call_args.kwargs["tol_linear"], DEFAULT_TOL_MM)

    def test_step_is_loaded_at_the_requested_voxel_tolerance(self) -> None:
        with patch("web.payloads.trimesh.load", side_effect=_fake_step_scene) as load:
            _upload(self.folder, _FAKE_STEP_BYTES, "part.stp", voxel_size_in="0.2")
        self.assertEqual(load.call_args.kwargs["tol_linear"], tessellation_tolerance_mm(0.2))

    def test_stl_is_loaded_without_a_tolerance(self) -> None:
        with patch("web.payloads.trimesh.load", wraps=trimesh.load) as load:
            _upload(self.folder, _stl_bytes(), "casting.stl", voxel_size_in="0.2")
        self.assertNotIn("tol_linear", load.call_args.kwargs)

    def test_single_nonempty_body_is_named_from_the_file_stem(self) -> None:
        def scene_with_empty_body(*_args: Any, **_kwargs: Any) -> trimesh.Scene:
            scene = _fake_step_scene()
            scene.add_geometry(trimesh.Trimesh(), geom_name="Construction")
            return scene

        with patch("web.payloads.trimesh.load", side_effect=scene_with_empty_body):
            entries = _upload(self.folder, _FAKE_STEP_BYTES, "Feeding Distance Test - Down Sprue.step")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["body_name"], "Feeding_Distance_Test_-_Down_Sprue")
        self.assertEqual(entries[0]["part_type"], "sprue")


class UploadRouteVoxelSizeTests(unittest.TestCase):
    """The ``voxel_size_in`` form field reaches the STEP load."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        upload_folder = root / "uploads"
        upload_folder.mkdir()
        database_path = root / "web.db"
        repo.init_db(database_path)
        app = Flask(__name__)
        app.config["DATABASE"] = database_path
        app.config["UPLOAD_FOLDER"] = upload_folder
        app.config["TEMP_FOLDER"] = root
        app.extensions["simulation_queue_manager"] = SimulationQueueManager()
        app.register_blueprint(api_bp)
        self.client = app.test_client()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _post_step(self, form: Dict[str, str]) -> float:
        data: Dict[str, Any] = {"files": (io.BytesIO(_FAKE_STEP_BYTES), "part.step"), **form}
        with patch("web.payloads.STEP_SUPPORTED", True), patch(
            "web.payloads.trimesh.load", side_effect=_fake_step_scene
        ) as load:
            response = self.client.post("/api/uploads", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        return load.call_args.kwargs["tol_linear"]

    def test_form_voxel_size_sets_the_tolerance(self) -> None:
        self.assertEqual(self._post_step({"voxel_size_in": "0.2"}), tessellation_tolerance_mm(0.2))

    def test_invalid_form_voxel_size_is_ignored(self) -> None:
        self.assertEqual(self._post_step({"voxel_size_in": "-1"}), DEFAULT_TOL_MM)
        self.assertEqual(self._post_step({"voxel_size_in": "nan"}), DEFAULT_TOL_MM)

    def test_missing_form_voxel_size_uses_the_default(self) -> None:
        self.assertEqual(self._post_step({}), DEFAULT_TOL_MM)


class UploadFilesOnDiskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_stl_upload_keeps_its_original_and_points_at_the_export(self) -> None:
        entries = _upload(self.folder, _stl_bytes(), "casting.stl")
        self.assertEqual(len(entries), 1)
        names = sorted(path.name for path in self.folder.iterdir())
        self.assertEqual(len(names), 2)
        self.assertTrue(any(name.endswith("_casting.stl") and "_b0_" not in name for name in names))
        self.assertTrue(Path(entries[0]["filepath"]).exists())
        self.assertIn("_b0_", Path(entries[0]["filepath"]).name)
        np.testing.assert_allclose(_extents(entries[0]), [1.0, 2.0, 3.0], atol=1e-6)

    @unittest.skipUnless(STEP_SUPPORTED, "cascadio not installed")
    def test_single_body_step_original_is_removed(self) -> None:
        entries = _upload(self.folder, RISER_STEP.read_bytes(), RISER_STEP.name)
        self.assertEqual(len(entries), 1)
        remaining = [path for path in self.folder.iterdir()]
        self.assertEqual(remaining, [Path(entries[0]["filepath"])])
        self.assertEqual(Path(entries[0]["filepath"]).suffix, ".stl")


@unittest.skipUnless(STEP_SUPPORTED, "cascadio not installed")
class InchStepUploadTests(unittest.TestCase):
    """Inch-authored Test_Model STEP files, uploaded for real."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_riser_uploads_at_its_inch_size_and_type(self) -> None:
        entry = _upload(self.folder, RISER_STEP.read_bytes(), RISER_STEP.name)[0]
        np.testing.assert_allclose(sorted(_extents(entry)), [3.0, 3.0, 5.0], rtol=1e-3)
        self.assertEqual(entry["part_type"], "riser")
        self.assertEqual(entry["body_name"], "Feeding_Distance_Test_-_Riser")

    def test_casting_uploads_at_its_inch_size(self) -> None:
        entry = _upload(self.folder, CASTING_STEP.read_bytes(), CASTING_STEP.name)[0]
        np.testing.assert_allclose(sorted(_extents(entry)), [1.0, 6.0, 12.0], rtol=1e-3)
        self.assertEqual(entry["part_type"], "cast")

    def test_voxel_tolerance_cuts_the_riser_face_count(self) -> None:
        default_faces = sum(
            len(mesh.faces) for mesh in trimesh.load(str(RISER_STEP), force="scene").geometry.values()
        )
        entry = _upload(self.folder, RISER_STEP.read_bytes(), RISER_STEP.name)[0]
        self.assertLess(entry["mesh_data"]["face_count"], default_faces / 2)
        # The exported STL carries the coarser tessellation too.
        exported = trimesh.load(entry["filepath"], force="mesh")
        self.assertEqual(len(exported.faces), entry["mesh_data"]["face_count"])


if __name__ == "__main__":
    unittest.main()
