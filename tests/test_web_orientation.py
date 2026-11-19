"""Tests for the web "Orient & Parting Line" tool.

Covers the parting-line routes (``web/routes_orientation.py``), the guarantee
that a generic project PUT never erases ``ui_state["parting_line"]``, and the
fixture the browser's port of :mod:`core.orientation` is checked against
(``web_frontend/tests/orientation.check.ts``).

Regenerate the fixture after a change to ``core/orientation.py`` with::

    python -m tests.test_web_orientation --write-fixture
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import trimesh
from flask import Flask

from core import orientation
from web import repository as repo
from web.routes import api_bp
from web.routes_orientation import orientation_bp
from web.services.simulation_queue import SimulationQueueManager

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "web_frontend" / "tests" / "orientation.fixture.json"


def _flanged_stl_bytes() -> bytes:
    """A 6 x 6 x 1 in flange under a 2 x 2 x 3 in boss: widest section at the bottom."""
    flange = trimesh.creation.box(extents=(6.0, 6.0, 1.0))
    flange.apply_translation((0.0, 0.0, 0.5))
    boss = trimesh.creation.box(extents=(2.0, 2.0, 3.0))
    boss.apply_translation((0.0, 0.0, 2.5))
    mesh = trimesh.util.concatenate([flange, boss])
    return bytes(mesh.export(file_type="stl"))


def _fixture_mesh() -> trimesh.Trimesh:
    """An off-centre, unequal box, so every axis and sign shows up in its bounds."""
    mesh = trimesh.creation.box(extents=(4.0, 2.0, 1.0))
    mesh.apply_translation((0.5, -0.3, 0.2))
    return mesh


def _vec(values: Any) -> List[float]:
    return [float(v) for v in values]


def build_fixture() -> Dict[str, Any]:
    """Cases the TypeScript port must reproduce, computed by ``core.orientation``."""
    mesh = _fixture_mesh()
    rotations = [(0, 0, 0), (30, -20, 75), (90, 0, 0), (0, 90, 0), (-45, 30, 170), (10, 89.9, -5)]
    placements = [
        {"rotation": [30.0, -20.0, 75.0], "position": [1.0, 2.0, 3.0], "scale": 1.0},
        {"rotation": [-45.0, 30.0, 170.0], "position": [-4.0, 0.5, 10.0], "scale": 2.0},
        {"rotation": [0.0, 0.0, 0.0], "position": [0.0, 0.0, 0.0], "scale": 1.0},
    ]
    world_axis_cases = [
        {"rotation": [0, 0, 0], "position": [0, 0, 0], "axis": "x", "angle": 90, "pivot": [0, 0, 0]},
        {"rotation": [90, 0, 0], "position": [1, 2, 3], "axis": "y", "angle": 90, "pivot": [1, 2, 3.5]},
        {"rotation": [30, -20, 75], "position": [1, 2, 3], "axis": "z", "angle": -45, "pivot": [0.5, 1, 2]},
        {"rotation": [-45, 30, 170], "position": [0, 0, 0], "axis": "x", "angle": 15, "pivot": [2, -1, 4]},
        {"rotation": [0, 0, 0], "position": [0, 0, 0], "axis": [1, 1, 0], "angle": 60, "pivot": [0, 0, 0]},
    ]
    face_cases = [
        {"rotation": [0, 0, 0], "position": [0, 0, 0], "normal": [1, 0, 0], "pivot": [0, 0, 0], "target": [0, 0, -1]},
        {"rotation": [0, 0, 0], "position": [0, 0, 5], "normal": [0, 0, 1], "pivot": [0, 0, 5], "target": [0, 0, -1]},
        {"rotation": [30, -20, 75], "position": [1, 2, 3], "normal": [0.2, -0.5, 0.84], "pivot": [1, 2, 3], "target": [0, 0, 1]},
        {"rotation": [0, 0, 0], "position": [0, 0, 0], "normal": [0, 0, -1], "pivot": [0, 0, 0], "target": [0, 0, -1]},
    ]
    return {
        "source": "core/orientation.py via tests/test_web_orientation.py build_fixture()",
        "vertices": _vec(np.asarray(mesh.vertices).reshape(-1)),
        "euler_to_matrix": [
            {"rotation": _vec(r), "matrix": [_vec(row) for row in orientation.euler_to_matrix(r)]} for r in rotations
        ],
        "matrix_to_euler": [
            {"matrix": [_vec(row) for row in orientation.euler_to_matrix(r)],
             "euler": _vec(orientation.matrix_to_euler(orientation.euler_to_matrix(r)))}
            for r in rotations
        ],
        "rotate_about_world_axis": [
            {
                **case,
                "result": [
                    _vec(part)
                    for part in orientation.rotate_about_world_axis(
                        case["rotation"], case["position"], case["axis"], case["angle"], case["pivot"]
                    )
                ],
            }
            for case in world_axis_cases
        ],
        "face_to_direction": [
            {
                **case,
                "result": [
                    _vec(part)
                    for part in orientation.face_to_direction(
                        case["rotation"], case["position"], case["normal"], case["pivot"], case["target"]
                    )
                ],
            }
            for case in face_cases
        ],
        "world_bounds": [
            {
                **placement,
                "bounds": [
                    _vec(row)
                    for row in orientation.world_bounds(
                        mesh, placement["rotation"], placement["position"], placement["scale"]
                    )
                ],
            }
            for placement in placements
        ],
        "snap_to_parting_dz": [
            {"bounds": [[0, 0, 1.5], [2, 2, 4.0]], "parting_z": 3.0, "side": side,
             "dz": orientation.snap_to_parting_dz(np.array([[0, 0, 1.5], [2, 2, 4.0]]), 3.0, side)}
            for side in ("above", "below")
        ],
        "drop_to_floor_dz": [
            {"bounds": [[0, 0, -2.25], [1, 1, 1]], "floor_z": 0.0,
             "dz": orientation.drop_to_floor_dz(np.array([[0, 0, -2.25], [1, 1, 1]]), 0.0)},
            {"bounds": [[0, 0, 3.0], [1, 1, 5]], "floor_z": 1.0,
             "dz": orientation.drop_to_floor_dz(np.array([[0, 0, 3.0], [1, 1, 5]]), 1.0)},
        ],
        "snap_angle": [
            {"angle": angle, "step": step, "result": orientation.snap_angle(angle, step)}
            for angle, step in [(37.0, 15.0), (-97.0, 45.0), (181.0, 90.0), (-179.0, 5.0), (12.3, 0.0)]
        ],
    }


def _close(a: Any, b: Any, tol: float = 1e-9) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    return a == b


class OrientationFixtureTestCase(unittest.TestCase):
    def test_fixture_matches_core_orientation(self) -> None:
        """The browser check reads this file; it must still describe core.orientation."""
        stored = json.loads(FIXTURE_PATH.read_text())
        self.assertTrue(
            _close(stored, build_fixture()),
            "web_frontend/tests/orientation.fixture.json is stale: run "
            "`python -m tests.test_web_orientation --write-fixture`.",
        )

    def test_babylon_bug_case_bounds(self) -> None:
        """The (30, -20, 75) / (1, 2, 3) case the viewport fix is verified against."""
        mesh = _fixture_mesh()
        bounds = orientation.world_bounds(mesh, (30, -20, 75), (1, 2, 3))
        # Built by hand from R = Rz @ Ry @ Rx so a convention change is caught here too.
        rx, ry, rz = np.radians([30.0, -20.0, 75.0])
        cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
        r = (
            np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        )
        world = np.asarray(mesh.vertices) @ r.T + np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(bounds[0], world.min(axis=0), atol=1e-12)
        np.testing.assert_allclose(bounds[1], world.max(axis=0), atol=1e-12)


class PartingLineRoutesTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.database_path = root / "test.db"
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
        app.register_blueprint(orientation_bp)
        self.client = app.test_client()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_project(self) -> str:
        response = self.client.post(
            "/api/projects",
            json={"name": "Flange", "alloy": "WCB", "voxel_size": 0.2, "refinement_level": 1},
        )
        self.assertEqual(response.status_code, 201)
        return str(response.get_json()["id"])

    def _add_casting(self, project_id: str, **placement: float) -> str:
        response = self.client.post(
            "/api/uploads",
            data={"files": (io.BytesIO(_flanged_stl_bytes()), "flange_cast.stl")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        uploaded = response.get_json()["files"][0]
        body = {"name": "Flange", "part_type": "cast", "filepath": uploaded["filepath"], **placement}
        response = self.client.post(f"/api/projects/{project_id}/parts", json=body)
        self.assertEqual(response.status_code, 201)
        return str(response.get_json()["id"])

    def test_suggestions_for_the_stored_casting(self) -> None:
        project_id = self._create_project()
        part_id = self._add_casting(project_id)

        response = self.client.get(f"/api/projects/{project_id}/parting-line")

        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertIsNone(payload["parting_line"])
        self.assertEqual(payload["part_id"], part_id)
        suggestions = payload["suggestions"]
        self.assertAlmostEqual(suggestions["bottom"], 0.0, places=4)
        self.assertAlmostEqual(suggestions["top"], 4.0, places=4)
        self.assertAlmostEqual(suggestions["center"], 2.0, places=4)
        # The flange is the widest section; the plane sits on its top face.
        self.assertAlmostEqual(suggestions["max_section"], 1.0, places=3)
        # With nothing stored, the split is taken at the widest section.
        self.assertAlmostEqual(payload["split_z"], suggestions["max_section"], places=4)
        self.assertAlmostEqual(payload["split"]["drag_depth"], 1.0, places=3)
        self.assertAlmostEqual(payload["split"]["cope_depth"], 3.0, places=3)

    def test_client_placement_overrides_the_stored_row(self) -> None:
        project_id = self._create_project()
        self._add_casting(project_id)

        # Upside down and lifted: the flange is now on top.
        response = self.client.get(
            f"/api/projects/{project_id}/parting-line",
            query_string={"rotation": "180,0,0", "position": "0,0,4", "z": "2"},
        )

        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertAlmostEqual(payload["bounds"][0][2], 0.0, places=4)
        self.assertAlmostEqual(payload["bounds"][1][2], 4.0, places=4)
        # Widest band on top: the plane goes on its top face, so the whole
        # casting draws from the drag without an undercut.
        self.assertAlmostEqual(payload["suggestions"]["max_section"], 4.0, places=3)
        self.assertAlmostEqual(payload["split_z"], 2.0)
        expected = orientation.cope_drag_split(
            trimesh.load(io.BytesIO(_flanged_stl_bytes()), file_type="stl"), (180, 0, 0), (0, 0, 4), 2.0
        )
        for key, value in expected.items():
            self.assertAlmostEqual(payload["split"][key], value, places=4)

    def test_put_stores_and_get_returns_the_parting_line(self) -> None:
        project_id = self._create_project()
        self._add_casting(project_id)

        response = self.client.put(
            f"/api/projects/{project_id}/parting-line",
            json={"z": 1.25, "visible": False, "locked_to_casting": True},
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertEqual(response.get_json()["parting_line"], {"z": 1.25, "visible": False, "locked_to_casting": True})

        payload = self.client.get(f"/api/projects/{project_id}/parting-line").get_json()
        self.assertEqual(payload["parting_line"]["z"], 1.25)
        self.assertAlmostEqual(payload["split_z"], 1.25)
        project = self.client.get(f"/api/projects/{project_id}").get_json()["project"]
        self.assertEqual(project["ui_state"]["parting_line"]["z"], 1.25)

    def test_put_null_clears_the_parting_line(self) -> None:
        project_id = self._create_project()
        self.client.put(f"/api/projects/{project_id}/parting-line", json={"z": 2.0})

        response = self.client.put(f"/api/projects/{project_id}/parting-line", json={"parting_line": None})

        self.assertEqual(response.status_code, 200)
        project = self.client.get(f"/api/projects/{project_id}").get_json()["project"]
        self.assertNotIn("parting_line", project["ui_state"])

    def test_put_rejects_a_bad_z(self) -> None:
        project_id = self._create_project()
        for body in ({"visible": True}, {"z": "high"}, {"z": float("inf")}, [1, 2]):
            with self.subTest(body=body):
                response = self.client.put(
                    f"/api/projects/{project_id}/parting-line",
                    data=json.dumps(body).replace("Infinity", "1e999"),
                    content_type="application/json",
                )
                self.assertEqual(response.status_code, 400)

    def test_generic_project_put_keeps_the_parting_line(self) -> None:
        project_id = self._create_project()
        self.client.put(f"/api/projects/{project_id}/parting-line", json={"z": 1.5})

        # The shell's settings autosave, with and without a ui_state of its own.
        self.client.put(f"/api/projects/{project_id}", json={"alloy": "CF8M"})
        self.client.put(f"/api/projects/{project_id}", json={"ui_state": {"something": 1}})

        project = self.client.get(f"/api/projects/{project_id}").get_json()["project"]
        self.assertEqual(project["alloy"], "CF8M")
        self.assertEqual(project["ui_state"]["parting_line"]["z"], 1.5)
        self.assertEqual(project["ui_state"]["something"], 1)

    def test_old_project_without_the_key_reports_no_parting_line(self) -> None:
        project_id = self._create_project()
        self._add_casting(project_id)
        project = self.client.get(f"/api/projects/{project_id}").get_json()["project"]
        self.assertNotIn("parting_line", project["ui_state"])
        payload = self.client.get(f"/api/projects/{project_id}/parting-line").get_json()
        self.assertIsNone(payload["parting_line"])

    def test_bad_query_and_missing_targets(self) -> None:
        project_id = self._create_project()
        self.assertEqual(self.client.get("/api/projects/nope/parting-line").status_code, 404)
        # No casting yet.
        self.assertEqual(self.client.get(f"/api/projects/{project_id}/parting-line").status_code, 404)
        self._add_casting(project_id)
        bad = self.client.get(f"/api/projects/{project_id}/parting-line", query_string={"rotation": "1,2"})
        self.assertEqual(bad.status_code, 400)
        bad_scale = self.client.get(f"/api/projects/{project_id}/parting-line", query_string={"scale": "0"})
        self.assertEqual(bad_scale.status_code, 400)


if __name__ == "__main__":
    if "--write-fixture" in sys.argv:
        FIXTURE_PATH.write_text(json.dumps(build_fixture(), indent=1) + "\n")
        print(f"wrote {FIXTURE_PATH}")
    else:
        unittest.main()
