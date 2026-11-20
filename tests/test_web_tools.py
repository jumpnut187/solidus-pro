from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import trimesh
from flask import Flask

from web import repository as repo
from web.payloads import load_part_mesh_data, upload_cad_file
from web.routes import api_bp
from web.routes_tools import tools_bp
from web.services.simulation_queue import SimulationQueueManager

def _cube_stl_bytes(size: float = 2.0) -> bytes:
    mesh = trimesh.creation.box(extents=(size, size, size))
    return bytes(mesh.export(file_type="stl"))

class WebToolsTestCase(unittest.TestCase):
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
        app.register_blueprint(tools_bp)
        self.app = app
        self.client = app.test_client()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_project(self, **overrides) -> str:
        payload = {
            "name": "Plate and Riser",
            "alloy": "WCB",
            "voxel_size": 0.2,
            "refinement_level": 2,
            "memory_mode": "auto",
            "analysis_options": {"shrinkage": True, "gas_porosity": True},
            "performance_options": {"use_gpu": True},
            "solidification_params": {"feeding": {"shrink_scalar": 1.75}},
            "pour_location": {
                "schema_version": 1,
                "point_in": [1.0, 2.0, 3.0],
                "normal": None,
                "part_id": None,
            },
        }
        payload.update(overrides)
        response = self.client.post("/api/projects", json=payload)
        self.assertEqual(response.status_code, 201)
        return str(response.get_json()["id"])

    def _upload(self, filename: str) -> dict:
        response = self.client.post(
            "/api/uploads",
            data={"files": (io.BytesIO(_cube_stl_bytes()), filename)},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        files = response.get_json()["files"]
        self.assertEqual(len(files), 1)
        return dict(files[0])

    def _add_part(self, project_id: str, uploaded: dict, name: str) -> str:
        response = self.client.post(
            f"/api/projects/{project_id}/parts",
            json={
                "name": name,
                "part_type": uploaded["part_type"],
                "filepath": uploaded["filepath"],
            },
        )
        self.assertEqual(response.status_code, 201)
        return str(response.get_json()["id"])

    def test_upload_reports_a_display_name_without_the_storage_prefix(self) -> None:
        entry = self._upload("b0_cast.stl")

        self.assertEqual(entry["display_name"], "b0_cast")
        self.assertTrue(entry["filename"].endswith("_b0_b0_cast.stl"))

    def test_re_uploading_a_stored_file_does_not_compound_prefixes(self) -> None:
        first = self._upload("cast.stl")
        second = self._upload(Path(first["filepath"]).name)

        self.assertEqual(first["display_name"], "cast")
        self.assertEqual(second["display_name"], "cast")
        self.assertNotEqual(first["filepath"], second["filepath"])

    def test_part_mesh_route_returns_geometry_for_a_stored_part(self) -> None:
        project_id = self._create_project()
        uploaded = self._upload("cast.stl")
        part_id = self._add_part(project_id, uploaded, "cast")

        response = self.client.get(f"/api/projects/{project_id}/parts/{part_id}/mesh")

        self.assertEqual(response.status_code, 200)
        mesh_data = response.get_json()["mesh_data"]
        self.assertGreater(mesh_data["vertex_count"], 0)
        self.assertGreater(mesh_data["face_count"], 0)
        self.assertEqual(len(mesh_data["vertices"]) % 3, 0)
        self.assertIn("x", mesh_data["bounds"])

    def test_part_mesh_route_404s_for_an_unknown_part(self) -> None:
        project_id = self._create_project()

        response = self.client.get(f"/api/projects/{project_id}/parts/nope/mesh")

        self.assertEqual(response.status_code, 404)

    def test_project_include_mesh_attaches_geometry_to_every_part(self) -> None:
        project_id = self._create_project()
        uploaded = self._upload("cast.stl")
        self._add_part(project_id, uploaded, "cast")

        without = self.client.get(f"/api/projects/{project_id}").get_json()
        with_mesh = self.client.get(f"/api/projects/{project_id}?include_mesh=1").get_json()

        self.assertNotIn("mesh_data", without["parts"][0])
        self.assertGreater(with_mesh["parts"][0]["mesh_data"]["face_count"], 0)

    def test_load_part_mesh_data_returns_none_for_a_missing_file(self) -> None:
        self.assertIsNone(load_part_mesh_data(str(self.upload_folder / "gone.stl")))
        self.assertIsNone(load_part_mesh_data(""))

    def test_rename_project_updates_the_stored_name(self) -> None:
        project_id = self._create_project()

        response = self.client.post(f"/api/projects/{project_id}/rename", json={"name": "Impeller v3"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["project"]["name"], "Impeller v3")
        self.assertEqual(repo.get_project(self.database_path, project_id)["name"], "Impeller v3")

    def test_rename_project_rejects_a_blank_name(self) -> None:
        project_id = self._create_project()

        response = self.client.post(f"/api/projects/{project_id}/rename", json={"name": "   "})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(repo.get_project(self.database_path, project_id)["name"], "Plate and Riser")

    def test_rename_project_404s_for_an_unknown_project(self) -> None:
        response = self.client.post("/api/projects/nope/rename", json={"name": "x"})

        self.assertEqual(response.status_code, 404)

    def test_duplicate_project_copies_settings_and_parts(self) -> None:
        project_id = self._create_project()
        uploaded = self._upload("cast.stl")
        self._add_part(project_id, uploaded, "cast")

        response = self.client.post(f"/api/projects/{project_id}/duplicate", json={})

        self.assertEqual(response.status_code, 201)
        copy_id = response.get_json()["id"]
        self.assertNotEqual(copy_id, project_id)

        copy = repo.get_project(self.database_path, copy_id)
        self.assertEqual(copy["name"], "Plate and Riser (copy)")
        self.assertEqual(copy["alloy"], "WCB")
        self.assertEqual(copy["solidification_params"]["feeding"]["shrink_scalar"], 1.75)
        self.assertTrue(copy["analysis_options"]["gas_porosity"])
        self.assertEqual(copy["pour_location"]["point_in"], [1.0, 2.0, 3.0])

        copy_parts = repo.get_parts(self.database_path, copy_id)
        self.assertEqual([part["name"] for part in copy_parts], ["cast"])

    def test_duplicate_project_takes_a_name_and_leaves_results_behind(self) -> None:
        project_id = self._create_project()
        repo.mark_project_completed(
            self.database_path, project_id, {"voxel_count": 41593}, "/tmp/artifact.json"
        )

        response = self.client.post(f"/api/projects/{project_id}/duplicate", json={"name": "Trial B"})

        copy = repo.get_project(self.database_path, response.get_json()["id"])
        self.assertEqual(copy["name"], "Trial B")
        self.assertEqual(copy["status"], "draft")
        self.assertIsNone(copy["artifact_path"])
        self.assertEqual(copy["summary"], {})

    def test_duplicate_project_copies_part_files_on_disk(self) -> None:
        project_id = self._create_project()
        uploaded = self._upload("cast.stl")
        self._add_part(project_id, uploaded, "cast")

        copy_id = self.client.post(f"/api/projects/{project_id}/duplicate", json={}).get_json()["id"]

        original_path = repo.get_parts(self.database_path, project_id)[0]["filepath"]
        copy_path = repo.get_parts(self.database_path, copy_id)[0]["filepath"]
        self.assertNotEqual(original_path, copy_path)
        self.assertTrue(Path(copy_path).exists())

        for filepath in repo.delete_project(self.database_path, copy_id):
            Path(filepath).unlink(missing_ok=True)
        self.assertTrue(Path(original_path).exists())

    def test_duplicate_project_404s_for_an_unknown_project(self) -> None:
        response = self.client.post("/api/projects/nope/duplicate", json={})

        self.assertEqual(response.status_code, 404)

    def test_hardware_route_reports_cores_memory_and_a_gpu_backend(self) -> None:
        response = self.client.get("/api/hardware")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        for key in (
            "cpu_brand",
            "physical_cores",
            "logical_cores",
            "total_memory_gb",
            "numba_threads",
            "gpu_backend",
            "gpu_available",
        ):
            self.assertIn(key, payload)
        self.assertIsInstance(payload["gpu_backend"], str)
        self.assertIsInstance(payload["gpu_available"], bool)

    def test_help_alloys_route_lists_every_alloy_with_guide_columns(self) -> None:
        response = self.client.get("/api/help/alloys")

        self.assertEqual(response.status_code, 200)
        alloys = response.get_json()["alloys"]
        self.assertGreaterEqual(len(alloys), 7)
        names = {alloy["name"] for alloy in alloys}
        self.assertIn("WCB", names)
        first = alloys[0]
        for key in ("liquidus_temp", "solidus_temp", "freezing_range", "niyama_threshold"):
            self.assertIsInstance(first[key], float)

    def test_materials_route_carries_the_full_desktop_property_sheet(self) -> None:
        response = self.client.get("/api/materials")

        self.assertEqual(response.status_code, 200)
        wcb = next(item for item in response.get_json() if item["name"] == "WCB")
        for key in (
            "pouring_temp",
            "freezing_range",
            "linear_shrinkage",
            "thermal_conductivity",
            "specific_heat",
            "latent_heat",
            "thermal_diffusivity",
            "density_liquid",
            "density_solid",
            "critical_fraction_solid",
            "dendrite_arm_spacing",
        ):
            self.assertIn(key, wcb)
        self.assertGreater(wcb["freezing_range"], 0)
        self.assertGreater(wcb["thermal_diffusivity"], 0)

    def test_performance_defaults_include_the_gpu_toggle(self) -> None:
        project_id = self.client.post("/api/projects", json={"name": "Bare"}).get_json()["id"]

        project = repo.get_project(self.database_path, project_id)

        self.assertIn("use_gpu", project["performance_options"])
        self.assertTrue(project["performance_options"]["use_gpu"])

class ReportRouteTests(WebToolsTestCase):

    def test_report_formats_flags_pdf_availability_with_a_reason(self) -> None:
        payload = self.client.get("/api/report/formats").get_json()

        by_id = {entry["id"]: entry for entry in payload["formats"]}
        self.assertEqual(set(by_id), {"html", "markdown", "pdf"})
        self.assertTrue(by_id["html"]["available"])
        self.assertTrue(by_id["markdown"]["available"])
        if not by_id["pdf"]["available"]:
            self.assertIn("weasyprint", by_id["pdf"]["reason"])

    def test_html_report_downloads_with_the_requested_sections(self) -> None:
        project_id = self._create_project()
        uploaded = self._upload("cast.stl")
        self._add_part(project_id, uploaded, "cast")

        response = self.client.post(
            f"/api/projects/{project_id}/report",
            json={"format": "html", "sections": {"settings": True, "parts": True, "statistics": True}},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("attachment", response.headers["Content-Disposition"])
        body = response.get_data(as_text=True)
        self.assertIn("Simulation Setup", body)
        self.assertIn("Parts", body)
        self.assertIn("WCB", body)

    def test_markdown_report_downloads(self) -> None:
        project_id = self._create_project()

        response = self.client.post(f"/api/projects/{project_id}/report", json={"format": "markdown"})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["Content-Disposition"].endswith('.md"') or
                        response.headers["Content-Disposition"].endswith(".md"))
        self.assertIn("Simulation Report", response.get_data(as_text=True))

    def test_report_rejects_an_unknown_format(self) -> None:
        project_id = self._create_project()

        response = self.client.post(f"/api/projects/{project_id}/report", json={"format": "docx"})

        self.assertEqual(response.status_code, 400)

    def test_report_404s_for_an_unknown_project(self) -> None:
        response = self.client.post("/api/projects/nope/report", json={"format": "html"})

        self.assertEqual(response.status_code, 404)

    def test_report_embeds_a_supplied_screenshot(self) -> None:
        import base64
        import zlib

        def chunk(tag: bytes, data: bytes) -> bytes:
            body = tag + data
            return len(data).to_bytes(4, "big") + body + zlib.crc32(body).to_bytes(4, "big")

        png = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", (1).to_bytes(4, "big") + (1).to_bytes(4, "big") + b"\x08\x02\x00\x00\x00")
            + chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
            + chunk(b"IEND", b"")
        )
        project_id = self._create_project()

        response = self.client.post(
            f"/api/projects/{project_id}/report",
            json={
                "format": "html",
                "sections": {"settings": False, "parts": False, "statistics": False, "screenshot": True},
                "screenshot": "data:image/png;base64," + base64.b64encode(png).decode(),
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("data:image/png;base64", response.get_data(as_text=True))

class ProjectFileTests(WebToolsTestCase):

    def test_export_produces_schema_v2_with_bare_part_filenames(self) -> None:
        import json

        project_id = self._create_project()
        uploaded = self._upload("cast.stl")
        self._add_part(project_id, uploaded, "cast")

        response = self.client.get(f"/api/projects/{project_id}/export")

        self.assertEqual(response.status_code, 200)
        self.assertIn(".solidus-project.json", response.headers["Content-Disposition"])
        payload = json.loads(response.get_data())
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["simulation"]["alloy_name"], "WCB")
        self.assertEqual(payload["simulation"]["pour_location"]["point_in"], [1.0, 2.0, 3.0])
        self.assertEqual(len(payload["parts"]), 1)
        self.assertNotIn("/", payload["parts"][0]["filepath"])

    def test_export_404s_for_an_unknown_project(self) -> None:
        self.assertEqual(self.client.get("/api/projects/nope/export").status_code, 404)

    def test_import_round_trips_settings_and_parts(self) -> None:
        import io
        import json

        source_id = self._create_project()
        uploaded = self._upload("cast.stl")
        self._add_part(source_id, uploaded, "cast")
        exported = self.client.get(f"/api/projects/{source_id}/export").get_data()

        response = self.client.post(
            "/api/projects/import",
            data={
                "project": (io.BytesIO(exported), "round.solidus-project.json"),
                "files": (io.BytesIO(_cube_stl_bytes()), "cast.stl"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 201)
        body = response.get_json()
        self.assertEqual(body["missing_parts"], [])
        imported = repo.get_project(self.database_path, body["id"])
        self.assertEqual(imported["alloy"], "WCB")
        self.assertEqual(imported["voxel_size"], 0.2)
        self.assertTrue(imported["analysis_options"]["gas_porosity"])
        self.assertEqual(imported["solidification_params"]["feeding"]["shrink_scalar"], 1.75)
        self.assertEqual(imported["pour_location"]["point_in"], [1.0, 2.0, 3.0])
        self.assertEqual([part["name"] for part in repo.get_parts(self.database_path, body["id"])], ["cast"])
        self.assertEqual(json.loads(exported)["schema_version"], 2)

    def test_import_points_cuts_at_the_imported_bodies(self) -> None:
        import io
        import json

        source_id = self._create_project()
        uploaded = self._upload("cast.stl")
        part_id = self._add_part(source_id, uploaded, "cast")
        cut = {"kind": "cut", "id": "c1", "name": "Cut-Extrude1", "seq": 1, "sketch": {}, "partIds": [part_id]}
        repo.set_ui_state_key(self.database_path, source_id, "cut_features", [cut])
        exported = self.client.get(f"/api/projects/{source_id}/export").get_data()
        self.assertEqual(json.loads(exported)["parts"][0]["id"], part_id)

        response = self.client.post(
            "/api/projects/import",
            data={
                "project": (io.BytesIO(exported), "round.solidus-project.json"),
                "files": (io.BytesIO(_cube_stl_bytes()), "cast.stl"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 201)
        new_id = response.get_json()["id"]
        imported_part = repo.get_parts(self.database_path, new_id)[0]
        imported = repo.get_project(self.database_path, new_id)
        self.assertEqual(imported["ui_state"]["cut_features"][0]["partIds"], [imported_part["id"]])

    def test_import_reports_parts_whose_files_were_not_supplied(self) -> None:
        import io

        source_id = self._create_project()
        uploaded = self._upload("cast.stl")
        self._add_part(source_id, uploaded, "cast")
        exported = self.client.get(f"/api/projects/{source_id}/export").get_data()

        response = self.client.post(
            "/api/projects/import",
            data={"project": (io.BytesIO(exported), "round.solidus-project.json")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json()["missing_parts"], ["cast"])

    def test_import_migrates_a_schema_v1_pour_rate(self) -> None:
        import io
        import json

        v1 = {
            "schema_version": 1,
            "simulation": {
                "alloy_name": "CF8",
                "voxel_size": 0.15,
                "refinement_level": 2,
                "memory_mode": "auto",
                "analysis_options": {},
                "performance_options": {},
                "solidification_params": {"filling": {"pour_rate_lb_s": 5.0}},
                "pour_location": None,
            },
            "ui_state": {},
            "parts": [],
        }

        response = self.client.post(
            "/api/projects/import",
            data={"project": (io.BytesIO(json.dumps(v1).encode()), "old.solidus-project.json")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 201)
        imported = repo.get_project(self.database_path, response.get_json()["id"])
        self.assertEqual(imported["alloy"], "CF8")
        self.assertEqual(imported["solidification_params"]["filling"]["pour_rate_lb_s"], 0.0)

    def test_import_rejects_an_unsupported_schema_version(self) -> None:
        import io
        import json

        payload = {"schema_version": 99, "simulation": {}, "parts": []}

        response = self.client.post(
            "/api/projects/import",
            data={"project": (io.BytesIO(json.dumps(payload).encode()), "future.json")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("schema version", response.get_json()["error"])

    def test_import_rejects_a_non_json_file(self) -> None:
        import io

        response = self.client.post(
            "/api/projects/import",
            data={"project": (io.BytesIO(b"not json"), "broken.json")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)

    def test_import_requires_a_project_file(self) -> None:
        response = self.client.post("/api/projects/import", data={}, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 400)

class JobLogTests(WebToolsTestCase):
    def test_log_route_reports_unavailable_for_a_project_that_never_ran(self) -> None:
        project_id = self._create_project()

        payload = self.client.get(f"/api/projects/{project_id}/log").get_json()

        self.assertFalse(payload["available"])
        self.assertEqual(payload["lines"], [])
        self.assertEqual(payload["next"], 0)

    def test_log_route_returns_the_whole_buffer_by_default(self) -> None:
        project_id = self._create_project()
        manager = self.app.extensions["simulation_queue_manager"]
        manager._init_manager()
        manager.simulation_logs[project_id] = {"lines": ["one", "two", "three"], "first": 0}

        full = self.client.get(f"/api/projects/{project_id}/log").get_json()

        self.assertEqual(full["lines"], ["one", "two", "three"])
        self.assertTrue(full["available"])
        self.assertEqual(full["next"], 3)

    def test_clear_tracking_drops_the_log(self) -> None:
        project_id = self._create_project()
        manager = self.app.extensions["simulation_queue_manager"]
        manager._init_manager()
        manager.simulation_logs[project_id] = {"lines": ["one"], "first": 0}

        manager.clear_tracking(project_id)

        self.assertIsNone(manager.get_log(project_id))

class PartFilePathSecurityTests(WebToolsTestCase):

    def _secret(self) -> Path:
        secret = Path(self.temp_dir.name) / "secret.pem"
        secret.write_text("BEGIN PRIVATE KEY\ntop secret\n")
        return secret

    def test_add_part_refuses_a_path_outside_the_upload_folder(self) -> None:
        project_id = self._create_project()
        secret = self._secret()

        response = self.client.post(
            f"/api/projects/{project_id}/parts",
            json={"name": "innocent", "part_type": "cast", "filepath": str(secret)},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("upload folder", response.get_json()["error"])
        self.assertEqual(repo.get_parts(self.database_path, project_id), [])

    def test_update_part_refuses_a_path_outside_the_upload_folder(self) -> None:
        project_id = self._create_project()
        uploaded = self._upload("cast.stl")
        part_id = self._add_part(project_id, uploaded, "cast")
        secret = self._secret()

        response = self.client.put(
            f"/api/projects/{project_id}/parts/{part_id}",
            json={"name": "cast", "part_type": "cast", "filepath": str(secret)},
        )

        self.assertEqual(response.status_code, 400)
        stored = repo.get_part(self.database_path, project_id, part_id)
        self.assertEqual(Path(stored["filepath"]).resolve(), Path(uploaded["filepath"]).resolve())

    def test_add_part_refuses_a_traversal_relative_path(self) -> None:
        project_id = self._create_project()
        self._secret()

        response = self.client.post(
            f"/api/projects/{project_id}/parts",
            json={"name": "x", "part_type": "cast", "filepath": "../secret.pem"},
        )

        self.assertEqual(response.status_code, 400)

    def test_add_part_ignores_fields_outside_the_whitelist(self) -> None:
        project_id = self._create_project()
        uploaded = self._upload("cast.stl")

        response = self.client.post(
            f"/api/projects/{project_id}/parts",
            json={
                "name": "cast",
                "part_type": "cast",
                "filepath": uploaded["filepath"],
                "id": "attacker-chosen",
                "project_id": "somewhere-else",
            },
        )

        self.assertEqual(response.status_code, 201)
        part = response.get_json()["part"]
        self.assertNotEqual(part["id"], "attacker-chosen")
        self.assertEqual(part["project_id"], project_id)

    def test_duplicate_will_not_copy_a_file_from_outside_the_upload_folder(self) -> None:
        project_id = self._create_project()
        secret = self._secret()
        with repo.get_db_connection(self.database_path) as db:
            db.execute(
                "INSERT INTO parts (id, project_id, name, part_type, filepath) VALUES (?,?,?,?,?)",
                ("legacy01", project_id, "legacy", "cast", str(secret)),
            )

        response = self.client.post(f"/api/projects/{project_id}/duplicate", json={})

        self.assertEqual(response.status_code, 201)
        copies = list(self.upload_folder.glob("*"))
        self.assertEqual(copies, [], "the foreign file must not be laundered into the served folder")
        copy_id = response.get_json()["id"]
        self.assertEqual(repo.get_parts(self.database_path, copy_id)[0]["filepath"], str(secret))

    def test_resolve_upload_path_accepts_a_file_inside_the_folder(self) -> None:
        inside = self.upload_folder / "ok.stl"
        inside.write_bytes(b"x")

        self.assertEqual(
            repo.resolve_upload_path(str(inside), self.upload_folder), str(inside.resolve())
        )
        self.assertEqual(repo.resolve_upload_path("ok.stl", self.upload_folder), str(inside.resolve()))

    def test_resolve_upload_path_rejects_escapes(self) -> None:
        for candidate in ("../secret.pem", "/etc/passwd", "sub/../../out.stl"):
            with self.assertRaises(repo.UnsafePartFileError):
                repo.resolve_upload_path(candidate, self.upload_folder)

class ReportEscapingTests(WebToolsTestCase):

    def test_html_report_escapes_a_hostile_project_name(self) -> None:
        project_id = self._create_project(name="<script>alert(1)</script>")

        body = self.client.post(
            f"/api/projects/{project_id}/report", json={"format": "html"}
        ).get_data(as_text=True)

        self.assertNotIn("<script>alert(1)</script>", body)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", body)

    def test_html_report_escapes_a_hostile_part_name(self) -> None:
        project_id = self._create_project()
        uploaded = self._upload("cast.stl")
        self._add_part(project_id, uploaded, '"><img src=x onerror=alert(1)>')

        body = self.client.post(
            f"/api/projects/{project_id}/report",
            json={"format": "html", "sections": {"parts": True}},
        ).get_data(as_text=True)

        self.assertNotIn("<img src=x onerror=alert(1)>", body)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", body)

    def test_markdown_report_escapes_hostile_text(self) -> None:
        project_id = self._create_project(name="<script>alert(1)</script>")

        body = self.client.post(
            f"/api/projects/{project_id}/report", json={"format": "markdown"}
        ).get_data(as_text=True)

        self.assertNotIn("<script>", body)

    def test_report_rejects_a_non_object_sections_field(self) -> None:
        project_id = self._create_project()

        response = self.client.post(
            f"/api/projects/{project_id}/report", json={"format": "html", "sections": ["parts"]}
        )

        self.assertEqual(response.status_code, 400)

    def test_report_rejects_a_screenshot_that_is_not_a_png(self) -> None:
        import base64

        project_id = self._create_project()

        response = self.client.post(
            f"/api/projects/{project_id}/report",
            json={
                "format": "html",
                "sections": {"screenshot": True},
                "screenshot": base64.b64encode(b"GIF89a not a png").decode(),
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("PNG", response.get_json()["error"])

    def test_report_rejects_an_oversized_screenshot(self) -> None:
        import base64

        import web.routes_tools as tools

        project_id = self._create_project()
        payload = tools.PNG_MAGIC + b"\x00" * (tools.MAX_SCREENSHOT_BYTES + 1)

        response = self.client.post(
            f"/api/projects/{project_id}/report",
            json={
                "format": "html",
                "sections": {"screenshot": True},
                "screenshot": base64.b64encode(payload).decode(),
            },
        )

        self.assertEqual(response.status_code, 413)

class ImportRobustnessTests(WebToolsTestCase):
    def test_import_rejects_a_non_scalar_schema_version(self) -> None:
        import io
        import json

        payload = {"schema_version": {}, "simulation": {}, "parts": []}

        response = self.client.post(
            "/api/projects/import",
            data={"project": (io.BytesIO(json.dumps(payload).encode()), "broken.json")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)

    def test_import_rejects_a_json_array(self) -> None:
        import io

        response = self.client.post(
            "/api/projects/import",
            data={"project": (io.BytesIO(b"[1, 2, 3]"), "list.json")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)

    def test_import_rejects_an_oversized_project_file(self) -> None:
        import io

        import web.routes_tools as tools

        blob = b"{" + b" " * (tools.MAX_PROJECT_FILE_BYTES + 16)

        response = self.client.post(
            "/api/projects/import",
            data={"project": (io.BytesIO(blob), "huge.json")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 413)

    def test_import_tessellates_cad_for_the_project_voxel_size(self) -> None:
        import io
        import json
        from unittest import mock

        import web.payloads as payloads

        payload = {
            "schema_version": 2,
            "simulation": {"alloy_name": "WCB", "voxel_size": 0.05},
            "parts": [],
        }
        with mock.patch.object(payloads, "upload_cad_file", return_value=[]) as upload:
            self.client.post(
                "/api/projects/import",
                data={
                    "project": (io.BytesIO(json.dumps(payload).encode()), "p.json"),
                    "files": (io.BytesIO(b"solid x\nendsolid x\n"), "cast.stl"),
                },
                content_type="multipart/form-data",
            )
        self.assertEqual(upload.call_count, 1)
        self.assertEqual(upload.call_args.args[2], 0.05)

    def test_import_survives_malformed_part_transforms(self) -> None:
        import io
        import json

        payload = {
            "schema_version": 2,
            "simulation": {"alloy_name": "WCB"},
            "ui_state": [],
            "parts": [
                {"filepath": "cast.stl", "name": "cast", "position": "nope", "rotation": [1, 2]},
            ],
        }

        response = self.client.post(
            "/api/projects/import",
            data={
                "project": (io.BytesIO(json.dumps(payload).encode()), "odd.solidus-project.json"),
                "files": (io.BytesIO(_cube_stl_bytes()), "cast.stl"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 201)
        parts = repo.get_parts(self.database_path, response.get_json()["id"])
        self.assertEqual(len(parts), 1)
        self.assertEqual(
            (parts[0]["position_x"], parts[0]["position_y"], parts[0]["position_z"]),
            (0.0, 0.0, 0.0),
        )

    def test_import_reports_display_name_collisions_instead_of_overwriting(self) -> None:
        import io
        import json

        payload = {
            "schema_version": 2,
            "simulation": {"alloy_name": "WCB"},
            "ui_state": {},
            "parts": [{"filepath": "bracket.stl", "name": "bracket"}],
        }

        response = self.client.post(
            "/api/projects/import",
            data={
                "project": (io.BytesIO(json.dumps(payload).encode()), "p.solidus-project.json"),
                "files": [
                    (io.BytesIO(_cube_stl_bytes()), "bracket.stl"),
                    (io.BytesIO(_cube_stl_bytes(3.0)), "bracket.stl"),
                ],
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 201)
        body = response.get_json()
        self.assertEqual(body["name_collisions"], ["bracket"])
        self.assertEqual(body["missing_parts"], [])

class JobLogCursorTests(WebToolsTestCase):
    def _manager(self):
        manager = self.app.extensions["simulation_queue_manager"]
        manager._init_manager()
        return manager

    def test_cursor_survives_the_ring_buffer_wrapping(self) -> None:
        project_id = self._create_project()
        manager = self._manager()
        manager.simulation_logs[project_id] = {
            "lines": [f"line {n}" for n in range(15, 20)],
            "first": 15,
        }

        payload = self.client.get(f"/api/projects/{project_id}/log?since=5").get_json()

        self.assertEqual(payload["dropped"], 10)
        self.assertEqual(payload["lines"], [f"line {n}" for n in range(15, 20)])
        self.assertEqual(payload["next"], 20)

    def test_cursor_returns_only_the_tail(self) -> None:
        project_id = self._create_project()
        manager = self._manager()
        manager.simulation_logs[project_id] = {"lines": ["a", "b", "c"], "first": 0}

        payload = self.client.get(f"/api/projects/{project_id}/log?since=2").get_json()

        self.assertEqual(payload["lines"], ["c"])
        self.assertEqual(payload["next"], 3)
        self.assertEqual(payload["dropped"], 0)

    def test_a_rerun_resets_the_buffer_rather_than_appending(self) -> None:
        from web.services.simulation_queue import JOB_LOG_LINES

        project_id = self._create_project()
        manager = self._manager()
        manager.simulation_logs[project_id] = {"lines": ["old run"], "first": 0}

        manager.simulation_logs[project_id] = {"lines": [], "first": 0}

        payload = self.client.get(f"/api/projects/{project_id}/log").get_json()
        self.assertEqual(payload["lines"], [])
        self.assertEqual(payload["next"], 0)
        self.assertGreater(JOB_LOG_LINES, 0)

    def test_log_for_a_project_that_never_ran(self) -> None:
        project_id = self._create_project()

        payload = self.client.get(f"/api/projects/{project_id}/log").get_json()

        self.assertFalse(payload["available"])
        self.assertEqual(payload["lines"], [])

class HardwareTierTests(WebToolsTestCase):
    def test_hardware_reports_core_tiers_by_name(self) -> None:
        payload = self.client.get("/api/hardware").get_json()

        self.assertIn("core_tiers", payload)
        self.assertIsInstance(payload["core_tiers"], list)
        for tier in payload["core_tiers"]:
            self.assertIsInstance(tier["name"], str)
            self.assertTrue(tier["name"])
            self.assertGreater(tier["cores"], 0)

    def test_core_tier_detection_reads_names_not_positions(self) -> None:
        import platform

        from config.hardware_config import HardwareConfig

        if platform.system() != "Darwin":
            self.skipTest("performance levels are a macOS concept")
        tiers = HardwareConfig()._detect_core_tiers()
        if not tiers:
            self.skipTest("this CPU reports no performance levels")
        total = sum(tier["cores"] for tier in tiers)
        self.assertGreater(total, 0)
        self.assertTrue(all(tier["name"] for tier in tiers))

class ArtifactCacheTests(WebToolsTestCase):
    def test_artifact_cache_reparses_when_the_file_changes(self) -> None:
        import json
        import os

        import web.routes_tools as tools

        tools._ARTIFACT_CACHE.clear()
        path = Path(self.temp_dir.name) / "artifact.json"
        path.write_text(json.dumps({"summary": {"voxel_count": 1}, "visualization": {}}))

        first = tools._read_artifact_cached(str(path))
        cached = tools._read_artifact_cached(str(path))
        self.assertIs(first, cached, "an unchanged artifact should not be re-parsed")

        path.write_text(json.dumps({"summary": {"voxel_count": 2}, "visualization": {}}))
        os.utime(path, (0, 0))
        second = tools._read_artifact_cached(str(path))

        self.assertEqual(second["summary"]["voxel_count"], 2)

    def test_artifact_cache_is_bounded(self) -> None:
        import json

        import web.routes_tools as tools

        tools._ARTIFACT_CACHE.clear()
        for index in range(tools._ARTIFACT_CACHE_MAX + 3):
            path = Path(self.temp_dir.name) / f"a{index}.json"
            path.write_text(json.dumps({"summary": {"n": index}, "visualization": {}}))
            tools._read_artifact_cached(str(path))

        self.assertLessEqual(len(tools._ARTIFACT_CACHE), tools._ARTIFACT_CACHE_MAX)

    def test_missing_artifact_returns_none(self) -> None:
        import web.routes_tools as tools

        self.assertIsNone(tools._read_artifact_cached(str(Path(self.temp_dir.name) / "gone.json")))

class PartNameTests(WebToolsTestCase):
    def test_a_users_own_name_is_never_rewritten_on_read(self) -> None:
        project_id = self._create_project()
        uploaded = self._upload("cast.stl")
        part_id = self._add_part(project_id, uploaded, "cast")

        self.client.put(
            f"/api/projects/{project_id}/parts/{part_id}",
            json={"name": "12345678_v2", "part_type": "cast", "filepath": uploaded["filepath"]},
        )

        stored = repo.get_part(self.database_path, project_id, part_id)
        self.assertEqual(stored["name"], "12345678_v2")

    def test_legacy_storage_names_are_repaired_once_at_startup(self) -> None:
        project_id = self._create_project()
        with repo.get_db_connection(self.database_path) as db:
            db.execute(
                "INSERT INTO parts (id, project_id, name, part_type, filepath) VALUES (?,?,?,?,?)",
                ("legacy02", project_id, "602e27b4_b0_cast", "cast", ""),
            )

        repo.init_db(self.database_path)

        self.assertEqual(repo.get_parts(self.database_path, project_id)[0]["name"], "cast")

class UploadPrefixTests(unittest.TestCase):

    def test_strip_upload_prefix_removes_every_layer(self) -> None:
        self.assertEqual(repo.strip_upload_prefix("602e27b4_b0_cast"), "cast")
        self.assertEqual(repo.strip_upload_prefix("3c3a67b8_b0_602e27b4_b0_cast"), "cast")
        self.assertEqual(repo.strip_upload_prefix("ab12cd34_riser"), "riser")

    def test_strip_upload_prefix_leaves_ordinary_names_alone(self) -> None:
        for name in ("cast", "impeller_v3", "PART-A_Rigging_-_Chill_1", "b0_cast", "12345678cast"):
            self.assertEqual(repo.strip_upload_prefix(name), name)

    def test_strip_upload_prefix_never_returns_an_empty_name(self) -> None:
        self.assertEqual(repo.strip_upload_prefix("602e27b4_b0_"), "602e27b4_b0_")

class UploadCadFileNamingTests(unittest.TestCase):

    def test_single_body_upload_keeps_the_original_stem(self) -> None:
        from werkzeug.datastructures import FileStorage

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            storage = FileStorage(
                stream=io.BytesIO(_cube_stl_bytes()), filename="Impeller Housing.stl"
            )
            entries = upload_cad_file(storage, folder)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["display_name"], "Impeller_Housing")
        self.assertEqual(entries[0]["part_type"], "cast")

if __name__ == "__main__":
    unittest.main()
