"""Tests for the application-shell server routes.

Covers the endpoints the web shell added for project management, hardware
information, the help guide, part-geometry restore, and the upload-naming fix.
"""

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
    """A small watertight STL, as an upload would arrive."""
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

    # -- helpers ---------------------------------------------------------

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

    # -- upload naming ---------------------------------------------------

    def test_upload_reports_a_display_name_without_the_storage_prefix(self) -> None:
        entry = self._upload("b0_cast.stl")

        self.assertEqual(entry["display_name"], "b0_cast")
        # The stored filename still carries exactly one prefix for collisions.
        self.assertTrue(entry["filename"].endswith("_b0_b0_cast.stl"))

    def test_re_uploading_a_stored_file_does_not_compound_prefixes(self) -> None:
        first = self._upload("cast.stl")
        # Re-uploading the file as it sits on disk is what produced names like
        # "3c3a67b8_b0_602e27b4_b0_cast" before the fix.
        second = self._upload(Path(first["filepath"]).name)

        self.assertEqual(first["display_name"], "cast")
        self.assertEqual(second["display_name"], "cast")
        self.assertNotEqual(first["filepath"], second["filepath"])

    def test_stored_part_names_lose_a_legacy_prefix_on_read(self) -> None:
        project_id = self._create_project()
        repo.add_part(
            self.database_path,
            project_id,
            {"name": "602e27b4_b0_cast", "part_type": "cast", "filepath": "x.stl"},
        )

        parts = repo.get_parts(self.database_path, project_id)

        self.assertEqual(parts[0]["name"], "cast")

    # -- part geometry restore -------------------------------------------

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

    # -- rename ----------------------------------------------------------

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

    # -- duplicate -------------------------------------------------------

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
        # A copy is a fresh setup to re-run, not a second row pointing at an
        # artifact it did not produce.
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

        # Deleting the copy must not take the original's geometry with it,
        # which is exactly what a shared filepath would have caused.
        for filepath in repo.delete_project(self.database_path, copy_id):
            Path(filepath).unlink(missing_ok=True)
        self.assertTrue(Path(original_path).exists())

    def test_duplicate_project_404s_for_an_unknown_project(self) -> None:
        response = self.client.post("/api/projects/nope/duplicate", json={})

        self.assertEqual(response.status_code, 404)

    # -- reference data --------------------------------------------------

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

        # `use_gpu` is read by the simulation queue; before the shell rewrite it
        # had no default and no control anywhere on the web side.
        self.assertIn("use_gpu", project["performance_options"])
        self.assertTrue(project["performance_options"]["use_gpu"])


class ReportRouteTests(WebToolsTestCase):
    """Report generation, including the formats the server cannot produce."""

    def test_report_formats_flags_pdf_availability_with_a_reason(self) -> None:
        payload = self.client.get("/api/report/formats").get_json()

        by_id = {entry["id"]: entry for entry in payload["formats"]}
        self.assertEqual(set(by_id), {"html", "markdown", "pdf"})
        self.assertTrue(by_id["html"]["available"])
        self.assertTrue(by_id["markdown"]["available"])
        # PDF depends on an optional native stack; whichever way it goes, an
        # unavailable format must carry a reason the UI can show.
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

        # Smallest valid PNG: an 1x1 opaque pixel.
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
    """Import and export of the desktop ``.solidus-project.json`` format."""

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
        # The desktop nests settings and calls the alloy `alloy_name`.
        self.assertEqual(payload["simulation"]["alloy_name"], "WCB")
        self.assertEqual(payload["simulation"]["pour_location"]["point_in"], [1.0, 2.0, 3.0])
        self.assertEqual(len(payload["parts"]), 1)
        # A server path means nothing on another machine; the importer matches
        # parts to uploaded files by name.
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
        # pour_location lives in ui_state on the web side and is mirrored out.
        self.assertEqual(imported["pour_location"]["point_in"], [1.0, 2.0, 3.0])
        self.assertEqual([part["name"] for part in repo.get_parts(self.database_path, body["id"])], ["cast"])
        self.assertEqual(json.loads(exported)["schema_version"], 2)

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
        # Named, not silently dropped: the user has to know what to re-attach.
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
                # 5.0 was the legacy default; v2 means "derive it" with 0.0.
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


class RiserDesignRouteTests(WebToolsTestCase):
    """The riser endpoints, without paying for a real solve."""

    def test_riser_design_404s_without_results(self) -> None:
        project_id = self._create_project()

        response = self.client.post(f"/api/projects/{project_id}/riser-design", json={})

        self.assertEqual(response.status_code, 404)

    def test_riser_design_rejects_an_artifact_without_a_surface_mask(self) -> None:
        from unittest.mock import patch

        project_id = self._create_project()
        artifact = {"summary": {}, "visualization": {"centers": [[0, 0, 0]], "sizes": [1.0]}}

        with patch("web.routes_tools._load_artifact", return_value=artifact):
            response = self.client.post(f"/api/projects/{project_id}/riser-design", json={})

        self.assertEqual(response.status_code, 400)
        self.assertIn("surface mask", response.get_json()["error"])

    def test_riser_design_rejects_a_downsampled_artifact(self) -> None:
        from unittest.mock import patch

        project_id = self._create_project()
        artifact = {
            # The stored count exceeds what was kept, so the surface set has
            # holes and every modulus would come out too large.
            "summary": {"voxel_count": 5000},
            "visualization": {
                "centers": [[0, 0, 0], [1, 0, 0]],
                "sizes": [1.0, 1.0],
                "surface_voxels": [True, True],
            },
        }

        with patch("web.routes_tools._load_artifact", return_value=artifact):
            response = self.client.post(f"/api/projects/{project_id}/riser-design", json={})

        self.assertEqual(response.status_code, 400)
        self.assertIn("downsampled", response.get_json()["error"])

    def test_riser_design_refuses_a_grid_beyond_the_inline_limit(self) -> None:
        from unittest.mock import patch

        import web.routes_tools as tools

        project_id = self._create_project()
        count = tools.RISER_MAX_VOXELS + 1
        artifact = {
            "summary": {"voxel_count": count},
            "visualization": {
                "centers": [[float(i), 0.0, 0.0] for i in range(count)],
                "sizes": [1.0] * count,
                "surface_voxels": [True] * count,
            },
        }

        with patch("web.routes_tools._load_artifact", return_value=artifact):
            response = self.client.post(f"/api/projects/{project_id}/riser-design", json={})

        self.assertEqual(response.status_code, 400)
        self.assertIn("riser-design limit", response.get_json()["error"])

    def test_apply_creates_riser_parts_with_geometry_on_disk(self) -> None:
        project_id = self._create_project()

        response = self.client.post(
            f"/api/projects/{project_id}/riser-design/apply",
            json={
                "risers": [
                    {
                        "diameter_in": 4.0,
                        "height_in": 6.0,
                        "position": [1.0, 2.0, 3.0],
                        "riser_type": "BLIND",
                    }
                ]
            },
        )

        self.assertEqual(response.status_code, 201)
        body = response.get_json()
        self.assertEqual(body["added"], 1)
        part = body["parts"][0]
        self.assertEqual(part["part_type"], "riser")
        self.assertTrue(Path(part["filepath"]).exists())

        mesh = trimesh.load(part["filepath"], force="mesh")
        # 40 percent of a diameter of penetration, then half the height up:
        # 3.0 - 1.6 + 3.0 = 4.4 for the centre of a 6 inch cylinder.
        self.assertAlmostEqual(float(mesh.centroid[2]), 4.4, places=3)
        self.assertAlmostEqual(float(mesh.bounds[1][2] - mesh.bounds[0][2]), 6.0, places=3)

    def test_apply_marks_an_open_riser_as_open_top(self) -> None:
        project_id = self._create_project()

        response = self.client.post(
            f"/api/projects/{project_id}/riser-design/apply",
            json={
                "risers": [
                    {"diameter_in": 3.0, "height_in": 4.0, "position": [0, 0, 0], "riser_type": "OPEN"}
                ]
            },
        )

        self.assertTrue(response.get_json()["parts"][0]["is_open_top"])

    def test_apply_rejects_an_empty_or_malformed_selection(self) -> None:
        project_id = self._create_project()

        self.assertEqual(
            self.client.post(f"/api/projects/{project_id}/riser-design/apply", json={"risers": []}).status_code,
            400,
        )
        self.assertEqual(
            self.client.post(
                f"/api/projects/{project_id}/riser-design/apply", json={"risers": [{"diameter_in": 1.0}]}
            ).status_code,
            400,
        )

    def test_dedupe_keeps_the_most_confident_of_coincident_risers(self) -> None:
        from types import SimpleNamespace

        from web.routes_tools import _dedupe_recommendations

        far = SimpleNamespace(position=(10.0, 0.0, 0.0), diameter=6.0, confidence=0.2)
        weak = SimpleNamespace(position=(0.0, 0.0, 0.0), diameter=6.0, confidence=0.3)
        strong = SimpleNamespace(position=(0.05, 0.0, 0.0), diameter=6.0, confidence=0.9)

        kept, merged = _dedupe_recommendations([weak, strong, far], voxel_size=0.2)

        self.assertEqual(merged, 1)
        self.assertEqual([item.confidence for item in kept], [0.9, 0.2])


class JobLogTests(WebToolsTestCase):
    def test_log_route_reports_unavailable_for_a_project_that_never_ran(self) -> None:
        project_id = self._create_project()

        payload = self.client.get(f"/api/projects/{project_id}/log").get_json()

        self.assertFalse(payload["available"])
        self.assertEqual(payload["lines"], [])
        self.assertEqual(payload["total"], 0)

    def test_log_route_returns_only_the_tail_after_since(self) -> None:
        project_id = self._create_project()
        manager = self.app.extensions["simulation_queue_manager"]
        manager._init_manager()
        manager.simulation_logs[project_id] = ["one", "two", "three"]

        full = self.client.get(f"/api/projects/{project_id}/log").get_json()
        tail = self.client.get(f"/api/projects/{project_id}/log?since=2").get_json()

        self.assertEqual(full["lines"], ["one", "two", "three"])
        self.assertTrue(full["available"])
        self.assertEqual(tail["lines"], ["three"])
        self.assertEqual(tail["total"], 3)

    def test_clear_tracking_drops_the_log(self) -> None:
        project_id = self._create_project()
        manager = self.app.extensions["simulation_queue_manager"]
        manager._init_manager()
        manager.simulation_logs[project_id] = ["one"]

        manager.clear_tracking(project_id)

        self.assertIsNone(manager.get_log(project_id))


class UploadPrefixTests(unittest.TestCase):
    """Unit coverage for the naming rule itself."""

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
    """The single-body upload path, exercised without the Flask layer."""

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
