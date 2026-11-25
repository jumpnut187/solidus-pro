from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import trimesh
from flask import Flask

from web import fields, repository as repo, skin
from web.payloads import write_artifact
from web.routes_fields import fields_bp
from web.services.simulation_queue import SimulationQueueManager

SPACING = 0.5
ORIGIN = (-2.0, -2.0, -2.0)
DIMS = (9, 9, 9)
HALF = 1.3

def _box() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(2 * HALF, 2 * HALF, 2 * HALF))

def _lattice() -> fields.Lattice:
    return fields.Lattice(dims=DIMS, origin=ORIGIN, spacing=SPACING)

def _centers() -> np.ndarray:
    nx, ny, nz = DIMS
    iz, iy, ix = np.mgrid[0:nz, 0:ny, 0:nx]
    return np.asarray(ORIGIN) + np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1) * SPACING

def _exact_distance(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    _, distance, _ = trimesh.proximity.closest_point(mesh, points)
    inside = mesh.contains(points)
    return np.where(inside, -distance, distance) / SPACING

def _inclusive_mask() -> np.ndarray:
    centers = _centers()
    touches = np.all(np.abs(centers) - SPACING / 2 <= HALF + 1e-9, axis=1)
    return touches.astype(np.uint8)

class SkinGridBuild(unittest.TestCase):
    def setUp(self) -> None:
        self.lattice = _lattice()
        self.mask = _inclusive_mask()
        self.values = skin.build_skin_grid(self.lattice, self.mask, [_box()])
        self.assertIsNotNone(self.values)
        assert self.values is not None
        self.distance = skin.dequantize(self.values)
        self.centers = _centers()
        self.exact = _exact_distance(_box(), self.centers)

    def test_shape_and_dtype(self) -> None:
        self.assertEqual(self.values.dtype, np.uint8)
        self.assertEqual(self.values.shape, (int(np.prod(DIMS)),))

    def test_mask_overhangs_the_box(self) -> None:
        outside_but_metal = (self.mask != 0) & (self.exact > 0)
        self.assertGreater(int(np.count_nonzero(outside_but_metal)), 0)

    def test_overhanging_metal_reads_outside(self) -> None:
        outside_but_metal = (self.mask != 0) & (self.exact > 0)
        self.assertTrue(np.all(self.distance[outside_but_metal] > 0))
        self.assertTrue(np.all(self.values[outside_but_metal] > 127))

    def test_sign_matches_geometry_everywhere(self) -> None:
        clear = np.abs(self.exact) > 1e-6
        self.assertTrue(np.all((self.distance[clear] < 0) == (self.exact[clear] < 0)))

    def test_distance_is_accurate_near_the_surface(self) -> None:
        near = np.abs(self.exact) < 1.5
        quantum = 2 * skin.SKIN_RANGE_VOXELS / 255
        error = np.abs(self.distance[near] - np.clip(self.exact[near], -2, 2))
        self.assertLess(float(error.max()), quantum + 1e-6)

    def test_far_cells_take_the_range_limit(self) -> None:
        self.assertEqual(int(self.values[0]), 255)
        center = int(np.argmin(np.sum(self.centers**2, axis=1)))
        self.assertEqual(int(self.values[center]), 0)

    def test_quantize_round_trips_the_surface(self) -> None:
        self.assertGreater(int(skin.quantize(np.asarray([0.0]))[0]), 127)
        self.assertLess(int(skin.quantize(np.asarray([-0.01]))[0]), 128)
        self.assertEqual(int(skin.quantize(np.asarray([-5.0]))[0]), 0)
        self.assertEqual(int(skin.quantize(np.asarray([5.0]))[0]), 255)

    def test_no_meshes_is_none(self) -> None:
        self.assertIsNone(skin.build_skin_grid(self.lattice, self.mask, []))

    def test_overlapping_parts_are_a_union(self) -> None:
        shifted = _box().copy()
        shifted.apply_translation((HALF, 0.0, 0.0))
        values = skin.build_skin_grid(self.lattice, self.mask, [_box(), shifted])
        assert values is not None
        distance = skin.dequantize(values)
        both = (np.abs(self.centers[:, 1]) < HALF - 0.3) & (np.abs(self.centers[:, 2]) < HALF - 0.3)
        both &= (self.centers[:, 0] > 0.3) & (self.centers[:, 0] < HALF - 0.3)
        self.assertGreater(int(np.count_nonzero(both)), 0)
        self.assertTrue(np.all(distance[both] < 0))

class SkinGridCaching(unittest.TestCase):
    def setUp(self) -> None:
        skin.clear_cache()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.stl = self.root / "box.stl"
        _box().export(str(self.stl))
        self.parts: List[Dict[str, Any]] = [{"filepath": str(self.stl), "part_type": "cast", "name": "box"}]
        self.lattice = _lattice()
        self.mask = _inclusive_mask()
        self.artifact = self.root / "run.json"
        self.artifact.write_text("{}", encoding="utf-8")

    def tearDown(self) -> None:
        skin.clear_cache()
        self.temp_dir.cleanup()

    def test_signature_changes_with_placement(self) -> None:
        base = skin.parts_signature(self.lattice, self.parts)
        moved = skin.parts_signature(self.lattice, [{**self.parts[0], "position_x": 0.5}])
        self.assertIsNotNone(base)
        self.assertNotEqual(base, moved)

    def test_signature_is_none_without_files(self) -> None:
        self.assertIsNone(skin.parts_signature(self.lattice, [{"filepath": str(self.root / "missing.stl")}]))
        self.assertIsNone(skin.parts_signature(self.lattice, []))

    def test_builds_then_reads_the_sidecar(self) -> None:
        first = skin.load_skin_grid("proj", self.lattice, self.mask, self.parts, self.artifact)
        self.assertIsNotNone(first)
        sidecar = skin.skin_path(self.artifact)
        self.assertTrue(sidecar.exists())
        skin.clear_cache()
        second = skin.load_skin_grid("proj", self.lattice, self.mask, self.parts, self.artifact)
        assert first is not None and second is not None
        np.testing.assert_array_equal(first.values, second.values)

    def test_stale_sidecar_is_rebuilt(self) -> None:
        first = skin.load_skin_grid("proj", self.lattice, self.mask, self.parts, self.artifact)
        assert first is not None
        skin.clear_cache()
        moved = [{**self.parts[0], "position_x": 1.0}]
        second = skin.load_skin_grid("proj", self.lattice, self.mask, moved, self.artifact)
        assert second is not None
        self.assertNotEqual(first.signature, second.signature)
        self.assertFalse(np.array_equal(first.values, second.values))

    def test_placement_is_honoured(self) -> None:
        moved = [{**self.parts[0], "position_x": 1.0}]
        grid = skin.load_skin_grid("proj", self.lattice, self.mask, moved, self.artifact)
        assert grid is not None
        distance = skin.dequantize(grid.values)
        centers = _centers()
        index = int(np.argmin(np.sum((centers - np.asarray([-1.0, 0.0, 0.0])) ** 2, axis=1)))
        self.assertGreater(distance[index], 0)

def _synthetic_artifact() -> Tuple[Dict[str, Any], np.ndarray]:
    mask = _inclusive_mask()
    centers = _centers()[mask != 0]
    return (
        {
            "summary": {"voxel_count": int(len(centers)), "total_volume": 1.0},
            "visualization": {
                "centers": centers.tolist(),
                "sizes": [SPACING] * len(centers),
                "part_types": [0] * len(centers),
                "shrinkage": [0.1] * len(centers),
            },
        },
        mask,
    )

class SkinEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        fields.clear_cache()
        skin.clear_cache()
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.database_path = root / "simulations.db"
        self.temp_folder = root / "temp"
        self.temp_folder.mkdir()
        repo.init_db(self.database_path)

        app = Flask(__name__)
        app.config["DATABASE"] = self.database_path
        app.config["TEMP_FOLDER"] = self.temp_folder
        app.extensions["simulation_queue_manager"] = SimulationQueueManager()
        app.register_blueprint(fields_bp)
        self.client = app.test_client()

        artifact, self.mask = _synthetic_artifact()
        self.project_id = repo.create_project(self.database_path, {"name": "Skin"})
        artifact_path = self.temp_folder / f"{self.project_id}.json"
        write_artifact(artifact_path, artifact)
        repo.mark_project_completed(self.database_path, self.project_id, artifact["summary"], str(artifact_path))
        self.stl = root / "box.stl"
        _box().export(str(self.stl))

    def tearDown(self) -> None:
        fields.clear_cache()
        skin.clear_cache()
        self.temp_dir.cleanup()

    def test_404_without_part_files(self) -> None:
        response = self.client.get(f"/api/projects/{self.project_id}/grid/skin")
        self.assertEqual(response.status_code, 404)

    def test_grid_with_parts(self) -> None:
        repo.add_part(
            self.database_path,
            self.project_id,
            {"name": "box", "part_type": "cast", "filepath": str(self.stl)},
            upload_root=self.stl.parent,
        )
        response = self.client.get(f"/api/projects/{self.project_id}/grid/skin")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Grid-Dims"], "7,7,7")
        self.assertEqual(response.headers["X-Grid-Dtype"], "uint8")
        self.assertEqual(float(response.headers["X-Skin-Range-Voxels"]), skin.SKIN_RANGE_VOXELS)
        values = np.frombuffer(response.data, dtype=np.uint8)
        self.assertEqual(values.shape, (7 * 7 * 7,))
        iz, iy, ix = np.mgrid[0:7, 0:7, 0:7]
        centers = -1.5 + np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1) * SPACING
        exact = _exact_distance(_box(), centers)
        self.assertGreater(int(np.count_nonzero(exact > 0)), 0)
        self.assertTrue(np.all(values[exact > 0] > 127))
        self.assertTrue(np.all(values[exact < -0.5] < 128))

    def test_unknown_project_is_404(self) -> None:
        response = self.client.get("/api/projects/nope/grid/skin")
        self.assertEqual(response.status_code, 404)

if __name__ == "__main__":
    unittest.main()
