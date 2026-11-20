from __future__ import annotations

import gzip
import json
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flask import Flask

from core.simulation_pipeline import _add_summary_statistics
from web import fields
from web import repository as repo
from web.payloads import GRID_SOURCE_KEY, write_artifact
from web.routes_fields import clear_envelope_cache, fields_bp
from web.services.simulation_queue import SimulationQueueManager

DIMS = (6, 5, 4)
ORIGIN = (-1.25, 2.5, 0.75)
SPACING = 0.25
FRAME_COUNT = 3
FRAME_TIMES = [0.0, 4.5, 9.0]
FILL_TIMES = [0.0, 0.75, 1.5]

HOLES = {(4, 1, 0), (4, 2, 0), (1, 3, 2)}

REAL_ARTIFACT = (
    Path(__file__).resolve().parents[1] / "web" / "static" / "temp" / "artifacts" / "ef17b6fc-ac9.json"
)

def _cell_center(ix: int, iy: int, iz: int) -> List[float]:
    return [
        ORIGIN[0] + SPACING * ix,
        ORIGIN[1] + SPACING * iy,
        ORIGIN[2] + SPACING * iz,
    ]

def _occupied_cells() -> List[Tuple[int, int, int]]:
    nx, ny, nz = DIMS
    return [
        (ix, iy, iz)
        for iz in range(nz)
        for iy in range(ny)
        for ix in range(nx)
        if (ix, iy, iz) not in HOLES
    ]

def _shrinkage_value(ix: int, iy: int, iz: int) -> float:
    return 5.0 + ix + 10.0 * iy + 100.0 * iz

def _synthetic_artifact() -> Dict[str, Any]:
    cells = _occupied_cells()
    centers = [_cell_center(*cell) for cell in cells]
    shrinkage = [_shrinkage_value(*cell) for cell in cells]
    part_types = [0 if iz < 2 else 1 for (_, _, iz) in cells]
    solidification = [20.0 + 5.0 * iz for (_, _, iz) in cells]

    history: List[List[float]] = []
    for frame_index in range(FRAME_COUNT):
        remaining = 1.0 - frame_index / (FRAME_COUNT - 1)
        history.append([float(np.clip(remaining, 0.0, 1.0))] * len(cells))

    fill_history: List[List[float]] = []
    for frame_index in range(FRAME_COUNT):
        level = (frame_index / (FRAME_COUNT - 1)) * (DIMS[2] - 1)
        fill_history.append([1.0 if iz <= level else 0.0 for (_, _, iz) in cells])

    return {
        "summary": {
            "voxel_count": len(cells),
            "total_volume": 12.5,
            "shrinkage_stats": {"max": 5.0, "mean": 1.0, "affected_count": 3},
            "shrink_scalar": 1.0,
        },
        "visualization": {
            "centers": centers,
            "sizes": [SPACING] * len(cells),
            "part_types": part_types,
            "shrinkage": shrinkage,
            "solidification_times": solidification,
            "fraction_liquid_history": history,
            "fraction_liquid_times": FRAME_TIMES,
            "fill_animation_history": fill_history,
            "fill_animation_times": FILL_TIMES,
        },
    }

def _flat_index(ix: int, iy: int, iz: int) -> int:
    nx, ny, _ = DIMS
    return ix + nx * (iy + ny * iz)

def _brute_force_nearest_values(
    values_by_cell: Dict[Tuple[int, int, int], float],
    target: Tuple[int, int, int],
) -> List[float]:
    best = float("inf")
    tied: List[float] = []
    for cell, value in values_by_cell.items():
        distance = sum((a - b) ** 2 for a, b in zip(cell, target))
        if distance < best - 1e-9:
            best = distance
            tied = [value]
        elif abs(distance - best) <= 1e-9:
            tied.append(value)
    return tied

class LatticeDerivationTests(unittest.TestCase):
    def test_derives_dims_origin_and_spacing_from_centers(self) -> None:
        artifact = _synthetic_artifact()
        centers = np.asarray(artifact["visualization"]["centers"], dtype=np.float64)
        sizes = np.asarray(artifact["visualization"]["sizes"], dtype=np.float64)

        lattice = fields.derive_lattice(centers, sizes)

        self.assertEqual(lattice.dims, DIMS)
        self.assertAlmostEqual(lattice.spacing, SPACING, places=9)
        for actual, expected in zip(lattice.origin, ORIGIN):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(lattice.cell_count, DIMS[0] * DIMS[1] * DIMS[2])

    def test_dims_survive_an_entirely_empty_lattice_plane(self) -> None:
        cells = [cell for cell in _occupied_cells() if cell[2] != 1]
        centers = np.asarray([_cell_center(*cell) for cell in cells], dtype=np.float64)

        lattice = fields.derive_lattice(centers, np.full(len(cells), SPACING))

        self.assertEqual(lattice.dims, DIMS)

class GridBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.artifact = _synthetic_artifact()
        self.grids = fields.build_field_grids(
            self.artifact["visualization"], self.artifact["summary"]
        )

    def test_mask_holds_part_type_code_plus_one_and_zero_outside(self) -> None:
        mask = self.grids.mask
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask.size, DIMS[0] * DIMS[1] * DIMS[2])

        for ix, iy, iz in _occupied_cells():
            expected = 1 if iz < 2 else 2
            self.assertEqual(int(mask[_flat_index(ix, iy, iz)]), expected, msg=f"cell {(ix, iy, iz)}")
        for hole in HOLES:
            self.assertEqual(int(mask[_flat_index(*hole)]), 0, msg=f"hole {hole}")

    def test_counts_and_volume_come_from_the_artifact(self) -> None:
        cells = _occupied_cells()
        self.assertEqual(self.grids.voxel_count, len(cells))
        self.assertEqual(self.grids.cast_voxel_count, sum(1 for c in cells if c[2] < 2))
        self.assertAlmostEqual(self.grids.total_volume_in3, 12.5)
        self.assertEqual(self.grids.part_type_counts[0], sum(1 for c in cells if c[2] < 2))
        self.assertEqual(self.grids.part_type_counts[1], sum(1 for c in cells if c[2] >= 2))

    def test_scalar_grid_reproduces_occupied_values(self) -> None:
        grid = self.grids.scalars["shrinkage"]
        dense = grid.dequantize()
        span = grid.maximum - grid.minimum
        tolerance = span / 255.0

        self.assertAlmostEqual(grid.minimum, 5.0)
        self.assertAlmostEqual(grid.maximum, _shrinkage_value(5, 4, 3))
        for cell in _occupied_cells():
            self.assertAlmostEqual(
                float(dense[_flat_index(*cell)]),
                _shrinkage_value(*cell),
                delta=tolerance,
                msg=f"cell {cell}",
            )

    def test_holes_take_the_nearest_inside_value_and_never_bleed_to_zero(self) -> None:
        grid = self.grids.scalars["shrinkage"]
        dense = grid.dequantize()
        tolerance = (grid.maximum - grid.minimum) / 255.0
        values_by_cell = {cell: _shrinkage_value(*cell) for cell in _occupied_cells()}

        self.assertGreaterEqual(float(np.min(dense)), grid.minimum - tolerance)

        for hole in HOLES:
            actual = float(dense[_flat_index(*hole)])
            candidates = _brute_force_nearest_values(values_by_cell, hole)
            self.assertTrue(
                any(abs(actual - candidate) <= tolerance for candidate in candidates),
                msg=f"hole {hole}: {actual} not among nearest values {sorted(set(candidates))}",
            )

    def test_solidification_time_is_exposed_as_its_own_mode(self) -> None:
        self.assertIn("solidification_time", self.grids.scalars)
        grid = self.grids.scalars["solidification_time"]
        self.assertAlmostEqual(grid.minimum, 20.0)
        self.assertAlmostEqual(grid.maximum, 35.0)

    def test_frames_are_quantized_bytes_with_their_times(self) -> None:
        stack = self.grids.stacks["fraction_liquid"]
        self.assertEqual(stack.frames.dtype, np.uint8)
        self.assertEqual(stack.count, FRAME_COUNT)
        self.assertEqual(stack.frames.shape, (FRAME_COUNT, DIMS[0] * DIMS[1] * DIMS[2]))
        self.assertEqual([float(t) for t in stack.times_s], FRAME_TIMES)

        expected = [255, 128, 0]
        for frame_index, level in enumerate(expected):
            occupied = [stack.frames[frame_index][_flat_index(*c)] for c in _occupied_cells()]
            self.assertTrue(
                all(abs(int(value) - level) <= 1 for value in occupied),
                msg=f"frame {frame_index} expected ~{level}, saw {sorted(set(int(v) for v in occupied))}",
            )

    def test_metadata_matches_the_grids(self) -> None:
        meta = fields.build_grid_metadata(self.grids)

        self.assertEqual(meta["dims"], list(DIMS))
        self.assertAlmostEqual(meta["spacing"], SPACING)
        self.assertIn("shrinkage", meta["modes"])
        self.assertIn("solidification_time", meta["modes"])
        self.assertIn("fraction_liquid", meta["modes"])
        self.assertEqual(meta["units"]["solidification_time"], "s")
        self.assertEqual(meta["frames"]["fraction_liquid"]["count"], FRAME_COUNT)
        self.assertEqual(meta["frames"]["fraction_liquid"]["times_s"], FRAME_TIMES)
        self.assertEqual(meta["voxelCount"], len(_occupied_cells()))

        codes = {entry["code"]: entry for entry in meta["partTypes"]}
        self.assertEqual(set(codes), {1, 2})
        self.assertEqual(codes[1]["id"], "cast")
        self.assertEqual(codes[2]["id"], "riser")

        self.assertEqual(meta["stats"]["shrinkage_max"], 5.0)
        self.assertEqual(meta["stats"]["voxel_count"], len(_occupied_cells()))

    def test_constant_field_is_reported_but_not_offered_as_a_mode(self) -> None:
        artifact = _synthetic_artifact()
        artifact["visualization"]["hotspot"] = [0.0] * len(_occupied_cells())
        grids = fields.build_field_grids(artifact["visualization"], artifact["summary"])
        meta = fields.build_grid_metadata(grids)

        self.assertNotIn("hotspot", meta["modes"])
        self.assertIn("hotspot", meta["emptyModes"])

    def test_empty_visualization_yields_no_voxels(self) -> None:
        grids = fields.build_field_grids({}, {})
        self.assertEqual(grids.voxel_count, 0)

class SidecarTests(unittest.TestCase):
    def setUp(self) -> None:
        fields.clear_cache()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.artifact = _synthetic_artifact()

    def tearDown(self) -> None:
        fields.clear_cache()
        self.temp_dir.cleanup()

    def test_round_trip_preserves_every_grid(self) -> None:
        grids = fields.build_field_grids(self.artifact["visualization"], self.artifact["summary"])
        path = self.root / "run.grid.npz"
        fields.save_sidecar(path, grids)

        restored = fields.load_sidecar(path)

        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(restored.lattice.dims, grids.lattice.dims)
        self.assertAlmostEqual(restored.lattice.spacing, grids.lattice.spacing)
        np.testing.assert_array_equal(restored.mask, grids.mask)
        self.assertEqual(set(restored.scalars), set(grids.scalars))
        np.testing.assert_array_equal(
            restored.scalars["shrinkage"].values, grids.scalars["shrinkage"].values
        )
        self.assertAlmostEqual(restored.scalars["shrinkage"].maximum, grids.scalars["shrinkage"].maximum)
        np.testing.assert_array_equal(
            restored.stacks["fraction_liquid"].frames, grids.stacks["fraction_liquid"].frames
        )
        self.assertEqual(restored.stats, grids.stats)
        self.assertEqual(restored.part_type_counts, grids.part_type_counts)

    def test_write_artifact_also_writes_the_sidecar(self) -> None:
        artifact_path = self.root / "artifacts" / "abc.json"
        write_artifact(artifact_path, self.artifact)

        sidecar = fields.sidecar_path(artifact_path)
        self.assertTrue(sidecar.exists())
        self.assertEqual(sidecar.name, "abc.grid.npz")
        self.assertIsNotNone(fields.load_sidecar(sidecar))

    def test_lazy_build_writes_a_sidecar_and_caches_by_mtime(self) -> None:
        artifact_path = self.root / "lazy.json"
        artifact_path.write_text(json.dumps(self.artifact), encoding="utf-8")
        sidecar = fields.sidecar_path(artifact_path)
        self.assertFalse(sidecar.exists())

        first = fields.load_grids("proj", artifact_path)
        self.assertIsNotNone(first)
        self.assertTrue(sidecar.exists())

        second = fields.load_grids("proj", artifact_path)
        self.assertIs(first, second)

    def test_rewriting_the_artifact_invalidates_the_cached_grids(self) -> None:
        artifact_path = self.root / "rerun.json"
        artifact_path.write_text(json.dumps(self.artifact), encoding="utf-8")
        first = fields.load_grids("proj", artifact_path)
        assert first is not None

        changed = _synthetic_artifact()
        changed["visualization"]["shrinkage"] = [
            _shrinkage_value(*cell) * 2.0 for cell in _occupied_cells()
        ]
        artifact_path.write_text(json.dumps(changed), encoding="utf-8")
        stat = artifact_path.stat()
        import os

        os.utime(artifact_path, ns=(stat.st_atime_ns + 2_000_000, stat.st_mtime_ns + 2_000_000))

        second = fields.load_grids("proj", artifact_path)
        assert second is not None
        self.assertIsNot(first, second)
        self.assertAlmostEqual(
            second.scalars["shrinkage"].maximum, 2.0 * first.scalars["shrinkage"].maximum
        )

class GridEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        fields.clear_cache()
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

        self.artifact = _synthetic_artifact()
        self.project_id = repo.create_project(self.database_path, {"name": "Grid endpoints"})
        artifact_path = self.temp_folder / f"{self.project_id}.json"
        write_artifact(artifact_path, self.artifact)
        repo.mark_project_completed(
            self.database_path, self.project_id, self.artifact["summary"], str(artifact_path)
        )

    def tearDown(self) -> None:
        fields.clear_cache()
        self.temp_dir.cleanup()

    def _body(self, response: Any) -> bytes:
        if response.headers.get("Content-Encoding") == "gzip":
            return gzip.decompress(response.data)
        return bytes(response.data)

    def test_metadata_endpoint(self) -> None:
        response = self.client.get(f"/api/projects/{self.project_id}/grid")

        self.assertEqual(response.status_code, 200)
        meta = response.get_json()
        self.assertEqual(meta["dims"], list(DIMS))
        self.assertAlmostEqual(meta["spacing"], SPACING)
        self.assertIn("shrinkage", meta["modes"])
        self.assertEqual(meta["frames"]["fraction_liquid"]["count"], FRAME_COUNT)
        self.assertEqual(meta["cellCount"], DIMS[0] * DIMS[1] * DIMS[2])

    def test_mask_endpoint_byte_layout(self) -> None:
        response = self.client.get(f"/api/projects/{self.project_id}/grid/mask")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/octet-stream")
        self.assertEqual(response.headers["X-Grid-Dims"], "6,5,4")
        body = self._body(response)
        self.assertEqual(len(body), DIMS[0] * DIMS[1] * DIMS[2])
        mask = np.frombuffer(body, dtype=np.uint8)
        for hole in HOLES:
            self.assertEqual(int(mask[_flat_index(*hole)]), 0)
        self.assertEqual(int(mask[_flat_index(0, 0, 0)]), 1)
        self.assertEqual(int(mask[_flat_index(0, 0, 3)]), 2)

    def test_scalar_endpoint_uint8_and_float32(self) -> None:
        cell_count = DIMS[0] * DIMS[1] * DIMS[2]

        quantized = self.client.get(f"/api/projects/{self.project_id}/grid/shrinkage")
        self.assertEqual(quantized.status_code, 200)
        self.assertEqual(quantized.headers["X-Grid-Dtype"], "uint8")
        minimum = float(quantized.headers["X-Field-Min"])
        maximum = float(quantized.headers["X-Field-Max"])
        bytes_body = self._body(quantized)
        self.assertEqual(len(bytes_body), cell_count)

        floats = self.client.get(f"/api/projects/{self.project_id}/grid/shrinkage?dtype=f32")
        self.assertEqual(floats.status_code, 200)
        self.assertEqual(floats.headers["X-Grid-Dtype"], "float32")
        float_body = self._body(floats)
        self.assertEqual(len(float_body), cell_count * 4)

        unpacked = struct.unpack(f"<{cell_count}f", float_body)
        raw = np.frombuffer(bytes_body, dtype=np.uint8)
        span = maximum - minimum
        for cell in [(0, 0, 0), (5, 4, 3), (2, 2, 1)]:
            index = _flat_index(*cell)
            expected = minimum + (float(raw[index]) / 255.0) * span
            self.assertAlmostEqual(unpacked[index], expected, places=3)
            self.assertAlmostEqual(unpacked[index], _shrinkage_value(*cell), delta=span / 255.0)

    def test_frames_endpoint_slices_the_stack(self) -> None:
        cell_count = DIMS[0] * DIMS[1] * DIMS[2]

        everything = self.client.get(
            f"/api/projects/{self.project_id}/grid/fraction_liquid/frames"
        )
        self.assertEqual(everything.status_code, 200)
        self.assertEqual(everything.headers["X-Frame-Total"], str(FRAME_COUNT))
        self.assertEqual(everything.headers["X-Frame-Count"], str(FRAME_COUNT))
        self.assertEqual(len(self._body(everything)), FRAME_COUNT * cell_count)

        window = self.client.get(
            f"/api/projects/{self.project_id}/grid/fraction_liquid/frames?start=1&count=2"
        )
        self.assertEqual(window.status_code, 200)
        self.assertEqual(window.headers["X-Frame-Start"], "1")
        self.assertEqual(window.headers["X-Frame-Count"], "2")
        window_bytes = self._body(window)
        self.assertEqual(len(window_bytes), 2 * cell_count)

        full = np.frombuffer(self._body(everything), dtype=np.uint8)
        sliced = np.frombuffer(window_bytes, dtype=np.uint8)
        np.testing.assert_array_equal(sliced, full[cell_count : 3 * cell_count])

    def test_frames_endpoint_clamps_an_out_of_range_window(self) -> None:
        response = self.client.get(
            f"/api/projects/{self.project_id}/grid/fraction_liquid/frames?start=99&count=5"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Frame-Count"], "0")
        self.assertEqual(len(self._body(response)), 0)

    def test_frames_endpoint_rejects_a_non_numeric_window(self) -> None:
        response = self.client.get(
            f"/api/projects/{self.project_id}/grid/fraction_liquid/frames?start=soon"
        )
        self.assertEqual(response.status_code, 400)

    def test_gzip_is_applied_only_when_the_client_asks(self) -> None:
        original = getattr(__import__("web.routes_fields", fromlist=["x"]), "_GZIP_MIN_BYTES")
        module = __import__("web.routes_fields", fromlist=["x"])
        module._GZIP_MIN_BYTES = 0
        try:
            compressed = self.client.get(
                f"/api/projects/{self.project_id}/grid/mask",
                headers={"Accept-Encoding": "gzip"},
            )
            plain = self.client.get(
                f"/api/projects/{self.project_id}/grid/mask",
                headers={"Accept-Encoding": "identity"},
            )
        finally:
            module._GZIP_MIN_BYTES = original

        self.assertEqual(compressed.headers.get("Content-Encoding"), "gzip")
        self.assertEqual(compressed.headers.get("Vary"), "Accept-Encoding")
        self.assertIsNone(plain.headers.get("Content-Encoding"))
        np.testing.assert_array_equal(
            np.frombuffer(gzip.decompress(compressed.data), dtype=np.uint8),
            np.frombuffer(plain.data, dtype=np.uint8),
        )

    def test_unknown_mode_and_unknown_project(self) -> None:
        self.assertEqual(
            self.client.get(f"/api/projects/{self.project_id}/grid/not_a_mode").status_code, 404
        )
        self.assertEqual(
            self.client.get(f"/api/projects/{self.project_id}/grid/shrinkage/frames").status_code, 404
        )
        self.assertEqual(self.client.get("/api/projects/nope/grid").status_code, 404)
        self.assertEqual(self.client.get("/api/projects/nope/grid/mask").status_code, 404)

    def test_health(self) -> None:
        response = self.client.get("/api/fields/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["ok"])

class FillAnimationTests(unittest.TestCase):

    def setUp(self) -> None:
        fields.clear_cache()
        self.artifact = _synthetic_artifact()
        self.grids = fields.build_field_grids(
            self.artifact["visualization"], self.artifact["summary"]
        )

    def tearDown(self) -> None:
        fields.clear_cache()

    def test_fill_frames_become_their_own_stack(self) -> None:
        stack = self.grids.stacks["fill_animation"]
        self.assertEqual(stack.count, FRAME_COUNT)
        self.assertEqual([float(t) for t in stack.times_s], FILL_TIMES)
        self.assertIsNot(stack, self.grids.stacks["fraction_liquid"])

    def test_fill_front_rises_with_the_frame(self) -> None:
        stack = self.grids.stacks["fill_animation"]
        filled_per_frame = []
        for frame_index in range(stack.count):
            frame = stack.frames[frame_index]
            filled_per_frame.append(
                sum(1 for cell in _occupied_cells() if frame[_flat_index(*cell)] > 127)
            )
        self.assertLess(filled_per_frame[0], filled_per_frame[1])
        self.assertLess(filled_per_frame[1], filled_per_frame[2])
        self.assertEqual(filled_per_frame[2], len(_occupied_cells()))

    def test_metadata_reports_the_fill_stack_under_its_own_key(self) -> None:
        meta = fields.build_grid_metadata(self.grids)
        self.assertIn("fill_animation", meta["modes"])
        self.assertEqual(meta["frames"]["fill"]["count"], FRAME_COUNT)
        self.assertEqual(meta["frames"]["fill"]["times_s"], FILL_TIMES)

    def test_the_animated_modes_carry_no_unit(self) -> None:
        meta = fields.build_grid_metadata(self.grids)
        self.assertNotIn("fill_animation", meta["units"])
        self.assertNotIn("fraction_liquid", meta["units"])
        self.assertEqual(meta["frames"]["fill"]["times_s"], FILL_TIMES)
        self.assertEqual(meta["units"]["solidification_time"], "s")

    def test_envelope_of_a_fill_frame_grows_over_time(self) -> None:
        early = fields.build_envelope(self.grids, "fill_animation", 0.5, frame_index=0)
        late = fields.build_envelope(self.grids, "fill_animation", 0.5, frame_index=2)
        assert early is not None and late is not None
        self.assertGreater(late.vertex_count, 0)
        self.assertGreater(_bounding_height(late), _bounding_height(early))

def _bounding_height(envelope: fields.Envelope) -> float:
    if envelope.vertex_count == 0:
        return 0.0
    return float(np.max(envelope.positions[:, 2]) - np.min(envelope.positions[:, 2]))

def _bounding_volume(envelope: fields.Envelope) -> float:
    if envelope.vertex_count == 0:
        return 0.0
    span = np.max(envelope.positions, axis=0) - np.min(envelope.positions, axis=0)
    return float(np.prod(span))

class EnvelopeTests(unittest.TestCase):

    def setUp(self) -> None:
        fields.clear_cache()
        clear_envelope_cache()
        self.artifact = _synthetic_artifact()
        self.grids = fields.build_field_grids(
            self.artifact["visualization"], self.artifact["summary"]
        )
        self.grid = self.grids.scalars["shrinkage"]

    def tearDown(self) -> None:
        fields.clear_cache()
        clear_envelope_cache()

    def _at(self, fraction: float) -> float:
        return self.grid.minimum + fraction * (self.grid.maximum - self.grid.minimum)

    def test_a_higher_threshold_encloses_less(self) -> None:
        loose = fields.build_envelope(self.grids, "shrinkage", self._at(0.2))
        tight = fields.build_envelope(self.grids, "shrinkage", self._at(0.8))
        assert loose is not None and tight is not None
        self.assertGreater(loose.vertex_count, 0)
        self.assertGreater(tight.vertex_count, 0)
        self.assertGreater(_bounding_volume(loose), _bounding_volume(tight))

    def test_envelope_stays_inside_the_metal(self) -> None:
        envelope = fields.build_envelope(self.grids, "shrinkage", self._at(0.1))
        assert envelope is not None
        nx, ny, nz = DIMS
        for x, y, z in envelope.positions:
            ix = round((x - ORIGIN[0]) / SPACING)
            iy = round((y - ORIGIN[1]) / SPACING)
            iz = round((z - ORIGIN[2]) / SPACING)
            self.assertTrue(-1 <= ix <= nx, msg=f"x {x}")
            self.assertTrue(-1 <= iy <= ny, msg=f"y {y}")
            self.assertTrue(-1 <= iz <= nz, msg=f"z {z}")

    def test_cast_only_drops_the_riser(self) -> None:
        threshold = self._at(0.05)
        everything = fields.build_envelope(self.grids, "shrinkage", threshold)
        cast_only = fields.build_envelope(self.grids, "shrinkage", threshold, cast_only=True)
        assert everything is not None and cast_only is not None
        self.assertGreater(cast_only.vertex_count, 0)
        self.assertLess(
            float(np.max(cast_only.positions[:, 2])),
            float(np.max(everything.positions[:, 2])),
        )

    def test_vertex_scalars_track_the_threshold(self) -> None:
        threshold = self._at(0.5)
        envelope = fields.build_envelope(self.grids, "shrinkage", threshold)
        assert envelope is not None and envelope.scalars is not None
        span = self.grid.maximum - self.grid.minimum
        deviation = np.abs(envelope.scalars - threshold) / span
        self.assertLess(float(np.median(deviation)), 0.2)

    def test_binary_layout(self) -> None:
        envelope = fields.build_envelope(self.grids, "shrinkage", self._at(0.3))
        assert envelope is not None and envelope.scalars is not None
        body = fields.envelope_bytes(envelope)

        vertices = envelope.vertex_count
        indices = int(envelope.indices.size)
        self.assertEqual(len(body), vertices * 7 * 4 + indices * 4)

        offset = 0
        positions = np.frombuffer(body, dtype="<f4", count=vertices * 3, offset=offset)
        offset += vertices * 3 * 4
        normals = np.frombuffer(body, dtype="<f4", count=vertices * 3, offset=offset)
        offset += vertices * 3 * 4
        parsed_indices = np.frombuffer(body, dtype="<u4", count=indices, offset=offset)
        offset += indices * 4
        scalars = np.frombuffer(body, dtype="<f4", count=vertices, offset=offset)

        np.testing.assert_allclose(positions, envelope.positions.reshape(-1), rtol=1e-6)
        np.testing.assert_allclose(normals, envelope.normals.reshape(-1), rtol=1e-6)
        np.testing.assert_array_equal(parsed_indices, envelope.indices)
        np.testing.assert_allclose(scalars, envelope.scalars, rtol=1e-6)
        self.assertLess(int(parsed_indices.max()), vertices)

    def test_scalars_can_be_skipped(self) -> None:
        envelope = fields.build_envelope(
            self.grids, "shrinkage", self._at(0.3), with_scalars=False
        )
        assert envelope is not None
        self.assertIsNone(envelope.scalars)
        expected = envelope.vertex_count * 6 * 4 + int(envelope.indices.size) * 4
        self.assertEqual(len(fields.envelope_bytes(envelope)), expected)

class EnvelopeEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        fields.clear_cache()
        clear_envelope_cache()
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

        self.artifact = _synthetic_artifact()
        self.project_id = repo.create_project(self.database_path, {"name": "Envelopes"})
        artifact_path = self.temp_folder / f"{self.project_id}.json"
        write_artifact(artifact_path, self.artifact)
        repo.mark_project_completed(
            self.database_path, self.project_id, self.artifact["summary"], str(artifact_path)
        )

    def tearDown(self) -> None:
        fields.clear_cache()
        clear_envelope_cache()
        self.temp_dir.cleanup()

    def _body(self, response: Any) -> bytes:
        if response.headers.get("Content-Encoding") == "gzip":
            return gzip.decompress(response.data)
        return bytes(response.data)

    def test_isosurface_endpoint_returns_a_parsable_block(self) -> None:
        response = self.client.get(
            f"/api/projects/{self.project_id}/grid/shrinkage/isosurface?threshold=30"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/octet-stream")
        self.assertEqual(response.headers["X-Has-Scalar"], "1")

        vertices = int(response.headers["X-Vertex-Count"])
        indices = int(response.headers["X-Index-Count"])
        self.assertGreater(vertices, 0)
        self.assertEqual(indices % 3, 0)

        body = self._body(response)
        self.assertEqual(len(body), vertices * 7 * 4 + indices * 4)
        positions = np.frombuffer(body, dtype="<f4", count=vertices * 3)
        self.assertTrue(np.all(np.isfinite(positions)))

    def test_isosurface_threshold_is_in_field_units(self) -> None:
        meta = self.client.get(f"/api/projects/{self.project_id}/grid").get_json()
        span = meta["ranges"]["shrinkage"]["max"] - meta["ranges"]["shrinkage"]["min"]
        low = meta["ranges"]["shrinkage"]["min"] + 0.1 * span
        response = self.client.get(
            f"/api/projects/{self.project_id}/grid/shrinkage/isosurface?threshold={low}"
        )
        self.assertAlmostEqual(float(response.headers["X-Threshold"]), low, places=6)
        self.assertGreater(int(response.headers["X-Vertex-Count"]), 0)

    def test_isosurface_without_scalars(self) -> None:
        response = self.client.get(
            f"/api/projects/{self.project_id}/grid/shrinkage/isosurface?threshold=30&scalars=0"
        )
        self.assertEqual(response.headers["X-Has-Scalar"], "0")
        vertices = int(response.headers["X-Vertex-Count"])
        indices = int(response.headers["X-Index-Count"])
        self.assertEqual(len(self._body(response)), vertices * 6 * 4 + indices * 4)

    def test_isosurface_of_a_frame(self) -> None:
        response = self.client.get(
            f"/api/projects/{self.project_id}/grid/fill_animation/isosurface?threshold=0.5&frame=2"
        )
        self.assertEqual(response.status_code, 200)
        self.assertGreater(int(response.headers["X-Vertex-Count"]), 0)

    def test_isosurface_rejects_bad_input(self) -> None:
        base = f"/api/projects/{self.project_id}/grid"
        self.assertEqual(self.client.get(f"{base}/nope/isosurface?threshold=1").status_code, 404)
        self.assertEqual(
            self.client.get(f"{base}/shrinkage/isosurface?threshold=abc").status_code, 400
        )
        self.assertEqual(
            self.client.get(f"{base}/shrinkage/isosurface?threshold=1&frame=x").status_code, 400
        )
        self.assertEqual(
            self.client.get("/api/projects/nope/grid/shrinkage/isosurface").status_code, 404
        )

    def test_repeated_requests_are_served_from_the_cache(self) -> None:
        url = f"/api/projects/{self.project_id}/grid/shrinkage/isosurface?threshold=30"
        first = self.client.get(url)
        second = self.client.get(url)
        self.assertEqual(first.headers["X-Vertex-Count"], second.headers["X-Vertex-Count"])
        self.assertEqual(self._body(first), self._body(second))

class SummaryStatisticsTests(unittest.TestCase):

    def _collect(self, **attributes: Any) -> Dict[str, Any]:
        engine = SimpleNamespace(
            alloy=SimpleNamespace(name="WCB"),
            time=612.5,
            **attributes,
        )
        result_data: Dict[str, Any] = {"voxel_count": 1000}
        _add_summary_statistics(result_data, engine)
        return result_data

    def test_scalars_are_flat_and_named_with_their_units(self) -> None:
        stats = self._collect(
            chvorinov_time=np.array([10.0, 20.0, 30.0]),
            total_shrinkage_volume=1.25,
            affected_voxel_count=250,
            hotspot_count=17,
            niyama_defect_count=3,
            max_stress_mpa=182.5,
            avg_sdas_um=48.0,
            feeding_blocked_count=9,
            feeding_path_result=SimpleNamespace(fed_fraction=0.87),
        )

        self.assertEqual(stats["alloy"], "WCB")
        self.assertAlmostEqual(stats["simulation_time_s"], 612.5)
        self.assertAlmostEqual(stats["chvorinov_time_max_s"], 30.0)
        self.assertAlmostEqual(stats["chvorinov_time_mean_s"], 20.0)
        self.assertAlmostEqual(stats["shrinkage_volume_in3"], 1.25)
        self.assertEqual(stats["shrinkage_voxel_count"], 250)
        self.assertAlmostEqual(stats["shrinkage_percentage"], 25.0)
        self.assertEqual(stats["hotspot_voxel_count"], 17)
        self.assertEqual(stats["niyama_defect_count"], 3)
        self.assertAlmostEqual(stats["max_stress_mpa"], 182.5)
        self.assertAlmostEqual(stats["avg_sdas_um"], 48.0)
        self.assertEqual(stats["feeding_blocked_count"], 9)
        self.assertAlmostEqual(stats["fed_fraction"], 0.87)

        for key, value in stats.items():
            self.assertIsInstance(value, (int, float, str), msg=key)

    def test_defect_volume_comes_from_the_engine(self) -> None:
        stats = self._collect(
            get_defect_volume_estimate=lambda: {
                "total_volume_in3": 2.5,
                "max_volume_in3": 4.0,
                "defect_count": 42,
                "casting_volume_in3": 300.0,
                "defect_percentage": 0.83,
                "threshold_used": 0.3,
            }
        )
        self.assertAlmostEqual(stats["defect_volume_in3"], 2.5)
        self.assertAlmostEqual(stats["defect_volume_max_in3"], 4.0)
        self.assertEqual(stats["defect_voxel_count"], 42)
        self.assertAlmostEqual(stats["casting_volume_in3"], 300.0)
        self.assertAlmostEqual(stats["defect_threshold"], 0.3)

    def test_an_analysis_that_did_not_run_is_absent_not_zero(self) -> None:
        stats = self._collect(hotspot_count=5)
        self.assertIn("hotspot_voxel_count", stats)
        for absent in [
            "hot_tear_voxel_count",
            "sand_burn_in_voxel_count",
            "gas_porosity_voxel_count",
            "max_stress_mpa",
            "avg_sdas_um",
            "defect_volume_in3",
        ]:
            self.assertNotIn(absent, stats)

    def test_fill_statistics_are_promoted_out_of_their_nested_dict(self) -> None:
        engine = SimpleNamespace(alloy=SimpleNamespace(name="WCB"), time=1.0)
        result_data: Dict[str, Any] = {
            "voxel_count": 10,
            "fill_time_stats": {
                "max_fill_time_s": 3.4,
                "filled_voxels": 9,
                "unfilled_voxels": 1,
            },
        }
        _add_summary_statistics(result_data, engine)
        self.assertAlmostEqual(result_data["fill_max_time_s"], 3.4)
        self.assertEqual(result_data["fill_filled_voxel_count"], 9)
        self.assertEqual(result_data["fill_unfilled_voxel_count"], 1)

    def test_a_broken_statistic_never_fails_the_run(self) -> None:
        def explode() -> Dict[str, Any]:
            raise RuntimeError("no mesh")

        stats = self._collect(get_defect_volume_estimate=explode, hotspot_count=1)
        self.assertNotIn("defect_volume_in3", stats)
        self.assertEqual(stats["hotspot_voxel_count"], 1)

    def test_the_flattener_carries_them_through_to_grid_metadata(self) -> None:
        artifact = _synthetic_artifact()
        artifact["summary"]["defect_volume_in3"] = 2.5
        artifact["summary"]["hotspot_voxel_count"] = 17
        grids = fields.build_field_grids(artifact["visualization"], artifact["summary"])
        meta = fields.build_grid_metadata(grids)
        self.assertAlmostEqual(meta["stats"]["defect_volume_in3"], 2.5)
        self.assertEqual(meta["stats"]["hotspot_voxel_count"], 17)

class LatticeBudgetTests(unittest.TestCase):

    def _diagonal_centers(self, count: int, spacing: float) -> np.ndarray:
        steps = np.arange(count, dtype=np.float64) * spacing
        return np.stack([steps, steps, steps], axis=1)

    def test_a_huge_lattice_is_refused_by_the_derivation(self) -> None:
        centers = self._diagonal_centers(4000, 0.05)
        with self.assertRaises(fields.LatticeTooLargeError) as caught:
            fields.derive_lattice(centers, np.full(len(centers), 0.05))
        error = caught.exception
        self.assertEqual(error.dims, (4000, 4000, 4000))
        self.assertGreater(error.cell_count, fields.MAX_LATTICE_CELLS)
        self.assertAlmostEqual(error.spacing, 0.05)

    def test_a_lattice_inside_the_budget_still_builds(self) -> None:
        centers = self._diagonal_centers(200, 0.05)
        lattice = fields.derive_lattice(centers, np.full(len(centers), 0.05))
        self.assertEqual(lattice.dims, (200, 200, 200))
        self.assertLessEqual(lattice.cell_count, fields.MAX_LATTICE_CELLS)

    def test_build_field_grids_propagates_the_refusal(self) -> None:
        centers = self._diagonal_centers(4000, 0.05)
        visualization = {
            "centers": centers.tolist(),
            "sizes": [0.05] * len(centers),
            "shrinkage": [0.5] * len(centers),
        }
        with self.assertRaises(fields.LatticeTooLargeError):
            fields.build_field_grids(visualization, {})

class GridCacheTests(unittest.TestCase):

    def _grids(self, cells: int) -> fields.FieldGrids:
        lattice = fields.Lattice(dims=(cells, 1, 1), origin=(0.0, 0.0, 0.0), spacing=1.0)
        return fields.FieldGrids(lattice=lattice, mask=np.ones(cells, dtype=np.uint8))

    def test_a_large_run_evicts_earlier_ones_rather_than_accumulating(self) -> None:
        cache = fields._GridCache(max_bytes=1000)
        for index in range(4):
            cache.put(("p", index, 0), self._grids(400))
        self.assertLessEqual(cache.nbytes, 1000)
        self.assertIsNone(cache.get(("p", 0, 0)))
        self.assertIsNotNone(cache.get(("p", 3, 0)))

    def test_a_run_larger_than_the_budget_is_simply_not_kept(self) -> None:
        cache = fields._GridCache(max_bytes=1000)
        cache.put(("p", 0, 0), self._grids(100))
        cache.put(("p", 1, 0), self._grids(5000))
        self.assertIsNone(cache.get(("p", 1, 0)))
        self.assertIsNotNone(cache.get(("p", 0, 0)))

    def test_recently_used_entries_survive(self) -> None:
        cache = fields._GridCache(max_bytes=1000)
        cache.put(("p", 0, 0), self._grids(400))
        cache.put(("p", 1, 0), self._grids(400))
        self.assertIsNotNone(cache.get(("p", 0, 0)))
        cache.put(("p", 2, 0), self._grids(400))
        self.assertIsNotNone(cache.get(("p", 0, 0)))
        self.assertIsNone(cache.get(("p", 1, 0)))

class StridedArtifactTests(unittest.TestCase):

    def setUp(self) -> None:
        fields.clear_cache()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.full = _synthetic_artifact()

    def tearDown(self) -> None:
        fields.clear_cache()
        self.temp_dir.cleanup()

    def _strided(self, step: int = 2) -> Dict[str, Any]:
        full_viz = self.full["visualization"]
        total = len(full_viz["centers"])
        keep = list(range(0, total, step))
        strided = {
            key: [value[index] for index in keep]
            for key, value in full_viz.items()
            if isinstance(value, list) and len(value) == total
        }
        strided["fraction_liquid_history"] = [
            [frame[index] for index in keep] for frame in full_viz["fraction_liquid_history"]
        ]
        strided["fraction_liquid_times"] = full_viz["fraction_liquid_times"]
        summary = dict(self.full["summary"])
        summary["voxel_count"] = total
        return {"summary": summary, "visualization": strided}

    def test_a_strided_payload_alone_produces_a_sieve(self) -> None:
        artifact = self._strided()
        grids = fields.build_field_grids(artifact["visualization"], artifact["summary"])
        occupied = len(_occupied_cells())
        self.assertLess(int(np.count_nonzero(grids.mask)), occupied)
        self.assertTrue(grids.downsampled)

    def _numpy_grid_source(self) -> Dict[str, Any]:
        full_viz = self.full["visualization"]
        source: Dict[str, Any] = {
            "centers": np.asarray(full_viz["centers"], dtype=np.float32),
            "sizes": np.asarray(full_viz["sizes"], dtype=np.float32),
            "part_types": np.asarray(full_viz["part_types"], dtype=np.float32),
            "shrinkage": np.asarray(full_viz["shrinkage"], dtype=np.float32),
            "solidification_times": np.asarray(full_viz["solidification_times"], dtype=np.float32),
            "fraction_liquid_history": [
                np.asarray(frame, dtype=np.float32)
                for frame in full_viz["fraction_liquid_history"]
            ],
            "fraction_liquid_times": list(full_viz["fraction_liquid_times"]),
        }
        return source

    def test_the_sidecar_uses_the_full_grid_source(self) -> None:
        artifact = self._strided()
        artifact_path = self.root / "run.json"
        write_artifact(
            artifact_path,
            {**artifact, GRID_SOURCE_KEY: self._numpy_grid_source()},
        )

        restored = fields.load_sidecar(fields.sidecar_path(artifact_path))
        assert restored is not None
        occupied = len(_occupied_cells())
        self.assertEqual(int(np.count_nonzero(restored.mask)), occupied)
        self.assertEqual(restored.voxel_count, occupied)
        self.assertEqual(
            restored.cast_voxel_count, sum(1 for cell in _occupied_cells() if cell[2] < 2)
        )
        self.assertFalse(restored.downsampled)

    def test_the_json_on_disk_stays_downsampled_and_carries_no_private_key(self) -> None:
        artifact = self._strided()
        artifact_path = self.root / "run.json"
        write_artifact(
            artifact_path,
            {**artifact, GRID_SOURCE_KEY: self._numpy_grid_source()},
        )
        on_disk = json.loads(artifact_path.read_text())
        self.assertNotIn(GRID_SOURCE_KEY, on_disk)
        self.assertLess(
            len(on_disk["visualization"]["centers"]), len(self.full["visualization"]["centers"])
        )

    def test_downsampling_is_inferred_when_only_the_json_survives(self) -> None:
        artifact = self._strided()
        artifact_path = self.root / "lazy.json"
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        grids = fields.load_grids("proj", artifact_path)
        assert grids is not None
        self.assertTrue(grids.downsampled)
        self.assertTrue(fields.build_grid_metadata(grids)["downsampled"])

    def test_grids_build_from_numpy_arrays_as_well_as_lists(self) -> None:
        grids = fields.build_field_grids(self._numpy_grid_source(), self.full["summary"])
        self.assertEqual(grids.lattice.dims, DIMS)
        self.assertEqual(int(np.count_nonzero(grids.mask)), len(_occupied_cells()))
        self.assertIn("shrinkage", grids.scalars)
        self.assertIn("solidification_time", grids.scalars)
        self.assertIn("fraction_liquid", grids.stacks)

    def test_a_full_payload_is_not_flagged(self) -> None:
        grids = fields.build_field_grids(
            self.full["visualization"], self.full["summary"]
        )
        self.assertFalse(grids.downsampled)
        self.assertFalse(fields.build_grid_metadata(grids)["downsampled"])

class SidecarRobustnessTests(unittest.TestCase):

    def setUp(self) -> None:
        fields.clear_cache()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.artifact = _synthetic_artifact()

    def tearDown(self) -> None:
        fields.clear_cache()
        self.temp_dir.cleanup()

    def test_a_truncated_sidecar_is_discarded_rather_than_raising(self) -> None:
        artifact_path = self.root / "run.json"
        write_artifact(artifact_path, self.artifact)
        sidecar = fields.sidecar_path(artifact_path)
        whole = sidecar.read_bytes()
        sidecar.write_bytes(whole[: len(whole) // 2])

        self.assertIsNone(fields.load_sidecar(sidecar))
        self.assertFalse(sidecar.exists())

    def test_a_truncated_sidecar_is_rebuilt_on_the_next_request(self) -> None:
        artifact_path = self.root / "run.json"
        write_artifact(artifact_path, self.artifact)
        sidecar = fields.sidecar_path(artifact_path)
        sidecar.write_bytes(b"not a zip file at all")

        grids = fields.load_grids("proj", artifact_path)
        self.assertIsNotNone(grids)
        assert grids is not None
        self.assertEqual(grids.voxel_count, len(_occupied_cells()))
        self.assertTrue(sidecar.exists())
        self.assertIsNotNone(fields.load_sidecar(sidecar))

    def test_garbage_of_every_shape_is_survivable(self) -> None:
        sidecar = self.root / "junk.grid.npz"
        for payload in [b"", b"\x00" * 64, b"PK\x03\x04 truncated"]:
            sidecar.write_bytes(payload)
            self.assertIsNone(fields.load_sidecar(sidecar))

    def test_a_half_written_artifact_is_not_a_server_error(self) -> None:
        artifact_path = self.root / "partial.json"
        artifact_path.write_text('{"summary": {}, "visual', encoding="utf-8")
        self.assertIsNone(fields.load_grids("proj", artifact_path))

    def test_write_artifact_never_publishes_a_partial_file(self) -> None:
        artifact_path = self.root / "atomic.json"
        write_artifact(artifact_path, self.artifact)
        first = json.loads(artifact_path.read_text())

        changed = _synthetic_artifact()
        changed["summary"]["voxel_count"] = 4242
        write_artifact(artifact_path, changed)
        second = json.loads(artifact_path.read_text())

        self.assertEqual(first["summary"]["voxel_count"], len(_occupied_cells()))
        self.assertEqual(second["summary"]["voxel_count"], 4242)
        self.assertEqual(sorted(q.suffix for q in self.root.iterdir()), [".json", ".npz"])

    def test_concurrent_sidecar_writers_publish_a_loadable_file(self) -> None:
        grids = fields.build_field_grids(
            self.artifact["visualization"], self.artifact["summary"]
        )
        sidecar = self.root / "concurrent.grid.npz"
        errors: List[BaseException] = []

        def write() -> None:
            try:
                for _ in range(4):
                    fields.save_sidecar(sidecar, grids)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        restored = fields.load_sidecar(sidecar)
        self.assertIsNotNone(restored)
        assert restored is not None
        np.testing.assert_array_equal(restored.mask, grids.mask)
        self.assertEqual([q.name for q in self.root.iterdir()], ["concurrent.grid.npz"])

class ConstantFieldTests(unittest.TestCase):

    def _grids_for(self, values: List[float]) -> fields.FieldGrids:
        artifact = _synthetic_artifact()
        artifact["visualization"]["hotspot"] = values
        return fields.build_field_grids(artifact["visualization"], artifact["summary"])

    def test_a_constant_non_zero_field_stays_selectable(self) -> None:
        grids = self._grids_for([0.42] * len(_occupied_cells()))
        grid = grids.scalars["hotspot"]
        self.assertTrue(grid.constant)
        self.assertFalse(grid.empty)

        meta = fields.build_grid_metadata(grids)
        self.assertIn("hotspot", meta["modes"])
        self.assertIn("hotspot", meta["constantModes"])
        self.assertNotIn("hotspot", meta["emptyModes"])
        self.assertAlmostEqual(meta["ranges"]["hotspot"]["min"], 0.42, places=6)
        self.assertAlmostEqual(meta["ranges"]["hotspot"]["max"], 0.42, places=6)

    def test_a_constant_zero_field_is_still_empty(self) -> None:
        grids = self._grids_for([0.0] * len(_occupied_cells()))
        grid = grids.scalars["hotspot"]
        self.assertTrue(grid.constant)
        self.assertTrue(grid.empty)

        meta = fields.build_grid_metadata(grids)
        self.assertNotIn("hotspot", meta["modes"])
        self.assertIn("hotspot", meta["emptyModes"])

    def test_a_varying_field_is_neither(self) -> None:
        grids = self._grids_for([i / len(_occupied_cells()) for i in range(len(_occupied_cells()))])
        grid = grids.scalars["hotspot"]
        self.assertFalse(grid.constant)
        self.assertFalse(grid.empty)
        meta = fields.build_grid_metadata(grids)
        self.assertIn("hotspot", meta["modes"])
        self.assertNotIn("hotspot", meta["constantModes"])

    def test_a_constant_field_survives_the_sidecar(self) -> None:
        grids = self._grids_for([0.42] * len(_occupied_cells()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "constant.grid.npz"
            fields.save_sidecar(path, grids)
            restored = fields.load_sidecar(path)
        assert restored is not None
        self.assertTrue(restored.scalars["hotspot"].constant)
        self.assertFalse(restored.scalars["hotspot"].empty)

class _SpyQueueManager:

    def __init__(self, result: Optional[Dict[str, Any]] = None) -> None:
        self.result = result
        self.calls = 0

    def get_result(self, project_id: str) -> Optional[Dict[str, Any]]:
        self.calls += 1
        return self.result

class ArtifactResolutionTests(unittest.TestCase):

    def setUp(self) -> None:
        fields.clear_cache()
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.database_path = root / "simulations.db"
        self.temp_folder = root / "temp"
        self.temp_folder.mkdir()
        repo.init_db(self.database_path)

        self.artifact = _synthetic_artifact()
        self.spy = _SpyQueueManager()

        app = Flask(__name__)
        app.config["DATABASE"] = self.database_path
        app.config["TEMP_FOLDER"] = self.temp_folder
        app.extensions["simulation_queue_manager"] = self.spy
        app.register_blueprint(fields_bp)
        self.client = app.test_client()

    def tearDown(self) -> None:
        fields.clear_cache()
        self.temp_dir.cleanup()

    def _completed_project(self) -> str:
        project_id = repo.create_project(self.database_path, {"name": "On disk"})
        artifact_path = self.temp_folder / f"{project_id}.json"
        write_artifact(artifact_path, self.artifact)
        repo.mark_project_completed(
            self.database_path, project_id, self.artifact["summary"], str(artifact_path)
        )
        return project_id

    def test_a_run_on_disk_never_touches_the_queue_manager(self) -> None:
        project_id = self._completed_project()
        for path in ["grid", "grid/mask", "grid/shrinkage", "grid/fraction_liquid/frames"]:
            response = self.client.get(f"/api/projects/{project_id}/{path}")
            self.assertEqual(response.status_code, 200, msg=path)
        self.assertEqual(self.spy.calls, 0)

    def test_repeated_requests_still_never_touch_it(self) -> None:
        project_id = self._completed_project()
        for _ in range(5):
            self.client.get(f"/api/projects/{project_id}/grid")
        self.assertEqual(self.spy.calls, 0)

    def test_a_run_with_no_artifact_on_disk_falls_back_to_the_queue(self) -> None:
        project_id = repo.create_project(self.database_path, {"name": "In flight"})
        self.spy.result = self.artifact
        response = self.client.get(f"/api/projects/{project_id}/grid")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["dims"], list(DIMS))
        self.assertGreaterEqual(self.spy.calls, 1)

    def test_a_missing_artifact_file_is_a_404(self) -> None:
        project_id = repo.create_project(self.database_path, {"name": "Gone"})
        repo.mark_project_completed(
            self.database_path,
            project_id,
            self.artifact["summary"],
            str(self.temp_folder / "does-not-exist.json"),
        )
        response = self.client.get(f"/api/projects/{project_id}/grid")
        self.assertEqual(response.status_code, 404)

class LatticeBudgetEndpointTests(unittest.TestCase):

    def setUp(self) -> None:
        fields.clear_cache()
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.database_path = root / "simulations.db"
        self.temp_folder = root / "temp"
        self.temp_folder.mkdir()
        repo.init_db(self.database_path)

        app = Flask(__name__)
        app.config["DATABASE"] = self.database_path
        app.config["TEMP_FOLDER"] = self.temp_folder
        app.extensions["simulation_queue_manager"] = _SpyQueueManager()
        app.register_blueprint(fields_bp)
        self.client = app.test_client()

        steps = np.arange(4000, dtype=np.float64) * 0.05
        centers = np.stack([steps, steps, steps], axis=1)
        artifact = {
            "summary": {"voxel_count": len(centers), "total_volume": 1.0},
            "visualization": {
                "centers": centers.tolist(),
                "sizes": [0.05] * len(centers),
                "shrinkage": [0.5] * len(centers),
            },
        }
        self.project_id = repo.create_project(self.database_path, {"name": "Too fine"})
        artifact_path = self.temp_folder / f"{self.project_id}.json"
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        repo.mark_project_completed(
            self.database_path, self.project_id, artifact["summary"], str(artifact_path)
        )

    def tearDown(self) -> None:
        fields.clear_cache()
        self.temp_dir.cleanup()

    def test_the_endpoint_explains_itself_instead_of_crashing(self) -> None:
        response = self.client.get(f"/api/projects/{self.project_id}/grid")
        self.assertEqual(response.status_code, 413)
        body = response.get_json()
        self.assertEqual(body["code"], "lattice_too_large")
        self.assertEqual(body["dims"], [4000, 4000, 4000])
        self.assertGreater(body["cellCount"], body["maxCellCount"])
        self.assertIn("coarser voxel size", body["error"])

    def test_every_grid_endpoint_answers_the_same_way(self) -> None:
        for path in ["grid/mask", "grid/shrinkage", "grid/shrinkage/isosurface?threshold=1"]:
            response = self.client.get(f"/api/projects/{self.project_id}/{path}")
            self.assertEqual(response.status_code, 413, msg=path)
            self.assertEqual(response.get_json()["code"], "lattice_too_large", msg=path)

class RemovedTetResultTests(unittest.TestCase):

    def setUp(self) -> None:
        fields.clear_cache()
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.database_path = root / "simulations.db"
        repo.init_db(self.database_path)
        app = Flask(__name__)
        app.config["DATABASE"] = self.database_path
        app.extensions["simulation_queue_manager"] = _SpyQueueManager()
        app.register_blueprint(fields_bp)
        self.client = app.test_client()

        artifact = _synthetic_artifact()
        artifact["visualization"]["mesh_type"] = "tet"
        self.project_id = repo.create_project(self.database_path, {"name": "Old tet run"})
        self.artifact_path = root / f"{self.project_id}.json"
        self.artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        repo.mark_project_completed(self.database_path, self.project_id, artifact["summary"], str(self.artifact_path))

    def tearDown(self) -> None:
        fields.clear_cache()
        self.temp_dir.cleanup()

    def _assert_removed(self) -> None:
        for path in ["grid", "grid/mask", "grid/shrinkage"]:
            response = self.client.get(f"/api/projects/{self.project_id}/{path}")
            self.assertEqual(response.status_code, 410, msg=path)
            self.assertIn("removed tetrahedral mesher", response.get_json()["error"], msg=path)

    def test_a_tet_marked_artifact_is_refused(self) -> None:
        self._assert_removed()

    def test_a_tet_marked_sidecar_is_refused_and_kept(self) -> None:
        sidecar = fields.sidecar_path(self.artifact_path)
        voxel_artifact = _synthetic_artifact()
        fields.save_sidecar(sidecar, fields.build_field_grids(voxel_artifact["visualization"], voxel_artifact["summary"]))
        with np.load(sidecar) as data:
            payload = {key: data[key] for key in data.files}
        meta = json.loads(bytes(payload["meta_json"]).decode("utf-8"))
        meta["mesh_type"] = "tet"
        payload["meta_json"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
        np.savez_compressed(sidecar, **payload)

        self._assert_removed()
        self.assertTrue(sidecar.exists())

@unittest.skipUnless(
    REAL_ARTIFACT.exists(),
    f"Seeded run artifact not present at {REAL_ARTIFACT}",
)
class RealArtifactTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        fields.clear_cache()
        cls.grids = fields.load_grids("ef17b6fc-ac9", REAL_ARTIFACT)

    @classmethod
    def tearDownClass(cls) -> None:
        fields.clear_cache()

    def test_lattice_matches_the_run(self) -> None:
        grids = self.grids
        assert grids is not None
        self.assertEqual(grids.lattice.dims, (77, 31, 43))
        self.assertAlmostEqual(grids.lattice.spacing, 0.2, places=6)
        self.assertEqual(grids.voxel_count, 41593)
        self.assertEqual(int(np.count_nonzero(grids.mask)), 41593)
        self.assertEqual(set(np.unique(grids.mask).tolist()), {0, 1, 2})

    def test_modes_include_solidification_time_and_frames(self) -> None:
        grids = self.grids
        assert grids is not None
        meta = fields.build_grid_metadata(grids)

        for mode in ["shrinkage", "porosity", "feed_modulus", "hotspot", "solidification_time"]:
            self.assertIn(mode, meta["modes"], msg=mode)
        self.assertEqual(meta["frames"]["fraction_liquid"]["count"], 93)
        self.assertAlmostEqual(meta["ranges"]["solidification_time"]["min"], 64.2396, places=3)
        self.assertAlmostEqual(meta["ranges"]["solidification_time"]["max"], 731.1978, places=3)
        self.assertNotIn("niyama", meta["modes"])

    def test_envelope_vertices_sit_on_the_threshold(self) -> None:
        grids = self.grids
        assert grids is not None
        grid = grids.scalars["solidification_time"]
        threshold = grid.minimum + 0.5 * (grid.maximum - grid.minimum)
        envelope = fields.build_envelope(grids, "solidification_time", threshold)
        assert envelope is not None and envelope.scalars is not None
        self.assertGreater(envelope.vertex_count, 0)
        span = grid.maximum - grid.minimum
        deviation = np.abs(envelope.scalars - threshold) / span
        self.assertLess(float(np.median(deviation)), 0.1)
        self.assertLess(float(np.percentile(deviation, 90)), 0.25)

    def test_envelope_is_cheap_enough_to_serve_per_request(self) -> None:
        import time

        grids = self.grids
        assert grids is not None
        grid = grids.scalars["solidification_time"]
        threshold = grid.minimum + 0.5 * (grid.maximum - grid.minimum)
        started = time.perf_counter()
        envelope = fields.build_envelope(grids, "solidification_time", threshold)
        elapsed = time.perf_counter() - started
        assert envelope is not None
        self.assertGreater(envelope.triangle_count, 0)
        self.assertLess(elapsed, 2.0)

    def test_no_scalar_grid_bleeds_below_its_own_minimum(self) -> None:
        grids = self.grids
        assert grids is not None
        for mode, grid in grids.scalars.items():
            dense = grid.dequantize()
            tolerance = max((grid.maximum - grid.minimum) / 255.0, 1e-6)
            self.assertGreaterEqual(
                float(np.min(dense)), grid.minimum - tolerance, msg=f"{mode} bled below its minimum"
            )

if __name__ == "__main__":
    unittest.main()
