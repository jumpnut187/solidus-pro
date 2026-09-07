"""
Tests for the dense field grids and the endpoints that serve them.

The fixture is a 6 x 5 x 4 lattice on a deliberately awkward origin and
spacing, with two part types and a pair of interior holes, so that a bug in
the index arithmetic or the hole filling shows up as a wrong number rather
than as a plausible-looking grid.
"""

from __future__ import annotations

import gzip
import json
import struct
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from flask import Flask

from web import fields
from web import repository as repo
from web.payloads import write_artifact
from web.routes_fields import fields_bp
from web.services.simulation_queue import SimulationQueueManager


DIMS = (6, 5, 4)
ORIGIN = (-1.25, 2.5, 0.75)
SPACING = 0.25
FRAME_COUNT = 3
FRAME_TIMES = [0.0, 4.5, 9.0]

#: Cells left empty inside the bounding box, so the lattice is not a solid
#: brick and the nearest-inside fill has something to do.
HOLES = {(4, 1, 0), (4, 2, 0), (1, 3, 2)}

#: The real completed run the contract points at. Present in a seeded
#: worktree, absent in a bare checkout.
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
    """Strictly positive and varying on every axis, so a bleed to zero is loud."""
    return 5.0 + ix + 10.0 * iy + 100.0 * iz


def _synthetic_artifact() -> Dict[str, Any]:
    cells = _occupied_cells()
    centers = [_cell_center(*cell) for cell in cells]
    shrinkage = [_shrinkage_value(*cell) for cell in cells]
    # Cast in the lower half, riser above it: two mask codes, not one.
    part_types = [0 if iz < 2 else 1 for (_, _, iz) in cells]
    solidification = [20.0 + 5.0 * iz for (_, _, iz) in cells]

    history: List[List[float]] = []
    for frame_index in range(FRAME_COUNT):
        remaining = 1.0 - frame_index / (FRAME_COUNT - 1)
        history.append([float(np.clip(remaining, 0.0, 1.0))] * len(cells))

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
        },
    }


def _flat_index(ix: int, iy: int, iz: int) -> int:
    nx, ny, _ = DIMS
    return ix + nx * (iy + ny * iz)


def _brute_force_nearest_values(
    values_by_cell: Dict[Tuple[int, int, int], float],
    target: Tuple[int, int, int],
) -> List[float]:
    """
    Every value that could legitimately fill ``target``.

    A hole equidistant from several occupied cells has no single right answer
    -- which one the Euclidean distance transform picks among ties is its own
    business -- so the assertion is membership in the tied set, not equality
    with one arbitrary member.
    """
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
        """
        Counting occupied planes instead of measuring the extent would shear
        the grid: a casting with a horizontal slot has a plane with no metal
        in it, and every cell above the slot would shift down one layer.
        """
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

        # The whole point of the fill: no cell anywhere sits below the field's
        # true minimum, so trilinear sampling cannot fade a surface toward the
        # bottom of the colormap at the metal boundary.
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

        # Nested summary statistics arrive flat, which is what GridMeta.stats is.
        self.assertEqual(meta["stats"]["shrinkage_max"], 5.0)
        self.assertEqual(meta["stats"]["voxel_count"], len(_occupied_cells()))

    def test_constant_field_is_reported_but_not_offered_as_a_mode(self) -> None:
        """A mode the run produced as all zeros has nothing to paint."""
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

        # Same file, same mtime: the LRU must hand back the very same object
        # rather than re-reading anything.
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
        # Force a distinct mtime even on a coarse-grained filesystem clock.
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

        # Little-endian is the wire contract; unpack it explicitly rather than
        # trusting the host's byte order to be the same one.
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


@unittest.skipUnless(
    REAL_ARTIFACT.exists(),
    f"Seeded run artifact not present at {REAL_ARTIFACT}",
)
class RealArtifactTests(unittest.TestCase):
    """
    The synthetic fixture proves the arithmetic; this proves the assumptions.

    A real run is where a wrong guess about the artifact's key names, its
    part-type codes or the uniformity of its lattice actually shows up.
    """

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
        # niyama came back identically zero on this run; it must not be
        # offered as something to look at.
        self.assertNotIn("niyama", meta["modes"])

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
