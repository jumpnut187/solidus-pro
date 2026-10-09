"""Surrogate / deep-model capture and inference on tet runs (wave 3, docs/tet_migration).

1. ``encode_channels`` on a tet mesh scatters lattice-sampled values, not the
   raw node cloud, and matches a voxel run of the same part.
2. ``capture_run`` on a tet run writes a record with ``mesh_type == "tet"``;
   ``load_dataset`` and ``DeepDataset`` read it together with voxel records.
3. The index stays append-only and old rows/records read as "voxel".
4. ``run_surrogate_pipeline`` on a tet mesh with a NumPy model fitted here
   returns node-length schedule fields.
"""

from __future__ import annotations

import contextlib
import csv
import io
import os
import tempfile
import unittest
from typing import Any, Sequence, Tuple
from unittest import mock

import numpy as np
import trimesh

from config.settings import Settings
from core import simulation_pipeline as sp
from core import tet_thermal
from core.materials import get_alloy
from core.part_manager import PART_TYPE_CAST, PART_TYPE_CHILL
from core.solidification import SolidificationEngine
from core.surrogate import pipeline as surrogate_pipeline
from core.surrogate.conditioning import build_conditioning_vector
from core.surrogate.dataset import (
    INDEX_FILENAME,
    _config_for_engine,
    capture_run,
    dataset_summary,
    load_dataset,
    load_record,
)
from core.surrogate.fields import (
    PART_TYPE_DIVISOR,
    _scatter,
    encode_channels,
    encoding_points,
    unproject_grid,
)
from core.surrogate.model import ModelManifest, NumpyWeightsModel
from core.tet_mesh import TetVolumeMesh

EDGE_LENGTH_IN = 0.2
#: Grid cell (2.0 in / 8 = 0.25 in) coarser than the 0.2 in pitch, so the voxel
#: path fills every interior grid cell.
RESOLUTION = 8
#: Multiples of the 0.2 in pitch, so voxel centres land on the part faces.
BOX_EXTENTS = (2.0, 1.6, 1.2)
CHILL_EXTENTS = (0.8, 0.8, 0.4)


def _box() -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=BOX_EXTENTS)
    mesh.apply_translation((0.0, 0.0, 0.5 * BOX_EXTENTS[2]))
    return mesh


def _chill() -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=CHILL_EXTENTS)
    mesh.apply_translation((0.6, 0.0, -0.5 * CHILL_EXTENTS[2]))
    return mesh


class _PartManager:
    """The ``PartManager`` members ``prepare_mesh_and_geometry`` reads (as tests/test_tet_pipeline.py)."""

    def __init__(self, parts: Sequence[Tuple[trimesh.Trimesh, int]]) -> None:
        self._parts = list(parts)

    def get_part_count(self) -> int:
        return len(self._parts)

    def get_combined_mesh(self) -> trimesh.Trimesh:
        return trimesh.util.concatenate([m for m, _ in self._parts])

    def get_combined_bounds(self) -> np.ndarray:
        return np.asarray(self.get_combined_mesh().bounds).T

    def get_total_volume(self) -> float:
        return float(sum(m.volume for m, _ in self._parts))

    def get_parts_with_types(self):
        return list(self._parts)


@contextlib.contextmanager
def _default_settings():
    settings = Settings()
    with mock.patch.object(sp, "get_settings", return_value=settings), mock.patch.object(
        tet_thermal, "get_settings", return_value=settings
    ), contextlib.redirect_stdout(io.StringIO()):
        yield


def _inputs(mesh_type: str) -> sp.PipelineInputs:
    return sp.PipelineInputs(
        "WCB",
        EDGE_LENGTH_IN,
        0,
        analysis_options={"shrinkage": True, "hotspot": True},
        mesh_type=mesh_type,
    )


def _geometry_engine(mesh_type: str, parts) -> Tuple[SolidificationEngine, np.ndarray, np.ndarray]:
    """Mesh + initialized engine + t=0 chill/riser fields, as the surrogate pipeline builds them."""
    with _default_settings():
        voxel_mesh = sp.prepare_mesh_and_geometry(
            _inputs(mesh_type), part_manager=_PartManager(parts), stl_loader=None
        )[4]
        options = sp.create_analysis_options({}, use_gpu=False, use_incremental=False)
        engine = SolidificationEngine(voxel_mesh, get_alloy("WCB"), options)
        engine.initialize()
        engine._calculate_feeding_distances()
        chill_boost = engine.compute_initial_chill_influence()
        riser_slowdown = engine._calculate_riser_influence()
    return engine, chill_boost, riser_slowdown


def _run(mesh_type: str) -> Any:
    with _default_settings():
        return sp.run_pipeline(_inputs(mesh_type), part_manager=_PartManager([(_box(), PART_TYPE_CAST)]))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.count_nonzero(a & b)) / float(np.count_nonzero(a | b))


class TetChannelEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        box = [(_box(), PART_TYPE_CAST)]
        cls.voxel = _geometry_engine("voxel", box)
        cls.tet = _geometry_engine("tet", box)

    def _channels(self, case):
        engine, chill_boost, riser_slowdown = case
        return encode_channels(engine, chill_boost, riser_slowdown, resolution=RESOLUTION)

    def test_tet_channels_match_a_voxel_run_of_the_same_box(self) -> None:
        voxel_channels, voxel_meta = self._channels(self.voxel)
        tet_channels, tet_meta = self._channels(self.tet)
        self.assertIsInstance(self.tet[0].mesh, TetVolumeMesh)
        # Same grid frame: voxel centres and tet nodes both span the part box.
        np.testing.assert_allclose(tet_meta.bbox, voxel_meta.bbox, atol=1e-3)

        voxel_occ = voxel_channels[0] > 0.5
        tet_occ = tet_channels[0] > 0.5
        self.assertGreater(_iou(voxel_occ, tet_occ), 0.9)
        overlap = voxel_occ & tet_occ
        np.testing.assert_array_equal(tet_channels[1][overlap], voxel_channels[1][overlap])
        moduli_corr = np.corrcoef(voxel_channels[3][overlap], tet_channels[3][overlap])[0, 1]
        self.assertGreater(moduli_corr, 0.9)

    def test_tet_encoding_uses_lattice_samples_not_nodes(self) -> None:
        engine = self.tet[0]
        mesh = engine.mesh
        points = encoding_points(mesh, RESOLUTION)
        self.assertIsNotNone(points.sampler)
        self.assertLessEqual(points.sampler.spacing, EDGE_LENGTH_IN + 1e-12)
        self.assertNotEqual(points.centers.shape[0], mesh.get_voxel_count())
        channels, meta = self._channels(self.tet)
        # Channel 3 is the scatter of the barycentric samples at the lattice centres.
        expected = _scatter(
            points.centers,
            np.clip(points.values(mesh.voxel_moduli) / meta.max_dim, 0.0, 1.0),
            meta,
        )
        np.testing.assert_allclose(channels[3], expected, atol=1e-6)
        node_scatter = _scatter(
            np.asarray(mesh.voxel_centers, dtype=np.float64),
            np.clip(mesh.voxel_moduli / meta.max_dim, 0.0, 1.0),
            meta,
        )
        self.assertFalse(np.allclose(channels[3], node_scatter, atol=1e-3))
        # Raw nodes only fill boundary cells that hold no lattice centre.
        self.assertLess(points.extra_nodes.size, mesh.get_voxel_count() // 2)

    def test_every_node_unprojects_from_an_occupied_cell(self) -> None:
        parts = [(_box(), PART_TYPE_CAST), (_chill(), PART_TYPE_CHILL)]
        engine, chill_boost, riser_slowdown = _geometry_engine("tet", parts)
        for resolution in (8, 16):
            channels, meta = encode_channels(engine, chill_boost, riser_slowdown, resolution=resolution)
            occupancy = unproject_grid(channels[0], engine.mesh.voxel_centers, meta)
            self.assertTrue(np.all(occupancy > 0.5), resolution)

    def test_points_for_another_resolution_are_refused(self) -> None:
        engine, chill_boost, riser_slowdown = self.tet
        points = encoding_points(engine.mesh, RESOLUTION)
        with self.assertRaises(ValueError):
            encode_channels(engine, chill_boost, riser_slowdown, resolution=2 * RESOLUTION, points=points)

    def test_chill_cells_encode_the_chill_code(self) -> None:
        parts = [(_box(), PART_TYPE_CAST), (_chill(), PART_TYPE_CHILL)]
        voxel_channels, _ = encode_channels(*_geometry_engine("voxel", parts), resolution=RESOLUTION)
        tet_channels, _ = encode_channels(*_geometry_engine("tet", parts), resolution=RESOLUTION)
        chill_code = PART_TYPE_CHILL / PART_TYPE_DIVISOR
        overlap = (voxel_channels[0] > 0.5) & (tet_channels[0] > 0.5)
        self.assertGreater(_iou(voxel_channels[0] > 0.5, tet_channels[0] > 0.5), 0.9)
        tet_chill = np.isclose(tet_channels[1], chill_code) & overlap
        voxel_chill = np.isclose(voxel_channels[1], chill_code) & overlap
        self.assertTrue(np.any(tet_chill))
        # Cells holding only one code agree; cells on the cast/chill interface may mix.
        pure = overlap & np.isin(np.round(tet_channels[1], 6), np.round([0.0, chill_code], 6)) & np.isin(
            np.round(voxel_channels[1], 6), np.round([0.0, chill_code], 6)
        )
        agreement = np.mean(tet_channels[1][pure] == voxel_channels[1][pure])
        self.assertGreater(agreement, 0.9)
        # Voxel centres on the cast/chill interface plane take the higher code
        # (chill), so the voxel chill is one layer thicker; tet chill cells are
        # a subset of it.
        self.assertFalse(np.any(tet_chill & ~voxel_chill))

    def test_tet_conditioning_uses_the_edge_length(self) -> None:
        engine = self.tet[0]
        config = _config_for_engine(engine, voxel_size=0.37)
        self.assertAlmostEqual(config["voxel_size"], engine.mesh.base_voxel_size)
        voxel_config = _config_for_engine(self.voxel[0], voxel_size=0.37)
        self.assertAlmostEqual(voxel_config["voxel_size"], 0.37)
        vector = build_conditioning_vector(get_alloy("WCB"), config).vector
        self.assertEqual(vector.shape, build_conditioning_vector(get_alloy("WCB"), voxel_config).vector.shape)


def _legacy_record(path: str, like: str) -> None:
    """Copy of the record at ``like`` without ``mesh_type``, as captured before wave 3."""
    with np.load(like, allow_pickle=True) as npz:
        data = {k: npz[k] for k in npz.files if k != "mesh_type"}
    np.savez_compressed(path, **data)


class TetCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tet_result = _run("tet")
        cls.voxel_result = _run("voxel")

    def test_capture_writes_tet_record_that_loads_with_voxel_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tet_path = capture_run(self.tet_result.engine, voxel_size=EDGE_LENGTH_IN, resolution=RESOLUTION, out_dir=tmp)
            voxel_path = capture_run(self.voxel_result.engine, voxel_size=EDGE_LENGTH_IN, resolution=RESOLUTION, out_dir=tmp)
            self.assertIsNotNone(tet_path)
            self.assertIsNotNone(voxel_path)
            with np.load(tet_path, allow_pickle=True) as npz:
                self.assertEqual(str(npz["mesh_type"]), "tet")
            legacy_path = os.path.join(tmp, "run_legacy000000.npz")
            _legacy_record(legacy_path, voxel_path)

            skipped = []
            records = load_dataset(tmp, skipped=skipped)
            self.assertEqual(skipped, [])
            by_path = {r["path"]: r for r in records}
            self.assertEqual(by_path[tet_path]["mesh_type"], "tet")
            self.assertEqual(by_path[voxel_path]["mesh_type"], "voxel")
            self.assertEqual(by_path[legacy_path]["mesh_type"], "voxel")
            for record in records:
                self.assertEqual(record["channels"].shape, (7, RESOLUTION, RESOLUTION, RESOLUTION))
                self.assertEqual(record["conditioning"].vector.shape, records[0]["conditioning"].vector.shape)

            tet_record = by_path[tet_path]
            active = tet_record["channels"][0].reshape(-1) > 0.5
            self.assertTrue(np.all(np.isfinite(tet_record["end"].reshape(-1)[active])))
            # Targets come from the same points as the channels: every occupied cell has a time.
            self.assertTrue(np.all(tet_record["end"].reshape(-1)[active] > 0.0))
            self.assertEqual(int(np.count_nonzero(tet_record["end"].reshape(-1)[~active])), 0)

            summary = dataset_summary(tmp)
            self.assertEqual([r["mesh_type"] for r in summary["runs"]], ["tet", "voxel"])

            try:
                from core.deep_model.dataset import DeepDataset
            except ImportError:  # torch absent
                return
            dataset = DeepDataset(records, resolution=RESOLUTION)
            self.assertEqual(len(dataset), 3)
            item = dataset[0]
            self.assertEqual(tuple(item.channels.shape), (7, RESOLUTION, RESOLUTION, RESOLUTION))
            self.assertTrue(all(bool(dataset[i].mask.any()) for i in range(len(dataset))))

    def test_legacy_index_is_appended_never_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            legacy = (
                "sample_id,timestamp,alloy,voxel_count,resolution,path\r\n"
                "aaaaaaaaaaaa,2026-01-01 00:00:00,WCB,125,64,/old/run_aaaaaaaaaaaa.npz\r\n"
            )
            index_path = os.path.join(tmp, INDEX_FILENAME)
            with open(index_path, "w", newline="") as fh:
                fh.write(legacy)
            path = capture_run(self.tet_result.engine, voxel_size=EDGE_LENGTH_IN, resolution=RESOLUTION, out_dir=tmp)
            self.assertIsNotNone(path)
            with open(index_path, newline="") as fh:
                text = fh.read()
            self.assertTrue(text.startswith(legacy))
            runs = dataset_summary(tmp)["runs"]
            self.assertEqual(len(runs), 2)
            self.assertEqual(runs[0]["mesh_type"], "voxel")
            self.assertEqual(runs[0]["sample_id"], "aaaaaaaaaaaa")
            self.assertEqual(runs[1]["mesh_type"], "tet")
            self.assertEqual(runs[1]["path"], path)
            self.assertNotIn(None, runs[1])

    def test_fresh_index_has_a_mesh_type_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_run(self.voxel_result.engine, voxel_size=EDGE_LENGTH_IN, resolution=RESOLUTION, out_dir=tmp)
            with open(os.path.join(tmp, INDEX_FILENAME), newline="") as fh:
                header = next(csv.reader(fh))
            self.assertEqual(header[-1], "mesh_type")
            self.assertEqual(header[:6], ["sample_id", "timestamp", "alloy", "voxel_count", "resolution", "path"])

    def test_surrogate_pipeline_on_tet_returns_node_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for result in (self.tet_result, self.voxel_result):
                self.assertIsNotNone(
                    capture_run(result.engine, voxel_size=EDGE_LENGTH_IN, resolution=RESOLUTION, out_dir=tmp)
                )
            records = load_dataset(tmp)
        channels = [r["channels"] for r in records]
        conditioning = [r["conditioning"] for r in records]
        model = NumpyWeightsModel(
            ModelManifest(name="tet-ml-test", version="0", kind="numpy_weights", backend="numpy", grid_resolution=RESOLUTION),
            NumpyWeightsModel._fit_weights(channels, conditioning, [r["end"] for r in records]),
            NumpyWeightsModel._fit_local_weights(channels, conditioning, [r["local"] for r in records]),
        )
        with _default_settings():
            result = surrogate_pipeline.run_surrogate_pipeline(
                _inputs("tet"), model=model, part_manager=_PartManager([(_box(), PART_TYPE_CAST)])
            )
        mesh = result.voxel_mesh
        self.assertIsInstance(mesh, TetVolumeMesh)
        n = mesh.get_voxel_count()
        engine = result.engine
        self.assertEqual(engine.solidification_start_times.shape, (n,))
        self.assertEqual(engine.local_solidification_time.shape, (n,))
        self.assertTrue(np.all(np.isfinite(engine.solidification_start_times)))
        self.assertTrue(np.all(engine.local_solidification_time > 0.0))
        self.assertEqual(len(result.visualization_data["solidification_times"]), n)


if __name__ == "__main__":
    unittest.main()
