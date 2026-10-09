"""Web sidecar, desktop result surface and inspectors on a tet run (wave 3, docs/tet_migration).

One real tet pipeline run of the small casting + riser + chill assembly from
``tests/test_tet_end_to_end.py`` (edge 0.2 in) feeds every test:

1. ``build_fork_artifact`` / ``write_artifact`` / ``load_grids``: the sidecar
   lattice is the ``TetLatticeSampler`` lattice, the mask is its
   ``mask_codes()``, node fields are sampled barycentrically and the frame
   count is the engine's snapshot count.
2. The grid endpoints: a tet run serves its grid, and a tet run whose sidecar
   is gone answers 404 ``tet_grid_missing`` instead of a wrong lattice.
3. ``ResultSurfaceRenderer.set_tet_mesh``: picking returns nodes, ``show``
   takes node-length arrays only.
4. ``CastingVisualizer``: the tet branch of ``set_result_mesh``, the
   inspector's ``mesh_type`` and the node-volume defect volume.

fTetWild is not bit-deterministic, so geometric checks use tolerances.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import vtk  # noqa: E402
from flask import Flask  # noqa: E402

from config.settings import Settings  # noqa: E402
from core.dense_grid import sample_node_values, tet_presentation_spacing  # noqa: E402
from core.part_manager import PART_TYPE_CAST, is_metal  # noqa: E402
from core.tet_lattice import TetLatticeSampler  # noqa: E402
from tests.test_tet_end_to_end import EDGE_LENGTH_IN, _assembly, _run  # noqa: E402
from ui.result_surface import ResultSurfaceRenderer, palette_hotspot  # noqa: E402
from web import fields  # noqa: E402
from web import repository as repo  # noqa: E402
from web.payloads import GRID_SOURCE_KEY, build_fork_artifact, write_artifact  # noqa: E402
from web.routes_fields import fields_bp  # noqa: E402
from web.services.simulation_queue import SimulationQueueManager  # noqa: E402

_RUN: Dict[str, Any] = {}


def setUpModule() -> None:
    result, _ = _run("tet", part_manager=_assembly())
    _RUN["result"] = result


def _result():
    return _RUN["result"]


def _linear(points: np.ndarray) -> np.ndarray:
    """A linear field (s) over the assembly: 10..~40 across the box."""
    p = np.asarray(points, dtype=np.float64)
    return 20.0 + 3.0 * p[:, 0] - 2.0 * p[:, 1] + 5.0 * p[:, 2]


def _artifact(linear_field: bool = True) -> Dict[str, Any]:
    """``build_fork_artifact`` under default settings, optionally with a linear solidification time."""
    with mock.patch("config.settings.get_settings", return_value=Settings()):
        artifact = build_fork_artifact(_result())
    if linear_field:
        nodes = np.asarray(_result().voxel_mesh.node_xyz)
        artifact[GRID_SOURCE_KEY]["solidification_times"] = _linear(nodes).astype(np.float32)
    return artifact


def _sampler() -> TetLatticeSampler:
    mesh = _result().voxel_mesh
    return TetLatticeSampler(mesh.node_xyz, mesh.tets, mesh.tet_part_types, EDGE_LENGTH_IN)


class TetSidecarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fields.clear_cache()
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.artifact = _artifact()
        cls.grid_source = cls.artifact[GRID_SOURCE_KEY]
        cls.visualization = cls.artifact["visualization"]
        cls.artifact_path = Path(cls.temp_dir.name) / "tet_run.json"
        write_artifact(cls.artifact_path, cls.artifact)
        cls.grids = fields.load_grids("tet", cls.artifact_path)
        cls.sampler = _sampler()

    @classmethod
    def tearDownClass(cls) -> None:
        fields.clear_cache()
        cls.temp_dir.cleanup()

    def test_grid_source_carries_tet_connectivity_as_numpy(self) -> None:
        mesh = _result().voxel_mesh
        self.assertEqual(self.grid_source["mesh_type"], "tet")
        self.assertEqual(self.grid_source["tets"].dtype, np.int32)
        self.assertEqual(self.grid_source["tets"].shape, mesh.tets.shape)
        self.assertEqual(self.grid_source["tet_part_types"].dtype, np.int8)
        self.assertAlmostEqual(self.grid_source["lattice_spacing"], EDGE_LENGTH_IN)
        self.assertEqual(len(self.grid_source["centers"]), mesh.get_voxel_count())

    def test_json_keeps_a_node_preview_and_never_the_tets(self) -> None:
        on_disk = json.loads(self.artifact_path.read_text(encoding="utf-8"))
        self.assertNotIn(GRID_SOURCE_KEY, on_disk)
        visualization = on_disk["visualization"]
        self.assertEqual(visualization["mesh_type"], "tet")
        self.assertNotIn("tets", visualization)
        n = _result().voxel_mesh.get_voxel_count()
        self.assertEqual(len(visualization["centers"]), n)
        self.assertEqual(len(visualization["part_types"]), n)

    def test_sidecar_lattice_is_the_sampler_lattice(self) -> None:
        self.assertIsNotNone(self.grids)
        self.assertEqual(self.grids.mesh_type, "tet")
        self.assertEqual(self.grids.lattice.dims, self.sampler.dims)
        self.assertAlmostEqual(self.grids.lattice.spacing, EDGE_LENGTH_IN)
        np.testing.assert_allclose(self.grids.lattice.origin, self.sampler.origin, atol=1e-9)
        meta = fields.build_grid_metadata(self.grids)
        self.assertEqual(meta["dims"], list(self.sampler.dims))
        self.assertEqual(meta["quantization"], {"dtype": "uint8", "levels": 256})
        self.assertEqual(meta["voxelCount"], _result().voxel_mesh.get_voxel_count())

    def test_mask_codes_match_tet_part_types(self) -> None:
        np.testing.assert_array_equal(self.grids.mask, self.sampler.mask_codes())
        inside_codes = set(np.unique(self.grids.mask[self.grids.mask > 0]).tolist())
        tet_codes = set((np.unique(_result().voxel_mesh.tet_part_types).astype(int) + 1).tolist())
        self.assertEqual(inside_codes, tet_codes)
        # Every mask code has a part-type row.
        table_codes = {row["code"] for row in fields.build_grid_metadata(self.grids)["partTypes"]}
        self.assertTrue(inside_codes <= table_codes)

    def test_a_linear_node_field_is_reproduced_at_inside_cells(self) -> None:
        grid = self.grids.scalars["solidification_time"]
        dense = grid.dequantize()
        expected = _linear(self.sampler.cell_centers)
        got = dense[self.sampler.cell_flat]
        quantum = (grid.maximum - grid.minimum) / 255.0
        self.assertLessEqual(float(np.max(np.abs(got - expected))), 0.5 * quantum + 1e-4)

    def test_frame_count_equals_the_snapshot_count(self) -> None:
        engine = _result().engine
        self.assertGreater(len(engine.fraction_liquid_history), 0)
        stack = self.grids.stacks["fraction_liquid"]
        self.assertEqual(stack.count, len(engine.fraction_liquid_history))
        self.assertEqual(stack.frames.shape, (stack.count, self.sampler.cell_count))
        self.assertEqual(stack.frames.dtype, np.uint8)
        np.testing.assert_allclose(stack.times_s, engine.fraction_liquid_times, rtol=1e-6)

    def test_frames_match_barycentric_samples(self) -> None:
        engine = _result().engine
        index = len(engine.fraction_liquid_history) // 2
        node_frame = np.asarray(self.grid_source["fraction_liquid_history"][index], dtype=np.float64)
        finite = np.isfinite(node_frame)
        cell_values, known = sample_node_values(
            self.sampler, np.where(finite, np.clip(node_frame, 0.0, 1.0), 0.0), finite
        )
        frame = self.grids.stacks["fraction_liquid"].frames[index][self.sampler.cell_flat]
        np.testing.assert_allclose(frame[known] / 255.0, cell_values[known], atol=0.5 / 255.0 + 1e-6)

    def test_sample_field_at_a_node_returns_the_node_value(self) -> None:
        mesh = _result().voxel_mesh
        interior = np.flatnonzero(np.asarray(mesh.voxel_moduli) > 2.0 * EDGE_LENGTH_IN)
        self.assertGreater(interior.size, 10)
        points = np.asarray(mesh.node_xyz)[interior]
        sampled = fields.sample_field_at(self.grids, "solidification_time", points)
        grid = self.grids.scalars["solidification_time"]
        quantum = (grid.maximum - grid.minimum) / 255.0
        np.testing.assert_allclose(sampled, _linear(points), atol=quantum + 1e-3)

    def test_envelope_closes_inside_the_mesh(self) -> None:
        grid = self.grids.scalars["solidification_time"]
        threshold = 0.5 * (grid.minimum + grid.maximum)
        envelope = fields.build_envelope(self.grids, "solidification_time", threshold)
        self.assertIsNotNone(envelope)
        self.assertGreater(envelope.triangle_count, 0)
        lo, hi = _result().voxel_mesh.bounds[:, 0], _result().voxel_mesh.bounds[:, 1]
        self.assertTrue(np.all(envelope.positions >= lo - EDGE_LENGTH_IN))
        self.assertTrue(np.all(envelope.positions <= hi + EDGE_LENGTH_IN))
        # Vertices where the contour closes at the metal boundary carry the
        # field there, not the threshold, so only check the scalars exist.
        self.assertEqual(envelope.scalars.shape, (envelope.vertex_count,))
        self.assertTrue(np.all(np.isfinite(envelope.scalars)))

    def test_sidecar_round_trip_keeps_mesh_type(self) -> None:
        path = fields.sidecar_path(self.artifact_path)
        loaded = fields.load_sidecar(path)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.mesh_type, "tet")
        self.assertAlmostEqual(loaded.lattice.spacing, EDGE_LENGTH_IN)

    def test_fill_counts_are_labelled_as_nodes(self) -> None:
        summary = {"fill_filled_voxel_count": 5, "fill_unfilled_voxel_count": 1,
                   "fill_time_stats": {"max_fill_time_s": 1.0, "filled_voxels": 5, "unfilled_voxels": 1}}
        result = _result()
        fake = SimpleNamespace(
            visualization_data=result.visualization_data,
            engine=result.engine,
            voxel_mesh=result.voxel_mesh,
            result_data={**result.result_data, **summary},
        )
        with mock.patch("config.settings.get_settings", return_value=Settings()):
            labelled = build_fork_artifact(fake)["summary"]
        self.assertNotIn("fill_filled_voxel_count", labelled)
        self.assertEqual(labelled["fill_nodes_filled_count"], 5)
        self.assertEqual(labelled["fill_nodes_unfilled_count"], 1)
        self.assertEqual(labelled["fill_time_stats"]["filled_nodes"], 5)


class TetGridEndpointTests(unittest.TestCase):
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

        artifact = _artifact(linear_field=False)
        self.project_id = repo.create_project(self.database_path, {"name": "Tet grid"})
        self.artifact_path = self.temp_folder / f"{self.project_id}.json"
        write_artifact(self.artifact_path, artifact)
        repo.mark_project_completed(self.database_path, self.project_id, artifact["summary"], str(self.artifact_path))

    def tearDown(self) -> None:
        fields.clear_cache()
        self.temp_dir.cleanup()

    def test_metadata_endpoint_serves_the_sampled_lattice(self) -> None:
        response = self.client.get(f"/api/projects/{self.project_id}/grid")
        self.assertEqual(response.status_code, 200)
        meta = response.get_json()
        self.assertEqual(meta["dims"], list(_sampler().dims))
        self.assertAlmostEqual(meta["spacing"], EDGE_LENGTH_IN)
        self.assertIn("solidification_time", meta["modes"])

    def test_a_tet_run_without_its_sidecar_asks_for_a_rerun(self) -> None:
        fields.sidecar_path(self.artifact_path).unlink()
        fields.clear_cache()
        with self.assertRaises(fields.TetGridUnavailableError):
            fields.load_grids(self.project_id, self.artifact_path)
        response = self.client.get(f"/api/projects/{self.project_id}/grid")
        self.assertEqual(response.status_code, 404)
        body = response.get_json()
        self.assertEqual(body["code"], "tet_grid_missing")
        self.assertIn("Re-run", body["error"])
        self.assertFalse(fields.sidecar_path(self.artifact_path).exists())


class TetResultSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mesh = _result().voxel_mesh
        cls.renderer = ResultSurfaceRenderer(vtk.vtkRenderer())
        cls.ok = cls.renderer.set_tet_mesh(cls.mesh.node_xyz, cls.mesh.tets, cls.mesh.tet_part_types, EDGE_LENGTH_IN)

    def _show(self, values: np.ndarray, **kwargs) -> bool:
        return self.renderer.show("hotspot", values, vmin=0.0, vmax=1.0, iso=0.5, palette=palette_hotspot, **kwargs)

    def test_lattice_is_the_sampler_lattice(self) -> None:
        self.assertTrue(self.ok)
        sampler = _sampler()
        self.assertEqual(self.renderer.lattice.dims, sampler.dims)
        self.assertAlmostEqual(self.renderer.lattice.spacing, EDGE_LENGTH_IN)

    def test_voxel_at_a_node_returns_that_node_or_one_an_edge_away(self) -> None:
        nodes = np.asarray(self.mesh.node_xyz)
        neighbors = np.asarray(self.mesh.voxel_neighbors)
        sizes = np.asarray(self.mesh.voxel_sizes, dtype=np.float64)
        interior = np.flatnonzero(np.asarray(self.mesh.voxel_moduli) > EDGE_LENGTH_IN)
        self.assertGreater(interior.size, 20)
        near = 0
        for node in interior:
            picked = self.renderer.voxel_at(nodes[node])
            self.assertGreaterEqual(picked, 0)
            distance = float(np.linalg.norm(nodes[picked] - nodes[node]))
            self.assertLessEqual(distance, 2.0 * sizes[node], f"node {node} picked {picked}")
            near += int(picked == node or picked in neighbors[node])
        self.assertGreaterEqual(near / interior.size, 0.9)

    def test_show_takes_node_arrays_and_refuses_other_lengths(self) -> None:
        n = self.mesh.get_voxel_count()
        nodes = np.asarray(self.mesh.node_xyz)
        values = (nodes[:, 2] - nodes[:, 2].min()) / np.ptp(nodes[:, 2])
        self.assertTrue(self._show(values))
        self.assertGreater(self.renderer.envelope_triangle_count, 0)
        self.assertEqual(len(self.renderer.view.values), n)
        self.assertFalse(self._show(values[:-1]))
        self.assertFalse(self._show(np.zeros(len(self.renderer._cell_of_voxel))))

    def test_node_flags_and_nan_values_follow_nearest_nodes(self) -> None:
        nodes = np.asarray(self.mesh.node_xyz)
        values = (nodes[:, 2] - nodes[:, 2].min()) / np.ptp(nodes[:, 2])
        metal = is_metal(np.asarray(self.mesh.voxel_part_types, dtype=np.int32))
        with_nan = np.where(metal, values, np.nan)
        cast_only = np.asarray(self.mesh.voxel_part_types) == PART_TYPE_CAST
        self.assertTrue(self._show(with_nan, valid=cast_only, flat_paint={"cast": (cast_only, (1, 2, 3))}))
        self.renderer.set_display_mask(np.ones(self.mesh.get_voxel_count(), dtype=bool))
        self.assertIsNotNone(self.renderer._display_mask)
        self.renderer.set_display_mask(np.ones(3, dtype=bool))
        self.assertIsNone(self.renderer._display_mask)


class TetVisualizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from ui import visualizer

        cls.visualizer = visualizer
        cls.mesh = _result().voxel_mesh

    def _bare(self) -> SimpleNamespace:
        return SimpleNamespace(
            result_surface=ResultSurfaceRenderer(vtk.vtkRenderer()),
            _result_lattice_key=None,
            _result_point_volumes=None,
            _voxel_display_mask=None,
            _configure_ssao=lambda: None,
        )

    def test_set_result_mesh_builds_the_tet_lattice(self) -> None:
        viz = self._bare()
        with mock.patch("config.settings.get_settings", return_value=Settings()):
            ok = self.visualizer.CastingVisualizer.set_result_mesh(viz, self.mesh)
        self.assertTrue(ok)
        self.assertIsNotNone(viz.result_surface._sampler)
        self.assertEqual(viz.result_surface.lattice.dims, _sampler().dims)
        np.testing.assert_allclose(viz._result_point_volumes, self.mesh.voxel_volumes)

    def test_defect_volume_sums_node_volumes(self) -> None:
        volumes = np.asarray(self.mesh.voxel_volumes, dtype=np.float64)
        intensity = np.zeros(len(volumes), dtype=np.float32)
        intensity[: len(volumes) // 3] = 1.0
        viz = SimpleNamespace(
            current_shrinkage_data=(self.mesh.voxel_centers, self.mesh.voxel_sizes, intensity),
            _result_point_volumes=volumes,
            measurement_tool=self.visualizer.MeasurementTool.__new__(self.visualizer.MeasurementTool),
        )
        got = self.visualizer.CastingVisualizer.calculate_defect_volume(viz, 0.5)
        self.assertAlmostEqual(got, float(volumes[: len(volumes) // 3].sum()), places=6)

    def test_shrink_inspector_passes_the_mesh_type(self) -> None:
        surface = SimpleNamespace(view=object(), pick_actors=lambda: [], voxel_at=lambda position: 0)
        viz = SimpleNamespace(
            result_surface=surface,
            _shrink_inspector_data={
                "mesh": self.mesh,
                "display_intensity": np.ones(self.mesh.get_voxel_count(), dtype=np.float32),
                "shrinkage_risk": None,
                "porosity_risk": None,
                "threshold": 0.5,
                "connectivity": 6,
            },
            _inspector_layer=None,
        )
        with mock.patch("core.defect_regions.extract_pocket_at", return_value=None) as extract:
            self.visualizer.CastingVisualizer._install_shrink_inspector(viz)
            self.assertIsNone(viz._inspector_layer.resolver(3))
        kwargs = extract.call_args.kwargs
        self.assertEqual(kwargs["mesh_type"], "tet")
        self.assertIs(kwargs["voxel_neighbors"], self.mesh.voxel_neighbors)
        self.assertIs(kwargs["voxel_volumes"], self.mesh.voxel_volumes)


if __name__ == "__main__":
    unittest.main()
