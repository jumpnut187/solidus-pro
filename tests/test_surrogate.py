"""Tests for the surrogate solidification backend (core/surrogate)."""

from __future__ import annotations

import io
import contextlib
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import trimesh

from core.materials import get_alloy
from core.simulation_pipeline import PipelineInputs, run_pipeline
from core.surrogate import (
    Backend,
    build_conditioning_vector,
    detect_backend,
    encode_channels,
    format_metrics,
    load_model,
    raw_from_conditioning,
    run_surrogate_pipeline,
    save_numpy_demo,
    unproject_grid,
)
from core.surrogate.dataset import DEFAULT_DATASET_DIR, capture_run, load_dataset
from core.surrogate.metrics import aggregate_schedule_metrics, compute_field_metrics, schedule_metrics
from core.surrogate.model import ModelManifest, NumpyDemoModel, NumpyWeightsModel


def _fake_engine(n=300, resolution=16):
    """A minimal engine-like object exposing what the codec needs."""
    rng = np.random.default_rng(0)

    class Mesh:
        def __init__(self):
            self.voxel_centers = rng.uniform(-5, 5, (n, 3))
            self.voxel_moduli = np.full(n, 0.5, dtype=np.float32)
            self.voxel_part_types = np.full(n, 0, dtype=np.int32)

    engine = SimpleNamespace()
    engine.mesh = Mesh()
    engine.surface_distances = np.full(n, 2.0, dtype=np.float32)

    def heights():
        z = engine.mesh.voxel_centers[:, 2]
        return (z - z.min()) / (z.max() - z.min() + 1e-9)

    engine._get_normalized_heights = heights
    return engine


def _loader_for(mesh):
    bounds = np.stack([mesh.bounds[0], mesh.bounds[1]], axis=1)
    return SimpleNamespace(
        trimesh_obj=mesh,
        is_valid=lambda: True,
        volume=float(getattr(mesh, "volume", 0.0) or 0.0),
        get_bounds_array=lambda: bounds,
    )


class ConditioningTests(unittest.TestCase):
    def test_vector_size_and_round_trip(self):
        alloy = get_alloy("WCB")
        cond = build_conditioning_vector(alloy, {"max_dim": 10.0, "voxel_size": 0.1})
        self.assertGreaterEqual(cond.size, 20)
        raw = raw_from_conditioning(cond)
        for name in ("liquidus_temp", "max_dim", "chvorinov_mold_constant"):
            self.assertIn(name, raw)
            self.assertGreater(raw[name], 0.0)


class FieldsCodecTests(unittest.TestCase):
    def test_encode_and_unproject_round_trip(self):
        engine = _fake_engine()
        boost = np.full(len(engine.mesh.voxel_centers), 1.0, dtype=np.float32)
        slow = np.full(len(engine.mesh.voxel_centers), 1.0, dtype=np.float32)
        channels, meta = encode_channels(engine, boost, slow, resolution=16)
        self.assertEqual(channels.shape, (7, 16, 16, 16))
        self.assertEqual(meta.resolution, 16)
        self.assertEqual(list(meta.channels), ["occupancy", "part_type", "surface_distance", "local_modulus", "normalized_height", "chill_boost", "riser_slowdown"])

        back = unproject_grid(channels[3], engine.mesh.voxel_centers, meta)
        self.assertEqual(back.shape, (len(engine.mesh.voxel_centers),))
        self.assertTrue(np.all(np.isfinite(back)))

    def test_occupancy_marks_only_filled_cells(self):
        engine = _fake_engine()
        boost = np.ones(len(engine.mesh.voxel_centers), dtype=np.float32)
        slow = np.ones(len(engine.mesh.voxel_centers), dtype=np.float32)
        channels, meta = encode_channels(engine, boost, slow, resolution=8)
        occ = channels[0]
        self.assertTrue(np.any(occ > 0.5))
        self.assertEqual(set(np.unique(occ)).issubset({0.0, 1.0}), True)

    def test_unproject_uses_scatter_cell_not_rounding(self):
        # Regression: unprojection must fetch the cell the voxel was scattered
        # into (floor), not round to the neighbour cell — rounding drops voxels
        # in the upper half of a cell onto empty cells.
        engine = _fake_engine(n=20)
        centers = np.asarray(engine.mesh.voxel_centers, dtype=np.float64)
        boost = np.ones(len(centers), dtype=np.float32)
        slow = np.ones(len(centers), dtype=np.float32)
        channels, meta = encode_channels(engine, boost, slow, resolution=16)
        cell = meta.cell_size
        bmin = meta.bbox[:, 0]
        res = meta.resolution
        grid = np.zeros((res, res, res), dtype=np.float32)
        floor_idx = np.floor((centers - bmin) / cell).astype(np.int64)
        floor_idx = np.clip(floor_idx, 0, res - 1)
        flat = floor_idx[:, 0] + res * (floor_idx[:, 1] + res * floor_idx[:, 2])
        grid.reshape(-1)[flat] = 1.0
        back = unproject_grid(grid, centers, meta)
        self.assertTrue(np.all(back == 1.0),
                        f"{int(np.sum(back == 0))}/{len(centers)} voxels dropped by rounding")


class ModelTests(unittest.TestCase):
    def test_demo_predict(self):
        model = NumpyDemoModel()
        engine = _fake_engine()
        boost = np.ones(len(engine.mesh.voxel_centers), dtype=np.float32)
        slow = np.ones(len(engine.mesh.voxel_centers), dtype=np.float32)
        channels, meta = encode_channels(engine, boost, slow, resolution=16)
        cond = build_conditioning_vector(get_alloy("WCB"), {"max_dim": 10.0, "voxel_size": 0.1})
        pred = model.predict(channels, cond)
        self.assertEqual(pred.shape, (3, 16, 16, 16))
        self.assertTrue(np.all(np.isfinite(pred)))
        self.assertTrue(np.all(pred[1] >= 0))  # end times non-negative

    def test_load_demo_and_trained(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_numpy_demo(tmp, resolution=16)
            res = load_model(tmp)
            self.assertIsNotNone(res.model, res.error)
            self.assertIsInstance(res.model, NumpyDemoModel)

        res = load_model(os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "surrogate", "demo"))
        self.assertIsNotNone(res.model, res.error)

    def test_weights_fit_and_predict(self):
        engine = _fake_engine()
        boost = np.ones(len(engine.mesh.voxel_centers), dtype=np.float32)
        slow = np.ones(len(engine.mesh.voxel_centers), dtype=np.float32)
        channels, _ = encode_channels(engine, boost, slow, resolution=16)
        cond = build_conditioning_vector(get_alloy("WCB"), {"max_dim": 10.0, "voxel_size": 0.1})
        end = 500 + np.abs(np.random.default_rng(1).normal(0, 100, (16, 16, 16)))

        weights = NumpyWeightsModel._fit_weights([channels], [cond], [end])
        self.assertEqual(weights.shape, (len(NumpyWeightsModel.FEATURE_NAMES),))

        manifest = ModelManifest(name="t", version="0", kind="numpy_weights", backend="numpy", grid_resolution=16)
        model = NumpyWeightsModel(manifest, weights)
        pred = model.predict(channels, cond)
        self.assertEqual(pred.shape, (3, 16, 16, 16))
        self.assertTrue(np.all(np.isfinite(pred)))


class MetricsTests(unittest.TestCase):
    def test_compute_field_metrics(self):
        ref = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.0, 2.5, 2.0])
        m = compute_field_metrics(pred, ref)
        self.assertAlmostEqual(m["mae"], 0.5, places=6)
        self.assertAlmostEqual(m["rmse"], np.sqrt((0.0 ** 2 + 0.5 ** 2 + 1.0 ** 2) / 3.0), places=6)

    def test_aggregate_and_format(self):
        sm = schedule_metrics(
            np.zeros(4), np.array([10.0, 20.0, 30.0, 40.0]), np.zeros(4),
            np.zeros(4), np.array([11.0, 21.0, 31.0, 41.0]), np.zeros(4),
        )
        head = aggregate_schedule_metrics(sm)
        self.assertIn("rmse_s", head)
        self.assertIn("confidence", head)
        self.assertIn("s", format_metrics(head))


class DatasetTests(unittest.TestCase):
    def test_capture_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SimpleNamespace(
                alloy=get_alloy("WCB"),
                mesh=SimpleNamespace(
                    voxel_centers=_fake_engine().mesh.voxel_centers,
                ),
                solidification_times=np.full(300, 100.0, dtype=np.float32),
                solidification_start_times=np.full(300, 50.0, dtype=np.float32),
                local_solidification_time=np.full(300, 50.0, dtype=np.float32),
            )
            engine._calculate_chill_influence = lambda: np.ones(300, dtype=np.float32)
            engine._calculate_riser_influence = lambda: np.ones(300, dtype=np.float32)
            path = capture_run(engine, voxel_size=0.5, resolution=8, out_dir=tmp)
            self.assertIsNotNone(path)
            records = load_dataset(tmp)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["end"].size, 8 ** 3)


class TrainingTests(unittest.TestCase):
    def _make_record(self, out_dir: str, seed: int = 0) -> str:
        rng = np.random.default_rng(seed)
        n = 200
        engine = SimpleNamespace(
            alloy=get_alloy("WCB"),
            mesh=SimpleNamespace(
                voxel_centers=rng.uniform(-4, 4, (n, 3)),
            ),
            solidification_times=np.full(n, 120.0 + seed * 10, dtype=np.float32),
            solidification_start_times=np.full(n, 60.0 + seed * 5, dtype=np.float32),
            local_solidification_time=np.full(n, 60.0 + seed * 5, dtype=np.float32),
        )
        engine._calculate_chill_influence = lambda: np.ones(n, dtype=np.float32)
        engine._calculate_riser_influence = lambda: np.ones(n, dtype=np.float32)
        path = capture_run(engine, voxel_size=0.5, resolution=8, out_dir=out_dir)
        self.assertIsNotNone(path)
        return path

    def test_retrain_from_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            out_dir = os.path.join(tmp, "model")
            for seed in range(5):
                self._make_record(data_dir, seed=seed)

            from core.surrogate.training import retrain_from_dataset

            summary = retrain_from_dataset(
                data_dir=data_dir,
                output_dir=out_dir,
                grid=8,
                name="Test Model",
                generate_first=False,
            )
            self.assertEqual(summary["records"], 5)
            self.assertIn("headline", summary)
            self.assertTrue(os.path.exists(os.path.join(out_dir, "manifest.json")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "weights.npz")))

            res = load_model(out_dir)
            self.assertIsNotNone(res.model, res.error)
            self.assertEqual(res.model.manifest.name, "Test Model")
            self.assertGreaterEqual(res.model.manifest.accuracy.get("confidence", 0.0), 0.0)
            self.assertLessEqual(res.model.manifest.accuracy.get("confidence", 0.0), 1.0)

    def test_retrain_raises_without_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            from core.surrogate.training import retrain_from_dataset

            with self.assertRaises(ValueError):
                retrain_from_dataset(data_dir=tmp, output_dir=tmp, grid=8)


class PipelineTests(unittest.TestCase):
    def test_surrogate_pipeline_end_to_end(self):
        box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        loader = _loader_for(box)
        inputs = PipelineInputs(
            alloy_name="WCB",
            voxel_size=0.5,
            refinement_level=1,
            memory_mode="ram",
            analysis_options={"shrinkage": True, "hotspot": True, "niyama": True, "fraction_liquid": True},
            max_voxels_for_visualization=50_000,
            show_progress=False,
        )
        model = load_model(os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "surrogate", "demo")).model
        self.assertIsNotNone(model)

        with contextlib.redirect_stdout(io.StringIO()):
            result = run_surrogate_pipeline(inputs, model=model, stl_loader=loader)

        engine = result.engine
        self.assertTrue(getattr(engine, "surrogate_run", False))
        self.assertTrue(engine.shrinkage_risk is not None)
        self.assertTrue(engine.hotspot_intensity is not None)
        self.assertTrue(engine.niyama_values is not None)
        self.assertGreater(len(engine.fraction_liquid_history), 1)
        self.assertTrue(np.all(np.isfinite(engine.solidification_times)))
        self.assertIn("solidification_times", result.visualization_data)
        self.assertIn("shrinkage", result.visualization_data)

    def test_surrogate_vs_traditional_parity(self):
        box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        loader = _loader_for(box)
        inputs = PipelineInputs(
            alloy_name="WCB",
            voxel_size=0.5,
            refinement_level=1,
            memory_mode="ram",
            analysis_options={"shrinkage": True, "hotspot": True, "fraction_liquid": True},
            max_voxels_for_visualization=50_000,
            show_progress=False,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            ref = run_pipeline(inputs, stl_loader=loader)
            model = load_model(os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "surrogate", "demo")).model
            sur = run_surrogate_pipeline(inputs, model=model, stl_loader=loader)

        self.assertEqual(set(ref.visualization_data.keys()), set(sur.visualization_data.keys()))
        ref_end = np.asarray(ref.engine.solidification_times, dtype=np.float64)
        sur_end = np.asarray(sur.engine.solidification_times, dtype=np.float64)
        m = compute_field_metrics(sur_end, ref_end)
        # Loose bound: the reference model mirrors Chvorinov physics, so it must
        # be in the right ballpark on a simple box (relative RMSE < 1.0).
        self.assertLess(m["relative_rmse"], 1.0)


    def test_fraction_liquid_window_matches_schedule_completion(self):
        # The fraction-liquid timeline must span exactly the schedule-defined
        # completion (max(start + local)), never the predicted end field —
        # an incoherent end head used to stretch the slider past completion.
        box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        loader = _loader_for(box)
        inputs = PipelineInputs(
            alloy_name="WCB",
            voxel_size=0.5,
            refinement_level=1,
            memory_mode="ram",
            analysis_options={"fraction_liquid": True},
            max_voxels_for_visualization=50_000,
            show_progress=False,
        )
        model = load_model(os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "surrogate", "demo")).model
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_surrogate_pipeline(inputs, model=model, stl_loader=loader)
        eng = result.engine
        start = np.asarray(eng.solidification_start_times, dtype=np.float64)
        local = np.asarray(eng.local_solidification_time, dtype=np.float64)
        active = local > 0.0
        expected_max = float(np.max((start + local)[active]))
        self.assertAlmostEqual(eng.time, expected_max, delta=1e-3)
        self.assertAlmostEqual(eng.get_fraction_liquid_time_range()[1], expected_max, delta=1e-3)
        self.assertAlmostEqual(
            float(np.max(eng.solidification_times)),
            expected_max,
            delta=1e-3,
        )


if __name__ == "__main__":
    unittest.main()