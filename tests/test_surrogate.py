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
from core.surrogate.metrics import (
    aggregate_schedule_metrics,
    compute_field_metrics,
    masked_schedule_metrics,
    schedule_evaluation_mask,
    schedule_metrics,
    skill_score,
)
from core.surrogate.model import ModelManifest, NumpyDemoModel, NumpyWeightsModel, _mushy_fraction
from core.surrogate.split import geometry_group_key, grouped_train_val_indices


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

    def _fit_inputs(self):
        engine = _fake_engine()
        boost = np.ones(len(engine.mesh.voxel_centers), dtype=np.float32)
        slow = np.ones(len(engine.mesh.voxel_centers), dtype=np.float32)
        channels, _ = encode_channels(engine, boost, slow, resolution=16)
        cond = build_conditioning_vector(get_alloy("WCB"), {"max_dim": 10.0, "voxel_size": 0.1})
        rng = np.random.default_rng(1)
        end = 500 + np.abs(rng.normal(0, 100, (16, 16, 16)))
        local = 0.1 * end
        return channels, cond, end, local

    def test_local_head_is_learned_and_bounded_by_end(self):
        channels, cond, end, local = self._fit_inputs()
        weights = NumpyWeightsModel._fit_weights([channels], [cond], [end])
        local_weights = NumpyWeightsModel._fit_local_weights([channels], [cond], [local])
        self.assertEqual(local_weights.shape, weights.shape)
        manifest = ModelManifest(name="t", version="0", kind="numpy_weights", backend="numpy", grid_resolution=16)
        model = NumpyWeightsModel(manifest, weights, local_weights)
        self.assertTrue(model.has_local_head)
        pred = model.predict(channels, cond)
        active = channels[0] > 0.5
        start, pred_end, pred_local = pred
        self.assertTrue(np.all(pred_local[active] <= pred_end[active] + 1e-4))
        self.assertTrue(np.all(pred_local[active] >= 0))
        np.testing.assert_allclose(start[active], pred_end[active] - pred_local[active], atol=1e-3)
        self.assertTrue(np.all(pred[:, ~active] == 0))
        # The learned head tracks the 10% local/end ratio, not the mushy fraction (~0.27).
        ratio = np.median(pred_local[active] / pred_end[active])
        self.assertLess(ratio, 0.2)
        self.assertIn("learned", model.describe())

    def test_missing_local_weights_falls_back_to_mushy_fraction(self):
        channels, cond, end, _ = self._fit_inputs()
        weights = NumpyWeightsModel._fit_weights([channels], [cond], [end])
        manifest = ModelManifest(name="old", version="0", kind="numpy_weights", backend="numpy", grid_resolution=16)
        with tempfile.TemporaryDirectory() as tmp:
            from core.surrogate.model import _save_manifest

            _save_manifest(manifest, tmp)
            # Artifact layout from before the local head existed: weights only.
            np.savez_compressed(os.path.join(tmp, "weights.npz"), weights=weights)
            res = load_model(tmp)
            self.assertIsNotNone(res.model, res.error)
            self.assertFalse(res.model.has_local_head)
            pred = res.model.predict(channels, cond)
            active = channels[0] > 0.5
            mushy = _mushy_fraction(raw_from_conditioning(cond))
            np.testing.assert_allclose(pred[2][active], pred[1][active] * mushy, rtol=1e-5)
            self.assertIn("mushy_fraction", res.model.describe())

    def test_load_reads_local_weights(self):
        channels, cond, end, local = self._fit_inputs()
        weights = NumpyWeightsModel._fit_weights([channels], [cond], [end])
        local_weights = NumpyWeightsModel._fit_local_weights([channels], [cond], [local])
        manifest = ModelManifest(name="new", version="0", kind="numpy_weights", backend="numpy", grid_resolution=16)
        with tempfile.TemporaryDirectory() as tmp:
            from core.surrogate.model import _save_manifest

            _save_manifest(manifest, tmp)
            np.savez_compressed(os.path.join(tmp, "weights.npz"), weights=weights, local_weights=local_weights)
            res = load_model(tmp)
            self.assertIsNotNone(res.model, res.error)
            self.assertTrue(res.model.has_local_head)
            np.testing.assert_allclose(res.model.local_weights, local_weights)


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

    def test_baseline_and_skill(self):
        ref = np.array([10.0, 20.0, 30.0, 40.0])
        # Constant-mean baseline RMSE is the population std (11.18 s).
        m_perfect = compute_field_metrics(ref, ref)
        self.assertAlmostEqual(m_perfect["baseline_rmse"], float(np.std(ref)), places=6)
        self.assertAlmostEqual(m_perfect["skill"], 1.0, places=6)
        m_mean = compute_field_metrics(np.full(4, ref.mean()), ref)
        self.assertAlmostEqual(m_mean["skill"], 0.0, places=6)
        m_worse = compute_field_metrics(np.full(4, 1000.0), ref)
        self.assertEqual(m_worse["skill"], 0.0)  # clipped, never negative
        self.assertEqual(skill_score(5.0, 10.0), 0.5)
        self.assertTrue(np.isnan(skill_score(float("nan"), 10.0)))

        sm = schedule_metrics(np.zeros(4), ref + 1.0, np.zeros(4), np.zeros(4), ref, np.zeros(4))
        head = aggregate_schedule_metrics(sm)
        self.assertAlmostEqual(head["baseline_rmse_s"], float(np.std(ref)), places=6)
        self.assertAlmostEqual(head["end_rmse_s"], 1.0, places=6)
        self.assertAlmostEqual(head["skill"], round(1.0 - 1.0 / float(np.std(ref)), 3), places=6)
        self.assertEqual(head["confidence"], head["skill"])
        self.assertIn("baseline", format_metrics(head))

    def test_aggregate_empty_is_nan(self):
        head = aggregate_schedule_metrics({})
        self.assertTrue(np.isnan(head["rmse_s"]))
        self.assertTrue(np.isnan(head["baseline_rmse_s"]))
        self.assertTrue(np.isnan(head["confidence"]))
        self.assertEqual(format_metrics(head), "accuracy unknown")

    def test_masked_schedule_metrics_ignores_unmasked_cells(self):
        pred = np.zeros((3, 2, 2, 2), dtype=np.float32)
        pred[1] = 100.0
        ref_end = np.zeros(8)
        ref_end[:3] = [100.0, 110.0, 90.0]
        mask = np.zeros(8, dtype=bool)
        mask[:3] = True
        m = masked_schedule_metrics(pred, np.zeros(8), ref_end, np.zeros(8), mask)
        self.assertEqual(m["end_time"]["n_compared"], 3)
        self.assertAlmostEqual(m["end_time"]["rmse"], np.sqrt((0 + 100 + 100) / 3.0), places=6)
        # Unmasked comparison would be dominated by 5 zero-vs-100 cells.
        full = compute_field_metrics(pred[1].reshape(-1), ref_end)
        self.assertGreater(full["rmse"], m["end_time"]["rmse"])

        empty = masked_schedule_metrics(pred, np.zeros(8), ref_end, np.zeros(8), np.zeros(8, dtype=bool))
        self.assertEqual(empty["end_time"]["n_compared"], 0)
        self.assertTrue(np.isnan(empty["end_time"]["rmse"]))

    def test_schedule_evaluation_mask_excludes_chill_and_zero_end(self):
        from core.part_manager import PART_TYPE_CHILL
        from core.surrogate.fields import PART_TYPE_DIVISOR

        channels = np.zeros((7, 2, 2, 2), dtype=np.float32)
        channels[0, 0] = 1.0  # first 4 cells active
        channels[1, 0, 0, 0] = PART_TYPE_CHILL / PART_TYPE_DIVISOR  # one chill cell
        ref_end = np.ones(8)
        ref_end[1] = 0.0  # active, non-chill, but end == 0
        mask = schedule_evaluation_mask(channels, ref_end)
        self.assertEqual(mask.shape, (8,))
        self.assertEqual(int(mask.sum()), 2)
        self.assertFalse(mask[0])
        self.assertFalse(mask[1])
        self.assertTrue(mask[2] and mask[3])
        self.assertFalse(mask[4:].any())


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
    def _make_record(self, out_dir: str, seed: int = 0, geometry_seed: int = None) -> str:
        rng = np.random.default_rng(seed if geometry_seed is None else geometry_seed)
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
            self.assertTrue(res.model.has_local_head)
            acc = res.model.manifest.accuracy
            self.assertGreaterEqual(acc.get("confidence", 0.0), 0.0)
            self.assertLessEqual(acc.get("confidence", 0.0), 1.0)
            self.assertEqual(acc["train_records"] + acc["val_records"], 5)
            self.assertEqual(acc["train_records"], summary["train"])
            self.assertEqual(acc["val_records"], summary["test"])
            self.assertGreater(acc["val_records"], 0)
            self.assertGreaterEqual(acc["train_groups"], 1)
            self.assertEqual(acc["train_groups"], summary["train_groups"])
            self.assertEqual(acc["val_groups"], summary["val_groups"])
            self.assertIn("baseline_rmse_s", acc)
            self.assertIn("skill", acc)
            self.assertGreater(acc["val_cells"], 0)
            with np.load(os.path.join(out_dir, "weights.npz")) as npz:
                self.assertIn("local_weights", npz.files)

    def test_train_numpy_fits_on_train_split_only(self):
        from unittest import mock

        from core.surrogate.training import train_numpy

        with tempfile.TemporaryDirectory() as tmp:
            # 3 geometries x 2 alloy-ish variants = 6 records.
            for geometry in range(3):
                for variant in range(2):
                    self._make_record(tmp, seed=geometry * 2 + variant, geometry_seed=geometry)
            records = load_dataset(tmp)
            self.assertEqual(len(records), 6)
            train_idx, val_idx = grouped_train_val_indices(records, val_fraction=0.2, seed=0)
            self.assertGreater(len(val_idx), 0)

            fit_end = mock.Mock(wraps=NumpyWeightsModel._fit_weights)
            fit_local = mock.Mock(wraps=NumpyWeightsModel._fit_local_weights)
            with mock.patch.object(NumpyWeightsModel, "_fit_weights", staticmethod(fit_end)), \
                 mock.patch.object(NumpyWeightsModel, "_fit_local_weights", staticmethod(fit_local)):
                result = train_numpy(records, grid=8)

            for fit in (fit_end, fit_local):
                fit.assert_called_once()
                channels_arg = fit.call_args.args[0]
                self.assertEqual(len(channels_arg), len(train_idx))
                fitted_keys = {geometry_group_key({"channels": c}) for c in channels_arg}
                val_keys = {geometry_group_key(records[i]) for i in val_idx}
                self.assertFalse(fitted_keys & val_keys)

            self.assertEqual(result["train_records"], len(train_idx))
            self.assertEqual(result["val_records"], len(val_idx))
            self.assertEqual(result["train_groups"] + result["val_groups"], 3)
            self.assertGreaterEqual(result["val_groups"], 1)
            expected_cells = sum(
                int(schedule_evaluation_mask(records[i]["channels"], records[i]["end"]).sum())
                for i in val_idx
            )
            self.assertEqual(result["val_cells"], expected_cells)
            self.assertEqual(result["metrics"]["end_time"]["n_compared"], expected_cells)
            self.assertTrue(np.isfinite(result["headline"]["end_rmse_s"]))

    def test_single_geometry_group_reports_unmeasured_accuracy(self):
        import json

        from core.surrogate.training import retrain_from_dataset, train_numpy

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            out_dir = os.path.join(tmp, "model")
            for seed in range(4):
                self._make_record(data_dir, seed=seed, geometry_seed=7)
            records = load_dataset(data_dir)
            result = train_numpy(records, grid=8)
            self.assertEqual(result["val_records"], 0)
            self.assertEqual(result["train_records"], 4)
            self.assertEqual(result["train_groups"], 1)
            self.assertEqual(result["val_groups"], 0)
            self.assertEqual(result["metrics"], {})
            self.assertTrue(np.isnan(result["headline"]["rmse_s"]))
            self.assertTrue(np.isnan(result["headline"]["confidence"]))

            summary = retrain_from_dataset(data_dir=data_dir, output_dir=out_dir, grid=8)
            self.assertEqual(summary["test"], 0)
            with open(os.path.join(out_dir, "manifest.json")) as fh:
                manifest = json.load(fh)
            acc = manifest["accuracy"]
            self.assertIsNone(acc["rmse_s"])
            self.assertIsNone(acc["confidence"])
            self.assertIsNone(acc["baseline_rmse_s"])
            self.assertEqual(acc["val_records"], 0)
            self.assertIn("No validation split", manifest["notes"])
            res = load_model(out_dir)
            self.assertIsNotNone(res.model, res.error)
            self.assertIn("n/a", res.model.describe())

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
            mesh_type="voxel",  # compares two meshings point-wise; fTetWild is not bit-deterministic
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

class TrainBaselineTests(unittest.TestCase):
    """The manifest baseline is the TRAIN-mean constant scored on val cells."""

    def test_train_numpy_baseline_is_train_mean_on_val(self):
        from core.surrogate.metrics import schedule_evaluation_mask
        from core.surrogate.split import grouped_train_val_indices
        from core.surrogate.training import train_numpy

        with tempfile.TemporaryDirectory() as tmp:
            for geometry in range(3):
                for variant in range(2):
                    TrainingTests._make_record(
                        TrainingTests(), tmp, seed=geometry * 2 + variant, geometry_seed=geometry
                    )
            records = load_dataset(tmp)
            train_idx, val_idx = grouped_train_val_indices(records, val_fraction=0.2, seed=0)
            self.assertTrue(val_idx)

            def _masked_end(i):
                r = records[i]
                m = schedule_evaluation_mask(r["channels"], r["end"])
                return np.asarray(r["end"]).reshape(-1)[m]

            train_end = np.concatenate([_masked_end(i) for i in train_idx])
            val_end = np.concatenate([_masked_end(i) for i in val_idx])
            expected = float(np.sqrt(np.mean((val_end - train_end.mean()) ** 2)))
            oracle = float(val_end.std())

            result = train_numpy(records, grid=8)
            baseline = result["headline"]["baseline_rmse_s"]
            self.assertAlmostEqual(baseline, expected, places=4)
            # Train-mean baseline is never below the oracle (val std) constant.
            self.assertGreaterEqual(baseline + 1e-9, oracle)
            self.assertEqual(result["metrics"]["end_time"]["baseline_rmse"], baseline)
