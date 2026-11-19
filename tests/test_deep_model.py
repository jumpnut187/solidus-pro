"""Tests for the deep learning solidification surrogate (core/deep_model).

Skipped gracefully when torch is absent — the deep module is an optional
dependency and the app must start without it.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import time
import unittest

import numpy as np
import trimesh

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from core.surrogate import load_model, run_surrogate_pipeline
from core.surrogate.conditioning import build_conditioning_vector
from core.surrogate.fields import GridMeta
from core.simulation_pipeline import PipelineInputs
from core.materials import get_alloy

from core.deep_model.autotrain import (
    AutoTrainState,
    dataset_signature,
    load_state,
    save_state,
    should_retrain,
)
from core.deep_model.model import (
    TARGET_NAMES,
    DeepModelManifest,
    FiLM,
    UNet3D,
    levels_for_resolution,
    read_deep_manifest,
)
from core.surrogate.split import geometry_group_key

HAVE_TORCH = torch is not None
try:
    import onnxruntime  # noqa: F401

    HAVE_ONNXRUNTIME = True
except ImportError:  # pragma: no cover
    HAVE_ONNXRUNTIME = False
HAVE_COREML = False
HAVE_TFLITE = False
if HAVE_TORCH:
    try:
        import coremltools  # noqa: F401

        HAVE_COREML = True
    except ImportError:
        pass
    try:
        import tflite_runtime  # noqa: F401
        import onnx2tf  # noqa: F401

        HAVE_TFLITE = True
    except ImportError:
        pass

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _conditioning():
    return build_conditioning_vector(get_alloy("WCB"), {"max_dim": 10.0, "voxel_size": 0.1})


def _write_synthetic_record(out_dir: str, seed: int, resolution: int = 8) -> str:
    """Write a schema-v2 .npz record in the capture_run format.

    Geometry varies with ``seed % 3`` (the box is shortened along x) so
    datasets built from several seeds contain several geometry groups, which
    the grouped train/val split requires.  Grid axis order is ``(z, y, x)``
    like the real capture; ``normalized_height`` varies along axis 0.
    """
    R = resolution
    x_hi = R - 1 - (seed % 3)
    meta = GridMeta(
        resolution=R,
        bbox=np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
        cell_size=2.0 / R,
        max_dim=2.0,
    )
    channels = np.zeros((7, R, R, R), dtype=np.float32)
    channels[0, 1:-1, 1:-1, 1:x_hi] = 1.0
    channels[1, 1:-1, 1:-1, 1:x_hi] = 0.0
    channels[2] = 0.5
    channels[3] = 0.5
    height = np.linspace(0, 1, R).reshape(R, 1, 1).repeat(R, axis=1).repeat(R, axis=2)
    channels[4] = height.astype(np.float32)
    end = np.full((R, R, R), 300.0 + seed * 50.0, dtype=np.float32)
    end[channels[0] == 0] = 0.0
    start = np.maximum(end - 120.0, 0.0)
    local = np.maximum(end - start, 0.0)
    cond = _conditioning()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"run_{seed:012d}.npz")
    np.savez_compressed(
        path,
        schema_version=2,
        channels=channels,
        solidification_start_times=start.reshape(-1),
        solidification_end_times=end.reshape(-1),
        local_solidification_time=local.reshape(-1),
        conditioning=cond.vector,
        conditioning_names=np.asarray(cond.names),
        alloy_name="WCB",
        grid_resolution=R,
        grid_meta=json.dumps(meta.to_dict()),
    )
    return path


def _two_record_dataset(tmp: str) -> str:
    data_dir = os.path.join(tmp, "data")
    _write_synthetic_record(data_dir, 0)
    _write_synthetic_record(data_dir, 1)
    return data_dir


def _loader_for(mesh):
    bounds = np.stack([mesh.bounds[0], mesh.bounds[1]], axis=1)
    return type(
        "_Loader",
        (),
        {
            "trimesh_obj": mesh,
            "is_valid": lambda self: True,
            "volume": float(getattr(mesh, "volume", 0.0) or 0.0),
            "get_bounds_array": lambda self: bounds,
        },
    )()


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class ArchitectureTests(unittest.TestCase):
    def test_forward_pass_shape_and_finite(self):
        for resolution in (8, 16, 32):
            model = UNet3D(levels=levels_for_resolution(resolution))
            channels = torch.rand(2, 7, resolution, resolution, resolution)
            cond = torch.rand(2, 23)
            out = model(channels, cond)
            self.assertEqual(out.shape, (2, 3, resolution, resolution, resolution))
            self.assertTrue(torch.isfinite(out).all())
            heads = model.forward_normalized(channels, cond)
            self.assertEqual(heads.shape, (2, len(TARGET_NAMES), resolution, resolution, resolution))

    def test_forward_derives_end_from_start_and_local(self):
        model = UNet3D(levels=levels_for_resolution(8))
        model.set_target_normalization([3.0, 4.0], [1.5, 0.5])
        channels = torch.rand(2, 7, 8, 8, 8)
        cond = torch.rand(2, 23)
        out = model(channels, cond)
        torch.testing.assert_close(out[:, 1], out[:, 0] + out[:, 2])
        self.assertTrue((out >= 0).all())

    def test_normalization_round_trip(self):
        model = UNet3D(levels=levels_for_resolution(8))
        mean, std = [2.5, 4.0], [0.7, 1.3]
        model.set_target_normalization(mean, std)
        channels = torch.rand(1, 7, 8, 8, 8)
        cond = torch.rand(1, 23)
        with torch.no_grad():
            z = model.forward_normalized(channels, cond)
            out = model(channels, cond)
        expected_start = torch.expm1(torch.clamp(z[:, 0] * std[0] + mean[0], min=0.0, max=11.0))
        expected_local = torch.expm1(torch.clamp(z[:, 1] * std[1] + mean[1], min=0.0, max=11.0))
        torch.testing.assert_close(out[:, 0], expected_start)
        torch.testing.assert_close(out[:, 2], expected_local)
        with self.assertRaises(ValueError):
            model.set_target_normalization([1.0, 1.0], [0.0, 1.0])

    def test_film_is_identity_at_init(self):
        film = FiLM(channels=16, conditioning_size=23)
        x = torch.randn(2, 16, 4, 4, 4)
        cond = torch.randn(2, 23)
        torch.testing.assert_close(film(x, cond), x)

    def test_default_model_size(self):
        model = UNet3D(levels=4)
        params = sum(p.numel() for p in model.parameters())
        self.assertGreater(params, 4_000_000)
        self.assertLess(params, 8_000_000)

    def test_resolution_levels(self):
        self.assertEqual(levels_for_resolution(8), 3)
        self.assertEqual(levels_for_resolution(16), 4)
        self.assertEqual(levels_for_resolution(32), 4)
        self.assertEqual(levels_for_resolution(64), 4)
        self.assertEqual(levels_for_resolution(128), 4)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TrainingTests(unittest.TestCase):
    def test_tiny_overfit_loss_decreases(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = _two_record_dataset(tmp)
            out_dir = os.path.join(tmp, "out")
            from core.deep_model.training import train_deep_model

            summary = train_deep_model(
                data_dir=data_dir,
                output_dir=out_dir,
                grid=8,
                epochs=6,
                batch_size=2,
                learning_rate=1e-3,
                name="Overfit Test",
                min_records=1,
            )
            self.assertEqual(summary["records"], 2)
            self.assertEqual(summary["train"], 1)
            self.assertEqual(summary["test"], 1)
            self.assertGreaterEqual(summary["first_loss"], summary["last_loss"])
            self.assertIsNotNone(summary["val_loss"])
            self.assertTrue(os.path.exists(os.path.join(out_dir, "weights.pt")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "manifest.json")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "deep_manifest.json")))

            deep = read_deep_manifest(out_dir)
            self.assertIsNotNone(deep)
            self.assertEqual(deep.kind, "torch_unet")
            self.assertEqual(deep.target_names, list(TARGET_NAMES))
            self.assertEqual(len(deep.target_mean), 2)
            self.assertTrue(all(sd > 0 for sd in deep.target_std))
            self.assertEqual(deep.training["train_records"], 1)
            self.assertEqual(deep.training["val_records"], 1)
            self.assertIn("val_loss", deep.notes)
            self.assertNotIn("best_val_mse", deep.notes)
            self.assertIn("baseline_end_rmse_s", deep.accuracy)

    def test_training_raises_without_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            from core.deep_model.training import train_deep_model

            with self.assertRaises(ValueError):
                train_deep_model(data_dir=tmp, output_dir=tmp, grid=8, min_records=1)

    def test_training_refuses_single_geometry_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            for seed in (0, 3, 6):  # same geometry (seed % 3 == 0), different end times
                _write_synthetic_record(data_dir, seed)
            from core.deep_model.training import train_deep_model

            with self.assertRaises(ValueError) as ctx:
                train_deep_model(data_dir=data_dir, output_dir=os.path.join(tmp, "out"),
                                 grid=8, epochs=1, min_records=1)
            self.assertIn("geometry group", str(ctx.exception))

    def test_early_stopping_keeps_best_checkpoint(self):
        from torch.utils.data import DataLoader

        from core.deep_model.dataset import DeepDataset, make_train_val_datasets
        from core.deep_model.inference import build_unet_from_artifact
        from core.deep_model.training import _validation_loss, train_deep_model
        from core.surrogate.model import _read_manifest

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            for seed in range(4):
                _write_synthetic_record(data_dir, seed)
            out_dir = os.path.join(tmp, "out")
            summary = train_deep_model(
                data_dir=data_dir, output_dir=out_dir, grid=8, epochs=40,
                batch_size=2, learning_rate=3e-2, min_records=1, patience=2,
                augment=False,
            )
            self.assertLessEqual(summary["best_epoch"], summary["epochs_run"])
            deep = read_deep_manifest(out_dir)
            self.assertEqual(deep.epochs, summary["epochs_run"])
            self.assertEqual(deep.training["best_epoch"], summary["best_epoch"])
            self.assertEqual(deep.training["early_stopped"], summary["early_stopped"])
            if summary["epochs_run"] == 40:
                self.assertFalse(summary["early_stopped"])
            else:
                self.assertTrue(summary["early_stopped"])
                self.assertEqual(summary["epochs_run"], summary["best_epoch"] + 2)

            # The saved weights are the best-val epoch, not the last one.
            model = build_unet_from_artifact(out_dir, _read_manifest(out_dir))
            _, val = make_train_val_datasets(data_dir, resolution=8)
            loader = DataLoader(val, batch_size=2, collate_fn=DeepDataset.collate)
            val_loss = _validation_loss(model, loader, torch.device("cpu"))
            self.assertAlmostEqual(
                val_loss, summary["val_loss"], delta=max(1e-3, 0.02 * summary["val_loss"])
            )

    def test_evaluation_matches_inference_path(self):
        from core.deep_model.dataset import make_train_val_datasets
        from core.deep_model.inference import load_deep_model
        from core.deep_model.training import _evaluate, _masked_end_mean, train_deep_model

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            for seed in range(4):
                _write_synthetic_record(data_dir, seed)
            out_dir = os.path.join(tmp, "out")
            summary = train_deep_model(
                data_dir=data_dir, output_dir=out_dir, grid=8, epochs=3,
                batch_size=2, min_records=1,
            )
            result = load_deep_model(out_dir, backend="cpu")
            self.assertEqual(result.error, "")
            train, val = make_train_val_datasets(data_dir, resolution=8)
            metrics = _evaluate(
                result.model.model, val, torch.device("cpu"), _masked_end_mean(train)
            )

            end_pred, end_ref = [], []
            for item, record in zip(val, val.records):
                pred = result.model.predict(item.channels.numpy(), record["conditioning"])
                mask = item.mask.numpy()
                end_pred.append(pred[1][mask])
                end_ref.append(item.end.numpy()[mask])
            end_pred, end_ref = np.concatenate(end_pred), np.concatenate(end_ref)
            rmse = float(np.sqrt(np.mean((end_pred - end_ref) ** 2)))
            self.assertAlmostEqual(metrics["end_time"]["rmse"], rmse, places=3)
            self.assertEqual(metrics["baseline"]["n_cells"], end_ref.size)
            self.assertIn("baseline_end_rmse_s", summary["headline"])
            self.assertGreaterEqual(summary["headline"]["confidence"], 0.0)
            self.assertLessEqual(summary["headline"]["confidence"], 1.0)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class GroupedSplitTests(unittest.TestCase):
    def test_make_train_val_datasets_groups_geometry(self):
        from core.deep_model.dataset import make_train_val_datasets

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            for seed in range(9):  # 3 geometries x 3 end-time variants
                _write_synthetic_record(data_dir, seed)
            train, val = make_train_val_datasets(data_dir, resolution=8)
            self.assertIsNotNone(val)
            self.assertEqual(len(train) + len(val), 9)
            train_keys = {geometry_group_key(r) for r in train.records}
            val_keys = {geometry_group_key(r) for r in val.records}
            self.assertFalse(train_keys & val_keys)
            self.assertEqual(train.geometry_groups + val.geometry_groups, 3)

    def test_single_geometry_has_no_validation(self):
        from core.deep_model.dataset import make_train_val_datasets

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            for seed in (0, 3, 6, 9, 12):
                _write_synthetic_record(data_dir, seed)
            train, val = make_train_val_datasets(data_dir, resolution=8)
            self.assertIsNone(val)
            self.assertEqual(len(train), 5)

    def test_augment_batch_preserves_z_and_alignment(self):
        from core.deep_model.dataset import augment_batch

        R = 8
        channels = torch.rand(2, 7, R, R, R)
        start = torch.rand(2, R, R, R)
        mask = torch.rand(2, R, R, R) > 0.5
        batch = {"channels": channels, "start": start, "mask": mask,
                 "conditioning": torch.rand(2, 23)}
        for seed in range(6):
            out = augment_batch(batch, torch.Generator().manual_seed(seed))
            # Same permutation applied to every grid: masked values line up.
            torch.testing.assert_close(out["channels"][:, 0][out["mask"]].sort().values,
                                       channels[:, 0][mask].sort().values)
            torch.testing.assert_close(out["start"][out["mask"]].sort().values,
                                       start[mask].sort().values)
            # z profile (axis 0 of the grid) is untouched.
            torch.testing.assert_close(out["channels"].mean(dim=(1, 3, 4)),
                                       channels.mean(dim=(1, 3, 4)))
            torch.testing.assert_close(out["conditioning"], batch["conditioning"])


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class DatasetResampleTests(unittest.TestCase):
    """Resampling must preserve schedule field values (regression: the
    occupancy re-binarization overwrote start/end/local with the occupancy
    mask, collapsing every training target to constant 1.0 on occupied
    cells and producing degenerate fraction-liquid playback)."""

    def _record(self) -> dict:
        from core.deep_model.dataset import resample_record

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_synthetic_record(tmp, 0, resolution=8)
            with np.load(path, allow_pickle=True) as npz:
                channels = np.asarray(npz["channels"], dtype=np.float32)
                start = np.asarray(npz["solidification_start_times"], dtype=np.float32).reshape(8, 8, 8)
                end = np.asarray(npz["solidification_end_times"], dtype=np.float32).reshape(8, 8, 8)
                local = np.asarray(npz["local_solidification_time"], dtype=np.float32).reshape(8, 8, 8)
            return resample_record(channels, start, end, local, 4)

    def test_resample_preserves_schedule_values(self):
        channels, start, end, local = self._record()
        for field in (start, end, local):
            self.assertEqual(field.shape, (4, 4, 4))
        self.assertAlmostEqual(float(end.max()), 300.0, places=4)
        self.assertAlmostEqual(float(np.median(end[end > 0])), 300.0, places=4)
        self.assertAlmostEqual(float(start[start > 0].max()), 180.0, places=4)
        self.assertAlmostEqual(float(np.median(start[start > 0])), 180.0, places=4)
        self.assertAlmostEqual(float(np.median(local[local > 0])), 120.0, places=4)
        self.assertEqual(set(np.unique(channels[0])).issubset({0.0, 1.0}), True)

    def test_resample_not_collapsed_to_occupancy(self):
        # Regression guard: fields must not be replaced by the 0/1 occupancy mask.
        channels, start, end, local = self._record()
        for field in (start, end, local):
            active = field[field > 0]
            self.assertGreater(active.size, 0)
            self.assertGreater(float(active.max()), 1.0)

    def test_deep_dataset_items_carry_real_targets(self):
        from core.deep_model.dataset import DeepDataset
        from core.surrogate.dataset import load_dataset

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            _write_synthetic_record(data_dir, 0, resolution=64)
            _write_synthetic_record(data_dir, 1, resolution=64)
            ds = DeepDataset(load_dataset(data_dir), resolution=32, split="all")
            for item in ds:
                for field in (item.start, item.end, item.local):
                    values = field.numpy()
                    active = values[values > 0]
                    self.assertGreater(active.size, 0)
                    self.assertGreater(float(active.max()), 1.0)
                self.assertTrue((item.end.numpy() > 0).any())



@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class InferenceTests(unittest.TestCase):
    def _train_small_model(self, tmp: str) -> str:
        data_dir = _two_record_dataset(tmp)
        out_dir = os.path.join(tmp, "out")
        from core.deep_model.training import train_deep_model

        summary = train_deep_model(
            data_dir=data_dir,
            output_dir=out_dir,
            grid=8,
            epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            min_records=1,
        )
        return summary["saved_to"]

    def test_load_via_shared_loader_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved_to = self._train_small_model(tmp)
            result = load_model(saved_to)
            self.assertEqual(result.error, "")
            self.assertIsNotNone(result.model)
            self.assertEqual(result.model.manifest.kind, "torch_unet")
            self.assertEqual(result.model.manifest.backend, "torch")

            pred = result.model.predict(
                np.zeros((7, 8, 8, 8), dtype=np.float32), _conditioning()
            )
            self.assertEqual(pred.shape, (3, 8, 8, 8))
            self.assertTrue(np.all(np.isfinite(pred)))
            self.assertTrue(np.all(pred >= 0.0))

    def test_predict_reconstructs_coherent_schedule(self):
        # Regression: end must equal start + local on active cells (bounded by
        # max_simulation_time). An incoherent end head inflates the
        # fraction-liquid timeline far past the true completion time.
        with tempfile.TemporaryDirectory() as tmp:
            saved_to = self._train_small_model(tmp)
            result = load_model(saved_to)
            self.assertEqual(result.error, "")

            rng = np.random.default_rng(0)
            channels = np.zeros((7, 8, 8, 8), dtype=np.float32)
            channels[0, 2:6, 2:6, 2:6] = 1.0
            channels[1, 2:6, 2:6, 2:6] = 0.0
            channels[2] = 0.5
            channels[3] = 0.5
            channels[4] = np.linspace(0, 1, 8)[:, None, None] * np.ones((1, 8, 8))
            cond = _conditioning()
            from core.surrogate.conditioning import raw_from_conditioning

            max_time = float(raw_from_conditioning(cond).get("max_simulation_time", 14400.0))

            pred = result.model.predict(channels, cond)
            start, end, local = pred[0], pred[1], pred[2]
            active = channels[0] > 0.5
            self.assertTrue(np.any(active))
            np.testing.assert_allclose(end[active], start[active] + local[active], rtol=1e-5, atol=1e-4)
            self.assertTrue(np.all(end[active] <= max_time + 1e-3))
            self.assertTrue(np.all((start >= 0) & (local >= 0) & (end >= 0)))
            self.assertTrue(np.all(end[~active] == 0.0))

    def test_pipeline_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            _write_synthetic_record(data_dir, 0, resolution=16)
            _write_synthetic_record(data_dir, 1, resolution=16)
            from core.deep_model.training import train_deep_model

            summary = train_deep_model(
                data_dir=data_dir,
                output_dir=os.path.join(tmp, "out"),
                grid=16,
                epochs=2,
                batch_size=1,
                learning_rate=1e-3,
                min_records=1,
            )
            result = load_model(summary["saved_to"])
            self.assertEqual(result.error, "")

            box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
            inputs = PipelineInputs(
                alloy_name="WCB",
                voxel_size=0.5,
                refinement_level=1,
                memory_mode="ram",
                analysis_options={
                    "shrinkage": True,
                    "hotspot": True,
                    "niyama": True,
                    "fraction_liquid": True,
                },
                max_voxels_for_visualization=50_000,
                show_progress=False,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                pipe_result = run_surrogate_pipeline(
                    inputs, model=result.model, stl_loader=_loader_for(box)
                )
            engine = pipe_result.engine
            self.assertTrue(getattr(engine, "surrogate_run", False))
            self.assertTrue(np.all(np.isfinite(engine.solidification_times)))
            self.assertIn("solidification_times", pipe_result.visualization_data)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class ExportSmokeTests(unittest.TestCase):
    def _train_small_model(self, tmp: str) -> str:
        data_dir = _two_record_dataset(tmp)
        from core.deep_model.training import train_deep_model

        summary = train_deep_model(
            data_dir=data_dir,
            output_dir=os.path.join(tmp, "out"),
            grid=8,
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            min_records=1,
        )
        return summary["saved_to"]

    def test_onnx_export_and_validation(self):
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as tmp:
            saved_to = self._train_small_model(tmp)
            from core.deep_model.export import export_for_target

            result = export_for_target(saved_to, "onnx", progress_cb=lambda _m: None)
            self.assertEqual(result["target"], "onnx")
            self.assertTrue(os.path.exists(result["artifact_path"]))
            self.assertTrue(result["validated"])
            self.assertIn("onnx", read_deep_manifest(saved_to).targets)

            sess = ort.InferenceSession(result["artifact_path"], providers=["CPUExecutionProvider"])
            out = sess.run(
                None,
                {
                    "channels": np.zeros((1, 7, 8, 8, 8), dtype=np.float32),
                    "conditioning": np.ones((1, 23), dtype=np.float32),
                },
            )
            self.assertEqual(np.asarray(out[0]).shape, (1, 3, 8, 8, 8))

    @unittest.skipUnless(HAVE_TFLITE, "onnx2tf / tflite-runtime not installed")
    def test_tflite_export_and_interpreter(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved_to = self._train_small_model(tmp)
            from core.deep_model.export import export_for_target

            result = export_for_target(saved_to, "tflite", progress_cb=lambda _m: None)
            self.assertEqual(result["target"], "tflite")
            self.assertTrue(os.path.exists(result["artifact_path"]))
            self.assertTrue(result["validated"])

    @unittest.skipUnless(HAVE_COREML, "coremltools not installed (macOS required)")
    def test_coreml_export(self):
        import platform

        if platform.system() != "Darwin":
            self.skipTest("Core ML export requires macOS")
        with tempfile.TemporaryDirectory() as tmp:
            saved_to = self._train_small_model(tmp)
            from core.deep_model.export import export_for_target

            result = export_for_target(saved_to, "coreml", progress_cb=lambda _m: None)
            self.assertEqual(result["target"], "coreml")
            self.assertTrue(os.path.isdir(result["artifact_path"]))
            self.assertTrue(result["validated"])
            self.assertIn("coreml", read_deep_manifest(saved_to).targets)

            # Regression: the GroupNorm export shim must reproduce the torch
            # forward (it once normalized per channel, i.e. InstanceNorm).
            from core.deep_model.backend import DeepBackend
            from core.deep_model.inference import load_deep_model

            coreml = load_deep_model(saved_to, backend="coreml").model
            torch_cpu = load_deep_model(saved_to, backend="cpu").model
            self.assertEqual(coreml._backend, DeepBackend.COREML)
            channels = np.zeros((7, 8, 8, 8), dtype=np.float32)
            channels[0, 2:6, 2:6, 2:6] = 1.0
            channels[2] = 0.5
            channels[3] = 0.5
            cond = _conditioning()
            pred_coreml = coreml.predict(channels, cond)
            pred_torch = torch_cpu.predict(channels, cond)
            active = channels[0] > 0.5
            np.testing.assert_allclose(
                pred_coreml[1][active], pred_torch[1][active], rtol=2e-2, atol=2.0
            )
            self.assertIs(coreml._coreml_model, coreml._coreml())

    def test_tflite_missing_toolchain_message(self):
        import importlib.util

        if importlib.util.find_spec("onnx2tf") is not None:
            self.skipTest("onnx2tf installed — nothing to assert")
        with tempfile.TemporaryDirectory() as tmp:
            saved_to = self._train_small_model(tmp)
            from core.deep_model.export import export_for_target

            with self.assertRaises(ImportError) as ctx:
                export_for_target(saved_to, "tflite", progress_cb=lambda _m: None)
            self.assertIn("onnx2tf", str(ctx.exception))


def _active_channels(resolution: int = 8) -> np.ndarray:
    channels = np.zeros((7, resolution, resolution, resolution), dtype=np.float32)
    channels[0, 2:6, 2:6, 2:6] = 1.0
    channels[2] = 0.5
    channels[3] = 0.5
    return channels


@unittest.skipUnless(HAVE_TORCH and HAVE_ONNXRUNTIME, "torch / onnxruntime not installed")
class OnnxInferenceTests(unittest.TestCase):
    """ONNX Runtime inference: validation against PyTorch, auto selection, fallback."""

    def _export_onnx(self, tmp: str) -> str:
        data_dir = _two_record_dataset(tmp)
        from core.deep_model.export import export_for_target
        from core.deep_model.training import train_deep_model

        summary = train_deep_model(
            data_dir=data_dir, output_dir=os.path.join(tmp, "out"), grid=8,
            epochs=1, batch_size=2, learning_rate=1e-3, min_records=1,
        )
        saved_to = summary["saved_to"]
        export_for_target(saved_to, "onnx", progress_cb=lambda _m: None)
        return saved_to

    def test_explicit_onnx_matches_torch(self):
        from core.deep_model.backend import DeepBackend
        from core.deep_model.inference import load_deep_model

        with tempfile.TemporaryDirectory() as tmp:
            saved_to = self._export_onnx(tmp)
            onnx_model = load_deep_model(saved_to, backend="onnx").model
            torch_model = load_deep_model(saved_to, backend="cpu").model
            self.assertIsNotNone(onnx_model.onnx_path)
            channels, cond = _active_channels(), _conditioning()

            pred_onnx = onnx_model.predict(channels, cond)
            self.assertEqual(onnx_model._backend, DeepBackend.ONNX)
            self.assertIn("onnx_provider", onnx_model.selection)
            self.assertNotIn("onnx_ms", onnx_model.selection)  # explicit: no benchmark
            pred_torch = torch_model.predict(channels, cond)
            active = channels[0] > 0.5
            np.testing.assert_allclose(pred_onnx[:, active], pred_torch[:, active], rtol=1e-3, atol=1e-2)

            # Later predictions reuse the session instead of re-checking.
            runner = onnx_model._onnx
            onnx_model.predict(channels, cond)
            self.assertIs(onnx_model._onnx, runner)
            self.assertIn("ONNX Runtime on", onnx_model.describe())

    def test_auto_times_both_and_keeps_the_faster(self):
        from core.deep_model.backend import DeepBackend
        from core.deep_model.inference import load_deep_model

        with tempfile.TemporaryDirectory() as tmp:
            saved_to = self._export_onnx(tmp)
            model = load_deep_model(saved_to, backend="auto").model
            self.assertIn("chosen on the first prediction", model.device_summary)
            model.predict(_active_channels(), _conditioning())

            sel = model.selection
            self.assertIn("onnx_ms", sel)
            self.assertIn("torch_ms", sel)
            expected = "onnx" if sel["onnx_ms"] < sel["torch_ms"] else model._torch_backend.value
            self.assertEqual(sel["chosen"], expected)
            self.assertEqual(model._backend.value, expected)
            self.assertEqual(model._onnx is not None, model._backend == DeepBackend.ONNX)

    def test_mismatch_falls_back_to_torch(self):
        from unittest import mock

        from core.deep_model import inference
        from core.deep_model.inference import load_deep_model

        with tempfile.TemporaryDirectory() as tmp:
            saved_to = self._export_onnx(tmp)
            model = load_deep_model(saved_to, backend="onnx").model
            torch_model = load_deep_model(saved_to, backend="cpu").model
            channels, cond = _active_channels(), _conditioning()
            # A negative tolerance can never be met, so the check must reject ONNX.
            with mock.patch.object(inference, "ONNX_MATCH_ATOL_S", -1.0), \
                    mock.patch.object(inference, "ONNX_MATCH_RTOL", 0.0):
                pred = model.predict(channels, cond)
            self.assertEqual(model._backend, model._torch_backend)
            self.assertIsNone(model._onnx)
            self.assertIn("differs from PyTorch", model.selection["onnx_rejected"])
            # The fallback may run on MPS while torch_model runs on CPU.
            np.testing.assert_allclose(pred, torch_model.predict(channels, cond), rtol=1e-4, atol=1e-2)

    def test_no_onnx_artifact_stays_on_torch(self):
        from core.deep_model.inference import load_deep_model

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = _two_record_dataset(tmp)
            from core.deep_model.training import train_deep_model

            summary = train_deep_model(
                data_dir=data_dir, output_dir=os.path.join(tmp, "out"), grid=8,
                epochs=1, batch_size=2, learning_rate=1e-3, min_records=1,
            )
            model = load_deep_model(summary["saved_to"], backend="onnx").model
            self.assertIsNone(model.onnx_path)
            self.assertEqual(model._backend, model._torch_backend)
            self.assertIn("no exported", model.selection["onnx_rejected"])

    def test_session_skips_provider_that_is_not_installed(self):
        from core.deep_model.onnx_runner import create_session, find_onnx_artifact

        with tempfile.TemporaryDirectory() as tmp:
            saved_to = self._export_onnx(tmp)
            path = find_onnx_artifact(saved_to)
            self.assertIsNotNone(path)
            _session, provider, skipped = create_session(
                path, ["NoSuchExecutionProvider", "CPUExecutionProvider"]
            )
            self.assertEqual(provider, "CPUExecutionProvider")
            self.assertEqual(len(skipped), 1)
            self.assertTrue(skipped[0].startswith("NoSuchExecutionProvider"))


@unittest.skipUnless(HAVE_ONNXRUNTIME, "onnxruntime not installed")
class OnnxProviderOrderTests(unittest.TestCase):
    """Provider ordering and overrides; no model or torch needed."""

    def _order(self, installed, env=""):
        from unittest import mock

        import onnxruntime as ort

        from core.deep_model import backend

        with mock.patch.object(ort, "get_available_providers", return_value=list(installed)), \
                mock.patch.dict(os.environ, {backend.ONNX_PROVIDERS_ENV: env}):
            return backend.available_onnx_providers()

    def test_npus_before_gpus_and_cpu_last(self):
        order = self._order([
            "CPUExecutionProvider", "DmlExecutionProvider", "CUDAExecutionProvider",
            "QNNExecutionProvider", "CoreMLExecutionProvider",
        ])
        self.assertEqual(
            order,
            ["QNNExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"],
        )

    def test_env_override_limits_and_orders(self):
        order = self._order(
            ["CPUExecutionProvider", "DmlExecutionProvider", "OpenVINOExecutionProvider"],
            env="DmlExecutionProvider, NotInstalledExecutionProvider",
        )
        self.assertEqual(order, ["DmlExecutionProvider", "CPUExecutionProvider"])

    def test_npu_provider_options(self):
        from core.deep_model.backend import onnx_provider_options

        self.assertEqual(onnx_provider_options("QNNExecutionProvider")["backend_type"], "htp")
        self.assertEqual(onnx_provider_options("OpenVINOExecutionProvider"), {"device_type": "NPU"})
        self.assertEqual(onnx_provider_options("CPUExecutionProvider"), {})

    def test_settings_normalize_backend(self):
        from config.settings import DeepModelSettings

        self.assertEqual(DeepModelSettings(inference_backend="ONNX").inference_backend, "onnx")
        self.assertEqual(DeepModelSettings(inference_backend="tpu").inference_backend, "auto")


class AutoTrainPolicyTests(unittest.TestCase):
    """Pure policy logic — no Qt, no torch required."""

    def _state(self, count=2, mtimes=(1.0, 2.0), trained_at=0.0, model_exists=True):
        return AutoTrainState(
            signature_count=count,
            signature_mtimes=tuple(mtimes),
            last_completed_at=trained_at,
            model_exists=model_exists,
        )

    def test_disabled(self):
        ok, _ = should_retrain(
            enabled=False, torch_available=True, training_active=False,
            export_in_progress=False, current_signature=(2, (1.0, 2.0)),
            state=self._state(),
        )
        self.assertFalse(ok)

    def test_torch_missing(self):
        ok, _ = should_retrain(
            enabled=True, torch_available=False, training_active=False,
            export_in_progress=False, current_signature=(2, (1.0, 2.0)),
            state=self._state(),
        )
        self.assertFalse(ok)

    def test_below_min_records(self):
        ok, reason = should_retrain(
            enabled=True, torch_available=True, training_active=False,
            export_in_progress=False, current_signature=(2, (1.0, 2.0)),
            state=self._state(count=2), min_records=4,
        )
        self.assertFalse(ok)
        self.assertIn("need at least", reason)

    def test_initial_training_without_model(self):
        ok, reason = should_retrain(
            enabled=True, torch_available=True, training_active=False,
            export_in_progress=False, current_signature=(5, (1.0, 2.0, 3.0, 4.0, 5.0)),
            state=self._state(model_exists=False),
        )
        self.assertTrue(ok)
        self.assertIn("initial training", reason)

    def test_unchanged_dataset_skips(self):
        ok, _ = should_retrain(
            enabled=True, torch_available=True, training_active=False,
            export_in_progress=False, current_signature=(2, (1.0, 2.0)),
            state=self._state(count=2, mtimes=(1.0, 2.0), trained_at=time.time() - 1000),
        )
        self.assertFalse(ok)

    def test_cooldown_respected(self):
        ok, reason = should_retrain(
            enabled=True, torch_available=True, training_active=False,
            export_in_progress=False, current_signature=(3, (1.0, 2.0, 3.0)),
            state=self._state(count=2, mtimes=(1.0, 2.0), trained_at=time.time() - 10),
            min_records=2, cooldown_seconds=300,
        )
        self.assertFalse(ok)
        self.assertIn("Cooldown", reason)

    def test_retrain_after_cooldown(self):
        ok, reason = should_retrain(
            enabled=True, torch_available=True, training_active=False,
            export_in_progress=False, current_signature=(3, (1.0, 2.0, 3.0)),
            state=self._state(count=2, mtimes=(1.0, 2.0), trained_at=time.time() - 1000),
            min_records=2, cooldown_seconds=300,
        )
        self.assertTrue(ok)
        self.assertIn("accumulated dataset", reason)

    def test_training_active_skips(self):
        ok, _ = should_retrain(
            enabled=True, torch_available=True, training_active=True,
            export_in_progress=False, current_signature=(3, (1.0, 2.0, 3.0)),
            state=self._state(count=2, mtimes=(1.0, 2.0), trained_at=time.time() - 1000),
        )
        self.assertFalse(ok)

    def test_export_in_progress_skips(self):
        ok, _ = should_retrain(
            enabled=True, torch_available=True, training_active=False,
            export_in_progress=True, current_signature=(3, (1.0, 2.0, 3.0)),
            state=self._state(count=2, mtimes=(1.0, 2.0), trained_at=time.time() - 1000),
        )
        self.assertFalse(ok)

    def test_state_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(count=7, mtimes=(1.0, 2.0, 3.0), trained_at=1234.0)
            save_state(tmp, state)
            loaded = load_state(tmp)
            self.assertEqual(loaded.signature_count, 7)
            self.assertEqual(loaded.signature_mtimes, (1.0, 2.0, 3.0))
            self.assertEqual(loaded.last_completed_at, 1234.0)
            self.assertTrue(loaded.model_exists)

    def test_dataset_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(dataset_signature(tmp))
            _write_synthetic_record(tmp, 0)
            sig = dataset_signature(tmp)
            self.assertEqual(sig[0], 1)
            _write_synthetic_record(tmp, 1)
            sig2 = dataset_signature(tmp)
            self.assertEqual(sig2[0], 2)
            self.assertNotEqual(sig, sig2)


if __name__ == "__main__":
    unittest.main()
