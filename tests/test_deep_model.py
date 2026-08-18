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
from core.deep_model.model import UNet3D, levels_for_resolution
from core.deep_model.model import DeepModelManifest, read_deep_manifest

HAVE_TORCH = torch is not None
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
    """Write a schema-v2 .npz record in the capture_run format."""
    rng = np.random.default_rng(seed)
    R = resolution
    meta = GridMeta(
        resolution=R,
        bbox=np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
        cell_size=2.0 / R,
        max_dim=2.0,
    )
    channels = np.zeros((7, R, R, R), dtype=np.float32)
    channels[0, 1:-1, 1:-1, 1:-1] = 1.0
    channels[1, 1:-1, 1:-1, 1:-1] = 0.0
    channels[2] = 0.5
    channels[3] = 0.5
    height = np.linspace(0, 1, R).reshape(1, 1, R).repeat(R, axis=1).repeat(R, axis=0)
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
            self.assertGreaterEqual(summary["first_loss"], summary["last_loss"])
            self.assertTrue(os.path.exists(os.path.join(out_dir, "weights.pt")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "manifest.json")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "deep_manifest.json")))

            deep = read_deep_manifest(out_dir)
            self.assertIsNotNone(deep)
            self.assertEqual(deep.kind, "torch_unet")

    def test_training_raises_without_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            from core.deep_model.training import train_deep_model

            with self.assertRaises(ValueError):
                train_deep_model(data_dir=tmp, output_dir=tmp, grid=8, min_records=1)


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
            channels[4] = np.linspace(0, 1, 8)[None, None, :] * np.ones((8, 8, 1))
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
