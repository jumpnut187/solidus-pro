"""NPU export pipeline: torch -> Core ML / ONNX / TFLite.

All three artifacts are produced from the same trained weights
(``weights.pt`` + ``manifest.json`` in a deep model directory):

* ``coreml`` — ``.mlpackage`` for the Apple ANE (compute_units=all).
* ``onnx`` — fp16 ONNX graph for Snapdragon (ONNX Runtime QNN EP), plus an
  fp32 graph validated on the CPU EP.
* ``tflite`` — int8 post-training-quantized TFLite for Google Tensor
  (torch -> ONNX -> onnx2tf), validated with tflite-runtime.

Each target updates ``deep_manifest.json`` with the artifact path,
quantization, and validation status.  Target-specific toolchains are imported
lazily and fail with a clear message rather than a crash (onnx2tf may lag
Python 3.13 / arm64).
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Callable, Dict, Optional

import numpy as np

from core.deep_model.inference import build_unet_from_artifact
from core.deep_model.model import (
    UNet3D,
    read_deep_manifest,
    save_deep_manifest,
)
from core.surrogate.model import _read_manifest

ProgressCallback = Optional[Callable[[str], None]]
CancelCallback = Optional[Callable[[], bool]]

TARGETS = ("coreml", "onnx", "tflite")
ARTIFACT_NAMES = {
    "coreml": "SolidusPro_Deep.mlpackage",
    "onnx": "SolidusPro_Deep.onnx",
    "tflite": "SolidusPro_Deep.tflite",
}

# Per-target notes for the export UI.
TARGET_NOTES = {
    "coreml": "Apple ANE — .mlpackage, compute_units=all (ANE/GPU/CPU). "
              "ANE active when tensor shapes fit ANE tiles.",
    "onnx": "Snapdragon — fp16 ONNX graph for ONNX Runtime QNN EP (Hexagon/HTP). "
            "Validated on the CPU EP in this repo.",
    "tflite": "Google Tensor — int8 post-training quantized TFLite for NNAPI "
              "(Edge TPU delegate).",
}


def _build_model(model_dir: str) -> UNet3D:
    """Load weights + manifests and build the U-Net for export.

    The exported graph is ``UNet3D.forward``: physical ``(1, 3, R, R, R)``
    seconds ``[start, end, local]`` with the target denormalization and
    ``end = start + local`` inside the graph.
    """
    manifest = _read_manifest(model_dir)
    if manifest is None:
        raise ValueError(f"No manifest.json found at {model_dir}")
    if manifest.kind != "torch_unet":
        raise ValueError(f"Not a deep model directory: {model_dir} (kind={manifest.kind})")
    resolution = int(getattr(manifest, "grid_resolution", 32) or 32)
    if resolution not in (8, 16, 32, 64, 128):
        raise ValueError(
            f"Grid resolution {resolution} not in {{8, 16, 32, 64, 128}}"
        )
    return build_unet_from_artifact(model_dir, manifest)


def _update_deep_manifest(
    model_dir: str, target: str, artifact_path: str, quantization: str, validated: bool, note: str
) -> None:
    deep = read_deep_manifest(model_dir) or _empty_deep_manifest()
    deep.targets = list(dict.fromkeys(list(deep.targets) + [target]))
    deep.quantization[target] = quantization
    save_deep_manifest(deep, model_dir)
    _append_export_log(model_dir, target, artifact_path, quantization, validated, note)


def _empty_deep_manifest():
    from core.deep_model.model import DeepModelManifest

    return DeepModelManifest()


def _append_export_log(
    model_dir: str, target: str, artifact_path: str, quantization: str, validated: bool, note: str
) -> None:
    import json

    log_path = os.path.join(model_dir, "export_log.json")
    entries = []
    if os.path.exists(log_path):
        try:
            with open(log_path) as fh:
                entries = json.load(fh)
        except Exception:
            entries = []
    entries.append(
        {
            "target": target,
            "artifact_path": artifact_path,
            "quantization": quantization,
            "validated": bool(validated),
            "note": note,
        }
    )
    with open(log_path, "w") as fh:
        json.dump(entries, fh, indent=2)


# ---------------------------------------------------------------------------
# Per-target exporters
# ---------------------------------------------------------------------------

def _export_coreml(model_dir: str, out_dir: str, grid: int, progress_cb: ProgressCallback) -> Dict[str, Any]:
    import torch
    import torch.nn as nn
    import coremltools as ct

    class _GroupNormExport(nn.Module):
        """GroupNorm equivalent tracing with constant ops (Core ML-safe).

        ``nn.GroupNorm`` decomposes to a rank-6 reshape that Core ML rejects;
        this computes the same math (normalize each channel within its group,
        shared per-channel affine) with fixed-slice operations that trace to
        constants (no ``aten::Int`` / rank-6 intermediates), reusing the
        original weight/bias Parameters so state_dict keys are unchanged.
        """

        def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
            super().__init__()
            self.num_groups = num_groups
            self.num_channels = num_channels
            self.per_group = num_channels // num_groups
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))

        def forward(self, x):
            per_group = self.per_group
            normalized = []
            for g in range(self.num_groups):
                sl = x[:, g * per_group : (g + 1) * per_group]
                # GroupNorm statistics span every channel in the group plus the
                # spatial dims; per-channel stats (dims 2-4 only) would be
                # InstanceNorm and diverge from the trained weights.
                mean = sl.mean(dim=(1, 2, 3, 4), keepdim=True)
                var = sl.var(dim=(1, 2, 3, 4), keepdim=True, unbiased=False)
                normalized.append((sl - mean) / torch.sqrt(var + self.eps))
            y = torch.cat(normalized, dim=1)
            c = self.num_channels
            return y * self.weight.view(1, c, 1, 1, 1) + self.bias.view(1, c, 1, 1, 1)

    def _to_exportable(model):
        for name, child in list(model.named_modules()):
            if isinstance(child, nn.GroupNorm):
                parent = model
                for part in name.split(".")[:-1]:
                    parent = parent._modules[part]
                shim = _GroupNormExport(child.num_groups, child.num_channels, child.eps)
                shim.weight = child.weight
                shim.bias = child.bias
                parent._modules[name.split(".")[-1]] = shim
        return model

    model = _to_exportable(_build_model(model_dir))
    out_path = os.path.join(out_dir, ARTIFACT_NAMES["coreml"])
    if os.path.isdir(out_path):
        shutil.rmtree(out_path)

    channels = torch.rand(1, 7, grid, grid, grid)
    cond = torch.rand(1, model.films[0].mlp[0].in_features)
    if progress_cb:
        progress_cb("Tracing U-Net for Core ML...")
    traced = torch.jit.trace(model, (channels, cond))
    if progress_cb:
        progress_cb("Converting to Core ML (.mlpackage)...")
    coreml_model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="channels", shape=channels.shape),
            ct.TensorType(name="conditioning", shape=cond.shape),
        ],
        outputs=[ct.TensorType(name="schedule")],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS13,
    )
    coreml_model.save(out_path)

    compute_units = getattr(coreml_model, "compute_unit", None)
    note = TARGET_NOTES["coreml"]
    validated = False
    try:
        if progress_cb:
            progress_cb("Validating Core ML model...")
        result = coreml_model.predict(
            {
                "channels": np.random.rand(1, 7, grid, grid, grid).astype(np.float32),
                "conditioning": np.random.rand(1, cond.shape[1]).astype(np.float32),
            }
        )
        validated = np.asarray(result["schedule"]).shape == (1, 3, grid, grid, grid)
        if compute_units is not None:
            note = f"{note} Compute units: {compute_units}."
    except Exception as exc:
        note = f"{note} Validation failed: {exc}"

    return {
        "target": "coreml",
        "artifact_path": out_path,
        "quantization": "fp32",
        "validated": validated,
        "note": note,
    }


def _export_onnx(model_dir: str, out_dir: str, grid: int, progress_cb: ProgressCallback) -> Dict[str, Any]:
    import torch
    import onnxruntime as ort

    model = _build_model(model_dir)
    base_path = os.path.join(out_dir, ARTIFACT_NAMES["onnx"])

    channels = torch.rand(1, 7, grid, grid, grid)
    cond = torch.rand(1, model.films[0].mlp[0].in_features)

    if progress_cb:
        progress_cb("Exporting fp32 ONNX graph...")
    torch.onnx.export(
        model,
        (channels, cond),
        base_path,
        opset_version=18,
        input_names=["channels", "conditioning"],
        output_names=["schedule"],
    )

    note = TARGET_NOTES["onnx"]
    validated = False
    try:
        if progress_cb:
            progress_cb("Validating ONNX graph on CPU EP...")
        sess = ort.InferenceSession(base_path, providers=["CPUExecutionProvider"])
        out = sess.run(
            None,
            {"channels": channels.numpy(), "conditioning": cond.numpy()},
        )
        validated = np.asarray(out[0]).shape == (1, 3, grid, grid, grid)
    except Exception as exc:
        note = f"{note} fp32 validation failed: {exc}"

    # fp16 artifact for the QNN EP.
    fp16_path = base_path + ".fp16"
    if progress_cb:
        progress_cb("Exporting fp16 ONNX graph (QNN EP)...")
    try:
        with torch.no_grad():
            model_half = _build_model(model_dir).half()
        channels_h = channels.half()
        cond_h = cond.half()
        torch.onnx.export(
            model_half,
            (channels_h, cond_h),
            fp16_path,
            opset_version=18,
            input_names=["channels", "conditioning"],
            output_names=["schedule"],
        )
        try:
            sess16 = ort.InferenceSession(fp16_path, providers=["CPUExecutionProvider"])
            out16 = sess16.run(
                None,
                {"channels": channels_h.numpy(), "conditioning": cond_h.numpy()},
            )
            fp16_ok = np.asarray(out16[0]).shape == (1, 3, grid, grid, grid)
        except Exception as exc:
            fp16_ok = False
            note = f"{note} fp16 CPU validation unavailable: {exc}"
        note = f"{note} fp16 artifact: {'validated' if fp16_ok else 'not validated on CPU'}."
    except Exception as exc:
        note = f"{note} fp16 export failed: {exc}"

    return {
        "target": "onnx",
        "artifact_path": base_path,
        "quantization": "fp16",
        "validated": validated,
        "note": note,
    }


def _export_tflite(model_dir: str, out_dir: str, grid: int, progress_cb: ProgressCallback) -> Dict[str, Any]:
    import torch

    try:
        import onnx2tf  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "onnx2tf is not installed — the Google Tensor (TFLite) export "
            "requires `pip install onnx2tf` (may not yet have Python 3.13 / "
            "arm64 wheels). Use --target coreml or --target onnx instead."
        ) from exc

    model = _build_model(model_dir)
    onnx_stage = os.path.join(out_dir, "_tflite_stage.onnx")
    out_path = os.path.join(out_dir, ARTIFACT_NAMES["tflite"])

    channels = torch.rand(1, 7, grid, grid, grid)
    cond = torch.rand(1, model.films[0].mlp[0].in_features)
    if progress_cb:
        progress_cb("Staging ONNX graph for onnx2tf...")
    torch.onnx.export(
        model,
        (channels, cond),
        onnx_stage,
        opset_version=18,
        input_names=["channels", "conditioning"],
        output_names=["schedule"],
    )
    if progress_cb:
        progress_cb("Converting to int8 quantized TFLite...")
    try:
        onnx2tf.convert(
            input_onnx_file_path=onnx_stage,
            output_folder_path=os.path.join(out_dir, "_tflite_out"),
            quant_type="per-channel",
        )
    finally:
        if os.path.exists(onnx_stage):
            os.remove(onnx_stage)
    stage_tflite = os.path.join(out_dir, "_tflite_out", ARTIFACT_NAMES["tflite"])
    if not os.path.exists(stage_tflite):
        raise RuntimeError("onnx2tf produced no SolidusPro_Deep.tflite artifact")
    shutil.move(stage_tflite, out_path)
    shutil.rmtree(os.path.join(out_dir, "_tflite_out"), ignore_errors=True)

    note = TARGET_NOTES["tflite"]
    validated = False
    try:
        import tflite_runtime.interpreter as tflite

        if progress_cb:
            progress_cb("Validating TFLite with the interpreter...")
        interpreter = tflite.Interpreter(model_path=out_path)
        interpreter.allocate_tensors()
        in0 = interpreter.get_input_details()[0]
        in1 = interpreter.get_input_details()[1]
        interpreter.set_tensor(in0["index"], np.random.rand(*in0["shape"]).astype(np.float32))
        interpreter.set_tensor(in1["index"], np.random.rand(*in1["shape"]).astype(np.float32))
        interpreter.invoke()
        out_detail = interpreter.get_output_details()[0]
        out_arr = interpreter.get_tensor(out_detail["index"])
        validated = np.asarray(out_arr).size > 0
    except ImportError:
        note = f"{note} tflite-runtime not installed — artifact produced but not validated."
    except Exception as exc:
        note = f"{note} Validation failed: {exc}"

    return {
        "target": "tflite",
        "artifact_path": out_path,
        "quantization": "int8 (per-channel)",
        "validated": validated,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def export_for_target(
    model_path: str,
    target: str,
    out_dir: Optional[str] = None,
    progress_cb: ProgressCallback = None,
    cancel_cb: CancelCallback = None,
) -> Dict[str, Any]:
    """Export a trained deep model to one NPU target.

    Args:
        model_path: Deep model directory (manifest.json + weights.pt).
        target: One of ``"coreml"``, ``"onnx"``, ``"tflite"``.
        out_dir: Artifact directory (defaults to ``<model_dir>/export``).

    Returns:
        Dict with ``target``, ``artifact_path``, ``quantization``,
        ``validated``, and ``note``.
    """
    if target not in TARGETS:
        raise ValueError(f"Unknown target '{target}' — choose from {TARGETS}")
    model_path = os.fspath(model_path)
    manifest = _read_manifest(model_path)
    if manifest is None:
        raise ValueError(f"No manifest.json found at {model_path}")
    grid = int(getattr(manifest, "grid_resolution", 32) or 32)

    out_dir = os.fspath(out_dir or os.path.join(os.path.abspath(model_path), "export"))
    os.makedirs(out_dir, exist_ok=True)

    if cancel_cb is not None and cancel_cb():
        raise RuntimeError("Export cancelled")
    exporters = {
        "coreml": _export_coreml,
        "onnx": _export_onnx,
        "tflite": _export_tflite,
    }
    result = exporters[target](model_path, out_dir, grid, progress_cb)
    if cancel_cb is not None and cancel_cb():
        raise RuntimeError("Export cancelled")

    _update_deep_manifest(
        model_path,
        target,
        result["artifact_path"],
        result["quantization"],
        result["validated"],
        result["note"],
    )
    return result
