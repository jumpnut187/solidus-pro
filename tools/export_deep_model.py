#!/usr/bin/env python3
"""
Export a trained deep surrogate to an NPU target.

All three artifacts are produced from the same trained weights
(``models/deep_model/trained``): Core ML for the Apple ANE, ONNX (fp16) for
Snapdragon QNN, TFLite (int8) for Google Tensor.

Requires torch + the per-target toolchain (see requirements.txt):

    python tools/export_deep_model.py --target coreml
    python tools/export_deep_model.py --target onnx
    python tools/export_deep_model.py --target tflite   # needs onnx2tf
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the deep surrogate for an NPU target.")
    parser.add_argument("--model", default="models/deep_model/trained",
                        help="Trained deep model directory (manifest.json + weights.pt).")
    parser.add_argument("--target", required=True, choices=["coreml", "onnx", "tflite"],
                        help="NPU target: Apple ANE (coreml), Snapdragon QNN (onnx), "
                             "Google Tensor (tflite).")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: <model_dir>/export).")
    args = parser.parse_args()

    try:
        from core.deep_model.export import export_for_target, TARGET_NOTES
    except ImportError as exc:
        print(f"\n{exc}")
        print("Install the deep-learning extras: "
              "venv/bin/pip install torch coremltools onnx onnxruntime")
        return 1

    if not os.path.isdir(args.model):
        print(f"\nNo trained model at {args.model} — train one first with "
              "tools/train_deep_model.py")
        return 1

    print(TARGET_NOTES[args.target])
    try:
        result = export_for_target(
            args.model,
            args.target,
            out_dir=args.out,
            progress_cb=print,
        )
    except ImportError as exc:
        print(f"\n{exc}")
        return 1
    except Exception as exc:
        print(f"\nExport failed: {exc}")
        return 1

    print(f"\n✓ {args.target.upper()} export complete")
    print(f"  Artifact: {result['artifact_path']}")
    print(f"  Quantization: {result['quantization']}")
    print(f"  Validated: {'yes' if result['validated'] else 'no'}")
    if result["note"] and result["note"] != TARGET_NOTES[args.target]:
        print(f"  Note: {result['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
