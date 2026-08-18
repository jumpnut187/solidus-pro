#!/usr/bin/env python3
"""
Train the deep learning solidification surrogate (3D U-Net + FiLM).

Thin CLI wrapper around ``core.deep_model.training``; the same logic is
available from the desktop UI (Deep Model > Train Deep Model...) and runs
automatically via auto-train when the feature is enabled.

Requires torch (optional dependency — see requirements.txt).

Examples:

    # Train on the captured dataset at the default 32^3 resolution
    python tools/train_deep_model.py

    # Train with custom hyperparameters
    python tools/train_deep_model.py --grid 64 --epochs 200 --batch-size 8
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the deep solidification surrogate.")
    parser.add_argument("--data-dir", default="data/surrogate_dataset",
                        help="Directory of captured .npz records.")
    parser.add_argument("--grid", type=int, default=32,
                        help="Grid resolution, one of {16, 32, 64, 128} (default 32).")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--name", default="Trained Deep Surrogate (3D U-Net)",
                        help="Model display name.")
    parser.add_argument("--output", default="models/deep_model/trained",
                        help="Output directory for the trained model.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of training records used.")
    parser.add_argument("--min-records", type=int, default=4,
                        help="Minimum dataset size to train (default 4).")
    args = parser.parse_args()

    try:
        from core.deep_model.training import train_deep_model
    except ImportError as exc:
        print(f"\n{exc}")
        print("Install the deep-learning extras: "
              "venv/bin/pip install torch coremltools")
        return 1

    try:
        result = train_deep_model(
            data_dir=args.data_dir,
            output_dir=args.output,
            grid=args.grid,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            name=args.name,
            limit=args.limit,
            min_records=args.min_records,
            progress_cb=print,
        )
    except ValueError as exc:
        print(f"\n{exc}")
        return 1
    except RuntimeError as exc:
        print(f"\nTraining failed: {exc}")
        return 1

    headline = result["headline"]
    print(f"Trained on {result['records']} records "
          f"({result['train']} train / {result['test']} test) "
          f"on {result['device']} for {result['epochs']} epochs")
    print(f"\n✓ Model saved to {result['saved_to']}")
    print(f"  RMSE {headline.get('rmse_s', 0):.1f}s · "
          f"MAE {headline.get('mae_s', 0):.1f}s · "
          f"conf {headline.get('confidence', 0):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
