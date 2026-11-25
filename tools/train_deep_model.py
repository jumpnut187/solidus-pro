#!/usr/bin/env python3

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
    parser.add_argument("--grid", type=int, default=64,
                        help="Grid resolution, one of {16, 32, 64, 128} (default 64).")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum training epochs (early stopping may end sooner).")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate.")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early-stopping patience in epochs on validation loss (default 15).")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable x/y flips and z-rotations during training.")
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
            patience=args.patience,
            augment=not args.no_augment,
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
          f"({result['train']} train / {result['test']} held-out geometries) "
          f"on {result['device']} for {result['epochs_run']} epochs "
          f"(best epoch {result['best_epoch']})")
    print(f"\n✓ Model saved to {result['saved_to']}")
    print(f"  end RMSE {headline.get('end_rmse_s', 0):.1f}s vs "
          f"constant-mean baseline {headline.get('baseline_end_rmse_s', 0):.1f}s · "
          f"MAE {headline.get('mae_s', 0):.1f}s · "
          f"skill {headline.get('confidence', 0):.2f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
