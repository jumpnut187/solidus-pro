#!/usr/bin/env python3
"""
Train a surrogate solidification model from simulation data.

Thin CLI wrapper around ``core.surrogate.training``; the same logic is
available from the desktop UI (Surrogate > Retrain Surrogate Model).

Examples:

    # Generate a dataset + train the default model
    python tools/train_surrogate_model.py --generate

    # Train on an existing captured dataset
    python tools/train_surrogate_model.py --data-dir data/surrogate_dataset
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.materials import get_all_alloy_names
from core.surrogate.dataset import DEFAULT_DATASET_DIR, dataset_summary
from core.surrogate.training import (
    DEFAULT_MODEL_OUTPUT,
    generate_dataset,
    retrain_from_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a surrogate solidification model.")
    parser.add_argument("--generate", action="store_true",
                        help="Generate a synthetic dataset first (offline batch).")
    parser.add_argument("--data-dir", default=DEFAULT_DATASET_DIR,
                        help="Directory of captured .npz records.")
    parser.add_argument("--alloys", default="",
                        help="Comma-separated alloy names for generation (default: all catalog).")
    parser.add_argument("--grid", type=int, default=64, help="Grid resolution (16-128).")
    parser.add_argument("--name", default="Trained Linear Surrogate",
                        help="Model display name.")
    parser.add_argument("--output", default=DEFAULT_MODEL_OUTPUT,
                        help="Output directory for the trained model.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of training records used.")
    args = parser.parse_args()

    if args.generate:
        alloys = [a.strip() for a in args.alloys.split(",") if a.strip()]
        if not alloys:
            alloys = get_all_alloy_names()
        print(f"Generating dataset across {len(alloys)} alloys x synthetic geometries...")
        generate_dataset(alloys, args.data_dir, args.grid)

    summary = dataset_summary(args.data_dir)
    print(f"Dataset: {summary['count']} records, alloys: {summary['alloys']}")

    try:
        result = retrain_from_dataset(
            data_dir=args.data_dir,
            output_dir=args.output,
            grid=args.grid,
            name=args.name,
            limit=args.limit,
        )
    except ValueError as exc:
        print(f"\n{exc}")
        return 1

    headline = result["headline"]
    print(f"Fitted on {result['records']} records "
          f"({result['train']} train / {result['test']} test)")
    print(f"\n✓ Model saved to {result['saved_to']}")
    print(f"  RMSE {headline.get('rmse_s', 0):.1f}s · "
          f"MAE {headline.get('mae_s', 0):.1f}s · "
          f"conf {headline.get('confidence', 0):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())