#!/usr/bin/env python3

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
          f"({result['train']} train / {result['test']} val; "
          f"{result.get('train_groups', '?')} / {result.get('val_groups', '?')} geometry groups)")
    print(f"\n✓ Model saved to {result['saved_to']}")
    nan = float("nan")
    print(f"  end RMSE {headline.get('end_rmse_s', nan):.1f}s · "
          f"MAE {headline.get('end_mae_s', nan):.1f}s · "
          f"constant-mean baseline {headline.get('baseline_rmse_s', nan):.1f}s · "
          f"skill {headline.get('skill', nan):.2f}")
    print("  (active, non-chill cells with reference end > 0 on geometry-grouped validation)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
