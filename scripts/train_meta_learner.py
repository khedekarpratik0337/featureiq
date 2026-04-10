#!/usr/bin/env python3
"""Train the FeatureIQ meta-learner and save to pretrained/ directory.

Usage:
    python scripts/train_meta_learner.py [--output PATH] [--n-tasks N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from featureiq.recommender.meta_learner import META_LEARNER_VERSION, MetaLearner


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FeatureIQ meta-learner")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the trained model. Defaults to pretrained/ directory.",
    )
    parser.add_argument(
        "--training-data",
        type=str,
        default=None,
        help="Path to JSON training data file from collect_openml_data.py",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        output_dir = Path(__file__).resolve().parent.parent / "featureiq" / "recommender" / "pretrained"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"meta_learner_{META_LEARNER_VERSION}.joblib")

    if args.training_data is None:
        print("No --training-data provided. Attempting to collect from OpenML...")
        try:
            from scripts.collect_openml_data import fetch_openml_tasks
            training_data = fetch_openml_tasks(n_tasks=50)
        except Exception as e:
            print(f"Failed to collect OpenML data: {e}")
            print("Please provide --training-data path to a pre-collected JSON file.")
            sys.exit(1)
    else:
        import json
        with open(args.training_data) as f:
            training_data = json.load(f)

    ml = MetaLearner()
    print(f"Training meta-learner on {len(training_data)} records...")
    scores = ml.train(training_data)

    print("\nValidation AUC per transformation:")
    for t_name, auc in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {t_name}: {auc:.4f}")

    ml.save(output_path)
    print(f"\nModel saved to: {output_path}")
    print(f"Version: {META_LEARNER_VERSION}")


if __name__ == "__main__":
    main()
