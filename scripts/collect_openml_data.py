"""CLI script to collect meta-learning training data from OpenML."""

from __future__ import annotations

import argparse
import logging

from featureiq.recommender.openml_collector import fetch_openml_tasks, save_training_data


def main() -> None:
    """Run the OpenML data collection pipeline."""
    parser = argparse.ArgumentParser(
        description="Collect meta-learning training data from OpenML"
    )
    parser.add_argument(
        "--n-tasks", type=int, default=50, help="Number of tasks to collect"
    )
    parser.add_argument(
        "--output", type=str, default="meta_training_data.joblib", help="Output path"
    )
    parser.add_argument(
        "--min-instances", type=int, default=100, help="Minimum dataset size"
    )
    parser.add_argument(
        "--max-instances", type=int, default=100000, help="Maximum dataset size"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    records = fetch_openml_tasks(
        n_tasks=args.n_tasks,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
    )
    save_training_data(records, args.output)
    print(f"Saved {len(records)} training records to {args.output}")


if __name__ == "__main__":
    main()
