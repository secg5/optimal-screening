#!/usr/bin/env python
"""
Example script to compute empirical P(Y=1|X) for different feature strata.

Usage:
    python scripts/analyze_strata.py --csv data/acs_income_sample.csv \
        --outcome "PINCP > 50k" \
        --strata AGEP SEX \
        --bins 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from optimal_screening.analysis import build_empirical_distribution, compute_empirical_probabilities, compute_quantiles


def main():
    parser = argparse.ArgumentParser(description="Compute empirical P(Y=1|X) for feature strata")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--outcome", type=str, required=True, help="Outcome column name")
    parser.add_argument("--strata", nargs="+", required=True, help="Feature columns defining strata")
    parser.add_argument("--bins", type=int, default=20, help="Number of histogram bins")
    parser.add_argument("--use-counts", action="store_true", help="Weight by stratum counts")
    parser.add_argument("--output", type=str, help="Output JSON file path (optional)")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.csv)
    rows = df.to_dict("records")

    print(f"Loaded {len(rows)} rows from {args.csv}")
    print(f"Outcome: {args.outcome}")
    print(f"Strata features: {args.strata}")
    print()

    # Compute empirical probabilities for each stratum
    empirical_probs = compute_empirical_probabilities(
        rows=rows,
        outcome_col=args.outcome,
        strata_features=args.strata,
    )

    print(f"Found {len(empirical_probs)} unique strata")
    print()

    # Show top 10 strata by probability
    sorted_strata = sorted(empirical_probs.items(), key=lambda x: x[1]["probability"], reverse=True)
    print("Top 10 strata by P(Y=1|X):")
    for i, (stratum_key, data) in enumerate(sorted_strata[:10], 1):
        features_str = ", ".join(f"{k}={v}" for k, v in data["features"].items())
        print(f"  {i}. {features_str}: P(Y=1|X)={data['probability']:.3f} ({data['positive_count']}/{data['count']})")
    print()

    # Build empirical distribution
    distribution = build_empirical_distribution(empirical_probs, bins=args.bins, use_counts=args.use_counts)

    print("Empirical distribution of P(Y=1|X):")
    print(f"  Mean: {distribution['mean']:.3f}")
    print(f"  Std:  {distribution['std']:.3f}")
    print(f"  Min:  {distribution['min']:.3f}")
    print(f"  Max:  {distribution['max']:.3f}")
    print()

    # Compute quantiles
    quantiles = compute_quantiles(empirical_probs, use_counts=args.use_counts)

    print("Quantiles of P(Y=1|X):")
    for q_name, q_value in sorted(quantiles.items()):
        print(f"  {q_name}: {q_value:.3f}")
    print()

    # Save results if output path provided
    if args.output:
        output_path = Path(args.output)
        results = {
            "config": {
                "csv_path": args.csv,
                "outcome_col": args.outcome,
                "strata_features": args.strata,
                "bins": args.bins,
                "use_counts": args.use_counts,
            },
            "n_strata": len(empirical_probs),
            "distribution": distribution,
            "quantiles": quantiles,
            "strata": {str(k): v for k, v in empirical_probs.items()},  # Convert tuple keys to strings for JSON
        }
        output_path.write_text(json.dumps(results, indent=2))
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
