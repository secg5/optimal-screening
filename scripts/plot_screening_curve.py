#!/usr/bin/env python
"""
Plot screening curve: true positives vs screening budget.

Usage:
    python scripts/plot_screening_curve.py --csv data/acs_income_sample.csv \
        --outcome "PINCP > 50k" \
        --strata AGEP SEX \
        --predictions runs/config_acs.json \
        --output runs/screening_curve.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from optimal_screening.analysis import compute_optimal_screening_curve, compute_screening_curve


def main():
    parser = argparse.ArgumentParser(description="Plot screening curve")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--outcome", type=str, required=True, help="Outcome column name")
    parser.add_argument("--strata", nargs="+", required=True, help="Feature columns defining risk strata")
    parser.add_argument("--predictions", type=str, help="JSON file with model predictions (optional)")
    parser.add_argument("--output", type=str, default="runs/screening_curve.png", help="Output plot path")
    parser.add_argument("--screen-lowest", action="store_true", default=True, help="Screen lowest risk first")
    parser.add_argument("--beta", type=float, help="Treatment budget for optimal screening (enables optimal mode)")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.csv)

    # Load predictions if provided
    if args.predictions:
        with open(args.predictions) as f:
            pred_data = json.load(f)
            predictions = pred_data.get("predictions", [])

        # Merge predictions with data
        for i, pred in enumerate(predictions):
            if i < len(df):
                df.loc[i, "probability"] = pred.get("probability", 0.5)
    else:
        # No predictions: use random baseline
        df["probability"] = 0.5

    rows = df.to_dict("records")

    print(f"Loaded {len(rows)} rows from {args.csv}")
    print(f"Outcome: {args.outcome}")
    print(f"Risk strata: {args.strata}")

    # Choose between optimal and simple screening
    if args.beta is not None:
        # Optimal screening with treatment budget
        print(f"Mode: Optimal screening with treatment budget β = {args.beta:.1%}")
        print()

        results = compute_optimal_screening_curve(
            rows=rows,
            outcome_col=args.outcome,
            strata_features=args.strata,
            prediction_col="probability",
            beta=args.beta,
        )

        # Print results
        print(f"Total samples: {results['total_samples']}")
        print(f"Total Y=1: {results['total_positive']}")
        print()
        print("Optimal Screening Results:")
        print(
            f"{'α':>6} {'TP':>6} {'B1∫':>8} {'B2∫':>8} {'B3∫':>8} {'B4∫':>8} "
            f"{'B1pop%':>8} {'B2pop%':>8} {'B3pop%':>8} {'B4pop%':>8} {'B3neg':>8} {'AvgRisk3':>10}"
        )
        print("-" * 130)
        for i, alpha in enumerate(results["alpha_values"]):
            tp = results["true_positives"][i]
            info = results["band_info"][i]
            print(
                f"{alpha:>6.1%} {tp:>6} "
                f"{info['band1_integral']:>8.3f} {info['band2_integral']:>8.3f} "
                f"{info['band3_integral']:>8.3f} {info['band4_integral']:>8.3f} "
                f"{info['band1_pop_prop']:>8.1%} {info['band2_pop_prop']:>8.1%} "
                f"{info['band3_pop_prop']:>8.1%} {info['band4_pop_prop']:>8.1%} "
                f"{info['band3_expected_negatives']:>8.3f} "
                f"{info['avg_risk_band3']:>10.3f}"
            )

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        alpha_pct = [a * 100 for a in results["alpha_values"]]
        ax.plot(alpha_pct, results["true_positives"], marker="o", linewidth=2, markersize=6, label="Optimal screening")

        # Add baseline and perfect lines
        if results["true_positives"]:
            no_screen_tp = results["true_positives"][0]
            ax.axhline(
                y=no_screen_tp, color="red", linestyle="--", alpha=0.7, label=f"No screening (α=0, TP={no_screen_tp})"
            )

        ax.axhline(
            y=results["total_positive"],
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Perfect (TP={results['total_positive']})",
        )

        ax.set_xlabel("Screening Budget α (%)", fontsize=12)
        ax.set_ylabel("True Positives Correctly Identified", fontsize=12)
        ax.set_title(f"Optimal Screening Curve\nTreatment Budget β = {args.beta:.1%}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        # Set x-axis from 0 to beta (in percentage)
        ax.set_xlim(0, args.beta * 100)

        # Save results
        results_to_save = {
            "beta": float(results["beta"]),
            "alpha_values": [float(a) for a in results["alpha_values"]],
            "true_positives": [int(tp) for tp in results["true_positives"]],
            "total_positive": int(results["total_positive"]),
            "total_samples": int(results["total_samples"]),
            "band_info": [
                {
                    "alpha": float(info["alpha"]),
                    "band1_integral": float(info["band1_integral"]),
                    "band2_integral": float(info["band2_integral"]),
                    "band3_integral": float(info["band3_integral"]),
                    "band4_integral": float(info["band4_integral"]),
                    "band1_pop_prop": float(info["band1_pop_prop"]),
                    "band2_pop_prop": float(info["band2_pop_prop"]),
                    "band3_pop_prop": float(info["band3_pop_prop"]),
                    "band4_pop_prop": float(info["band4_pop_prop"]),
                    "band3_expected_negatives": float(info["band3_expected_negatives"]),
                    "avg_risk_band3": float(info["avg_risk_band3"]),
                }
                for info in results["band_info"]
            ],
        }

    else:
        # Simple screening (lowest or highest risk first)
        print(f"Mode: Simple screening - {'lowest' if args.screen_lowest else 'highest'} risk first")
        print()

        curve_data = compute_screening_curve(
            rows=rows,
            outcome_col=args.outcome,
            strata_features=args.strata,
            prediction_col="probability",
            screen_lowest_risk=args.screen_lowest,
        )

        # Print summary
        print(f"Total samples: {curve_data['total_samples']}")
        print(f"Total Y=1: {curve_data['total_positive']}")
        print()
        print("Screening Budget → True Positives:")
        for budget, tp in zip(curve_data["screening_budgets"], curve_data["true_positives"]):
            print(f"  α={budget:.0%} → TP={tp} ({tp/curve_data['total_positive']:.1%} of all positives)")

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        screening_pct = [b * 100 for b in curve_data["screening_budgets"]]
        ax.plot(
            screening_pct, curve_data["true_positives"], marker="o", linewidth=2, markersize=6, label="Screening curve"
        )

        # Add baseline: no screening (all model predictions)
        no_screen_tp = curve_data["true_positives"][0]
        ax.axhline(y=no_screen_tp, color="red", linestyle="--", alpha=0.7, label=f"No screening (TP={no_screen_tp})")

        # Add perfect screening (all Y=1 identified)
        ax.axhline(
            y=curve_data["total_positive"],
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Perfect (TP={curve_data['total_positive']})",
        )

        ax.set_xlabel("Proportion of Population Screened (%)", fontsize=12)
        ax.set_ylabel("True Positives Correctly Identified", fontsize=12)
        ax.set_title(
            f"Screening Curve: True Positives vs Screening Budget\nStrategy: {curve_data['strategy']}", fontsize=14
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)

        results_to_save = curve_data

    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")

    # Save data to JSON
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(results_to_save, indent=2))
    print(f"Data saved to {json_path}")


if __name__ == "__main__":
    main()
