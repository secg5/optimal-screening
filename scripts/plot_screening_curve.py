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

from optimal_screening.analysis import compute_optimal_screening_curve, compute_random_screening_curve, compute_screening_curve
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Plot screening curve")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--outcome", type=str, required=True, help="Outcome column name")
    parser.add_argument("--strata", nargs="+", required=True, help="Feature columns defining risk strata")
    parser.add_argument("--predictions", type=str, help="JSON file with model predictions (optional)")
    parser.add_argument("--output", type=str, default="runs/screening_curve.png", help="Output plot path")
    parser.add_argument("--screen-lowest", action="store_true", default=True, help="Screen lowest risk first")
    parser.add_argument("--beta", type=float, help="Treatment budget for optimal screening (enables optimal mode)")
    parser.add_argument("--compare-llm", action="store_true", help="Compare empirical vs LLM probabilities (requires --predictions with bucket data)")
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
                if args.compare_llm:
                    # Also load bucket data for comparison mode
                    bucket = pred.get("bucket", 1)
                    df.loc[i, "llm_bucket"] = bucket
                    # Convert bucket to probability for 2-bucket system
                    # Bucket 1 -> 0.25, Bucket 2 -> 0.75
                    df.loc[i, "bucket_probability"] = (bucket - 0.5) / 2.0
                    # Add naive screening score
                    df.loc[i, "naive_score"] = pred.get("naive", 0.5)

        # Trim data to match predictions in comparison mode
        if args.compare_llm and len(predictions) < len(df):
            df = df.iloc[:len(predictions)]
    else:
        # No predictions: use random baseline
        # TODO Change for uniform probability
        df["probability"] = 0.5

    rows = df.to_dict("records")

    print(f"Loaded {len(rows)} rows from {args.csv}")
    print(f"Outcome: {args.outcome}")
    print(f"Risk strata: {args.strata}")

    # Choose between optimal and simple screening
    if args.beta is not None:
        # Optimal screening with treatment budget
        if args.compare_llm:
            print(f"Mode: LLM Comparison with treatment budget β = {args.beta:.1%}")
            print()

            # Compute three curves for comparison
            print("Computing optimal screening curves...")
            print("  1. Empirical probabilities (from strata)")
            results_empirical = compute_optimal_screening_curve(
                rows=rows,
                outcome_col=args.outcome,
                strata_features=args.strata,
                prediction_col="probability",
                beta=args.beta,
            )

            print("  2. LLM predicted probabilities")
            results_llm_prob = compute_optimal_screening_curve(
                rows=rows,
                outcome_col=args.outcome,
                strata_features=args.strata,
                prediction_col="probability",
                beta=args.beta,
                use_custom_risk_col="probability",  # Use LLM probability column
            )

            print("  3. LLM bucket-based probabilities (bucket/10)")
            results_bucket = compute_optimal_screening_curve(
                rows=rows,
                outcome_col=args.outcome,
                strata_features=args.strata,
                prediction_col="probability",
                beta=args.beta,
                use_custom_risk_col="bucket_probability",  # Use bucket/10 column
            )

            print("  4. Naive screening scores")
            results_naive = compute_optimal_screening_curve(
                rows=rows,
                outcome_col=args.outcome,
                strata_features=args.strata,
                prediction_col="probability",
                beta=args.beta,
                use_custom_risk_col="naive_score",  # Use naive screening score column
            )

            print("  5. Random screening baseline")
            results_random = compute_random_screening_curve(
                rows=rows,
                outcome_col=args.outcome,
                strata_features=args.strata,
                prediction_col="probability",
                beta=args.beta,
                seed=42,
            )

            # Use empirical results for overall stats
            results = results_empirical

        else:
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

        # Helper function to create plot
        def create_optimal_plot(use_percentage=True):
            fig, ax = plt.subplots(figsize=(12, 7) if args.compare_llm else (10, 6))

            total_pos = results["total_positive"]

            if args.compare_llm:
                # Plot all five curves
                alpha_pct_emp = [a * 100 for a in results_empirical["alpha_values"]]
                alpha_pct_llm = [a * 100 for a in results_llm_prob["alpha_values"]]
                alpha_pct_bucket = [a * 100 for a in results_bucket["alpha_values"]]
                alpha_pct_naive = [a * 100 for a in results_naive["alpha_values"]]
                alpha_pct_random = [a * 100 for a in results_random["alpha_values"]]

                if use_percentage:
                    tp_emp = [tp / total_pos * 100 for tp in results_empirical["true_positives"]]
                    tp_llm = [tp / total_pos * 100 for tp in results_llm_prob["true_positives"]]
                    tp_bucket = [tp / total_pos * 100 for tp in results_bucket["true_positives"]]
                    tp_naive = [tp / total_pos * 100 for tp in results_naive["true_positives"]]
                    tp_random = [tp / total_pos * 100 for tp in results_random["true_positives"]]
                else:
                    tp_emp = results_empirical["true_positives"]
                    tp_llm = results_llm_prob["true_positives"]
                    tp_bucket = results_bucket["true_positives"]
                    tp_naive = results_naive["true_positives"]
                    tp_random = results_random["true_positives"]

                ax.plot(
                    alpha_pct_emp,
                    tp_emp,
                    marker="o",
                    linewidth=2.5,
                    markersize=7,
                    label="Empirical (from strata)",
                    color="blue",
                )

                ax.plot(
                    alpha_pct_llm,
                    tp_llm,
                    marker="s",
                    linewidth=2.5,
                    markersize=7,
                    label="LLM probability",
                    color="orange",
                )

                ax.plot(
                    alpha_pct_bucket,
                    tp_bucket,
                    marker="^",
                    linewidth=2.5,
                    markersize=7,
                    label="LLM bucket (quantile)",
                    color="green",
                )

                ax.plot(
                    alpha_pct_naive,
                    tp_naive,
                    marker="d",
                    linewidth=2.5,
                    markersize=7,
                    label="Naive screening",
                    color="red",
                )

                ax.plot(
                    alpha_pct_random,
                    tp_random,
                    marker="x",
                    linewidth=2.5,
                    markersize=8,
                    label="Random screening (baseline)",
                    color="gray",
                    alpha=0.7,
                    linestyle="--",
                )
            else:
                # Plot single curve
                alpha_pct = [a * 100 for a in results["alpha_values"]]
                if use_percentage:
                    tp_data = [tp / total_pos * 100 for tp in results["true_positives"]]
                else:
                    tp_data = results["true_positives"]
                ax.plot(alpha_pct, tp_data, marker="o", linewidth=2, markersize=6, label="Optimal screening")

            # Add baseline and perfect lines
            if results["true_positives"]:
                no_screen_tp = results["true_positives"][0]
                if use_percentage:
                    no_screen_val = no_screen_tp / total_pos * 100
                    perfect_val = 100.0
                    baseline_label = f"No screening (α=0, {no_screen_val:.1f}%)"
                    perfect_label = "Perfect (100%)"
                else:
                    no_screen_val = no_screen_tp
                    perfect_val = total_pos
                    baseline_label = f"No screening (α=0, TP={no_screen_tp})"
                    perfect_label = f"Perfect (TP={total_pos})"

                ax.axhline(
                    y=no_screen_val,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label=baseline_label,
                )

            ax.axhline(
                y=perfect_val,
                color="green",
                linestyle="--",
                alpha=0.7,
                label=perfect_label,
            )

            ax.set_xlabel("Screening Budget α (%)", fontsize=12 if not args.compare_llm else 13)
            ylabel = "True Positives Identified (%)" if use_percentage else "True Positives Correctly Identified"
            ax.set_ylabel(ylabel, fontsize=12 if not args.compare_llm else 13)

            if args.compare_llm:
                ax.set_title(
                    f"Optimal Screening Curves: Empirical vs LLM\nTreatment Budget β = {args.beta:.1%}",
                    fontsize=15,
                    fontweight="bold",
                )
            else:
                ax.set_title(
                    f"Optimal Screening Curve\nTreatment Budget β = {args.beta:.1%}",
                    fontsize=14,
                    fontweight="bold",
                )

            ax.legend(fontsize=10 if not args.compare_llm else 11, loc="best")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, args.beta * 100)

            if use_percentage:
                ax.set_ylim(0, 105)

            return fig

        # Create both plots
        fig_pct = create_optimal_plot(use_percentage=True)
        fig_count = create_optimal_plot(use_percentage=False)

        # Save results
        if args.compare_llm:
            results_to_save = {
                "beta": float(results["beta"]),
                "total_positive": int(results["total_positive"]),
                "total_samples": int(results["total_samples"]),
                "empirical_results": {
                    "alpha_values": [float(a) for a in results_empirical["alpha_values"]],
                    "true_positives": [int(tp) for tp in results_empirical["true_positives"]],
                },
                "llm_probability_results": {
                    "alpha_values": [float(a) for a in results_llm_prob["alpha_values"]],
                    "true_positives": [int(tp) for tp in results_llm_prob["true_positives"]],
                },
                "llm_bucket_results": {
                    "alpha_values": [float(a) for a in results_bucket["alpha_values"]],
                    "true_positives": [int(tp) for tp in results_bucket["true_positives"]],
                },
                "naive_results": {
                    "alpha_values": [float(a) for a in results_naive["alpha_values"]],
                    "true_positives": [int(tp) for tp in results_naive["true_positives"]],
                },
                "random_results": {
                    "alpha_values": [float(a) for a in results_random["alpha_values"]],
                    "true_positives": [int(tp) for tp in results_random["true_positives"]],
                },
            }
        else:
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

        # Helper function to create simple screening plot
        def create_simple_plot(use_percentage=True):
            fig, ax = plt.subplots(figsize=(10, 6))

            total_pos = curve_data["total_positive"]
            screening_pct = [b * 100 for b in curve_data["screening_budgets"]]

            if use_percentage:
                y_data = [tp / total_pos * 100 for tp in curve_data["true_positives"]]
                no_screen_val = curve_data["true_positives"][0] / total_pos * 100
                perfect_val = 100.0
                baseline_label = f"No screening ({no_screen_val:.1f}%)"
                perfect_label = "Perfect (100%)"
                ylabel = "True Positives Identified (%)"
            else:
                y_data = curve_data["true_positives"]
                no_screen_val = curve_data["true_positives"][0]
                perfect_val = total_pos
                baseline_label = f"No screening (TP={no_screen_val})"
                perfect_label = f"Perfect (TP={total_pos})"
                ylabel = "True Positives Correctly Identified"

            ax.plot(screening_pct, y_data, marker="o", linewidth=2, markersize=6, label="Screening curve")

            # Add baseline: no screening (all model predictions)
            ax.axhline(y=no_screen_val, color="red", linestyle="--", alpha=0.7, label=baseline_label)

            # Add perfect screening (all Y=1 identified)
            ax.axhline(y=perfect_val, color="green", linestyle="--", alpha=0.7, label=perfect_label)

            ax.set_xlabel("Proportion of Population Screened (%)", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(
                f"Screening Curve: True Positives vs Screening Budget\nStrategy: {curve_data['strategy']}", fontsize=14
            )
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 100)

            if use_percentage:
                ax.set_ylim(0, 105)

            return fig

        # Create both plots
        fig_pct = create_simple_plot(use_percentage=True)
        fig_count = create_simple_plot(use_percentage=False)

        results_to_save = curve_data

    # Save both plots
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save percentage plot
    output_pct_path = output_path.with_stem(output_path.stem + "_pct")
    fig_pct.tight_layout()
    fig_pct.savefig(output_pct_path, dpi=150, bbox_inches="tight")
    print(f"\nPercentage plot saved to {output_pct_path}")
    plt.close(fig_pct)

    # Save count plot
    output_count_path = output_path.with_stem(output_path.stem + "_count")
    fig_count.tight_layout()
    fig_count.savefig(output_count_path, dpi=150, bbox_inches="tight")
    print(f"Count plot saved to {output_count_path}")
    plt.close(fig_count)

    # Save data to JSON
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(results_to_save, indent=2))
    print(f"Data saved to {json_path}")


if __name__ == "__main__":
    main()
