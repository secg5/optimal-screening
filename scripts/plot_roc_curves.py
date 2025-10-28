#!/usr/bin/env python
"""
Plot ROC curves comparing empirical probabilities vs LLM predictions.

Usage:
    python scripts/plot_roc_curves.py --csv data/acs_income_sample.csv \
        --outcome "PINCP > 50k" \
        --strata AGEP SEX \
        --predictions runs/config_acs.json \
        --output runs/roc_curves.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimal_screening.analysis import compute_empirical_probabilities


def compute_roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute ROC curve and AUC.

    Args:
        y_true: True binary labels (0 or 1)
        y_scores: Predicted probabilities or scores

    Returns:
        Tuple of (fpr, tpr, auc)
    """
    # Sort by predicted score (descending)
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_scores_sorted = y_scores[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]

    # Compute TPR and FPR at each threshold
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos

    tpr_list = [0.0]
    fpr_list = [0.0]

    tp = 0
    fp = 0

    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1

        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0

        # Only add point if it changes the curve
        if tpr != tpr_list[-1] or fpr != fpr_list[-1]:
            tpr_list.append(tpr)
            fpr_list.append(fpr)

    # Ensure we end at (1, 1)
    if tpr_list[-1] != 1.0 or fpr_list[-1] != 1.0:
        tpr_list.append(1.0)
        fpr_list.append(1.0)

    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)

    # Compute AUC using trapezoidal rule
    auc = np.trapz(tpr, fpr)

    return fpr, tpr, auc


def main():
    parser = argparse.ArgumentParser(description="Plot ROC curves for probability comparison")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--outcome", type=str, required=True, help="Outcome column name")
    parser.add_argument("--strata", nargs="+", required=True, help="Feature columns defining risk strata")
    parser.add_argument("--predictions", type=str, required=True, help="JSON file with model predictions")
    parser.add_argument("--output", type=str, default="runs/roc_curves.png", help="Output plot path")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.csv)

    # Load predictions
    with open(args.predictions) as f:
        pred_data = json.load(f)
        predictions = pred_data.get("predictions", [])

    # Merge predictions with data
    for i, pred in enumerate(predictions):
        if i < len(df):
            df.loc[i, "llm_probability"] = pred.get("probability", 0.5)
            bucket = pred.get("bucket", 1)
            df.loc[i, "llm_bucket"] = bucket
            # Convert bucket to probability for 2-bucket system
            # Bucket 1 -> 0.25, Bucket 2 -> 0.75
            df.loc[i, "bucket_probability"] = (bucket - 0.5) / 2.0

    # Trim data to match predictions
    if len(predictions) < len(df):
        df = df.iloc[: len(predictions)]

    rows = df.to_dict("records")

    print(f"Loaded {len(rows)} rows from {args.csv}")
    print(f"Outcome: {args.outcome}")
    print(f"Risk strata: {args.strata}")
    print()

    # Compute empirical probabilities from strata
    empirical_probs = compute_empirical_probabilities(rows, args.outcome, args.strata)

    # Prepare arrays
    y_true = []
    empirical_scores = []
    llm_scores = []
    bucket_scores = []

    for row in rows:
        # Get true outcome
        outcome = row.get(args.outcome)
        if isinstance(outcome, bool):
            y_val = int(outcome)
        elif isinstance(outcome, str):
            y_val = int(outcome.lower() in ("true", "1", "yes", "t", "y"))
        else:
            y_val = int(bool(outcome))

        y_true.append(y_val)

        # Get empirical probability
        stratum_key = tuple(row.get(f) for f in args.strata)
        emp_prob = empirical_probs.get(stratum_key, {}).get("probability", 0.5)
        empirical_scores.append(emp_prob)

        # Get LLM scores
        llm_scores.append(row.get("llm_probability", 0.5))
        bucket_scores.append(row.get("bucket_probability", 0.5))

    y_true = np.array(y_true)
    empirical_scores = np.array(empirical_scores)
    llm_scores = np.array(llm_scores)
    bucket_scores = np.array(bucket_scores)

    print("Computing ROC curves...")
    fpr_emp, tpr_emp, auc_emp = compute_roc_curve(y_true, empirical_scores)
    fpr_llm, tpr_llm, auc_llm = compute_roc_curve(y_true, llm_scores)
    fpr_bucket, tpr_bucket, auc_bucket = compute_roc_curve(y_true, bucket_scores)

    print(f"  Empirical AUC: {auc_emp:.4f}")
    print(f"  LLM Probability AUC: {auc_llm:.4f}")
    print(f"  LLM Bucket AUC: {auc_bucket:.4f}")
    print()

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot ROC curves
    ax.plot(
        fpr_emp,
        tpr_emp,
        marker="o",
        linewidth=2.5,
        markersize=4,
        label=f"Empirical (AUC = {auc_emp:.3f})",
        color="blue",
        alpha=0.8,
    )

    ax.plot(
        fpr_llm,
        tpr_llm,
        marker="s",
        linewidth=2.5,
        markersize=4,
        label=f"LLM Probability (AUC = {auc_llm:.3f})",
        color="orange",
        alpha=0.8,
    )

    ax.plot(
        fpr_bucket,
        tpr_bucket,
        marker="^",
        linewidth=2.5,
        markersize=4,
        label=f"LLM Bucket (AUC = {auc_bucket:.3f})",
        color="green",
        alpha=0.8,
    )

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.5, label="Random (AUC = 0.500)")

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curves: Empirical vs LLM Predictions", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    # Save plot
    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"ROC curve plot saved to {output_path}")

    # Save AUC scores to JSON
    results = {
        "empirical_auc": float(auc_emp),
        "llm_probability_auc": float(auc_llm),
        "llm_bucket_auc": float(auc_bucket),
        "n_samples": len(y_true),
        "n_positive": int(np.sum(y_true)),
        "n_negative": int(len(y_true) - np.sum(y_true)),
    }

    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(results, indent=2))
    print(f"AUC scores saved to {json_path}")


if __name__ == "__main__":
    main()
