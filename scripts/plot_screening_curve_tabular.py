#!/usr/bin/env python
"""Plot optimal screening curves (TP Rate vs α) using tabular model risk scores.

Trains Logistic Regression, Random Forest, and XGBoost on a train split,
then computes optimal screening curves on the test split using each model's
predicted probabilities as risk scores.  Optionally includes gpt-4o-mini
LLM predictions.  Also includes empirical (strata-based) and random screening
baselines.

Usage:
    python scripts/plot_screening_curve_tabular.py \
        --csv data/acs_income_sample.csv \
        --outcome "PINCP > 50k" \
        --strata AGEP SEX \
        --beta 0.3 \
        --strata-only \
        --test-size 0.5 \
        --output runs/screening_tabular_acs_strata_only_50pct.png

    python scripts/plot_screening_curve_tabular.py \
        --csv data/landmines_granada_binary.csv \
        --outcome mines_outcome \
        --strata coca_within_7km old_mine_within_250m warm_weather \
        --beta 0.3 \
        --strata-only \
        --test-size 0.5 \
        --output runs/screening_tabular_landmines_strata_only_50pct.png

    # Include gpt-4o-mini predictions:
    python scripts/plot_screening_curve_tabular.py \
        --csv data/acs_income_sample.csv \
        --outcome "PINCP > 50k" \
        --strata AGEP SEX \
        --beta 0.3 \
        --strata-only \
        --test-size 0.5 \
        --include-llm \
        --dataset-class acs_income.ACSDataset \
        --output runs/screening_tabular_acs_strata_only_50pct.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


sys.path.insert(0, "src")

from optimal_screening.analysis import (
    compute_empirical_probabilities,
    compute_optimal_screening_curve,
    compute_random_screening_curve,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in X.columns:
        try:
            pd.to_numeric(X[col], errors="raise")
            numeric_cols.append(col)
        except (ValueError, TypeError):
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def _build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _build_pipelines(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([
            ("pre", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("pre", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]),
        "XGBoost": Pipeline([
            ("pre", preprocessor),
            ("clf", XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss")),
        ]),
    }


def _import_dataset_class(short_path: str):
    """Import a dataset class given 'module.ClassName' under optimal_screening.datasets."""
    full_path = f"optimal_screening.datasets.{short_path}"
    mod, _, attr = full_path.rpartition(".")
    module = __import__(mod, fromlist=[attr])
    return getattr(module, attr)


def _predict_llm(
    dataset: Any,
    rows: list[dict[str, Any]],
    variables: list[str],
    model_name: str = "gpt-4o-mini",
    requests_per_second: float = 5.0,
) -> np.ndarray:
    """Get probability predictions from an OpenAI LLM for each row."""
    from optimal_screening.models.openai_model import OAModel

    model = OAModel(model_name=model_name)
    model.load_model()

    interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
    probabilities: list[float] = []

    for i, row in enumerate(rows):
        prompt_json = dataset.get_prompt(row, variables)
        prompts = json.loads(prompt_json)
        prob_prompt = prompts["probability_prompt"]

        result = model._single_predict(prob_prompt)
        prob = float(result.get("probability", 0.5))
        prob = max(0.0, min(1.0, prob))
        probabilities.append(prob)

        if interval > 0:
            time.sleep(interval)

        if (i + 1) % 50 == 0 or (i + 1) == len(rows):
            print(f"    LLM progress: {i + 1}/{len(rows)}")

    return np.array(probabilities)


MODEL_STYLES: dict[str, dict[str, str]] = {
    "Logistic Regression": {"color": "#1f77b4", "marker": "o"},
    "Random Forest": {"color": "#2ca02c", "marker": "s"},
    "XGBoost": {"color": "#ff7f0e", "marker": "^"},
    "gpt-4o-mini": {"color": "#d62728", "marker": "P"},
    "Empirical (strata)": {"color": "#9467bd", "marker": "D"},
    "Random": {"color": "gray", "marker": "x"},
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot screening curves using tabular model risk scores")
    parser.add_argument("--csv", required=True, help="Path to CSV data file")
    parser.add_argument("--outcome", required=True, help="Outcome column name")
    parser.add_argument("--strata", nargs="+", required=True, help="Feature columns defining risk strata")
    parser.add_argument("--beta", type=float, required=True, help="Treatment budget β")
    parser.add_argument("--output", default="runs/screening_tabular.png", help="Output plot path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--include-llm", action="store_true", help="Include gpt-4o-mini LLM predictions")
    parser.add_argument(
        "--dataset-class",
        default=None,
        help="Dataset class for LLM prompt generation, e.g. 'acs_income.ACSDataset' or 'generic.GenericCSVDataset'",
    )
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="OpenAI model name (default: gpt-4o-mini)")
    parser.add_argument("--strata-only", action="store_true", help="Restrict ML models to strata features only")
    args = parser.parse_args()

    # --- Load data ---
    df = pd.read_csv(args.csv)
    outcome_col = args.outcome
    feature_cols = args.strata if args.strata_only else [c for c in df.columns if c != outcome_col]

    y = df[outcome_col].apply(lambda v: int(str(v).lower() in ("true", "1", "yes", "t", "y") if isinstance(v, str) else int(v) > 0 if isinstance(v, (int, float)) else 0))
    X = df[feature_cols]

    print(f"Dataset: {args.csv}")
    print(f"Samples: {len(X)}, Features: {X.shape[1]}, Positive rate: {y.mean():.2%}")

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y,
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # --- Detect column types & build pipelines ---
    numeric_cols, categorical_cols = _detect_column_types(X_train)
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)
    pipelines = _build_pipelines(preprocessor)

    # --- Train models and collect test-set predictions ---
    model_probs: dict[str, np.ndarray] = {}
    for name, pipeline in pipelines.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_test)[:, 1]
        model_probs[name] = probs
        auc = roc_auc_score(y_test, probs)
        print(f"  AUC: {auc:.4f}")

    # --- LLM predictions (optional) ---
    if args.include_llm:
        if not os.getenv("OPENAI_API_KEY"):
            print("\nSkipping LLM: OPENAI_API_KEY not set.")
        else:
            dataset_class_path = args.dataset_class or "generic.GenericCSVDataset"
            DatasetClass = _import_dataset_class(dataset_class_path)
            dataset = DatasetClass(csv_path=args.csv, outcome_col=outcome_col)
            dataset.load_data()

            # Build row dicts for the test set matching the dataset's internal row format
            test_indices = X_test.index.tolist()
            test_rows = [dataset._rows[i] for i in test_indices]
            variables = [c for c in df.columns if c != outcome_col]

            print(f"\nRunning LLM predictions ({args.llm_model})...")
            llm_probs = _predict_llm(
                dataset=dataset,
                rows=test_rows,
                variables=variables,
                model_name=args.llm_model,
            )
            model_probs[args.llm_model] = llm_probs
            auc = roc_auc_score(y_test, llm_probs)
            print(f"  {args.llm_model} — AUC: {auc:.4f}")

    # --- Compute empirical strata probabilities from training set ---
    train_df_for_strata = X_train.copy()
    train_df_for_strata[outcome_col] = y_train.values
    train_rows = train_df_for_strata.to_dict("records")
    train_empirical_probs = compute_empirical_probabilities(train_rows, outcome_col, args.strata)
    train_stratum_risk = {
        k: v["probability"] for k, v in train_empirical_probs.items()
    }

    # --- Build row dicts for the test set (needed by screening functions) ---
    test_df = X_test.copy()
    test_df[outcome_col] = y_test.values
    # Add each model's predictions as columns
    for name, probs in model_probs.items():
        col_name = f"_risk_{name}"
        test_df[col_name] = probs
    # Add training-set empirical strata risk (avoids leakage from test outcomes)
    test_df["_risk_empirical"] = [
        train_stratum_risk.get(tuple(row.get(f) for f in args.strata), 0.5)
        for row in X_test.to_dict("records")
    ]

    rows = test_df.to_dict("records")

    # --- Compute screening curves ---
    print(f"\nComputing screening curves (β={args.beta:.1%})...")

    all_results: dict[str, dict] = {}

    # 1. Empirical (strata-based) — uses training-set strata probabilities
    print("  Empirical (strata)...")
    all_results["Empirical (strata)"] = compute_optimal_screening_curve(
        rows=rows,
        outcome_col=outcome_col,
        strata_features=args.strata,
        prediction_col="_risk_empirical",
        beta=args.beta,
        use_custom_risk_col="_risk_empirical",
    )

    # 2. Each tabular model
    for name in model_probs:
        col_name = f"_risk_{name}"
        print(f"  {name}...")
        all_results[name] = compute_optimal_screening_curve(
            rows=rows,
            outcome_col=outcome_col,
            strata_features=args.strata,
            prediction_col=col_name,
            beta=args.beta,
            use_custom_risk_col=col_name,
        )

    # 3. Random baseline
    print("  Random baseline...")
    all_results["Random"] = compute_random_screening_curve(
        rows=rows,
        outcome_col=outcome_col,
        strata_features=args.strata,
        prediction_col="_risk_empirical",
        beta=args.beta,
        seed=args.seed,
        use_custom_risk_col="_risk_empirical",
    )

    # --- Plot ---
    total_pos = all_results["Empirical (strata)"]["total_positive"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, res in all_results.items():
        style = MODEL_STYLES.get(label, {"color": "black", "marker": "."})
        alpha_pct = [a * 100 for a in res["alpha_values"]]
        tp_pct = [tp / total_pos * 100 for tp in res["true_positives"]]
        linestyle = "--" if label == "Random" else "-"
        line_alpha = 0.7 if label == "Random" else 1.0
        ax.plot(
            alpha_pct, tp_pct,
            marker=style["marker"],
            color=style["color"],
            linewidth=2.5,
            markersize=6,
            label=label,
            linestyle=linestyle,
            alpha=line_alpha,
        )

    ax.set_xlabel("Screening Budget α (%)", fontsize=13)
    ax.set_ylabel("True Positive Rate (%)", fontsize=13)
    dataset_name = Path(args.csv).stem
    ax.set_title(
        f"Optimal Screening: True Positive Rate vs Screening Budget\n"
        f"Dataset: {dataset_name} | β = {args.beta:.0%}",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, args.beta * 100)
    ax.set_ylim(0, 105)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    plt.close(fig)

    # Save data to JSON
    json_path = output_path.with_suffix(".json")
    results_to_save = {}
    for label, res in all_results.items():
        results_to_save[label] = {
            "alpha_values": [float(a) for a in res["alpha_values"]],
            "true_positives": [int(tp) for tp in res["true_positives"]],
        }
    results_to_save["meta"] = {
        "beta": args.beta,
        "total_positive": total_pos,
        "total_samples": all_results["Empirical (strata)"]["total_samples"],
        "dataset": args.csv,
    }
    json_path.write_text(json.dumps(results_to_save, indent=2))
    print(f"Data saved to {json_path}")


if __name__ == "__main__":
    main()
