#!/usr/bin/env python
"""Train tabular models and call gpt-4o-mini, then plot risk-score densities.

Usage:
    python scripts/run_risk_density.py \
        --dataset-class acs_income.ACSDataset \
        --csv-path data/acs_income_sample.csv \
        --outcome-col "PINCP > 50k"

    python scripts/run_risk_density.py \
        --dataset-class landmines.LandmineDataset \
        --csv-path data/landmines_processed.csv \
        --outcome-col mines_outcome \
        --skip-llm \
        --output-dir runs/
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
from scipy.stats import gaussian_kde
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


sys.path.insert(0, "src")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_dataset_class(short_path: str):
    """Import a dataset class given 'module.ClassName' under optimal_screening.datasets."""
    full_path = f"optimal_screening.datasets.{short_path}"
    mod, _, attr = full_path.rpartition(".")
    module = __import__(mod, fromlist=[attr])
    return getattr(module, attr)


def _detect_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split columns into numeric and categorical based on convertibility to float."""
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
    """Build a ColumnTransformer that scales numerics and one-hot encodes categoricals."""
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _build_tabular_pipelines(
    preprocessor: ColumnTransformer,
) -> dict[str, Pipeline]:
    """Return name -> sklearn Pipeline for each tabular model."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("pre", preprocessor),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("pre", preprocessor),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("pre", preprocessor),
                ("clf", XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss")),
            ]
        ),
    }


# ---------------------------------------------------------------------------
# LLM prediction
# ---------------------------------------------------------------------------


def _predict_llm(
    dataset: Any,
    rows: list[dict[str, Any]],
    variables: list[str],
    model_name: str = "gpt-4o-mini",
    requests_per_second: float = 5.0,
) -> np.ndarray:
    """Get probability predictions from an OpenAI LLM for each row.

    Uses only the probability prompt (single API call per row).
    """
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

MODEL_STYLES: dict[str, dict[str, str]] = {
    "Logistic Regression": {"color": "#1f77b4", "linestyle": "-"},
    "Random Forest": {"color": "#2ca02c", "linestyle": "-"},
    "XGBoost": {"color": "#ff7f0e", "linestyle": "-"},
    "gpt-4o-mini": {"color": "#d62728", "linestyle": "--"},
}


def _plot_densities(
    scores: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
    output_path: Path,
    title: str = "Risk Score Density by Model",
) -> None:
    """Plot KDE density curves for each model's risk scores with performance metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x_grid = np.linspace(0, 1, 500)

    for name, values in scores.items():
        style = MODEL_STYLES.get(name, {"color": "gray", "linestyle": "-"})
        m = metrics.get(name, {})
        label_parts = [name]
        if m:
            label_parts.append(f"(AUC={m.get('auc', 0):.3f}, Brier={m.get('brier', 0):.3f})")
        label = " ".join(label_parts)

        if len(values) < 2 or values.std() < 1e-12:
            ax.axvline(x=values.mean(), label=label, **style, linewidth=2)
            continue
        kde = gaussian_kde(values, bw_method="scott")
        density = kde(x_grid)
        ax.plot(x_grid, density, label=label, linewidth=2.5, **style)

    ax.set_xlabel("Risk Score", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nDensity plot saved to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tabular models + call LLM, then plot risk-score densities")
    parser.add_argument(
        "--dataset-class",
        required=True,
        help="Dataset class under optimal_screening.datasets, e.g. 'acs_income.ACSDataset'",
    )
    parser.add_argument("--csv-path", required=True, help="Path to the CSV data file")
    parser.add_argument("--outcome-col", required=True, help="Name of the binary outcome column")
    parser.add_argument(
        "--variables",
        nargs="*",
        default=None,
        help="Feature columns to use (default: all except outcome)",
    )
    parser.add_argument("--output-dir", default="runs", help="Output directory (default: runs)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip the gpt-4o-mini predictions")
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="OpenAI model name for LLM predictions (default: gpt-4o-mini)",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    # --- Load dataset ---
    DatasetClass = _import_dataset_class(args.dataset_class)
    dataset = DatasetClass(
        csv_path=args.csv_path,
        variables=args.variables,
        outcome_col=args.outcome_col,
    )
    dataset.load_data()
    X, y = dataset.to_dataframe()

    print(f"Dataset: {args.dataset_class}")
    print(f"Samples: {len(X)}, Features: {X.shape[1]}, Positive rate: {y.mean():.2%}")

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # --- Detect column types ---
    numeric_cols, categorical_cols = _detect_column_types(X_train)
    print(f"Numeric features: {len(numeric_cols)}, Categorical features: {len(categorical_cols)}")

    # --- Build and train tabular models ---
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)
    pipelines = _build_tabular_pipelines(preprocessor)

    risk_scores: dict[str, np.ndarray] = {}
    model_metrics: dict[str, dict[str, float]] = {}

    for name, pipeline in pipelines.items():
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_test)[:, 1]
        risk_scores[name] = probs

        auc = roc_auc_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        model_metrics[name] = {"auc": auc, "brier": brier}
        print(f"  {name} — AUC: {auc:.4f}, Brier: {brier:.4f}, mean risk: {probs.mean():.4f}")

    # --- LLM predictions ---
    if not args.skip_llm:
        if not os.getenv("OPENAI_API_KEY"):
            print("\nSkipping LLM: OPENAI_API_KEY not set. Use --skip-llm to suppress this message.")
        else:
            print(f"\nRunning LLM predictions ({args.llm_model})...")
            # Get the test rows as dicts for prompt generation
            test_indices = X_test.index.tolist()
            test_rows = [dataset._rows[i] for i in test_indices]
            variables = dataset.variables or list(X.columns)

            llm_probs = _predict_llm(
                dataset=dataset,
                rows=test_rows,
                variables=variables,
                model_name=args.llm_model,
            )
            risk_scores[args.llm_model] = llm_probs

            auc = roc_auc_score(y_test, llm_probs)
            brier = brier_score_loss(y_test, llm_probs)
            model_metrics[args.llm_model] = {"auc": auc, "brier": brier}
            print(f"  {args.llm_model} — AUC: {auc:.4f}, Brier: {brier:.4f}, mean risk: {llm_probs.mean():.4f}")

    # --- Plot ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset_class.split(".")[-1].replace("Dataset", "").lower()
    output_path = output_dir / f"risk_density_{dataset_name}.png"

    _plot_densities(
        risk_scores,
        model_metrics,
        output_path,
        title=f"Risk Score Density — {dataset_name}",
    )


if __name__ == "__main__":
    main()
