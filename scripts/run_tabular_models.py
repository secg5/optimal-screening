#!/usr/bin/env python
"""Train ML models on tabular data, compute optimal screening curves, and save results to JSON.

Trains Logistic Regression, Random Forest, and XGBoost on a train split, then computes
optimal screening curves on the test split using each model's predicted probabilities as
risk scores. Optionally includes gpt-4o-mini LLM predictions.
Does NOT produce plots — run plot_figures.py to generate figures from the saved artifact.

The output JSON contains:
  - Screening curve data (alpha_values, true_positives) per model
  - Risk scores (predicted probabilities on test set) per model
  - y_test (true labels for the test set)
  - Metadata (beta, dataset, split info)

Usage:
    # Landmines — binary strata:
    python scripts/run_tabular_models.py \
        --csv data/landmines_granada_binary.csv \
        --outcome mines_outcome \
        --strata coca_within_7km old_mine_within_250m warm_weather \
        --beta 0.1 \
        --strata-only \
        --test-size 0.5 \
        --output runs/screening_tabular_landmines_strata_only_50pct.json

    # Landmines — raw features, Granada as test set:
    python scripts/run_tabular_models.py \
        --csv data/landmines_raw.csv \
        --outcome mines_outcome \
        --strata Municipio \
        --beta 0.1 \
        --test-municipio GRANADA \
        --output runs/screening_tabular_landmines_granada_test.json

    # ACS 2018 — all features as strata:
    python scripts/run_tabular_models.py \
        --csv data/acs_income_2018.csv \
        --outcome "PINCP > 50k" \
        --strata AGEP COW SCHL MAR OCCP POBP RELP WKHP SEX RAC1P \
        --beta 0.3 \
        --strata-only \
        --test-size 0.5 \
        --output runs/screening_tabular_acs2018_all_strata_50pct.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
    cache_path: str | None = None,
) -> np.ndarray:
    """Get probability predictions from an OpenAI LLM for each row."""
    if cache_path and Path(cache_path).exists():
        print(f"    Loading cached LLM predictions from {cache_path}")
        return np.array(json.loads(Path(cache_path).read_text()))

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

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(probabilities))
        print(f"    LLM predictions cached to {cache_path}")

    return np.array(probabilities)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tabular models and compute screening curves")
    parser.add_argument("--csv", required=True, help="Path to CSV data file")
    parser.add_argument("--outcome", required=True, help="Outcome column name")
    parser.add_argument("--strata", nargs="+", required=True, help="Feature columns defining risk strata")
    parser.add_argument("--beta", type=float, required=True, help="Treatment budget β")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction (default: 0.2)")
    parser.add_argument("--test-municipio", default=None, help="Use this Municipio value as test set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--include-llm", action="store_true", help="Include gpt-4o-mini LLM predictions")
    parser.add_argument("--dataset-class", default=None, help="Dataset class for LLM prompt generation")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="OpenAI model name (default: gpt-4o-mini)")
    parser.add_argument("--llm-cache", default=None, help="Path to cache LLM predictions as JSON")
    parser.add_argument("--strata-only", action="store_true", help="Restrict ML models to strata features only")
    args = parser.parse_args()

    # --- Load data ---
    df = pd.read_csv(args.csv)
    outcome_col = args.outcome
    feature_cols = args.strata if args.strata_only else [c for c in df.columns if c != outcome_col]

    y = df[outcome_col].apply(
        lambda v: int(str(v).lower() in ("true", "1", "yes", "t", "y") if isinstance(v, str)
                      else int(v) > 0 if isinstance(v, (int, float)) else 0)
    )
    X = df[feature_cols]

    print(f"Dataset: {args.csv}")
    print(f"Samples: {len(X)}, Features: {X.shape[1]}, Positive rate: {y.mean():.2%}")

    # --- Train/test split ---
    if args.test_municipio:
        municipio_mask = df["Municipio"].str.upper() == args.test_municipio.upper()
        X_train, X_test = X[~municipio_mask], X[municipio_mask]
        y_train, y_test = y[~municipio_mask], y[municipio_mask]
        print(f"Train: {len(X_train)} (all except {args.test_municipio}), Test: {len(X_test)} ({args.test_municipio})")
    else:
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

            test_indices = X_test.index.tolist()
            test_rows = [dataset._rows[i] for i in test_indices]
            variables = [c for c in df.columns if c != outcome_col]

            print(f"\nRunning LLM predictions ({args.llm_model})...")
            llm_probs = _predict_llm(
                dataset=dataset, rows=test_rows, variables=variables,
                model_name=args.llm_model, cache_path=args.llm_cache,
            )
            model_probs[args.llm_model] = llm_probs
            auc = roc_auc_score(y_test, llm_probs)
            print(f"  {args.llm_model} — AUC: {auc:.4f}")

    # --- Build row dicts for screening curve computation ---
    test_df = X_test.copy()
    test_df[outcome_col] = y_test.values
    for name, probs in model_probs.items():
        test_df[f"_risk_{name}"] = probs
    rows = test_df.to_dict("records")

    # --- Compute screening curves ---
    print(f"\nComputing screening curves (β={args.beta:.1%})...")

    all_results: dict[str, dict] = {}

    for name in model_probs:
        col_name = f"_risk_{name}"
        print(f"  {name}...")
        all_results[name] = compute_optimal_screening_curve(
            rows=rows, outcome_col=outcome_col, strata_features=args.strata,
            prediction_col=col_name, beta=args.beta, use_custom_risk_col=col_name,
        )

    first_col = f"_risk_{next(iter(model_probs))}"
    print("  Random baseline...")
    all_results["Random"] = compute_random_screening_curve(
        rows=rows, outcome_col=outcome_col, strata_features=args.strata,
        prediction_col=first_col, beta=args.beta, seed=args.seed, use_custom_risk_col=first_col,
    )

    # --- Save JSON artifact ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_to_save = {}
    for label, res in all_results.items():
        results_to_save[label] = {
            "alpha_values":  [float(a) for a in res["alpha_values"]],
            "true_positives": [int(tp) for tp in res["true_positives"]],
        }

    results_to_save["meta"] = {
        "beta": args.beta,
        "total_positive": next(iter(all_results.values()))["total_positive"],
        "total_samples":  next(iter(all_results.values()))["total_samples"],
        "dataset": args.csv,
    }

    # Risk scores: predicted probabilities and true labels for the test set
    results_to_save["risk_scores"] = {
        name: probs.tolist() for name, probs in model_probs.items()
    }
    results_to_save["risk_scores"]["y_test"] = y_test.tolist()

    output_path.write_text(json.dumps(results_to_save, indent=2))
    print(f"\nData saved to {output_path}")


if __name__ == "__main__":
    main()
