from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .datasets.acs_income import ACSDataset  # noqa: F401
from .datasets.generic import GenericCSVDataset  # noqa: F401


# Optional dotenv import hoisted to top-level
try:  # pragma: no cover - environment dependent
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - missing optional dep
    load_dotenv = None  # type: ignore[assignment]
from .models.base import BaseModel
from .models.bedrock_model import BedrockModel
from .models.local_model import LocalModel
from .models.openai_model import OAModel
from .prediction_processing import process_predictions
from .utils.io import ensure_dir, load_yaml


def _import_from_string_under_datasets(short_path: str):
    """Import a class given a short path under optimal_screening.datasets.

    Expects values like "module.ClassName" (e.g., "acs_income.ACSDataset").
    """
    if "." not in short_path:
        raise ValueError(
            "dataset.class must be of the form 'module.ClassName' under 'optimal_screening.datasets', e.g. 'acs_income.ACSDataset'",
        )
    full_path = f"optimal_screening.datasets.{short_path}"
    mod, _, attr = full_path.rpartition(".")
    module = __import__(mod, fromlist=[attr])
    return getattr(module, attr)


def _build_dataset(cfg: dict[str, Any]):
    ds = cfg.get("dataset", {})
    csv_path = ds.get("csv_path")
    variables = ds.get("variables")
    outcome_col = ds.get("outcome_col")
    cls_path: str | None = ds.get("class")
    if not csv_path:
        raise ValueError("dataset.csv_path is required")

    if not cls_path:
        raise ValueError(
            "dataset.class is required. Provide something like 'acs_income.ACSDataset' (under optimal_screening.datasets).",
        )

    DatasetClass = _import_from_string_under_datasets(cls_path)
    return DatasetClass(
        csv_path=csv_path,
        variables=variables,
        outcome_col=outcome_col,
    )


def _build_model(cfg: dict[str, Any]) -> BaseModel:
    m = cfg.get("model", {})
    mtype = (m.get("type") or "openai").lower()
    model_name = m.get("model_name") or "gpt-4.1-mini"
    params = m.get("params") or {}
    if mtype in {"openai", "oa"}:
        return OAModel(model_name=model_name, **params)
    if mtype in {"bedrock", "aws"}:
        return BedrockModel(model_name=model_name, **params)
    if mtype in {"local", "vllm"}:
        return LocalModel(model_name=model_name, **params)
    raise ValueError(f"Unknown model type: {mtype}")


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run optimal-screening experiment")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max rows to process (debug). Omit to process the full dataset.",
    )
    args = parser.parse_args(argv)

    # Load credentials from .env if python-dotenv is available
    if load_dotenv is not None:
        load_dotenv()

    cfg = load_yaml(args.config)
    run_cfg = cfg.get("run", {})
    outdir = ensure_dir(run_cfg.get("output_dir") or "runs")

    dataset = _build_dataset(cfg)
    rows = dataset.preprocess_data(dataset.load_data())
    variables = dataset.variables or (rows[0].keys() if rows else [])

    model = _build_model(cfg)
    model.load_model()

    limit_value = len(rows) if args.limit is None else max(0, min(len(rows), args.limit))
    rows_to_process = rows[:limit_value]

    predictions: list[dict[str, Any]] = []
    for i, row in tqdm(
        enumerate(rows_to_process),
        total=len(rows_to_process),
        desc="Processing rows",
    ):
        prompt = dataset.get_prompt(row, variables)
        if i == 0:
            print("=" * 80)
            print(f"PROMPT FOR ROW {i}:")
            print("=" * 80)
            print(prompt)
            print("=" * 80)
        result = model.predict(prompt)
        # result is now {"bucket": int, "probability": float, "naive": float}
        preds = process_predictions(result.get("probability", 0.5))
        pred_entry = {
            "index": i,
            "bucket": result.get("bucket", 1),
            "probability": preds,
        }
        # Add naive score if present
        if "naive" in result:
            pred_entry["naive"] = result.get("naive", 0.5)
        predictions.append(pred_entry)

    meta = {
        "dataset": {
            "name": type(dataset).__name__,
            "csv_path": str(getattr(dataset, "csv_path", "")),
            "variables": list(variables),
            "outcome_col": getattr(dataset, "outcome_col", None),
        },
        "model": {
            "class": type(model).__name__,
            "name": getattr(model, "model_name", ""),
            "params": getattr(model, "params", {}),
        },
        "env": {
            "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
            "has_aws_region": bool(os.getenv("AWS_REGION")),
        },
    }

    # Derive output filename from config name; replace dots with underscores.
    cfg_name = Path(args.config).stem  # e.g., "v55.00" -> "v55.00"
    base = re.sub(r"[^A-Za-z0-9]", "_", cfg_name)  # -> "v55_00"
    outfile = outdir / f"{base}.json"
    if outfile.exists():
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        outfile = outdir / f"{base}_{stamp}.json"

    payload = {"meta": meta, "predictions": predictions}
    outfile.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results -> {outfile}")


__all__ = ["run"]
