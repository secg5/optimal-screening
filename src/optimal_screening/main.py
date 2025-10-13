from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


from .datasets.generic import GenericCSVDataset
from .models.base import BaseModel
from .models.bedrock_model import BedrockModel
from .models.local_model import LocalModel
from .models.openai_model import OAModel
from .prediction_processing import process_predictions
from .utils.io import ensure_dir, load_yaml_or_json


def _build_dataset(cfg: dict[str, Any]):
    ds = cfg.get("dataset", {})
    name = (ds.get("name") or "generic_csv").lower()
    csv_path = ds.get("csv_path")
    variables = ds.get("variables")
    if not csv_path:
        raise ValueError("dataset.csv_path is required")
    if name == "generic_csv":
        return GenericCSVDataset(csv_path=csv_path, variables=variables)
    raise ValueError(f"Unknown dataset: {name}")


def _build_model(cfg: dict[str, Any]) -> BaseModel:
    m = cfg.get("model", {})
    mtype = (m.get("type") or "openai").lower()
    model_name = m.get("model_name") or "gpt-4.1-mini"
    quantiles = m.get("quantiles") or [0.1, 0.5, 0.9]
    params = m.get("params") or {}
    if mtype in {"openai", "oa"}:
        return OAModel(model_name=model_name, quantiles=quantiles, **params)
    if mtype in {"bedrock", "aws"}:
        return BedrockModel(model_name=model_name, quantiles=quantiles, **params)
    if mtype in {"local", "vllm"}:
        return LocalModel(model_name=model_name, quantiles=quantiles, **params)
    raise ValueError(f"Unknown model type: {mtype}")


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run optimal-screening experiment")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML/JSON config")
    parser.add_argument("--limit", type=int, default=5, help="Max rows to process (debug)")
    args = parser.parse_args(argv)

    # Load credentials from .env if python-dotenv is available
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass

    cfg = load_yaml_or_json(args.config)
    run_cfg = cfg.get("run", {})
    outdir = ensure_dir(run_cfg.get("output_dir") or "runs")

    dataset = _build_dataset(cfg)
    rows = dataset.preprocess_data(dataset.load_data())
    variables = dataset.variables or (rows[0].keys() if rows else [])

    model = _build_model(cfg)
    model.load_model()

    predictions: list[dict[str, Any]] = []
    for i, row in enumerate(rows[: args.limit]):
        prompt = dataset.get_prompt(row, variables)
        raw = model.predict(prompt)
        preds = process_predictions(raw)
        predictions.append({"index": i, "predictions": preds})

    meta = {
        "dataset": {
            "name": type(dataset).__name__,
            "csv_path": str(getattr(dataset, "csv_path", "")),
            "variables": list(variables),
        },
        "model": {
            "class": type(model).__name__,
            "name": getattr(model, "model_name", ""),
            "quantiles": getattr(model, "quantiles", []),
            "params": getattr(model, "params", {}),
        },
        "env": {
            "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
            "has_aws_region": bool(os.getenv("AWS_REGION")),
        },
    }

    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    outfile = outdir / f"run_{stamp}.json"
    payload = {"meta": meta, "predictions": predictions}
    outfile.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results -> {outfile}")


__all__ = ["run"]
