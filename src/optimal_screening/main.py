from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from .models.hf_model import HFModel
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
    if mtype in {"huggingface", "hf"}:
        return HFModel(model_name=model_name, **params)
    if mtype in {"bedrock", "aws"}:
        return BedrockModel(model_name=model_name, **params)
    if mtype in {"local", "vllm"}:
        return LocalModel(model_name=model_name, **params)
    raise ValueError(f"Unknown model type: {mtype}")


class RateLimiter:
    """Simple thread-safe rate limiter."""

    def __init__(self, requests_per_second: float | None) -> None:
        self._interval = 0.0
        if requests_per_second and requests_per_second > 0:
            self._interval = 1.0 / requests_per_second
        self._lock = threading.Lock()
        self._next_time = 0.0

    def acquire(self) -> None:
        if self._interval <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                wait_time = self._next_time - now
                if wait_time <= 0:
                    self._next_time = now + self._interval
                    return
            time.sleep(wait_time if wait_time > 0 else self._interval)


def _default_worker_count(model: BaseModel) -> int:
    if isinstance(model, LocalModel):
        cpu_count = os.cpu_count() or 1
        return max(1, min(cpu_count, 4))
    return 2


def _default_rate_limit(model: BaseModel) -> float | None:
    if isinstance(model, (OAModel, BedrockModel, HFModel)):
        return 1.0  # Safe default: 1 request per second
    return None


def _parallel_settings(model: BaseModel, run_cfg: dict[str, Any]) -> tuple[int, float | None]:
    parallel_cfg = run_cfg.get("parallelism") or {}
    if not isinstance(parallel_cfg, dict):
        parallel_cfg = {}

    max_workers = parallel_cfg.get("max_workers") or parallel_cfg.get("workers")
    if max_workers is not None:
        max_workers = int(max_workers)
    else:
        max_workers = _default_worker_count(model)

    requests_per_second = parallel_cfg.get("requests_per_second")
    if requests_per_second is None and parallel_cfg.get("requests_per_minute"):
        requests_per_minute = float(parallel_cfg["requests_per_minute"])
        requests_per_second = requests_per_minute / 60.0 if requests_per_minute > 0 else None
    if requests_per_second is not None:
        requests_per_second = max(float(requests_per_second), 0.0)
    else:
        requests_per_second = _default_rate_limit(model)

    return max(1, max_workers), requests_per_second


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

    predictions: list[dict[str, Any]]
    if not rows_to_process:
        predictions = []
    else:
        prompt_cache: dict[int, str] = {}
        # Print prompt for first row to aid debugging
        first_prompt = dataset.get_prompt(rows_to_process[0], variables)
        prompt_cache[0] = first_prompt
        print("=" * 80)
        print("PROMPT FOR ROW 0:")
        print("=" * 80)
        print(first_prompt)
        print("=" * 80)

        max_workers, requests_per_second = _parallel_settings(model, run_cfg)
        rate_limiter = RateLimiter(requests_per_second)
        predictions_by_index: dict[int, dict[str, Any]] = {}

        def _predict_single(idx: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            prompt = prompt_cache.get(idx)
            if prompt is None:
                prompt = dataset.get_prompt(row, variables)
            rate_limiter.acquire()
            result = model.predict(prompt)
            preds = process_predictions(result.get("probability", 0.5))
            pred_entry: dict[str, Any] = {
                "index": idx,
                "bucket": result.get("bucket", 1),
                "probability": preds,
            }
            if "naive" in result:
                pred_entry["naive"] = result.get("naive", 0.5)
            return idx, pred_entry

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_predict_single, idx, row) for idx, row in enumerate(rows_to_process)]
            with tqdm(total=len(rows_to_process), desc="Processing rows") as progress:
                for future in as_completed(futures):
                    idx, result = future.result()
                    predictions_by_index[idx] = result
                    progress.update(1)

        predictions = [predictions_by_index[i] for i in range(len(rows_to_process))]

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
            "has_hf_token": bool(
                os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            ),
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
