from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


Row = Mapping[str, Any]


class BaseDataset(ABC):
    """Abstract dataset interface for CSV-backed datasets.

    Subclasses should implement encoding for variables and any preprocessing.
    """

    def __init__(
        self,
        csv_path: str | Path,
        variables: Sequence[str] | None = None,
        outcome_col: str | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.variables = list(variables) if variables is not None else None
        self.outcome_col = outcome_col
        self._rows: list[Row] | None = None

    def load_data(self) -> list[Row]:
        """Load data from CSV into a list of rows (dicts)."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        with self.csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self._rows = [dict(r) for r in reader]
        return self._rows

    def preprocess_data(self, rows: Iterable[Row]) -> list[Row]:
        """Preprocess rows. Default: identity (copy into a list)."""
        return [dict(r) for r in rows]

    @abstractmethod
    def encode_variable(self, name: str, value: Any) -> str:
        """Encode a single variable into text suitable for LLM prompts."""

    def get_prompt(self, row: Row, variables: Sequence[str] | None = None) -> str:
        """Generate dual prompts for bucket and probability prediction.

        Returns a JSON string containing two separate prompts that will be sent
        as separate API calls by the model's predict() method.

        - Excludes `outcome_col` from the covariate list.
        - Adds explicit line describing the outcome.
        """
        import json

        selected_all = list(variables) if variables is not None else (self.variables or list(row.keys()))

        # Exclude outcome from the covariate list
        excluded = {self.outcome_col} if self.outcome_col else set()
        covariates = [k for k in selected_all if k not in excluded]

        parts: list[str] = []
        if self.outcome_col:
            # Outcome may be a derived label (e.g., "PINCP > 50k"), so we only state its definition.
            parts.append(f"Outcome: {self.outcome_col}")

        # Add remaining covariates
        parts.extend(self.encode_variable(k, row.get(k)) for k in covariates)

        features = "\n".join(parts)

        # Bucket prompt
        bucket_header = (
            "You are an expert assistant that classifies individuals based on their likelihood "
            "of the specified outcome occurring. "
            "Analyze the features carefully and classify the individual into one of 2 buckets "
            "(1=low likelihood, 2=high likelihood). "
            "Return only a JSON object with one key: 'bucket' (integer 1-2). "
            "Do not include explanations or other fields."
        )
        bucket_prompt = bucket_header + "\n\n" + features

        # Probability prompt
        probability_header = (
            "You are an expert assistant that estimates the probability of the specified outcome "
            "occurring for individuals with the given characteristics. "
            "Analyze the features carefully and provide a probability estimate between 0 and 1, "
            "where 0 means the outcome is impossible and 1 means the outcome is certain. "
            "Return only a JSON object with one key: 'probability' (float 0-1). "
            "Do not include explanations or other fields."
        )
        probability_prompt = probability_header + "\n\n" + features

        # Naive screening prompt
        naive_header = (
            "You are an expert assistant helping to optimize a screening strategy. "
            "Context: We have a limited treatment budget (β) and want to maximize the number of treated individuals "
            "who actually have a positive outcome (Y=1). We also have a smaller screening budget (α) to obtain true labels. "
            "Our strategy: (1) Screen α proportion of the population to reveal their true outcomes, "
            "(2) Treat all screened individuals with Y=1, (3) Use remaining treatment budget for highest risk unscreened individuals. "
            "\n\n"
            "Given the individual's features below, provide a screening score between 0 and 1 indicating how valuable "
            "it would be to screen this person (0=not valuable to screen, 1=very valuable to screen). "
            "Consider that screening is most valuable when: (a) the person is uncertain (moderate risk), "
            "(b) screening could change the treatment decision, (c) the person might be Y=1 but not obvious from features. "
            "\n\n"
            "Return only a JSON object with one key: 'naive' (float 0-1). "
            "Do not include explanations or other fields."
        )
        naive_prompt = naive_header + "\n\n" + features

        # Return JSON string with all three prompts
        return json.dumps({
            "bucket_prompt": bucket_prompt,
            "probability_prompt": probability_prompt,
            "naive_prompt": naive_prompt
        })
