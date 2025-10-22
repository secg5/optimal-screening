from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


Row = Mapping[str, Any]


class BaseDataset(ABC):
    """Abstract dataset interface for CSV-backed datasets.

    Subclasses should implement encoding for variables and any preprocessing.
    """

    def __init__(
        self,
        csv_path: str | Path,
        variables: Sequence[str] | None = None,
        treatment_col: str | None = None,
        outcome_col: str | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.variables = list(variables) if variables is not None else None
        self.treatment_col = treatment_col
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
        """Generate a prompt with explicit treatment/outcome and remaining covariates.

        - Excludes `treatment_col` and `outcome_col` from the generic variable list.
        - Adds explicit lines describing treatment and outcome.
        """
        selected_all = list(variables) if variables is not None else (self.variables or list(row.keys()))

        # Exclude treatment/outcome from the generic covariate list
        excluded = {c for c in (self.treatment_col, self.outcome_col) if c}
        covariates = [k for k in selected_all if k not in excluded]

        parts: list[str] = []
        if self.treatment_col:
            parts.append(
                f"Treatment column ({self.treatment_col}): "
                + self.encode_variable(self.treatment_col, row.get(self.treatment_col))
            )
        if self.outcome_col:
            # Outcome may be a derived label (e.g., "PINCP > 50k"), so we only state its definition.
            parts.append(f"Outcome column: {self.outcome_col}")

        # Add remaining covariates
        parts.extend(self.encode_variable(k, row.get(k)) for k in covariates)

        header = (
            "You are an assistant that estimates the probability that this row should be screened "
            "to determine whether the specified treatment was effective for the outcome. "
            "Return a single JSON object with a key 'probability' whose value is a number between 0 and 1. "
            "Do not include explanations."
        )
        return header + "\n\n" + "\n".join(parts)
