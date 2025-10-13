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

    def __init__(self, csv_path: str | Path, variables: Sequence[str] | None = None) -> None:
        self.csv_path = Path(csv_path)
        self.variables = list(variables) if variables is not None else None
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
        """Generate a prompt string for a single row using selected variables."""
        selected = list(variables) if variables is not None else (self.variables or list(row.keys()))
        parts = [self.encode_variable(k, row.get(k)) for k in selected]
        header = (
            "You are an assistant that estimates numeric quantiles for a target variable. "
            "Return a JSON object mapping quantile to value."
        )
        return header + "\n\n" + "\n".join(parts)

