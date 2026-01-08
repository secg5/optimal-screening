from __future__ import annotations

from typing import Any

from .base import BaseDataset


class GenericCSVDataset(BaseDataset):
    """Basic CSV dataset with simple key: value encodings."""

    def encode_variable(self, name: str, value: Any) -> str:
        if value is None or value == "":
            return f"{name}: (missing)"
        return f"{name}: {value}"
