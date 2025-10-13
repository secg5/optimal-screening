from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """Abstract model interface for quantile prediction from prompts."""

    def __init__(self, model_name: str, quantiles: list[float], **params: Any) -> None:
        self.model_name = model_name
        self.quantiles = quantiles
        self.params = params

    @abstractmethod
    def load_model(self) -> None:
        """Load underlying model/resources and validate credentials if needed."""

    @abstractmethod
    def predict(self, prompt: str) -> dict[str, float]:
        """Return predicted quantiles as a mapping of quantile->value."""

