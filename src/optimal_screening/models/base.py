from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """Abstract model interface for probability screening from prompts."""

    def __init__(self, model_name: str, **params: Any) -> None:
        self.model_name = model_name
        self.params = params

    @abstractmethod
    def load_model(self) -> None:
        """Load underlying model/resources and validate credentials if needed."""

    @abstractmethod
    def predict(self, prompt: str) -> float:
        """Return probability (0..1) that the row should be screened."""
