from __future__ import annotations

from typing import Any

from .base import BaseModel


class LocalModel(BaseModel):
    """Local vLLM-backed model.

    Requires `vllm` to be installed and a model path or name available locally.
    """

    def __init__(self, model_name: str, quantiles: list[float], **params: Any) -> None:
        super().__init__(model_name, quantiles, **params)
        self._llm = None

    def load_model(self) -> None:
        try:
            from vllm import LLM  # type: ignore

            self._llm = LLM(model=self.model_name)
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Failed to initialize vLLM model") from exc

    def predict(self, prompt: str) -> dict[str, float]:
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        try:
            from vllm import SamplingParams  # type: ignore

            sp = SamplingParams(**{k: v for k, v in self.params.items() if k in {"temperature", "max_tokens"}})
            outputs = self._llm.generate([prompt], sp)
            text = outputs[0].outputs[0].text.strip() if outputs and outputs[0].outputs else ""
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("vLLM generation failed") from exc

        # Simple parsing expectation: JSON mapping quantile->value
        import json

        try:
            data = json.loads(text)
            return {str(k): float(v) for k, v in data.items()}
        except Exception:
            return {}

