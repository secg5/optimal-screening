from __future__ import annotations

from typing import Any

from .base import BaseModel

# Optional vLLM imports hoisted to top-level
try:  # pragma: no cover - optional dependency
    from vllm import LLM, SamplingParams  # type: ignore
except Exception:  # pragma: no cover
    LLM = None  # type: ignore[assignment]
    SamplingParams = None  # type: ignore[assignment]


class LocalModel(BaseModel):
    """Local vLLM-backed model.

    Requires `vllm` to be installed and a model path or name available locally.
    """

    def __init__(self, model_name: str, **params: Any) -> None:
        super().__init__(model_name, **params)
        self._llm = None

    def load_model(self) -> None:
        if LLM is None:
            raise RuntimeError("vllm package not installed; install 'vllm'.")
        try:
            self._llm = LLM(model=self.model_name)
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Failed to initialize vLLM model") from exc

    def predict(self, prompt: str) -> float:
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if SamplingParams is None:
            raise RuntimeError("vllm package not installed; install 'vllm'.")
        try:
            sp = SamplingParams(**{k: v for k, v in self.params.items() if k in {"temperature", "max_tokens"}})
            outputs = self._llm.generate([prompt], sp)
            text = outputs[0].outputs[0].text.strip() if outputs and outputs[0].outputs else ""
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("vLLM generation failed") from exc

        # Expect JSON with 'probability' or a bare number in [0,1]
        import json
        import re

        try:
            data = json.loads(text)
            if isinstance(data, dict) and "probability" in data:
                return float(data["probability"])  # type: ignore[arg-type]
        except Exception:
            pass

        m = re.search(r"\b(?:(?:0(?:\.\d+)?)|(?:1(?:\.0+)?))\b", text)
        if m:
            return float(m.group(0))

        raise RuntimeError("Could not parse probability from local model response")
