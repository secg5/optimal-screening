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

    def predict(self, prompt: str) -> dict[str, Any]:
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

        # Try JSON parsing (supports both keys, or single key)
        import json
        import re

        text = text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                result = {}

                # Handle bucket if present
                if "bucket" in data:
                    result["bucket"] = int(data["bucket"])

                # Handle probability if present
                if "probability" in data:
                    result["probability"] = float(data["probability"])

                # If we got at least one key, return
                if result:
                    return result
        except Exception:
            pass

        # Fallback: parse a bare number in [0,1] and infer bucket
        m = re.search(r"\b(?:(?:0(?:\.\d+)?)|(?:1(?:\.0+)?))\b", text)
        if m:
            prob = float(m.group(0))
            # Map probability to bucket for 2-bucket system
            bucket = 1 if prob < 0.5 else 2
            return {"bucket": bucket, "probability": prob}

        raise RuntimeError("Could not parse bucket/probability from local model response")
