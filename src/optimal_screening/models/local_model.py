from __future__ import annotations

from typing import Any

import torch

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
        self.engine_params = self._extract_engine_params(self.params)
        self.sampling_params = self._extract_sampling_params(self.params)
        self._llm = None

    @staticmethod
    def _extract_engine_params(params: dict[str, Any]) -> dict[str, Any]:
        engine_params = params.get("engine_params") or params.get("engine")
        if isinstance(engine_params, dict):
            return engine_params
        return {}

    @staticmethod
    def _extract_sampling_params(params: dict[str, Any]) -> dict[str, Any]:
        sampling_params = params.get("sampling_params") or params.get("sampling")
        sampling: dict[str, Any] = sampling_params if isinstance(sampling_params, dict) else {}
        legacy_sampling = {k: v for k, v in params.items() if k in {"temperature", "max_tokens"}}
        return {**legacy_sampling, **sampling}

    def load_model(self) -> None:
        if LLM is None:
            raise RuntimeError("vllm package not installed; install 'vllm'.")
        try:
            # Check number of GPUs available
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            print(f"Number of GPUs available: {num_gpus}")
            self._llm = LLM(model=self.model_name, **self.engine_params)
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Failed to initialize vLLM model") from exc

    def predict(self, prompt: str) -> dict[str, Any]:
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if SamplingParams is None:
            raise RuntimeError("vllm package not installed; install 'vllm'.")

        import json
        import re

        # Check if prompt is multi-prompt format (JSON with bucket_prompt, probability_prompt, naive_prompt)
        try:
            multi_prompts = json.loads(prompt)
            if (
                isinstance(multi_prompts, dict)
                and "bucket_prompt" in multi_prompts
                and "probability_prompt" in multi_prompts
            ):
                # Make separate API calls for each prompt
                bucket_result = self._single_predict(multi_prompts["bucket_prompt"])
                probability_result = self._single_predict(multi_prompts["probability_prompt"])

                result = {
                    "bucket": bucket_result.get("bucket", 1),
                    "probability": probability_result.get("probability", 0.5),
                }

                # Add naive score if present
                if "naive_prompt" in multi_prompts:
                    naive_result = self._single_predict(multi_prompts["naive_prompt"])
                    result["naive"] = naive_result.get("naive", 0.5)

                return result
        except (json.JSONDecodeError, KeyError):
            pass  # Not multi-prompt format, continue with single prediction

        # Single prediction mode (backward compatible)
        return self._single_predict(prompt)

    def _single_predict(self, prompt: str) -> dict[str, Any]:
        """Make a single vLLM generation and parse the response."""
        import json
        import re

        try:
            sp = SamplingParams(**self.sampling_params)
            outputs = self._llm.generate([prompt], sp)
            text = outputs[0].outputs[0].text.strip() if outputs and outputs[0].outputs else ""
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("vLLM generation failed") from exc

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
