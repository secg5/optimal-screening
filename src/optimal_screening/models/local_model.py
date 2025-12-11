from __future__ import annotations

import json
import re
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
        super().__init__(model_name, is_api_model=False, **params)
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
            llm_kwargs = {"model": self.model_name, **self.engine_params}
            llm_kwargs.setdefault("tensor_parallel_size", num_gpus)
            self._llm = LLM(**llm_kwargs)
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Failed to initialize vLLM model") from exc

    def predict(self, prompt: str) -> dict[str, Any]:
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if SamplingParams is None:
            raise RuntimeError("vllm package not installed; install 'vllm'.")

        return self.predict_batch([prompt])[0]

    def predict_batch(self, prompts: list[str]) -> list[dict[str, Any]]:
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if SamplingParams is None:
            raise RuntimeError("vllm package not installed; install 'vllm'.")

        results: list[dict[str, Any]] = [{} for _ in prompts]
        single_indices: list[int] = []
        single_prompts: list[str] = []
        bucket_indices: list[int] = []
        bucket_prompts: list[str] = []
        probability_indices: list[int] = []
        probability_prompts: list[str] = []
        naive_indices: list[int] = []
        naive_prompts: list[str] = []

        for idx, prompt in enumerate(prompts):
            try:
                multi_prompts = json.loads(prompt)
                if (
                    isinstance(multi_prompts, dict)
                    and "bucket_prompt" in multi_prompts
                    and "probability_prompt" in multi_prompts
                ):
                    bucket_indices.append(idx)
                    bucket_prompts.append(multi_prompts["bucket_prompt"])
                    probability_indices.append(idx)
                    probability_prompts.append(multi_prompts["probability_prompt"])
                    if "naive_prompt" in multi_prompts:
                        naive_indices.append(idx)
                        naive_prompts.append(multi_prompts["naive_prompt"])
                    continue
            except (json.JSONDecodeError, TypeError):
                pass

            single_indices.append(idx)
            single_prompts.append(prompt)

        if single_prompts:
            try:
                texts = self._run_generate(single_prompts)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("vLLM generation failed") from exc
            for i, text in enumerate(texts):
                results[single_indices[i]] = self._parse_prediction_text(text)

        if bucket_prompts:
            try:
                bucket_outputs = self._run_generate(bucket_prompts)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("vLLM generation failed") from exc
            for i, text in enumerate(bucket_outputs):
                parsed = self._parse_prediction_text(text)
                if "bucket" in parsed:
                    results[bucket_indices[i]]["bucket"] = parsed["bucket"]

        if probability_prompts:
            try:
                probability_outputs = self._run_generate(probability_prompts)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("vLLM generation failed") from exc
            for i, text in enumerate(probability_outputs):
                parsed = self._parse_prediction_text(text)
                if "probability" in parsed:
                    results[probability_indices[i]]["probability"] = parsed["probability"]

        if naive_prompts:
            try:
                naive_outputs = self._run_generate(naive_prompts)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("vLLM generation failed") from exc
            for i, text in enumerate(naive_outputs):
                parsed = self._parse_prediction_text(text)
                if "naive" in parsed:
                    results[naive_indices[i]]["naive"] = parsed["naive"]

        for res in results:
            res.setdefault("bucket", 1)
            res.setdefault("probability", 0.5)

        return results

    def _run_generate(self, prompts: list[str]) -> list[str]:
        sp = SamplingParams(**self.sampling_params)
        outputs = self._llm.generate(prompts, sp)
        texts: list[str] = []
        for output in outputs:
            text = output.outputs[0].text.strip() if output and output.outputs else ""
            texts.append(text.strip())
        return texts

    def _parse_prediction_text(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()

        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                result: dict[str, Any] = {}

                if "bucket" in data:
                    result["bucket"] = int(data["bucket"])

                if "probability" in data:
                    result["probability"] = float(data["probability"])

                if "naive" in data:
                    result["naive"] = float(data["naive"])

                if result:
                    return result
        except Exception:
            pass

        m = re.search(r"\b(?:(?:0(?:\.\d+)?)|(?:1(?:\.0+)?))\b", cleaned)
        if m:
            prob = float(m.group(0))
            bucket = 1 if prob < 0.5 else 2
            return {"bucket": bucket, "probability": prob}

        raise RuntimeError("Could not parse bucket/probability from local model response")
