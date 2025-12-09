from __future__ import annotations

import json
import os
import re
from typing import Any

from .base import BaseModel


# Optional OpenAI client import hoisted to top-level
try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


# Match a floating point number in [0,1], e.g., 0.42, 1, 0, 1.0
_PROB_RE = re.compile(r"\b(?:(?:0(?:\.\d+)?)|(?:1(?:\.0+)?))\b")
_DEFAULT_BASE_URL = "https://router.huggingface.co/v1"


class HFModel(BaseModel):
    """Hugging Face Inference Router model using the OpenAI client."""

    def __init__(self, model_name: str, base_url: str | None = None, **params: Any) -> None:
        base_url = base_url or params.pop("base_url", _DEFAULT_BASE_URL)
        super().__init__(model_name, **params)
        self.base_url = base_url
        self._client = None

    def load_model(self) -> None:
        api_key = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_API_KEY")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        if not api_key:
            raise RuntimeError("Missing HF_TOKEN or HUGGINGFACE_API_KEY for HFModel.")
        if OpenAI is None:
            raise RuntimeError("openai package not installed; install 'openai'.")
        try:
            self._client = OpenAI(api_key=api_key, base_url=self.base_url)
        except Exception as exc:  # pragma: no cover - SDK optional
            raise RuntimeError("Failed to initialize Hugging Face OpenAI-compatible client") from exc

    def predict(self, prompt: str) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Check if prompt is multi-prompt format (JSON with bucket_prompt, probability_prompt, naive_prompt)
        try:
            multi_prompts = json.loads(prompt)
            if (
                isinstance(multi_prompts, dict)
                and "bucket_prompt" in multi_prompts
                and "probability_prompt" in multi_prompts
            ):
                bucket_result = self._single_predict(multi_prompts["bucket_prompt"])
                probability_result = self._single_predict(multi_prompts["probability_prompt"])

                result = {
                    "bucket": bucket_result.get("bucket", 1),
                    "probability": probability_result.get("probability", 0.5),
                }

                if "naive_prompt" in multi_prompts:
                    naive_result = self._single_predict(multi_prompts["naive_prompt"])
                    result["naive"] = naive_result.get("naive", 0.5)

                return result
        except (json.JSONDecodeError, KeyError):
            pass

        return self._single_predict(prompt)

    def _single_predict(self, prompt: str) -> dict[str, Any]:
        """Make a single API call and parse the response via chat completions."""

        try:
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **self.params,
            )
            text = resp.choices[0].message.content  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Hugging Face chat completion failed") from exc

        if not text:
            raise RuntimeError("Empty response from Hugging Face model")

        text = text.strip()

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

                if "bucket" in data:
                    result["bucket"] = int(data["bucket"])

                if "probability" in data:
                    result["probability"] = float(data["probability"])

                if result:
                    return result
        except Exception:
            pass

        m = _PROB_RE.search(text)
        if m:
            prob = float(m.group(0))
            bucket = 1 if prob < 0.5 else 2
            return {"bucket": bucket, "probability": prob}

        raise RuntimeError(
            f"Could not parse bucket/probability from Hugging Face response. Response text: {text[:500]}"
        )
