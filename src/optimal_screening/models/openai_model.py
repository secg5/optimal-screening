from __future__ import annotations

import json
import os
import re
from typing import Any

from .base import BaseModel


# Match a floating point number in [0,1], e.g., 0.42, 1, 0, 1.0
_PROB_RE = re.compile(r"\b(?:(?:0(?:\.\d+)?)|(?:1(?:\.0+)?))\b")

# Optional OpenAI client import hoisted to top-level
try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


class OAModel(BaseModel):
    """OpenAI-backed model returning a screening probability.

    Expects output as either JSON like {"probability": 0.42} or a bare
    probability number between 0 and 1 in the text.
    """

    def __init__(self, model_name: str, **params: Any) -> None:
        super().__init__(model_name, **params)
        self._client = None

    def load_model(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY for OAModel.")
        if OpenAI is None:
            raise RuntimeError("openai package not installed; install 'openai'.")
        try:
            self._client = OpenAI(api_key=api_key)
        except Exception as exc:  # pragma: no cover - SDK optional
            raise RuntimeError("Failed to initialize OpenAI client") from exc

    def predict(self, prompt: str) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Use Responses API if available; fall back to Chat Completions.
        text: str | None = None
        try:  # Newer SDK
            resp = self._client.responses.create(
                model=self.model_name,
                input=prompt,
                **self.params,
            )
            text = resp.output_text  # type: ignore[attr-defined]
        except Exception:
            try:  # Older chat.completions
                resp = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **self.params,
                )
                text = resp.choices[0].message.content  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("OpenAI response failed") from exc

        if not text:
            raise RuntimeError("Empty response from OpenAI model")

        text = text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line if it's ```json or just ```
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Try JSON parsing (supports both keys, or single key)
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

        # Fallback: parse a single float in [0,1] and infer bucket
        m = _PROB_RE.search(text)
        if m:
            prob = float(m.group(0))
            # Map probability to bucket for 2-bucket system
            # 0.0-0.5 -> bucket 1, 0.5-1.0 -> bucket 2
            bucket = 1 if prob < 0.5 else 2
            return {"bucket": bucket, "probability": prob}

        raise RuntimeError(f"Could not parse bucket/probability from model response. Response text: {text[:500]}")
