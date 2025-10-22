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

    def predict(self, prompt: str) -> float:
        if self._client is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        print(prompt)  # Use Responses API if available; fall back to Chat Completions.
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
        # Try JSON with a 'probability' key
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "probability" in data:
                return float(data["probability"])  # type: ignore[arg-type]
        except Exception:
            pass

        # Fallback: parse a single float in [0,1]
        m = _PROB_RE.search(text)
        if m:
            return float(m.group(0))

        raise RuntimeError("Could not parse probability from model response")
