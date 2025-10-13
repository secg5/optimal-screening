from __future__ import annotations

import json
import os
import re
from typing import Any

from .base import BaseModel


_NUM_RE = re.compile(r"([01](?:\.\d+)?)\s*[:=]\s*(-?\d+(?:\.\d+)?)")


class OAModel(BaseModel):
    """OpenAI-backed model using the official SDK.

    Expects output either as JSON or simple lines like `0.1: 12.3`.
    """

    def __init__(self, model_name: str, quantiles: list[float], **params: Any) -> None:
        super().__init__(model_name, quantiles, **params)
        self._client = None

    def load_model(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY for OAModel.")
        try:
            from openai import OpenAI  # type: ignore

            self._client = OpenAI(api_key=api_key)
        except Exception as exc:  # pragma: no cover - SDK optional
            raise RuntimeError("Failed to initialize OpenAI client") from exc

    def predict(self, prompt: str) -> dict[str, float]:
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
        # Try JSON first
        try:
            data = json.loads(text)
            return {str(k): float(v) for k, v in data.items()}
        except Exception:
            pass

        # Fallback: parse lines like "0.1: 12.3"
        preds: dict[str, float] = {}
        for q, v in _NUM_RE.findall(text):
            preds[str(q)] = float(v)
        return preds

