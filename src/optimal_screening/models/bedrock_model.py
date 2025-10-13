from __future__ import annotations

import json
import os
from typing import Any

from .base import BaseModel


class BedrockModel(BaseModel):
    """AWS Bedrock model calling the Chat/Converse APIs via boto3.

    Credentials must be provided via environment (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION).
    This class focuses on interface compatibility; exact wire format may require adjustment per provider/model.
    """

    def __init__(self, model_name: str, quantiles: list[float], **params: Any) -> None:
        super().__init__(model_name, quantiles, **params)
        self._client = None

    def load_model(self) -> None:
        if not os.getenv("AWS_REGION"):
            raise RuntimeError("Missing AWS_REGION for Bedrock model.")
        try:
            import boto3  # type: ignore

            self._client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION"))
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Failed to initialize boto3 bedrock-runtime client") from exc

    def predict(self, prompt: str) -> dict[str, float]:
        if self._client is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Prefer Converse if available; otherwise fall back to invoke_model with JSON body.
        body = {
            "inputText": prompt,
            "textGenerationConfig": {k: v for k, v in self.params.items() if k in {"temperature", "maxTokens"}},
        }
        try:
            response = self._client.invoke_model(
                modelId=self.model_name,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(response.get("body", b"{}"))
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Bedrock invocation failed") from exc

        # Attempt to extract JSON or key:value pairs from provider-specific fields
        text = ""
        if isinstance(payload, dict):
            # Common providers wrap in 'outputText' or 'results'
            if "outputText" in payload:
                text = str(payload["outputText"]).strip()
            elif "results" in payload and payload["results"]:
                text = str(payload["results"][0].get("outputText", "")).strip()

        if not text:
            # Fallback to raw body string
            text = json.dumps(payload)

        try:
            data = json.loads(text)
            return {str(k): float(v) for k, v in data.items()}
        except Exception:
            pass

        # If JSON parsing fails, return empty mapping to signal downstream parsing is needed.
        return {}

