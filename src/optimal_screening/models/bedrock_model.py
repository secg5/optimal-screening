from __future__ import annotations

import json
import os
from typing import Any

from .base import BaseModel


# Optional boto3 import hoisted to top-level
try:  # pragma: no cover - optional dependency
    import boto3  # type: ignore
except Exception:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]


class BedrockModel(BaseModel):
    """AWS Bedrock model calling the Chat/Converse APIs via boto3.

    Credentials must be provided via environment (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION).
    This class focuses on interface compatibility; exact wire format may require adjustment per provider/model.
    """

    def __init__(self, model_name: str, **params: Any) -> None:
        super().__init__(model_name, **params)
        self._client = None

    def load_model(self) -> None:
        if not os.getenv("AWS_REGION"):
            raise RuntimeError("Missing AWS_REGION for Bedrock model.")
        if boto3 is None:
            raise RuntimeError("boto3 is not installed; install 'boto3'.")
        try:
            self._client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION"))
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Failed to initialize boto3 bedrock-runtime client") from exc

    def predict(self, prompt: str) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Check if prompt is multi-prompt format (JSON with bucket_prompt, probability_prompt, naive_prompt)
        try:
            multi_prompts = json.loads(prompt)
            if isinstance(multi_prompts, dict) and "bucket_prompt" in multi_prompts and "probability_prompt" in multi_prompts:
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
        """Make a single Bedrock invocation and parse the response."""
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

        # Try JSON parsing (supports both keys, or single key)
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

        # Fallback: attempt to find a float in [0,1] and infer bucket
        import re

        m = re.search(r"\b(?:(?:0(?:\.\d+)?)|(?:1(?:\.0+)?))\b", text)
        if m:
            prob = float(m.group(0))
            # Map probability to bucket for 2-bucket system
            bucket = 1 if prob < 0.5 else 2
            return {"bucket": bucket, "probability": prob}

        raise RuntimeError("Could not parse bucket/probability from Bedrock response")
