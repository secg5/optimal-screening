from __future__ import annotations

from optimal_screening.models import hf_model
from optimal_screening.models.hf_model import HFModel


def test_hf_model_loads_with_token(monkeypatch):
    created: dict[str, str] = {}

    class DummyOpenAI:
        def __init__(self, *, api_key: str, base_url: str):
            created["api_key"] = api_key
            created["base_url"] = base_url

    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setattr(hf_model, "OpenAI", DummyOpenAI)

    model = HFModel("Qwen/Qwen3-Next", base_url="https://custom.example.com/v1")
    model.load_model()

    assert created == {"api_key": "secret-token", "base_url": "https://custom.example.com/v1"}


def test_hf_model_predict_parses_probability(monkeypatch):
    params_capture: dict[str, dict[str, float]] = {}

    class FakeResponses:
        def create(self, *args, **kwargs):
            raise RuntimeError("responses endpoint unavailable")

    class FakeChat:
        def __init__(self):
            self.completions = self

        def create(self, model: str, messages: list[dict[str, str]], **kwargs):
            params_capture["kwargs"] = kwargs

            class Message:
                content = "0.61"

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeClient:
        def __init__(self):
            self.responses = FakeResponses()
            self.chat = FakeChat()

    model = HFModel("Qwen/Qwen3-Next", base_url="https://router.huggingface.co/v1", temperature=0.2)
    model._client = FakeClient()

    result = model.predict("Return a probability between 0 and 1")

    assert result == {"bucket": 2, "probability": 0.61}
    assert params_capture["kwargs"] == {"temperature": 0.2}
