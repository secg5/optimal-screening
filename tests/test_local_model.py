from __future__ import annotations

import json

from optimal_screening.models import local_model
from optimal_screening.models.local_model import LocalModel


def test_local_model_passes_engine_params(monkeypatch):
    created: dict[str, dict[str, object]] = {}

    class DummyLLM:
        def __init__(self, *args, **kwargs):
            created["args"] = args
            created["kwargs"] = kwargs

    monkeypatch.setattr(local_model, "LLM", DummyLLM)

    model = LocalModel("fake-model", engine_params={"tensor_parallel_size": 2})
    model.load_model()

    assert isinstance(model._llm, DummyLLM)
    assert created["kwargs"] == {"model": "fake-model", "tensor_parallel_size": 2}


def test_local_model_sampling_params(monkeypatch):
    captured: dict[str, object] = {}

    class DummySamplingParams:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    class DummyOutput:
        def __init__(self, text: str):
            self.text = text

    class DummyGenerated:
        def __init__(self, text: str):
            self.outputs = [DummyOutput(text)]

    class DummyLLM:
        def generate(self, prompts, sampling_params):
            captured["prompts"] = prompts
            captured["sampling_params_instance"] = sampling_params
            return [DummyGenerated("0.73")]

    monkeypatch.setattr(local_model, "SamplingParams", DummySamplingParams)

    model = LocalModel(
        "fake-model",
        max_tokens=12,
        sampling_params={"temperature": 0.4, "top_p": 0.9},
    )
    model._llm = DummyLLM()

    result = model.predict("Return a probability between 0 and 1")

    assert result == {"bucket": 2, "probability": 0.73}
    assert captured["prompts"] == ["Return a probability between 0 and 1"]
    assert captured["kwargs"] == {"max_tokens": 12, "temperature": 0.4, "top_p": 0.9}
    assert isinstance(captured["sampling_params_instance"], DummySamplingParams)


def test_local_model_predict_batch_handles_multi_prompts(monkeypatch):
    captured: dict[str, object] = {}

    class DummySamplingParams:
        def __init__(self, **kwargs):
            captured["sampling_kwargs"] = kwargs

    class DummyOutput:
        def __init__(self, text: str):
            self.text = text

    class DummyGenerated:
        def __init__(self, text: str):
            self.outputs = [DummyOutput(text)]

    class DummyLLM:
        def generate(self, prompts, sampling_params):
            captured.setdefault("prompts", []).append(list(prompts))
            outputs = []
            for prompt in prompts:
                if "bucket prompt" in prompt:
                    outputs.append(DummyGenerated("{\"bucket\": 2}"))
                elif "probability prompt" in prompt:
                    outputs.append(DummyGenerated("{\"probability\": 0.25}"))
                elif "naive prompt" in prompt:
                    outputs.append(DummyGenerated("{\"naive\": 0.1}"))
                else:
                    outputs.append(DummyGenerated("0.8"))
            return outputs

    monkeypatch.setattr(local_model, "SamplingParams", DummySamplingParams)

    model = LocalModel("fake-model", sampling_params={"temperature": 0.5})
    model._llm = DummyLLM()

    prompts = [
        json.dumps(
            {
                "bucket_prompt": "bucket prompt 1",
                "probability_prompt": "probability prompt 1",
                "naive_prompt": "naive prompt 1",
            }
        ),
        "single prompt",
    ]

    results = model.predict_batch(prompts)

    assert results == [
        {"bucket": 2, "probability": 0.25, "naive": 0.1},
        {"bucket": 2, "probability": 0.8},
    ]
    assert captured["sampling_kwargs"] == {"temperature": 0.5}
    assert captured["prompts"] == [
        ["single prompt"],
        ["bucket prompt 1"],
        ["probability prompt 1"],
        ["naive prompt 1"],
    ]
