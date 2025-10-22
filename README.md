# optimal-screening

## Quickstart: Run the ACS Example

This example runs the pipeline on `data/acs_income_sample.csv` using the ACS dataset encoder and an OpenAI model.

1) Install dependencies:

```
pip install -e .
```

2) Configure credentials and data:

- Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
- Ensure the CSV exists at `data/acs_income_sample.csv`.

3) Run with the provided config:

```
PYTHONPATH=src \
python -m optimal_screening --config examples/config.acs.yaml --limit 5
```

Notes
- The config `examples/config.acs.yaml` uses `dataset.class: acs_income.ACSDataset` and points to `data/acs_income_sample.csv`.
- Results are saved under `runs/run_*.json` with metadata and a per-row `probability`.
- To use a different backend (e.g., Bedrock or vLLM), adjust `model.type` in the config.
