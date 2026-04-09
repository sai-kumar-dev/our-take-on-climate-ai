# Architecture

## Runtime Layers

1. Data pipeline
   `run_pipeline.py` builds the processed training table from raw climate, soil, and crop files.
2. Training
   `train_model.py` trains a versioned artifact and writes evaluation, model card, and run manifest files.
3. Inference API
   `src/app_api_entry.py` exposes prediction, context, simulation, feedback, and LLM explanation endpoints.
4. UI
   `src/ui_app_source.py` provides the farmer-facing Streamlit experience.

## Context Providers

- `live_weather_pending`
  Placeholder for future IMD or approved live provider integration.
- `historical_training_context`
  Current production fallback using historical district-month climatology and validation bands.

The inference layer uses a composite context provider so live observed plus forecast data can be added later without rewriting the API contract.

## Governance Outputs

Every training run now writes:
- `evaluation_report.json`
- `run_manifest.json`
- `model_card.md`
- registry entries in `artifacts/registry`

## Research Boundaries

The current model target is historical crop-share distribution. It is useful for localized pattern guidance, but it is not yet a true agronomic outcome model.
