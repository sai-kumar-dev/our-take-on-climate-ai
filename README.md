# Climate-Aware Crop Recommendation System

[![Python](https://img.shields.io/badge/Python-ML%20Pipeline-3776AB?logo=python&logoColor=white)](requirements.txt)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ranking%20Model-E76F51)](requirements.txt)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](src/app_api_entry.py)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)](src/ui_app_source.py)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-6C5CE7)](artifacts/evaluation/shap)
[![Groq](https://img.shields.io/badge/Groq-LLM%20Explanations-111111)](src/climate_pipeline/scenario_explainer.py)

This repository implements a climate-aware crop recommendation platform that combines district-level climate, soil, and crop-pattern data into a multi-output machine learning ranking system.

The project is designed as an end-to-end stack: a configurable data pipeline, an XGBoost-based ranking model, reproducible evaluation artifacts, a FastAPI serving layer, a Streamlit user interface, SHAP-based interpretability, and Groq-powered explanation modules for both farmer-facing guidance and scenario analysis.

For a deeper engineering walkthrough, see [PROJECT_GUIDE.md](PROJECT_GUIDE.md).

## 1. Features

- Multi-source data integration from climate, soil, and crop-distribution tables
- District-level crop guidance with localized context lookup
- Multi-output ranking model that predicts a crop suitability distribution instead of a single hard label
- Scenario simulation for rainfall, heat, and irrigation stress testing
- SHAP explainability for both global and local model interpretation
- Groq LLM explanation engine for scenario analysis and user-facing guidance
- Full-stack application with FastAPI backend and Streamlit frontend
- Versioned artifacts, model registry entries, model cards, and run manifests for reproducibility

## 2. System Overview

```text
User Input
    ->
Streamlit Frontend
    ->
FastAPI API Layer
    ->
Inference Service
    ->
Preprocessor + XGBoost Multi-Output Model
    ->
Prediction, Confidence, SHAP/Counterfactual Explanation
    ->
Scenario Simulation + Groq Explanation Modules
    ->
Structured Response for UI and Evaluation Artifacts
```

At runtime, the UI collects region, month, and field-level inputs. The API resolves localized historical context, validates and clips inputs, generates a ranked crop shortlist, and can then run what-if scenario simulations plus structured explanations.

## 3. Project Structure

```text
.
├── app.py
├── app_ui.py
├── run_pipeline.py
├── train_model.py
├── run_all_evaluations.py
├── run_all.bat
├── requirements.txt
├── .env.example
├── configs/
│   ├── data_new_config.json
│   └── training_data_new.json
├── data/
│   ├── raw/
│   │   ├── data_new/
│   │   └── demo/
│   └── processed/
│       ├── data_new_final_ml_dataset.csv
│       └── final_ml_dataset.csv
├── artifacts/
│   ├── data_new_training/
│   ├── demo_training/
│   ├── evaluation/
│   ├── feedback_store/
│   └── registry/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── MODEL_GOVERNANCE.md
│   └── TARGET_REDESIGN.md
├── logs/
├── src/
│   ├── app_api_entry.py
│   ├── ui_app_source.py
│   ├── env_loader.py
│   └── climate_pipeline/
│       ├── pipeline.py
│       ├── transforms.py
│       ├── training.py
│       ├── inference.py
│       ├── evaluation.py
│       ├── scenario_explainer.py
│       ├── llm_guide.py
│       ├── context_providers.py
│       ├── feedback.py
│       └── experiment_tracking.py
└── tests/
```

Major directories:

- `configs/`: declarative pipeline and training configuration
- `data/`: raw source tables and processed ML-ready datasets
- `artifacts/`: trained models, registries, evaluation outputs, and feedback storage
- `src/climate_pipeline/`: core ML, inference, evaluation, and explanation code
- `src/app_api_entry.py`: FastAPI application entrypoint
- `src/ui_app_source.py`: Streamlit frontend implementation
- `docs/`: supplementary architecture and governance notes
- `tests/`: unit and integration tests

Note: there is no separate `frontend/` directory in this repo. The frontend entrypoint is `app_ui.py`, which wraps `src/ui_app_source.py`.

## 4. Installation

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd our_take_on_climate_ai
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

Create a local environment file:

```bash
copy .env.example .env
```

Useful `.env` variables:

- `GROQ_API_KEY`: enables Groq-powered explanation features
- `GROQ_MODEL`: selects the Groq model used by explanation modules
- `MODEL_ARTIFACT_DIR`: overrides the default trained artifact directory
- `FEEDBACK_SIGNING_SECRET`: signs stored feedback records
- `API_BASE_URL`: points the frontend to a non-default API address

## 5. Running the Project

Run the backend:

```bash
uvicorn app:app --reload
```

Run the frontend:

```bash
streamlit run app_ui.py
```

Recommended Windows shortcut:

```bash
run_all.bat start
```

Other useful helper commands:

```bash
run_all.bat status
run_all.bat logs api
run_all.bat logs ui
run_all.bat doctor
run_all.bat stop
```

Important note:

- `app.py` is the FastAPI wrapper entrypoint
- `app_ui.py` is the Streamlit wrapper entrypoint

## 6. Running Evaluation

Run the full evaluation stack against the production-style training artifact:

```bash
python run_all_evaluations.py --artifact-dir artifacts/data_new_training
```

The evaluation runner resolves the saved model, reconstructs the held-out temporal split from artifact metadata, and writes outputs to:

```text
artifacts/evaluation/
├── performance/
├── stability/
├── scenario/
├── shap/
├── tables/
└── summaries/
```

Generated outputs include:

- `metrics.json` and `metrics.csv`
- performance and class-wise charts
- perturbation stability metrics
- scenario comparison outputs
- `scenario_explanations.json` and `scenario_explanations.md`
- SHAP summary, bar, and force plots
- `report_summary.md` and `final_metrics_table.csv`

## 7. Example Input / Output

Example `/predict` request:

```json
{
  "region": "Pune",
  "state": "Maharashtra",
  "target_time": "2024-07",
  "irrigation_index": 0.62,
  "rotation_score": 0.58,
  "features": {
    "temp_avg": 28.5,
    "rain_total": 215.0,
    "humidity_avg": 73.0,
    "max_temp": 33.5,
    "max_temp_3d": 34.0,
    "rain_lag_14": 96.0,
    "pH": 6.8,
    "N": 62.0,
    "P": 41.0,
    "K": 47.0,
    "N_class": "medium",
    "P_class": "medium",
    "K_class": "medium",
    "fertility_class": "medium",
    "state_context": "maharashtra",
    "target_month": "07",
    "target_season": "kharif"
  }
}
```

Example `/predict` response structure:

```json
{
  "status": "ok",
  "request_id": "8c1d...",
  "prediction_time_ms": 84.2,
  "recommendations": [
    {"crop": "sugarcane", "score": 0.62},
    {"crop": "coconut", "score": 0.19},
    {"crop": "banana", "score": 0.09}
  ],
  "confidence": 0.81,
  "confidence_breakdown": {
    "data_confidence": 0.92,
    "geo_confidence": 0.88,
    "rule_model_agreement": 0.71
  },
  "explanation": "Higher soil pH and irrigation support favored sugarcane over coconut.",
  "top_features": [
    {
      "feature": "soil pH",
      "feature_key": "pH",
      "impact": 0.0831,
      "direction": "supports",
      "descriptor": "high soil pH"
    }
  ],
  "why_not": [
    {
      "crop": "coconut",
      "score_gap": 0.043,
      "reason": "Coconut ranks below sugarcane under the current conditions."
    }
  ],
  "warnings": [],
  "model_version": "model_v1_..."
}
```

## 8. Results

Reference evaluation currently stored in `artifacts/evaluation/summaries/report_summary.md` and generated from the local artifact bundle on April 14, 2026 reports:

- Top-1 Accuracy: `0.828808`
- Top-3 Accuracy: `0.964178`
- Macro F1: `0.631731`
- NDCG: `0.950921`
- Top-1 perturbation consistency under `+-3%` noise: `0.934766`
- Top-3 perturbation consistency under `+-3%` noise: `0.815611`

Observed scenario behavior on the held-out split:

- `low_rainfall`: mean JS divergence `0.000634`, top-1 change rate `0.015837`
- `high_temperature`: mean JS divergence `0.000666`, top-1 change rate `0.006787`
- `increased_irrigation`: mean JS divergence `0.001577`, top-1 change rate `0.016214`

SHAP summary from the saved evaluation highlights `pH`, `K`, `state_context`, and `N` among the strongest contributors for the dominant explained crop in the reference run.

## 9. Scenario Explanation With Groq

The project includes two Groq-backed explanation paths:

- `src/climate_pipeline/llm_guide.py`: conversational, farmer-oriented explanation for predictions
- `src/climate_pipeline/scenario_explainer.py`: structured scenario impact analysis for baseline vs. modified rankings

Why this matters:

- raw ranking shifts are useful for engineers but hard to interpret for end users
- scenario explanations connect feature changes to crop movement using agronomic reasoning
- UI users can switch between analytical, farmer-facing, and metrics views for each scenario run

Structured scenario explanation format:

```json
{
  "scenario_summary": "...",
  "environmental_change": "...",
  "crop_response_analysis": "...",
  "ranking_changes": "...",
  "key_drivers": ["rainfall", "soil_ph"],
  "stability_assessment": "...",
  "confidence_note": "..."
}
```

Related API route:

```bash
POST /scenario-explain
```

## 10. Reproducibility

Key inputs and outputs:

- Raw source tables: `data/raw/data_new/`
- Processed dataset: `data/processed/data_new_final_ml_dataset.csv`
- Training config: `configs/training_data_new.json`
- Pipeline config: `configs/data_new_config.json`
- Primary trained artifact: `artifacts/data_new_training/`
- Registry files: `artifacts/registry/model_registry.jsonl` and `artifacts/registry/training_runs.jsonl`

To fully regenerate the project outputs:

```bash
python run_pipeline.py --config configs/data_new_config.json
python train_model.py --config configs/training_data_new.json
python run_all_evaluations.py --artifact-dir artifacts/data_new_training
```

To run the test suite:

```bash
python -m unittest tests.test_api
python -m unittest tests.test_inference
python -m unittest tests.test_training_pipeline
python -m unittest tests.test_pipeline_orchestration
```

Reproducibility note:

- use the same environment for training, inference, and evaluation
- this repo pins `scikit-learn`, `xgboost`, `shap`, and `matplotlib` in `requirements.txt`
- `run_all_evaluations.py` re-executes itself inside the repo-local `.venv` when available to reduce model/pickle version drift

## 11. Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn preprocessing and utilities
- XGBoost for multi-output ranking
- FastAPI for the serving layer
- Streamlit for the UI
- SHAP for explainability
- Groq for LLM-backed explanations
- Matplotlib for evaluation visualizations

## 12. Contributor Tips

- Keep configs in `configs/` as the source of truth for pipeline and training behavior
- Update tests when adding features, scenarios, or endpoint fields
- Prefer writing outputs to `artifacts/` instead of ad hoc local folders
- If you change model artifacts, rerun the evaluation pipeline and review the generated summaries

## 13. Future Improvements

- Integrate live weather features instead of relying only on historical same-month climatology
- Add stronger causal or counterfactual scenario modeling
- Expand multilingual farmer-facing guidance
- Add monitoring for production drift and model promotion gates
- Support additional ranking models and ensemble strategies

## 14. License / Credits

License:

- This repository does not currently include a committed `LICENSE` file. Add one before public redistribution.

Credits:

- Built with XGBoost, FastAPI, Streamlit, SHAP, and Groq
- Includes local architecture and governance support through `docs/ARCHITECTURE.md`, `docs/MODEL_GOVERNANCE.md`, and `docs/TARGET_REDESIGN.md`
- Uses versioned artifacts, model cards, and registry files to support reproducible ML experimentation
