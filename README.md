# Climate Crop Guidance

This repository contains a local FastAPI + Streamlit system for district-level crop guidance.

Current scope:
- ranks crops from district-season training patterns plus field inputs
- uses a provider-based context layer with historical district-month fallback today
- does not yet use live IMD weather data in production
- stores training lineage, model cards, and run manifests for local governance
- should still be treated as shortlist guidance, not guaranteed agronomic advice

## Architecture

Key layers:
- `run_pipeline.py`
  builds the processed dataset from configured raw sources
- `train_model.py`
  trains a versioned artifact and writes evaluation plus governance outputs
- `src/app_api_entry.py`
  serves the FastAPI endpoints
- `src/ui_app_source.py`
  serves the Streamlit UI
- `src/climate_pipeline/context_providers.py`
  contains the provider abstraction for historical and future live context sources

Reference docs:
- `docs/ARCHITECTURE.md`
- `docs/MODEL_GOVERNANCE.md`
- `docs/TARGET_REDESIGN.md`

## Local Requirements

Large raw datasets and model artifacts are intentionally kept local and are not committed to git.

Expected local assets for runtime:
- `artifacts/data_new_training/trained_model.pkl`
- `artifacts/data_new_training/feature_config.json`
- `artifacts/data_new_training/scaler.pkl`
- `data/processed/final_ml_dataset.csv`

Expected local assets for retraining:
- `data/processed/data_new_final_ml_dataset.csv`
- raw files referenced by `configs/data_new_config.json` if you want to rebuild the processed dataset

Install the direct Python dependencies from `requirements.txt` in the local virtual environment.

## Local Config

The app supports a project-level `.env` file.

Start by copying:

```powershell
Copy-Item .env.example .env
```

Useful keys:
- `GROQ_API_KEY`
  enables the Groq layman explanation flow in the UI
- `GROQ_MODEL`
  chooses the Groq model
- `MODEL_ARTIFACT_DIR`
  points the API at a specific trained artifact folder
- `FEEDBACK_SIGNING_SECRET`
  integrity-signs stored feedback
- `API_BASE_URL`
  overrides the UI target when not using the local default

## Run

From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
run_all.bat start
```

Useful commands:
- `run_all.bat status`
- `run_all.bat stop`
- `run_all.bat doctor`
- `run_all.bat logs api`
- `run_all.bat logs ui`

## Data Pipeline

Rebuild the processed dataset:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --config configs/data_new_config.json
```

Train a model:

```powershell
.\.venv\Scripts\python.exe train_model.py --config configs/training_data_new.json
```

Every training run now writes:
- `evaluation_report.json`
- `run_manifest.json`
- `model_card.md`
- registry entries under `artifacts/registry`

## Test

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_api
.\.venv\Scripts\python.exe -m unittest tests.test_inference
.\.venv\Scripts\python.exe -m unittest tests.test_training_pipeline
.\.venv\Scripts\python.exe -m unittest tests.test_pipeline_orchestration
```

## Containers And CI

Container files:
- `Dockerfile`
- `docker-compose.yml`

CI:
- `.github/workflows/ci.yml`

The compose stack expects local artifacts and datasets to exist in the mounted workspace.

## Repo Notes

- `src/project_doctor.py`
  checks whether the local environment has the files needed to run, retrain, and package
- `app.py`
  is the API wrapper entrypoint
- `app_ui.py`
  is the UI wrapper entrypoint
