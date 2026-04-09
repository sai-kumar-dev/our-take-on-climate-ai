# Climate Crop Guidance

This repository contains a local FastAPI + Streamlit prototype for district-level crop guidance.

Current scope:
- ranks crops from district-season training patterns plus field inputs
- uses historical same-month district context for autofill and validation
- does not yet use live IMD weather data
- should be treated as shortlist guidance, not guaranteed agronomic advice

## Local Requirements

Large raw datasets and model artifacts are intentionally kept local and are not committed to git.

The app expects these local assets to exist:
- `artifacts/data_new_training/trained_model.pkl`
- `artifacts/data_new_training/feature_config.json`
- `artifacts/data_new_training/scaler.pkl`
- `data/processed/final_ml_dataset.csv`
- `data/processed/data_new_final_ml_dataset.csv` for retraining

Install the direct Python dependencies from `requirements.txt` in the local virtual environment.

## Local Config

The app now supports a project-level `.env` file.

Start by copying:

```powershell
Copy-Item .env.example .env
```

Useful keys:
- `GROQ_API_KEY` to enable the Groq layman explanation flow in the UI
- `GROQ_MODEL` to choose the Groq model
- `MODEL_ARTIFACT_DIR` to point at a specific trained artifact folder
- `FEEDBACK_SIGNING_SECRET` to integrity-sign stored feedback
- `API_BASE_URL` only if you are running the UI against a non-default API address

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

## Test

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_api
.\.venv\Scripts\python.exe -m unittest tests.test_inference
.\.venv\Scripts\python.exe -m unittest tests.test_training_pipeline
```

## Repo Notes

- `src/app_api_entry.py` contains the API entrypoint.
- `src/ui_app_source.py` contains the Streamlit UI.
- `run_all.bat` manages the local API/UI stack.
- `src/project_doctor.py` checks whether the local environment has the files needed to run and retrain.
