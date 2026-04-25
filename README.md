# Climate-Aware Crop Recommendation System

[![Python](https://img.shields.io/badge/Python-ML%20Pipeline-3776AB?logo=python&logoColor=white)](requirements.txt)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ranking%20Model-E76F51)](requirements.txt)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](src/app_api_entry.py)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)](src/ui_app_source.py)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-6C5CE7)](src/climate_pipeline/evaluation.py)
[![Groq](https://img.shields.io/badge/Groq-LLM%20Explanations-111111)](src/climate_pipeline/scenario_explainer.py)

This repository contains a climate-aware crop recommendation platform for district-level ranking under climate and soil variability.

The public release `v1.0.0-public-artifact` ships:

- application code
- training and evaluation support files
- a representative reproducibility subset under `data/`
- release fixity files: `MANIFEST.yaml`, `dataset_version.json`, and `CHECKSUMS.txt`

The public release does not ship the full raw upstream source bundle or the full 24,120-row processed training table. Instead, it provides a demo artifact plus a documented reconstruction pathway in [data/RECONSTRUCTION_GUIDE.md](data/RECONSTRUCTION_GUIDE.md).

For a fast repository map, see [docs/REPO_GUIDE.md](docs/REPO_GUIDE.md). For a deeper engineering walkthrough, see [PROJECT_GUIDE.md](PROJECT_GUIDE.md).

## Start Here

1. Read the project overview in [System Overview](#2-system-overview) and the engineering walkthrough in [PROJECT_GUIDE.md](PROJECT_GUIDE.md).
2. Run the minimal local setup in [Quickstart](#quickstart) to launch the API and UI.
3. Explore the public artifact in [data/](data/), starting with [data/README.md](data/README.md), [data/DATA_CARD.md](data/DATA_CARD.md), and [data/sample_dataset.csv](data/sample_dataset.csv).
4. Review reproducibility boundaries in [data/RECONSTRUCTION_GUIDE.md](data/RECONSTRUCTION_GUIDE.md) and [data/SOURCE_PROVENANCE.md](data/SOURCE_PROVENANCE.md).
5. Verify release-facing metadata in [CITATION.cff](CITATION.cff), [MANIFEST.yaml](MANIFEST.yaml), [dataset_version.json](dataset_version.json), and [CHECKSUMS.txt](CHECKSUMS.txt).
6. Generate or inspect local evaluation outputs under `artifacts/evaluation/` using `python run_all_evaluations.py --artifact-dir artifacts/data_new_training`.

## Quickstart

Minimal local setup:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
streamlit run app_ui.py
```

Windows shortcut:

```bash
run_all.bat start
```

## Repository Navigation

- Source code: [src/](src)
- Public data artifact: [data/](data) and [data/README.md](data/README.md)
- Reconstruction docs: [data/RECONSTRUCTION_GUIDE.md](data/RECONSTRUCTION_GUIDE.md)
- Provenance and citation guidance: [data/SOURCE_PROVENANCE.md](data/SOURCE_PROVENANCE.md) and [data/DATASET_CITATION.txt](data/DATASET_CITATION.txt)
- Architecture and governance notes: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/MODEL_GOVERNANCE.md](docs/MODEL_GOVERNANCE.md), and [docs/TARGET_REDESIGN.md](docs/TARGET_REDESIGN.md)
- Engineering walkthrough: [PROJECT_GUIDE.md](PROJECT_GUIDE.md)
- Tests: [tests/](tests)
- Release metadata: [CITATION.cff](CITATION.cff), [MANIFEST.yaml](MANIFEST.yaml), [dataset_version.json](dataset_version.json), and [CHECKSUMS.txt](CHECKSUMS.txt)
- Evaluation outputs: local `artifacts/evaluation/` after running `python run_all_evaluations.py --artifact-dir artifacts/data_new_training`
- Repository map: [docs/REPO_GUIDE.md](docs/REPO_GUIDE.md)

## Table Of Contents

- [Start Here](#start-here)
- [Quickstart](#quickstart)
- [Repository Navigation](#repository-navigation)
- [Features](#1-features)
- [System Overview](#2-system-overview)
- [Project Structure](#3-project-structure)
- [Installation](#4-installation)
- [Running The Project](#5-running-the-project)
- [Evaluation](#6-evaluation)
- [Public Data Artifact](#7-public-data-artifact)
- [Reproducibility](#8-reproducibility)
- [Example API Payload](#9-example-api-payload)
- [License And Citation](#10-license-and-citation)

## 1. Features

- Multi-source integration across climate, soil, and crop-distribution inputs
- District-level ranking instead of a single hard crop label
- Scenario simulation for rainfall, heat, and irrigation stress testing
- SHAP-based model explanation support
- Groq-backed structured explanation modules
- FastAPI backend and Streamlit frontend
- Release-hardened public artifact package with provenance, citation, and fixity metadata

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

At runtime, the UI collects region, month, and field-level inputs. The API resolves localized historical context, validates and clips inputs, generates a ranked crop shortlist, and can run what-if scenario simulations plus structured explanations.

## 3. Project Structure

```text
.
|-- app.py
|-- app_ui.py
|-- run_pipeline.py
|-- train_model.py
|-- run_all_evaluations.py
|-- run_all.bat
|-- requirements.txt
|-- CHECKSUMS.txt
|-- MANIFEST.yaml
|-- dataset_version.json
|-- CITATION.cff
|-- NOTICE
|-- data/
|   |-- sample_dataset.csv
|   |-- data_dictionary.csv
|   |-- sample_dataset_schema.json
|   |-- DATA_CARD.md
|   |-- SOURCE_PROVENANCE.md
|   |-- RECONSTRUCTION_GUIDE.md
|   |-- NOTICE_DATA_SUBSET.md
|   |-- DATASET_CITATION.txt
|   `-- sample_inputs/
|-- artifacts/
|-- configs/
|-- docs/
|-- scripts/
|-- src/
`-- tests/
```

Public release boundary:

- `data/` is the shipped dataset artifact for this release.
- The subset is a representative reproducibility subset and demo artifact.
- The subset alone is not the full processed training table and should not be described as complete empirical reproducibility.
- Local `data/processed/` outputs remain build products and are outside the public release boundary.
- `artifacts/` is a generated local output directory used for training and evaluation runs; it is not versioned in git by default.

Structure note:

- Root-level entrypoints such as `app.py`, `app_ui.py`, `run_pipeline.py`, `train_model.py`, and `run_all_evaluations.py` are intentionally kept at the repository root for discoverable CLI use.
- Reviewer-facing tracked materials are concentrated in `README.md`, `data/`, `docs/`, and the root release metadata files.

## 4. Installation

Create a virtual environment in a local checkout:

```bash
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

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional local environment file:

```bash
copy .env.example .env
```

Useful `.env` variables:

- `GROQ_API_KEY`
- `GROQ_MODEL`
- `MODEL_ARTIFACT_DIR`
- `FEEDBACK_SIGNING_SECRET`
- `API_BASE_URL`

## 5. Running The Project

Backend:

```bash
uvicorn app:app --reload
```

Frontend:

```bash
streamlit run app_ui.py
```

Windows helper:

```bash
run_all.bat start
```

Other helpers:

```bash
run_all.bat status
run_all.bat logs api
run_all.bat logs ui
run_all.bat doctor
run_all.bat stop
```

## 6. Evaluation

Reference evaluation for the production-style artifact bundle:

```bash
python run_all_evaluations.py --artifact-dir artifacts/data_new_training
```

Outputs are written under local `artifacts/evaluation/`, including metrics tables, scenario analyses, SHAP plots, and summary reports.

Reference metrics currently stored in `artifacts/evaluation/summaries/report_summary.md` report:

- Top-1 Accuracy: `0.828808`
- Top-3 Accuracy: `0.964178`
- Macro F1: `0.631731`
- NDCG: `0.950921`

## 7. Public Data Artifact

The shipped dataset is `data/sample_dataset.csv`. It is a representative reproducibility subset of the final ML-ready table and is intended for:

- artifact review
- schema inspection
- demo requests
- tests and examples
- fixity verification

It is not intended to imply that the full raw data or full processed training data ships in the repository.

### Coverage Summary

| Metric | Released subset | Full final table | Meaning |
| --- | ---: | ---: | --- |
| States | `26` | `26` | All states in the full table are represented. |
| State-season groups | `78` | `78` | All `state x season` groups are represented. |
| Regions | `133` | `480` | Region coverage is intentionally partial. |
| Months covered | `44` | `54` | Month coverage is intentionally partial. |
| Dominant crops | `23` | `23` | All crops that are top-ranked anywhere in the full table appear as dominant at least once. |
| Target columns positive at least once | `41` | `41` | Every `crop_prob_*` target has at least one positive exemplar. |

### Coverage Interpretation

- `41/41` target-column coverage means every target column is positive at least once. It does not mean every crop is dominant somewhere.
- `23/23` dominant-crop coverage is the stricter argmax-style coverage metric and is separate from target-column positivity.
- `133/480` region coverage and `44/54` month coverage are explicitly partial and should be described that way.
- Quality variation in the released subset is limited: `soil_imputed` includes both `0` and `1`, `data_confidence` ranges from `0.9375` to `1.0`, while `climate_gap_filled` and `time_step_missing` are only `0` in the public subset.

### Scope Decision

No extra anchor rows were added for `v1.0.0-public-artifact`. The current 211-row subset already covers all states, all state-season groups, all dominant crops, and all target columns while remaining clearly bounded as a partial demo artifact.

## 8. Reproducibility

Reproducibility is split into two layers.

Public artifact reproducibility:

- verify file integrity with `CHECKSUMS.txt`
- inspect release metadata in `MANIFEST.yaml` and `dataset_version.json`
- use `data/sample_dataset.csv`, `data/sample_dataset_schema.json`, and `data/sample_inputs/` for schema-faithful demos and tests

Documented reconstruction pathway:

- [data/RECONSTRUCTION_GUIDE.md](data/RECONSTRUCTION_GUIDE.md) explains how to acquire upstream sources and rebuild the full final-table interface locally
- the public subset alone does not recreate the full 24,120-row training table or all empirical metrics
- if the prepared upstream interface tables are available locally under `data/raw/data_new/`, run:

```bash
python run_pipeline.py --config configs/data_new_config.json
python train_model.py --config configs/training_data_new.json
python run_all_evaluations.py --artifact-dir artifacts/data_new_training
```

Test suite:

```bash
python -m unittest tests.test_api
python -m unittest tests.test_inference
python -m unittest tests.test_training_pipeline
python -m unittest tests.test_pipeline_orchestration
```

## 9. Example API Payload

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

## 10. License And Citation

Code and original repository-authored documentation are released under [LICENSE](LICENSE) with the Apache-2.0 terms.

Repository notice file: [NOTICE](NOTICE)

The data subset under `data/` has a separate provenance boundary:

- read [data/NOTICE_DATA_SUBSET.md](data/NOTICE_DATA_SUBSET.md)
- read [data/SOURCE_PROVENANCE.md](data/SOURCE_PROVENANCE.md)
- read [data/DATASET_CITATION.txt](data/DATASET_CITATION.txt)

The Apache-2.0 code license does not grant broader rights in third-party upstream source data. If you use this repository, cite the project via [CITATION.cff](CITATION.cff) and cite upstream data sources separately.
