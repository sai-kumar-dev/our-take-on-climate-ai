# Climate-Aware Crop Recommendation System

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-ML%20Pipeline-3776AB?logo=python&logoColor=white)](requirements.txt)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ranking%20Model-E76F51)](requirements.txt)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](src/app_api_entry.py)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)](src/ui_app_source.py)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-6C5CE7)](src/climate_pipeline/evaluation.py)

---

## Climate-Aware Crop Recommendation Using Machine Learning, Explainability, and Scenario Simulation

An end-to-end climate-aware crop recommendation platform combining:

- District-level climate + soil + agricultural context  
- Machine learning ranking models for crop suitability  
- Scenario simulation under climate stress conditions  
- SHAP-based interpretability  
- FastAPI + Streamlit deployment stack  
- Reproducibility-ready public research artifact release

This repository accompanies the public artifact release:

**Release:** `v1.0.0-public-artifact`

---

# Start Here

If you are new to the repository:

### For Visitors / Recruiters
1. Read the Project Overview below  
2. See Results and Features  
3. Explore `docs/REPO_GUIDE.md`  
4. Review architecture under `docs/`

### For Users
1. Follow Quickstart  
2. Run API + UI  
3. Explore scenario simulations

### For Reviewers / Researchers
1. See `data/README.md`
2. Read `data/RECONSTRUCTION_GUIDE.md`
3. Review `data/SOURCE_PROVENANCE.md`
4. Use `CITATION.cff` for citation metadata

---

# Table of Contents

- Overview
- Features
- Architecture
- Repository Navigation
- Quickstart
- Running the System
- Evaluation
- Example API I/O
- Results
- Scenario Explanations
- Reproducibility
- Tech Stack
- Future Work
- License and Citation

---

# Project Overview

This project implements a climate-aware crop recommendation system that predicts **ranked crop suitability** rather than a single hard label.

It integrates:
- Climate variables
- Soil indicators
- Regional agricultural context
- Scenario perturbations
- Explainable ML outputs

The system supports:
- Crop recommendation
- Climate what-if analysis
- Interpretability
- Research reproducibility
- API + UI deployment

---

# Features

## Machine Learning
- Multi-output crop suitability ranking
- XGBoost-based ranking pipeline
- Feature engineering for climate and soil indicators
- Confidence-aware predictions

## Scenario Simulation
Supports:
- Low rainfall stress
- High temperature stress
- Irrigation shifts
- What-if crop ranking changes

## Explainability
- SHAP global explanations
- Local feature attribution
- Scenario explanation engine
- “Why this crop / why not that crop”

## Full Stack
- FastAPI backend
- Streamlit frontend
- Config-driven training pipeline
- Model artifacts + evaluation bundles

---

# System Overview

```text
User Input
   ↓
Streamlit Frontend
   ↓
FastAPI API Layer
   ↓
Inference Service
   ↓
Feature Processing + Ranking Model
   ↓
Crop Ranking + Confidence
   ↓
SHAP + Scenario Explanations
   ↓
Structured Recommendation Response
````

---

# Repository Navigation

## Core Source

```text
src/
└── climate_pipeline/
    ├── training.py
    ├── inference.py
    ├── evaluation.py
    ├── transforms.py
    ├── pipeline.py
    ├── scenario_explainer.py
```

## API / UI

```text
app.py
app_ui.py
src/app_api_entry.py
src/ui_app_source.py
```

## Configuration

```text
configs/
```

## Evaluation Artifacts

```text
artifacts/
```

## Public Data Artifact

```text
data/
```

Includes:

* sample dataset
* data card
* provenance documentation
* reconstruction guide

## Documentation

```text
docs/
├── REPO_GUIDE.md
├── ARCHITECTURE.md
├── MODEL_GOVERNANCE.md
```

## Release Metadata

```text
CITATION.cff
MANIFEST.yaml
CHECKSUMS.txt
dataset_version.json
```

---

# Quickstart

## Clone

```bash
git clone https://github.com/sai-kumar-dev/our-take-on-climate-ai.git
cd our-take-on-climate-ai
```

## Create Environment

```bash
python -m venv .venv
```

Activate:

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy environment config:

```bash
copy .env.example .env
```

---

# Running the System

## Start API

```bash
uvicorn app:app --reload
```

## Start UI

```bash
streamlit run app_ui.py
```

## Helper Runner

```bash
run_all.bat start
```

Other helpers:

```bash
run_all.bat status
run_all.bat doctor
run_all.bat stop
```

---

# Running Evaluation

```bash
python run_all_evaluations.py --artifact-dir artifacts/data_new_training
```

Generated outputs:

```text
artifacts/evaluation/
├── performance/
├── stability/
├── scenario/
├── shap/
├── summaries/
```

Includes:

* Metrics
* Robustness tests
* Scenario outputs
* SHAP analysis
* Evaluation summaries

---

# Example API Input

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
      "pH": 6.8,
      "N": 62.0,
      "P": 41.0,
      "K": 47.0,
      "target_month": "07",
      "target_season": "kharif"
  }
}
```

---

# Example Output

```json
{
 "recommendations":[
   {"crop":"sugarcane","score":0.62},
   {"crop":"coconut","score":0.19},
   {"crop":"banana","score":0.09}
 ],
 "confidence":0.81,
 "explanation":"Higher soil pH and irrigation support favored sugarcane."
}
```

---

# Reference Results

Example evaluation snapshot:

| Metric         | Value  |
| -------------- | ------ |
| Top-1 Accuracy | 0.8288 |
| Top-3 Accuracy | 0.9641 |
| Macro F1       | 0.6317 |
| NDCG           | 0.9509 |

Scenario stability:

* Low rainfall stress tested
* High temperature perturbation tested
* Increased irrigation scenario tested

SHAP highlights:

* pH
* K
* State context
* Nitrogen

---

# Scenario Explanations

Structured explanation engine supports:

```json
{
 "scenario_summary":"...",
 "ranking_changes":"...",
 "key_drivers":["rainfall","soil_ph"],
 "stability_assessment":"..."
}
```

API route:

```http
POST /scenario-explain
```

---

# Reproducibility

## Public Artifact Included

This repository includes a **Representative Reproducibility Subset** and release metadata.

See:

```text
data/README.md
data/RECONSTRUCTION_GUIDE.md
data/SOURCE_PROVENANCE.md
```

## Release Metadata

```text
CITATION.cff
CHECKSUMS.txt
MANIFEST.yaml
dataset_version.json
```

## Tests

```bash
python -m unittest tests.test_inference
python -m unittest tests.test_training_pipeline
```

---

# Project Structure

```text
.
├── app.py
├── app_ui.py
├── configs/
├── data/
├── docs/
├── artifacts/
├── scripts/
├── src/
└── tests/
```

See:

```text
docs/REPO_GUIDE.md
```

for full repository map.

---

# Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* FastAPI
* Streamlit
* SHAP
* Matplotlib

Optional explanation integration:

* Groq LLM APIs

---

# Contributor Notes

* Configs are source of truth
* Update tests when changing pipeline logic
* Write generated outputs to `artifacts/`
* Re-run evaluation when changing model artifacts

---

# Future Improvements

* Live weather integration
* Counterfactual reasoning
* Ensemble ranking models
* Drift monitoring
* Multilingual farmer guidance
* MLOps promotion gates

---

# Research Artifact Release

Public artifact release includes:

* Source code
* Public sample data subset
* Reproducibility package
* Citation metadata
* Checksums and manifests

Release tag:

```text
v1.0.0-public-artifact
```

---

# License

Source code in this repository is licensed under:

## Apache License 2.0

See:

```text
LICENSE
NOTICE
```

Data subset notices:

```text
data/NOTICE_DATA_SUBSET.md
data/SOURCE_PROVENANCE.md
```

Upstream data sources retain their respective licensing terms.

---

# Citation

If you use this repository in research:

Use:

```text
CITATION.cff
```

or GitHub’s **Cite this repository** button.

---

# Credits

Built using:

* XGBoost
* FastAPI
* Streamlit
* SHAP

Supporting documentation:

* Architecture notes
* Governance notes
* Reproducibility artifact package

---

## Repository Status

Research Artifact Release: Stable
Version: v1.0.0-public-artifact

Repository structure and navigation cleared for public release.
