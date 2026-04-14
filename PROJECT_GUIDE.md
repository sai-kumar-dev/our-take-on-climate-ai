# Project Guide

## 1. System Architecture

This repository is an end-to-end machine learning application for district-level crop recommendation under climate and soil variability. It is organized as a layered system rather than a single notebook or script.

High-level architecture:

```text
Raw CSV Data
    ->
Config-Driven Data Pipeline
    ->
Processed ML Dataset
    ->
Training + Calibration + Governance Artifacts
    ->
FastAPI Inference Service
    ->
Streamlit Frontend
    ->
Scenario Simulation + SHAP + Groq Explanations
```

Core layers:

### Data Layer

- raw inputs live under `data/raw/`
- pipeline configs define source paths and schema mapping
- processed ML-ready outputs are written to `data/processed/`

### Model Layer

- preprocessing and training logic live in `src/climate_pipeline/training.py`
- the production-style artifact bundle is stored in `artifacts/data_new_training/`
- evaluation, registry, and governance metadata are written alongside the trained model

### Serving Layer

- FastAPI app entrypoint: `src/app_api_entry.py`
- main inference engine: `src/climate_pipeline/inference.py`
- prediction, scenario simulation, feedback capture, catalog, context lookup, and explanation endpoints all route through the same service object

### Frontend Layer

- Streamlit entrypoint: `app_ui.py`
- UI implementation: `src/ui_app_source.py`
- the UI is designed for both technical and beginner users, with tabs, scenario cards, farmer-facing explanation views, and feedback capture

### Explainability Layer

- local model explanation uses SHAP when available
- fallback explanation logic uses counterfactual feature perturbation
- scenario analysis uses a dedicated Groq-backed structured explanation module in `src/climate_pipeline/scenario_explainer.py`
- conversational farmer guidance uses `src/climate_pipeline/llm_guide.py`

## 2. Data Pipeline

The data pipeline is configuration-driven and starts from three source families:

- climate time series
- soil measurements
- crop production or crop area history

The default production configuration is `configs/data_new_config.json`.

### 2.1 Source Files

Configured source tables:

- `data/raw/data_new/climate_daily.csv`
- `data/raw/data_new/soil_samples.csv`
- `data/raw/data_new/crop_monthly.csv`

The config also defines the column mapping for each source, which allows the pipeline to remain stable even if source column names differ from the internal canonical names.

### 2.2 Pipeline Entry Point

Run:

```bash
python run_pipeline.py --config configs/data_new_config.json
```

This calls `run_pipeline_from_path()` in `src/climate_pipeline/pipeline.py`.

### 2.3 Climate Feature Engineering

Implemented in `prepare_climate_features()` inside `src/climate_pipeline/transforms.py`.

Key operations:

- canonicalize region and state metadata
- parse time information to monthly or seasonal keys
- fill raw climate gaps
- aggregate daily records to district-time summaries
- derive climate statistics per district and time step

Generated climate features include:

- `temp_avg`
- `rain_total`
- `humidity_avg`
- `rain_variance`
- `max_temp`
- `max_temp_3d`
- `max_rain_1d`
- `dry_spell_days`
- `temp_lag_7`
- `rain_lag_14`
- `climate_gap_filled`
- `time_step_missing`

### 2.4 Soil Feature Engineering

Implemented in `prepare_soil_features()`.

Key operations:

- map raw soil columns to canonical `pH`, `N`, `P`, and `K`
- aggregate soil observations at district level, optionally with time alignment when available
- impute missing numeric soil values
- derive soil quality indicators
- assign categorical nutrient bands

Generated soil features include:

- `pH`
- `N`
- `P`
- `K`
- `N_class`
- `P_class`
- `K_class`
- `soil_health_index`
- `fertility_class`
- `soil_imputed`

### 2.5 Crop Label Construction

Implemented in `prepare_crop_labels()`.

Key operations:

- normalize crop identifiers
- aggregate either crop area or production weights
- compute within-district time-step crop share
- pivot to one column per crop using the prefix `crop_prob_`

This yields a target matrix such as:

```text
crop_prob_sugarcane
crop_prob_coconut
crop_prob_banana
...
```

Each row is therefore a probability distribution over crops, not a single categorical label.

### 2.6 Merge and Validation

Implemented in `merge_datasets()` and `validate_final_dataset()`.

The final merge combines:

- climate features
- soil features
- crop probability targets
- geographic metadata
- temporal metadata
- confidence and completeness markers

Outputs written by the pipeline:

- `data/processed/data_new_final_ml_dataset.csv`
- `reports/data_new_inspection_report.json`
- `reports/data_new_validation_report.json`
- `reports/data_new_summary_stats.csv`

## 3. Model Design

### 3.1 Why XGBoost

The project uses XGBoost because it fits the operating constraints of tabular agricultural data well:

- handles non-linear interactions between weather, soil, and context
- performs strongly on structured tabular data
- supports robust feature importance and SHAP analysis
- works well with modest-sized but heterogeneous datasets

### 3.2 Why a Ranking Formulation

This system does not treat crop recommendation as a single-label classification task. Instead, it predicts a distribution over crops and ranks the crops by predicted suitability.

That design is useful because:

- multiple crops may be plausible for the same district-month profile
- the UI can show shortlists rather than one rigid answer
- scenario simulation can compare ranking movement, not only class flips
- evaluation can include ranking-sensitive metrics such as NDCG and top-k accuracy

### 3.3 Preprocessing

Implemented through `FeaturePreprocessor` in `training.py`.

Preprocessing steps:

- numeric imputation with `SimpleImputer`
- categorical imputation with `SimpleImputer`
- numeric scaling using `StandardScaler` or `MinMaxScaler`
- one-hot encoding for categorical features via `OneHotEncoder`

Default categorical features include:

- `N_class`
- `P_class`
- `K_class`
- `fertility_class`
- `state_context`
- `region_context`
- `target_month`
- `target_season`

### 3.4 Training Strategy

The XGBoost backend is trained as a multi-output regression problem using:

- `XGBRegressor`
- wrapped by `MultiOutputRegressor`

The model predicts one value per crop label. Predictions are normalized into a probability distribution using `normalize_probability_matrix()`.

### 3.5 Calibration

Calibration is built into the training pipeline through a `ProbabilityCalibrator` abstraction.

Current calibration method:

- temperature scaling

Configured in:

- `configs/training_data_new.json`

This improves the reliability of the predicted distribution, especially for ranking confidence and downstream comparisons.

### 3.6 Outputs

The trained artifact bundle contains:

- `trained_model.pkl`
- `model_v*.pkl`
- `calibrator.pkl`
- `scaler.pkl`
- `feature_config.json`
- `evaluation_report.json`
- `model_card.md`
- `run_manifest.json`

## 4. Evaluation Pipeline

The evaluation pipeline is implemented in `src/climate_pipeline/evaluation.py` and invoked by `run_all_evaluations.py`.

Command:

```bash
python run_all_evaluations.py --artifact-dir artifacts/data_new_training
```

The runner automatically re-executes inside the repo-local `.venv` when available, which helps avoid model/pickle version mismatches.

### 4.1 Performance Evaluation

Implemented in `run_performance()`.

Metrics:

- Top-1 Accuracy
- Top-3 Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- NDCG

Generated artifacts:

- `artifacts/evaluation/performance/metrics.json`
- `artifacts/evaluation/performance/metrics.csv`
- `artifacts/evaluation/performance/performance_bar_chart.png`
- `artifacts/evaluation/performance/classwise_accuracy.png`

### 4.2 Stability Evaluation

Implemented in `run_stability()`.

Method:

- perturb numeric inputs with random `+-3%` noise
- compare baseline and perturbed predictions

Metrics:

- Top-1 consistency
- Top-3 consistency
- Jensen-Shannon divergence
- Mean probability delta

Generated artifacts:

- `artifacts/evaluation/stability/stability_metrics.csv`
- `artifacts/evaluation/stability/stability_summary.json`
- `artifacts/evaluation/stability/stability_histogram.png`

### 4.3 Scenario Evaluation

Implemented in `run_scenario()`.

Offline evaluation scenarios:

- `low_rainfall`
- `high_temperature`
- `increased_irrigation`

For each scenario, the evaluator:

- modifies the held-out test inputs
- reruns the model
- computes probability shifts
- computes divergence metrics
- builds a baseline vs. scenario ranking payload
- generates structured scenario explanations

Generated artifacts:

- `artifacts/evaluation/scenario/scenario_results.csv`
- `artifacts/evaluation/scenario/scenario_comparison.png`
- `artifacts/evaluation/scenario/scenario_explanations.json`
- `artifacts/evaluation/scenario/scenario_explanations.md`

### 4.4 SHAP Evaluation

Implemented in `run_shap()`.

Behavior:

- uses SHAP only when the backend is XGBoost
- selects a dominant label from the held-out sample
- computes SHAP values over transformed features
- generates global and local plots

Artifacts:

- `artifacts/evaluation/shap/shap_summary.png`
- `artifacts/evaluation/shap/shap_bar.png`
- `artifacts/evaluation/shap/shap_force.png`

### 4.5 Summary Outputs

Generated under:

- `artifacts/evaluation/summaries/report_summary.md`
- `artifacts/evaluation/summaries/final_metrics_table.csv`

These files are the canonical reference for the current local evaluation state.

## 5. Scenario Engine

The project contains two related scenario paths:

- online scenario simulation used by the API and Streamlit application
- offline scenario evaluation used to benchmark how predictions behave over a held-out test split

They share the same idea: start from a baseline input profile, modify a small number of environmental or management variables, rerun the model, and compare the resulting crop ranking.

### 5.1 Runtime Scenarios

Runtime scenario definitions are maintained in `src/climate_pipeline/inference.py` through `PRESET_SCENARIOS`.

Current user-facing scenarios include:

- `low_rainfall`
- `heatwave`
- `high_irrigation`

Each preset contains:

- a stable internal name
- a display name for the UI
- the feature-level modifiers applied to the request payload
- descriptive text used in outputs

### 5.2 How Scenario Inputs Are Built

The runtime flow is:

1. accept a baseline prediction payload
2. validate and normalize user features
3. clone the baseline numeric feature set
4. apply scenario-specific adjustments
5. rerun the same trained model and calibrator
6. compute baseline-vs-scenario comparison metrics
7. return both the modified rankings and explanatory metadata

The important design choice is that the same trained model is reused. The scenario engine does not train a separate model. This isolates the effect of the input change and makes scenario comparisons easier to interpret.

### 5.3 Scenario Modifiers

Examples of scenario logic:

- low rainfall reduces rainfall-linked variables such as `rain_total` or related moisture indicators
- heatwave increases temperature-linked variables such as `temp_avg`, `max_temp`, or `max_temp_3d`
- high irrigation increases `irrigation_index`

Because the model is trained on a multivariate feature space, even a small direct change can create indirect ranking movement through interactions with soil, state context, or seasonal fields.

### 5.4 Comparison Outputs

Scenario comparison is not limited to checking whether the top crop changes. The service computes richer differences such as:

- baseline ranking
- scenario ranking
- per-crop score deltas
- top-rank movement
- overlap between baseline and scenario top-k crops
- divergence between the full predicted distributions

This is important because many practical cases are stable at rank 1 but still show meaningful movement in ranks 2 and 3.

### 5.5 UI Explanation Modes

The frontend exposes scenario results in three modes:

- `Analysis`: structured technical interpretation
- `Farmer view`: plain-language guidance for non-technical users and farmer backgrounds
- `Metrics`: raw comparison numbers and rank shifts

That split allows the same scenario engine to support both debugging and decision support.

## 6. LLM Explanation Module

This repository now has two Groq-backed explanation modules serving different needs.

### 6.1 Interactive Guide Module

File:

- `src/climate_pipeline/llm_guide.py`

Purpose:

- answer user questions about a prediction in simple language
- support multiple languages
- convert raw model outputs into beginner-friendly guidance

Technical design:

- uses Groq's OpenAI-compatible Responses-style interface
- consumes the prediction payload returned by the inference service
- constrains tone to be practical, short, and cautious
- avoids promising yield or income

This module is best viewed as a guided explanation layer on top of a single prediction.

### 6.2 Scenario Explainer Module

File:

- `src/climate_pipeline/scenario_explainer.py`

Purpose:

- explain why the crop ranking changes between baseline and scenario conditions
- produce structured JSON suitable for storage, UI rendering, and report generation

Primary entry points:

- `generate_scenario_explanation(...)`
- `call_groq_llm(prompt: str) -> str`
- `format_for_ui(...)`
- `render_explanations_markdown(...)`

### 6.3 Prompt Design

The prompt design is intentionally restrictive.

System role:

- agricultural AI expert

Prompt goals:

- formal, report-ready tone
- analytical rather than conversational
- explicitly tied to supplied feature changes
- explicit comparison between baseline and scenario rankings
- cautious treatment of uncertainty

The prompt includes:

- baseline ranking and scores
- scenario ranking and scores
- explicit feature changes
- comparison metrics
- crop trait hints derived from known sensitivity groups
- instructions to avoid unsupported climate, soil, pest, or management claims

### 6.4 Structured Output Contract

The scenario explanation is normalized to the following schema:

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

This contract matters for three reasons:

- evaluation code can save explanations consistently
- the UI can render fixed sections without fragile text parsing
- downstream report generation can use the same structure

### 6.5 Grounding and Failure Handling

The module is designed to remain useful even if the upstream LLM is unavailable or returns weak output.

Safeguards include:

- local `.env` loading for `GROQ_API_KEY`
- JSON parsing and repair attempts
- schema validation and key normalization
- evidence-grounding checks against actual feature names and ranking movement
- disallowed phrase filtering for unsupported claims
- deterministic fallback explanation generation when the LLM is unavailable

This is an important production decision. Scenario explanation is treated as an explainability enhancement, not as a hard dependency for the core prediction pipeline.

### 6.6 Trait Highlighting

To make the explanation more decision-oriented, the explainer incorporates crop sensitivity groups such as:

- water-sensitive crops
- heat-resilient crops
- nutrient-dependent crops

These groupings are used as hints rather than as a substitute for model evidence. The explainer still checks whether the actual ranking and score deltas support the narrative.

## 7. API Design

The backend is implemented in `src/app_api_entry.py` using FastAPI.

### 7.1 Application Factory

`create_app()` builds the API with shared service objects and application state such as:

- inference service
- feedback store
- metrics tracker
- rate limiters

Using an application factory rather than a single global app object makes testing and controlled instantiation easier.

### 7.2 Main Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | `GET` | basic service liveness |
| `/sanity` | `GET` | lightweight self-check for model availability and startup state |
| `/metrics` | `GET` | internal API usage counters |
| `/catalog` | `GET` | available crops, states, feature schema, and scenario metadata |
| `/context` | `GET` | localized historical context and metadata |
| `/predict` | `POST` | baseline crop recommendation request |
| `/simulate` | `POST` | run preset scenarios on a baseline request |
| `/scenario-explain` | `POST` | generate structured explanation for a scenario result |
| `/feedback` | `POST` | submit human feedback for later review |
| `/llm-guide` | `POST` | ask a Groq-backed question about a prediction |

### 7.3 Prediction Response Shape

The prediction service returns more than ranked crops. It also includes:

- confidence and confidence breakdown
- explanation text
- top contributing features
- why-not analysis for nearby alternatives
- warnings and input-quality checks
- drift-style or context checks
- model version metadata

This richer payload lets the frontend avoid re-computing business logic and keeps explanation layers consistent with the original prediction.

### 7.4 Scenario API Design

`POST /simulate` accepts the baseline input and one or more scenario names. The service returns:

- baseline prediction
- one result block per scenario
- rank and score comparisons
- metadata used by the UI for explanation rendering

`POST /scenario-explain` is separated from `POST /simulate` so the UI can trigger explanation generation on demand rather than paying the LLM latency cost for every scenario automatically.

### 7.5 Feedback and Governance

The backend also supports:

- signed feedback records
- rate limiting for feedback and LLM routes
- storage metadata in the catalog/context layer
- API metrics that help with debugging and operational visibility

## 8. Frontend Design

The frontend is implemented with Streamlit in `src/ui_app_source.py` and exposed through the small wrapper `app_ui.py`.

### 8.1 Frontend Role

The UI is not only a form layer. It is also responsible for:

- collecting structured agronomic inputs
- surfacing catalog values and context hints
- displaying ranked recommendations and confidence
- running scenario simulation
- requesting scenario explanations on demand
- collecting user feedback

### 8.2 Input Flow

The user interaction sequence is roughly:

1. select region and state
2. choose target month or time context
3. enter climate and soil variables
4. submit the baseline prediction
5. inspect the ranked crops and explanations
6. trigger scenario simulation
7. optionally request detailed explanation or farmer-friendly guidance

This keeps the workflow aligned with how an end user typically reasons: first inspect the current recommendation, then ask what changes under stress or management interventions.

### 8.3 Output Visualization

The frontend presents multiple explanation layers:

- ranked crops with scores
- confidence summary
- feature-based explanation
- why-not comparison for alternatives
- scenario cards with rank movement
- on-demand scenario explanation
- farmer-facing guide text
- feedback form

This is a practical pattern because it prevents one oversized explanation block from trying to satisfy every type of user.

### 8.4 Scenario Experience

The scenario UX is intentionally progressive:

- the user first sees baseline-vs-scenario rank movement
- an expander exposes the explanation controls
- inside that block, the user can toggle between `Analysis`, `Farmer view`, and `Metrics`

This approach keeps the default UI lightweight while still supporting advanced interpretation.

### 8.5 Frontend Startup

Recommended command:

```bash
streamlit run app_ui.py
```

The actual UI implementation lives in `src/ui_app_source.py`, but `app_ui.py` is the stable wrapper entrypoint documented for local use.

## 9. Artifacts System

The repository uses an explicit artifact layout instead of mixing outputs into arbitrary folders. This is one of the strongest production-oriented parts of the project.

### 9.1 Training Artifacts

Primary artifact bundles are stored under directories such as:

- `artifacts/data_new_training/`
- `artifacts/demo_training/`

These contain serialized model components and metadata such as:

- trained model pickle
- calibrator
- feature config
- evaluation report
- model card
- run manifest

### 9.2 Registry

The registry lives in:

- `artifacts/registry/model_registry.jsonl`
- `artifacts/registry/training_runs.jsonl`

Registry records are produced by `record_training_run()` in `src/climate_pipeline/experiment_tracking.py`.

This registry layer is useful because it:

- keeps a lightweight history of trained runs
- records model version identifiers
- links artifacts to configs and governance metadata
- supports later auditability without introducing a full external experiment platform

### 9.3 Evaluation Outputs

All evaluation outputs are centralized under:

- `artifacts/evaluation/performance/`
- `artifacts/evaluation/stability/`
- `artifacts/evaluation/scenario/`
- `artifacts/evaluation/shap/`
- `artifacts/evaluation/tables/`
- `artifacts/evaluation/summaries/`

This separation makes it easy to publish report figures without mixing them with training-time artifacts.

### 9.4 Feedback Store

Human feedback records are stored under:

- `artifacts/feedback_store/`

That location keeps user feedback operationally separate from model artifacts and evaluation outputs.

### 9.5 Logs and Reports

Supporting outputs also exist under:

- `logs/`
- `reports/`

The rough division is:

- `reports/` for data validation and analysis reports
- `logs/` for application runtime support

## 10. How to Extend

This section describes the safest extension points for evolving the project.

### 10.1 Add a New Crop

To add a new crop target:

1. ensure the raw crop source includes the crop with enough coverage
2. update any crop normalization logic in the data transforms if names need mapping
3. rerun the pipeline so `prepare_crop_labels()` generates the new `crop_prob_*` column
4. retrain the model
5. rerun the evaluation pipeline
6. verify UI and API catalog outputs

Important note:

- if the new crop is rare, ranking metrics may look unstable even if the average performance remains strong

### 10.2 Add a New Feature

To add a new climate, soil, or context feature:

1. add the feature derivation in `src/climate_pipeline/transforms.py`
2. include it in the processed dataset
3. ensure training config and inference schema recognize it
4. update tests
5. retrain and reevaluate

Be careful to keep the feature available in both training and inference. A feature that only exists in offline data but not at request time will create serving problems.

### 10.3 Add a New Scenario

To add a new runtime scenario:

1. add a preset to `PRESET_SCENARIOS` in `src/climate_pipeline/inference.py`
2. define the feature adjustments clearly
3. verify that the scenario appears in the catalog and UI
4. check whether the scenario should also be added to offline evaluation in `src/climate_pipeline/evaluation.py`
5. confirm the explanation text remains meaningful under the new modifier

### 10.4 Retrain the Model

Use:

```bash
python run_pipeline.py --config configs/data_new_config.json
python train_model.py --config configs/training_data_new.json
python run_all_evaluations.py --artifact-dir artifacts/data_new_training
```

After retraining:

- inspect the new model card
- review evaluation summaries
- compare ranking behavior and stability against the previous artifact

### 10.5 Replace the Backend Model

If you want to replace XGBoost with another model family, preserve these interfaces:

- fit and predict over the same target matrix shape
- return one score per crop
- integrate with the same preprocessing path
- provide a stable model-loading contract for inference
- keep evaluation outputs comparable

The more faithfully the new backend preserves these interfaces, the less UI and API code needs to change.

## 11. Troubleshooting

### 11.1 Scikit-Learn or Pickle Version Mismatch

Symptom:

- warnings or failures when loading serialized artifacts

Cause:

- the local Python environment does not match the versions used when the artifact was created

Fix:

```bash
pip install -r requirements.txt
python run_all_evaluations.py --artifact-dir artifacts/data_new_training
```

Operational note:

- `run_all_evaluations.py` attempts to re-run itself inside the repo-local `.venv` to reduce this problem

### 11.2 Groq Explanations Not Working

Symptom:

- scenario explanations fall back to deterministic text
- the interactive guide reports that Groq is not configured

Check:

- `.env` exists
- `GROQ_API_KEY` is present
- `GROQ_MODEL` is valid if overridden

The scenario explainer loads `.env` locally, but missing credentials will still disable live LLM calls.

### 11.3 Missing Model Artifacts

Symptom:

- API startup or evaluation cannot find model files

Check:

- `artifacts/data_new_training/` exists
- `artifacts/registry/` exists
- `MODEL_ARTIFACT_DIR` is not pointing to a stale path

If artifacts are missing, retrain or point the environment variable to the correct bundle.

### 11.4 Frontend Command Confusion

Use:

```bash
streamlit run app_ui.py
```

Do not use `streamlit run app.py`. `app.py` is the FastAPI wrapper, not the Streamlit application.

### 11.5 API Startup Command Confusion

Use either:

```bash
uvicorn app:app --reload
```

or the direct source entrypoint:

```bash
uvicorn src.app_api_entry:app --reload
```

Both work in this repository because `app.py` re-exports the FastAPI app.

### 11.6 SHAP Errors

Possible causes:

- SHAP not installed
- model backend incompatible with the current SHAP path
- artifact missing transformed feature metadata

Fix:

- reinstall requirements
- use the XGBoost artifact bundle
- rerun evaluation after confirming the model files are complete

### 11.7 Frontend Cannot Reach Backend

Check:

- FastAPI is running on the expected host and port
- the frontend `API_BASE_URL` is correct
- Windows firewall or a stale local process is not blocking the port

If you use the helper script:

```bash
run_all.bat doctor
run_all.bat status
```

These commands are the fastest way to detect local runtime issues on Windows.

### 11.8 Path and Working Directory Issues

Many commands assume the repository root as the working directory. If imports or file paths fail:

- `cd` into the repository root first
- run wrapper scripts from the root
- avoid launching the app from nested directories

## Closing Note

This project is strongest when treated as a full system rather than as an isolated model file. The data pipeline, trained artifact bundle, evaluation suite, API contracts, UI flows, and LLM explanation layers all depend on shared assumptions. The safest engineering practice is therefore to make changes through the config and pipeline paths, then retrain, reevaluate, and verify the application end to end.
