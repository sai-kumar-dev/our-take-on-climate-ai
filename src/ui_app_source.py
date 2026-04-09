from __future__ import annotations

import html
import os
from datetime import datetime
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
PRIMARY_NUMERIC_FIELDS = [
    "temp_avg",
    "rain_total",
    "humidity_avg",
    "max_temp",
    "max_temp_3d",
    "rain_lag_14",
    "pH",
    "N",
    "P",
    "K",
    "irrigation_index",
    "rotation_score",
]
FIELD_LABELS = {
    "temp_avg": "Average temperature (C)",
    "rain_total": "Rainfall in current window (mm)",
    "humidity_avg": "Average humidity (%)",
    "max_temp": "Maximum temperature (C)",
    "max_temp_3d": "Recent hottest spell (C)",
    "rain_lag_14": "Recent rainfall, last 14 days (mm)",
    "pH": "Soil pH",
    "N": "Nitrogen level",
    "P": "Phosphorus level",
    "K": "Potassium level",
    "irrigation_index": "Irrigation support (0-1)",
    "rotation_score": "Rotation diversity score (0-1)",
}
FIELD_HELP = {
    "irrigation_index": "0 means rainfed only, 1 means reliable irrigation is available.",
    "rotation_score": "Higher means better crop rotation or field diversification history.",
}
PREFERRED_LANGUAGES = ["English", "Hindi", "Marathi", "Kannada", "Telugu", "Tamil"]


def api_url(path: str) -> str:
    return f"{API_BASE_URL}{path}"


def fetch_json(path: str, params: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.get(api_url(path), params=params, timeout=20)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def post_json(path: str, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.post(api_url(path), json=payload, timeout=30)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def pretty(value: str) -> str:
    if not value:
        return ""
    tokens = str(value).replace("_", " ").split()
    return " ".join(token.upper() if token in {"n", "p", "k"} else token.capitalize() for token in tokens)


def season_for(month_value: str) -> str:
    try:
        month_number = int(month_value)
    except (TypeError, ValueError):
        return "unknown"
    if month_number in {6, 7, 8, 9, 10}:
        return "kharif"
    if month_number in {11, 12, 1, 2, 3}:
        return "rabi"
    return "zaid"


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f7f1e6 0%, #f1eadb 100%);
        }
        [data-baseweb="input"] input,
        [data-baseweb="base-input"] input,
        [data-baseweb="select"] > div,
        textarea,
        .stNumberInput input {
            background: #fffaf2 !important;
            color: #243830 !important;
        }
        .stSelectbox label, .stNumberInput label, .stTextArea label, .stRadio label, .stSlider label {
            color: #243830 !important;
            font-weight: 600;
        }
        .result-card {
            background: #fffaf2;
            border: 1px solid #dfd2bc;
            border-radius: 16px;
            padding: 1rem 1.1rem;
        }
        .soft-note {
            background: #f3ebdb;
            border-left: 4px solid #1f5d46;
            padding: 0.9rem 1rem;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def numeric_default(feature: str, context: dict[str, Any], catalog: dict[str, Any]) -> float:
    feature_defaults = context.get("feature_defaults", {})
    if feature in feature_defaults and feature_defaults[feature] not in (None, ""):
        return float(feature_defaults[feature])
    catalog_defaults = catalog.get("numeric_features", {}).get(feature, {})
    return float(catalog_defaults.get("default", 0.0) or 0.0)


def categorical_default(feature: str, context: dict[str, Any], catalog: dict[str, Any]) -> str:
    feature_defaults = context.get("feature_defaults", {})
    if feature in feature_defaults and feature_defaults[feature]:
        return str(feature_defaults[feature]).strip().lower()
    levels = catalog.get("categorical_features", {}).get(feature, {}).get("levels", [])
    if levels:
        return str(levels[0]).strip().lower()
    return "unknown"


def numeric_step(feature: str) -> float:
    if feature == "pH":
        return 0.1
    if feature in {"irrigation_index", "rotation_score"}:
        return 0.05
    return 1.0


def build_target_time(month_value: str) -> str:
    return f"{datetime.now().year}-{month_value}"


def render_prediction(prediction: dict[str, Any]) -> None:
    recommendations = prediction.get("recommendations", [])
    if not recommendations:
        st.warning("No recommendation was returned for this input.")
        return

    top_choice = recommendations[0]
    confidence = float(prediction.get("confidence", 0.0) or 0.0) * 100.0
    st.markdown(
        f"""
        <div class="result-card">
            <h3 style="margin:0 0 0.4rem 0; color:#1f5d46;">Top recommendation: {html.escape(pretty(top_choice.get("crop", "")))}</h3>
            <p style="margin:0; color:#243830;">Confidence: {confidence:.1f}%</p>
            <p style="margin:0.7rem 0 0 0; color:#243830;">{html.escape(prediction.get("farmer_message", prediction.get("explanation", "")))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    score_frame = pd.DataFrame(
        [
            {
                "Crop": pretty(item.get("crop", "")),
                "Suitability": round(float(item.get("score", 0.0)) * 100.0, 2),
            }
            for item in recommendations
        ]
    )
    st.subheader("Crop ranking")
    st.dataframe(score_frame, use_container_width=True, hide_index=True)

    top_features = prediction.get("top_features", [])
    if top_features:
        st.subheader("Why this crop is leading")
        for feature in top_features:
            direction = "supports" if float(feature.get("impact", 0.0)) >= 0 else "holds back"
            st.write(f"- {pretty(feature.get('feature_key', ''))}: {direction} the ranking.")

    why_not = prediction.get("why_not", [])
    if why_not:
        st.subheader("Why others ranked lower")
        for item in why_not:
            st.write(f"- {pretty(item.get('crop', ''))}: {item.get('reason', '')}")

    warnings = prediction.get("warnings", [])
    if warnings:
        st.subheader("Input checks")
        for item in warnings:
            st.warning(item)

    localized_context = prediction.get("localized_context", {})
    if localized_context:
        with st.expander("Localized context used for autofill and validation"):
            st.json(localized_context)


def render_simulation(simulation: dict[str, Any]) -> None:
    scenario_results = simulation.get("scenario_results", {})
    if not scenario_results:
        st.info("No scenario output available.")
        return

    st.subheader("Scenario comparison")
    for scenario_name, payload in scenario_results.items():
        display_name = payload.get("display_name", pretty(scenario_name))
        prediction = payload.get("prediction", {})
        top_choice = prediction.get("recommendations", [{}])[0]
        st.markdown(
            f"""
            <div class="result-card">
                <strong style="color:#1f5d46;">{html.escape(display_name)}</strong><br>
                <span style="color:#243830;">Top crop: {html.escape(pretty(top_choice.get("crop", "")))}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        comparison = payload.get("comparison", {})
        rows = comparison.get("rows", [])[:5]
        if rows:
            table = pd.DataFrame(
                [
                    {
                        "Crop": pretty(row.get("crop", "")),
                        "Base score": round(float(row.get("base_score", 0.0)) * 100.0, 2),
                        "Scenario score": round(float(row.get("scenario_score", 0.0)) * 100.0, 2),
                        "Rank change": row.get("rank_change", 0),
                    }
                    for row in rows
                ]
            )
            st.dataframe(table, use_container_width=True, hide_index=True)


def render_feedback_form(region: str, state: str, prediction: dict[str, Any], payload: dict[str, Any]) -> None:
    recommendations = prediction.get("recommendations", [])
    default_crop = pretty(recommendations[0].get("crop", "")) if recommendations else ""
    with st.form("feedback_form", clear_on_submit=False):
        st.subheader("Farmer feedback")
        preferred_language = st.selectbox("Preferred language", PREFERRED_LANGUAGES, index=0)
        selected_crop = st.text_input("Recommended crop", value=default_crop)
        actual_crop = st.text_input("Actual crop used or planned")
        outcome_label = st.selectbox("Was this useful?", ["useful", "partly_useful", "not_useful"], index=0)
        helpfulness_rating = st.slider("Helpfulness", min_value=1, max_value=5, value=4)
        clarity_rating = st.slider("Clarity", min_value=1, max_value=5, value=4)
        consent_for_training = st.checkbox("Allow this feedback for future human-reviewed model improvement")
        comment = st.text_area("Farmer notes", placeholder="What felt useful, missing, or unclear?")
        submitted = st.form_submit_button("Submit feedback")

    if submitted:
        feedback_payload = {
            "request_id": prediction.get("request_id"),
            "region": region,
            "state": state,
            "preferred_language": preferred_language,
            "selected_crop": selected_crop,
            "actual_crop": actual_crop,
            "outcome_label": outcome_label,
            "helpfulness_rating": helpfulness_rating,
            "clarity_rating": clarity_rating,
            "consent_for_training": consent_for_training,
            "comment": comment,
            "input_snapshot": payload,
            "prediction_snapshot": {
                "recommendations": prediction.get("recommendations", []),
                "confidence": prediction.get("confidence"),
            },
        }
        response, error = post_json("/feedback", feedback_payload)
        if error:
            st.error(f"Feedback could not be saved: {error}")
        else:
            st.success(
                f"Feedback stored with review status: {response.get('review_status', 'unknown')}."
            )


def main() -> None:
    st.set_page_config(
        page_title="Climate Crop Advisory",
        page_icon=":seedling:",
        layout="wide",
    )
    apply_styles()

    st.title("Climate Crop Advisory")
    st.markdown(
        '<div class="soft-note">We use district-season context from the trained model to prefill inputs, '
        "flag unlikely values, and keep the form simple for first-time users.</div>",
        unsafe_allow_html=True,
    )

    health, health_error = fetch_json("/health")
    catalog, catalog_error = fetch_json("/catalog")
    sanity, _ = fetch_json("/sanity")
    if health_error:
        st.error(f"API health check failed: {health_error}")
        st.stop()
    if catalog_error or not catalog:
        st.error(f"Catalog load failed: {catalog_error or 'No response'}")
        st.stop()

    left, right = st.columns([1.4, 1.0])
    with right:
        st.caption(f"API: {API_BASE_URL}")
        st.caption(f"Model version: {health.get('model_version', 'unknown')}")
        st.caption(f"Mode: {health.get('mode', 'unknown')}")
        if sanity and sanity.get("sanity_checks", {}).get("available"):
            st.caption("Sanity checks: available")

    states = catalog.get("states", [])
    state_names = [item.get("state", "") for item in states if item.get("state")]
    default_state = "Maharashtra" if "Maharashtra" in state_names else (state_names[0] if state_names else "")
    month_options = catalog.get("temporal_context", {}).get("target_months", [])
    if not month_options:
        month_options = [
            {"value": f"{month:02d}", "label": datetime(2000, month, 1).strftime("%B")}
            for month in range(1, 13)
        ]

    with left:
        st.subheader("Field setup")
        selected_state = st.selectbox(
            "State",
            options=state_names or [""],
            index=state_names.index(default_state) if default_state in state_names else 0,
        )
        available_regions = next(
            (item.get("regions", []) for item in states if item.get("state") == selected_state),
            [],
        )
        default_region = "Pune" if "Pune" in available_regions else (available_regions[0] if available_regions else "")
        selected_region = st.selectbox(
            "District or region",
            options=available_regions or [""],
            index=available_regions.index(default_region) if default_region in available_regions else 0,
        )
        month_label_map = {
            f"{item['label']} ({pretty(item.get('season', season_for(item['value'])) )})": item["value"]
            for item in month_options
        }
        current_month = datetime.now().strftime("%m")
        default_month_label = next(
            (label for label, value in month_label_map.items() if value == current_month),
            next(iter(month_label_map.keys())),
        )
        selected_month_label = st.selectbox(
            "Target month",
            options=list(month_label_map.keys()),
            index=list(month_label_map.keys()).index(default_month_label),
        )
        target_month = month_label_map[selected_month_label]
        target_time = build_target_time(target_month)

    context, context_error = fetch_json(
        "/context",
        params={"state": selected_state, "region": selected_region, "target_time": target_time},
    )
    if context_error or not context:
        st.warning(
            f"Localized context could not be loaded, so the form is using model defaults. {context_error or ''}"
        )
        context = {
            "feature_defaults": {},
            "validation_bands": {},
            "crop_prior": [],
            "target_month": target_month,
            "target_season": season_for(target_month),
            "resolved_region": selected_region,
            "resolved_state": selected_state,
        }

    crop_prior = context.get("crop_prior", [])
    if crop_prior:
        st.subheader("Local seasonal history")
        st.write(
            ", ".join(
                f"{pretty(item.get('crop', ''))} ({round(float(item.get('score', 0.0)) * 100.0, 1)}%)"
                for item in crop_prior[:5]
            )
        )

    st.subheader("Main inputs")
    input_columns = st.columns(2)
    numeric_values: dict[str, float] = {}
    for index, feature in enumerate(PRIMARY_NUMERIC_FIELDS):
        with input_columns[index % 2]:
            numeric_values[feature] = st.number_input(
                FIELD_LABELS.get(feature, pretty(feature)),
                value=numeric_default(feature, context, catalog),
                step=numeric_step(feature),
                help=FIELD_HELP.get(feature),
            )
            band = context.get("validation_bands", {}).get(feature)
            if band:
                st.caption(f"Typical local range: {band.get('typical_min')} to {band.get('typical_max')}")

    st.subheader("Soil classes")
    class_columns = st.columns(4)
    categorical_values: dict[str, str] = {}
    for index, feature in enumerate(["N_class", "P_class", "K_class", "fertility_class"]):
        with class_columns[index]:
            levels = [
                str(item).strip().lower()
                for item in catalog.get("categorical_features", {}).get(feature, {}).get("levels", [])
            ]
            if not levels:
                levels = ["low", "medium", "high"]
            default_value = categorical_default(feature, context, catalog)
            if default_value not in levels:
                levels = [default_value, *levels]
            categorical_values[feature] = st.selectbox(
                pretty(feature),
                options=levels,
                index=levels.index(default_value),
            )

    available_scenarios = catalog.get("available_scenarios", [])
    scenario_options = {
        item.get("display_name", pretty(item.get("name", ""))): item.get("name")
        for item in available_scenarios
    }
    selected_scenarios = st.multiselect(
        "Demo scenarios",
        options=list(scenario_options.keys()),
        default=list(scenario_options.keys())[:2],
    )

    payload = {
        "region": selected_region,
        "state": selected_state,
        "target_time": target_time,
        "irrigation_index": numeric_values["irrigation_index"],
        "rotation_score": numeric_values["rotation_score"],
        "features": {
            **context.get("feature_defaults", {}),
            **numeric_values,
            **categorical_values,
            "state_context": str(selected_state).strip().lower(),
            "target_month": target_month,
            "target_season": season_for(target_month),
        },
    }

    predict_col, simulate_col = st.columns(2)
    if predict_col.button("Get recommendation", use_container_width=True):
        prediction, error = post_json("/predict", payload)
        if error:
            st.error(f"Prediction failed: {error}")
        else:
            st.session_state["last_prediction"] = prediction
            st.session_state["last_payload"] = payload

    if simulate_col.button("Run demo scenarios", use_container_width=True):
        simulation_payload = {
            **payload,
            "scenario_names": [scenario_options[item] for item in selected_scenarios],
        }
        simulation, error = post_json("/simulate", simulation_payload)
        if error:
            st.error(f"Scenario simulation failed: {error}")
        else:
            st.session_state["last_simulation"] = simulation
            st.session_state["last_payload"] = payload

    prediction = st.session_state.get("last_prediction")
    if prediction:
        render_prediction(prediction)
        render_feedback_form(
            selected_region,
            selected_state,
            prediction,
            st.session_state.get("last_payload", payload),
        )

    simulation = st.session_state.get("last_simulation")
    if simulation:
        render_simulation(simulation)


if __name__ == "__main__":
    main()
