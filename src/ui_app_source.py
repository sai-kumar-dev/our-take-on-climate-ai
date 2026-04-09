from __future__ import annotations

import html
import json
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
BEGINNER_STEPS = [
    {
        "title": "Pick your district",
        "body": "Choose the state, district, and month. The app pulls the closest historical district-season context from training data.",
    },
    {
        "title": "Review field inputs",
        "body": "Check weather, soil, and irrigation values. If a value looks unrealistic, change it before asking for guidance.",
    },
    {
        "title": "Read the result slowly",
        "body": "Use the top crop as a shortlist, read the explanation, then review warnings and next-step checks before acting.",
    },
]
FIELD_GUIDE = {
    "temp_avg": {
        "plain": "Average daytime field temperature for the current crop window.",
        "why": "Temperature strongly affects crop stress, growth speed, and flowering.",
    },
    "rain_total": {
        "plain": "Total rainfall during the current period you want to judge.",
        "why": "Rainfall changes water availability and the need for irrigation.",
    },
    "humidity_avg": {
        "plain": "Average air moisture around the crop period.",
        "why": "Humidity affects heat stress, disease pressure, and moisture loss.",
    },
    "max_temp": {
        "plain": "Highest likely temperature in the current period.",
        "why": "Short heat spikes can damage sensitive crops even when the average looks fine.",
    },
    "max_temp_3d": {
        "plain": "Recent hottest stretch over a few days.",
        "why": "This helps the model notice heatwave-like conditions.",
    },
    "rain_lag_14": {
        "plain": "Rainfall from the recent 14-day period.",
        "why": "Recent rain often matters more than seasonal total for soil moisture.",
    },
    "pH": {
        "plain": "Acidity or alkalinity of the soil.",
        "why": "Wrong pH can reduce nutrient availability even when fertilizer is present.",
    },
    "N": {
        "plain": "Nitrogen level in the soil.",
        "why": "Nitrogen supports leaf growth and crop vigor.",
    },
    "P": {
        "plain": "Phosphorus level in the soil.",
        "why": "Phosphorus supports root growth and early crop establishment.",
    },
    "K": {
        "plain": "Potassium level in the soil.",
        "why": "Potassium supports water balance, stress resistance, and grain or fruit quality.",
    },
    "irrigation_index": {
        "plain": "How dependable irrigation support is on this field.",
        "why": "Reliable irrigation can keep some water-intensive crops viable in weaker rainfall periods.",
    },
    "rotation_score": {
        "plain": "How healthy and varied the recent crop rotation has been.",
        "why": "Better rotation often improves soil condition and lowers disease pressure.",
    },
}
RESULT_GUIDE = [
    ("Top pattern match", "The crop that best matches the entered field profile and district-month history."),
    ("Confidence", "Higher confidence means the entered conditions look more familiar to the trained model and agree better with the rule checks."),
    ("Warnings", "Warnings tell you where the inputs look unusual, missing, or risky for the model."),
    ("Scenario check", "Scenarios show whether the ranking changes if rainfall or heat conditions become worse."),
    ("AI guide", "The AI guide turns the result into a conversational answer, but it still works only from the current prediction data."),
]
SCENARIO_GUIDE = {
    "low_rainfall": "Tests how the ranking changes if the period gets meaningfully drier.",
    "heatwave": "Tests how the ranking changes if temperatures jump higher than normal.",
    "high_irrigation": "Tests whether stronger irrigation would improve the ranking for water-sensitive crops.",
}


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
            background:
                radial-gradient(circle at top left, rgba(222, 196, 150, 0.36), transparent 34%),
                linear-gradient(180deg, #f7f1e6 0%, #efe5d4 100%);
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
        .hero-shell {
            background: linear-gradient(135deg, #fff7ea 0%, #f4ead8 100%);
            border: 1px solid #d8c6a7;
            border-radius: 24px;
            padding: 1.4rem 1.5rem;
            box-shadow: 0 18px 40px rgba(62, 46, 22, 0.08);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-family: Georgia, "Times New Roman", serif;
            color: #1b4532;
            font-size: 2rem;
            line-height: 1.1;
            margin: 0 0 0.4rem 0;
        }
        .hero-copy {
            color: #31443a;
            margin: 0;
            font-size: 1rem;
        }
        .mini-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 1rem 0 0 0;
        }
        .mini-card, .guide-card, .insight-card {
            background: rgba(255, 250, 242, 0.95);
            border: 1px solid #dfd2bc;
            border-radius: 18px;
            padding: 0.95rem 1rem;
        }
        .mini-kicker {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #7b6950;
            margin-bottom: 0.15rem;
        }
        .mini-value {
            color: #1f5d46;
            font-size: 1.2rem;
            font-weight: 700;
        }
        .step-number {
            display: inline-block;
            width: 1.75rem;
            height: 1.75rem;
            border-radius: 999px;
            background: #1f5d46;
            color: #fffaf2;
            text-align: center;
            line-height: 1.75rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }
        .section-title {
            font-family: Georgia, "Times New Roman", serif;
            color: #234332;
            margin-top: 0.3rem;
            margin-bottom: 0.2rem;
        }
        .confidence-strip {
            height: 0.75rem;
            border-radius: 999px;
            background: #e7dcc7;
            overflow: hidden;
            margin: 0.6rem 0 0.35rem 0;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #8b5e34 0%, #d48d35 45%, #1f5d46 100%);
        }
        .small-note {
            color: #6b5d47;
            font-size: 0.92rem;
        }
        .answer-shell {
            background: rgba(255, 252, 246, 0.98);
            border: 1px solid #d9ccb5;
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 30px rgba(62, 46, 22, 0.06);
        }
        .result-heading {
            font-family: Georgia, "Times New Roman", serif;
            color: #1f5d46;
            margin: 0 0 0.35rem 0;
        }
        @media (max-width: 900px) {
            .mini-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .hero-shell {
                padding: 1.1rem 1rem;
            }
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
    return month_value


def payload_signature(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def invalidate_stale_results(
    base_signature: str,
    simulation_signature: str,
) -> None:
    previous_prediction_signature = st.session_state.get("prediction_signature")
    previous_simulation_signature = st.session_state.get("simulation_signature")
    if previous_prediction_signature is not None and previous_prediction_signature != base_signature:
        st.session_state.pop("last_prediction", None)
        st.session_state.pop("last_payload", None)
        st.session_state.pop("last_llm_guide", None)
        st.session_state.pop("llm_guide_signature", None)
    if previous_simulation_signature is not None and previous_simulation_signature != simulation_signature:
        st.session_state.pop("last_simulation", None)


def confidence_summary(confidence_percent: float) -> tuple[str, str]:
    if confidence_percent >= 80:
        return "High confidence", "The model sees this field pattern often and the signals mostly agree."
    if confidence_percent >= 60:
        return "Moderate confidence", "This is a reasonable shortlist, but local validation still matters."
    return "Low confidence", "Use this as a first hint only and double-check the field conditions."


def band_summary(feature: str, value: float, context: dict[str, Any]) -> str:
    band = context.get("validation_bands", {}).get(feature, {})
    typical_min = band.get("typical_min")
    typical_max = band.get("typical_max")
    if typical_min is None or typical_max is None:
        return "No local comparison available yet."
    if value < typical_min:
        return "Below the typical local range."
    if value > typical_max:
        return "Above the typical local range."
    return "Within the typical local range."


def feature_story(feature: dict[str, Any]) -> str:
    feature_key = str(feature.get("feature_key", "")).strip()
    guide = FIELD_GUIDE.get(feature_key, {})
    label = pretty(feature_key or str(feature.get("feature", "this factor")))
    direction = "helped" if float(feature.get("impact", 0.0)) >= 0 else "held back"
    why = guide.get("why")
    if why:
        return f"{label} {direction} the ranking. {why}"
    return f"{label} {direction} the ranking."


def render_hero(
    catalog: dict[str, Any],
    health: dict[str, Any],
    guidance_scope: dict[str, Any],
    live_weather_status: str,
    llm_support: dict[str, Any],
) -> None:
    coverage = catalog.get("coverage", {})
    ai_guide_status = "Ready" if llm_support.get("enabled") else "Setup needed"
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-title">Simple crop guidance for first-time users</div>
            <p class="hero-copy">
                This screen turns district-season history plus your field inputs into an explainable crop shortlist.
                It is designed to be read slowly, with checks and plain-language help at every step.
            </p>
            <div class="mini-grid">
                <div class="mini-card">
                    <div class="mini-kicker">Districts</div>
                    <div class="mini-value">{coverage.get('region_count', 0)}</div>
                </div>
                <div class="mini-card">
                    <div class="mini-kicker">Crops</div>
                    <div class="mini-value">{coverage.get('crop_count', 0)}</div>
                </div>
                <div class="mini-card">
                    <div class="mini-kicker">Model mode</div>
                    <div class="mini-value">{html.escape(pretty(str(health.get('mode', 'unknown'))))}</div>
                </div>
                <div class="mini-card">
                    <div class="mini-kicker">Live weather</div>
                    <div class="mini-value">{html.escape(pretty(str(live_weather_status)))}</div>
                </div>
                <div class="mini-card">
                    <div class="mini-kicker">AI guide</div>
                    <div class="mini-value">{html.escape(ai_guide_status)}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    scope_note = guidance_scope.get(
        "product_note",
        "This prototype uses historical district-month context from training data.",
    )
    st.markdown(
        f'<div class="soft-note">{html.escape(scope_note)}</div>',
        unsafe_allow_html=True,
    )


def render_beginner_steps() -> None:
    st.subheader("How to use this page")
    step_columns = st.columns(len(BEGINNER_STEPS))
    for index, step in enumerate(BEGINNER_STEPS, start=1):
        with step_columns[index - 1]:
            st.markdown(
                f"""
                <div class="guide-card">
                    <div class="step-number">{index}</div>
                    <div class="section-title">{html.escape(step['title'])}</div>
                    <div class="small-note">{html.escape(step['body'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_field_guide(context: dict[str, Any]) -> None:
    with st.expander("Need help with the field inputs?", expanded=False):
        st.write("Use these plain-language descriptions if the form terms feel technical.")
        for feature in PRIMARY_NUMERIC_FIELDS:
            guide = FIELD_GUIDE.get(feature, {})
            if not guide:
                continue
            st.markdown(f"**{FIELD_LABELS.get(feature, pretty(feature))}**")
            st.write(guide.get("plain", ""))
            st.caption(guide.get("why", ""))
            default_value = context.get("feature_defaults", {}).get(feature)
            if default_value not in (None, ""):
                st.caption(f"Current district-month default: {default_value}")


def render_result_guide() -> None:
    with st.expander("How to read the result", expanded=False):
        for title, body in RESULT_GUIDE:
            st.markdown(f"**{title}**")
            st.write(body)


def render_ai_guide(
    prediction: dict[str, Any],
    llm_support: dict[str, Any],
    payload: dict[str, Any],
    region: str,
    state: str,
) -> None:
    st.subheader("Ask the AI guide")
    if not llm_support.get("enabled"):
        st.info(
            "The AI guide is optional and currently turned off. Add `GROQ_API_KEY` to enable Groq answers here."
        )
        return

    supported_languages = llm_support.get("supported_languages", PREFERRED_LANGUAGES) or PREFERRED_LANGUAGES
    default_language = supported_languages[0] if supported_languages else "English"
    selected_language = st.selectbox(
        "Answer language",
        options=supported_languages,
        index=0,
        key="llm_guide_language",
        help="This affects only the conversational AI answer, not the core model prediction.",
    )
    default_question = (
        "Why is this crop ranked first, and what should I verify before making a decision?"
    )
    user_question = st.text_area(
        "Question for the AI guide",
        value=default_question if "llm_guide_question" not in st.session_state else st.session_state["llm_guide_question"],
        key="llm_guide_question",
        height=120,
        help="Try asking what this means, why a crop came first, or what to check next on the field.",
    )
    request_signature = payload_signature(
        {
            "prediction_request_id": prediction.get("request_id"),
            "question": user_question,
            "language": selected_language or default_language,
        }
    )

    if st.button("Ask AI guide", key="ask_ai_guide", width="stretch"):
        with st.spinner("Preparing a simpler answer from the current prediction..."):
            guide_response, error = post_json(
                "/llm-guide",
                {
                    "prediction": prediction,
                    "input_snapshot": payload,
                    "preferred_language": selected_language or default_language,
                    "user_question": user_question,
                    "region": region,
                    "state": state,
                },
            )
        if error:
            st.error(f"AI guide failed: {error}")
        else:
            st.session_state["last_llm_guide"] = guide_response
            st.session_state["llm_guide_signature"] = request_signature

    guide_response = st.session_state.get("last_llm_guide")
    if not guide_response or st.session_state.get("llm_guide_signature") != request_signature:
        st.caption(
            "This answer will stay grounded in the current prediction, warnings, and district context. It does not pull live weather on its own."
        )
        return

    st.markdown(
        """
        <div class="answer-shell">
            <div class="mini-kicker">AI guide answer</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(guide_response.get("answer", "No answer returned."))
    st.caption(guide_response.get("disclaimer", ""))
    latency = guide_response.get("latency_ms")
    provider = pretty(str(guide_response.get("provider", "groq")))
    model_name = str(guide_response.get("model", "unknown"))
    if latency is not None:
        st.caption(f"Provider: {provider} | Model: {model_name} | Latency: {latency} ms")
    else:
        st.caption(f"Provider: {provider} | Model: {model_name}")


def render_prediction(
    prediction: dict[str, Any],
    llm_support: dict[str, Any],
    payload: dict[str, Any],
    region: str,
    state: str,
) -> None:
    recommendations = prediction.get("recommendations", [])
    if not recommendations:
        st.warning("No recommendation was returned for this input.")
        return

    top_choice = recommendations[0]
    confidence = float(prediction.get("confidence", 0.0) or 0.0) * 100.0
    confidence_title, confidence_copy = confidence_summary(confidence)
    farmer_action = str(prediction.get("farmer_action", "")).strip()
    guidance_scope = prediction.get("guidance_scope", {})
    st.markdown(
        f"""
        <div class="result-card">
            <h3 class="result-heading">Top pattern match: {html.escape(pretty(top_choice.get("crop", "")))}</h3>
            <p style="margin:0; color:#243830;"><strong>{html.escape(confidence_title)}</strong> - {confidence:.1f}%</p>
            <div class="confidence-strip"><div class="confidence-fill" style="width:{max(0.0, min(confidence, 100.0)):.1f}%;"></div></div>
            <p class="small-note" style="margin:0;">{html.escape(confidence_copy)}</p>
            <p style="margin:0.7rem 0 0 0; color:#243830;">{html.escape(prediction.get("farmer_message", prediction.get("explanation", "")))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_features = prediction.get("top_features", [])
    why_not = prediction.get("why_not", [])
    warnings = prediction.get("warnings", [])
    localized_context = prediction.get("localized_context", {})
    summary_tab, explain_tab, help_tab, ai_tab = st.tabs(
        ["Result summary", "Why the app said this", "Help and glossary", "Ask AI guide"]
    )

    with summary_tab:
        highlight_columns = st.columns(3)
        with highlight_columns[0]:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="mini-kicker">Best next step</div>
                    <div class="small-note">{html.escape(farmer_action or "Review the field inputs and compare the top two crops.")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with highlight_columns[1]:
            match_level = localized_context.get("match_level", "unknown")
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="mini-kicker">Context source</div>
                    <div class="small-note">{html.escape(pretty(str(match_level)))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with highlight_columns[2]:
            recommended_use = guidance_scope.get(
                "recommended_use",
                "Treat this as shortlist guidance, not a final planting instruction.",
            )
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="mini-kicker">Use it like this</div>
                    <div class="small-note">{html.escape(recommended_use)}</div>
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
        st.dataframe(score_frame, width="stretch", hide_index=True)

        if why_not:
            st.subheader("Why the others came lower")
            for item in why_not:
                st.markdown(
                    f"""
                    <div class="guide-card">
                        <div class="section-title">{html.escape(pretty(item.get("crop", "")))}</div>
                        <div class="small-note">{html.escape(item.get("reason", ""))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if warnings:
            st.subheader("Important checks before deciding")
            for item in warnings:
                st.warning(item)

    with explain_tab:
        st.subheader("Simple explanation")
        st.write(prediction.get("explanation", "No explanation was returned."))

        if top_features:
            st.subheader("What most influenced the result")
            for feature in top_features:
                st.markdown(
                    f"""
                    <div class="guide-card">
                        <div class="section-title">{html.escape(pretty(feature.get("feature_key", "")))}</div>
                        <div class="small-note">{html.escape(feature_story(feature))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        confidence_breakdown = prediction.get("confidence_breakdown", {})
        if confidence_breakdown:
            st.subheader("Confidence breakdown")
            breakdown_rows = []
            for key, value in confidence_breakdown.items():
                if key == "data_confidence":
                    meaning = "How complete and realistic the entered field values look."
                elif key == "geo_confidence":
                    meaning = "How confidently the district mapping and local context were resolved."
                else:
                    meaning = "How much the learned model agrees with the agronomy rule check."
                breakdown_rows.append(
                    {
                        "Part": pretty(key),
                        "Score": round(float(value) * 100.0, 1),
                        "Meaning": meaning,
                    }
                )
            st.dataframe(pd.DataFrame(breakdown_rows), width="stretch", hide_index=True)

        if localized_context:
            with st.expander("See the district-month context behind this result", expanded=False):
                st.json(localized_context)

    with help_tab:
        render_result_guide()
        st.subheader("Quick glossary")
        glossary_rows = []
        for feature in ["temp_avg", "rain_total", "humidity_avg", "pH", "N", "P", "K", "irrigation_index", "rotation_score"]:
            guide = FIELD_GUIDE.get(feature, {})
            glossary_rows.append(
                {
                    "Term": FIELD_LABELS.get(feature, pretty(feature)),
                    "Meaning": guide.get("plain", ""),
                    "Why it matters": guide.get("why", ""),
                }
            )
        st.dataframe(pd.DataFrame(glossary_rows), width="stretch", hide_index=True)

    with ai_tab:
        render_ai_guide(prediction, llm_support, payload, region, state)


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
        guide_copy = SCENARIO_GUIDE.get(scenario_name)
        if guide_copy:
            st.caption(guide_copy)
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
            st.dataframe(table, width="stretch", hide_index=True)


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
        page_title="Climate Crop Guidance",
        page_icon=":seedling:",
        layout="wide",
    )
    apply_styles()

    health, health_error = fetch_json("/health")
    catalog, catalog_error = fetch_json("/catalog")
    sanity, _ = fetch_json("/sanity")
    if health_error:
        st.error(f"API health check failed: {health_error}")
        st.stop()
    if catalog_error or not catalog:
        st.error(f"Catalog load failed: {catalog_error or 'No response'}")
        st.stop()

    guidance_scope = catalog.get("guidance_scope", {})
    live_weather_status = catalog.get("temporal_context", {}).get("live_weather_status", "unknown")
    llm_support = catalog.get("llm_support", {})
    render_hero(catalog, health, guidance_scope, live_weather_status, llm_support)
    render_beginner_steps()

    left, right = st.columns([1.4, 1.0])
    with right:
        st.markdown(
            f"""
            <div class="guide-card">
                <div class="mini-kicker">System info</div>
                <div class="small-note">API: {html.escape(API_BASE_URL)}</div>
                <div class="small-note">Model version: {html.escape(str(health.get('model_version', 'unknown')))}</div>
                <div class="small-note">Mode: {html.escape(str(health.get('mode', 'unknown')))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
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
            "Target month for historical district context",
            options=list(month_label_map.keys()),
            index=list(month_label_map.keys()).index(default_month_label),
        )
        target_month = month_label_map[selected_month_label]
        target_time = build_target_time(target_month)
        st.caption("Current autofill uses same-month historical district context. Month drives the context lookup today; live year-specific weather autofill is still pending.")

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
    source_summary = context.get("source_summary", {})
    if source_summary:
        st.caption(
            "Autofill source: "
            f"climate={pretty(str(source_summary.get('climate', 'unknown')))}, "
            f"soil={pretty(str(source_summary.get('soil', 'unknown')))}, "
            f"crop prior={pretty(str(source_summary.get('crop_prior', 'unknown')))}"
        )

    st.subheader("Main inputs")
    render_field_guide(context)
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
                st.caption(
                    f"Typical local range: {band.get('typical_min')} to {band.get('typical_max')}."
                )
                st.caption(band_summary(feature, numeric_values[feature], context))

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
        help="Use these to see whether the crop ranking changes under tougher weather conditions.",
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
    selected_scenario_names = [scenario_options[item] for item in selected_scenarios]
    base_signature = payload_signature(payload)
    simulation_signature = payload_signature({**payload, "scenario_names": selected_scenario_names})
    invalidate_stale_results(base_signature, simulation_signature)

    predict_col, simulate_col = st.columns(2)
    if predict_col.button("Get recommendation", width="stretch"):
        prediction, error = post_json("/predict", payload)
        if error:
            st.error(f"Prediction failed: {error}")
        else:
            st.session_state["last_prediction"] = prediction
            st.session_state["last_payload"] = payload
            st.session_state["prediction_signature"] = base_signature

    if simulate_col.button("Run demo scenarios", width="stretch"):
        simulation_payload = {
            **payload,
            "scenario_names": selected_scenario_names,
        }
        simulation, error = post_json("/simulate", simulation_payload)
        if error:
            st.error(f"Scenario simulation failed: {error}")
        else:
            st.session_state["last_simulation"] = simulation
            st.session_state["last_payload"] = payload
            st.session_state["simulation_signature"] = simulation_signature

    prediction = st.session_state.get("last_prediction")
    if prediction:
        render_prediction(
            prediction,
            llm_support,
            st.session_state.get("last_payload", payload),
            selected_region,
            selected_state,
        )
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
