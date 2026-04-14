from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests


DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "30"))
DEFAULT_MAX_COMPLETION_TOKENS = int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "900"))
DEFAULT_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
STRICT_SCHEMA_MODELS = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
REQUIRED_KEYS = (
    "scenario_summary",
    "environmental_change",
    "crop_response_analysis",
    "ranking_changes",
    "key_drivers",
    "stability_assessment",
    "confidence_note",
)
FEATURE_NAME_ALIASES = {
    "rain_total": "rainfall",
    "rain_lag_14": "recent rainfall",
    "humidity_avg": "humidity",
    "temp_avg": "temperature",
    "max_temp": "maximum temperature",
    "max_temp_3d": "recent heat",
    "irrigation_index": "irrigation",
    "pH": "soil pH",
    "N": "nitrogen",
    "P": "phosphorus",
    "K": "potassium",
    "soil_health_index": "soil health",
}
KEY_DRIVER_ALIASES = {
    "rain_total": "rainfall",
    "rain_lag_14": "recent_rainfall",
    "humidity_avg": "humidity",
    "temp_avg": "temperature",
    "max_temp": "max_temperature",
    "max_temp_3d": "recent_heat",
    "irrigation_index": "irrigation",
    "pH": "soil_ph",
    "N": "nitrogen",
    "P": "phosphorus",
    "K": "potassium",
    "soil_health_index": "soil_health",
}
DISALLOWED_UNOBSERVED_PHRASES = (
    "soil type",
    "soil texture",
    "pest",
    "disease",
    "market demand",
    "farmer preference",
    "management practice",
    "irrigation management",
    "unobserved",
    "not considered",
    "other factors",
)
WATER_SENSITIVE_CROPS = {
    "banana",
    "coconut",
    "rice",
    "sugarcane",
    "arcanut processed",
    "arecanut",
    "cardamom",
    "black pepper",
    "pineapple",
    "tapioca",
}
HEAT_RESILIENT_CROPS = {
    "castor seed",
    "groundnut",
    "guar seed",
    "jowar",
    "maize",
    "ragi",
    "sesamum",
    "small millets",
    "sunflower",
}
NUTRIENT_DEPENDENT_CROPS = {
    "banana",
    "cotton lint",
    "maize",
    "onion",
    "potato",
    "sugarcane",
    "tobacco",
    "turmeric",
}
SYSTEM_PROMPT = (
    "You are an agricultural AI expert explaining scenario-driven changes in crop recommendation rankings. "
    "Write in a formal, analytical, report-ready tone. Use only the supplied evidence. "
    "Do not invent unsupported climate, soil, pest, irrigation, or management facts. "
    "If the shift is weak or uncertain, say so explicitly."
)
SCENARIO_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "scenario_summary": {"type": "string"},
        "environmental_change": {"type": "string"},
        "crop_response_analysis": {"type": "string"},
        "ranking_changes": {"type": "string"},
        "key_drivers": {"type": "array", "items": {"type": "string"}},
        "stability_assessment": {"type": "string"},
        "confidence_note": {"type": "string"},
    },
    "required": list(REQUIRED_KEYS),
}

_DOTENV_LOADED = False


def generate_scenario_explanation(
    baseline: dict,
    scenario: dict,
    scenario_name: str,
    feature_changes: dict,
) -> dict:
    load_local_env()
    comparison = derive_comparison_metrics(baseline=baseline, scenario=scenario)
    prompt = build_scenario_prompt(
        baseline=baseline,
        scenario=scenario,
        scenario_name=scenario_name,
        feature_changes=feature_changes,
        comparison=comparison,
    )
    try:
        raw_response = call_groq_llm(prompt)
        parsed_response = parse_json_response(raw_response)
        normalized = normalize_explanation_payload(
            payload=parsed_response,
            baseline=baseline,
            scenario=scenario,
            scenario_name=scenario_name,
            feature_changes=feature_changes,
            comparison=comparison,
            fallback_used=False,
        )
        if explanation_needs_repair(
            explanation=normalized,
            baseline=baseline,
            scenario=scenario,
            feature_changes=feature_changes,
        ):
            repair_prompt = build_repair_prompt(
                baseline=baseline,
                scenario=scenario,
                scenario_name=scenario_name,
                feature_changes=feature_changes,
                comparison=comparison,
                current_explanation=normalized,
            )
            repaired_response = parse_json_response(call_groq_llm(repair_prompt))
            normalized = normalize_explanation_payload(
                payload=repaired_response,
                baseline=baseline,
                scenario=scenario,
                scenario_name=scenario_name,
                feature_changes=feature_changes,
                comparison=comparison,
                fallback_used=False,
            )
        return enrich_explanation_payload(
            explanation=normalized,
            baseline=baseline,
            scenario=scenario,
            feature_changes=feature_changes,
            comparison=comparison,
        )
    except Exception as exc:
        fallback_payload = build_fallback_explanation(
            baseline=baseline,
            scenario=scenario,
            scenario_name=scenario_name,
            feature_changes=feature_changes,
            comparison=comparison,
            error_message=str(exc),
        )
        normalized = normalize_explanation_payload(
            payload=fallback_payload,
            baseline=baseline,
            scenario=scenario,
            scenario_name=scenario_name,
            feature_changes=feature_changes,
            comparison=comparison,
            fallback_used=True,
        )
        return enrich_explanation_payload(
            explanation=normalized,
            baseline=baseline,
            scenario=scenario,
            feature_changes=feature_changes,
            comparison=comparison,
        )


def call_groq_llm(prompt: str) -> str:
    load_local_env()
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": DEFAULT_TEMPERATURE,
        "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
        "response_format": build_response_format(model),
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.post(
                DEFAULT_GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=DEFAULT_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("Groq returned an empty completion.")
            return content
        except Exception as exc:
            last_error = exc
            if attempt == 2:
                break
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"Groq request failed after retries: {last_error}") from last_error


def format_for_ui(explanation_dict: dict[str, Any]) -> dict[str, Any]:
    scenario_name = humanize_name(str(explanation_dict.get("scenario_name", "Scenario")))
    return {
        "title": f"{scenario_name} Impact Analysis",
        "sections": [
            {
                "heading": "What Changed",
                "content": explanation_dict.get("environmental_change", ""),
            },
            {
                "heading": "Impact on Crops",
                "content": explanation_dict.get("crop_response_analysis", ""),
            },
            {
                "heading": "Ranking Changes",
                "content": explanation_dict.get("ranking_changes", ""),
            },
            {
                "heading": "Interpretation",
                "content": " ".join(
                    part
                    for part in [
                        explanation_dict.get("stability_assessment", ""),
                        explanation_dict.get("confidence_note", ""),
                    ]
                    if part
                ),
            },
        ],
    }


def render_explanations_markdown(records: list[dict[str, Any]]) -> str:
    lines = []
    for record in records:
        scenario_title = humanize_name(str(record.get("scenario_name", "Scenario")))
        explanation = record.get("explanation", {})
        lines.extend(
            [
                f"### Scenario: {scenario_title}",
                "",
                f"* {explanation.get('scenario_summary', '')}",
                f"* {explanation.get('environmental_change', '')}",
                f"* {explanation.get('crop_response_analysis', '')}",
                f"* {explanation.get('ranking_changes', '')}",
                f"* {explanation.get('stability_assessment', '')}",
                f"* {explanation.get('confidence_note', '')}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def build_response_format(model: str) -> dict[str, Any]:
    if model in STRICT_SCHEMA_MODELS:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "scenario_explanation",
                "schema": SCENARIO_RESPONSE_SCHEMA,
                "strict": True,
            },
        }
    return {"type": "json_object"}


def load_local_env() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return

    env_candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    for env_path in env_candidates:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value
        break
    _DOTENV_LOADED = True


def build_scenario_prompt(
    baseline: dict,
    scenario: dict,
    scenario_name: str,
    feature_changes: dict,
    comparison: dict,
) -> str:
    trait_hints = crop_trait_hints(baseline=baseline, scenario=scenario, feature_changes=feature_changes)
    top_crop_deltas = top_crop_delta_summary(baseline=baseline, scenario=scenario)
    schema_hint = json.dumps(
        {
            "scenario_summary": "string",
            "environmental_change": "string",
            "crop_response_analysis": "string",
            "ranking_changes": "string",
            "key_drivers": ["string"],
            "stability_assessment": "string",
            "confidence_note": "string",
        },
        indent=2,
    )
    return (
        "Explain the impact of the scenario on crop recommendations.\n\n"
        f"Scenario name: {humanize_name(scenario_name)}\n\n"
        "Baseline recommendation:\n"
        f"{json.dumps(baseline, indent=2, ensure_ascii=False)}\n\n"
        "Scenario-modified recommendation:\n"
        f"{json.dumps(scenario, indent=2, ensure_ascii=False)}\n\n"
        "Environmental feature changes:\n"
        f"{json.dumps(feature_changes, indent=2, ensure_ascii=False)}\n\n"
        "Derived comparison metrics:\n"
        f"{json.dumps(comparison, indent=2, ensure_ascii=False)}\n\n"
        "Agronomic response hints to use only if directly relevant:\n"
        f"{json.dumps(trait_hints, indent=2, ensure_ascii=False)}\n\n"
        "Top crop score shifts:\n"
        f"{json.dumps(top_crop_deltas, indent=2, ensure_ascii=False)}\n\n"
        "Instructions:\n"
        "- Use only the variables, crops, scores, and ranking changes shown above.\n"
        "- Tie agronomic reasoning directly to the changed variables.\n"
        "- Explicitly compare the baseline and scenario rankings.\n"
        "- Explain why crops moved up or down using water, temperature, or nutrient logic only when supported.\n"
        "- Mention the actual crops that improved or weakened, not generic crop groups only.\n"
        "- Mention whether the shift is minor, moderate, or significant and justify that judgment using the supplied score and overlap information.\n"
        "- If the ranking stayed similar, explicitly state that the ordering is stable and note the small score deltas.\n"
        "- If a crop is highlighted as water-sensitive, heat-resilient, or nutrient-dependent, connect that trait to the changed variables.\n"
        "- Do not mention any variable that was not provided.\n"
        "- Avoid generic phrases such as 'some crops changed slightly' unless you immediately identify which crops changed.\n"
        "- Keep each field concise but evidence-based, ideally 1-3 sentences.\n"
        "- Do not output markdown or code fences.\n"
        "- Return valid JSON only with this exact structure:\n"
        f"{schema_hint}\n"
    )


def derive_comparison_metrics(baseline: dict, scenario: dict) -> dict[str, Any]:
    baseline_ranking = [str(item) for item in baseline.get("ranking", [])]
    scenario_ranking = [str(item) for item in scenario.get("ranking", [])]
    baseline_scores = {str(key): float(value) for key, value in baseline.get("scores", {}).items()}
    scenario_scores = {str(key): float(value) for key, value in scenario.get("scores", {}).items()}
    all_crops = list(dict.fromkeys(baseline_ranking + scenario_ranking + list(baseline_scores) + list(scenario_scores)))

    baseline_positions = {crop: index + 1 for index, crop in enumerate(baseline_ranking)}
    scenario_positions = {crop: index + 1 for index, crop in enumerate(scenario_ranking)}
    moved_up = []
    moved_down = []
    for crop in all_crops:
        if crop in baseline_positions and crop in scenario_positions:
            if scenario_positions[crop] < baseline_positions[crop]:
                moved_up.append(crop)
            elif scenario_positions[crop] > baseline_positions[crop]:
                moved_down.append(crop)

    score_deltas = {
        crop: round(float(scenario_scores.get(crop, 0.0) - baseline_scores.get(crop, 0.0)), 6)
        for crop in all_crops
    }
    abs_deltas = [abs(value) for value in score_deltas.values()]
    top3_overlap = len(set(baseline_ranking[:3]).intersection(set(scenario_ranking[:3]))) / 3.0 if baseline_ranking or scenario_ranking else 1.0
    top_crop_changed = str(baseline.get("top_crop", "")) != str(scenario.get("top_crop", ""))

    return {
        "top_crop_changed": top_crop_changed,
        "baseline_top_crop": str(baseline.get("top_crop", "")),
        "scenario_top_crop": str(scenario.get("top_crop", "")),
        "moved_up": moved_up,
        "moved_down": moved_down,
        "entered_top_ranking": [crop for crop in scenario_ranking if crop not in baseline_ranking],
        "dropped_from_top_ranking": [crop for crop in baseline_ranking if crop not in scenario_ranking],
        "top3_overlap_ratio": round(float(top3_overlap), 6),
        "max_abs_score_delta": round(float(max(abs_deltas) if abs_deltas else 0.0), 6),
        "mean_abs_score_delta": round(float(sum(abs_deltas) / len(abs_deltas) if abs_deltas else 0.0), 6),
        "score_deltas": score_deltas,
        "change_magnitude": assess_change_magnitude(
            top_crop_changed=top_crop_changed,
            top3_overlap=top3_overlap,
            max_abs_score_delta=max(abs_deltas) if abs_deltas else 0.0,
        ),
    }


def crop_trait_hints(baseline: dict, scenario: dict, feature_changes: dict) -> dict[str, list[str]]:
    crops = list(
        dict.fromkeys(
            [str(item) for item in baseline.get("ranking", [])]
            + [str(item) for item in scenario.get("ranking", [])]
        )
    )
    changed_feature_keys = {normalize_feature_key(key) for key in feature_changes}
    hints: dict[str, list[str]] = {"water_sensitive": [], "heat_resilient": [], "nutrient_dependent": []}
    for crop in crops:
        crop_key = normalize_crop_name(crop)
        if is_water_related(changed_feature_keys) and crop_key in WATER_SENSITIVE_CROPS:
            hints["water_sensitive"].append(crop)
        if is_heat_related(changed_feature_keys) and crop_key in HEAT_RESILIENT_CROPS:
            hints["heat_resilient"].append(crop)
        if is_nutrient_related(changed_feature_keys) and crop_key in NUTRIENT_DEPENDENT_CROPS:
            hints["nutrient_dependent"].append(crop)
    return hints


def build_repair_prompt(
    baseline: dict,
    scenario: dict,
    scenario_name: str,
    feature_changes: dict,
    comparison: dict,
    current_explanation: dict,
) -> str:
    gaps = identify_signal_gaps(
        explanation=current_explanation,
        baseline=baseline,
        scenario=scenario,
        feature_changes=feature_changes,
    )
    return (
        "Revise the explanation so it becomes more concrete and evidence-based.\n\n"
        f"Scenario name: {humanize_name(scenario_name)}\n"
        f"Baseline: {json.dumps(baseline, ensure_ascii=False)}\n"
        f"Scenario: {json.dumps(scenario, ensure_ascii=False)}\n"
        f"Feature changes: {json.dumps(feature_changes, ensure_ascii=False)}\n"
        f"Comparison metrics: {json.dumps(comparison, ensure_ascii=False)}\n"
        f"Current explanation: {json.dumps(current_explanation, ensure_ascii=False)}\n"
        f"Problems to fix: {json.dumps(gaps, ensure_ascii=False)}\n\n"
        "Rewrite all fields so they explicitly mention the changed variables, the affected crops, the ranking movement, "
        "and whether the shift is minor, moderate, or significant. Return valid JSON only."
    )


def normalize_explanation_payload(
    payload: dict,
    baseline: dict,
    scenario: dict,
    scenario_name: str,
    feature_changes: dict,
    comparison: dict,
    fallback_used: bool,
) -> dict:
    fallback = build_fallback_explanation(
        baseline=baseline,
        scenario=scenario,
        scenario_name=scenario_name,
        feature_changes=feature_changes,
        comparison=comparison,
        error_message="LLM output normalization fallback",
    )
    normalized: dict[str, Any] = {}
    for key in REQUIRED_KEYS:
        if key == "key_drivers":
            raw_value = payload.get(key, fallback.get(key, []))
            if isinstance(raw_value, str):
                raw_list = [item.strip() for item in raw_value.split(",") if item.strip()]
            elif isinstance(raw_value, list):
                raw_list = [str(item).strip() for item in raw_value if str(item).strip()]
            else:
                raw_list = []
            normalized[key] = raw_list or fallback["key_drivers"]
            continue
        value = str(payload.get(key, fallback.get(key, ""))).strip()
        normalized[key] = value or fallback[key]

    if fallback_used and "fallback" not in normalized["confidence_note"].casefold():
        normalized["confidence_note"] = (
            normalized["confidence_note"].rstrip(".")
            + ". This response used the rule-based fallback because the Groq call was unavailable."
        )
    return normalized


def enrich_explanation_payload(
    explanation: dict,
    baseline: dict,
    scenario: dict,
    feature_changes: dict,
    comparison: dict,
) -> dict:
    enriched = dict(explanation)
    enriched = ground_explanation_payload(
        explanation=enriched,
        baseline=baseline,
        scenario=scenario,
        feature_changes=feature_changes,
        comparison=comparison,
    )
    feature_names = [humanize_feature_name(key) for key in feature_changes.keys()]
    crops = [str(item) for item in baseline.get("ranking", [])[:5]] + [str(item) for item in scenario.get("ranking", [])[:5]]
    crops = list(dict.fromkeys([crop for crop in crops if crop]))

    environmental_text = enriched.get("environmental_change", "")
    if feature_names and not contains_any(environmental_text, feature_names):
        enriched["environmental_change"] = environmental_text.rstrip(".") + ". " + (
            "Changed variables: " + ", ".join(feature_names) + "."
        )

    crop_text = enriched.get("crop_response_analysis", "")
    if crops and not contains_any(crop_text, crops):
        enriched["crop_response_analysis"] = crop_text.rstrip(".") + ". " + build_crop_response_text(
            baseline=baseline,
            scenario=scenario,
            feature_changes=feature_changes,
            comparison=comparison,
        )

    ranking_text = enriched.get("ranking_changes", "")
    if not contains_any(ranking_text, crops):
        enriched["ranking_changes"] = ranking_text.rstrip(".") + ". " + build_ranking_change_text(
            baseline=baseline,
            scenario=scenario,
            comparison=comparison,
        )

    stability_text = enriched.get("stability_assessment", "")
    if "top-3 overlap ratio" not in stability_text.casefold():
        enriched["stability_assessment"] = stability_text.rstrip(".") + ". " + build_stability_text(comparison)

    if not enriched.get("key_drivers"):
        enriched["key_drivers"] = feature_names[:3]
    return enriched


def ground_explanation_payload(
    explanation: dict,
    baseline: dict,
    scenario: dict,
    feature_changes: dict,
    comparison: dict,
) -> dict:
    grounded = dict(explanation)
    scenario_summary_fallback = build_scenario_summary_text(
        baseline=baseline,
        scenario=scenario,
        feature_changes=feature_changes,
        comparison=comparison,
    )
    environmental_fallback = build_environmental_change_text(feature_changes)
    crop_response_fallback = build_crop_response_text(
        baseline=baseline,
        scenario=scenario,
        feature_changes=feature_changes,
        comparison=comparison,
    )
    ranking_fallback = build_ranking_change_text(baseline=baseline, scenario=scenario, comparison=comparison)
    stability_fallback = build_stability_text(comparison)
    confidence_fallback = build_confidence_note(comparison)

    grounded["scenario_summary"] = prefer_grounded_text(
        text=str(grounded.get("scenario_summary", "")),
        fallback=scenario_summary_fallback,
        required_terms=required_summary_terms(baseline, scenario, feature_changes),
        feature_changes=feature_changes,
    )
    grounded["environmental_change"] = prefer_grounded_text(
        text=str(grounded.get("environmental_change", "")),
        fallback=environmental_fallback,
        required_terms=[humanize_feature_name(key) for key in feature_changes.keys()],
        feature_changes=feature_changes,
    )
    grounded["crop_response_analysis"] = prefer_grounded_text(
        text=str(grounded.get("crop_response_analysis", "")),
        fallback=crop_response_fallback,
        required_terms=required_crop_terms(baseline, scenario, feature_changes),
        feature_changes=feature_changes,
    )
    grounded["ranking_changes"] = prefer_grounded_text(
        text=str(grounded.get("ranking_changes", "")),
        fallback=ranking_fallback,
        required_terms=required_ranking_terms(baseline, scenario),
        feature_changes=feature_changes,
    )
    grounded["stability_assessment"] = prefer_grounded_text(
        text=str(grounded.get("stability_assessment", "")),
        fallback=stability_fallback,
        required_terms=["minor", "moderate", "significant", "stable", "top-3 overlap ratio"],
        feature_changes=feature_changes,
    )
    grounded["confidence_note"] = confidence_fallback
    grounded["key_drivers"] = build_key_drivers(feature_changes)
    return grounded


def explanation_needs_repair(
    explanation: dict,
    baseline: dict,
    scenario: dict,
    feature_changes: dict,
) -> bool:
    return bool(identify_signal_gaps(explanation=explanation, baseline=baseline, scenario=scenario, feature_changes=feature_changes))


def identify_signal_gaps(
    explanation: dict,
    baseline: dict,
    scenario: dict,
    feature_changes: dict,
) -> list[str]:
    gaps = []
    feature_names = [humanize_feature_name(key) for key in feature_changes.keys()]
    crops = [str(item) for item in baseline.get("ranking", [])[:5]] + [str(item) for item in scenario.get("ranking", [])[:5]]
    crops = list(dict.fromkeys([crop for crop in crops if crop]))

    if feature_names and not contains_any(explanation.get("environmental_change", ""), feature_names):
        gaps.append("environmental_change does not explicitly mention the changed variables")
    if feature_names and not contains_any(explanation.get("crop_response_analysis", ""), feature_names):
        gaps.append("crop_response_analysis does not tie crop response to the changed variables")
    if crops and not contains_any(explanation.get("crop_response_analysis", ""), crops):
        gaps.append("crop_response_analysis does not name the affected crops")
    if crops and not contains_any(explanation.get("ranking_changes", ""), crops):
        gaps.append("ranking_changes does not name the crops that moved or remained stable")
    if not contains_any(explanation.get("stability_assessment", ""), ["minor", "moderate", "significant", "stable"]):
        gaps.append("stability_assessment does not classify the size or stability of the ranking shift")
    if any(
        contains_disallowed_phrase(str(explanation.get(field, "")), feature_changes)
        for field in (
            "scenario_summary",
            "environmental_change",
            "crop_response_analysis",
            "ranking_changes",
            "stability_assessment",
            "confidence_note",
        )
    ):
        gaps.append("one or more fields include unsupported or unobserved claims")
    return gaps


def build_fallback_explanation(
    baseline: dict,
    scenario: dict,
    scenario_name: str,
    feature_changes: dict,
    comparison: dict,
    error_message: str,
) -> dict:
    scenario_title = humanize_name(scenario_name)
    formatted_changes = format_feature_changes(feature_changes)
    ranking_text = build_ranking_change_text(baseline=baseline, scenario=scenario, comparison=comparison)
    crop_response_text = build_crop_response_text(
        baseline=baseline,
        scenario=scenario,
        feature_changes=feature_changes,
        comparison=comparison,
    )
    stability_text = build_stability_text(comparison)
    baseline_top_crop = str(baseline.get("top_crop", ""))
    scenario_top_crop = str(scenario.get("top_crop", "")) or baseline_top_crop
    if baseline_top_crop and scenario_top_crop and baseline_top_crop != scenario_top_crop:
        crop_transition_text = f"The top recommendation shifts from {baseline_top_crop} to {scenario_top_crop}"
    else:
        crop_transition_text = f"The scenario remains centered on {scenario_top_crop}"

    return {
        "scenario_summary": (
            f"The {scenario_title} scenario evaluates how the recommendation profile responds when {formatted_changes}. "
            f"{crop_transition_text}, and the model registers a {comparison['change_magnitude']} shift in the crop ranking."
        ),
        "environmental_change": (
            f"Relative to the baseline recommendation, the scenario changes {formatted_changes}. "
            "The explanation is limited to these supplied feature shifts and the resulting score movement."
        ),
        "crop_response_analysis": crop_response_text,
        "ranking_changes": ranking_text,
        "key_drivers": [humanize_feature_name(key) for key in feature_changes.keys()] or ["ranking shift"],
        "stability_assessment": stability_text,
        "confidence_note": (
            "This explanation is grounded in the provided feature and score deltas. "
            f"The Groq response was not used directly ({error_message})."
        ),
    }


def build_scenario_summary_text(
    baseline: dict,
    scenario: dict,
    feature_changes: dict,
    comparison: dict,
) -> str:
    scenario_top_crop = str(scenario.get("top_crop", "")) or str(baseline.get("top_crop", ""))
    return (
        f"This scenario tests the recommendation response to {format_feature_changes(feature_changes)}. "
        f"The model still ranks {scenario_top_crop} first, and the overall shift is {comparison.get('change_magnitude', 'minor')}."
    )


def build_environmental_change_text(feature_changes: dict) -> str:
    return (
        f"The modified inputs are {format_feature_changes(feature_changes)}. "
        "No additional environmental or management variables are assumed beyond these supplied changes."
    )


def build_confidence_note(comparison: dict) -> str:
    return (
        "This explanation is grounded only in the supplied feature changes, baseline scores, scenario scores, "
        f"and ranking overlap. The reported effect is {comparison.get('change_magnitude', 'minor')}, so the interpretation "
        "should be read as a ranking-based response rather than a broader agronomic claim."
    )


def top_crop_delta_summary(baseline: dict, scenario: dict) -> list[dict[str, Any]]:
    baseline_scores = {str(key): float(value) for key, value in baseline.get("scores", {}).items()}
    scenario_scores = {str(key): float(value) for key, value in scenario.get("scores", {}).items()}
    all_crops = list(dict.fromkeys(list(baseline_scores) + list(scenario_scores)))
    rows = []
    for crop in all_crops:
        baseline_score = float(baseline_scores.get(crop, 0.0))
        scenario_score = float(scenario_scores.get(crop, 0.0))
        rows.append(
            {
                "crop": crop,
                "baseline_score": round(baseline_score, 6),
                "scenario_score": round(scenario_score, 6),
                "delta": round(scenario_score - baseline_score, 6),
            }
        )
    rows.sort(key=lambda item: abs(float(item["delta"])), reverse=True)
    return rows[:5]


def parse_json_response(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def build_ranking_change_text(baseline: dict, scenario: dict, comparison: dict) -> str:
    baseline_top = str(baseline.get("top_crop", ""))
    scenario_top = str(scenario.get("top_crop", ""))
    baseline_ranking = [str(item) for item in baseline.get("ranking", [])]
    scenario_ranking = [str(item) for item in scenario.get("ranking", [])]
    moved_up = comparison.get("moved_up", [])[:3]
    moved_down = comparison.get("moved_down", [])[:3]
    entered = comparison.get("entered_top_ranking", [])[:2]
    dropped = comparison.get("dropped_from_top_ranking", [])[:2]
    top3_baseline = baseline_ranking[:3]
    top3_scenario = scenario_ranking[:3]

    parts = []
    if baseline_top == scenario_top:
        parts.append(f"The top recommendation remains {baseline_top}.")
    else:
        parts.append(f"The top recommendation changes from {baseline_top} to {scenario_top}.")
    if top3_baseline and top3_baseline == top3_scenario:
        parts.append(f"The top three crops remain {', '.join(top3_scenario)}.")
    if moved_up:
        parts.append(f"Upward movement is most visible for {', '.join(moved_up)}.")
    if moved_down:
        parts.append(f"Downward movement is most visible for {', '.join(moved_down)}.")
    if entered:
        parts.append(f"{', '.join(entered)} enters the compared ranking window.")
    if dropped:
        parts.append(f"{', '.join(dropped)} falls out of the compared ranking window.")
    return " ".join(parts)


def build_crop_response_text(
    baseline: dict,
    scenario: dict,
    feature_changes: dict,
    comparison: dict,
) -> str:
    score_deltas = comparison.get("score_deltas", {})
    ordered = sorted(score_deltas.items(), key=lambda item: abs(float(item[1])), reverse=True)
    changed_feature_keys = {normalize_feature_key(key) for key in feature_changes}
    clauses = []

    for crop, delta in ordered[:4]:
        delta_value = float(delta)
        if abs(delta_value) < 1e-9:
            continue
        crop_key = normalize_crop_name(crop)
        direction = "gains" if delta_value > 0 else "loses"
        delta_text = f"{delta_value:+.6f}"
        reason = "its relative suitability signal shifts under the modified environment"
        if is_water_related(changed_feature_keys) and crop_key in WATER_SENSITIVE_CROPS:
            reason = (
                "it is more sensitive to water availability, so the changed moisture signal materially affects its suitability"
            )
        elif is_heat_related(changed_feature_keys) and crop_key in HEAT_RESILIENT_CROPS and delta_value > 0:
            reason = "it is comparatively more resilient under warmer conditions, so it is penalized less than competing crops"
        elif is_heat_related(changed_feature_keys) and delta_value < 0:
            reason = "the higher temperature signal weakens its relative suitability against more heat-tolerant alternatives"
        elif is_nutrient_related(changed_feature_keys) and crop_key in NUTRIENT_DEPENDENT_CROPS:
            reason = "its recommendation score is sensitive to soil fertility cues, so nutrient-related changes alter its standing"
        clauses.append(f"{crop} {direction} probability ({delta_text}) because {reason}.")

    if not clauses:
        return (
            "The scenario changes the ranking only modestly, so the crop-level response appears comparative rather than disruptive. "
            "The main effect is a small redistribution of probability mass among the leading crops."
        )
    return " ".join(clauses[:3])


def build_stability_text(comparison: dict) -> str:
    magnitude = comparison.get("change_magnitude", "minor")
    top_crop_changed = bool(comparison.get("top_crop_changed", False))
    top3_overlap = float(comparison.get("top3_overlap_ratio", 1.0))
    max_abs_delta = float(comparison.get("max_abs_score_delta", 0.0))
    mean_abs_delta = float(comparison.get("mean_abs_score_delta", 0.0))

    if magnitude == "significant":
        descriptor = "The scenario effect is significant and should be treated as a meaningful ranking shift."
    elif magnitude == "moderate":
        descriptor = "The scenario effect is moderate: the ordering moves, but the model signal is not fully destabilized."
    else:
        descriptor = "The scenario effect is minor, which suggests that the recommendation remains relatively stable."

    crop_change_text = "The top crop changes under the scenario." if top_crop_changed else "The top crop remains unchanged."
    return (
        f"{descriptor} {crop_change_text} "
        f"The top-3 overlap ratio is {top3_overlap:.2f}, the maximum score delta is {max_abs_delta:.3f}, "
        f"and the mean absolute score delta is {mean_abs_delta:.3f}."
    )


def build_key_drivers(feature_changes: dict) -> list[str]:
    drivers = []
    for feature_name in feature_changes.keys():
        driver = KEY_DRIVER_ALIASES.get(feature_name, normalize_feature_key(feature_name))
        if driver not in drivers:
            drivers.append(driver)
    return drivers[:5]


def required_summary_terms(baseline: dict, scenario: dict, feature_changes: dict) -> list[str]:
    terms = [humanize_feature_name(key) for key in feature_changes.keys()]
    top_crop_terms = [str(scenario.get("top_crop", "")) or str(baseline.get("top_crop", ""))]
    return [term for term in terms + top_crop_terms if term]


def required_crop_terms(baseline: dict, scenario: dict, feature_changes: dict) -> list[str]:
    crops = [str(item) for item in baseline.get("ranking", [])[:3]] + [str(item) for item in scenario.get("ranking", [])[:3]]
    features = [humanize_feature_name(key) for key in feature_changes.keys()]
    return [term for term in list(dict.fromkeys(crops + features)) if term]


def required_ranking_terms(baseline: dict, scenario: dict) -> list[str]:
    crops = [str(item) for item in baseline.get("ranking", [])[:5]] + [str(item) for item in scenario.get("ranking", [])[:5]]
    return [term for term in list(dict.fromkeys(crops)) if term]


def prefer_grounded_text(
    text: str,
    fallback: str,
    required_terms: list[str],
    feature_changes: dict,
) -> str:
    candidate = " ".join(str(text).split())
    if not candidate:
        return fallback
    if contains_disallowed_phrase(candidate, feature_changes):
        return fallback
    if required_terms and not contains_any(candidate, required_terms):
        return fallback
    return candidate


def assess_change_magnitude(top_crop_changed: bool, top3_overlap: float, max_abs_score_delta: float) -> str:
    if top_crop_changed or top3_overlap < 0.34 or max_abs_score_delta >= 0.08:
        return "significant"
    if top3_overlap < 0.67 or max_abs_score_delta >= 0.03:
        return "moderate"
    return "minor"


def format_feature_changes(feature_changes: dict) -> str:
    if not feature_changes:
        return "no explicit environmental variables are changed"
    items = [f"{humanize_feature_name(key)} ({value})" for key, value in feature_changes.items()]
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def humanize_name(value: str) -> str:
    return str(value).replace("_", " ").strip().title()


def humanize_feature_name(feature_name: str) -> str:
    return FEATURE_NAME_ALIASES.get(feature_name, str(feature_name).replace("_", " ").strip())


def normalize_feature_key(value: str) -> str:
    return str(value).strip().casefold().replace(" ", "_")


def normalize_crop_name(value: str) -> str:
    return str(value).strip().casefold().replace("_", " ")


def is_water_related(feature_keys: set[str]) -> bool:
    return any("rain" in key or "humidity" in key or "irrigation" in key for key in feature_keys)


def is_heat_related(feature_keys: set[str]) -> bool:
    return any("temp" in key or "heat" in key for key in feature_keys)


def is_nutrient_related(feature_keys: set[str]) -> bool:
    nutrient_tokens = {"ph", "p_h", "soil", "nitrogen", "phosphorus", "potassium", "n", "p", "k"}
    return any(key in nutrient_tokens or key.startswith("soil") for key in feature_keys)


def contains_any(text: str, candidates: list[str]) -> bool:
    normalized_text = str(text).casefold()
    return any(str(candidate).casefold() in normalized_text for candidate in candidates if str(candidate).strip())


def contains_disallowed_phrase(text: str, feature_changes: dict) -> bool:
    normalized_text = str(text).casefold()
    for phrase in DISALLOWED_UNOBSERVED_PHRASES:
        if phrase in normalized_text:
            return True

    normalized_feature_names = {
        normalize_feature_key(feature_name) for feature_name in feature_changes.keys()
    }
    if "soil" not in normalized_feature_names and "soil " in normalized_text:
        if "soil ph" not in normalized_text and "soil health" not in normalized_text:
            return True
    return False
