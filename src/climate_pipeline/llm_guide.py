from __future__ import annotations

import json
import os
from typing import Any

import requests

DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TIMEOUT_SECONDS = 20.0
SUPPORTED_GUIDE_LANGUAGES = ["English", "Hindi", "Marathi", "Kannada", "Telugu", "Tamil"]


class LlmGuideError(RuntimeError):
    """Base exception for the optional LLM guide layer."""


class LlmGuideNotConfiguredError(LlmGuideError):
    """Raised when the Groq client is requested without an API key."""


class LlmGuideUpstreamError(LlmGuideError):
    """Raised when Groq returns an error or empty response."""


class GroqGuideClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        session: requests.Session | Any | None = None,
    ) -> None:
        self.api_key = (api_key or os.getenv("GROQ_API_KEY") or "").strip()
        self.model = (model or os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL).strip()
        self.base_url = (base_url or os.getenv("GROQ_BASE_URL") or DEFAULT_GROQ_BASE_URL).rstrip("/")
        self.timeout_seconds = float(
            timeout_seconds or os.getenv("GROQ_TIMEOUT_SECONDS") or DEFAULT_TIMEOUT_SECONDS
        )
        self.session = session or requests.Session()

    @classmethod
    def from_env(cls) -> "GroqGuideClient":
        return cls()

    def is_enabled(self) -> bool:
        return bool(self.api_key)

    def support_metadata(self) -> dict[str, Any]:
        enabled = self.is_enabled()
        return {
            "enabled": enabled,
            "provider": "groq",
            "model": self.model,
            "api_style": "responses",
            "interactive_guide": True,
            "status": "configured" if enabled else "needs_groq_api_key",
            "supported_languages": SUPPORTED_GUIDE_LANGUAGES,
            "quality_note": "LLM answers are generated from the app's prediction payload and should be checked before field action.",
        }

    def generate_answer(
        self,
        *,
        prediction: dict[str, Any],
        input_snapshot: dict[str, Any] | None = None,
        preferred_language: str | None = None,
        user_question: str | None = None,
        region: str | None = None,
        state: str | None = None,
    ) -> dict[str, Any]:
        if not self.is_enabled():
            raise LlmGuideNotConfiguredError(
                "Groq AI guide is not configured. Set GROQ_API_KEY to enable it."
            )

        prompt = build_guide_prompt(
            prediction=prediction,
            input_snapshot=input_snapshot,
            preferred_language=preferred_language,
            user_question=user_question,
            region=region,
            state=state,
        )
        payload = {
            "model": self.model,
            "instructions": (
                "You are a careful crop guidance explainer for first-time farmers. "
                "Use only the provided prediction data. Do not claim live weather access, "
                "do not guarantee yield or income, and do not present this as a final planting order. "
                "Keep the answer short, practical, and beginner-friendly. "
                "Explain terms in layman language. Prefer three short sections with simple headings: "
                "'What this means', 'Why it came first', and 'What to check next'. "
                "If there are warnings, mention them in plain words. "
                "If a non-English language is requested, answer fully in that language when possible. "
                "Otherwise, use simple English."
            ),
            "input": prompt,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = self.session.post(
                f"{self.base_url}/responses",
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            detail = ""
            if getattr(exc, "response", None) is not None:
                detail = _safe_text(getattr(exc.response, "text", ""), 400)
            raise LlmGuideUpstreamError(
                f"Groq request failed. {detail}".strip()
            ) from exc

        response_payload = response.json()
        answer = extract_output_text(response_payload)
        if not answer:
            raise LlmGuideUpstreamError("Groq returned an empty guide answer.")

        metadata = response_payload.get("metadata", {}) or {}
        return {
            "provider": "groq",
            "model": self.model,
            "source": "groq_responses_api",
            "preferred_language": _safe_text(preferred_language or "English", 32) or "English",
            "answer": answer,
            "disclaimer": (
                "AI-generated explanation based only on this app's prediction details. "
                "Please verify with local agronomy advice before planting."
            ),
            "upstream_total_time_s": metadata.get("total_time"),
        }


def build_guide_prompt(
    *,
    prediction: dict[str, Any],
    input_snapshot: dict[str, Any] | None,
    preferred_language: str | None,
    user_question: str | None,
    region: str | None,
    state: str | None,
) -> str:
    recommendations = prediction.get("recommendations", [])[:3]
    top_features = prediction.get("top_features", [])[:4]
    warnings = prediction.get("warnings", [])[:6]
    why_not = prediction.get("why_not", [])[:3]
    localized_context = prediction.get("localized_context", {}) or {}
    guidance_scope = prediction.get("guidance_scope", {}) or {}
    input_features = ((input_snapshot or {}).get("features") or {}) if isinstance(input_snapshot, dict) else {}

    lines = [
        f"Preferred language: {_safe_text(preferred_language or 'English', 32) or 'English'}",
        f"Region: {_safe_text(region or localized_context.get('resolved_region') or 'Unknown', 80) or 'Unknown'}",
        f"State: {_safe_text(state or localized_context.get('resolved_state') or 'Unknown', 80) or 'Unknown'}",
        f"Target time: {_safe_text(localized_context.get('target_time') or 'unknown', 32) or 'unknown'}",
        f"Top crop: {_safe_text(_top_crop_name(recommendations), 80) or 'Unknown'}",
        f"Top confidence percent: {round(float(prediction.get('confidence', 0.0) or 0.0) * 100.0, 1)}",
        f"Farmer message: {_safe_text(prediction.get('farmer_message'), 280) or 'Not provided'}",
        f"Rule explanation: {_safe_text(prediction.get('explanation'), 280) or 'Not provided'}",
        f"Recommended use: {_safe_text(guidance_scope.get('recommended_use'), 220) or 'Use as shortlist guidance only.'}",
        f"Context source: {_safe_text(localized_context.get('data_source') or localized_context.get('match_level') or 'unknown', 80) or 'unknown'}",
        "Top crop ranking:",
    ]
    for item in recommendations:
        lines.append(
            f"- {_safe_text(item.get('crop'), 80) or 'Unknown'}: {round(float(item.get('score', 0.0) or 0.0) * 100.0, 2)}"
        )

    if top_features:
        lines.append("Main influences:")
        for item in top_features:
            descriptor = _safe_text(item.get("descriptor") or item.get("feature"), 120) or "unknown factor"
            direction = _safe_text(item.get("direction"), 32) or "influenced"
            lines.append(f"- {descriptor}: {direction}")

    if why_not:
        lines.append("Why lower-ranked crops lost:")
        for item in why_not:
            crop_name = _safe_text(item.get("crop"), 80) or "Unknown"
            reason = _safe_text(item.get("reason"), 200) or "No reason provided."
            lines.append(f"- {crop_name}: {reason}")

    if warnings:
        lines.append("Warnings to mention:")
        for warning in warnings:
            lines.append(f"- {_safe_text(warning, 220) or 'Input warning'}")

    if input_features:
        lines.append("User-entered field values:")
        for feature in [
            "temp_avg",
            "rain_total",
            "humidity_avg",
            "pH",
            "N",
            "P",
            "K",
            "irrigation_index",
            "rotation_score",
        ]:
            if feature in input_features:
                lines.append(f"- {feature}: {_safe_text(input_features.get(feature), 64)}")

    question = _safe_text(user_question, 320)
    if question:
        lines.append(f"User question: {question}")
        lines.append(
            "Answer the user's question directly first in layman language, then briefly explain what to check next."
        )
    else:
        lines.append(
            "Please explain this result for a beginner farmer in three short parts with clear headings: "
            "'What this means', 'Why it came first', and 'What to check next'. "
            "Keep the wording simple enough for a first-time farmer."
        )

    return "\n".join(lines)


def extract_output_text(payload: dict[str, Any]) -> str:
    direct = _safe_text(payload.get("output_text"), 4000)
    if direct:
        return direct

    fragments: list[str] = []
    for item in payload.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"}:
                text = _safe_text(content.get("text"), 4000)
                if text:
                    fragments.append(text)
    return "\n".join(fragments).strip()


def _top_crop_name(recommendations: list[dict[str, Any]]) -> str:
    if not recommendations:
        return ""
    return str(recommendations[0].get("crop", "")).replace("_", " ").title()


def _safe_text(value: Any, max_length: int) -> str | None:
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    collapsed = " ".join(text.split())
    if not collapsed:
        return None
    return collapsed[:max_length]
