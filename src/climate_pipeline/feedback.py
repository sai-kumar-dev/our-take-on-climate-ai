from __future__ import annotations

import hashlib
import hmac
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from .utils import ensure_parent_dir

EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d\-\s]{7,}\d)(?!\w)")
URL_PATTERN = re.compile(r"\bhttps?://\S+\b", re.IGNORECASE)


class FeedbackStore:
    """Append-only feedback storage for future supervised or RL-style learning loops."""

    def __init__(self, storage_dir: Path, signing_secret: str | None = None) -> None:
        self.storage_dir = storage_dir
        self.signing_secret = signing_secret.encode("utf-8") if signing_secret else None
        self._lock = Lock()

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        feedback_id = uuid4().hex
        submitted_at = datetime.now(timezone.utc).isoformat()
        record = {
            "schema_version": 2,
            "feedback_id": feedback_id,
            "submitted_at": submitted_at,
            **sanitize_feedback_payload(payload),
        }

        canonical_payload = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        record["record_hash"] = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
        if self.signing_secret:
            record["integrity_signature"] = hmac.new(
                self.signing_secret,
                canonical_payload.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()

        storage_path = self.storage_dir / f"feedback-{submitted_at[:10]}.jsonl"
        ensure_parent_dir(storage_path)
        with self._lock:
            with storage_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")

        return {
            "feedback_id": feedback_id,
            "submitted_at": submitted_at,
            "storage_file": storage_path.name,
            "integrity_protected": bool(self.signing_secret),
            "training_consent_recorded": bool(record.get("consent_for_training", False)),
        }

    def get_storage_info(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "storage_mode": "append_only_jsonl",
            "collects_contact_details": False,
            "integrity_protected": bool(self.signing_secret),
            "consent_flag_supported": True,
            "pii_redaction_enabled": True,
            "training_use_mode": "offline_human_review_only",
            "human_review_required": True,
        }


def sanitize_feedback_payload(payload: dict[str, Any]) -> dict[str, Any]:
    consent_for_training = bool(payload.get("consent_for_training", False))
    return {
        "request_id": _sanitize_text(payload.get("request_id"), 128),
        "region": _sanitize_text(payload.get("region"), 128),
        "state": _sanitize_text(payload.get("state"), 128),
        "preferred_language": _sanitize_text(payload.get("preferred_language"), 64),
        "selected_crop": _sanitize_text(payload.get("selected_crop"), 128),
        "actual_crop": _sanitize_text(payload.get("actual_crop"), 128),
        "outcome_label": _sanitize_text(payload.get("outcome_label"), 64),
        "helpfulness_rating": _sanitize_int(payload.get("helpfulness_rating"), minimum=1, maximum=5),
        "clarity_rating": _sanitize_int(payload.get("clarity_rating"), minimum=1, maximum=5),
        "consent_for_training": consent_for_training,
        "training_use_status": "pending_human_review" if consent_for_training else "consent_not_granted",
        "eligible_for_training": False,
        "review_status": "pending_human_review" if consent_for_training else "not_requested",
        "comment": _sanitize_text(payload.get("comment"), 1200),
        "input_snapshot": _sanitize_json_like(payload.get("input_snapshot"), depth=0),
        "prediction_snapshot": _sanitize_json_like(payload.get("prediction_snapshot"), depth=0),
    }


def _sanitize_json_like(value: Any, depth: int) -> Any:
    if depth >= 3:
        return _sanitize_scalar(value, max_length=240)

    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in list(value.items())[:40]:
            clean_key = _sanitize_text(key, 80)
            if not clean_key:
                continue
            sanitized[clean_key] = _sanitize_json_like(item, depth + 1)
        return sanitized

    if isinstance(value, (list, tuple)):
        return [_sanitize_json_like(item, depth + 1) for item in list(value)[:20]]

    return _sanitize_scalar(value, max_length=240)


def _sanitize_scalar(value: Any, max_length: int) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _sanitize_text(value, max_length)


def _sanitize_text(value: Any, max_length: int) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    text = EMAIL_PATTERN.sub("[redacted-email]", text)
    text = PHONE_PATTERN.sub("[redacted-phone]", text)
    text = URL_PATTERN.sub("[redacted-url]", text)
    return text[:max_length]


def _sanitize_int(value: Any, minimum: int, maximum: int) -> int | None:
    try:
        if value is None or value == "":
            return None
        number = int(value)
    except (TypeError, ValueError):
        return None
    return max(minimum, min(maximum, number))
