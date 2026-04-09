from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(root_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (root_dir / path).resolve()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path)
        except ImportError as exc:
            raise ImportError(
                "Excel input requires a pandas Excel engine such as openpyxl."
            ) from exc
    raise ValueError(f"Unsupported file type: {path.suffix}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.casefold().strip()
    text = text.replace("&", " and ")
    text = text.replace(",", " ")
    text = re.sub(r"[\.\(\)\[\]/]+", " ", text)
    text = re.sub(r"[-_]+", " ", text)
    return " ".join(text.split())


def normalize_region_name(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"\b(dist|district)\b", " ", text)
    text = re.sub(r"\b(dt)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def pretty_text(value: str) -> str:
    return " ".join(part.capitalize() for part in value.split())


def slugify(value: Any) -> str:
    normalized = normalize_text(value)
    return normalized.replace("/", "_").replace("-", "_").replace(" ", "_")


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def canonicalize_region(
    region: Any,
    state: Any,
    aliases: dict[str, str] | None = None,
) -> tuple[str, str]:
    aliases = aliases or {}
    region_norm = normalize_region_name(region)
    state_norm = normalize_region_name(state)
    scoped_key = f"{state_norm}|{region_norm}" if state_norm else region_norm
    canonical_region = aliases.get(scoped_key) or aliases.get(region_norm) or region_norm
    canonical_state = aliases.get(state_norm) or state_norm
    return pretty_text(canonical_region), pretty_text(canonical_state)


def canonicalize_region_with_metadata(
    region: Any,
    state: Any,
    aliases: dict[str, str] | None = None,
) -> tuple[str, str, float]:
    aliases = aliases or {}
    region_norm = normalize_region_name(region)
    state_norm = normalize_region_name(state)

    if not region_norm:
        return "", pretty_text(state_norm), 0.0

    scoped_key = f"{state_norm}|{region_norm}" if state_norm else region_norm
    alias_applied = scoped_key in aliases or region_norm in aliases or state_norm in aliases
    canonical_region = aliases.get(scoped_key) or aliases.get(region_norm) or region_norm
    canonical_state = aliases.get(state_norm) or state_norm

    geo_confidence = 0.85
    if canonical_state:
        geo_confidence += 0.10
    if not alias_applied:
        geo_confidence += 0.05

    return pretty_text(canonical_region), pretty_text(canonical_state), min(1.0, geo_confidence)


def make_region_key(region: str, state: str) -> str:
    state_key = normalize_text(state)
    region_key = normalize_text(region)
    return f"{state_key}__{region_key}" if state_key else region_key


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    return value
