from __future__ import annotations

from typing import Any

import pandas as pd


SPATIAL_LEVEL_HINTS = {
    "district": "district",
    "district_name": "district",
    "state": "state",
    "state_name": "state",
    "block": "block",
    "tehsil": "tehsil",
    "taluk": "taluk",
    "village": "village",
}


def inspect_dataset(
    dataset_name: str,
    frame: pd.DataFrame,
    dataset_cfg: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "schema": [
            {"column": column, "dtype": str(frame[column].dtype)}
            for column in frame.columns
        ],
        "missing_values": {column: int(value) for column, value in frame.isna().sum().items()},
        "missing_value_percentage": {
            column: round(float(value), 2)
            for column, value in frame.isna().mean().mul(100).items()
        },
        "duplicate_rows": int(frame.duplicated().sum()),
        "time_resolution": infer_time_resolution(frame, dataset_cfg),
        "spatial_resolution": infer_spatial_resolution(frame, dataset_cfg),
    }


def infer_time_resolution(frame: pd.DataFrame, dataset_cfg: dict[str, Any]) -> str:
    columns = dataset_cfg.get("columns", {})

    if "date" in columns and columns["date"] in frame.columns:
        parsed = pd.to_datetime(
            frame[columns["date"]],
            errors="coerce",
            dayfirst=dataset_cfg.get("dayfirst", False),
        ).dropna()
        parsed = parsed.sort_values().drop_duplicates()
        if len(parsed) < 2:
            return "date-like"

        diffs = parsed.diff().dropna().dt.days
        if diffs.empty:
            return "date-like"

        min_gap = float(diffs.min())
        if min_gap <= 1:
            return "daily"
        if min_gap <= 31:
            return "monthly"
        if min_gap <= 92:
            return "seasonal_or_quarterly"
        if min_gap <= 366:
            return "yearly"
        return "irregular"

    if "year" in columns and "month" in columns:
        return "monthly"
    if "year" in columns and "season" in columns:
        return "seasonal"
    if "year" in columns:
        return "yearly"

    return "unknown"


def infer_spatial_resolution(frame: pd.DataFrame, dataset_cfg: dict[str, Any]) -> str:
    if dataset_cfg.get("spatial_level"):
        return str(dataset_cfg["spatial_level"])

    lowered_columns = {column.casefold(): column for column in frame.columns}
    for hint, level in SPATIAL_LEVEL_HINTS.items():
        if hint in lowered_columns:
            return level
    return "unknown"
