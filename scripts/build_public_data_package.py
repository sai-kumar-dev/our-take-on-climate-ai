from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
FULL_DATASET_PATH = ROOT_DIR / "data" / "processed" / "data_new_final_ml_dataset.csv"
OUTPUT_DIR = ROOT_DIR / "data"
SAMPLE_DATASET_PATH = OUTPUT_DIR / "sample_dataset.csv"
DATA_DICTIONARY_PATH = OUTPUT_DIR / "data_dictionary.csv"
SCHEMA_PATH = OUTPUT_DIR / "sample_dataset_schema.json"
SAMPLE_INPUT_DIR = OUTPUT_DIR / "sample_inputs"
DATASET_VERSION_PATH = ROOT_DIR / "dataset_version.json"
MANIFEST_PATH = ROOT_DIR / "MANIFEST.yaml"
CHECKSUMS_PATH = ROOT_DIR / "CHECKSUMS.txt"
RELEASE_VERSION = "1.0.0"
RELEASE_TAG = "v1.0.0-public-artifact"
BUILDER_SCRIPT_PATH = Path("scripts/build_public_data_package.py")
ALLOWED_UNTRACKED_RELEASE_PREFIXES = ("data/", "scripts/")
ALLOWED_UNTRACKED_RELEASE_FILES = {
    "LICENSE",
    "NOTICE",
    "CITATION.cff",
    "MANIFEST.yaml",
    "dataset_version.json",
    "CHECKSUMS.txt",
}
RELEASE_EXCLUDED_PATHS = {
    "data/processed/final_ml_dataset.csv",
}
DATA_NOTICE_TEXT = (
    "This representative subset is provided for reproducibility demonstration. "
    "Users should consult original upstream sources for full source licensing."
)


def relpath(path: Path) -> str:
    return path.relative_to(ROOT_DIR).as_posix()


MODEL_INPUT_FEATURES = [
    "temp_avg",
    "rain_total",
    "humidity_avg",
    "rain_variance",
    "max_temp",
    "max_temp_3d",
    "max_rain_1d",
    "dry_spell_days",
    "temp_lag_7",
    "rain_lag_14",
    "pH",
    "N",
    "P",
    "K",
    "soil_health_index",
    "N_class",
    "P_class",
    "K_class",
    "fertility_class",
    "state_context",
    "region_context",
    "target_month",
    "target_season",
]

CATEGORICAL_INPUT_FEATURES = {
    "N_class",
    "P_class",
    "K_class",
    "fertility_class",
    "state_context",
    "region_context",
    "target_month",
    "target_season",
}


FIXED_COLUMN_METADATA: dict[str, dict[str, str]] = {
    "region": {
        "description": "Human-readable district or region name used in the final ML-ready table.",
        "unit": "text",
        "source": "Merged district-level records after name canonicalization.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Canonicalized district labels aligned across climate, soil, and crop inputs.",
        "role": "identifier",
    },
    "state": {
        "description": "State name associated with the district row.",
        "unit": "text",
        "source": "Merged district-level records after name canonicalization.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Used as fallback grouping key during imputation and for slice analysis.",
        "role": "identifier",
    },
    "region_key": {
        "description": "Stable internal district identifier combining normalized state and district tokens.",
        "unit": "text",
        "source": "Derived key from canonicalized district and state names.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Primary join key across prepared climate, soil, and crop tables.",
        "role": "identifier",
    },
    "time": {
        "description": "Monthly district-time key in YYYY-MM format.",
        "unit": "YYYY-MM",
        "source": "Derived temporal index after alignment to monthly granularity.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Climate and crop sources are aligned to the same monthly period key.",
        "role": "identifier",
    },
    "temp_avg": {
        "description": "Average temperature for the district-month.",
        "unit": "degC",
        "source": "Prepared climate table built from weather data described in project references.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Aggregated from daily observations to a monthly mean.",
        "role": "feature",
    },
    "rain_total": {
        "description": "Total rainfall accumulated over the district-month.",
        "unit": "mm",
        "source": "Prepared climate table built from district rainfall data.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Aggregated from daily rainfall observations to a monthly sum.",
        "role": "feature",
    },
    "humidity_avg": {
        "description": "Average relative humidity for the district-month.",
        "unit": "percent",
        "source": "Prepared climate table built from weather data described in project references.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Aggregated from daily humidity observations to a monthly mean.",
        "role": "feature",
    },
    "rain_variance": {
        "description": "Variance of daily rainfall within the district-month.",
        "unit": "mm^2",
        "source": "Prepared climate table built from district rainfall data.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Computed from daily rainfall after temporal aggregation.",
        "role": "feature",
    },
    "max_temp": {
        "description": "Maximum daily temperature observed within the district-month.",
        "unit": "degC",
        "source": "Prepared climate table built from weather data described in project references.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Monthly maximum over daily maximum temperature values.",
        "role": "feature",
    },
    "max_temp_3d": {
        "description": "Peak 3-step rolling average of maximum temperature within the district-month.",
        "unit": "degC",
        "source": "Prepared climate table built from weather data described in project references.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Computed as the maximum rolling mean over the daily max temperature series.",
        "role": "feature",
    },
    "max_rain_1d": {
        "description": "Maximum single-day rainfall observed within the district-month.",
        "unit": "mm/day",
        "source": "Prepared climate table built from district rainfall data.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Monthly maximum over daily rainfall totals.",
        "role": "feature",
    },
    "dry_spell_days": {
        "description": "Longest run of low-rainfall days within the district-month.",
        "unit": "days",
        "source": "Prepared climate table built from district rainfall data.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Counts the longest consecutive period with rainfall at or below the configured dry-spell threshold.",
        "role": "feature",
    },
    "temp_lag_7": {
        "description": "Coarse lag proxy for recent temperature conditions.",
        "unit": "degC",
        "source": "Prepared climate table built from weather data described in project references.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "After monthly aggregation, the previous period temperature is reused as a compatibility proxy for a shorter lag.",
        "role": "feature",
    },
    "rain_lag_14": {
        "description": "Coarse lag proxy for recent rainfall conditions.",
        "unit": "mm",
        "source": "Prepared climate table built from district rainfall data.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "After monthly aggregation, the previous period rainfall total is reused as a compatibility proxy for a shorter lag.",
        "role": "feature",
    },
    "pH": {
        "description": "Soil acidity or alkalinity indicator for the district.",
        "unit": "pH",
        "source": "Prepared soil table derived from soil property sources described in project references.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Region-level mean after spatial alignment and numeric imputation.",
        "role": "feature",
    },
    "N": {
        "description": "District-level nitrogen proxy used by the model.",
        "unit": "kg/ha proxy",
        "source": "Prepared soil table derived from soil property sources described in project references.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "The internal prep report states these N values are proxy estimates rather than direct observed lab N measurements.",
        "role": "feature",
    },
    "P": {
        "description": "District-level phosphorus proxy used by the model.",
        "unit": "kg/ha proxy",
        "source": "Prepared soil table derived from soil property sources described in project references.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "The internal prep report states these P values are proxy estimates rather than direct observed lab P measurements.",
        "role": "feature",
    },
    "K": {
        "description": "District-level potassium proxy used by the model.",
        "unit": "kg/ha proxy",
        "source": "Prepared soil table derived from soil property sources described in project references.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "The internal prep report states these K values are proxy estimates rather than direct observed lab K measurements.",
        "role": "feature",
    },
    "N_class": {
        "description": "Categorical nitrogen status bucket.",
        "unit": "category",
        "source": "Derived from N using configured nutrient bins.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Binned into low, medium, or high categories before modeling.",
        "role": "feature",
    },
    "P_class": {
        "description": "Categorical phosphorus status bucket.",
        "unit": "category",
        "source": "Derived from P using configured nutrient bins.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Binned into low, medium, or high categories before modeling.",
        "role": "feature",
    },
    "K_class": {
        "description": "Categorical potassium status bucket.",
        "unit": "category",
        "source": "Derived from K using configured nutrient bins.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Binned into low, medium, or high categories before modeling.",
        "role": "feature",
    },
    "soil_health_index": {
        "description": "Simple composite soil support score used in the ML table.",
        "unit": "0-100 index",
        "source": "Derived from pH and N/P/K proxies.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Code comments explicitly describe this as a simple composite baseline, not an official scientific soil health card metric.",
        "role": "feature",
    },
    "irrigation_index": {
        "description": "Bounded management-context feature representing relative irrigation support.",
        "unit": "0-1 index",
        "source": "Context defaults plus heuristic inference from climate and soil conditions.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "When not available from context defaults, inferred heuristically from rainfall, dry spells, humidity, heat, and soil support signals.",
        "role": "feature",
    },
    "rotation_score": {
        "description": "Bounded management-context feature representing crop-rotation suitability.",
        "unit": "0-1 index",
        "source": "Context defaults plus heuristic inference from soil and climate conditions.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Heuristic score built from soil health, nutrient balance, pH balance, and humidity when no direct value is available.",
        "role": "feature",
    },
    "fertility_class": {
        "description": "Categorical fertility summary token used in modeling and UI defaults.",
        "unit": "category",
        "source": "Context defaults and derived soil support context.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Filled from configured defaults when not explicitly available.",
        "role": "feature",
    },
    "state_context": {
        "description": "Normalized state token retained as a model context feature.",
        "unit": "text",
        "source": "Derived from state.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Lowercased string field used as categorical context during training and inference.",
        "role": "feature",
    },
    "region_context": {
        "description": "Normalized region token retained as a model context feature.",
        "unit": "text",
        "source": "Derived from region_key.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Lowercased stable region identifier used as categorical context during training and inference.",
        "role": "feature",
    },
    "target_month": {
        "description": "Zero-padded month token extracted from the district-time key.",
        "unit": "MM",
        "source": "Derived from the aligned time index.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Stored in the released CSV text as a zero-padded string token from 01 to 12.",
        "role": "feature",
    },
    "target_season": {
        "description": "Season label derived from the target month.",
        "unit": "category",
        "source": "Derived from the aligned time index.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Mapped to kharif, rabi, or zaid using the project season lookup.",
        "role": "feature",
    },
    "time_step_missing": {
        "description": "Indicator for rows introduced to complete missing district-time periods.",
        "unit": "0/1 flag",
        "source": "Dataset quality metadata.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Set during time-grid completion before climate imputation.",
        "role": "quality",
    },
    "climate_gap_filled": {
        "description": "Indicator that climate values were filled or imputed for the row.",
        "unit": "0/1 flag",
        "source": "Dataset quality metadata.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Tracks climate gaps originating in raw series or introduced during merge alignment.",
        "role": "quality",
    },
    "soil_imputed": {
        "description": "Indicator that at least one soil field required imputation.",
        "unit": "0/1 flag",
        "source": "Dataset quality metadata.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Triggered when soil values are missing and filled from district, state, or global summaries.",
        "role": "quality",
    },
    "geo_confidence": {
        "description": "Confidence score for the geographic alignment used during source joins.",
        "unit": "0-1 score",
        "source": "Geographic canonicalization and merge metadata.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Average of available geo-confidence signals from prepared source tables.",
        "role": "quality",
    },
    "data_confidence": {
        "description": "Composite quality score summarizing interpolation and imputation burden for the row.",
        "unit": "0-1 score",
        "source": "Dataset quality metadata.",
        "engineered_or_raw": "engineered",
        "preprocessing_notes": "Computed from climate interpolation, soil imputation, missing time-step, and merge-imputation penalties.",
        "role": "quality",
    },
}


def crop_label_to_name(column: str) -> str:
    crop = column.removeprefix("crop_prob_").replace("_", " ")
    return crop


def crop_column_metadata(column: str) -> dict[str, str]:
    crop_name = crop_label_to_name(column)
    return {
        "description": f"Normalized district-time probability target for {crop_name}.",
        "unit": "probability",
        "source": "Prepared crop table derived from historical area or production records.",
        "engineered_or_raw": "target",
        "preprocessing_notes": "Aggregated by district, month, and crop, then row-normalized so all crop_prob_* targets sum to 1.",
        "role": "target",
    }


def build_column_metadata(columns: list[str]) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for column in columns:
        if column in FIXED_COLUMN_METADATA:
            metadata[column] = FIXED_COLUMN_METADATA[column]
        elif column.startswith("crop_prob_"):
            metadata[column] = crop_column_metadata(column)
        else:
            metadata[column] = {
                "description": "Project feature with no public mapping note yet.",
                "unit": "unknown",
                "source": "unknown",
                "engineered_or_raw": "engineered",
                "preprocessing_notes": "Review manually before release.",
                "role": "feature",
            }
    return metadata


def select_sample_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    label_cols = [column for column in df.columns if column.startswith("crop_prob_")]
    df = df.copy()
    df["top_crop"] = df[label_cols].idxmax(axis=1)

    selected_indices: set[int] = set()

    # Coverage anchor 1: every state-season combination with rainfall extremes.
    for (_, _), group in df.groupby(["state", "target_season"], dropna=False):
        low = group.sort_values(
            ["rain_total", "data_confidence", "soil_health_index", "region_key", "time"],
            ascending=[True, False, False, True, True],
        )
        high = group.sort_values(
            ["rain_total", "data_confidence", "soil_health_index", "region_key", "time"],
            ascending=[False, False, False, True, True],
        )
        selected_indices.add(int(low.index[0]))
        selected_indices.add(int(high.index[0]))

    # Coverage anchor 2: each crop target appears with at least one positive exemplar.
    for column in label_cols:
        selected_indices.add(int(df[column].idxmax()))

    # Coverage anchor 3: each state contributes a low-confidence row.
    for _, group in df.groupby("state", dropna=False):
        candidate = group.sort_values(
            ["data_confidence", "soil_imputed", "climate_gap_filled", "region_key", "time"],
            ascending=[True, False, False, True, True],
        )
        selected_indices.add(int(candidate.index[0]))

    sample = df.loc[sorted(selected_indices)].drop(columns=["top_crop"]).copy()
    sample = sample.sort_values(["state", "region", "time"]).reset_index(drop=True)

    stats = {
        "selection_strategy": {
            "state_season_extremes": "Per state and season, include lowest- and highest-rainfall rows with quality-aware tie breaks.",
            "crop_exemplars": "For every crop_prob_* target, include the row with the maximum observed probability.",
            "quality_exemplars": "For every state, include the lowest-confidence row to preserve imputation and quality variation.",
            "deduplication": "Union the selected rows and remove duplicates deterministically.",
        },
        "row_count": int(len(sample)),
        "state_count": int(sample["state"].nunique()),
        "region_count": int(sample["region_key"].nunique()),
        "time_min": str(sample["time"].min()),
        "time_max": str(sample["time"].max()),
        "season_counts": {str(key): int(value) for key, value in sample["target_season"].value_counts().to_dict().items()},
        "soil_imputed_counts": {str(key): int(value) for key, value in sample["soil_imputed"].value_counts().to_dict().items()},
        "data_confidence_range": {
            "min": float(sample["data_confidence"].min()),
            "max": float(sample["data_confidence"].max()),
        },
        "positive_target_coverage": int(sum(int((sample[column] > 0).any()) for column in label_cols)),
        "target_column_count": int(len(label_cols)),
    }
    return sample, stats


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    metadata = build_column_metadata(df.columns.tolist())
    records = []
    for column in df.columns:
        item = metadata[column]
        records.append(
            {
                "feature_name": column,
                "description": item["description"],
                "unit": item["unit"],
                "source": item["source"],
                "engineered_or_raw": item["engineered_or_raw"],
                "preprocessing_notes": item["preprocessing_notes"],
            }
        )
    return pd.DataFrame(records)


def normalize_public_frame(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "target_month" in normalized.columns:
        month_tokens = (
            normalized["target_month"]
            .astype("string")
            .str.extract(r"(\d{1,2})", expand=False)
            .str.zfill(2)
        )
        normalized["target_month"] = month_tokens
    return normalized


def crop_names_from_columns(columns: list[str]) -> list[str]:
    return [crop_label_to_name(column) for column in columns]


def dominant_crop_names(frame: pd.DataFrame, label_cols: list[str]) -> list[str]:
    if frame.empty or not label_cols:
        return []
    dominant = frame[label_cols].idxmax(axis=1).dropna().unique().tolist()
    return sorted(crop_label_to_name(column) for column in dominant)


def build_coverage_summary(
    full_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    label_cols: list[str],
) -> dict[str, Any]:
    full_state_season = full_df[["state", "target_season"]].drop_duplicates().shape[0]
    sample_state_season = sample_df[["state", "target_season"]].drop_duplicates().shape[0]
    full_months = sorted(full_df["time"].astype("string").dropna().unique().tolist())
    sample_months = sorted(sample_df["time"].astype("string").dropna().unique().tolist())
    positive_target_columns = [column for column in label_cols if bool(sample_df[column].gt(0).any())]

    return {
        "states": {
            "sample_count": int(sample_df["state"].nunique()),
            "full_count": int(full_df["state"].nunique()),
            "fraction": f"{int(sample_df['state'].nunique())}/{int(full_df['state'].nunique())}",
            "note": "All states present in the full final table appear in the released subset.",
        },
        "state_season_groups": {
            "sample_count": int(sample_state_season),
            "full_count": int(full_state_season),
            "fraction": f"{int(sample_state_season)}/{int(full_state_season)}",
            "note": "State-season coverage is complete even though district and month coverage are partial.",
        },
        "regions": {
            "sample_count": int(sample_df["region_key"].nunique()),
            "full_count": int(full_df["region_key"].nunique()),
            "fraction": f"{int(sample_df['region_key'].nunique())}/{int(full_df['region_key'].nunique())}",
            "note": "Region coverage is intentionally partial. The subset is a demo artifact, not a national district release.",
        },
        "months": {
            "sample_count": int(len(sample_months)),
            "full_count": int(len(full_months)),
            "fraction": f"{int(len(sample_months))}/{int(len(full_months))}",
            "note": "Month coverage is intentionally partial and should not be described as a complete temporal reconstruction.",
            "missing_from_sample": [month for month in full_months if month not in sample_months],
        },
        "dominant_crops": {
            "sample_count": int(len(dominant_crop_names(sample_df, label_cols))),
            "full_count": int(len(dominant_crop_names(full_df, label_cols))),
            "fraction": f"{int(len(dominant_crop_names(sample_df, label_cols)))}/{int(len(dominant_crop_names(full_df, label_cols)))}",
            "note": "Counts crops that appear as the row-wise top target at least once.",
            "values": dominant_crop_names(sample_df, label_cols),
        },
        "positive_target_columns": {
            "sample_count": int(len(positive_target_columns)),
            "full_count": int(len(label_cols)),
            "fraction": f"{int(len(positive_target_columns))}/{int(len(label_cols))}",
            "note": "Counts target columns that are positive at least once, regardless of whether they are ever dominant.",
            "values": crop_names_from_columns(positive_target_columns),
        },
    }


def build_quality_variation_summary(sample_df: pd.DataFrame) -> dict[str, Any]:
    def unique_values(column: str) -> list[Any]:
        values = sample_df[column].dropna().unique().tolist()
        return sorted(values)

    return {
        "target_seasons_present": sorted(sample_df["target_season"].dropna().astype("string").unique().tolist()),
        "soil_imputed_values": unique_values("soil_imputed"),
        "climate_gap_filled_values": unique_values("climate_gap_filled"),
        "time_step_missing_values": unique_values("time_step_missing"),
        "geo_confidence_range": {
            "min": float(sample_df["geo_confidence"].min()),
            "max": float(sample_df["geo_confidence"].max()),
        },
        "data_confidence_range": {
            "min": float(sample_df["data_confidence"].min()),
            "max": float(sample_df["data_confidence"].max()),
        },
        "note": (
            "Observed quality variation in the released subset is limited. "
            "Rows include both soil-imputed and non-imputed cases, but do not include positive "
            "time_step_missing or climate_gap_filled examples."
        ),
    }


def build_role_counts(columns: list[str]) -> dict[str, int]:
    metadata = build_column_metadata(columns)
    counts = {
        "identifier_column_count": 0,
        "feature_column_count": 0,
        "quality_column_count": 0,
        "target_column_count": 0,
    }
    for column in columns:
        role = metadata[column]["role"]
        if role == "identifier":
            counts["identifier_column_count"] += 1
        elif role == "feature":
            counts["feature_column_count"] += 1
        elif role == "quality":
            counts["quality_column_count"] += 1
        elif role == "target":
            counts["target_column_count"] += 1
    return counts


def build_schema(
    full_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    sampling_stats: dict[str, Any],
) -> dict[str, Any]:
    metadata = build_column_metadata(full_df.columns.tolist())
    label_cols = [column for column in full_df.columns if column.startswith("crop_prob_")]
    coverage_summary = build_coverage_summary(full_df, sample_df, label_cols)
    quality_variation = build_quality_variation_summary(sample_df)
    role_counts = build_role_counts(full_df.columns.tolist())
    columns = []
    for column in full_df.columns:
        item = metadata[column]
        columns.append(
            {
                "name": column,
                "dtype": "string" if column == "target_month" else str(full_df[column].dtype),
                "nullable_in_full_dataset": bool(full_df[column].isna().any()),
                "nullable_in_sample_dataset": bool(sample_df[column].isna().any()),
                "role": item["role"],
                "unit": item["unit"],
                "description": item["description"],
                "source": item["source"],
                "engineered_or_raw": item["engineered_or_raw"],
                "preprocessing_notes": item["preprocessing_notes"],
            }
        )

    return {
        "dataset_name": "Representative Reproducibility Subset",
        "dataset_version": RELEASE_VERSION,
        "release_tag": RELEASE_TAG,
        "schema_compatible_with_full_dataset": True,
        "public_release_boundary": {
            "artifact_type": "demo artifact",
            "description": (
                "Representative reproducibility subset with a documented reconstruction pathway. "
                "The subset alone is not a claim of complete empirical reproducibility for the full study."
            ),
            "licensing_notice": DATA_NOTICE_TEXT,
        },
        "full_dataset": {
            "path": relpath(FULL_DATASET_PATH),
            "availability": "Not shipped in the public release. Local build source path only.",
            "row_count": int(len(full_df)),
            "column_count": int(len(full_df.columns)),
            "state_count": int(full_df["state"].nunique()),
            "region_count": int(full_df["region_key"].nunique()),
            "time_step_count": int(full_df["time"].nunique()),
            "time_min": str(full_df["time"].min()),
            "time_max": str(full_df["time"].max()),
            "target_column_count": int(len(label_cols)),
        },
        "sample_dataset": {
            "path": relpath(SAMPLE_DATASET_PATH),
            "availability": "Shipped in the public release.",
            "row_count": int(len(sample_df)),
            "column_count": int(len(sample_df.columns)),
            "state_count": int(sample_df["state"].nunique()),
            "region_count": int(sample_df["region_key"].nunique()),
            "time_step_count": int(sample_df["time"].nunique()),
            "time_min": str(sample_df["time"].min()),
            "time_max": str(sample_df["time"].max()),
        },
        "role_counts": role_counts,
        "coverage_summary": coverage_summary,
        "quality_variation": quality_variation,
        "scope_decision": (
            "No extra anchor rows were added for release v1.0.0-public-artifact. "
            "The existing 211-row subset already covers all states, all state-season groups, "
            "all dominant crops, and all target columns while remaining explicitly partial for regions and months."
        ),
        "sampling_stats": sampling_stats,
        "columns": columns,
    }


def build_sample_inputs(sample_df: pd.DataFrame) -> list[dict[str, Any]]:
    selectors = [
        (
            "01_high_rainfall_kharif.json",
            sample_df.sort_values(
                ["rain_total", "data_confidence", "region_key", "time"],
                ascending=[False, False, True, True],
            ),
        ),
        (
            "02_high_heat_example.json",
            sample_df.sort_values(
                ["max_temp", "data_confidence", "region_key", "time"],
                ascending=[False, False, True, True],
            ),
        ),
        (
            "03_high_soil_health_example.json",
            sample_df.sort_values(
                ["soil_health_index", "data_confidence", "region_key", "time"],
                ascending=[False, False, True, True],
            ),
        ),
        (
            "04_low_confidence_imputed_example.json",
            sample_df.sort_values(
                ["data_confidence", "soil_imputed", "region_key", "time"],
                ascending=[True, False, True, True],
            ),
        ),
        (
            "05_high_irrigation_example.json",
            sample_df.sort_values(
                ["irrigation_index", "data_confidence", "region_key", "time"],
                ascending=[False, False, True, True],
            ),
        ),
    ]

    selected_rows: list[tuple[str, pd.Series]] = []
    seen_keys: set[tuple[str, str]] = set()
    for filename, ordered in selectors:
        chosen_row = None
        for _, row in ordered.iterrows():
            key = (str(row["region_key"]), str(row["time"]))
            if key in seen_keys:
                continue
            chosen_row = row
            seen_keys.add(key)
            break
        if chosen_row is None:
            chosen_row = ordered.iloc[0]
        selected_rows.append((filename, chosen_row))
    return [
        {
            "filename": filename,
            "payload": row_to_predict_request(row),
        }
        for filename, row in selected_rows
    ]


def row_to_predict_request(row: pd.Series) -> dict[str, Any]:
    def clean_value(value: Any, *, feature_name: str | None = None) -> Any:
        if pd.isna(value):
            return None
        if feature_name in CATEGORICAL_INPUT_FEATURES:
            if feature_name == "target_month":
                return str(value).zfill(2)
            return str(value)
        if isinstance(value, (float, int)):
            return float(value) if isinstance(value, float) else int(value)
        return value

    features = {
        feature: clean_value(row[feature], feature_name=feature)
        for feature in MODEL_INPUT_FEATURES
        if feature in row.index
    }
    return {
        "region": clean_value(row["region"]),
        "state": clean_value(row["state"]),
        "target_time": clean_value(row["time"]),
        "irrigation_index": clean_value(row["irrigation_index"]),
        "rotation_score": clean_value(row["rotation_score"]),
        "features": features,
    }


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def list_release_files(*, exclude: set[str] | None = None) -> list[Path]:
    excluded = {item.replace("\\", "/") for item in (exclude or set())}
    excluded.update(RELEASE_EXCLUDED_PATHS)
    try:
        tracked_result = subprocess.run(
            ["git", "ls-files", "--cached"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        untracked_result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        tracked_candidates = {
            Path(line.strip()).as_posix()
            for line in tracked_result.stdout.splitlines()
            if line.strip()
        }
        untracked_candidates = {
            Path(line.strip()).as_posix()
            for line in untracked_result.stdout.splitlines()
            if line.strip()
        }
        allowed_untracked = {
            path
            for path in untracked_candidates
            if path in ALLOWED_UNTRACKED_RELEASE_FILES
            or path.startswith(ALLOWED_UNTRACKED_RELEASE_PREFIXES)
        }
        candidate_paths = sorted(tracked_candidates | allowed_untracked)
    except (OSError, subprocess.CalledProcessError):
        candidate_paths = [
            path.relative_to(ROOT_DIR).as_posix()
            for path in ROOT_DIR.rglob("*")
            if path.is_file()
            and ".git" not in path.parts
            and ".venv" not in path.parts
            and "__pycache__" not in path.parts
            and ".tmp" not in path.parts
        ]

    release_files: list[Path] = []
    for relative_path in sorted(candidate_paths):
        if relative_path in excluded:
            continue
        absolute_path = ROOT_DIR / relative_path
        if absolute_path.is_file():
            release_files.append(absolute_path)
    return release_files


def checksum_records(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": relpath(path),
            "sha256": file_sha256(path),
            "bytes": int(path.stat().st_size),
        }
        for path in paths
    ]


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value, ensure_ascii=False)
    return json.dumps(str(value), ensure_ascii=False)


def to_yaml_lines(value: Any, *, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        if not value:
            return [f"{prefix}{{}}"]
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(to_yaml_lines(item, indent=indent + 2))
            else:
                lines.append(f"{prefix}{key}: {yaml_scalar(item)}")
        return lines
    if isinstance(value, list):
        if not value:
            return [f"{prefix}[]"]
        lines = []
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{prefix}-")
                lines.extend(to_yaml_lines(item, indent=indent + 2))
            elif isinstance(item, list):
                lines.append(f"{prefix}-")
                lines.extend(to_yaml_lines(item, indent=indent + 2))
            else:
                lines.append(f"{prefix}- {yaml_scalar(item)}")
        return lines
    return [f"{prefix}{yaml_scalar(value)}"]


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text("\n".join(to_yaml_lines(payload)) + "\n", encoding="utf-8")


def build_dataset_version_payload(
    full_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    schema: dict[str, Any],
) -> dict[str, Any]:
    label_cols = [column for column in full_df.columns if column.startswith("crop_prob_")]
    role_counts = build_role_counts(full_df.columns.tolist())
    coverage_summary = build_coverage_summary(full_df, sample_df, label_cols)
    quality_variation = build_quality_variation_summary(sample_df)
    return {
        "dataset_name": "Representative Reproducibility Subset",
        "version": RELEASE_VERSION,
        "release_tag": RELEASE_TAG,
        "artifact_type": "demo artifact",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "builder_script": BUILDER_SCRIPT_PATH.as_posix(),
        "row_count": int(len(sample_df)),
        "column_count": int(len(sample_df.columns)),
        **role_counts,
        "state_count": int(sample_df["state"].nunique()),
        "district_count": int(sample_df["region_key"].nunique()),
        "month_count": int(sample_df["time"].nunique()),
        "time_min": str(sample_df["time"].min()),
        "time_max": str(sample_df["time"].max()),
        "sample_dataset_path": relpath(SAMPLE_DATASET_PATH),
        "sample_dataset_sha256": file_sha256(SAMPLE_DATASET_PATH),
        "schema_path": relpath(SCHEMA_PATH),
        "schema_sha256": file_sha256(SCHEMA_PATH),
        "coverage_summary": coverage_summary,
        "quality_variation": quality_variation,
        "scope_decision": schema["scope_decision"],
        "licensing_notice": DATA_NOTICE_TEXT,
    }


def build_manifest_payload(
    full_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    schema: dict[str, Any],
    checksums: list[dict[str, Any]],
) -> dict[str, Any]:
    label_cols = [column for column in full_df.columns if column.startswith("crop_prob_")]
    role_counts = build_role_counts(full_df.columns.tolist())
    coverage_summary = build_coverage_summary(full_df, sample_df, label_cols)
    quality_variation = build_quality_variation_summary(sample_df)
    return {
        "manifest_version": 1,
        "release_tag": RELEASE_TAG,
        "artifact_name": "Climate-Aware Crop Recommendation System Public Artifact",
        "artifact_type": "demo artifact",
        "dataset_version": RELEASE_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "builder_script": BUILDER_SCRIPT_PATH.as_posix(),
        "public_release_boundary": {
            "summary": (
                "Representative reproducibility subset with a documented reconstruction pathway."
            ),
            "scope_note": schema["scope_decision"],
            "licensing_notice": DATA_NOTICE_TEXT,
        },
        "dataset": {
            "path": relpath(SAMPLE_DATASET_PATH),
            "row_count": int(len(sample_df)),
            "column_count": int(len(sample_df.columns)),
            **role_counts,
            "states": {
                "count": int(sample_df["state"].nunique()),
                "values": sorted(sample_df["state"].dropna().astype("string").unique().tolist()),
            },
            "districts": {
                "count": int(sample_df["region_key"].nunique()),
                "values": sorted(sample_df["region"].dropna().astype("string").unique().tolist()),
            },
            "months_covered": {
                "count": int(sample_df["time"].nunique()),
                "values": sorted(sample_df["time"].dropna().astype("string").unique().tolist()),
            },
            "time_span": {
                "min": str(sample_df["time"].min()),
                "max": str(sample_df["time"].max()),
            },
            "crop_coverage": {
                "dominant_crops": {
                    "count": int(len(coverage_summary["dominant_crops"]["values"])),
                    "of_full_dataset": int(coverage_summary["dominant_crops"]["full_count"]),
                    "values": coverage_summary["dominant_crops"]["values"],
                },
                "positive_target_columns": {
                    "count": int(coverage_summary["positive_target_columns"]["sample_count"]),
                    "of_full_dataset": int(coverage_summary["positive_target_columns"]["full_count"]),
                    "values": coverage_summary["positive_target_columns"]["values"],
                },
            },
            "coverage_summary": coverage_summary,
            "quality_variation": quality_variation,
        },
        "checksums": {
            "algorithm": "SHA-256",
            "scope": "All release files except MANIFEST.yaml and CHECKSUMS.txt.",
            "files": checksums,
        },
    }


def write_checksums(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [
        f"# SHA-256 checksums for {RELEASE_TAG}",
        "# Scope: all release files except MANIFEST.yaml and CHECKSUMS.txt.",
    ]
    lines.extend(f"{record['sha256']}  {record['path']}" for record in records)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not FULL_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Full dataset not found: "
            f"{FULL_DATASET_PATH}. "
            "The public release does not ship the full processed source table. "
            "Rebuild it locally via run_pipeline.py as described in data/RECONSTRUCTION_GUIDE.md, "
            "then rerun scripts/build_public_data_package.py."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    full_df = normalize_public_frame(pd.read_csv(FULL_DATASET_PATH, dtype={"target_month": "string"}))
    sample_df, sampling_stats = select_sample_dataset(full_df)
    sample_df = normalize_public_frame(sample_df)
    sample_df.to_csv(SAMPLE_DATASET_PATH, index=False)

    data_dictionary = build_data_dictionary(full_df)
    data_dictionary.to_csv(DATA_DICTIONARY_PATH, index=False)

    schema = build_schema(full_df, sample_df, sampling_stats)
    schema["sample_dataset_sha256"] = file_sha256(SAMPLE_DATASET_PATH)
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for item in build_sample_inputs(sample_df):
        output_path = SAMPLE_INPUT_DIR / item["filename"]
        output_path.write_text(json.dumps(item["payload"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    dataset_version_payload = build_dataset_version_payload(full_df, sample_df, schema)
    DATASET_VERSION_PATH.write_text(
        json.dumps(dataset_version_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest_release_files = list_release_files(
        exclude={MANIFEST_PATH.name, CHECKSUMS_PATH.name},
    )
    manifest_payload = build_manifest_payload(
        full_df,
        sample_df,
        schema,
        checksum_records(manifest_release_files),
    )
    write_yaml(MANIFEST_PATH, manifest_payload)

    checksum_release_files = list_release_files(exclude={CHECKSUMS_PATH.name, MANIFEST_PATH.name})
    write_checksums(CHECKSUMS_PATH, checksum_records(checksum_release_files))

    print(
        json.dumps(
            {
                "sample_dataset": relpath(SAMPLE_DATASET_PATH),
                "sample_rows": int(len(sample_df)),
                "data_dictionary": relpath(DATA_DICTIONARY_PATH),
                "schema": relpath(SCHEMA_PATH),
                "dataset_version": relpath(DATASET_VERSION_PATH),
                "manifest": relpath(MANIFEST_PATH),
                "checksums": relpath(CHECKSUMS_PATH),
                "sample_inputs": sorted(relpath(path) for path in SAMPLE_INPUT_DIR.glob("*.json")),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
