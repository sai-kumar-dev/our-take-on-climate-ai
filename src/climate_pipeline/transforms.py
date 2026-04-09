from __future__ import annotations

from typing import Any

import pandas as pd

from .utils import canonicalize_region_with_metadata, coerce_numeric, make_region_key, slugify


DEFAULT_SOIL_BINS = {
    "N": [280, 560],
    "P": [10, 25],
    "K": [110, 280],
}

DEFAULT_CONTEXT_FEATURES = {
    "irrigation_index": 0.0,
    "rotation_score": 0.5,
    "fertility_class": "medium",
}
MONTH_TO_SEASON = {
    1: "rabi",
    2: "rabi",
    3: "rabi",
    4: "zaid",
    5: "zaid",
    6: "kharif",
    7: "kharif",
    8: "kharif",
    9: "kharif",
    10: "kharif",
    11: "rabi",
    12: "rabi",
}

SEASON_ORDER = {
    "Kharif": 0,
    "Rabi": 1,
    "Zaid": 2,
}

SEASON_INDEX = {value: key for key, value in SEASON_ORDER.items()}
SEASON_START_MONTH = {
    "Kharif": 6,
    "Rabi": 11,
    "Zaid": 4,
}


def prepare_climate_features(
    raw_frame: pd.DataFrame,
    dataset_cfg: dict[str, Any],
    pipeline_cfg: dict[str, Any],
) -> pd.DataFrame:
    frame, _ = prepare_base_frame(
        raw_frame,
        dataset_cfg,
        pipeline_cfg,
        allow_static_time=False,
    )

    climate_columns = [column for column in ["temperature", "rainfall", "humidity", "max_temperature"] if column in frame.columns]
    for column in climate_columns:
        frame[column] = coerce_numeric(frame[column])

    frame = fill_raw_climate_gaps(frame, climate_columns)

    records: list[dict[str, Any]] = []
    dry_threshold = float(pipeline_cfg.get("dry_spell_rain_threshold_mm", 1.0))
    group_columns = ["region_key", "region", "state", "time", "time_order", "period_start"]
    for key, group in frame.groupby(group_columns, dropna=False, sort=True):
        region_key, region, state, time_value, time_order, period_start = key
        ordered = group.sort_values("event_date")
        rainfall = ordered["rainfall"] if "rainfall" in ordered.columns else pd.Series(dtype="float64")
        temperature = ordered["temperature"] if "temperature" in ordered.columns else pd.Series(dtype="float64")
        humidity = ordered["humidity"] if "humidity" in ordered.columns else pd.Series(dtype="float64")
        max_temperature = ordered["max_temperature"] if "max_temperature" in ordered.columns else temperature

        records.append(
            {
                "region_key": region_key,
                "region": region,
                "state": state,
                "time": time_value,
                "time_order": time_order,
                "period_start": period_start,
                "geo_confidence": round(float(ordered["geo_confidence"].mean()), 4),
                "temp_avg": float(temperature.mean()) if temperature.notna().any() else pd.NA,
                "rain_total": float(rainfall.sum()) if rainfall.notna().any() else pd.NA,
                "humidity_avg": float(humidity.mean()) if humidity.notna().any() else pd.NA,
                "rain_variance": float(rainfall.var(ddof=0)) if rainfall.notna().any() else pd.NA,
                "max_temp": float(max_temperature.max()) if max_temperature.notna().any() else pd.NA,
                "max_temp_3d": rolling_three_step_max(max_temperature),
                "max_rain_1d": float(rainfall.max()) if rainfall.notna().any() else pd.NA,
                "dry_spell_days": longest_dry_spell(rainfall.fillna(0), dry_threshold),
                "climate_obs_count": int(len(group)),
                "climate_interpolation_rate": round(float(ordered["raw_climate_missing_fraction"].mean()), 4),
                "climate_gap_filled": int(ordered["raw_climate_gap_filled"].max()),
                "time_step_missing": 0,
            }
        )

    climate_frame = pd.DataFrame(records)
    target_time_level = str(pipeline_cfg.get("target_time_level", "monthly")).casefold()
    climate_frame = complete_time_grid(
        climate_frame,
        target_time_level,
        flag_column="time_step_missing",
    )
    climate_frame = fill_climate_time_gaps(climate_frame)
    climate_frame = add_lag_features(climate_frame, target_time_level)
    return climate_frame


def prepare_soil_features(
    raw_frame: pd.DataFrame,
    dataset_cfg: dict[str, Any],
    pipeline_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    frame, time_mode = prepare_base_frame(
        raw_frame,
        dataset_cfg,
        pipeline_cfg,
        allow_static_time=True,
    )

    soil_columns = [column for column in ["ph", "n", "p", "k"] if column in frame.columns]
    for column in soil_columns:
        frame[column] = coerce_numeric(frame[column])

    if soil_columns:
        frame["soil_row_missing_fraction"] = frame[soil_columns].isna().mean(axis=1).round(4)
    else:
        frame["soil_row_missing_fraction"] = 1.0

    group_columns = ["region_key", "region", "state", "geo_confidence"]
    if time_mode == "temporal":
        group_columns.extend(["time", "time_order", "period_start"])

    records: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_columns, dropna=False, sort=True):
        record = {
            "region_key": key[0],
            "region": key[1],
            "state": key[2],
            "geo_confidence": round(float(group["geo_confidence"].mean()), 4),
            "pH": float(group["ph"].mean()) if "ph" in group and group["ph"].notna().any() else pd.NA,
            "N": float(group["n"].mean()) if "n" in group and group["n"].notna().any() else pd.NA,
            "P": float(group["p"].mean()) if "p" in group and group["p"].notna().any() else pd.NA,
            "K": float(group["k"].mean()) if "k" in group and group["k"].notna().any() else pd.NA,
            "soil_imputation_rate": round(float(group["soil_row_missing_fraction"].mean()), 4),
            "soil_imputed": int(group["soil_row_missing_fraction"].gt(0).any()),
        }
        if time_mode == "temporal":
            record["time"] = key[4]
            record["time_order"] = key[5]
            record["period_start"] = key[6]
        records.append(record)

    soil_frame = pd.DataFrame(records)
    soil_numeric_columns = [column for column in ["pH", "N", "P", "K"] if column in soil_frame.columns]
    if not soil_frame.empty and soil_numeric_columns:
        soil_frame = impute_numeric_columns(
            soil_frame,
            soil_numeric_columns,
            group_key="region_key",
            fallback_key="state",
            strategy="mean",
        )
    soil_frame = apply_soil_scoring(soil_frame, pipeline_cfg)
    return soil_frame, time_mode


def prepare_crop_labels(
    raw_frame: pd.DataFrame,
    dataset_cfg: dict[str, Any],
    pipeline_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    frame, _ = prepare_base_frame(
        raw_frame,
        dataset_cfg,
        pipeline_cfg,
        allow_static_time=False,
    )

    if "crop" not in frame.columns:
        raise ValueError("Crop dataset must map a raw crop column to canonical 'crop'.")

    if "area" in frame.columns and frame["area"].notna().any():
        frame["weight"] = coerce_numeric(frame["area"]).clip(lower=0).fillna(0)
    elif "production" in frame.columns and frame["production"].notna().any():
        frame["weight"] = coerce_numeric(frame["production"]).clip(lower=0).fillna(0)
    else:
        frame["weight"] = 1.0

    frame["crop"] = frame["crop"].astype("string").str.strip()
    frame["crop_key"] = frame["crop"].map(slugify)

    index_columns = ["region_key", "region", "state", "time", "time_order", "period_start"]
    grouped = (
        frame.groupby(index_columns + ["crop_key"], dropna=False)
        .agg(weight=("weight", "sum"), geo_confidence=("geo_confidence", "mean"))
        .reset_index()
    )
    totals = grouped.groupby(["region_key", "time"], dropna=False)["weight"].transform("sum")
    grouped["crop_probability"] = grouped["weight"] / totals.where(totals != 0, pd.NA)

    pivot = grouped.pivot_table(
        index=index_columns,
        columns="crop_key",
        values="crop_probability",
        fill_value=0.0,
    ).reset_index()
    crop_probability_columns = [
        f"crop_prob_{column}"
        for column in pivot.columns
        if column not in set(index_columns)
    ]
    pivot.columns = [
        column if column in set(index_columns) else f"crop_prob_{column}"
        for column in pivot.columns
    ]

    geo_confidence = (
        grouped.groupby(index_columns, dropna=False)["geo_confidence"]
        .mean()
        .reset_index()
        .rename(columns={"geo_confidence": "geo_confidence"})
    )
    crop_frame = pivot.merge(geo_confidence, on=index_columns, how="left")
    crop_frame = normalize_crop_probabilities(crop_frame, crop_probability_columns)
    return crop_frame, sorted(crop_probability_columns)


def merge_datasets(
    climate_frame: pd.DataFrame,
    soil_frame: pd.DataFrame,
    crop_frame: pd.DataFrame,
    crop_probability_columns: list[str],
    soil_time_mode: str,
    pipeline_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    join_diagnostics = build_join_diagnostics(
        climate_frame,
        soil_frame,
        crop_frame,
        soil_time_mode,
    )

    merged = climate_frame.merge(
        crop_frame,
        on=["region_key", "time"],
        how="outer",
        suffixes=("_climate", "_crop"),
    )
    merged = coalesce_merged_metadata(
        merged,
        prefix_pairs=[("region", ["region_climate", "region_crop"]), ("state", ["state_climate", "state_crop"])],
    )
    merged["time_order"] = coerce_numeric(
        merged.get("time_order_climate", pd.Series(pd.NA, index=merged.index))
    ).astype("Int64").combine_first(
        coerce_numeric(merged.get("time_order_crop", pd.Series(pd.NA, index=merged.index))).astype("Int64")
    )
    merged["period_start"] = pd.to_datetime(
        merged.get("period_start_climate", pd.Series(pd.NaT, index=merged.index)),
        errors="coerce",
    ).combine_first(
        pd.to_datetime(
            merged.get("period_start_crop", pd.Series(pd.NaT, index=merged.index)),
            errors="coerce",
        )
    )

    if soil_time_mode == "temporal" and "time" in soil_frame.columns:
        merged = merged.merge(
            soil_frame,
            on=["region_key", "time"],
            how="left",
            suffixes=("", "_soil"),
        )
    else:
        soil_region_only = soil_frame.drop(columns=["time", "time_order", "period_start"], errors="ignore")
        merged = merged.merge(
            soil_region_only,
            on=["region_key"],
            how="left",
            suffixes=("", "_soil"),
        )

    merged = coalesce_merged_metadata(
        merged,
        prefix_pairs=[("region", ["region", "region_soil"]), ("state", ["state", "state_soil"])],
    )
    merged["geo_confidence"] = average_available_columns(
        merged,
        ["geo_confidence_climate", "geo_confidence_crop", "geo_confidence", "geo_confidence_soil"],
    ).round(4)

    climate_feature_columns = [
        column
        for column in [
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
        ]
        if column in merged.columns
    ]
    soil_numeric_columns = [column for column in ["pH", "N", "P", "K", "soil_health_index"] if column in merged.columns]

    merged["has_climate"] = merged[climate_feature_columns].notna().any(axis=1) if climate_feature_columns else False
    merged["has_crop_labels"] = (
        merged[crop_probability_columns].fillna(0).sum(axis=1).gt(0)
        if crop_probability_columns
        else False
    )
    merged["has_soil"] = merged[soil_numeric_columns].notna().any(axis=1) if soil_numeric_columns else False

    merged["climate_imputed_from_merge"] = (~merged["has_climate"]).astype(int)
    merged["soil_imputed_from_merge"] = (~merged["has_soil"]).astype(int)

    final_frame = merged[merged["has_crop_labels"]].copy()

    if climate_feature_columns:
        climate_missing_before = final_frame[climate_feature_columns].isna()
        final_frame = impute_numeric_columns(
            final_frame,
            climate_feature_columns,
            group_key="region_key",
            fallback_key="state",
            strategy="mean",
        )
        final_frame["climate_gap_filled"] = (
            final_frame.get("climate_gap_filled", 0).fillna(0).astype(int)
            | climate_missing_before.any(axis=1).astype(int)
        )
        climate_missing_rate = climate_missing_before.mean(axis=1).round(4)
        existing_rate = coerce_numeric(
            final_frame.get("climate_interpolation_rate", pd.Series(0.0, index=final_frame.index))
        ).fillna(0)
        final_frame["climate_interpolation_rate"] = existing_rate.combine(climate_missing_rate, max).round(4)
        final_frame["time_step_missing"] = (
            final_frame.get("time_step_missing", 0).fillna(0).astype(int)
            | climate_missing_before.all(axis=1).astype(int)
        )

    if soil_numeric_columns:
        soil_missing_before = final_frame[soil_numeric_columns].isna()
        final_frame = impute_numeric_columns(
            final_frame,
            soil_numeric_columns,
            group_key="region_key",
            fallback_key="state",
            strategy="mean",
        )
        soil_rate = soil_missing_before.mean(axis=1).round(4)
        existing_soil_rate = coerce_numeric(
            final_frame.get("soil_imputation_rate", pd.Series(0.0, index=final_frame.index))
        ).fillna(0)
        final_frame["soil_imputation_rate"] = existing_soil_rate.combine(soil_rate, max).round(4)
        final_frame["soil_imputed"] = (
            final_frame.get("soil_imputed", 0).fillna(0).astype(int)
            | soil_missing_before.any(axis=1).astype(int)
        )

    final_frame = apply_soil_scoring(final_frame, pipeline_cfg)
    final_frame = add_context_features(final_frame, pipeline_cfg)
    final_frame = add_spatiotemporal_context(final_frame)
    final_frame = normalize_crop_probabilities(final_frame, crop_probability_columns)

    zero_label_rows = final_frame[crop_probability_columns].sum(axis=1).eq(0) if crop_probability_columns else pd.Series(False, index=final_frame.index)
    final_frame = final_frame.loc[~zero_label_rows].copy()

    final_frame["time_step_missing"] = final_frame.get("time_step_missing", 0).fillna(0).astype(int)
    final_frame["soil_imputed"] = final_frame.get("soil_imputed", 0).fillna(0).astype(int)
    final_frame["climate_gap_filled"] = final_frame.get("climate_gap_filled", 0).fillna(0).astype(int)
    final_frame["geo_confidence"] = coerce_numeric(final_frame["geo_confidence"]).fillna(0.0).clip(lower=0, upper=1)
    final_frame["data_confidence"] = compute_data_confidence(final_frame).round(4)

    for column in [
        "geo_confidence_climate",
        "geo_confidence_crop",
        "geo_confidence_soil",
        "region_soil",
        "state_soil",
        "region_climate",
        "region_crop",
        "state_climate",
        "state_crop",
        "time_order_climate",
        "time_order_crop",
        "period_start_climate",
        "period_start_crop",
        "has_climate",
        "has_crop_labels",
        "has_soil",
    ]:
        if column in final_frame.columns:
            final_frame = final_frame.drop(columns=[column])

    ordered_columns = [
        "region",
        "state",
        "region_key",
        "time",
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
        "N_class",
        "P_class",
        "K_class",
        "soil_health_index",
        "irrigation_index",
        "rotation_score",
        "fertility_class",
        "state_context",
        "region_context",
        "target_month",
        "target_season",
        "time_step_missing",
        "climate_gap_filled",
        "soil_imputed",
        "geo_confidence",
        "data_confidence",
    ] + crop_probability_columns
    existing_columns = [column for column in ordered_columns if column in final_frame.columns]
    final_frame = final_frame[existing_columns].sort_values(["region", "time"]).reset_index(drop=True)

    rows_after_outer_merge = int(join_diagnostics["outer_region_time_keys"])
    rows_ready = int(len(final_frame))
    rows_lost = max(rows_after_outer_merge - rows_ready, 0)
    merge_report = {
        "rows_after_outer_merge": rows_after_outer_merge,
        "rows_ready_for_training": rows_ready,
        "rows_lost_after_merge": rows_lost,
        "percent_rows_lost_after_merge": round((rows_lost / rows_after_outer_merge) * 100, 2) if rows_after_outer_merge else 0.0,
        "rows_without_climate_features_before_imputation": int((~merged["has_climate"]).sum()),
        "rows_without_soil_features_before_imputation": int((~merged["has_soil"]).sum()),
        "rows_without_crop_labels": int((~merged["has_crop_labels"]).sum()),
        "join_diagnostics": join_diagnostics,
    }
    return final_frame, merge_report


def validate_final_dataset(
    final_frame: pd.DataFrame,
    crop_probability_columns: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    numeric_columns = final_frame.select_dtypes(include=["number"]).columns.tolist()
    summary_stats = (
        final_frame[numeric_columns].describe().transpose().round(4)
        if numeric_columns
        else pd.DataFrame()
    )
    if not summary_stats.empty:
        summary_stats["missing_pct"] = final_frame[summary_stats.index].isna().mean().mul(100).round(2)

    missing_pct = final_frame.isna().mean().mul(100).round(2).to_dict()
    feature_ranges = {
        column: {
            "min": None if final_frame[column].dropna().empty else float(final_frame[column].min()),
            "max": None if final_frame[column].dropna().empty else float(final_frame[column].max()),
        }
        for column in numeric_columns
    }

    if crop_probability_columns:
        label_sums = final_frame[crop_probability_columns].sum(axis=1)
        negative_rows = int((final_frame[crop_probability_columns] < 0).any(axis=1).sum())
        nan_rows = int(final_frame[crop_probability_columns].isna().any(axis=1).sum())
        rows_sum_to_one = int(label_sums.round(6).eq(1.0).sum())
        label_distribution = {
            column: {
                "mean_probability": round(float(final_frame[column].mean()), 6),
                "total_probability": round(float(final_frame[column].sum()), 6),
                "non_zero_rows": int(final_frame[column].gt(0).sum()),
            }
            for column in crop_probability_columns
        }
    else:
        negative_rows = 0
        nan_rows = 0
        rows_sum_to_one = 0
        label_distribution = {}

    validation_report = {
        "row_count": int(len(final_frame)),
        "column_count": int(len(final_frame.columns)),
        "region_count": int(final_frame["region_key"].nunique()) if "region_key" in final_frame.columns else 0,
        "time_step_count": int(final_frame["time"].nunique()) if "time" in final_frame.columns else 0,
        "avg_time_steps_per_region": round(
            float(final_frame.groupby("region_key")["time"].nunique().mean()),
            4,
        ) if {"region_key", "time"}.issubset(final_frame.columns) and not final_frame.empty else 0.0,
        "missing_value_percentage": missing_pct,
        "feature_ranges": feature_ranges,
        "label_consistency": {
            "rows_checked": int(len(final_frame)),
            "rows_sum_to_one": rows_sum_to_one,
            "rows_with_negative_probabilities": negative_rows,
            "rows_with_nan_probabilities": nan_rows,
        },
        "label_distribution": label_distribution,
        "quality_feature_summary": {
            "rows_with_missing_time_steps": int(final_frame.get("time_step_missing", pd.Series(dtype="int64")).sum()) if "time_step_missing" in final_frame.columns else 0,
            "rows_with_climate_gap_fill": int(final_frame.get("climate_gap_filled", pd.Series(dtype="int64")).sum()) if "climate_gap_filled" in final_frame.columns else 0,
            "rows_with_soil_imputation": int(final_frame.get("soil_imputed", pd.Series(dtype="int64")).sum()) if "soil_imputed" in final_frame.columns else 0,
            "mean_geo_confidence": round(float(final_frame.get("geo_confidence", pd.Series([0])).mean()), 4) if not final_frame.empty else 0.0,
            "mean_data_confidence": round(float(final_frame.get("data_confidence", pd.Series([0])).mean()), 4) if not final_frame.empty else 0.0,
        },
        "sample_rows": final_frame.head(5).to_dict(orient="records"),
    }
    return validation_report, summary_stats


def prepare_base_frame(
    raw_frame: pd.DataFrame,
    dataset_cfg: dict[str, Any],
    pipeline_cfg: dict[str, Any],
    allow_static_time: bool,
) -> tuple[pd.DataFrame, str]:
    columns_map = dataset_cfg.get("columns", {})
    rename_map = {
        raw_name: canonical_name
        for canonical_name, raw_name in columns_map.items()
        if raw_name in raw_frame.columns
    }
    frame = raw_frame.rename(columns=rename_map).copy()

    if "region" not in frame.columns:
        raise ValueError("Dataset must provide a region column mapping.")
    if "state" not in frame.columns:
        frame["state"] = dataset_cfg.get("default_state", "")

    aliases = pipeline_cfg.get("region_aliases", {})
    canonical_records = [
        canonicalize_region_with_metadata(region, state, aliases)
        for region, state in zip(frame["region"], frame["state"])
    ]
    frame["region"] = [region for region, _, _ in canonical_records]
    frame["state"] = [state for _, state, _ in canonical_records]
    frame["geo_confidence"] = [geo_confidence for _, _, geo_confidence in canonical_records]
    frame["region_key"] = [
        make_region_key(region, state)
        for region, state in zip(frame["region"], frame["state"])
    ]

    frame, time_mode = add_time_columns(
        frame,
        dataset_cfg,
        pipeline_cfg.get("target_time_level", "monthly"),
        allow_static_time=allow_static_time,
    )
    return frame, time_mode


def add_time_columns(
    frame: pd.DataFrame,
    dataset_cfg: dict[str, Any],
    target_time_level: str,
    allow_static_time: bool,
) -> tuple[pd.DataFrame, str]:
    has_date = "date" in frame.columns
    has_year = "year" in frame.columns
    has_month = "month" in frame.columns
    has_season = "season" in frame.columns

    if has_date:
        frame["event_date"] = pd.to_datetime(
            frame["date"],
            errors="coerce",
            dayfirst=dataset_cfg.get("dayfirst", False),
        )
        frame["year"] = frame["event_date"].dt.year.astype("Int64")
        frame["month"] = frame["event_date"].dt.month.astype("Int64")
    else:
        if has_year:
            frame["year"] = coerce_numeric(frame["year"]).astype("Int64")
        if has_month:
            frame["month"] = coerce_numeric(frame["month"]).astype("Int64")
        if has_year and has_month:
            event_dates = pd.to_datetime(
                {
                    "year": frame["year"].fillna(2000).astype(int),
                    "month": frame["month"].fillna(1).astype(int),
                    "day": 1,
                },
                errors="coerce",
            )
            frame["event_date"] = event_dates.where(frame["year"].notna() & frame["month"].notna())
        elif has_year:
            event_dates = pd.to_datetime(
                {
                    "year": frame["year"].fillna(2000).astype(int),
                    "month": 1,
                    "day": 1,
                },
                errors="coerce",
            )
            frame["event_date"] = event_dates.where(frame["year"].notna())

    if has_season:
        frame["season"] = frame["season"].map(normalize_season_name)

    target_time_level = str(target_time_level).casefold()
    if target_time_level == "monthly":
        if "event_date" not in frame.columns or frame["event_date"].isna().all():
            if allow_static_time:
                frame["time"] = pd.NA
                frame["time_order"] = pd.NA
                frame["period_start"] = pd.NaT
                return frame, "static"
            raise ValueError("Monthly alignment requires either a date column or year + month columns.")

        frame["time"] = frame["event_date"].dt.to_period("M").astype(str)
        frame["time_order"] = [monthly_time_order(year, month) for year, month in zip(frame["year"], frame["month"])]
        frame["period_start"] = frame["event_date"].dt.to_period("M").dt.to_timestamp()
        return frame, "temporal"

    if target_time_level == "seasonal":
        if "year" not in frame.columns or frame["year"].isna().all():
            if allow_static_time:
                frame["time"] = pd.NA
                frame["time_order"] = pd.NA
                frame["period_start"] = pd.NaT
                return frame, "static"
            raise ValueError("Seasonal alignment requires year information.")

        if "season" not in frame.columns or frame["season"].isna().all():
            if "month" not in frame.columns or frame["month"].isna().all():
                if allow_static_time:
                    frame["time"] = pd.NA
                    frame["time_order"] = pd.NA
                    frame["period_start"] = pd.NaT
                    return frame, "static"
                raise ValueError("Seasonal alignment requires a season column or a month/date column.")
            frame["season_year"], frame["season"] = zip(
                *[derive_season(year, month) for year, month in zip(frame["year"], frame["month"])]
            )
        else:
            frame["season_year"] = frame["year"]

        frame["time"] = frame["season_year"].astype("string") + "-" + frame["season"].astype("string")
        frame["time_order"] = [season_time_order(year, season) for year, season in zip(frame["season_year"], frame["season"])]
        frame["period_start"] = [season_period_start(year, season) for year, season in zip(frame["season_year"], frame["season"])]
        return frame, "temporal"

    raise ValueError("target_time_level must be either 'monthly' or 'seasonal'.")


def fill_raw_climate_gaps(
    frame: pd.DataFrame,
    climate_columns: list[str],
) -> pd.DataFrame:
    if frame.empty or not climate_columns:
        frame["raw_climate_missing_fraction"] = 0.0
        frame["raw_climate_gap_filled"] = 0
        return frame

    pieces = []
    for _, group in frame.groupby("region_key", sort=False):
        group = group.sort_values(["period_start", "event_date"]).copy()
        missing_before = group[climate_columns].isna()
        group[climate_columns] = group[climate_columns].interpolate(limit_direction="both")
        group[climate_columns] = group[climate_columns].ffill().bfill()
        group["raw_climate_missing_fraction"] = missing_before.mean(axis=1).round(4)
        group["raw_climate_gap_filled"] = missing_before.any(axis=1).astype(int)
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def fill_climate_time_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    feature_columns = [
        column
        for column in [
            "temp_avg",
            "rain_total",
            "humidity_avg",
            "rain_variance",
            "max_temp",
            "max_temp_3d",
            "max_rain_1d",
            "dry_spell_days",
        ]
        if column in frame.columns
    ]

    pieces = []
    for _, group in frame.groupby("region_key", sort=False):
        group = group.sort_values("time_order").copy()
        missing_before = group[feature_columns].isna()
        group[feature_columns] = group[feature_columns].interpolate(limit_direction="both")
        group[feature_columns] = group[feature_columns].ffill().bfill()
        group["climate_interpolation_rate"] = (
            coerce_numeric(group.get("climate_interpolation_rate", pd.Series(0.0, index=group.index))).fillna(0)
            .combine(missing_before.mean(axis=1).round(4), max)
        )
        group["climate_gap_filled"] = (
            group.get("climate_gap_filled", 0).fillna(0).astype(int)
            | missing_before.any(axis=1).astype(int)
            | group.get("time_step_missing", 0).fillna(0).astype(int)
        )
        group["climate_obs_count"] = coerce_numeric(group.get("climate_obs_count", pd.Series(0, index=group.index))).fillna(0).astype(int)
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def add_lag_features(frame: pd.DataFrame, target_time_level: str) -> pd.DataFrame:
    if frame.empty:
        frame["temp_lag_7"] = pd.NA
        frame["rain_lag_14"] = pd.NA
        return frame

    pieces = []
    for _, group in frame.groupby("region_key", sort=False):
        group = group.sort_values("time_order").copy()

        # Once the data is monthly or seasonal, exact 7-day and 14-day lags are no longer
        # available. We keep the feature names for downstream compatibility and use the
        # previous aggregated period as a coarse proxy.
        group["temp_lag_7"] = group["temp_avg"].shift(1)
        group["rain_lag_14"] = group["rain_total"].shift(1)
        group["temp_lag_7"] = group["temp_lag_7"].fillna(group["temp_avg"])
        group["rain_lag_14"] = group["rain_lag_14"].fillna(group["rain_total"])
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def complete_time_grid(
    frame: pd.DataFrame,
    target_time_level: str,
    flag_column: str,
) -> pd.DataFrame:
    if frame.empty or "time_order" not in frame.columns:
        return frame

    completed = []
    for _, group in frame.groupby("region_key", sort=False):
        group = group.sort_values("time_order").copy()
        min_order = int(group["time_order"].min())
        max_order = int(group["time_order"].max())
        expected = pd.DataFrame({"region_key": group["region_key"].iloc[0], "time_order": list(range(min_order, max_order + 1))})
        expected["time"] = expected["time_order"].map(lambda value: format_time_from_order(value, target_time_level))
        expected["period_start"] = expected["time_order"].map(lambda value: period_start_from_order(value, target_time_level))
        expected["region"] = group["region"].iloc[0]
        expected["state"] = group["state"].iloc[0]
        expected["geo_confidence"] = group["geo_confidence"].iloc[0]

        expanded = expected.merge(group, on=["region_key", "time_order"], how="left", suffixes=("_expected", ""))
        expanded[flag_column] = expanded["time"].isna().astype(int)
        expanded["time"] = expanded["time"].fillna(expanded["time_expected"])
        expanded["period_start"] = expanded["period_start"].combine_first(expanded["period_start_expected"])
        expanded["region"] = expanded["region"].combine_first(expanded["region_expected"])
        expanded["state"] = expanded["state"].combine_first(expanded["state_expected"])
        expanded["geo_confidence"] = average_available_columns(
            expanded,
            ["geo_confidence", "geo_confidence_expected"],
        ).round(4)
        expanded = expanded.drop(
            columns=[
                column
                for column in ["time_expected", "period_start_expected", "region_expected", "state_expected", "geo_confidence_expected"]
                if column in expanded.columns
            ]
        )
        completed.append(expanded)

    return pd.concat(completed, ignore_index=True)


def normalize_crop_probabilities(
    frame: pd.DataFrame,
    crop_probability_columns: list[str],
) -> pd.DataFrame:
    if not crop_probability_columns:
        return frame

    for column in crop_probability_columns:
        frame[column] = coerce_numeric(frame[column]).fillna(0.0).clip(lower=0)

    row_sums = frame[crop_probability_columns].sum(axis=1)
    valid_rows = row_sums > 0
    frame.loc[valid_rows, crop_probability_columns] = frame.loc[valid_rows, crop_probability_columns].div(row_sums[valid_rows], axis=0)
    frame.loc[~valid_rows, crop_probability_columns] = 0.0
    return frame


def add_context_features(
    frame: pd.DataFrame,
    pipeline_cfg: dict[str, Any],
) -> pd.DataFrame:
    defaults = DEFAULT_CONTEXT_FEATURES | pipeline_cfg.get("context_defaults", {})
    for column, value in defaults.items():
        if column not in frame.columns:
            frame[column] = value
        else:
            frame[column] = frame[column].fillna(value)

    if "irrigation_index" in frame.columns:
        frame["irrigation_index"] = coerce_numeric(frame["irrigation_index"]).fillna(
            float(defaults["irrigation_index"])
        )
        if frame["irrigation_index"].nunique(dropna=True) <= 1:
            frame["irrigation_index"] = infer_irrigation_index(frame, float(defaults["irrigation_index"]))

    if "rotation_score" in frame.columns:
        frame["rotation_score"] = coerce_numeric(frame["rotation_score"]).fillna(
            float(defaults["rotation_score"])
        )
        if frame["rotation_score"].nunique(dropna=True) <= 1:
            frame["rotation_score"] = infer_rotation_score(frame, float(defaults["rotation_score"]))

    if "fertility_class" in frame.columns:
        frame["fertility_class"] = (
            frame["fertility_class"].astype("string").fillna(str(defaults["fertility_class"])).str.strip().str.lower()
        )
    return frame


def infer_irrigation_index(frame: pd.DataFrame, default_value: float) -> pd.Series:
    """Derive bounded irrigation variation for demo rows when no source signal exists."""
    if frame.empty:
        return pd.Series(dtype="float64", index=frame.index)

    rain = coerce_numeric(frame.get("rain_total", pd.Series(default_value, index=frame.index))).fillna(0.0)
    soil = coerce_numeric(frame.get("soil_health_index", pd.Series(70.0, index=frame.index))).fillna(70.0)
    dry_spell = coerce_numeric(frame.get("dry_spell_days", pd.Series(1.0, index=frame.index))).fillna(1.0)
    humidity = coerce_numeric(frame.get("humidity_avg", pd.Series(60.0, index=frame.index))).fillna(60.0)
    max_temp = coerce_numeric(frame.get("max_temp", pd.Series(30.0, index=frame.index))).fillna(30.0)

    rain_component = 1.0 - normalize_series(rain)
    soil_component = 1.0 - normalize_series(soil)
    dry_component = normalize_series(dry_spell)
    humidity_component = 1.0 - normalize_series(humidity)
    heat_component = normalize_series(max_temp)
    irrigation = (
        0.10
        + (0.35 * rain_component)
        + (0.25 * dry_component)
        + (0.15 * humidity_component)
        + (0.10 * heat_component)
        + (0.05 * soil_component)
    )
    return irrigation.clip(lower=0.0, upper=1.0).round(4)


def infer_rotation_score(frame: pd.DataFrame, default_value: float) -> pd.Series:
    """Estimate rotation-management quality from exogenous soil and climate signals."""
    if frame.empty:
        return pd.Series(dtype="float64", index=frame.index)

    soil_component = normalize_series(
        coerce_numeric(frame.get("soil_health_index", pd.Series(70.0, index=frame.index))).fillna(70.0)
    )
    ph = coerce_numeric(frame.get("pH", pd.Series(6.5, index=frame.index))).fillna(6.5)
    n_series = normalize_series(coerce_numeric(frame.get("N", pd.Series(350.0, index=frame.index))).fillna(350.0))
    p_series = normalize_series(coerce_numeric(frame.get("P", pd.Series(18.0, index=frame.index))).fillna(18.0))
    k_series = normalize_series(coerce_numeric(frame.get("K", pd.Series(200.0, index=frame.index))).fillna(200.0))
    humidity = normalize_series(
        coerce_numeric(frame.get("humidity_avg", pd.Series(60.0, index=frame.index))).fillna(60.0)
    )

    nutrient_frame = pd.concat([n_series, p_series, k_series], axis=1)
    nutrient_balance = 1.0 - nutrient_frame.std(axis=1, ddof=0).clip(lower=0.0, upper=1.0)
    ph_balance = (1.0 - (ph.sub(6.5).abs() / 2.5)).clip(lower=0.0, upper=1.0)
    rotation = (
        0.15
        + (0.40 * soil_component.clip(lower=0.0, upper=1.0))
        + (0.20 * nutrient_balance.clip(lower=0.0, upper=1.0))
        + (0.15 * ph_balance)
        + (0.10 * humidity.clip(lower=0.0, upper=1.0))
    )
    return rotation.clip(lower=0.0, upper=1.0).round(4)


def add_spatiotemporal_context(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    period_start = pd.to_datetime(frame.get("period_start", pd.Series(pd.NaT, index=frame.index)), errors="coerce")
    month_series = period_start.dt.month
    if month_series.isna().all() and "time" in frame.columns:
        month_series = (
            frame["time"]
            .astype("string")
            .str.extract(r"(\d{2})$", expand=False)
            .astype("Int64")
        )

    frame["state_context"] = frame.get("state", "").astype("string").str.strip().str.lower()
    frame["region_context"] = frame.get("region_key", "").astype("string").str.strip().str.lower()
    frame["target_month"] = month_series.astype("Int64").astype("string").str.zfill(2)
    frame["target_season"] = (
        month_series.map(MONTH_TO_SEASON).fillna("unknown").astype("string").str.strip().str.lower()
    )
    return frame


def compute_data_confidence(frame: pd.DataFrame) -> pd.Series:
    climate_penalty = (
        coerce_numeric(frame.get("climate_interpolation_rate", pd.Series(0.0, index=frame.index))).fillna(0) * 0.35
    )
    soil_penalty = (
        coerce_numeric(frame.get("soil_imputation_rate", pd.Series(0.0, index=frame.index))).fillna(0) * 0.25
    )
    time_penalty = coerce_numeric(frame.get("time_step_missing", pd.Series(0, index=frame.index))).fillna(0) * 0.20
    climate_merge_penalty = coerce_numeric(frame.get("climate_imputed_from_merge", pd.Series(0, index=frame.index))).fillna(0) * 0.10
    soil_merge_penalty = coerce_numeric(frame.get("soil_imputed_from_merge", pd.Series(0, index=frame.index))).fillna(0) * 0.10
    quality_score = 1 - (climate_penalty + soil_penalty + time_penalty + climate_merge_penalty + soil_merge_penalty)
    return quality_score.clip(lower=0, upper=1)


def build_join_diagnostics(
    climate_frame: pd.DataFrame,
    soil_frame: pd.DataFrame,
    crop_frame: pd.DataFrame,
    soil_time_mode: str,
) -> dict[str, Any]:
    climate_keys = climate_frame[["region_key", "time"]].drop_duplicates()
    crop_keys = crop_frame[["region_key", "time"]].drop_duplicates()
    climate_crop = climate_keys.merge(crop_keys, on=["region_key", "time"], how="outer", indicator=True)

    diagnostics = {
        "outer_region_time_keys": int(len(climate_crop)),
        "climate_only_region_time_pairs": int(climate_crop["_merge"].eq("left_only").sum()),
        "crop_only_region_time_pairs": int(climate_crop["_merge"].eq("right_only").sum()),
        "sample_unmatched_climate_pairs": format_key_samples(climate_crop.loc[climate_crop["_merge"].eq("left_only")]),
        "sample_unmatched_crop_pairs": format_key_samples(climate_crop.loc[climate_crop["_merge"].eq("right_only")]),
    }

    if soil_time_mode == "temporal" and {"region_key", "time"}.issubset(soil_frame.columns):
        soil_keys = soil_frame[["region_key", "time"]].drop_duplicates()
        base_to_soil = climate_crop[["region_key", "time"]].drop_duplicates().merge(
            soil_keys,
            on=["region_key", "time"],
            how="left",
            indicator=True,
        )
        diagnostics["base_pairs_without_soil_match"] = int(base_to_soil["_merge"].eq("left_only").sum())
        diagnostics["sample_pairs_without_soil_match"] = format_key_samples(base_to_soil.loc[base_to_soil["_merge"].eq("left_only")])
    else:
        soil_keys = soil_frame[["region_key"]].drop_duplicates()
        base_to_soil = climate_crop[["region_key"]].drop_duplicates().merge(
            soil_keys,
            on=["region_key"],
            how="left",
            indicator=True,
        )
        diagnostics["regions_without_soil_match"] = int(base_to_soil["_merge"].eq("left_only").sum())
        diagnostics["sample_regions_without_soil_match"] = base_to_soil.loc[
            base_to_soil["_merge"].eq("left_only"),
            "region_key",
        ].head(10).tolist()

    return diagnostics


def coalesce_merged_metadata(
    frame: pd.DataFrame,
    prefix_pairs: list[tuple[str, list[str]]],
) -> pd.DataFrame:
    for target, candidates in prefix_pairs:
        available = [column for column in candidates if column in frame.columns]
        if not available:
            continue
        series = frame[available[0]]
        for candidate in available[1:]:
            series = series.combine_first(frame[candidate])
        frame[target] = series
    return frame


def average_available_columns(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return pd.Series(0.0, index=frame.index)
    return frame[available].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)


def normalize_series(series: pd.Series) -> pd.Series:
    numeric = coerce_numeric(series).astype("float64")
    if numeric.dropna().empty:
        return pd.Series(0.0, index=series.index, dtype="float64")

    min_value = float(numeric.min(skipna=True))
    max_value = float(numeric.max(skipna=True))
    value_range = max_value - min_value
    if value_range <= 1e-9:
        return pd.Series(0.0, index=series.index, dtype="float64")
    return ((numeric - min_value) / value_range).fillna(0.0)


def monthly_time_order(year: Any, month: Any) -> int | pd.NA:
    if pd.isna(year) or pd.isna(month):
        return pd.NA
    return int(year) * 12 + int(month) - 1


def season_time_order(year: Any, season: str) -> int | pd.NA:
    if pd.isna(year) or pd.isna(season):
        return pd.NA
    return int(year) * 3 + SEASON_ORDER.get(str(season), 0)


def format_time_from_order(order: int, target_time_level: str) -> str:
    if str(target_time_level).casefold() == "monthly":
        year_value = order // 12
        month_value = (order % 12) + 1
        return f"{year_value:04d}-{month_value:02d}"

    year_value = order // 3
    season_value = SEASON_INDEX.get(order % 3, "Kharif")
    return f"{year_value}-{season_value}"


def period_start_from_order(order: int, target_time_level: str) -> pd.Timestamp:
    if str(target_time_level).casefold() == "monthly":
        year_value = order // 12
        month_value = (order % 12) + 1
        return pd.Timestamp(year=year_value, month=month_value, day=1)

    year_value = order // 3
    season_value = SEASON_INDEX.get(order % 3, "Kharif")
    return season_period_start(year_value, season_value)


def normalize_season_name(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    token = str(value).strip().casefold()
    if token in {"kharif", "monsoon"}:
        return "Kharif"
    if token in {"rabi", "winter"}:
        return "Rabi"
    if token in {"zaid", "summer"}:
        return "Zaid"
    return token.capitalize()


def derive_season(year: Any, month: Any) -> tuple[int | pd.NA, str]:
    if pd.isna(year) or pd.isna(month):
        return pd.NA, "Unknown"

    year_value = int(year)
    month_value = int(month)
    if month_value in {6, 7, 8, 9, 10}:
        return year_value, "Kharif"
    if month_value in {11, 12}:
        return year_value, "Rabi"
    if month_value in {1, 2, 3}:
        return year_value - 1, "Rabi"
    if month_value in {4, 5}:
        return year_value, "Zaid"
    return year_value, "Unknown"


def season_period_start(year: Any, season: str) -> pd.Timestamp:
    if pd.isna(year):
        return pd.NaT
    month_value = SEASON_START_MONTH.get(str(season), 1)
    return pd.Timestamp(year=int(year), month=month_value, day=1)


def rolling_three_step_max(series: pd.Series) -> float | pd.NA:
    if series.notna().any():
        return float(series.rolling(window=3, min_periods=1).mean().max())
    return pd.NA


def longest_dry_spell(rainfall: pd.Series, threshold: float) -> int:
    longest = 0
    current = 0
    for value in rainfall:
        if pd.isna(value) or float(value) <= threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def classify_nutrient(series: pd.Series, bins: list[float]) -> pd.Series:
    return pd.cut(
        series,
        bins=[float("-inf"), bins[0], bins[1], float("inf")],
        labels=["low", "medium", "high"],
    ).astype("string")


def apply_soil_scoring(
    frame: pd.DataFrame,
    pipeline_cfg: dict[str, Any],
) -> pd.DataFrame:
    bins_cfg = pipeline_cfg.get("soil_class_bins", DEFAULT_SOIL_BINS)
    bins = {
        nutrient: bins_cfg.get(nutrient, default_bins)
        for nutrient, default_bins in DEFAULT_SOIL_BINS.items()
    }

    for nutrient in ["N", "P", "K"]:
        if nutrient in frame.columns:
            frame[nutrient] = coerce_numeric(frame[nutrient])

    if "pH" in frame.columns:
        frame["pH"] = coerce_numeric(frame["pH"])

    if "N" in frame.columns:
        frame["N_class"] = classify_nutrient(frame["N"], bins["N"])
    if "P" in frame.columns:
        frame["P_class"] = classify_nutrient(frame["P"], bins["P"])
    if "K" in frame.columns:
        frame["K_class"] = classify_nutrient(frame["K"], bins["K"])

    # This is an intentionally simple composite baseline, not a scientific SHC score.
    ph_score = (
        (1 - (frame["pH"] - 6.5).abs() / 2.5).clip(lower=0, upper=1)
        if "pH" in frame.columns
        else pd.Series(pd.NA, index=frame.index)
    )
    n_score = (
        (frame["N"] / bins["N"][1]).clip(lower=0, upper=1)
        if "N" in frame.columns
        else pd.Series(pd.NA, index=frame.index)
    )
    p_score = (
        (frame["P"] / bins["P"][1]).clip(lower=0, upper=1)
        if "P" in frame.columns
        else pd.Series(pd.NA, index=frame.index)
    )
    k_score = (
        (frame["K"] / bins["K"][1]).clip(lower=0, upper=1)
        if "K" in frame.columns
        else pd.Series(pd.NA, index=frame.index)
    )

    score_frame = pd.concat([ph_score, n_score, p_score, k_score], axis=1)
    frame["soil_health_index"] = score_frame.mean(axis=1, skipna=True).mul(100).round(2)

    for column in ["N_class", "P_class", "K_class"]:
        if column in frame.columns:
            frame[column] = frame[column].fillna("unknown")

    return frame


def impute_numeric_columns(
    frame: pd.DataFrame,
    columns: list[str],
    group_key: str,
    fallback_key: str | None = None,
    strategy: str = "median",
) -> pd.DataFrame:
    for column in columns:
        if column not in frame.columns:
            continue

        frame[column] = coerce_numeric(frame[column])
        reducer = "median" if strategy == "median" else "mean"

        group_values = frame.groupby(group_key)[column].transform(reducer)
        frame[column] = frame[column].fillna(group_values)

        if fallback_key and fallback_key in frame.columns:
            fallback_values = frame.groupby(fallback_key)[column].transform(reducer)
            frame[column] = frame[column].fillna(fallback_values)

        global_value = frame[column].median() if strategy == "median" else frame[column].mean()
        frame[column] = frame[column].fillna(global_value)
    return frame


def format_key_samples(frame: pd.DataFrame, limit: int = 10) -> list[str]:
    if frame.empty:
        return []
    if {"region_key", "time"}.issubset(frame.columns):
        samples = frame[["region_key", "time"]].head(limit).itertuples(index=False, name=None)
        return [f"{region_key}|{time_value}" for region_key, time_value in samples]
    if "region_key" in frame.columns:
        return frame["region_key"].head(limit).tolist()
    return []
