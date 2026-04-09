from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import make_region_key, normalize_region_name

CLIMATE_FEATURES = {
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
}
SOIL_FEATURES = {
    "pH",
    "N",
    "P",
    "K",
    "soil_health_index",
    "N_class",
    "P_class",
    "K_class",
    "fertility_class",
}
MANAGEMENT_FEATURES = {
    "irrigation_index",
    "rotation_score",
    "time_step_missing",
    "climate_gap_filled",
    "soil_imputed",
    "geo_confidence",
}
LOCATION_CONTEXT_FEATURES = {
    "region_context",
    "state_context",
    "target_month",
    "target_season",
}
SOURCE_WEIGHTS = {
    "region_month": 1.0,
    "region_all": 0.92,
    "state_month": 0.84,
    "state_all": 0.76,
    "global_month": 0.68,
    "global": 0.58,
}
MONTH_LABELS = {
    "01": "January",
    "02": "February",
    "03": "March",
    "04": "April",
    "05": "May",
    "06": "June",
    "07": "July",
    "08": "August",
    "09": "September",
    "10": "October",
    "11": "November",
    "12": "December",
}


def month_to_season(month_text: str) -> str:
    try:
        month_value = int(month_text)
    except (TypeError, ValueError):
        return "unknown"
    if month_value in {6, 7, 8, 9, 10}:
        return "kharif"
    if month_value in {11, 12, 1, 2, 3}:
        return "rabi"
    return "zaid"


class LocalizedFeatureProvider:
    def __init__(
        self,
        dataset_path: Path | None,
        numeric_features: list[str],
        categorical_features: list[str],
        label_columns: list[str],
    ) -> None:
        self.dataset_path = dataset_path
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.label_columns = list(label_columns)
        self.available = False
        self.buffer_strategy = "same_month_local_climatology"
        self.live_weather_status = "planned_pending_external_approval"
        self._cache: dict[tuple[str, str, str], dict[str, Any]] = {}

        if dataset_path is None or not dataset_path.exists():
            self.frame = pd.DataFrame()
            self.region_catalog: list[dict[str, Any]] = []
            self.region_lookup_by_key: dict[str, dict[str, Any]] = {}
            self.region_lookup_by_region: dict[str, list[dict[str, Any]]] = {}
            return

        available_columns = pd.read_csv(dataset_path, nrows=0).columns.tolist()
        desired_columns = {
            "region",
            "state",
            "region_key",
            "time",
            *self.numeric_features,
            *self.categorical_features,
            *self.label_columns,
        }
        usecols = [column for column in desired_columns if column in available_columns]
        frame = pd.read_csv(dataset_path, usecols=usecols)
        if frame.empty:
            self.frame = frame
            self.region_catalog = []
            self.region_lookup_by_key = {}
            self.region_lookup_by_region = {}
            return

        frame = frame.copy()
        if "region_key" not in frame.columns and {"region", "state"}.issubset(frame.columns):
            frame["region_key"] = frame.apply(
                lambda row: make_region_key(row.get("region", ""), row.get("state", "")),
                axis=1,
            )
        if "target_month" not in frame.columns:
            frame["target_month"] = frame.get("time", "").astype("string").str[-2:].fillna("")
        frame["target_month"] = frame["target_month"].astype("string").str.zfill(2)
        if "target_season" not in frame.columns:
            frame["target_season"] = frame["target_month"].map(month_to_season)
        frame["target_season"] = frame["target_season"].astype("string").str.strip().str.lower()
        if "state_context" not in frame.columns:
            frame["state_context"] = frame.get("state", "").astype("string").str.strip().str.lower()
        if "region_context" not in frame.columns:
            frame["region_context"] = frame.get("region_key", "").astype("string").str.strip().str.lower()
        frame["region_norm"] = frame.get("region", "").map(normalize_region_name)
        frame["state_norm"] = frame.get("state", "").map(normalize_region_name)
        self.frame = frame
        self.region_catalog = self._build_region_catalog()
        self.region_lookup_by_key = {
            str(row["region_key"]).strip().lower(): {
                "region": str(row["region"]).strip(),
                "state": str(row["state"]).strip(),
                "region_key": str(row["region_key"]).strip().lower(),
                "region_norm": str(row["region_norm"]).strip(),
                "state_norm": str(row["state_norm"]).strip(),
            }
            for row in (
                frame[["region", "state", "region_key", "region_norm", "state_norm"]]
                .drop_duplicates(subset=["region_key"])
                .to_dict(orient="records")
            )
        }
        region_lookup: dict[str, list[dict[str, Any]]] = {}
        for record in self.region_lookup_by_key.values():
            region_lookup.setdefault(record["region_norm"], []).append(record)
        self.region_lookup_by_region = region_lookup
        self.available = True

    def get_region_catalog(self) -> list[dict[str, Any]]:
        return list(self.region_catalog)

    def get_temporal_catalog(self) -> dict[str, Any]:
        return {
            "autofill_enabled": self.available,
            "buffer_strategy": self.buffer_strategy,
            "live_weather_status": self.live_weather_status,
            "target_months": [
                {"value": value, "label": label, "season": month_to_season(value)}
                for value, label in MONTH_LABELS.items()
            ],
            "seasons": [
                {"value": "kharif", "label": "Kharif"},
                {"value": "rabi", "label": "Rabi"},
                {"value": "zaid", "label": "Zaid"},
            ],
            "fallback_order": [
                "region_month",
                "region_all",
                "state_month",
                "state_all",
                "global_month",
                "global",
            ],
        }

    def resolve(
        self,
        region: str | None,
        state: str | None,
        target_time: str | None = None,
    ) -> dict[str, Any]:
        target_month, resolved_target_time = self._resolve_target_time(target_time)
        cache_key = (
            normalize_region_name(region),
            normalize_region_name(state),
            target_month,
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        empty_payload = {
            "available": False,
            "resolved_region": region or "",
            "resolved_state": state or "",
            "region_key": None,
            "target_time": resolved_target_time,
            "target_month": target_month,
            "target_season": month_to_season(target_month),
            "match_level": "unavailable",
            "context_confidence": 0.55,
            "buffer_strategy": self.buffer_strategy,
            "data_source": "training_defaults_only",
            "feature_defaults": {},
            "validation_bands": {},
            "feature_provenance": {},
            "crop_prior": [],
            "observation_counts": {},
        }
        if not self.available or self.frame.empty:
            self._cache[cache_key] = empty_payload
            return empty_payload

        resolved_region = region or ""
        resolved_state = state or ""
        region_key: str | None = None
        if region:
            region_key, resolved_region, resolved_state = self._resolve_region(region, state)

        climate_frame, climate_source = self._select_slice(
            region_key=region_key,
            state=resolved_state,
            target_month=target_month,
            prefer_month=True,
        )
        soil_frame, soil_source = self._select_slice(
            region_key=region_key,
            state=resolved_state,
            target_month=target_month,
            prefer_month=False,
        )
        prior_frame, prior_source = self._select_slice(
            region_key=region_key,
            state=resolved_state,
            target_month=target_month,
            prefer_month=True,
        )

        feature_defaults: dict[str, Any] = {}
        validation_bands: dict[str, dict[str, Any]] = {}
        feature_provenance: dict[str, str] = {}

        for feature in self.numeric_features:
            if feature in CLIMATE_FEATURES:
                source_frame = climate_frame
                source_name = climate_source
            elif feature in SOIL_FEATURES or feature in MANAGEMENT_FEATURES:
                source_frame = soil_frame
                source_name = soil_source
            else:
                source_frame = climate_frame if not climate_frame.empty else soil_frame
                source_name = climate_source if not climate_frame.empty else soil_source
            stats = self._numeric_stats(source_frame, feature)
            if stats:
                feature_defaults[feature] = stats["default"]
                validation_bands[feature] = stats
                feature_provenance[feature] = source_name

        for feature in self.categorical_features:
            if feature in LOCATION_CONTEXT_FEATURES:
                continue
            if feature in SOIL_FEATURES or feature in MANAGEMENT_FEATURES:
                source_frame = soil_frame
                source_name = soil_source
            else:
                source_frame = climate_frame if not climate_frame.empty else soil_frame
                source_name = climate_source if not climate_frame.empty else soil_source
            mode_value = self._mode_value(source_frame, feature)
            if mode_value is not None:
                feature_defaults[feature] = mode_value
                feature_provenance[feature] = source_name

        state_context = normalize_region_name(resolved_state).replace("  ", " ").strip()
        feature_defaults["state_context"] = state_context
        feature_defaults["region_context"] = (region_key or "").strip().lower()
        feature_defaults["target_month"] = target_month
        feature_defaults["target_season"] = month_to_season(target_month)
        feature_provenance["state_context"] = "request_context"
        feature_provenance["region_context"] = "request_context"
        feature_provenance["target_month"] = "request_context"
        feature_provenance["target_season"] = "request_context"

        observation_counts = {
            "climate_context_rows": int(len(climate_frame)),
            "soil_context_rows": int(len(soil_frame)),
            "prior_context_rows": int(len(prior_frame)),
        }
        context_confidence = round(
            min(
                1.0,
                (0.6 * SOURCE_WEIGHTS.get(climate_source, 0.58))
                + (0.4 * SOURCE_WEIGHTS.get(soil_source, 0.58)),
            ),
            4,
        )
        crop_prior = self._crop_prior(prior_frame, prior_source)
        resolved_payload = {
            "available": True,
            "resolved_region": resolved_region,
            "resolved_state": resolved_state,
            "region_key": region_key,
            "target_time": resolved_target_time,
            "target_month": target_month,
            "target_season": month_to_season(target_month),
            "match_level": climate_source,
            "context_confidence": context_confidence,
            "buffer_strategy": self.buffer_strategy,
            "data_source": "historical_same_month_climatology",
            "feature_defaults": feature_defaults,
            "validation_bands": validation_bands,
            "feature_provenance": feature_provenance,
            "crop_prior": crop_prior,
            "observation_counts": observation_counts,
            "source_summary": {
                "climate": climate_source,
                "soil": soil_source,
                "crop_prior": prior_source,
            },
        }
        self._cache[cache_key] = resolved_payload
        return resolved_payload

    def _resolve_target_time(self, target_time: str | None) -> tuple[str, str]:
        if target_time:
            parsed = pd.to_datetime(target_time, errors="coerce")
            if pd.notna(parsed):
                return parsed.strftime("%m"), parsed.strftime("%Y-%m")
            cleaned = str(target_time).strip()
            if cleaned.isdigit() and 1 <= int(cleaned) <= 12:
                month_value = f"{int(cleaned):02d}"
                return month_value, f"{datetime.utcnow().year}-{month_value}"
        now = datetime.now(timezone.utc)
        month_value = f"{int(now.month):02d}"
        return month_value, f"{now.year}-{month_value}"

    def _resolve_region(
        self,
        region: str,
        state: str | None,
    ) -> tuple[str | None, str, str]:
        state_norm = normalize_region_name(state)
        region_key = make_region_key(region, state or "")
        key_match = self.region_lookup_by_key.get(region_key.strip().lower())
        if key_match:
            return key_match["region_key"], key_match["region"], key_match["state"]

        region_norm = normalize_region_name(region)
        region_matches = self.region_lookup_by_region.get(region_norm, [])
        if state_norm:
            scoped_matches = [item for item in region_matches if item["state_norm"] == state_norm]
            if scoped_matches:
                match = scoped_matches[0]
                return match["region_key"], match["region"], match["state"]
        if len(region_matches) == 1:
            match = region_matches[0]
            return match["region_key"], match["region"], match["state"]
        return None, str(region).strip(), str(state or "").strip()

    def _select_slice(
        self,
        region_key: str | None,
        state: str | None,
        target_month: str,
        prefer_month: bool,
    ) -> tuple[pd.DataFrame, str]:
        if self.frame.empty:
            return self.frame.copy(), "global"

        state_norm = normalize_region_name(state)
        candidates: list[tuple[str, pd.DataFrame]] = []
        if prefer_month:
            if region_key:
                candidates.append(
                    (
                        "region_month",
                        self.frame[
                            self.frame["region_context"].eq(region_key.strip().lower())
                            & self.frame["target_month"].eq(target_month)
                        ],
                    )
                )
            if region_key:
                candidates.append(
                    (
                        "region_all",
                        self.frame[self.frame["region_context"].eq(region_key.strip().lower())],
                    )
                )
            if state_norm:
                candidates.append(
                    (
                        "state_month",
                        self.frame[
                            self.frame["state_norm"].eq(state_norm)
                            & self.frame["target_month"].eq(target_month)
                        ],
                    )
                )
                candidates.append(
                    (
                        "state_all",
                        self.frame[self.frame["state_norm"].eq(state_norm)],
                    )
                )
            candidates.append(("global_month", self.frame[self.frame["target_month"].eq(target_month)]))
            candidates.append(("global", self.frame))
        else:
            if region_key:
                candidates.append(
                    (
                        "region_all",
                        self.frame[self.frame["region_context"].eq(region_key.strip().lower())],
                    )
                )
            if state_norm:
                candidates.append(
                    (
                        "state_all",
                        self.frame[self.frame["state_norm"].eq(state_norm)],
                    )
                )
            candidates.append(("global", self.frame))

        for source_name, candidate in candidates:
            if not candidate.empty:
                return candidate, source_name
        return self.frame, "global"

    def _build_region_catalog(self) -> list[dict[str, Any]]:
        if self.frame.empty:
            return []
        region_frame = (
            self.frame[["state", "region"]]
            .fillna("")
            .drop_duplicates(subset=["state", "region"])
            .sort_values(["state", "region"])
        )
        states: list[dict[str, Any]] = []
        for state_name, state_frame in region_frame.groupby("state", dropna=False, sort=True):
            regions = [str(value).strip() for value in state_frame["region"].tolist() if str(value).strip()]
            if not regions:
                continue
            states.append(
                {
                    "state": str(state_name).strip() or "Unknown",
                    "region_count": len(regions),
                    "regions": regions,
                }
            )
        return states

    def _numeric_stats(self, frame: pd.DataFrame, feature: str) -> dict[str, Any]:
        if frame.empty or feature not in frame.columns:
            return {}
        series = pd.to_numeric(frame[feature], errors="coerce").dropna()
        if series.empty:
            return {}
        quantiles = series.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        return {
            "default": round(float(quantiles.loc[0.5]), 4),
            "typical_min": round(float(quantiles.loc[0.1]), 4),
            "typical_max": round(float(quantiles.loc[0.9]), 4),
            "q1": round(float(quantiles.loc[0.25]), 4),
            "q3": round(float(quantiles.loc[0.75]), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "mean": round(float(series.mean()), 4),
            "observation_count": int(len(series)),
        }

    def _mode_value(self, frame: pd.DataFrame, feature: str) -> str | None:
        if frame.empty or feature not in frame.columns:
            return None
        series = frame[feature].astype("string").str.strip().str.lower().replace("", pd.NA).dropna()
        if series.empty:
            return None
        modes = series.mode(dropna=True)
        if modes.empty:
            return None
        return str(modes.iloc[0]).strip().lower()

    def _crop_prior(self, frame: pd.DataFrame, source_name: str) -> list[dict[str, Any]]:
        if frame.empty:
            return []
        available_labels = [column for column in self.label_columns if column in frame.columns]
        if not available_labels:
            return []
        priors = frame[available_labels].apply(pd.to_numeric, errors="coerce").fillna(0.0).mean(axis=0)
        if float(priors.sum()) <= 0.0:
            return []
        priors = priors / priors.sum()
        rows = []
        for label, score in priors.sort_values(ascending=False).head(5).items():
            rows.append(
                {
                    "crop": label.removeprefix("crop_prob_").replace("_", " ").strip().title(),
                    "score": round(float(score), 6),
                    "source": source_name,
                }
            )
        return rows
