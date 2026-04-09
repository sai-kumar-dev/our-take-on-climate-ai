from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .training import (
    CropSuitabilityModelBundle,
    apply_scenario,
    crop_name_from_label,
    estimate_local_feature_effects,
    extract_feature_importance,
    humanize_feature_name,
    jensen_shannon_divergence,
    module_available,
)
from .runtime_context import LocalizedFeatureProvider, month_to_season
from .utils import ensure_parent_dir, resolve_path

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = ROOT_DIR / "artifacts" / "demo_training"
PREDICTION_CONTEXT_FIELDS = {
    "irrigation_index",
    "rotation_score",
    "fertility_class",
    "geo_confidence",
    "data_confidence",
    "time_step_missing",
    "climate_gap_filled",
    "soil_imputed",
}
UNIT_RANGE_FEATURES = {
    "irrigation_index",
    "rotation_score",
    "geo_confidence",
    "data_confidence",
}
CLIMATE_BUFFER_FEATURES = {
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
SOIL_RELATED_FEATURES = {"pH", "N", "P", "K", "soil_health_index"}
LOCATION_CONTEXT_FEATURES = {"state_context", "region_context", "target_month", "target_season"}
PRESET_SCENARIOS = {
    "low_rainfall": {
        "display_name": "Low Rainfall",
        "feature_multipliers": {
            "rain_total": 0.6,
            "rain_lag_14": 0.6,
            "humidity_avg": 0.9,
        },
    },
    "heatwave": {
        "display_name": "Heatwave",
        "feature_additions": {
            "temp_avg": 4.0,
            "max_temp": 5.0,
            "max_temp_3d": 5.0,
            "humidity_avg": -5.0,
        },
    },
    "high_irrigation": {
        "display_name": "High Irrigation",
        "feature_additions": {
            "irrigation_index": 0.3,
        },
    },
}


@dataclass
class PreparedInferenceInput:
    frame: pd.DataFrame
    warnings: list[str]
    input_quality: dict[str, Any]
    region: str | None
    data_confidence: float
    geo_confidence: float
    drift_report: dict[str, Any]
    localized_context: dict[str, Any]


class InferenceValidationError(ValueError):
    """Raised when inference input cannot be safely processed."""


class CropSuitabilityInferenceService:
    """Loads trained artifacts and serves validated crop suitability predictions."""

    def __init__(
        self,
        artifact_dir: Path,
        model_bundle: CropSuitabilityModelBundle,
        scaler: Any,
        feature_config: dict[str, Any],
        logger: logging.Logger,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.model_bundle = model_bundle
        self.scaler = scaler
        self.feature_config = feature_config
        self.logger = logger

        preprocessor_cfg = feature_config["preprocessor"]
        self.numeric_features = preprocessor_cfg["numeric_features"]
        self.categorical_features = preprocessor_cfg["categorical_features"]
        self.categorical_levels = preprocessor_cfg.get("categorical_levels", {})
        self.numeric_fill_values = preprocessor_cfg.get("numeric_fill_values", {})
        self.categorical_fill_value = preprocessor_cfg.get("categorical_fill_value", "unknown")
        self.numeric_summary = (
            feature_config.get("training_metadata", {})
            .get("dataset_profile", {})
            .get("numeric_summary", {})
        )
        self.label_columns = feature_config["labels"]
        self.mode = str(feature_config.get("mode", "demo")).casefold()
        self.sanity_mode = str(feature_config.get("sanity_mode", "warn")).casefold()
        self.model_metadata = feature_config.get("model_metadata", {})
        self.model_version = self.model_metadata.get("model_version", "unknown")
        self.inference_settings = feature_config.get("inference_settings", {})
        self.calibration_config = feature_config.get("calibration", {})
        self.required_features = list(self.inference_settings.get("required_features", []))
        self.drift_zscore_threshold = float(self.inference_settings.get("drift_zscore_threshold", 3.0))
        self.warmup_enabled = bool(self.inference_settings.get("warmup_enabled", True))
        self.warmup_explainer_count = int(self.inference_settings.get("warmup_explainer_count", 2))
        self.evaluation_report = self._load_json_artifact("evaluation_report.json")
        self.sanity_checks = self.evaluation_report.get("sanity_checks", {})
        dataset_path = self.feature_config.get("training_metadata", {}).get("dataset_profile", {}).get("dataset_path")
        self.context_provider = LocalizedFeatureProvider(
            dataset_path=Path(str(dataset_path)) if dataset_path else None,
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            label_columns=self.label_columns,
        )
        self._explainer_cache: dict[int, Any] = {}
        self._warmup_completed = False
        self._catalog_cache: dict[str, Any] | None = None

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: str | Path | None = None,
        root_dir: Path | None = None,
    ) -> CropSuitabilityInferenceService:
        root = root_dir or ROOT_DIR
        resolved_dir = resolve_path(root, str(artifact_dir or DEFAULT_ARTIFACT_DIR))
        model_path = resolved_dir / "trained_model.pkl"
        if not model_path.exists():
            versioned_models = sorted(resolved_dir.glob("model_v*.pkl"))
            if not versioned_models:
                raise FileNotFoundError(f"No model artifact found in {resolved_dir}")
            model_path = versioned_models[-1]

        model_bundle = joblib.load(model_path)
        scaler = joblib.load(resolved_dir / "scaler.pkl")
        with (resolved_dir / "feature_config.json").open("r", encoding="utf-8") as handle:
            feature_config = json.load(handle)
        logger = configure_inference_logger(root)
        logger.info(
            "inference_service_loaded artifact_dir=%s backend=%s model_version=%s mode=%s",
            resolved_dir,
            model_bundle.backend,
            feature_config.get("model_metadata", {}).get("model_version", "unknown"),
            feature_config.get("mode", "unknown"),
        )
        return cls(
            artifact_dir=resolved_dir,
            model_bundle=model_bundle,
            scaler=scaler,
            feature_config=feature_config,
            logger=logger,
        )

    def predict(self, payload: dict[str, Any], top_n: int = 3) -> dict[str, Any]:
        prepared = self.prepare_input(payload)
        probabilities = self._predict_probabilities(prepared.frame)
        return self._build_prediction_response(
            frame=prepared.frame,
            probabilities=probabilities,
            top_n=top_n,
            region=prepared.region,
            warnings=prepared.warnings,
            input_quality=prepared.input_quality,
            data_confidence=prepared.data_confidence,
            geo_confidence=prepared.geo_confidence,
            drift_report=prepared.drift_report,
            localized_context=prepared.localized_context,
        )

    def simulate_scenarios(
        self,
        payload: dict[str, Any],
        scenario_names: list[str] | None = None,
        top_n: int = 3,
    ) -> dict[str, Any]:
        prepared = self.prepare_input(payload)
        base_probabilities = self._predict_probabilities(prepared.frame)
        base_prediction = self._build_prediction_response(
            frame=prepared.frame,
            probabilities=base_probabilities,
            top_n=top_n,
            region=prepared.region,
            warnings=prepared.warnings,
            input_quality=prepared.input_quality,
            data_confidence=prepared.data_confidence,
            geo_confidence=prepared.geo_confidence,
            drift_report=prepared.drift_report,
            localized_context=prepared.localized_context,
        )

        selected_names = scenario_names or list(PRESET_SCENARIOS.keys())
        scenario_results: dict[str, Any] = {}
        for scenario_name in selected_names:
            if scenario_name not in PRESET_SCENARIOS:
                continue

            scenario_cfg = PRESET_SCENARIOS[scenario_name]
            scenario_frame = apply_scenario(prepared.frame, scenario_cfg)
            scenario_frame = self._clip_frame_to_safe_ranges(scenario_frame)
            scenario_probabilities = self._predict_probabilities(scenario_frame)
            scenario_drift_report, scenario_drift_warnings = self._detect_feature_drift(scenario_frame)
            scenario_prediction = self._build_prediction_response(
                frame=scenario_frame,
                probabilities=scenario_probabilities,
                top_n=top_n,
                region=prepared.region,
                warnings=[f"Scenario applied: {scenario_cfg['display_name']}.", *scenario_drift_warnings],
                input_quality={**prepared.input_quality, "scenario": scenario_name},
                data_confidence=prepared.data_confidence,
                geo_confidence=prepared.geo_confidence,
                drift_report=scenario_drift_report,
                localized_context=prepared.localized_context,
            )
            scenario_results[scenario_name] = {
                "display_name": scenario_cfg["display_name"],
                "prediction": scenario_prediction,
                "comparison": self._build_comparison_table(base_probabilities, scenario_probabilities),
            }

        return {
            "base_prediction": base_prediction,
            "scenario_results": scenario_results,
            "available_scenarios": [
                {"name": name, "display_name": cfg["display_name"]}
                for name, cfg in PRESET_SCENARIOS.items()
            ],
            "model_version": self.model_version,
        }

    def warmup(self) -> None:
        if self._warmup_completed or not self.warmup_enabled:
            return

        warmup_row: dict[str, Any] = {}
        for feature in self.numeric_features:
            summary = self.numeric_summary.get(feature, {})
            warmup_row[feature] = float(summary.get("mean", self.numeric_fill_values.get(feature, 0.0)))
        for feature in self.categorical_features:
            allowed = self.categorical_levels.get(feature, [])
            warmup_row[feature] = str(allowed[0]).strip().lower() if allowed else self.categorical_fill_value

        warmup_frame = pd.DataFrame([warmup_row])
        _ = self._predict_probabilities(warmup_frame)
        if module_available("shap"):
            for label_index in range(min(self.warmup_explainer_count, len(self.label_columns))):
                try:
                    self._get_shap_explainer(label_index)
                except Exception:
                    break

        self._warmup_completed = True
        self.logger.info(
            "inference_warmup_complete model_version=%s warmup_explainer_count=%s",
            self.model_version,
            min(self.warmup_explainer_count, len(self.label_columns)),
        )

    def prepare_input(self, payload: dict[str, Any]) -> PreparedInferenceInput:
        features = dict(payload.get("features") or {})
        for field in PREDICTION_CONTEXT_FIELDS:
            if field in payload and payload[field] is not None:
                features[field] = payload[field]

        if not features:
            raise InferenceValidationError("No feature values provided. Supply a 'features' object with model inputs.")

        localized_context = self.context_provider.resolve(
            region=payload.get("region"),
            state=payload.get("state"),
            target_time=payload.get("target_time"),
        )
        feature_defaults = dict(localized_context.get("feature_defaults", {}))
        provided_features = {key for key, value in features.items() if value is not None and value != ""}
        merged_features = {**feature_defaults, **features}
        merged_features = self._apply_context_consistency_adjustments(
            merged_features=merged_features,
            feature_defaults=feature_defaults,
            provided_features=provided_features,
        )

        warnings: list[str] = []
        provided_feature_count = 0
        filled_feature_count = 0
        context_autofill_count = 0
        clipped_feature_count = 0
        invalid_feature_count = 0
        localized_outlier_count = 0
        missing_required_features: list[str] = []

        row: dict[str, Any] = {}
        for feature in self.numeric_features:
            raw_value = merged_features.get(feature)
            if raw_value is None or raw_value == "":
                row[feature] = float(self.numeric_fill_values.get(feature, 0.0))
                filled_feature_count += 1
                warnings.append(f"Missing numeric feature '{feature}' filled with training default.")
                if feature in self.required_features:
                    missing_required_features.append(feature)
                continue

            try:
                numeric_value = float(raw_value)
                if feature in provided_features:
                    provided_feature_count += 1
                    if self._is_outside_local_band(feature, numeric_value, localized_context):
                        localized_outlier_count += 1
                        warnings.append(self._local_band_warning(feature, payload, localized_context))
                else:
                    context_autofill_count += 1
            except (TypeError, ValueError):
                row[feature] = float(self.numeric_fill_values.get(feature, 0.0))
                filled_feature_count += 1
                invalid_feature_count += 1
                warnings.append(f"Invalid numeric feature '{feature}' replaced with training default.")
                if feature in self.required_features:
                    missing_required_features.append(feature)
                continue

            clipped_value, was_clipped = self._clip_numeric_value(feature, numeric_value)
            row[feature] = clipped_value
            if was_clipped:
                clipped_feature_count += 1
                warnings.append(f"Feature '{feature}' was clipped to a safe range.")

        for feature in self.categorical_features:
            raw_value = merged_features.get(feature)
            if raw_value is None or raw_value == "":
                row[feature] = self.categorical_fill_value
                filled_feature_count += 1
                warnings.append(f"Missing categorical feature '{feature}' filled with '{self.categorical_fill_value}'.")
                if feature in self.required_features:
                    missing_required_features.append(feature)
                continue

            value = str(raw_value).strip().lower()
            allowed = {str(item).strip().lower() for item in self.categorical_levels.get(feature, [])}
            if allowed and value not in allowed:
                row[feature] = self.categorical_fill_value
                clipped_feature_count += 1
                warnings.append(f"Categorical feature '{feature}' unseen during training; using fallback token.")
                if feature in self.required_features and self.mode == "production":
                    missing_required_features.append(feature)
            else:
                row[feature] = value if value else self.categorical_fill_value
                if feature in provided_features:
                    provided_feature_count += 1
                else:
                    context_autofill_count += 1

        if self.mode == "production" and missing_required_features:
            required_display = sorted(set(missing_required_features))
            raise InferenceValidationError(
                f"Production mode requires these features: {required_display}"
            )

        frame = pd.DataFrame([row])
        total_features = len(self.numeric_features) + len(self.categorical_features)
        completeness = 1.0 - (filled_feature_count / total_features if total_features else 0.0)
        quality_score = max(
            0.0,
            min(
                1.0,
                completeness
                - (0.05 * clipped_feature_count)
                - (0.10 * invalid_feature_count)
                - (0.04 * localized_outlier_count),
            ),
        )
        drift_report, drift_warnings = self._detect_feature_drift(frame)
        warnings.extend(drift_warnings)
        input_quality = {
            "provided_feature_count": provided_feature_count,
            "filled_feature_count": filled_feature_count,
            "context_autofill_count": context_autofill_count,
            "clipped_feature_count": clipped_feature_count,
            "invalid_feature_count": invalid_feature_count,
            "localized_outlier_count": localized_outlier_count,
            "quality_score": round(quality_score, 4),
            "mode": self.mode,
            "context_source": localized_context.get("match_level"),
            "context_confidence": localized_context.get("context_confidence"),
            "target_month": localized_context.get("target_month"),
            "target_season": localized_context.get("target_season"),
        }

        quality_fallback = quality_score * max(0.65, float(localized_context.get("context_confidence", 0.65)))
        data_confidence_raw = features.get("data_confidence")
        geo_confidence_raw = features.get("geo_confidence")
        data_confidence = self._resolve_confidence_value(data_confidence_raw, quality_fallback)
        geo_confidence = self._resolve_confidence_value(
            geo_confidence_raw,
            float(localized_context.get("context_confidence", frame.get("geo_confidence", pd.Series([1.0])).iloc[0])),
        )

        self.logger.info(
            "input_prepared region=%s mode=%s provided=%s filled=%s context_autofilled=%s clipped=%s invalid=%s localized_outliers=%s drifted=%s",
            payload.get("region", "unknown"),
            self.mode,
            provided_feature_count,
            filled_feature_count,
            context_autofill_count,
            clipped_feature_count,
            invalid_feature_count,
            localized_outlier_count,
            drift_report["drifted_feature_count"],
        )
        return PreparedInferenceInput(
            frame=frame,
            warnings=deduplicate_preserve_order(warnings),
            input_quality=input_quality,
            region=payload.get("region"),
            data_confidence=data_confidence,
            geo_confidence=geo_confidence,
            drift_report=drift_report,
            localized_context=localized_context,
        )

    def _apply_context_consistency_adjustments(
        self,
        merged_features: dict[str, Any],
        feature_defaults: dict[str, Any],
        provided_features: set[str],
    ) -> dict[str, Any]:
        adjusted = dict(merged_features)
        if not feature_defaults:
            return adjusted

        base_temp = self._safe_float(feature_defaults.get("temp_avg"), fallback=None)
        current_temp = self._safe_float(adjusted.get("temp_avg"), fallback=base_temp)
        if current_temp is not None and base_temp is not None:
            temp_delta = current_temp - base_temp
            for feature in ["temp_lag_7", "max_temp", "max_temp_3d"]:
                if feature not in provided_features:
                    baseline = self._safe_float(feature_defaults.get(feature), fallback=current_temp)
                    adjusted[feature] = round(float(baseline + temp_delta), 4)

        base_rain = self._safe_float(feature_defaults.get("rain_total"), fallback=None)
        current_rain = self._safe_float(adjusted.get("rain_total"), fallback=base_rain)
        if current_rain is not None and base_rain is not None:
            baseline = max(base_rain, 1e-6)
            rain_ratio = max(min(current_rain / baseline, 4.0), 0.0)
            if "rain_lag_14" not in provided_features:
                lag_rain = self._safe_float(feature_defaults.get("rain_lag_14"), fallback=current_rain)
                adjusted["rain_lag_14"] = round(float(lag_rain * max(rain_ratio, 0.1)), 4)
            if "max_rain_1d" not in provided_features:
                max_rain = self._safe_float(feature_defaults.get("max_rain_1d"), fallback=max(current_rain * 0.3, 0.0))
                adjusted["max_rain_1d"] = round(float(max_rain * max(rain_ratio, 0.1)), 4)
            if "rain_variance" not in provided_features:
                rain_variance = self._safe_float(
                    feature_defaults.get("rain_variance"),
                    fallback=max((current_rain * 0.22) ** 2, 0.0),
                )
                adjusted["rain_variance"] = round(float(rain_variance * max(rain_ratio, 0.1) ** 2), 4)
            if "dry_spell_days" not in provided_features:
                default_dry_spell = self._safe_float(
                    feature_defaults.get("dry_spell_days"),
                    fallback=max(0.0, min(31.0, 31.0 - (current_rain / 5.0))),
                )
                inverse_ratio = 1.0 if rain_ratio <= 0 else min(4.0, 1.0 / max(rain_ratio, 0.1))
                adjusted["dry_spell_days"] = round(float(min(31.0, max(0.0, default_dry_spell * inverse_ratio))), 4)

        if not {"soil_health_index"} & provided_features:
            soil_health = self._compute_soil_health_index(adjusted)
            if soil_health is not None:
                adjusted["soil_health_index"] = soil_health

        if "target_season" not in provided_features and adjusted.get("target_month"):
            adjusted["target_season"] = month_to_season(str(adjusted["target_month"]))
        return adjusted

    def _is_outside_local_band(
        self,
        feature: str,
        value: float,
        localized_context: dict[str, Any],
    ) -> bool:
        validation_bands = localized_context.get("validation_bands", {})
        feature_band = validation_bands.get(feature, {})
        if not feature_band:
            return False
        lower = feature_band.get("typical_min")
        upper = feature_band.get("typical_max")
        if lower is None or upper is None:
            return False
        return float(value) < float(lower) or float(value) > float(upper)

    def _local_band_warning(
        self,
        feature: str,
        payload: dict[str, Any],
        localized_context: dict[str, Any],
    ) -> str:
        feature_band = localized_context.get("validation_bands", {}).get(feature, {})
        lower = feature_band.get("typical_min")
        upper = feature_band.get("typical_max")
        region = localized_context.get("resolved_region") or payload.get("region") or "this region"
        month = localized_context.get("target_month") or "the chosen month"
        return (
            f"Feature '{feature}' is outside the typical local range for {region} in month {month} "
            f"({lower} to {upper})."
        )

    def _compute_soil_health_index(self, features: dict[str, Any]) -> float | None:
        ph = self._safe_float(features.get("pH"), fallback=None)
        n_value = self._safe_float(features.get("N"), fallback=None)
        p_value = self._safe_float(features.get("P"), fallback=None)
        k_value = self._safe_float(features.get("K"), fallback=None)
        if None in {ph, n_value, p_value, k_value}:
            return None
        ph_score = max(0.0, 1.0 - abs(float(ph) - 6.8) / 2.0)
        n_score = min(max(float(n_value) / 560.0, 0.0), 1.0)
        p_score = min(max(float(p_value) / 25.0, 0.0), 1.0)
        k_score = min(max(float(k_value) / 280.0, 0.0), 1.0)
        return round(((ph_score + n_score + p_score + k_score) / 4.0) * 100.0, 4)

    def _safe_float(self, value: Any, fallback: float | None) -> float | None:
        try:
            if value is None or value == "":
                return fallback
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def _predict_probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities = np.asarray(self.model_bundle.predict(frame)[0], dtype=float)
        return probabilities / np.clip(probabilities.sum(), 1e-9, None)

    def _build_prediction_response(
        self,
        frame: pd.DataFrame,
        probabilities: np.ndarray,
        top_n: int,
        region: str | None,
        warnings: list[str],
        input_quality: dict[str, Any],
        data_confidence: float,
        geo_confidence: float,
        drift_report: dict[str, Any],
        localized_context: dict[str, Any],
    ) -> dict[str, Any]:
        rule_distribution = self._rule_based_distribution(frame)
        rule_model_agreement = float(
            np.clip(1.0 - jensen_shannon_divergence(probabilities, rule_distribution)[0], 0.0, 1.0)
        )
        recommendations = self._rank_predictions(probabilities, top_n)
        confidence_breakdown = {
            "data_confidence": round(data_confidence, 4),
            "geo_confidence": round(geo_confidence, 4),
            "rule_model_agreement": round(rule_model_agreement, 4),
        }
        confidence = self._compute_confidence(**confidence_breakdown)
        explanation_payload = self._build_explanation_payload(frame, probabilities)

        response = {
            "recommendations": recommendations,
            "confidence": confidence,
            "confidence_breakdown": confidence_breakdown,
            "confidence_components": confidence_breakdown,
            "explanation": explanation_payload["explanation"],
            "top_features": explanation_payload["top_features"],
            "why_not": explanation_payload["why_not"],
            "explanation_method": explanation_payload["method"],
            "warnings": deduplicate_preserve_order(warnings),
            "input_quality": input_quality,
            "drift_report": drift_report,
            "model_version": self.model_version,
            "localized_context": localized_context,
        }
        self.logger.info(
            "prediction_complete region=%s model_version=%s top_crop=%s confidence=%.4f warnings=%s drift_detected=%s explanation_method=%s",
            region or "unknown",
            self.model_version,
            recommendations[0]["crop"] if recommendations else "none",
            confidence,
            len(warnings),
            drift_report["drift_detected"],
            explanation_payload["method"],
        )
        return response

    def _clip_frame_to_safe_ranges(self, frame: pd.DataFrame) -> pd.DataFrame:
        clipped_frame = frame.copy()
        for feature in self.numeric_features:
            if feature not in clipped_frame.columns:
                continue
            value = float(pd.to_numeric(clipped_frame[feature], errors="coerce").iloc[0])
            clipped_value, _ = self._clip_numeric_value(feature, value)
            clipped_frame.at[0, feature] = clipped_value
        return clipped_frame

    def _build_comparison_table(
        self,
        base_probabilities: np.ndarray,
        scenario_probabilities: np.ndarray,
    ) -> dict[str, Any]:
        base_ranks = np.argsort(base_probabilities)[::-1]
        scenario_ranks = np.argsort(scenario_probabilities)[::-1]
        base_rank_lookup = {int(index): rank + 1 for rank, index in enumerate(base_ranks)}
        scenario_rank_lookup = {int(index): rank + 1 for rank, index in enumerate(scenario_ranks)}

        rows = []
        for label_index, label in enumerate(self.label_columns):
            index = int(label_index)
            rows.append(
                {
                    "crop": crop_name_from_label(label),
                    "base_score": round(float(base_probabilities[index]), 6),
                    "scenario_score": round(float(scenario_probabilities[index]), 6),
                    "score_delta": round(float(scenario_probabilities[index] - base_probabilities[index]), 6),
                    "base_rank": base_rank_lookup[index],
                    "scenario_rank": scenario_rank_lookup[index],
                    "rank_change": base_rank_lookup[index] - scenario_rank_lookup[index],
                }
            )
        rows.sort(key=lambda item: item["scenario_rank"])
        return {
            "top_crop_changed": int(base_ranks[0]) != int(scenario_ranks[0]),
            "rows": rows,
        }

    def get_sanity_summary(self) -> dict[str, Any]:
        if not self.sanity_checks:
            return {
                "available": False,
                "reason": "No sanity check artifact available.",
            }
        if not self.sanity_checks.get("available", False):
            return self.sanity_checks

        scenarios = {}
        for name, result in self.sanity_checks.get("scenarios", {}).items():
            scenarios[name] = {
                "status": result.get("status", "UNKNOWN"),
                "passed": bool(result.get("passed", False)),
                "expected_direction": result.get("expected_direction"),
                "group_probability_delta": round(float(result.get("group_probability_delta", 0.0)), 4),
                "min_group_delta": round(float(result.get("min_group_delta", 0.0)), 4),
                "top_predictions": result.get("top_predictions", []),
            }

        return {
            "available": True,
            "baseline_top_predictions": self.sanity_checks.get("baseline_top_predictions", []),
            "scenarios": scenarios,
        }

    def get_catalog(self) -> dict[str, Any]:
        if self._catalog_cache is not None:
            return self._catalog_cache

        dataset_profile = self.feature_config.get("training_metadata", {}).get("dataset_profile", {})
        states = self.context_provider.get_region_catalog() or self._load_region_catalog(dataset_profile.get("dataset_path"))
        supported_crops = [crop_name_from_label(label) for label in self.label_columns]

        coverage = {
            "row_count": int(dataset_profile.get("rows", 0) or 0),
            "region_count": int(dataset_profile.get("region_count", 0) or 0),
            "state_count": len(states),
            "crop_count": len(supported_crops),
            "time_min": dataset_profile.get("time_min"),
            "time_max": dataset_profile.get("time_max"),
        }

        self._catalog_cache = {
            "model_version": self.model_version,
            "mode": self.mode,
            "coverage": coverage,
            "states": states,
            "supported_crops": supported_crops,
            "numeric_features": {
                feature: self._build_numeric_feature_catalog(feature)
                for feature in self.numeric_features
            },
            "categorical_features": {
                feature: {
                    "levels": list(self.categorical_levels.get(feature, [])),
                    "default": self.categorical_fill_value,
                }
                for feature in self.categorical_features
            },
            "available_scenarios": [
                {"name": name, "display_name": cfg["display_name"]}
                for name, cfg in PRESET_SCENARIOS.items()
            ],
            "temporal_context": self.context_provider.get_temporal_catalog(),
        }
        return self._catalog_cache

    def get_localized_context(
        self,
        region: str | None,
        state: str | None,
        target_time: str | None,
    ) -> dict[str, Any]:
        return self.context_provider.resolve(region=region, state=state, target_time=target_time)

    def _resolve_confidence_value(self, raw_value: Any, fallback: float) -> float:
        try:
            if raw_value is None or raw_value == "":
                return float(np.clip(fallback, 0.0, 1.0))
            return float(np.clip(float(raw_value), 0.0, 1.0))
        except (TypeError, ValueError):
            return float(np.clip(fallback, 0.0, 1.0))

    def _clip_numeric_value(self, feature: str, value: float) -> tuple[float, bool]:
        if feature in UNIT_RANGE_FEATURES:
            clipped = float(np.clip(value, 0.0, 1.0))
            return clipped, clipped != value

        if feature == "pH":
            clipped = float(np.clip(value, 0.0, 14.0))
            return clipped, clipped != value

        summary = self.numeric_summary.get(feature, {})
        if not summary:
            return value, False

        feature_min = float(summary.get("min", value))
        feature_max = float(summary.get("max", value))
        feature_std = abs(float(summary.get("std", 0.0) or 0.0))
        margin = max(feature_std * 3.0, abs(feature_max - feature_min) * 0.25, 0.5)
        lower_bound = feature_min - margin
        upper_bound = feature_max + margin
        clipped = float(np.clip(value, lower_bound, upper_bound))
        return clipped, clipped != value

    def _detect_feature_drift(self, frame: pd.DataFrame) -> tuple[dict[str, Any], list[str]]:
        drifted_features: list[dict[str, Any]] = []
        warnings: list[str] = []
        for feature in self.numeric_features:
            if feature not in frame.columns or feature not in self.numeric_summary:
                continue
            value = float(pd.to_numeric(frame[feature], errors="coerce").iloc[0])
            summary = self.numeric_summary[feature]
            mean_value = float(summary.get("mean", value))
            std_value = abs(float(summary.get("std", 0.0) or 0.0))

            if std_value > 1e-9:
                zscore = abs(value - mean_value) / std_value
            else:
                observed_min = float(summary.get("min", mean_value))
                observed_max = float(summary.get("max", mean_value))
                tolerance = max(abs(mean_value) * 0.25, abs(observed_max - observed_min), 0.15)
                if feature in UNIT_RANGE_FEATURES:
                    tolerance = max(tolerance, 0.5)
                if abs(value - mean_value) <= tolerance:
                    zscore = 0.0
                else:
                    zscore = abs(value - mean_value) / max(tolerance, 1e-9)

            if zscore > self.drift_zscore_threshold:
                drifted_features.append(
                    {
                        "feature": feature,
                        "value": value,
                        "training_mean": mean_value,
                        "zscore": round(float(zscore), 4),
                    }
                )
                warnings.append(f"Feature drift detected for '{feature}'.")

        drift_report = {
            "drift_detected": bool(drifted_features),
            "drifted_feature_count": len(drifted_features),
            "threshold": self.drift_zscore_threshold,
            "drifted_features": drifted_features[:10],
        }
        if drift_report["drift_detected"]:
            self.logger.warning("feature_drift_detected features=%s", [item["feature"] for item in drifted_features[:10]])
        return drift_report, warnings

    def _rank_predictions(self, probabilities: np.ndarray, top_n: int) -> list[dict[str, Any]]:
        ranked_indices = np.argsort(probabilities)[::-1][:top_n]
        return [
            {
                "crop": crop_name_from_label(self.label_columns[int(index)]),
                "score": round(float(probabilities[int(index)]), 6),
            }
            for index in ranked_indices
        ]

    def _compute_confidence(
        self,
        data_confidence: float,
        geo_confidence: float,
        rule_model_agreement: float,
    ) -> float:
        confidence = (
            0.45 * float(np.clip(data_confidence, 0.0, 1.0))
            + 0.20 * float(np.clip(geo_confidence, 0.0, 1.0))
            + 0.35 * float(np.clip(rule_model_agreement, 0.0, 1.0))
        )
        return round(float(np.clip(confidence, 0.0, 1.0)), 4)

    def _rule_based_distribution(self, frame: pd.DataFrame) -> np.ndarray:
        row = frame.iloc[0]
        rain_total = float(pd.to_numeric(row.get("rain_total", 0.0), errors="coerce") or 0.0)
        irrigation = float(pd.to_numeric(row.get("irrigation_index", 0.0), errors="coerce") or 0.0)
        temp_avg = float(pd.to_numeric(row.get("temp_avg", 0.0), errors="coerce") or 0.0)
        humidity = float(pd.to_numeric(row.get("humidity_avg", 0.0), errors="coerce") or 0.0)
        soil_ph = float(pd.to_numeric(row.get("pH", 6.5), errors="coerce") or 6.5)

        low_rain = max(0.0, min(1.0, (60.0 - rain_total) / 60.0))
        high_rain = max(0.0, min(4.0, (rain_total - 80.0) / 300.0))
        heat = max(0.0, min(2.5, (temp_avg - 26.0) / 6.0))
        cool = max(0.0, min(1.5, (26.0 - temp_avg) / 6.0))
        humid = max(0.0, min(2.0, (humidity - 60.0) / 25.0))
        ph_balance = max(0.0, 1.0 - min(abs(soil_ph - 6.5) / 2.5, 1.0))

        scores: list[float] = []
        for label in self.label_columns:
            crop = crop_name_from_label(label).casefold()
            score = 1.0
            if crop == "rice":
                score += 1.2 * high_rain + 0.8 * irrigation + 0.4 * humid - 0.8 * heat - 0.6 * low_rain
            elif crop in {"sugarcane", "banana", "coconut"}:
                score += 1.0 * high_rain + 0.6 * irrigation + 0.4 * humid - 0.5 * low_rain - 0.3 * heat
            elif crop in {"millet", "ragi"}:
                score += 0.9 * low_rain + 0.5 * heat - 0.4 * high_rain
            elif crop == "maize":
                score += 0.4 * high_rain + 0.3 * cool + 0.2 * ph_balance
            elif crop in {"jowar", "small millets"}:
                score += 0.8 * low_rain + 0.4 * heat - 0.3 * high_rain
            elif crop == "grapes":
                score += 0.4 * cool + 0.3 * ph_balance - 0.3 * high_rain
            elif crop in {"onion", "potato", "wheat"}:
                score += 0.5 * cool + 0.2 * ph_balance - 0.4 * heat - 0.2 * high_rain
            elif crop in {"black pepper", "cardamom"}:
                score += 0.5 * humid + 0.2 * ph_balance - 0.3 * heat - 0.2 * low_rain
            else:
                score += 0.15 * ph_balance
            scores.append(max(score, 0.01))

        vector = np.asarray(scores, dtype=float)
        vector = vector / np.clip(vector.sum(), 1e-9, None)
        return vector

    def _build_explanation_payload(self, frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, Any]:
        top_indices = np.argsort(probabilities)[::-1]
        top_index = int(top_indices[0])
        runner_up_index = int(top_indices[1]) if len(top_indices) > 1 else int(top_indices[0])

        method = "counterfactual_feature_impact"
        top_features = self._top_shap_impacts(frame, top_index, top_n=5)
        if top_features:
            method = "shap"
        else:
            top_features = self._counterfactual_feature_impacts(frame, top_index, top_n=5)

        explanation = self._compose_explanation_sentence(top_features, top_index, runner_up_index)
        why_not = self._build_why_not_reasons(frame, probabilities, top_index, top_indices[1:4], top_features)
        return {
            "method": method,
            "explanation": explanation,
            "top_features": top_features,
            "why_not": why_not,
        }

    def _compose_explanation_sentence(
        self,
        top_features: list[dict[str, Any]],
        top_index: int,
        runner_up_index: int,
    ) -> str:
        positive_features = [item for item in top_features if float(item.get("impact", 0.0)) >= 0.0]
        chosen_features = positive_features[:2]
        top_crop = crop_name_from_label(self.label_columns[top_index])
        runner_up_crop = crop_name_from_label(self.label_columns[runner_up_index])

        if chosen_features:
            descriptors = [item.get("descriptor") or item.get("feature", "feature") for item in chosen_features]
            verb = "favor" if len(descriptors) > 1 else "favors"
            return f"{' and '.join(descriptors)} {verb} {top_crop} over {runner_up_crop}."

        fallback_descriptors = [item.get("feature", "feature") for item in top_features[:2]]
        if fallback_descriptors:
            return f"{top_crop} stays ahead of {runner_up_crop}, led mainly by {' and '.join(fallback_descriptors)}."
        return f"Top predicted crop is {top_crop} based on the available climate, soil, and management signals."

    def _build_why_not_reasons(
        self,
        frame: pd.DataFrame,
        probabilities: np.ndarray,
        top_index: int,
        alternative_indices: np.ndarray,
        top_features: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidate_features = [
            item.get("feature_key")
            for item in top_features
            if item.get("feature_key") in self.numeric_features
        ]
        if not candidate_features:
            candidate_features = self._top_importance_features(top_index, top_n=6)

        reasons: list[dict[str, Any]] = []
        for alternative_index in alternative_indices:
            alt_index = int(alternative_index)
            comparative_effects = estimate_local_feature_effects(
                model_bundle=self.model_bundle,
                reference_frame=frame,
                candidate_features=candidate_features,
                top_index=top_index,
                runner_up_index=alt_index,
            )
            alternative_crop = crop_name_from_label(self.label_columns[alt_index])
            top_crop = crop_name_from_label(self.label_columns[top_index])
            if comparative_effects:
                phrases = [
                    f"{effect['direction']} {humanize_feature_name(effect['feature'])}"
                    for effect in comparative_effects[:2]
                ]
                verb = "favor" if len(phrases) > 1 else "favors"
                reason = (
                    f"{alternative_crop.capitalize()} is less suitable because {' and '.join(phrases)} "
                    f"{verb} {top_crop} instead."
                )
            else:
                reason = f"{alternative_crop.capitalize()} ranks below {top_crop} under the current conditions."
            reasons.append(
                {
                    "crop": alternative_crop,
                    "score_gap": round(float(probabilities[top_index] - probabilities[alt_index]), 4),
                    "reason": reason,
                }
            )
        return reasons

    def _top_importance_features(self, label_index: int, top_n: int) -> list[str]:
        estimator = self._estimator_for_label(label_index)
        raw_importance = extract_feature_importance(
            estimator,
            len(self.model_bundle.preprocessor.feature_names_out),
        )
        source_scores: dict[str, float] = {}
        for source_name, score in zip(self.model_bundle.preprocessor.feature_sources_out, raw_importance, strict=False):
            source_scores[source_name] = source_scores.get(source_name, 0.0) + float(score)
        ranked = sorted(source_scores.items(), key=lambda item: item[1], reverse=True)
        return [feature for feature, _ in ranked[:top_n]]

    def _counterfactual_feature_impacts(
        self,
        frame: pd.DataFrame,
        label_index: int,
        top_n: int,
    ) -> list[dict[str, Any]]:
        base_prediction = self.model_bundle.predict(frame)[0]
        candidates = self._top_importance_features(label_index, top_n=max(top_n * 2, 6))
        if not candidates:
            candidates = self.numeric_features[: max(top_n * 2, 6)]

        impacts: dict[str, float] = {}
        for feature in candidates:
            if feature not in frame.columns:
                continue
            baseline_value = self._reference_feature_value(feature, frame)
            if baseline_value is None:
                continue
            counterfactual_frame = frame.copy()
            counterfactual_frame.at[0, feature] = baseline_value
            counterfactual_prediction = self.model_bundle.predict(counterfactual_frame)[0]
            impacts[feature] = float(base_prediction[label_index] - counterfactual_prediction[label_index])
        return self._format_feature_impacts(impacts, frame, top_n)

    def _top_shap_impacts(
        self,
        frame: pd.DataFrame,
        label_index: int,
        top_n: int,
    ) -> list[dict[str, Any]]:
        if not module_available("shap"):
            return []

        try:
            transformed = self.model_bundle.preprocessor.transform(frame)
            explainer = self._get_shap_explainer(label_index)
            shap_values = np.asarray(explainer.shap_values(transformed), dtype=float)
            if shap_values.ndim == 1:
                shap_row = shap_values
            else:
                shap_row = shap_values[0]

            impacts: dict[str, float] = {}
            for source_name, value in zip(
                self.model_bundle.preprocessor.feature_sources_out,
                shap_row,
                strict=False,
            ):
                impacts[source_name] = impacts.get(source_name, 0.0) + float(value)
            return self._format_feature_impacts(impacts, frame, top_n)
        except Exception:
            return []

    def _format_feature_impacts(
        self,
        impacts: dict[str, float],
        frame: pd.DataFrame,
        top_n: int,
    ) -> list[dict[str, Any]]:
        ranked = sorted(impacts.items(), key=lambda item: abs(item[1]), reverse=True)
        non_zero_ranked = [item for item in ranked if abs(float(item[1])) > 1e-6]
        ranked = non_zero_ranked or ranked
        results: list[dict[str, Any]] = []
        for feature, impact in ranked[:top_n]:
            results.append(
                {
                    "feature": humanize_feature_name(feature),
                    "feature_key": feature,
                    "impact": round(float(impact), 4),
                    "direction": "supports" if float(impact) >= 0.0 else "reduces",
                    "descriptor": self._feature_descriptor(feature, frame),
                }
            )
        return results

    def _feature_descriptor(self, feature: str, frame: pd.DataFrame) -> str:
        label = humanize_feature_name(feature)
        if feature in self.numeric_features:
            summary = self.numeric_summary.get(feature, {})
            value = float(pd.to_numeric(frame[feature], errors="coerce").iloc[0])
            mean_value = float(summary.get("mean", value))
            std_value = abs(float(summary.get("std", 0.0) or 0.0))
            if std_value > 0:
                zscore = (value - mean_value) / std_value
                if zscore >= 0.5:
                    return f"high {label}"
                if zscore <= -0.5:
                    return f"low {label}"
            return label

        raw_value = str(frame.iloc[0].get(feature, self.categorical_fill_value)).strip().lower()
        if raw_value and raw_value != self.categorical_fill_value:
            return f"{raw_value} {label}"
        return label

    def _reference_feature_value(self, feature: str, frame: pd.DataFrame) -> Any | None:
        if feature in self.numeric_features:
            summary = self.numeric_summary.get(feature, {})
            return float(summary.get("mean", self.numeric_fill_values.get(feature, 0.0)))
        if feature in self.categorical_features:
            allowed_values = self.categorical_levels.get(feature, [])
            if allowed_values:
                return str(allowed_values[0]).strip().lower()
            return self.categorical_fill_value
        return None

    def _get_shap_explainer(self, label_index: int) -> Any:
        if label_index in self._explainer_cache:
            return self._explainer_cache[label_index]

        import shap

        explainer = shap.TreeExplainer(self._estimator_for_label(label_index))
        self._explainer_cache[label_index] = explainer
        return explainer

    def _estimator_for_label(self, label_index: int) -> Any:
        if self.model_bundle.model is not None and hasattr(self.model_bundle.model, "estimators_"):
            return self.model_bundle.model.estimators_[label_index]
        if self.model_bundle.models is not None:
            return self.model_bundle.models[self.label_columns[label_index]]
        raise InferenceValidationError("No estimator available in model artifact.")

    def _load_json_artifact(self, filename: str) -> dict[str, Any]:
        artifact_path = self.artifact_dir / filename
        if not artifact_path.exists():
            return {}
        try:
            with artifact_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return {}

    def _build_numeric_feature_catalog(self, feature: str) -> dict[str, Any]:
        summary = self.numeric_summary.get(feature, {})
        return {
            "default": float(self.numeric_fill_values.get(feature, 0.0)),
            "mean": float(summary.get("mean", self.numeric_fill_values.get(feature, 0.0))),
            "min": float(summary.get("min", self.numeric_fill_values.get(feature, 0.0))),
            "max": float(summary.get("max", self.numeric_fill_values.get(feature, 0.0))),
            "std": float(summary.get("std", 0.0) or 0.0),
        }

    def _load_region_catalog(self, dataset_path_raw: Any) -> list[dict[str, Any]]:
        if not dataset_path_raw:
            return []

        dataset_path = Path(str(dataset_path_raw))
        if not dataset_path.exists():
            return []

        try:
            region_frame = pd.read_csv(dataset_path, usecols=["state", "region", "region_key"])
        except Exception:
            return []

        if "region" not in region_frame.columns:
            return []
        if "state" not in region_frame.columns:
            region_frame["state"] = ""

        region_frame = (
            region_frame.fillna("")
            .drop_duplicates(subset=["state", "region"])
            .sort_values(["state", "region"])
        )

        states: list[dict[str, Any]] = []
        for state_name, state_frame in region_frame.groupby("state", dropna=False, sort=True):
            regions = [str(value).strip() for value in state_frame["region"].tolist() if str(value).strip()]
            if not regions:
                continue
            display_state = str(state_name).strip() or "Unknown"
            states.append(
                {
                    "state": display_state,
                    "region_count": len(regions),
                    "regions": regions,
                }
            )
        return states


def configure_inference_logger(root_dir: Path) -> logging.Logger:
    logger = logging.getLogger("climate_pipeline.inference")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path = root_dir / "logs" / "inference.log"
    ensure_parent_dir(log_path)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def deduplicate_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
