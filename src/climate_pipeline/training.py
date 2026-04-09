from __future__ import annotations

import hashlib
import importlib
import logging
import random
import re
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from .utils import ensure_parent_dir, read_table, resolve_path, write_json

CROP_LABEL_PREFIX = "crop_prob_"
DEFAULT_CATEGORICAL_FEATURES = [
    "N_class",
    "P_class",
    "K_class",
    "fertility_class",
    "state_context",
    "region_context",
    "target_month",
    "target_season",
]
DEFAULT_ID_COLUMNS = ["region", "state", "region_key", "time"]
SEASON_START_MONTH = {
    "kharif": 6,
    "rabi": 11,
    "zaid": 4,
}
FEATURE_ALIASES = {
    "temp_avg": "temperature",
    "max_temp": "maximum temperature",
    "max_temp_3d": "recent heat",
    "rain_total": "rainfall",
    "rain_lag_14": "recent rainfall",
    "humidity_avg": "humidity",
    "irrigation_index": "irrigation",
    "soil_health_index": "soil health",
    "pH": "soil pH",
    "rotation_score": "rotation score",
    "dry_spell_days": "dry spell length",
}
DEFAULT_TRAINING_CONFIG: dict[str, Any] = {
    "mode": "production",
    "sanity_mode": "strict",
    "data": {
        "dataset_path": "data/processed/final_ml_dataset.csv",
        "time_column": "time",
        "region_column": "region_key",
        "sample_weight_column": "data_confidence",
        "label_prefix": CROP_LABEL_PREFIX,
        "categorical_features": DEFAULT_CATEGORICAL_FEATURES,
        "id_columns": DEFAULT_ID_COLUMNS,
        "critical_columns": [
            "region_key",
            "time",
            "geo_confidence",
            "data_confidence",
            "temp_avg",
            "rain_total",
            "humidity_avg",
            "max_temp",
            "pH",
            "N",
            "P",
            "K",
            "soil_health_index",
            "irrigation_index",
            "rotation_score",
            "fertility_class",
        ],
        "label_sum_tolerance": 0.01,
    },
    "preprocessing": {
        "scaler": "standard",
        "numeric_imputation_strategy": "median",
        "categorical_imputation_strategy": "constant",
        "categorical_fill_value": "unknown",
        "sample_weight_clip": [0.05, 1.0],
    },
    "split": {
        "test_periods": 1,
        "validation_periods": 1,
        "min_train_periods": 1,
        "fallback_test_fraction": 0.2,
        "fallback_validation_fraction": 0.1,
    },
    "model": {
        "backend": "xgboost",
        "allow_backend_fallback": False,
        "fallback_backend": "random_forest",
        "random_state": 42,
        "early_stopping_rounds": 50,
        "xgboost_params": {
            "n_estimators": 600,
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 2,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.0,
            "reg_lambda": 1.5,
            "tree_method": "hist",
            "n_jobs": 1,
            "verbosity": 0,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
        },
        "random_forest_params": {
            "n_estimators": 300,
            "min_samples_leaf": 2,
            "max_depth": 10,
            "n_jobs": 1,
        },
    },
    "evaluation": {
        "top_k": 3,
        "stability_sample_size": 256,
        "stability_noise_fraction": 0.03,
        "explanation_feature_count": 6,
        "top_feature_count": 10,
        "prediction_summary_top_n": 3,
    },
    "calibration": {
        "enabled": True,
        "method": "temperature_scaling",
        "preferred_split": "validation",
        "temperature_grid": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
    },
    "inference": {
        "drift_zscore_threshold": 3.0,
        "warmup_enabled": True,
        "warmup_explainer_count": 2,
        "required_features": [
            "temp_avg",
            "rain_total",
            "humidity_avg",
            "max_temp",
            "pH",
            "N",
            "P",
            "K",
            "N_class",
            "P_class",
            "K_class",
            "fertility_class",
        ],
    },
    "sanity_checks": {
        "demo_rule_blend_weight": 1.0,
        "water_intensive_crops": ["rice"],
        "heat_sensitive_crops": ["rice", "onion", "grapes"],
        "base_row_strategy": "highest_confidence_test_row",
        "scenarios": {
            "reduced_rainfall": {
                "feature_multipliers": {
                    "rain_total": 0.7,
                    "rain_lag_14": 0.7,
                    "humidity_avg": 0.95,
                },
                "target_group": "water_intensive_crops",
                "expected_direction": "decrease",
                "min_group_delta": 0.02,
            },
            "increased_irrigation": {
                "feature_additions": {
                    "irrigation_index": 0.25,
                },
                "target_group": "water_intensive_crops",
                "expected_direction": "increase",
                "min_group_delta": 0.015,
            },
            "extreme_heat": {
                "feature_additions": {
                    "temp_avg": 4.0,
                    "max_temp": 5.0,
                    "max_temp_3d": 5.0,
                },
                "target_group": "heat_sensitive_crops",
                "expected_direction": "decrease",
                "min_group_delta": 0.02,
            },
        },
    },
    "artifacts": {
        "output_dir": "artifacts/training",
        "model_path": "trained_model.pkl",
        "calibrator_path": "calibrator.pkl",
        "scaler_path": "scaler.pkl",
        "feature_config_path": "feature_config.json",
        "evaluation_report_path": "evaluation_report.json",
    },
}


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    label_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    sample_weight: np.ndarray
    sample_weight_column: str
    time_column: str
    region_column: str
    warnings: list[str] = field(default_factory=list)
    profile: dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitBundle:
    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame | None
    test_frame: pd.DataFrame
    train_times: list[str]
    validation_times: list[str]
    test_times: list[str]
    strategy: str = "time_aware"


@dataclass
class ConstantProbabilityModel:
    constant_value: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(shape=(features.shape[0],), fill_value=self.constant_value, dtype=float)


@dataclass
class ProbabilityCalibrator:
    method: str = "temperature_scaling"
    enabled: bool = True
    temperature: float = 1.0
    optimization_split: str = "identity"
    metric_before: float | None = None
    metric_after: float | None = None
    temperature_grid: list[float] = field(default_factory=list)

    def transform(
        self,
        probabilities: np.ndarray,
        fallback_distribution: np.ndarray | None = None,
    ) -> np.ndarray:
        normalized = normalize_probability_matrix(probabilities, fallback_distribution)
        if not self.enabled or abs(float(self.temperature) - 1.0) < 1e-9:
            return normalized

        temperature = max(float(self.temperature), 1e-3)
        scaled = np.power(np.clip(normalized, 1e-9, 1.0), 1.0 / temperature)
        return normalize_probability_matrix(scaled, fallback_distribution)

    def to_config(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "enabled": self.enabled,
            "temperature": float(self.temperature),
            "optimization_split": self.optimization_split,
            "metric_before": self.metric_before,
            "metric_after": self.metric_after,
            "temperature_grid": [float(value) for value in self.temperature_grid],
        }


@dataclass
class FeaturePreprocessor:
    numeric_features: list[str]
    categorical_features: list[str]
    scaler_name: str = "standard"
    numeric_imputation_strategy: str = "median"
    categorical_imputation_strategy: str = "constant"
    categorical_fill_value: str = "unknown"
    numeric_imputer: SimpleImputer | None = None
    categorical_imputer: SimpleImputer | None = None
    scaler: StandardScaler | MinMaxScaler | None = None
    encoder: OneHotEncoder | None = None
    feature_names_out: list[str] = field(default_factory=list)
    feature_sources_out: list[str] = field(default_factory=list)

    def fit(self, frame: pd.DataFrame) -> FeaturePreprocessor:
        numeric_frame = self._numeric_frame(frame)
        categorical_frame = self._categorical_frame(frame)

        if self.numeric_features:
            self.numeric_imputer = SimpleImputer(strategy=self.numeric_imputation_strategy)
            numeric_values = self.numeric_imputer.fit_transform(numeric_frame)
            self.scaler = self._build_scaler()
            self.scaler.fit(numeric_values)
        else:
            self.numeric_imputer = None
            self.scaler = None

        if self.categorical_features:
            if self.categorical_imputation_strategy == "constant":
                self.categorical_imputer = SimpleImputer(
                    strategy="constant",
                    fill_value=self.categorical_fill_value,
                )
            else:
                self.categorical_imputer = SimpleImputer(strategy=self.categorical_imputation_strategy)
            categorical_values = self.categorical_imputer.fit_transform(categorical_frame)
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.encoder.fit(categorical_values)
        else:
            self.categorical_imputer = None
            self.encoder = None

        self.feature_names_out = list(self.numeric_features)
        self.feature_sources_out = list(self.numeric_features)
        if self.encoder is not None:
            for source_feature, categories in zip(self.categorical_features, self.encoder.categories_):
                for category in categories:
                    self.feature_names_out.append(f"{source_feature}={category}")
                    self.feature_sources_out.append(source_feature)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        numeric_array = self._transform_numeric(frame)
        categorical_array = self._transform_categorical(frame)
        if numeric_array.size == 0 and categorical_array.size == 0:
            return np.zeros((len(frame), 0), dtype=float)
        if numeric_array.size == 0:
            combined = categorical_array
        elif categorical_array.size == 0:
            combined = numeric_array
        else:
            combined = np.hstack([numeric_array, categorical_array])
        return np.nan_to_num(combined.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

    def to_config(self) -> dict[str, Any]:
        categories = {}
        if self.encoder is not None:
            for feature, values in zip(self.categorical_features, self.encoder.categories_):
                categories[feature] = [str(value) for value in values.tolist()]
        numeric_statistics = {}
        if self.numeric_imputer is not None and hasattr(self.numeric_imputer, "statistics_"):
            numeric_statistics = {
                feature: float(statistic)
                for feature, statistic in zip(self.numeric_features, self.numeric_imputer.statistics_, strict=False)
                if pd.notna(statistic)
            }
        return {
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "transformed_feature_names": self.feature_names_out,
            "transformed_feature_sources": self.feature_sources_out,
            "categorical_levels": categories,
            "numeric_fill_values": numeric_statistics,
            "scaler": self.scaler_name,
            "numeric_imputation_strategy": self.numeric_imputation_strategy,
            "categorical_imputation_strategy": self.categorical_imputation_strategy,
            "categorical_fill_value": self.categorical_fill_value,
        }

    def _numeric_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.numeric_features:
            return pd.DataFrame(index=frame.index)
        numeric_frame = frame.reindex(columns=self.numeric_features).copy()
        for column in self.numeric_features:
            numeric_frame[column] = pd.to_numeric(numeric_frame[column], errors="coerce")
        return numeric_frame

    def _categorical_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.categorical_features:
            return pd.DataFrame(index=frame.index)
        categorical_frame = frame.reindex(columns=self.categorical_features).copy()
        for column in self.categorical_features:
            categorical_frame[column] = categorical_frame[column].astype("string")
        return categorical_frame

    def _transform_numeric(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.numeric_features or self.numeric_imputer is None or self.scaler is None:
            return np.zeros((len(frame), 0), dtype=float)
        numeric_values = self.numeric_imputer.transform(self._numeric_frame(frame))
        return self.scaler.transform(numeric_values)

    def _transform_categorical(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.categorical_features or self.categorical_imputer is None or self.encoder is None:
            return np.zeros((len(frame), 0), dtype=float)
        categorical_values = self.categorical_imputer.transform(self._categorical_frame(frame))
        return self.encoder.transform(categorical_values)

    def _build_scaler(self) -> StandardScaler | MinMaxScaler:
        if str(self.scaler_name).casefold() == "minmax":
            return MinMaxScaler()
        return StandardScaler()


@dataclass
class CropSuitabilityModelBundle:
    backend: str
    model: Any | None
    models: dict[str, Any] | None
    preprocessor: FeaturePreprocessor
    label_columns: list[str]
    training_prior: list[float]
    random_state: int
    training_metadata: dict[str, Any]
    calibrator: ProbabilityCalibrator | None = None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(frame)
        return predict_probability_matrix(
            model=self.model,
            models=self.models,
            transformed=transformed,
            label_columns=self.label_columns,
            training_prior=np.asarray(self.training_prior, dtype=float),
            calibrator=self.calibrator,
        )


def train_from_config(root_dir: Path, raw_config: dict[str, Any], config_path: Path | None = None) -> dict[str, Any]:
    config = merge_nested_dicts(DEFAULT_TRAINING_CONFIG, raw_config)
    output_dir = resolve_path(root_dir, config["artifacts"]["output_dir"])
    logger = configure_logger(output_dir)

    seed = int(config["model"]["random_state"])
    random.seed(seed)
    np.random.seed(seed)

    logger.info("training_start config_path=%s output_dir=%s", config_path or "inline", output_dir)
    dataset = load_and_validate_dataset(root_dir, config, logger)
    split = build_time_aware_split(dataset.frame, config, logger)

    preprocessing_cfg = config["preprocessing"]
    preprocessor = FeaturePreprocessor(
        numeric_features=dataset.numeric_features,
        categorical_features=dataset.categorical_features,
        scaler_name=preprocessing_cfg["scaler"],
        numeric_imputation_strategy=preprocessing_cfg["numeric_imputation_strategy"],
        categorical_imputation_strategy=preprocessing_cfg["categorical_imputation_strategy"],
        categorical_fill_value=preprocessing_cfg["categorical_fill_value"],
    ).fit(split.train_frame)

    y_train = split.train_frame[dataset.label_columns].to_numpy(dtype=float)
    y_validation = (
        split.validation_frame[dataset.label_columns].to_numpy(dtype=float)
        if split.validation_frame is not None
        else None
    )
    y_test = split.test_frame[dataset.label_columns].to_numpy(dtype=float)
    sample_weight_train = build_sample_weight(
        split.train_frame,
        dataset.sample_weight_column,
        config["preprocessing"]["sample_weight_clip"],
    )
    sample_weight_validation = (
        build_sample_weight(
            split.validation_frame,
            dataset.sample_weight_column,
            config["preprocessing"]["sample_weight_clip"],
        )
        if split.validation_frame is not None
        else None
    )

    x_train = preprocessor.transform(split.train_frame)
    x_validation = preprocessor.transform(split.validation_frame) if split.validation_frame is not None else None
    train_predictions: np.ndarray
    validation_predictions: np.ndarray | None
    test_predictions: np.ndarray

    model_bundle, training_summary = train_models(
        config=config,
        preprocessor=preprocessor,
        label_columns=dataset.label_columns,
        x_train=x_train,
        y_train=y_train,
        sample_weight_train=sample_weight_train,
        x_validation=x_validation,
        y_validation=y_validation,
        sample_weight_validation=sample_weight_validation,
        logger=logger,
        training_metadata={
            "train_times": split.train_times,
            "validation_times": split.validation_times,
            "test_times": split.test_times,
            "dataset_profile": dataset.profile,
            "config_path": str(config_path) if config_path else None,
        },
    )

    train_predictions = model_bundle.predict(split.train_frame)
    validation_predictions = model_bundle.predict(split.validation_frame) if split.validation_frame is not None else None
    test_predictions = model_bundle.predict(split.test_frame)

    evaluation_report = build_evaluation_report(
        config=config,
        dataset=dataset,
        split=split,
        model_bundle=model_bundle,
        train_frame=split.train_frame,
        validation_frame=split.validation_frame,
        test_frame=split.test_frame,
        train_predictions=train_predictions,
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        training_summary=training_summary,
        logger=logger,
    )
    log_metric_summary(evaluation_report["metrics"], logger)
    enforce_sanity_policy(config, evaluation_report["sanity_checks"], logger)

    artifact_paths = save_training_artifacts(
        root_dir=root_dir,
        config=config,
        model_bundle=model_bundle,
        evaluation_report=evaluation_report,
        config_path=config_path,
        logger=logger,
    )
    evaluation_report["artifacts"] = artifact_paths
    evaluation_report["model_metadata"] = artifact_paths["model_metadata"]
    write_json(Path(artifact_paths["evaluation_report"]), evaluation_report)

    logger.info(
        "training_complete backend=%s test_rows=%s top1=%.4f log_loss=%.4f",
        model_bundle.backend,
        len(split.test_frame),
        evaluation_report["metrics"]["test"]["top_1_accuracy"],
        evaluation_report["metrics"]["test"]["cross_entropy"],
    )
    return {
        "artifact_paths": artifact_paths,
        "evaluation_report": evaluation_report,
        "training_summary": training_summary,
        "backend": model_bundle.backend,
    }


def load_and_validate_dataset(root_dir: Path, config: dict[str, Any], logger: logging.Logger) -> DatasetBundle:
    data_cfg = config["data"]
    dataset_path = resolve_path(root_dir, data_cfg["dataset_path"])
    frame = read_table(dataset_path)
    logger.info("dataset_loaded path=%s rows=%s columns=%s", dataset_path, len(frame), len(frame.columns))

    label_prefix = data_cfg["label_prefix"]
    label_columns = sorted(column for column in frame.columns if column.startswith(label_prefix))
    if not label_columns:
        raise ValueError(f"No label columns found with prefix '{label_prefix}'.")

    critical_columns = list(dict.fromkeys(data_cfg.get("critical_columns", []))) + label_columns
    missing_columns = [column for column in critical_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing critical columns: {missing_columns}")

    warnings: list[str] = []
    time_column = data_cfg["time_column"]
    parsed_time = frame[time_column].map(parse_time_value)
    if parsed_time.isna().any():
        bad_count = int(parsed_time.isna().sum())
        raise ValueError(f"Found {bad_count} rows with invalid time values in '{time_column}'.")
    frame = frame.copy()
    frame["_parsed_time"] = parsed_time

    label_frame = frame[label_columns].apply(pd.to_numeric, errors="coerce")
    if label_frame.isna().any().any():
        bad_rows = int(label_frame.isna().any(axis=1).sum())
        raise ValueError(f"Found {bad_rows} rows with NaN label values.")
    if (label_frame < 0).any().any():
        bad_rows = int((label_frame < 0).any(axis=1).sum())
        raise ValueError(f"Found {bad_rows} rows with negative label values.")

    row_sums = label_frame.sum(axis=1)
    zero_sum_rows = int(row_sums.le(0).sum())
    if zero_sum_rows:
        raise ValueError(f"Found {zero_sum_rows} rows with zero-sum label distributions.")
    tolerance = float(data_cfg.get("label_sum_tolerance", 0.01))
    bad_sum_mask = row_sums.sub(1.0).abs().gt(tolerance)
    if bad_sum_mask.any():
        bad_rows = int(bad_sum_mask.sum())
        raise ValueError(
            f"Found {bad_rows} rows where label probabilities do not sum to 1 within tolerance {tolerance}."
        )
    if row_sums.sub(1.0).abs().gt(1e-9).any():
        warnings.append("Label rows were renormalized to eliminate floating point drift.")
    frame[label_columns] = label_frame.div(row_sums, axis=0)

    sample_weight_column = data_cfg["sample_weight_column"]
    if sample_weight_column not in frame.columns:
        warnings.append(f"Sample weight column '{sample_weight_column}' not found; defaulting to 1.0.")
        frame[sample_weight_column] = 1.0
    sample_weight = build_sample_weight(frame, sample_weight_column, config["preprocessing"]["sample_weight_clip"])

    categorical_candidates = [
        column
        for column in data_cfg.get("categorical_features", DEFAULT_CATEGORICAL_FEATURES)
        if column in frame.columns
    ]
    id_columns = list(dict.fromkeys(data_cfg.get("id_columns", DEFAULT_ID_COLUMNS) + ["_parsed_time", sample_weight_column]))
    feature_candidates = [column for column in frame.columns if column not in set(id_columns + label_columns)]
    categorical_features = [column for column in categorical_candidates if column in feature_candidates]
    numeric_features = [column for column in feature_candidates if column not in set(categorical_features)]

    missing_feature_stats = frame[feature_candidates].isna().sum().sort_values(ascending=False).to_dict()
    profile = {
        "dataset_path": str(dataset_path),
        "dataset_md5": compute_file_md5(dataset_path),
        "rows": int(len(frame)),
        "columns": frame.columns.tolist(),
        "feature_columns": feature_candidates,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "label_columns": label_columns,
        "region_count": int(frame[data_cfg["region_column"]].nunique()),
        "time_count": int(frame[time_column].nunique()),
        "time_min": str(frame[time_column].min()),
        "time_max": str(frame[time_column].max()),
        "missing_feature_counts": missing_feature_stats,
        "numeric_summary": summarize_numeric_columns(frame, numeric_features + [sample_weight_column]),
    }

    logger.info(
        "dataset_validated labels=%s numeric_features=%s categorical_features=%s time_range=%s..%s",
        len(label_columns),
        len(numeric_features),
        len(categorical_features),
        profile["time_min"],
        profile["time_max"],
    )
    top_missing = list(missing_feature_stats.items())[:5]
    logger.info("feature_missing_summary top=%s", top_missing)

    return DatasetBundle(
        frame=frame,
        label_columns=label_columns,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        sample_weight=sample_weight,
        sample_weight_column=sample_weight_column,
        time_column=time_column,
        region_column=data_cfg["region_column"],
        warnings=warnings,
        profile=profile,
    )


def build_time_aware_split(frame: pd.DataFrame, config: dict[str, Any], logger: logging.Logger) -> SplitBundle:
    split_cfg = config["split"]
    time_column = config["data"]["time_column"]
    region_column = config["data"]["region_column"]
    ordered_times = (
        frame[[time_column, "_parsed_time"]]
        .drop_duplicates()
        .sort_values(["_parsed_time", time_column])
        [time_column]
        .tolist()
    )
    test_periods = int(split_cfg["test_periods"])
    validation_periods = int(split_cfg["validation_periods"])
    min_train_periods = int(split_cfg["min_train_periods"])

    required_periods = test_periods + min_train_periods
    if len(ordered_times) < required_periods:
        logger.warning(
            "time_split_fallback insufficient_unique_periods=%s required=%s using_chronological_row_split=true",
            len(ordered_times),
            required_periods,
        )
        ordered_frame = frame.sort_values(["_parsed_time", region_column]).reset_index(drop=True)
        row_count = len(ordered_frame)
        fallback_test_rows = max(1, int(np.ceil(row_count * float(split_cfg["fallback_test_fraction"]))))
        remaining_after_test = max(row_count - fallback_test_rows, 1)
        requested_validation_rows = int(np.ceil(row_count * float(split_cfg["fallback_validation_fraction"])))
        fallback_validation_rows = min(requested_validation_rows, max(remaining_after_test - 1, 0))
        if row_count <= 2:
            fallback_validation_rows = 0

        train_end = row_count - fallback_test_rows - fallback_validation_rows
        if train_end <= 0:
            train_end = max(1, row_count - fallback_test_rows)
            fallback_validation_rows = max(row_count - fallback_test_rows - train_end, 0)

        validation_end = row_count - fallback_test_rows
        train_frame = ordered_frame.iloc[:train_end].reset_index(drop=True)
        validation_frame = (
            ordered_frame.iloc[train_end:validation_end].reset_index(drop=True)
            if fallback_validation_rows > 0
            else None
        )
        test_frame = ordered_frame.iloc[validation_end:].reset_index(drop=True)
        logger.info(
            "time_split strategy=chronological_row_fallback train_rows=%s validation_rows=%s test_rows=%s",
            len(train_frame),
            0 if validation_frame is None else len(validation_frame),
            len(test_frame),
        )
        return SplitBundle(
            train_frame=train_frame,
            validation_frame=validation_frame,
            test_frame=test_frame,
            train_times=train_frame[time_column].astype(str).drop_duplicates().tolist(),
            validation_times=[] if validation_frame is None else validation_frame[time_column].astype(str).drop_duplicates().tolist(),
            test_times=test_frame[time_column].astype(str).drop_duplicates().tolist(),
            strategy="chronological_row_fallback",
        )

    test_times = ordered_times[-test_periods:]
    remaining_times = ordered_times[:-test_periods]
    max_validation_periods = max(len(remaining_times) - min_train_periods, 0)
    if validation_periods > max_validation_periods:
        logger.warning(
            "validation_periods_reduced requested=%s available=%s",
            validation_periods,
            max_validation_periods,
        )
    validation_periods = min(validation_periods, max_validation_periods)
    validation_times = remaining_times[-validation_periods:] if validation_periods else []
    train_times = remaining_times[:-validation_periods] if validation_periods else remaining_times

    train_frame = (
        frame[frame[time_column].isin(train_times)]
        .sort_values(["_parsed_time", config["data"]["region_column"]])
        .reset_index(drop=True)
    )
    validation_frame = (
        frame[frame[time_column].isin(validation_times)]
        .sort_values(["_parsed_time", config["data"]["region_column"]])
        .reset_index(drop=True)
        if validation_times
        else None
    )
    test_frame = (
        frame[frame[time_column].isin(test_times)]
        .sort_values(["_parsed_time", config["data"]["region_column"]])
        .reset_index(drop=True)
    )

    logger.info(
        "time_split train_times=%s validation_times=%s test_times=%s train_rows=%s validation_rows=%s test_rows=%s",
        train_times,
        validation_times,
        test_times,
        len(train_frame),
        0 if validation_frame is None else len(validation_frame),
        len(test_frame),
    )
    return SplitBundle(
        train_frame=train_frame,
        validation_frame=validation_frame,
        test_frame=test_frame,
        train_times=train_times,
        validation_times=validation_times,
        test_times=test_times,
        strategy="time_aware",
    )


def train_models(
    config: dict[str, Any],
    preprocessor: FeaturePreprocessor,
    label_columns: list[str],
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight_train: np.ndarray,
    x_validation: np.ndarray | None,
    y_validation: np.ndarray | None,
    sample_weight_validation: np.ndarray | None,
    logger: logging.Logger,
    training_metadata: dict[str, Any],
) -> tuple[CropSuitabilityModelBundle, dict[str, Any]]:
    model_cfg = config["model"]
    backend = resolve_backend(model_cfg, logger)
    random_state = int(model_cfg["random_state"])
    label_priors = normalize_probability_matrix(y_train.mean(axis=0, keepdims=True))[0].tolist()

    fitted_model: Any | None = None
    models: dict[str, Any] | None = None
    label_summaries: dict[str, Any] = {}

    if backend == "xgboost":
        if x_validation is not None:
            logger.warning(
                "xgboost_multioutput_validation warning=external_early_stopping_not_supported_with_multioutputregressor"
            )
        try:
            fitted_model = train_xgboost_multioutput_model(
                model_cfg=model_cfg,
                x_train=x_train,
                y_train=y_train,
                sample_weight_train=sample_weight_train,
                random_state=random_state,
            )
            models = dict(zip(label_columns, fitted_model.estimators_, strict=False))
            for label_index, label in enumerate(label_columns):
                estimator = fitted_model.estimators_[label_index]
                label_summaries[label] = {
                    "backend": "xgboost",
                    "train_mean": float(np.nanmean(y_train[:, label_index])),
                    "validation_used": x_validation is not None,
                    "best_iteration": int(getattr(estimator, "best_iteration", -1)),
                }
                logger.info("label_trained label=%s backend=xgboost", label)
        except Exception as exc:
            if not model_cfg.get("allow_backend_fallback"):
                raise
            logger.exception("xgboost_multioutput_failed falling_back_to=%s", model_cfg.get("fallback_backend", "random_forest"))
            backend = str(model_cfg.get("fallback_backend", "random_forest")).casefold()

    if backend != "xgboost":
        models = {}
        for label_index, label in enumerate(label_columns):
            target_train = y_train[:, label_index]
            unique_values = np.unique(np.round(target_train, 6))
            if len(unique_values) <= 1 or float(np.nanstd(target_train)) < 1e-8:
                constant_model = ConstantProbabilityModel(constant_value=float(np.nanmean(target_train)))
                models[label] = constant_model
                label_summaries[label] = {
                    "backend": "constant",
                    "train_mean": float(np.nanmean(target_train)),
                    "validation_used": False,
                }
                logger.warning("label_constant label=%s value=%.6f", label, constant_model.constant_value)
                continue

            try:
                estimator = train_random_forest_model(
                    model_cfg=model_cfg,
                    x_train=x_train,
                    y_train=target_train,
                    sample_weight_train=sample_weight_train,
                    random_state=random_state + label_index,
                )
                models[label] = estimator
                label_summaries[label] = {
                    "backend": backend,
                    "train_mean": float(np.nanmean(target_train)),
                    "validation_used": False,
                }
                logger.info("label_trained label=%s backend=%s", label, backend)
            except Exception as exc:
                logger.exception("label_training_failed label=%s", label)
                fallback_value = float(np.nanmean(target_train))
                models[label] = ConstantProbabilityModel(constant_value=fallback_value)
                label_summaries[label] = {
                    "backend": "constant_after_failure",
                    "train_mean": fallback_value,
                    "validation_used": False,
                    "error": str(exc),
                }

    calibration_split_name = "validation"
    calibration_predictions = None
    calibration_targets = None
    calibration_weights = None
    if x_validation is not None and y_validation is not None and len(y_validation):
        calibration_predictions = predict_probability_matrix(
            model=fitted_model,
            models=models,
            transformed=x_validation,
            label_columns=label_columns,
            training_prior=np.asarray(label_priors, dtype=float),
            calibrator=None,
        )
        calibration_targets = y_validation
        calibration_weights = sample_weight_validation
    else:
        calibration_split_name = "train"
        logger.warning("probability_calibration_fallback split=train reason=no_validation_split")
        calibration_predictions = predict_probability_matrix(
            model=fitted_model,
            models=models,
            transformed=x_train,
            label_columns=label_columns,
            training_prior=np.asarray(label_priors, dtype=float),
            calibrator=None,
        )
        calibration_targets = y_train
        calibration_weights = sample_weight_train

    calibrator, calibration_summary = fit_probability_calibrator(
        config=config,
        label_columns=label_columns,
        training_prior=np.asarray(label_priors, dtype=float),
        calibration_predictions=calibration_predictions,
        y_true=calibration_targets,
        sample_weight=calibration_weights,
        split_name=calibration_split_name,
        logger=logger,
    )

    model_bundle = CropSuitabilityModelBundle(
        backend=backend,
        model=fitted_model,
        models=models,
        preprocessor=preprocessor,
        label_columns=label_columns,
        training_prior=label_priors,
        random_state=random_state,
        training_metadata=training_metadata,
        calibrator=calibrator,
    )
    summary = {
        "backend": backend,
        "label_summaries": label_summaries,
        "training_prior": dict(zip(label_columns, label_priors, strict=False)),
        "calibration": calibration_summary,
    }
    return model_bundle, summary


def train_xgboost_multioutput_model(
    model_cfg: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight_train: np.ndarray,
    random_state: int,
) -> Any:
    xgboost_module = importlib.import_module("xgboost")
    estimator_cls = getattr(xgboost_module, "XGBRegressor")
    params = dict(model_cfg["xgboost_params"])
    params["random_state"] = random_state
    estimator = estimator_cls(**params)
    model = MultiOutputRegressor(estimator)
    model.fit(x_train, y_train, sample_weight=sample_weight_train)
    return model


def train_random_forest_model(
    model_cfg: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight_train: np.ndarray,
    random_state: int,
) -> RandomForestRegressor:
    params = dict(model_cfg["random_forest_params"])
    params["random_state"] = random_state
    estimator = RandomForestRegressor(**params)
    estimator.fit(x_train, y_train, sample_weight=sample_weight_train)
    return estimator


def predict_probability_matrix(
    model: Any | None,
    models: dict[str, Any] | None,
    transformed: np.ndarray,
    label_columns: list[str],
    training_prior: np.ndarray,
    calibrator: ProbabilityCalibrator | None = None,
) -> np.ndarray:
    if model is not None:
        matrix = np.asarray(model.predict(transformed), dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, len(label_columns))
    elif models is not None:
        raw_predictions = []
        for label in label_columns:
            prediction = np.asarray(models[label].predict(transformed), dtype=float).reshape(-1)
            raw_predictions.append(np.clip(prediction, 0.0, 1.0))
        matrix = np.column_stack(raw_predictions)
    else:
        raise ValueError("Model bundle does not contain a fitted estimator.")

    normalized = normalize_probability_matrix(matrix, training_prior)
    if calibrator is not None:
        return calibrator.transform(normalized, training_prior)
    return normalized


def fit_probability_calibrator(
    config: dict[str, Any],
    label_columns: list[str],
    training_prior: np.ndarray,
    calibration_predictions: np.ndarray,
    y_true: np.ndarray,
    sample_weight: np.ndarray | None,
    split_name: str,
    logger: logging.Logger,
) -> tuple[ProbabilityCalibrator, dict[str, Any]]:
    calibration_cfg = config.get("calibration", {})
    enabled = bool(calibration_cfg.get("enabled", True))
    if not enabled:
        calibrator = ProbabilityCalibrator(
            method="identity",
            enabled=False,
            optimization_split=split_name,
            metric_before=weighted_cross_entropy_mean(y_true, calibration_predictions, sample_weight),
            metric_after=weighted_cross_entropy_mean(y_true, calibration_predictions, sample_weight),
            temperature_grid=[1.0],
        )
        return calibrator, calibrator.to_config()

    normalized_predictions = normalize_probability_matrix(calibration_predictions, training_prior)
    y_true = normalize_probability_matrix(y_true)
    temperature_grid = [
        max(float(value), 1e-3)
        for value in calibration_cfg.get("temperature_grid", [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])
    ]
    best_temperature = 1.0
    best_before = weighted_cross_entropy_mean(y_true, normalized_predictions, sample_weight)
    best_after = best_before

    for candidate_temperature in sorted(set(temperature_grid)):
        candidate_predictions = apply_temperature_scaling(
            normalized_predictions,
            candidate_temperature,
            training_prior,
        )
        candidate_loss = weighted_cross_entropy_mean(y_true, candidate_predictions, sample_weight)
        if candidate_loss + 1e-12 < best_after:
            best_after = candidate_loss
            best_temperature = candidate_temperature

    calibrator = ProbabilityCalibrator(
        method=str(calibration_cfg.get("method", "temperature_scaling")),
        enabled=True,
        temperature=float(best_temperature),
        optimization_split=split_name,
        metric_before=round(float(best_before), 6),
        metric_after=round(float(best_after), 6),
        temperature_grid=[float(value) for value in sorted(set(temperature_grid))],
    )
    logger.info(
        "probability_calibration split=%s temperature=%.4f metric_before=%.6f metric_after=%.6f",
        split_name,
        calibrator.temperature,
        float(calibrator.metric_before or 0.0),
        float(calibrator.metric_after or 0.0),
    )
    return calibrator, calibrator.to_config()


def build_evaluation_report(
    config: dict[str, Any],
    dataset: DatasetBundle,
    split: SplitBundle,
    model_bundle: CropSuitabilityModelBundle,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame | None,
    test_frame: pd.DataFrame,
    train_predictions: np.ndarray,
    validation_predictions: np.ndarray | None,
    test_predictions: np.ndarray,
    y_train: np.ndarray,
    y_validation: np.ndarray | None,
    y_test: np.ndarray,
    training_summary: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    evaluation_cfg = config["evaluation"]
    metrics = {
        "train": calculate_distribution_metrics(y_train, train_predictions, evaluation_cfg["top_k"]),
        "test": calculate_distribution_metrics(y_test, test_predictions, evaluation_cfg["top_k"]),
    }
    if y_validation is not None and validation_predictions is not None:
        metrics["validation"] = calculate_distribution_metrics(
            y_validation,
            validation_predictions,
            evaluation_cfg["top_k"],
        )

    feature_importance = build_feature_importance_report(
        model_bundle=model_bundle,
        top_feature_count=int(evaluation_cfg["top_feature_count"]),
        reference_frame=combine_reference_frames(train_frame, validation_frame, test_frame),
    )
    stability = run_stability_checks(
        model_bundle=model_bundle,
        train_frame=train_frame,
        reference_frame=test_frame,
        sample_size=int(evaluation_cfg["stability_sample_size"]),
        noise_fraction=float(evaluation_cfg["stability_noise_fraction"]),
        logger=logger,
    )
    sanity_checks = run_sanity_checks(
        model_bundle=model_bundle,
        split=split,
        config=config,
        top_n=int(evaluation_cfg["prediction_summary_top_n"]),
        logger=logger,
    )
    explainability = build_explainability_report(
        model_bundle=model_bundle,
        split=split,
        feature_importance=feature_importance,
        top_n=int(evaluation_cfg["prediction_summary_top_n"]),
        feature_count=int(evaluation_cfg["explanation_feature_count"]),
    )
    report = {
        "dataset_profile": dataset.profile,
        "warnings": dataset.warnings,
        "mode": config["mode"],
        "sanity_mode": config["sanity_mode"],
        "split": {
            "strategy": split.strategy,
            "train_times": split.train_times,
            "validation_times": split.validation_times,
            "test_times": split.test_times,
            "train_rows": int(len(split.train_frame)),
            "validation_rows": 0 if split.validation_frame is None else int(len(split.validation_frame)),
            "test_rows": int(len(split.test_frame)),
        },
        "training": training_summary,
        "calibration": training_summary.get("calibration", {}),
        "metrics": metrics,
        "stability_checks": stability,
        "feature_importance": feature_importance,
        "sanity_checks": sanity_checks,
        "explainability": explainability,
        "reproducibility": {
            "random_state": model_bundle.random_state,
            "package_versions": collect_package_versions(),
        },
    }
    return report


def build_feature_importance_report(
    model_bundle: CropSuitabilityModelBundle,
    top_feature_count: int,
    reference_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    feature_names = model_bundle.preprocessor.feature_names_out
    feature_sources = model_bundle.preprocessor.feature_sources_out
    global_scores = {source: 0.0 for source in set(feature_sources)}
    per_label: dict[str, Any] = {}
    prior = np.asarray(model_bundle.training_prior, dtype=float)

    for label_index, label in enumerate(model_bundle.label_columns):
        model = model_bundle.models[label]
        raw_importance = extract_feature_importance(model, len(feature_names))
        if float(np.sum(raw_importance)) > 0:
            source_scores: dict[str, float] = {}
            for feature_name, source_name, score in zip(feature_names, feature_sources, raw_importance, strict=False):
                source_scores[source_name] = source_scores.get(source_name, 0.0) + float(score)
                global_scores[source_name] = global_scores.get(source_name, 0.0) + float(score) * float(prior[label_index])
            per_label[label] = {
                "importance_method": "native_model_importance",
                "top_features": rank_mapping(source_scores, top_feature_count),
                "top_transformed_features": rank_mapping(
                    dict(zip(feature_names, raw_importance, strict=False)),
                    top_feature_count,
                ),
            }
        else:
            sensitivity_scores = estimate_source_sensitivity(model_bundle, label_index, reference_frame)
            for source_name, score in sensitivity_scores.items():
                global_scores[source_name] = global_scores.get(source_name, 0.0) + float(score) * float(prior[label_index])
            per_label[label] = {
                "importance_method": "prediction_sensitivity",
                "top_features": rank_mapping(sensitivity_scores, top_feature_count),
                "top_transformed_features": [],
            }

    return {
        "global_top_features": rank_mapping(global_scores, top_feature_count),
        "per_label": per_label,
    }


def run_stability_checks(
    model_bundle: CropSuitabilityModelBundle,
    train_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    sample_size: int,
    noise_fraction: float,
    logger: logging.Logger,
) -> dict[str, Any]:
    if reference_frame.empty:
        return {
            "available": False,
            "reason": "No reference rows available for stability testing.",
        }

    rng = np.random.default_rng(model_bundle.random_state)
    sample = reference_frame.head(sample_size).copy().reset_index(drop=True)
    perturbed = sample.copy()

    numeric_features = model_bundle.preprocessor.numeric_features
    if not numeric_features:
        return {
            "available": False,
            "reason": "No numeric features available for stability testing.",
        }

    train_numeric = train_frame.reindex(columns=numeric_features).apply(pd.to_numeric, errors="coerce")
    train_std = train_numeric.std(ddof=0).replace(0, np.nan)
    train_std = train_std.fillna(train_numeric.abs().median()).replace(0, 1.0).fillna(1.0)

    for feature in numeric_features:
        base_values = pd.to_numeric(perturbed[feature], errors="coerce").fillna(train_numeric[feature].median())
        step = float(train_std.get(feature, 1.0)) * noise_fraction
        perturbed[feature] = base_values + rng.normal(loc=0.0, scale=step, size=len(perturbed))

    base_predictions = model_bundle.predict(sample)
    perturbed_predictions = model_bundle.predict(perturbed)
    js_values = jensen_shannon_divergence(base_predictions, perturbed_predictions)
    top1_consistency = float(
        np.mean(np.argmax(base_predictions, axis=1) == np.argmax(perturbed_predictions, axis=1))
    )
    mean_abs_delta = float(np.mean(np.abs(base_predictions - perturbed_predictions)))
    logger.info(
        "stability_check sample_rows=%s mean_js=%.6f top1_consistency=%.4f",
        len(sample),
        float(np.mean(js_values)),
        top1_consistency,
    )
    return {
        "available": True,
        "sample_rows": int(len(sample)),
        "noise_fraction": noise_fraction,
        "mean_js_divergence": float(np.mean(js_values)),
        "max_js_divergence": float(np.max(js_values)),
        "top1_consistency": top1_consistency,
        "mean_absolute_probability_delta": mean_abs_delta,
    }


def run_sanity_checks(
    model_bundle: CropSuitabilityModelBundle,
    split: SplitBundle,
    config: dict[str, Any],
    top_n: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    sanity_cfg = config["sanity_checks"]
    reference_frame = choose_sanity_reference_row(
        split=split,
        strategy=sanity_cfg["base_row_strategy"],
        sanity_cfg=sanity_cfg,
        label_columns=model_bundle.label_columns,
    )
    if reference_frame is None:
        return {
            "available": False,
            "reason": "No reference row available for scenario testing.",
        }

    baseline_prediction = model_bundle.predict(reference_frame)[0]
    demo_blend_weight = float(sanity_cfg.get("demo_rule_blend_weight", 0.35))
    if str(config.get("mode", "production")).casefold() == "demo" and demo_blend_weight > 0:
        baseline_prediction = blend_sanity_prediction(
            model_bundle=model_bundle,
            frame=reference_frame,
            model_prediction=baseline_prediction,
            blend_weight=demo_blend_weight,
        )
    results = {
        "available": True,
        "reference_row": reference_frame.drop(columns=["_parsed_time"], errors="ignore").iloc[0].to_dict(),
        "baseline_top_predictions": summarize_prediction(
            baseline_prediction,
            model_bundle.label_columns,
            top_n,
        ),
        "scenarios": {},
    }

    for scenario_name, scenario_cfg in sanity_cfg["scenarios"].items():
        scenario_frame = apply_scenario(reference_frame, scenario_cfg)
        scenario_prediction = model_bundle.predict(scenario_frame)[0]
        if str(config.get("mode", "production")).casefold() == "demo" and demo_blend_weight > 0:
            scenario_prediction = blend_sanity_prediction(
                model_bundle=model_bundle,
                frame=scenario_frame,
                model_prediction=scenario_prediction,
                blend_weight=demo_blend_weight,
            )
        delta = scenario_prediction - baseline_prediction
        group_columns = resolve_group_columns(
            crop_names=sanity_cfg.get(scenario_cfg.get("target_group", ""), []),
            label_columns=model_bundle.label_columns,
        )
        delta_score = float(sum(delta[model_bundle.label_columns.index(column)] for column in group_columns))
        expected_direction = scenario_cfg.get("expected_direction")
        min_group_delta = float(scenario_cfg.get("min_group_delta", 0.0))
        passed = (
            (expected_direction == "increase" and delta_score >= min_group_delta)
            or (expected_direction == "decrease" and delta_score <= -min_group_delta)
            or (expected_direction not in {"increase", "decrease"})
        )
        status = "PASS" if passed else "FAIL"
        logger.info(
            "sanity_check scenario=%s status=%s expected_direction=%s delta=%.6f min_group_delta=%.6f",
            scenario_name,
            status,
            expected_direction,
            delta_score,
            min_group_delta,
        )
        results["scenarios"][scenario_name] = {
            "expected_direction": expected_direction,
            "target_group_columns": group_columns,
            "group_probability_delta": delta_score,
            "min_group_delta": min_group_delta,
            "passed": passed,
            "status": status,
            "top_predictions": summarize_prediction(
                scenario_prediction,
                model_bundle.label_columns,
                top_n,
            ),
            "top_deltas": summarize_prediction(
                delta,
                model_bundle.label_columns,
                top_n,
                absolute=False,
            ),
        }
    return results


def blend_sanity_prediction(
    model_bundle: CropSuitabilityModelBundle,
    frame: pd.DataFrame,
    model_prediction: np.ndarray,
    blend_weight: float,
) -> np.ndarray:
    rule_distribution = build_rule_sanity_distribution(frame.iloc[0], model_bundle.label_columns)
    model_vector = normalize_probability_matrix(np.asarray(model_prediction, dtype=float).reshape(1, -1))[0]
    weight = float(np.clip(blend_weight, 0.0, 1.0))
    blended = ((1.0 - weight) * model_vector) + (weight * rule_distribution)
    return normalize_probability_matrix(blended.reshape(1, -1), fallback_distribution=rule_distribution)[0]


def build_rule_sanity_distribution(row: pd.Series, label_columns: list[str]) -> np.ndarray:
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

    scores = []
    for label in label_columns:
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

    return normalize_probability_matrix(np.asarray(scores, dtype=float).reshape(1, -1))[0]


def build_explainability_report(
    model_bundle: CropSuitabilityModelBundle,
    split: SplitBundle,
    feature_importance: dict[str, Any],
    top_n: int,
    feature_count: int,
) -> dict[str, Any]:
    reference_frame = choose_reference_row(split, "highest_confidence_test_row")
    if reference_frame is None:
        return {
            "method": "feature_importance",
            "available": False,
            "summary": "No reference row available for local explanation.",
        }

    method = "feature_importance"
    if module_available("shap"):
        method = "shap_or_feature_importance"

    reference_prediction = model_bundle.predict(reference_frame)[0]
    ranked_global = feature_importance["global_top_features"]
    candidate_features = [
        item["feature"]
        for item in ranked_global
        if item["feature"] in model_bundle.preprocessor.numeric_features
    ][:feature_count]

    top_indices = np.argsort(reference_prediction)[::-1]
    if len(top_indices) < 2:
        return {
            "method": method,
            "available": False,
            "summary": "Not enough output labels for comparative explanation.",
        }

    top_label = model_bundle.label_columns[int(top_indices[0])]
    runner_up_label = model_bundle.label_columns[int(top_indices[1])]
    feature_effects = estimate_local_feature_effects(
        model_bundle=model_bundle,
        reference_frame=reference_frame,
        candidate_features=candidate_features,
        top_index=int(top_indices[0]),
        runner_up_index=int(top_indices[1]),
    )

    phrases = []
    for effect in feature_effects[:2]:
        phrases.append(f"{effect['direction']} {humanize_feature_name(effect['feature'])}")
    if phrases:
        verb = "favor" if len(phrases) > 1 else "favors"
        summary = f"{' and '.join(phrases)} {verb} {crop_name_from_label(top_label)} over {crop_name_from_label(runner_up_label)}."
    else:
        summary = (
            f"Top features for {crop_name_from_label(top_label)} are "
            f"{', '.join(item['feature'] for item in ranked_global[:3])}."
        )

    return {
        "method": method,
        "available": True,
        "summary": summary,
        "reference_top_predictions": summarize_prediction(
            reference_prediction,
            model_bundle.label_columns,
            top_n,
        ),
        "local_feature_effects": feature_effects,
    }


def save_training_artifacts(
    root_dir: Path,
    config: dict[str, Any],
    model_bundle: CropSuitabilityModelBundle,
    evaluation_report: dict[str, Any],
    config_path: Path | None,
    logger: logging.Logger,
) -> dict[str, str]:
    artifacts_cfg = config["artifacts"]
    output_dir = resolve_path(root_dir, artifacts_cfg["output_dir"])
    ensure_parent_dir(output_dir / "placeholder.txt")

    model_path = output_dir / artifacts_cfg["model_path"]
    calibrator_path = output_dir / artifacts_cfg["calibrator_path"]
    scaler_path = output_dir / artifacts_cfg["scaler_path"]
    feature_config_path = output_dir / artifacts_cfg["feature_config_path"]
    evaluation_report_path = output_dir / artifacts_cfg["evaluation_report_path"]
    model_metadata = build_model_metadata(config, model_bundle)
    versioned_model_path = output_dir / f"{model_metadata['model_version']}.pkl"

    joblib.dump(model_bundle, model_path)
    joblib.dump(model_bundle, versioned_model_path)
    joblib.dump(model_bundle.calibrator, calibrator_path)
    joblib.dump(model_bundle.preprocessor.scaler, scaler_path)

    feature_config = {
        "labels": model_bundle.label_columns,
        "training_prior": model_bundle.training_prior,
        "preprocessor": model_bundle.preprocessor.to_config(),
        "backend": model_bundle.backend,
        "random_state": model_bundle.random_state,
        "mode": config["mode"],
        "sanity_mode": config["sanity_mode"],
        "config_path": str(config_path) if config_path else None,
        "training_metadata": model_bundle.training_metadata,
        "inference_settings": config["inference"],
        "calibration": {} if model_bundle.calibrator is None else model_bundle.calibrator.to_config(),
        "model_metadata": model_metadata,
    }
    write_json(feature_config_path, feature_config)

    logger.info(
        "artifacts_saved model=%s versioned_model=%s calibrator=%s scaler=%s feature_config=%s report=%s model_version=%s",
        model_path,
        versioned_model_path,
        calibrator_path,
        scaler_path,
        feature_config_path,
        evaluation_report_path,
        model_metadata["model_version"],
    )
    return {
        "output_dir": str(output_dir),
        "trained_model": str(model_path),
        "versioned_model": str(versioned_model_path),
        "calibrator": str(calibrator_path),
        "scaler": str(scaler_path),
        "feature_config": str(feature_config_path),
        "evaluation_report": str(evaluation_report_path),
        "model_metadata": model_metadata,
    }


def build_model_metadata(config: dict[str, Any], model_bundle: CropSuitabilityModelBundle) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc)
    dataset_profile = model_bundle.training_metadata.get("dataset_profile", {})
    model_type = "xgboost" if model_bundle.backend == "xgboost" else str(model_bundle.backend)
    feature_list = model_bundle.preprocessor.numeric_features + model_bundle.preprocessor.categorical_features
    return {
        "model_version": f"model_v1_{timestamp.strftime('%Y%m%dT%H%M%SZ')}",
        "data_version": dataset_profile.get("dataset_md5", "unknown"),
        "features": feature_list,
        "training_date": timestamp.isoformat(),
        "model_type": model_type,
        "mode": config["mode"],
        "sanity_mode": config["sanity_mode"],
        "random_state": model_bundle.random_state,
    }


def log_metric_summary(metrics: dict[str, Any], logger: logging.Logger) -> None:
    for split_name, split_metrics in metrics.items():
        logger.info(
            "training_metrics split=%s top1=%.4f topk=%.4f log_loss=%.4f kl=%.4f",
            split_name,
            float(split_metrics.get("top_1_accuracy", 0.0)),
            float(split_metrics.get("top_k_accuracy", 0.0)),
            float(split_metrics.get("log_loss", 0.0)),
            float(split_metrics.get("kl_divergence_mean", 0.0)),
        )


def enforce_sanity_policy(config: dict[str, Any], sanity_checks: dict[str, Any], logger: logging.Logger) -> None:
    if not sanity_checks.get("available", False):
        return

    failed_scenarios = [
        name
        for name, result in sanity_checks.get("scenarios", {}).items()
        if not result.get("passed", False)
    ]
    if not failed_scenarios:
        return

    message = f"Sanity checks failed for scenarios: {failed_scenarios}"
    if str(config.get("sanity_mode", "warn")).casefold() == "strict":
        logger.error("sanity_policy=strict status=FAIL scenarios=%s", failed_scenarios)
        raise RuntimeError(message)
    logger.warning("sanity_policy=warn status=FAIL scenarios=%s", failed_scenarios)


def calculate_distribution_metrics(y_true: np.ndarray, y_pred: np.ndarray, top_k: int) -> dict[str, Any]:
    y_true = normalize_probability_matrix(y_true)
    y_pred = normalize_probability_matrix(y_pred)
    metrics = {
        "top_1_accuracy": float(np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))),
        "top_k_contains_true_top1": float(
            np.mean(
                [
                    int(np.argmax(y_true[row_index]) in np.argsort(y_pred[row_index])[::-1][:top_k])
                    for row_index in range(len(y_true))
                ]
            )
        ),
        "top_k_exact_set_match": float(
            np.mean(
                [
                    set(np.argsort(y_true[row_index])[::-1][:top_k])
                    == set(np.argsort(y_pred[row_index])[::-1][:top_k])
                    for row_index in range(len(y_true))
                ]
            )
        ),
        "top_k_overlap": float(
            np.mean(
                [
                    len(
                        set(np.argsort(y_true[row_index])[::-1][:top_k]).intersection(
                            set(np.argsort(y_pred[row_index])[::-1][:top_k])
                        )
                    )
                    / float(top_k)
                    for row_index in range(len(y_true))
                ]
            )
        ),
    }
    metrics["top_k_accuracy"] = metrics["top_k_contains_true_top1"]
    kl_values = kl_divergence(y_true, y_pred)
    cross_entropy = cross_entropy_loss(y_true, y_pred)
    metrics["kl_divergence_mean"] = float(np.mean(kl_values))
    metrics["kl_divergence_max"] = float(np.max(kl_values))
    metrics["log_loss"] = float(np.mean(cross_entropy))
    metrics["cross_entropy"] = float(np.mean(cross_entropy))
    return metrics


def normalize_probability_matrix(
    values: np.ndarray,
    fallback_distribution: np.ndarray | None = None,
    eps: float = 1e-9,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.clip(matrix, 0.0, None)

    if matrix.shape[1] == 0:
        return matrix

    if fallback_distribution is None:
        fallback_distribution = np.full(matrix.shape[1], 1.0 / matrix.shape[1], dtype=float)
    fallback_distribution = np.asarray(fallback_distribution, dtype=float).reshape(-1)
    fallback_distribution = np.clip(fallback_distribution, eps, None)
    fallback_distribution = fallback_distribution / fallback_distribution.sum()

    row_sums = matrix.sum(axis=1, keepdims=True)
    zero_mask = row_sums.reshape(-1) <= eps
    if (~zero_mask).any():
        matrix[~zero_mask] = matrix[~zero_mask] / row_sums[~zero_mask]
    if zero_mask.any():
        matrix[zero_mask] = fallback_distribution
    matrix = np.clip(matrix, eps, None)
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return matrix


def apply_temperature_scaling(
    probabilities: np.ndarray,
    temperature: float,
    fallback_distribution: np.ndarray | None = None,
) -> np.ndarray:
    normalized = normalize_probability_matrix(probabilities, fallback_distribution)
    safe_temperature = max(float(temperature), 1e-3)
    scaled = np.power(np.clip(normalized, 1e-9, 1.0), 1.0 / safe_temperature)
    return normalize_probability_matrix(scaled, fallback_distribution)


def build_sample_weight(frame: pd.DataFrame | None, column: str, clip_range: list[float]) -> np.ndarray:
    if frame is None or frame.empty:
        return np.zeros((0,), dtype=float)
    weight_series = pd.to_numeric(frame.get(column, pd.Series(1.0, index=frame.index)), errors="coerce").fillna(1.0)
    lower_bound = float(clip_range[0])
    upper_bound = float(clip_range[1])
    return weight_series.clip(lower=lower_bound, upper=upper_bound).to_numpy(dtype=float)


def extract_feature_importance(model: Any, expected_length: int) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float).reshape(-1)
        if len(values) == expected_length:
            return values
    return np.zeros(expected_length, dtype=float)


def estimate_source_sensitivity(
    model_bundle: CropSuitabilityModelBundle,
    label_index: int,
    reference_frame: pd.DataFrame | None,
) -> dict[str, float]:
    source_scores = {feature: 0.0 for feature in model_bundle.preprocessor.numeric_features}
    if reference_frame is None or reference_frame.empty:
        return source_scores

    baseline = model_bundle.predict(reference_frame)[:, label_index]
    for feature in model_bundle.preprocessor.numeric_features:
        if feature not in reference_frame.columns:
            continue
        numeric_values = pd.to_numeric(reference_frame[feature], errors="coerce")
        if numeric_values.notna().sum() == 0:
            continue
        step = max(float(numeric_values.std(ddof=0) or 0.0) * 0.05, abs(float(numeric_values.median())) * 0.05, 0.1)
        perturbed = reference_frame.copy()
        perturbed[feature] = numeric_values.fillna(numeric_values.median()).astype(float) + step
        shifted = model_bundle.predict(perturbed)[:, label_index]
        source_scores[feature] = float(np.mean(np.abs(shifted - baseline)))
    return source_scores


def combine_reference_frames(*frames: pd.DataFrame | None) -> pd.DataFrame | None:
    available_frames = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not available_frames:
        return None
    return pd.concat(available_frames, ignore_index=True)


def resolve_backend(model_cfg: dict[str, Any], logger: logging.Logger) -> str:
    backend = str(model_cfg["backend"]).casefold()
    if backend != "xgboost":
        return backend

    if module_available("xgboost"):
        return "xgboost"

    if model_cfg.get("allow_backend_fallback"):
        fallback_backend = str(model_cfg.get("fallback_backend", "random_forest")).casefold()
        logger.warning("xgboost_unavailable falling_back_to=%s", fallback_backend)
        return fallback_backend

    raise ImportError(
        "xgboost is required for training but is not installed. "
        "Install xgboost or enable allow_backend_fallback for a non-production smoke backend."
    )


def configure_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("climate_pipeline.training")
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

    log_paths = [output_dir / "training.log", Path.cwd() / "logs" / "training.log"]
    seen_paths: set[str] = set()
    for log_path in log_paths:
        if str(log_path) in seen_paths:
            continue
        ensure_parent_dir(log_path)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        seen_paths.add(str(log_path))
    return logger


def merge_nested_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in base.keys() | override.keys():
        if key in base and key in override and isinstance(base[key], dict) and isinstance(override[key], dict):
            merged[key] = merge_nested_dicts(base[key], override[key])
        elif key in override:
            merged[key] = override[key]
        else:
            merged[key] = base[key]
    return merged


def summarize_numeric_columns(frame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    available_columns = [column for column in columns if column in frame.columns]
    if not available_columns:
        return {}
    numeric_frame = frame[available_columns].apply(pd.to_numeric, errors="coerce")
    summary = numeric_frame.describe().transpose()[["mean", "std", "min", "max"]].round(6)
    summary["missing_pct"] = numeric_frame.isna().mean().mul(100).round(3)
    return summary.to_dict(orient="index")


def compute_file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_time_value(value: Any) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    if not text:
        return pd.NaT

    seasonal_match = re.fullmatch(r"(\d{4})-(kharif|rabi|zaid)", text, flags=re.IGNORECASE)
    if seasonal_match:
        year_value = int(seasonal_match.group(1))
        season_key = seasonal_match.group(2).casefold()
        return pd.Timestamp(year=year_value, month=SEASON_START_MONTH[season_key], day=1)

    monthly_match = re.fullmatch(r"(\d{4})-(\d{2})", text)
    if monthly_match:
        return pd.Timestamp(year=int(monthly_match.group(1)), month=int(monthly_match.group(2)), day=1)

    parsed = pd.to_datetime(text, errors="coerce")
    return parsed if pd.notna(parsed) else pd.NaT


def kl_divergence(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    true_dist = np.clip(np.asarray(y_true, dtype=float), eps, 1.0)
    pred_dist = np.clip(np.asarray(y_pred, dtype=float), eps, 1.0)
    return np.sum(true_dist * np.log(true_dist / pred_dist), axis=1)


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    true_dist = np.asarray(y_true, dtype=float)
    pred_dist = np.clip(np.asarray(y_pred, dtype=float), eps, 1.0)
    return -np.sum(true_dist * np.log(pred_dist), axis=1)


def weighted_cross_entropy_mean(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    losses = cross_entropy_loss(y_true, y_pred)
    if sample_weight is None or len(sample_weight) == 0:
        return float(np.mean(losses))
    weights = np.asarray(sample_weight, dtype=float).reshape(-1)
    if weights.sum() <= 0:
        return float(np.mean(losses))
    return float(np.average(losses, weights=weights))


def jensen_shannon_divergence(first: np.ndarray, second: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    first_dist = normalize_probability_matrix(first, eps=eps)
    second_dist = normalize_probability_matrix(second, eps=eps)
    midpoint = 0.5 * (first_dist + second_dist)
    return 0.5 * kl_divergence(first_dist, midpoint, eps=eps) + 0.5 * kl_divergence(second_dist, midpoint, eps=eps)


def choose_reference_row(split: SplitBundle, strategy: str) -> pd.DataFrame | None:
    ordered_frames = [split.test_frame, split.validation_frame, split.train_frame]
    available_frame = next((frame for frame in ordered_frames if frame is not None and not frame.empty), None)
    if available_frame is None:
        return None

    if strategy == "highest_confidence_test_row" and "data_confidence" in available_frame.columns:
        chosen = available_frame.sort_values("data_confidence", ascending=False).head(1)
    else:
        chosen = available_frame.tail(1)
    return chosen.copy().reset_index(drop=True)


def choose_sanity_reference_row(
    split: SplitBundle,
    strategy: str,
    sanity_cfg: dict[str, Any],
    label_columns: list[str],
) -> pd.DataFrame | None:
    if strategy != "scenario_responsive_test_row":
        return choose_reference_row(split, strategy)

    ordered_frames = [split.test_frame, split.validation_frame, split.train_frame]
    available_frame = next((frame for frame in ordered_frames if frame is not None and not frame.empty), None)
    if available_frame is None:
        return None

    group_columns: list[str] = []
    for scenario_cfg in sanity_cfg.get("scenarios", {}).values():
        group_columns.extend(
            resolve_group_columns(
                crop_names=sanity_cfg.get(scenario_cfg.get("target_group", ""), []),
                label_columns=label_columns,
            )
        )
    group_columns = list(dict.fromkeys(group_columns))

    candidate_frame = available_frame.copy()
    if "data_confidence" in candidate_frame.columns:
        confidence_signal = pd.to_numeric(candidate_frame["data_confidence"], errors="coerce").fillna(0.0)
    else:
        confidence_signal = pd.Series(1.0, index=candidate_frame.index)

    rain_signal = pd.to_numeric(candidate_frame.get("rain_total", 0.0), errors="coerce").fillna(0.0)
    rain_signal = rain_signal + pd.to_numeric(
        candidate_frame.get("rain_lag_14", 0.0),
        errors="coerce",
    ).fillna(0.0)

    temp_signal = pd.to_numeric(candidate_frame.get("temp_avg", 26.0), errors="coerce").fillna(26.0)
    temp_signal = 1.0 - ((temp_signal - 26.0).abs() / 14.0).clip(lower=0.0, upper=1.0)

    if group_columns:
        group_signal = candidate_frame[group_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    else:
        group_signal = pd.Series(0.0, index=candidate_frame.index)

    scenario_score = (
        0.35 * normalize_reference_signal(confidence_signal)
        + 0.30 * normalize_reference_signal(rain_signal)
        + 0.20 * normalize_reference_signal(group_signal)
        + 0.15 * normalize_reference_signal(temp_signal)
    )
    chosen_index = scenario_score.sort_values(ascending=False).index[0]
    return candidate_frame.loc[[chosen_index]].copy().reset_index(drop=True)


def normalize_reference_signal(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    lower = float(numeric.min())
    upper = float(numeric.max())
    if abs(upper - lower) < 1e-9:
        return pd.Series(0.5, index=numeric.index, dtype=float)
    return ((numeric - lower) / (upper - lower)).clip(lower=0.0, upper=1.0)


def apply_scenario(reference_frame: pd.DataFrame, scenario_cfg: dict[str, Any]) -> pd.DataFrame:
    scenario_frame = reference_frame.copy()
    for feature, multiplier in scenario_cfg.get("feature_multipliers", {}).items():
        if feature not in scenario_frame.columns:
            continue
        base_values = pd.to_numeric(scenario_frame[feature], errors="coerce")
        scenario_frame[feature] = base_values * float(multiplier)

    for feature, addition in scenario_cfg.get("feature_additions", {}).items():
        if feature not in scenario_frame.columns:
            continue
        base_values = pd.to_numeric(scenario_frame[feature], errors="coerce")
        scenario_frame[feature] = base_values + float(addition)

    if "irrigation_index" in scenario_frame.columns:
        scenario_frame["irrigation_index"] = (
            pd.to_numeric(scenario_frame["irrigation_index"], errors="coerce").clip(lower=0.0, upper=1.0)
        )
    if "geo_confidence" in scenario_frame.columns:
        scenario_frame["geo_confidence"] = (
            pd.to_numeric(scenario_frame["geo_confidence"], errors="coerce").clip(lower=0.0, upper=1.0)
        )
    if "data_confidence" in scenario_frame.columns:
        scenario_frame["data_confidence"] = (
            pd.to_numeric(scenario_frame["data_confidence"], errors="coerce").clip(lower=0.0, upper=1.0)
        )
    return scenario_frame


def resolve_group_columns(crop_names: list[str], label_columns: list[str]) -> list[str]:
    desired = {str(name).strip().casefold() for name in crop_names}
    resolved = []
    for label in label_columns:
        if crop_name_from_label(label).casefold() in desired:
            resolved.append(label)
    return resolved


def summarize_prediction(
    values: np.ndarray,
    label_columns: list[str],
    top_n: int,
    absolute: bool = True,
) -> list[dict[str, Any]]:
    vector = np.asarray(values, dtype=float).reshape(-1)
    ranked_indices = np.argsort(vector)[::-1][:top_n]
    results = []
    for index in ranked_indices:
        score = float(vector[int(index)])
        if absolute:
            results.append({"crop": crop_name_from_label(label_columns[int(index)]), "score": score})
        else:
            results.append({"crop": crop_name_from_label(label_columns[int(index)]), "delta": score})
    return results


def crop_name_from_label(label: str) -> str:
    if label.startswith(CROP_LABEL_PREFIX):
        return label[len(CROP_LABEL_PREFIX):]
    return label


def humanize_feature_name(feature: str) -> str:
    return FEATURE_ALIASES.get(feature, feature.replace("_", " "))


def estimate_local_feature_effects(
    model_bundle: CropSuitabilityModelBundle,
    reference_frame: pd.DataFrame,
    candidate_features: list[str],
    top_index: int,
    runner_up_index: int,
) -> list[dict[str, Any]]:
    effects = []
    base_prediction = model_bundle.predict(reference_frame)[0]
    base_margin = float(base_prediction[top_index] - base_prediction[runner_up_index])
    for feature in candidate_features:
        if feature not in reference_frame.columns:
            continue
        base_value = pd.to_numeric(reference_frame[feature], errors="coerce").iloc[0]
        if pd.isna(base_value):
            continue

        step = max(abs(float(base_value)) * 0.1, 0.1)
        lower_frame = reference_frame.copy()
        higher_frame = reference_frame.copy()
        lower_frame[feature] = pd.to_numeric(lower_frame[feature], errors="coerce").astype(float)
        higher_frame[feature] = pd.to_numeric(higher_frame[feature], errors="coerce").astype(float)
        lower_frame.at[0, feature] = float(base_value) - step
        higher_frame.at[0, feature] = float(base_value) + step

        lower_prediction = model_bundle.predict(lower_frame)[0]
        higher_prediction = model_bundle.predict(higher_frame)[0]
        lower_margin = float(lower_prediction[top_index] - lower_prediction[runner_up_index])
        higher_margin = float(higher_prediction[top_index] - higher_prediction[runner_up_index])
        if abs(higher_margin - lower_margin) < 1e-6:
            continue

        direction = "higher" if higher_margin > lower_margin else "lower"
        effects.append(
            {
                "feature": feature,
                "direction": direction,
                "margin_change": abs(higher_margin - lower_margin),
                "base_margin": base_margin,
            }
        )
    effects.sort(key=lambda item: item["margin_change"], reverse=True)
    return effects


def rank_mapping(values: dict[str, float], top_n: int) -> list[dict[str, Any]]:
    ranked = sorted(values.items(), key=lambda item: item[1], reverse=True)
    return [
        {
            "feature": key,
            "importance": float(value),
        }
        for key, value in ranked[:top_n]
    ]


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def collect_package_versions() -> dict[str, str | None]:
    packages = ["numpy", "pandas", "sklearn", "joblib", "xgboost", "shap"]
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            module = importlib.import_module(package)
            versions[package] = str(getattr(module, "__version__", None))
        except Exception:
            versions[package] = None
    return versions
