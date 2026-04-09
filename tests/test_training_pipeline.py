from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from climate_pipeline.training import (  # noqa: E402
    DEFAULT_TRAINING_CONFIG,
    FeaturePreprocessor,
    build_time_aware_split,
    configure_logger,
    load_and_validate_dataset,
    merge_nested_dicts,
    normalize_probability_matrix,
    train_from_config,
)


class TrainingPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = merge_nested_dicts(DEFAULT_TRAINING_CONFIG, {
            "mode": "demo",
            "sanity_mode": "warn",
            "data": {
                "dataset_path": "data/processed/final_ml_dataset.csv",
            },
            "model": {
                "backend": "xgboost",
                "allow_backend_fallback": True,
                "fallback_backend": "random_forest",
                "random_state": 42,
                "xgboost_params": {
                    "n_estimators": 20,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "min_child_weight": 1,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 0.0,
                    "reg_lambda": 1.0,
                    "tree_method": "hist",
                    "n_jobs": 1,
                    "verbosity": 0,
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                },
                "random_forest_params": {
                    "n_estimators": 25,
                    "min_samples_leaf": 1,
                    "max_depth": 6,
                    "n_jobs": 1,
                },
            },
            "artifacts": {
                "output_dir": "artifacts/test_output",
            },
        })

    def test_dataset_validation_finds_expected_labels(self) -> None:
        logger = configure_logger(ROOT_DIR / "artifacts" / "test_logs")
        dataset = load_and_validate_dataset(ROOT_DIR, self.config, logger)
        self.assertGreater(len(dataset.label_columns), 0)
        self.assertIn("crop_prob_rice", dataset.label_columns)
        self.assertTrue(np.allclose(dataset.frame[dataset.label_columns].sum(axis=1).to_numpy(), 1.0))

    def test_time_split_is_ordered(self) -> None:
        logger = configure_logger(ROOT_DIR / "artifacts" / "test_logs")
        dataset = load_and_validate_dataset(ROOT_DIR, self.config, logger)
        split = build_time_aware_split(dataset.frame, self.config, logger)
        self.assertLess(max(split.train_times), min(split.test_times))

    def test_preprocessor_emits_dense_nan_free_matrix(self) -> None:
        logger = configure_logger(ROOT_DIR / "artifacts" / "test_logs")
        dataset = load_and_validate_dataset(ROOT_DIR, self.config, logger)
        preprocessor = FeaturePreprocessor(
            numeric_features=dataset.numeric_features,
            categorical_features=dataset.categorical_features,
        ).fit(dataset.frame)
        transformed = preprocessor.transform(dataset.frame)
        self.assertEqual(transformed.shape[0], len(dataset.frame))
        self.assertFalse(np.isnan(transformed).any())

    def test_probability_normalization_handles_zero_rows(self) -> None:
        raw = np.array([[0.4, 0.6], [0.0, 0.0]])
        normalized = normalize_probability_matrix(raw, np.array([0.7, 0.3]))
        self.assertTrue(np.allclose(normalized.sum(axis=1), 1.0))
        self.assertTrue(np.allclose(normalized[1], np.array([0.7, 0.3])))

    def test_time_split_falls_back_safely_for_single_period(self) -> None:
        logger = configure_logger(ROOT_DIR / "artifacts" / "test_logs")
        dataset = load_and_validate_dataset(ROOT_DIR, self.config, logger)
        frame = dataset.frame.copy()
        frame["time"] = "2024-06"
        frame["_parsed_time"] = pd.Timestamp("2024-06-01")
        split = build_time_aware_split(frame, self.config, logger)
        self.assertEqual(split.strategy, "chronological_row_fallback")
        self.assertGreater(len(split.train_frame), 0)
        self.assertGreater(len(split.test_frame), 0)

    def test_end_to_end_training_creates_expected_artifacts(self) -> None:
        artifacts_root = ROOT_DIR / "artifacts"
        artifacts_root.mkdir(parents=True, exist_ok=True)
        output_dir = artifacts_root / "test_run_e2e"
        output_dir.mkdir(parents=True, exist_ok=True)
        override = merge_nested_dicts(
            self.config,
            {
                "artifacts": {
                    "output_dir": str(output_dir),
                },
                "split": {
                    "test_periods": 1,
                    "validation_periods": 1,
                    "min_train_periods": 1,
                },
            },
        )
        result = train_from_config(ROOT_DIR, override)
        artifact_paths = result["artifact_paths"]
        for key in ["trained_model", "calibrator", "scaler", "feature_config", "evaluation_report"]:
            self.assertTrue(Path(artifact_paths[key]).exists(), key)
        test_metrics = result["evaluation_report"]["metrics"]["test"]
        self.assertIn("top_1_accuracy", test_metrics)
        self.assertIn("cross_entropy", test_metrics)
        self.assertIn("calibration", result["evaluation_report"])
        self.assertIn("temperature", result["evaluation_report"]["calibration"])

    def test_strict_sanity_mode_fails_when_scenarios_fail(self) -> None:
        strict_config = merge_nested_dicts(
            self.config,
            {
                "mode": "production",
                "sanity_mode": "strict",
                "artifacts": {
                    "output_dir": "artifacts/test_run_strict_failure",
                },
            },
        )
        with self.assertRaises(RuntimeError):
            train_from_config(ROOT_DIR, strict_config)


if __name__ == "__main__":
    unittest.main()
