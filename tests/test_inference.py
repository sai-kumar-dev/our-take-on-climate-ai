from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from climate_pipeline.inference import CropSuitabilityInferenceService  # noqa: E402
from climate_pipeline.training import DEFAULT_TRAINING_CONFIG, merge_nested_dicts, train_from_config  # noqa: E402


class InferenceServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact_dir = ROOT_DIR / "artifacts" / "test_inference_model"
        cls.artifact_dir.mkdir(parents=True, exist_ok=True)
        config = merge_nested_dicts(
            DEFAULT_TRAINING_CONFIG,
            {
                "mode": "demo",
                "sanity_mode": "warn",
                "artifacts": {
                    "output_dir": str(cls.artifact_dir),
                },
                "model": {
                    "backend": "xgboost",
                    "allow_backend_fallback": False,
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
                },
            },
        )
        train_from_config(ROOT_DIR, config)
        cls.service = CropSuitabilityInferenceService.from_artifact_dir(cls.artifact_dir, root_dir=ROOT_DIR)
        cls.dataset = pd.read_csv(ROOT_DIR / "data" / "processed" / "final_ml_dataset.csv")
        cls.sample_row = cls.dataset.iloc[0].to_dict()

    def test_valid_input_returns_ranked_probabilities(self) -> None:
        payload = {
            "region": "Mysuru",
            "features": {
                feature: self.sample_row[feature]
                for feature in self.service.numeric_features + self.service.categorical_features
            },
        }
        result = self.service.predict(payload, top_n=len(self.service.label_columns))
        self.assertEqual(len(result["recommendations"]), len(self.service.label_columns))
        self.assertAlmostEqual(sum(item["score"] for item in result["recommendations"]), 1.0, places=5)
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)
        self.assertIsInstance(result["explanation"], str)
        self.assertIn("confidence_breakdown", result)
        self.assertIn("top_features", result)
        self.assertTrue(result["top_features"])
        self.assertIn("why_not", result)

    def test_missing_input_is_filled_with_warnings(self) -> None:
        payload = {
            "region": "Pune",
            "features": {
                "temp_avg": 30.0,
                "rain_total": 42.0,
            },
        }
        result = self.service.predict(payload)
        self.assertGreater(len(result["warnings"]), 0)
        self.assertGreater(result["input_quality"]["context_autofill_count"], 0)
        self.assertTrue(result["recommendations"])
        self.assertIn("confidence_breakdown", result)

    def test_extreme_values_are_clipped(self) -> None:
        payload = {
            "region": "Pune",
            "features": {
                "temp_avg": 120.0,
                "rain_total": 99999.0,
                "humidity_avg": -50.0,
                "pH": 25.0,
                "N_class": "extreme",
                "P_class": "medium",
                "K_class": "medium",
                "fertility_class": "medium",
            },
            "irrigation_index": 5.0,
            "rotation_score": -2.0,
            "geo_confidence": 7.0,
        }
        result = self.service.predict(payload)
        self.assertGreater(result["input_quality"]["clipped_feature_count"], 0)
        self.assertGreater(len(result["warnings"]), 0)
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)
        self.assertIn("top_features", result)

    def test_physical_bounds_are_enforced_for_real_world_features(self) -> None:
        payload = {
            "region": "Pune",
            "features": {
                "temp_avg": 30.0,
                "rain_total": -25.0,
                "humidity_avg": -10.0,
                "dry_spell_days": -3.0,
                "pH": 6.7,
                "N": -50.0,
                "P": -10.0,
                "K": -5.0,
                "N_class": "medium",
                "P_class": "medium",
                "K_class": "medium",
                "fertility_class": "medium",
            },
        }
        prepared = self.service.prepare_input(payload)
        row = prepared.frame.iloc[0]
        self.assertGreaterEqual(float(row["rain_total"]), 0.0)
        self.assertGreaterEqual(float(row["humidity_avg"]), 0.0)
        self.assertGreaterEqual(float(row["dry_spell_days"]), 0.0)
        self.assertGreaterEqual(float(row["N"]), 0.0)
        self.assertGreaterEqual(float(row["P"]), 0.0)
        self.assertGreaterEqual(float(row["K"]), 0.0)

    def test_scenario_simulation_returns_comparison_rows(self) -> None:
        payload = {
            "region": "Pune",
            "features": {
                "temp_avg": 29.0,
                "rain_total": 35.0,
                "humidity_avg": 70.0,
                "max_temp": 34.0,
                "max_temp_3d": 34.0,
                "rain_lag_14": 30.0,
                "pH": 6.7,
                "N": 340.0,
                "P": 18.0,
                "K": 220.0,
                "N_class": "medium",
                "P_class": "medium",
                "K_class": "medium",
                "fertility_class": "medium",
            },
            "irrigation_index": 0.5,
            "rotation_score": 0.6,
        }
        result = self.service.simulate_scenarios(payload, scenario_names=["low_rainfall"])
        self.assertIn("base_prediction", result)
        self.assertIn("scenario_results", result)
        self.assertIn("low_rainfall", result["scenario_results"])
        low_rainfall_result = result["scenario_results"]["low_rainfall"]
        comparison = low_rainfall_result["comparison"]
        self.assertIn("rows", comparison)
        self.assertTrue(comparison["rows"])
        self.assertIn("top_crop_before", comparison)
        self.assertIn("top_crop_after", comparison)
        self.assertIn("effect_strength", comparison)
        self.assertIn("distribution_shift", comparison)
        self.assertIn("applied_changes", low_rainfall_result)
        self.assertTrue(low_rainfall_result["applied_changes"])
        self.assertIn("average_abs_score_delta", comparison)
        self.assertIn("scenario_adjustment", low_rainfall_result)
        self.assertIn("rule_shift", low_rainfall_result)

    def test_explain_scenario_returns_structured_explanation(self) -> None:
        payload = {
            "region": "Pune",
            "features": {
                "temp_avg": 29.0,
                "rain_total": 35.0,
                "humidity_avg": 70.0,
                "max_temp": 34.0,
                "max_temp_3d": 34.0,
                "rain_lag_14": 30.0,
                "pH": 6.7,
                "N": 340.0,
                "P": 18.0,
                "K": 220.0,
                "N_class": "medium",
                "P_class": "medium",
                "K_class": "medium",
                "fertility_class": "medium",
            },
            "irrigation_index": 0.5,
            "rotation_score": 0.6,
        }
        mocked_explanation = {
            "scenario_summary": "Scenario summary.",
            "environmental_change": "Environmental change.",
            "crop_response_analysis": "Crop response.",
            "ranking_changes": "Ranking change.",
            "key_drivers": ["rainfall"],
            "stability_assessment": "Stable.",
            "confidence_note": "Grounded.",
        }
        with mock.patch(
            "climate_pipeline.inference.generate_scenario_explanation",
            return_value=mocked_explanation,
        ):
            result = self.service.explain_scenario(payload, scenario_name="low_rainfall")
        self.assertEqual(result["scenario_name"], "low_rainfall")
        self.assertIn("scenario_result", result)
        self.assertIn("scenario_explanation", result["scenario_result"])
        self.assertEqual(result["scenario_result"]["scenario_explanation"]["scenario_summary"], "Scenario summary.")
        self.assertIn("scenario_explanation_ui", result["scenario_result"])

    def test_catalog_exposes_model_coverage(self) -> None:
        catalog = self.service.get_catalog()
        self.assertIn("coverage", catalog)
        self.assertIn("supported_crops", catalog)
        self.assertIn("numeric_features", catalog)
        self.assertIn("temporal_context", catalog)
        self.assertGreaterEqual(catalog["coverage"]["crop_count"], 1)
        self.assertEqual(catalog["coverage"]["crop_count"], len(catalog["supported_crops"]))
        self.assertIn("temp_avg", catalog["numeric_features"])

    def test_localized_context_returns_region_month_defaults(self) -> None:
        context = self.service.get_localized_context(
            region="Pune",
            state="Maharashtra",
            target_time="2024-06",
        )
        self.assertTrue(context["available"])
        self.assertEqual(context["resolved_region"], "Pune")
        self.assertEqual(context["target_month"], "06")
        self.assertIn("rain_total", context["feature_defaults"])
        self.assertIn("temp_avg", context["validation_bands"])

    def test_partial_input_uses_localized_autofill(self) -> None:
        payload = {
            "region": "Pune",
            "state": "Maharashtra",
            "target_time": "2024-06",
            "features": {
                "temp_avg": 29.0,
                "rain_total": 35.0,
                "humidity_avg": 70.0,
                "pH": 6.7,
                "N": 340.0,
                "P": 18.0,
                "K": 220.0,
                "N_class": "medium",
                "P_class": "medium",
                "K_class": "medium",
                "fertility_class": "medium",
            },
        }
        result = self.service.predict(payload)
        self.assertTrue(result["recommendations"])
        self.assertGreater(result["input_quality"]["context_autofill_count"], 0)
        self.assertIn("localized_context", result)
        self.assertEqual(result["localized_context"]["target_month"], "06")

    def test_zero_padded_target_month_is_accepted(self) -> None:
        self.assertEqual(self.service._normalize_categorical_value("target_month", "06"), "6")
        self.assertEqual(self.service._normalize_categorical_value("target_month", "6"), "6")
        self.assertEqual(self.service._normalize_categorical_value("target_season", "Kharif"), "kharif")


if __name__ == "__main__":
    unittest.main()
