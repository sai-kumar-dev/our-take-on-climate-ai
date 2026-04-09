from __future__ import annotations

import json
import sys
import unittest
from unittest import mock
from pathlib import Path

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app import create_app  # noqa: E402
from climate_pipeline.feedback import FeedbackStore  # noqa: E402
from climate_pipeline.inference import CropSuitabilityInferenceService  # noqa: E402
from climate_pipeline.training import DEFAULT_TRAINING_CONFIG, merge_nested_dicts, train_from_config  # noqa: E402


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact_dir = ROOT_DIR / "artifacts" / "test_api_model"
        cls.feedback_dir = ROOT_DIR / "artifacts" / "test_feedback_store"
        cls.artifact_dir.mkdir(parents=True, exist_ok=True)
        cls.feedback_dir.mkdir(parents=True, exist_ok=True)
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
        service = CropSuitabilityInferenceService.from_artifact_dir(cls.artifact_dir, root_dir=ROOT_DIR)
        feedback_store = FeedbackStore(storage_dir=cls.feedback_dir, signing_secret="test-secret")
        cls.client = TestClient(create_app(service=service, feedback_store=feedback_store))

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("request_id", payload)

    def test_sanity_endpoint(self) -> None:
        response = self.client.get("/sanity")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("sanity_checks", payload)

    def test_metrics_endpoint(self) -> None:
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("total_requests", payload)
        self.assertIn("avg_latency_ms", payload)
        self.assertIn("error_rate", payload)

    def test_catalog_endpoint(self) -> None:
        response = self.client.get("/catalog")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("coverage", payload)
        self.assertIn("supported_crops", payload)
        self.assertIn("feedback", payload)
        self.assertIn("temporal_context", payload)
        self.assertIn("guidance_scope", payload)
        self.assertGreaterEqual(payload["coverage"]["crop_count"], 1)

    def test_context_endpoint(self) -> None:
        response = self.client.get(
            "/context",
            params={
                "state": "Maharashtra",
                "region": "Pune",
                "target_time": "2024-06",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["available"])
        self.assertEqual(payload["resolved_region"], "Pune")
        self.assertEqual(payload["target_month"], "06")
        self.assertIn("feature_defaults", payload)
        self.assertIn("guidance_scope", payload)

    def test_simulate_endpoint(self) -> None:
        response = self.client.post(
            "/simulate",
            json={
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
                    "fertility_class": "medium"
                },
                "irrigation_index": 0.5,
                "rotation_score": 0.6,
                "scenario_names": ["low_rainfall", "heatwave"]
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("base_prediction", payload)
        self.assertIn("scenario_results", payload)
        self.assertIn("low_rainfall", payload["scenario_results"])

    def test_predict_endpoint_with_valid_payload(self) -> None:
        response = self.client.post(
            "/predict",
            json={
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
                    "fertility_class": "medium"
                },
                "irrigation_index": 0.5,
                "rotation_score": 0.6
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("recommendations", payload)
        self.assertIn("confidence", payload)
        self.assertIn("explanation", payload)
        self.assertIn("top_features", payload)
        self.assertIn("confidence_breakdown", payload)
        self.assertIn("request_id", payload)
        self.assertIn("prediction_time_ms", payload)
        self.assertIn("guidance_scope", payload)

    def test_predict_endpoint_rejects_empty_payload(self) -> None:
        response = self.client.post("/predict", json={"region": "Pune", "features": {}})
        self.assertEqual(response.status_code, 422)

    def test_predict_endpoint_handles_extreme_values(self) -> None:
        response = self.client.post(
            "/predict",
            json={
                "region": "Pune",
                "features": {
                    "temp_avg": 300.0,
                    "rain_total": 999999.0,
                    "pH": -4.0,
                    "N_class": "unknown",
                    "P_class": "medium",
                    "K_class": "medium",
                    "fertility_class": "medium"
                },
                "irrigation_index": 3.0
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertGreaterEqual(len(payload["warnings"]), 1)
        self.assertIn("why_not", payload)

    def test_feedback_endpoint_persists_feedback(self) -> None:
        response = self.client.post(
            "/feedback",
            json={
                "request_id": "predict-request-123",
                "region": "Pune",
                "state": "Maharashtra",
                "preferred_language": "Marathi",
                "selected_crop": "sugarcane",
                "actual_crop": "sugarcane",
                "outcome_label": "useful",
                "helpfulness_rating": 5,
                "clarity_rating": 4,
                "consent_for_training": True,
                "comment": "Simple enough to explain to a first-time farmer. Reach me at test@example.com or +91 98765 43210.",
                "input_snapshot": {
                    "features": {
                        "temp_avg": 29.0,
                        "rain_total": 35.0,
                    },
                },
                "prediction_snapshot": {
                    "recommendations": [
                        {"crop": "sugarcane", "score": 0.62},
                    ],
                    "confidence": 0.81,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "stored")
        self.assertEqual(payload["linked_request_id"], "predict-request-123")
        self.assertTrue(payload["integrity_protected"])
        self.assertEqual(payload["review_status"], "pending_human_review")
        self.assertFalse(payload["eligible_for_training"])

        storage_path = self.feedback_dir / payload["storage_file"]
        self.assertTrue(storage_path.exists())
        with storage_path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle.readlines() if line.strip()]
        self.assertTrue(lines)
        last_record = json.loads(lines[-1])
        self.assertEqual(last_record["region"], "Pune")
        self.assertEqual(last_record["preferred_language"], "Marathi")
        self.assertTrue(last_record["consent_for_training"])
        self.assertEqual(last_record["review_status"], "pending_human_review")
        self.assertEqual(last_record["training_use_status"], "pending_human_review")
        self.assertFalse(last_record["eligible_for_training"])
        self.assertIn("[redacted-email]", last_record["comment"])
        self.assertIn("[redacted-phone]", last_record["comment"])
        self.assertIn("record_hash", last_record)
        self.assertIn("integrity_signature", last_record)

    def test_create_app_is_lazy_about_default_service_loading(self) -> None:
        feedback_store = FeedbackStore(storage_dir=self.feedback_dir / "lazy_store", signing_secret="test-secret")
        with mock.patch("src.app_api_entry.get_service", side_effect=AssertionError("get_service should stay lazy")):
            app = create_app(service=None, feedback_store=feedback_store)
        self.assertIsNotNone(app)

    def test_feedback_rate_limit_rejects_burst_submissions(self) -> None:
        service = CropSuitabilityInferenceService.from_artifact_dir(self.artifact_dir, root_dir=ROOT_DIR)
        feedback_store = FeedbackStore(storage_dir=self.feedback_dir / "rate_limit_store", signing_secret="test-secret")
        rate_limited_client = TestClient(
            create_app(
                service=service,
                feedback_store=feedback_store,
                feedback_rate_limit_count=1,
                feedback_rate_limit_window_seconds=3600,
            )
        )
        payload = {
            "request_id": "predict-request-123",
            "region": "Pune",
            "state": "Maharashtra",
            "preferred_language": "English",
            "selected_crop": "sugarcane",
            "actual_crop": "sugarcane",
            "outcome_label": "useful",
            "helpfulness_rating": 4,
            "clarity_rating": 4,
            "consent_for_training": False,
            "comment": "First submission",
        }
        first_response = rate_limited_client.post("/feedback", json=payload)
        self.assertEqual(first_response.status_code, 200)
        second_response = rate_limited_client.post("/feedback", json={**payload, "comment": "Second submission"})
        self.assertEqual(second_response.status_code, 429)


if __name__ == "__main__":
    unittest.main()
