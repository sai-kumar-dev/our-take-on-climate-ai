from __future__ import annotations

import json
import secrets
import shutil
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from climate_pipeline.pipeline import run_pipeline_from_config  # noqa: E402


class PipelineOrchestrationTests(unittest.TestCase):
    def test_pipeline_builds_outputs_from_small_raw_inputs(self) -> None:
        tmp_root = ROOT_DIR / "artifacts" / "test_pipeline_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        temp_dir = tmp_root / f"run_{secrets.token_hex(4)}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            root = temp_dir
            climate_path = root / "climate.csv"
            soil_path = root / "soil.csv"
            crop_path = root / "crop.csv"

            pd.DataFrame(
                [
                    {"region": "Pune", "state": "Maharashtra", "date": "2024-06-01", "temperature": 28.0, "rainfall": 12.0, "humidity": 70.0},
                    {"region": "Pune", "state": "Maharashtra", "date": "2024-06-02", "temperature": 29.0, "rainfall": 5.0, "humidity": 72.0},
                    {"region": "Pune", "state": "Maharashtra", "date": "2024-06-03", "temperature": 30.0, "rainfall": 8.0, "humidity": 71.0},
                ]
            ).to_csv(climate_path, index=False)

            pd.DataFrame(
                [
                    {"region": "Pune", "state": "Maharashtra", "ph": 6.8, "n_kg_ha": 320.0, "p_kg_ha": 18.0, "k_kg_ha": 210.0},
                ]
            ).to_csv(soil_path, index=False)

            pd.DataFrame(
                [
                    {"region": "Pune", "state": "Maharashtra", "year": 2024, "month": 6, "crop": "sugarcane", "area": 60.0, "production": 100.0},
                    {"region": "Pune", "state": "Maharashtra", "year": 2024, "month": 6, "crop": "maize", "area": 40.0, "production": 50.0},
                ]
            ).to_csv(crop_path, index=False)

            config = {
                "target_time_level": "monthly",
                "dry_spell_rain_threshold_mm": 1.0,
                "soil_class_bins": {"N": [280, 560], "P": [10, 25], "K": [110, 280]},
                "region_aliases": {},
                "context_defaults": {
                    "irrigation_index": 0.45,
                    "rotation_score": 0.65,
                    "fertility_class": "medium",
                },
                "datasets": {
                    "climate": {
                        "path": str(climate_path),
                        "spatial_level": "district",
                        "columns": {
                            "region": "region",
                            "state": "state",
                            "date": "date",
                            "temperature": "temperature",
                            "rainfall": "rainfall",
                            "humidity": "humidity",
                        },
                    },
                    "soil": {
                        "path": str(soil_path),
                        "spatial_level": "district",
                        "columns": {
                            "region": "region",
                            "state": "state",
                            "ph": "ph",
                            "n": "n_kg_ha",
                            "p": "p_kg_ha",
                            "k": "k_kg_ha",
                        },
                    },
                    "crop": {
                        "path": str(crop_path),
                        "spatial_level": "district",
                        "columns": {
                            "region": "region",
                            "state": "state",
                            "year": "year",
                            "month": "month",
                            "crop": "crop",
                            "area": "area",
                            "production": "production",
                        },
                    },
                },
                "outputs": {
                    "inspection_report": "reports/inspection.json",
                    "validation_report": "reports/validation.json",
                    "final_dataset": "data/final_ml_dataset.csv",
                    "summary_stats": "reports/summary_stats.csv",
                },
            }

            result = run_pipeline_from_config(root_dir=root, raw_config=config)
            final_dataset_path = Path(result["outputs"]["final_dataset"])
            validation_report_path = Path(result["outputs"]["validation_report"])
            inspection_report_path = Path(result["outputs"]["inspection_report"])
            self.assertTrue(final_dataset_path.exists())
            self.assertTrue(validation_report_path.exists())
            self.assertTrue(inspection_report_path.exists())

            final_frame = pd.read_csv(final_dataset_path)
            self.assertGreaterEqual(len(final_frame), 1)
            self.assertIn("crop_prob_sugarcane", final_frame.columns)
            self.assertIn("crop_prob_maize", final_frame.columns)
            self.assertIn("target_month", final_frame.columns)

            validation_report = json.loads(validation_report_path.read_text(encoding="utf-8"))
            self.assertIn("merge_report", validation_report)
            self.assertEqual(validation_report["row_count"], len(final_frame))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
