from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import project_doctor  # noqa: E402
import train_model  # noqa: E402
from climate_pipeline.utils import read_config  # noqa: E402


class PublicArtifactPathTests(unittest.TestCase):
    def make_temp_root(self) -> Path:
        temp_root = ROOT_DIR / ".tmp" / f"public-artifact-tests-{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(temp_root, ignore_errors=True))
        return temp_root

    def touch_file(self, root_dir: Path, relative_path: str, content: str = "") -> None:
        file_path = root_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    def test_train_cli_default_uses_shipped_sample_config(self) -> None:
        self.assertEqual(train_model.DEFAULT_TRAINING_CONFIG_PATH, "configs/training_config.json")
        config = read_config(ROOT_DIR / train_model.DEFAULT_TRAINING_CONFIG_PATH)
        self.assertEqual(config["data"]["dataset_path"], "data/sample_dataset.csv")

    def test_project_doctor_accepts_public_artifact_without_private_assets(self) -> None:
        temp_root = self.make_temp_root()
        for relative_path in [
            "src/app_api_entry.py",
            "src/ui_app_source.py",
            "run_all.bat",
            "train_model.py",
            "run_pipeline.py",
            "configs/training_config.json",
            "data/sample_dataset.csv",
            "Dockerfile",
            "docker-compose.yml",
            ".github/workflows/ci.yml",
        ]:
            self.touch_file(temp_root, relative_path)

        with mock.patch.object(project_doctor, "ROOT_DIR", temp_root):
            results, ready_to_run, ready_to_retrain = project_doctor.check_paths()

        status_by_label = {item.label: item.status for item in results}
        self.assertTrue(ready_to_run)
        self.assertFalse(ready_to_retrain)
        self.assertEqual(status_by_label["Production model"], "optional-missing")
        self.assertEqual(status_by_label["Full training dataset"], "optional-missing")


if __name__ == "__main__":
    unittest.main()
