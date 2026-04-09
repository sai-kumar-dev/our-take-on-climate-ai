from __future__ import annotations

import os
import shutil
import sys
import unittest
from pathlib import Path
from uuid import uuid4

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env_loader import load_project_env, parse_env_file  # noqa: E402


class EnvLoaderTests(unittest.TestCase):
    def make_temp_root(self) -> Path:
        temp_root = ROOT_DIR / ".tmp" / f"env-loader-tests-{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(temp_root, ignore_errors=True))
        return temp_root

    def test_parse_env_file_handles_comments_quotes_and_export(self) -> None:
        temp_root = self.make_temp_root()
        env_path = temp_root / ".env"
        env_path.write_text(
            "\n".join(
                [
                    "# comment",
                    "GROQ_API_KEY='abc123'",
                    'GROQ_MODEL="llama-test"',
                    "export API_BASE_URL=http://127.0.0.1:8000",
                ]
            ),
            encoding="utf-8",
        )
        parsed = parse_env_file(env_path)

        self.assertEqual(parsed["GROQ_API_KEY"], "abc123")
        self.assertEqual(parsed["GROQ_MODEL"], "llama-test")
        self.assertEqual(parsed["API_BASE_URL"], "http://127.0.0.1:8000")

    def test_load_project_env_sets_missing_values_without_overriding(self) -> None:
        root_dir = self.make_temp_root()
        (root_dir / ".env").write_text(
            "GROQ_API_KEY=from-file\nAPI_BASE_URL=http://127.0.0.1:8001\n",
            encoding="utf-8",
        )
        original_key = os.environ.get("GROQ_API_KEY")
        original_api_base_url = os.environ.get("API_BASE_URL")
        try:
            os.environ["GROQ_API_KEY"] = "from-env"
            os.environ.pop("API_BASE_URL", None)
            loaded_path = load_project_env(root_dir=root_dir)
            self.assertEqual(loaded_path, root_dir / ".env")
            self.assertEqual(os.environ["GROQ_API_KEY"], "from-env")
            self.assertEqual(os.environ["API_BASE_URL"], "http://127.0.0.1:8001")
        finally:
            if original_key is None:
                os.environ.pop("GROQ_API_KEY", None)
            else:
                os.environ["GROQ_API_KEY"] = original_key
            if original_api_base_url is None:
                os.environ.pop("API_BASE_URL", None)
            else:
                os.environ["API_BASE_URL"] = original_api_base_url


if __name__ == "__main__":
    unittest.main()
