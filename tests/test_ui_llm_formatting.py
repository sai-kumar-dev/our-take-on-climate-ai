from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.ui_app_source import clean_llm_text, split_llm_sections  # noqa: E402


class UiLlmFormattingTests(unittest.TestCase):
    def test_clean_llm_text_removes_basic_markdown_noise(self) -> None:
        raw = "## **What this means**\nThis is *simple* text."
        cleaned = clean_llm_text(raw)
        self.assertNotIn("##", cleaned)
        self.assertNotIn("**", cleaned)
        self.assertNotIn("*", cleaned)
        self.assertIn("What this means", cleaned)

    def test_split_llm_sections_extracts_expected_farmer_sections(self) -> None:
        answer = "\n".join(
            [
                "**What this means**",
                "Sugarcane is the current best match for these inputs.",
                "",
                "**Why it came first**",
                "The soil pH and district pattern supported it.",
                "",
                "**What to check next**",
                "Please verify water availability and local seed access.",
            ]
        )
        sections = split_llm_sections(answer)
        self.assertEqual(
            sections,
            [
                ("What this means", "Sugarcane is the current best match for these inputs."),
                ("Why it came first", "The soil pH and district pattern supported it."),
                ("What to check next", "Please verify water availability and local seed access."),
            ],
        )

    def test_split_llm_sections_falls_back_to_single_section(self) -> None:
        sections = split_llm_sections("This is a plain answer without headings.")
        self.assertEqual(sections, [("Simple explanation", "This is a plain answer without headings.")])


if __name__ == "__main__":
    unittest.main()
