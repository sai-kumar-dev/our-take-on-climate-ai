from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from climate_pipeline.llm_guide import (  # noqa: E402
    GroqGuideClient,
    LlmGuideNotConfiguredError,
    build_guide_prompt,
    extract_output_text,
)


class GroqGuideClientTests(unittest.TestCase):
    def test_support_metadata_is_disabled_without_api_key(self) -> None:
        client = GroqGuideClient(api_key="")
        metadata = client.support_metadata()
        self.assertFalse(metadata["enabled"])
        self.assertEqual(metadata["provider"], "groq")

    def test_generate_answer_requires_api_key(self) -> None:
        client = GroqGuideClient(api_key="")
        with self.assertRaises(LlmGuideNotConfiguredError):
            client.generate_answer(
                prediction={"recommendations": [{"crop": "sugarcane", "score": 0.8}]},
                region="Pune",
                state="Maharashtra",
            )

    def test_generate_answer_uses_responses_api(self) -> None:
        session = mock.Mock()
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "output_text": "Sugarcane looks strongest here. Check irrigation before planting.",
            "metadata": {"total_time": "0.42"},
        }
        session.post.return_value = response

        client = GroqGuideClient(api_key="test-key", session=session)
        result = client.generate_answer(
            prediction={
                "recommendations": [{"crop": "sugarcane", "score": 0.8}],
                "confidence": 0.74,
                "explanation": "Soil pH and rainfall support sugarcane.",
            },
            input_snapshot={"features": {"temp_avg": 27.0, "rain_total": 110.0}},
            preferred_language="English",
            user_question="Why is sugarcane on top?",
            region="Pune",
            state="Maharashtra",
        )

        self.assertIn("Sugarcane looks strongest", result["answer"])
        session.post.assert_called_once()
        self.assertEqual(session.post.call_args.args[0], "https://api.groq.com/openai/v1/responses")
        sent_payload = session.post.call_args.kwargs["json"]
        self.assertEqual(sent_payload["model"], client.model)
        self.assertIn("Pune", sent_payload["input"])

    def test_extract_output_text_falls_back_to_output_messages(self) -> None:
        payload = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "First line."},
                        {"type": "output_text", "text": "Second line."},
                    ],
                }
            ]
        }
        self.assertEqual(extract_output_text(payload), "First line.\nSecond line.")

    def test_build_guide_prompt_includes_core_prediction_details(self) -> None:
        prompt = build_guide_prompt(
            prediction={
                "recommendations": [{"crop": "sugarcane", "score": 0.81}],
                "confidence": 0.7,
                "explanation": "Higher soil pH favored sugarcane.",
                "warnings": ["Rainfall is above the typical local range."],
            },
            input_snapshot={"features": {"temp_avg": 28.0, "pH": 6.8}},
            preferred_language="Hindi",
            user_question="Why did this happen?",
            region="Pune",
            state="Maharashtra",
        )
        self.assertIn("Preferred language: Hindi", prompt)
        self.assertIn("Top crop: Sugarcane", prompt)
        self.assertIn("Why did this happen?", prompt)


if __name__ == "__main__":
    unittest.main()
