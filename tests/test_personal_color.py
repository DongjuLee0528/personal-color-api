import unittest
from unittest.mock import patch

import numpy as np

from src.analyzers import PersonalColorAnalyzer, classify_from_metrics


class PersonalColorAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = PersonalColorAnalyzer()

    def test_classify_from_metrics_warm_vs_cool(self) -> None:
        warm_metrics = {"L": 60.0, "a": 25.0, "b": 30.0, "ITA": 35.0}
        cool_metrics = {"L": 40.0, "a": -5.0, "b": -10.0, "ITA": 5.0}

        season, tone = classify_from_metrics(warm_metrics)
        self.assertEqual(tone, "Warm")
        self.assertIn(season, {"Spring", "Autumn"})

        season_cool, tone_cool = classify_from_metrics(cool_metrics)
        self.assertEqual(tone_cool, "Cool")
        self.assertIn(season_cool, {"Summer", "Winter"})

    def test_analyze_returns_guardrail_when_no_face(self) -> None:
        image_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        with patch("src.analyzers.personal_color.detect_landmarks", return_value=None):
            result = self.analyzer.analyze(image_bgr)

        self.assertEqual(result.get("status"), "guardrail")
        self.assertEqual(result.get("code"), "NO_FACE")

    def test_analyze_guardrail_on_low_quality_roi(self) -> None:
        image_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        fake_landmarks = np.zeros((468, 2), dtype=np.float32)
        roi_metrics = {
            "left_cheek": {"mean_l": 5.0, "variance": 1.0, "pixels": 500.0, "reason": "ROI_TOO_DARK"}
        }

        with patch("src.analyzers.personal_color.detect_landmarks", return_value=fake_landmarks), patch(
            "src.analyzers.personal_color.extract_skin_rois",
            return_value=({}, roi_metrics),
        ):
            result = self.analyzer.analyze(image_bgr)

        self.assertEqual(result.get("status"), "guardrail")
        self.assertEqual(result.get("code"), "LOW_QUALITY")
        self.assertIn("landmarks", result)
        self.assertIsNone(result.get("metrics"))

    def test_analyze_successful_flow(self) -> None:
        image_bgr = np.full((128, 128, 3), (180, 170, 160), dtype=np.uint8)
        fake_landmarks = np.zeros((468, 2), dtype=np.float32)
        roi_patch = np.full((32, 32, 3), (200, 180, 160), dtype=np.uint8)
        roi_metrics = {
            "left_cheek": {"mean_l": 70.0, "variance": 15.0, "pixels": 1024.0},
            "right_cheek": {"mean_l": 68.0, "variance": 14.0, "pixels": 1024.0},
        }
        rois = {"left_cheek": roi_patch, "right_cheek": roi_patch}

        with patch("src.analyzers.personal_color.detect_landmarks", return_value=fake_landmarks), patch(
            "src.analyzers.personal_color.extract_skin_rois",
            return_value=(rois, roi_metrics),
        ):
            result = self.analyzer.analyze(image_bgr)

        self.assertEqual(result.get("status"), "ok")
        self.assertIn(result.get("season"), DEFAULT_SEASONS)
        self.assertIn(result.get("tone"), {"Warm", "Cool"})
        self.assertIn("metrics", result)


DEFAULT_SEASONS = {"Spring", "Summer", "Autumn", "Winter"}


if __name__ == "__main__":
    unittest.main()
