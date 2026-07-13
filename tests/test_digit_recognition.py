from __future__ import annotations

import unittest
from pathlib import Path

import cv2

from scripts.targets.generate_numbered_squares import PRESETS, render
from src.vision.hog import extract_hog
from src.vision.recognize_digits import HogSvmDigitRecognizer, extract_digit_patch


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "assets" / "models" / "digit_hog_linear_svm.npz"


class DigitRecognitionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.recognizer = HogSvmDigitRecognizer.load(MODEL)

    def test_hog_feature_shape_and_finiteness(self) -> None:
        patch = extract_digit_patch(
            cv2.cvtColor(render(PRESETS["01"])[0], cv2.COLOR_GRAY2BGR),
            (PRESETS["01"][0].cx_mm * 300 / 25.4, PRESETS["01"][0].cy_mm * 300 / 25.4),
            PRESETS["01"][0].size_mm * 300 / 25.4,
            PRESETS["01"][0].angle_deg,
        )
        features = extract_hog(patch)
        self.assertEqual(features.shape, (1764,))
        self.assertTrue(bool((features >= 0).all()))

    def test_all_generated_digits(self) -> None:
        for case, squares in PRESETS.items():
            gray, _ = render(squares)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            scale = 300 / 25.4
            for square in squares:
                with self.subTest(case=case, digit=square.digit):
                    result = self.recognizer.recognize_square(
                        image,
                        (square.cx_mm * scale, square.cy_mm * scale),
                        square.size_mm * scale,
                        square.angle_deg,
                    )
                    self.assertEqual(result.digit, square.digit)
                    self.assertGreater(result.confidence, 0.8)


if __name__ == "__main__":
    unittest.main()
