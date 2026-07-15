from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np

from scripts.targets.generate_numbered_squares import PRESETS, render
from src.vision.hog import extract_hog
from src.vision.measure import Measurement, MeasurementResult
from src.vision.measurement_tasks import (
    numbered_measurement_from_recognitions,
    recognize_numbered_squares,
)
from src.vision.recognize_digits import (
    DigitRecognition,
    HogSvmDigitRecognizer,
    extract_digit_patch,
)


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

    def test_all_digits_are_recognized_once_then_selected_without_recognition(self) -> None:
        shapes = tuple(
            Measurement("square", size, center, 0.0, 1.0, "test")
            for size, center in (
                (50.0, (50.0, 80.0)),
                (60.0, (105.0, 150.0)),
                (70.0, (160.0, 220.0)),
            )
        )
        geometry = MeasurementResult(shapes, "separate", 2.0, 1.0)
        recognizer = Mock()
        recognizer.recognize_square.side_effect = [
            DigitRecognition(digit, 0.9, np.zeros((28, 28), dtype=np.uint8))
            for digit in (2, 3, 4)
        ]

        squares = recognize_numbered_squares(
            np.full((594, 420, 3), 255, dtype=np.uint8),
            geometry,
            recognizer,
        )
        selected_three = numbered_measurement_from_recognitions(
            geometry,
            squares,
            3,
            1500.0,
        )
        selected_four = numbered_measurement_from_recognitions(
            geometry,
            squares,
            4,
            1500.0,
        )

        self.assertEqual(recognizer.recognize_square.call_count, 3)
        self.assertEqual([item.digit for item in squares], [2, 3, 4])
        self.assertEqual(selected_three.x_mm, 60.0)
        self.assertEqual(selected_four.x_mm, 70.0)


if __name__ == "__main__":
    unittest.main()
