from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.application import (
    MeasurementApplication,
    _draw_a4_location,
    _normal_summary,
)
from src.vision.measure import Measurement, MeasurementResult
from src.vision.measurement_tasks import NormalMeasurementResult


class ApplicationPresentationTest(unittest.TestCase):
    def test_missing_camera_has_required_message(self) -> None:
        capture = Mock()
        capture.isOpened.return_value = False
        with patch("src.application.cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(RuntimeError, "^找不到camera$"):
                MeasurementApplication.capture_frame()
        capture.release.assert_called_once()

    def test_disconnected_camera_frame_has_required_message(self) -> None:
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)
        with patch("src.application.cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(RuntimeError, "^找不到camera$"):
                MeasurementApplication.capture_frame()
        capture.release.assert_called_once()

    def test_normal_summary_is_chinese_and_lists_all_targets(self) -> None:
        first = Measurement("square", 50.0, (60.0, 80.0), 0.0, 1.0, "contour")
        second = Measurement("square", 70.0, (140.0, 180.0), 5.0, 1.0, "contour")
        geometry = MeasurementResult((first, second), "separate", 10.0, 1.0)
        task = NormalMeasurementResult(1500.0, geometry, first, first.x_mm)
        summary = _normal_summary(task, 0.25)
        self.assertIn("识别成功", summary)
        self.assertIn("多个分离目标", summary)
        self.assertIn("2 个正方形", summary)
        self.assertIn("50.00 mm", summary)
        self.assertIn("70.00 mm", summary)

    def test_a4_location_draws_green_outline(self) -> None:
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        corners = np.asarray([[20, 20], [140, 20], [140, 100], [20, 100]], dtype=np.float64)
        debug = _draw_a4_location(image, corners)
        self.assertEqual(debug.shape, image.shape)
        self.assertGreater(int(np.count_nonzero(debug[:, :, 1])), 0)
        self.assertEqual(int(np.count_nonzero(image)), 0)


if __name__ == "__main__":
    unittest.main()
