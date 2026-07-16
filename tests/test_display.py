from __future__ import annotations

import unittest

import numpy as np

from src.application import ApplicationResult
from src.display.measurement import build_measurement_view, draw_a4_location, normal_summary
from src.vision.measure import Measurement, MeasurementResult
from src.vision.measurement_tasks import NormalMeasurementResult


class MeasurementDisplayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.first = Measurement(
            "square",
            50.0,
            (60.0, 80.0),
            0.0,
            1.0,
            "contour",
        )
        self.second = Measurement(
            "square",
            70.0,
            (140.0, 180.0),
            5.0,
            1.0,
            "contour",
        )
        self.geometry = MeasurementResult(
            (self.first, self.second),
            "separate",
            10.0,
            1.0,
        )
        self.task = NormalMeasurementResult(
            1500.0,
            self.geometry,
            self.first,
            self.first.x_mm,
        )

    def test_normal_summary_is_chinese_and_lists_all_targets(self) -> None:
        summary = normal_summary(self.task, 0.25)
        self.assertIn("识别成功", summary)
        self.assertIn("多个分离目标", summary)
        self.assertIn("2 个正方形", summary)
        self.assertIn("50.00 mm", summary)
        self.assertIn("70.00 mm", summary)

    def test_a4_location_draws_green_outline(self) -> None:
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        corners = np.asarray(
            [[20, 20], [140, 20], [140, 100], [20, 100]],
            dtype=np.float64,
        )
        debug = draw_a4_location(image, corners)
        self.assertEqual(debug.shape, image.shape)
        self.assertGreater(int(np.count_nonzero(debug[:, :, 1])), 0)
        self.assertEqual(int(np.count_nonzero(image)), 0)

    def test_build_view_generates_all_ui_outputs(self) -> None:
        input_image = np.zeros((120, 160, 3), dtype=np.uint8)
        rectified = np.full((297, 210, 3), 255, dtype=np.uint8)
        corners = np.asarray(
            [[20, 20], [140, 20], [140, 100], [20, 100]],
            dtype=np.float64,
        )
        result = ApplicationResult(
            mode="normal",
            x_mm=self.task.x_mm,
            distance_mm=self.task.distance_mm,
            reprojection_error_px=0.25,
            plane_axis_angle_deg=90.0,
            tilted_plane=False,
            task=self.task,
            selected_index=0,
            input_image=input_image,
            undistorted_image=input_image,
            a4_corners=corners,
            rectified_a4=rectified,
        )
        view = build_measurement_view(result)
        self.assertIn("识别成功", view.summary)
        self.assertIn("目标尺寸 x：50.00 mm", view.target_text)
        self.assertEqual(view.located_a4.shape, input_image.shape)
        self.assertEqual(view.annotated_a4.shape, rectified.shape)


if __name__ == "__main__":
    unittest.main()
