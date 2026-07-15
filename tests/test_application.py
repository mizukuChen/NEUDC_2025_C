from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from src.application import (
    CameraFrameBuffer,
    MeasurementApplication,
    _draw_a4_location,
    _normal_summary,
)
from src.vision.measure import Measurement, MeasurementResult
from src.vision.measurement_tasks import NormalMeasurementResult
from src.vision.fisheye import FisheyeCalibration
from src.vision.rectify import rectify_a4


class ApplicationPresentationTest(unittest.TestCase):
    def test_persistent_camera_buffer_returns_three_frames(self) -> None:
        capture = Mock()
        capture.isOpened.return_value = True
        capture.grab.return_value = True
        counter = 0

        def read():
            nonlocal counter
            counter += 1
            return True, np.full((8, 12, 3), counter % 255, dtype=np.uint8)

        capture.read.side_effect = read
        with patch("src.application.cv2.VideoCapture", return_value=capture):
            buffer = CameraFrameBuffer(0, 1920, 1080, warmup_frames=0)
            buffer.start()
            frames = buffer.frames(timeout_s=1.0)
            buffer.stop()

        self.assertEqual(len(frames), 3)
        self.assertTrue(all(frame.shape == (8, 12, 3) for frame in frames))
        capture.release.assert_called()

    def test_direct_rectification_matches_two_step_pipeline(self) -> None:
        width, height = 160, 120
        calibration = FisheyeCalibration(
            camera_matrix=np.asarray(
                [[105.0, 0.0, 80.0], [0.0, 105.0, 60.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            distortion=np.asarray([-0.03, 0.004, 0.0, 0.0], dtype=np.float64).reshape(4, 1),
            image_size=(width, height),
            rms_px=0.1,
            used_images=(),
            rejected_images=(),
            pattern_size=(8, 12),
            square_size_mm=29.0,
        )
        yy, xx = np.indices((height, width))
        raw = np.dstack(
            [
                (xx * 255 // width).astype(np.uint8),
                (yy * 255 // height).astype(np.uint8),
                ((xx + yy) * 255 // (width + height)).astype(np.uint8),
            ]
        )
        corners = np.asarray(
            [[42, 18], [120, 20], [126, 102], [35, 100]],
            dtype=np.float64,
        )
        application = MeasurementApplication()
        fixed, interpolation, map_x, map_y, _ = application._camera_undistortion_maps(
            calibration,
            (width, height),
        )
        undistorted = cv2.remap(raw, fixed, interpolation, cv2.INTER_LINEAR)
        expected = rectify_a4(undistorted, corners, 420, 594)
        actual = application._direct_rectify_frames(
            [raw],
            corners,
            map_x,
            map_y,
        )[0]
        difference = np.mean(
            np.abs(expected.astype(np.int16) - actual.astype(np.int16))
        )
        self.assertLess(float(difference), 3.0)
        self.assertIs(
            application._camera_undistortion_maps(calibration, (width, height))[0],
            fixed,
        )

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

    def test_capture_frames_uses_three_stable_yuyv_frames(self) -> None:
        capture = Mock()
        capture.isOpened.return_value = True
        capture.grab.return_value = True
        frames = [np.full((8, 12, 3), value, dtype=np.uint8) for value in (1, 2, 3)]
        capture.read.side_effect = [(True, frame) for frame in frames]
        with patch("src.application.cv2.VideoCapture", return_value=capture) as video_capture:
            actual = MeasurementApplication.capture_frames(warmup_frames=2)
        self.assertEqual(len(actual), 3)
        np.testing.assert_array_equal(actual[-1], frames[-1])
        self.assertEqual(capture.grab.call_count, 2)
        self.assertEqual(capture.read.call_count, 3)
        capture.release.assert_called_once()
        video_capture.assert_called_once_with(0, cv2.CAP_V4L2)

    def test_tilted_camera_frames_route_without_distance_output(self) -> None:
        application = MeasurementApplication()
        application._fisheye = object()
        application._correction = None
        raw_frames = tuple(np.full((120, 180, 3), value, dtype=np.uint8) for value in (1, 2, 3))
        work_image = np.full((594, 420, 3), 240, dtype=np.uint8)
        shape = Measurement("square", 60.0, (105.0, 148.5), 0.0, 0.9, "overlap_lowres")
        geometry = MeasurementResult((shape,), "overlap", 2.0, 0.8, work_image)
        task = NormalMeasurementResult(None, geometry, shape, shape.x_mm)
        plane = SimpleNamespace(
            corners=np.asarray([[10, 10], [50, 10], [50, 100], [10, 100]], dtype=np.float32)
        )
        distance = SimpleNamespace(
            corrected_distance_mm=1500.0,
            reprojection_error_px=0.5,
            rvec=np.asarray([[0.0], [np.deg2rad(55.0)], [0.0]]),
        )

        with (
            patch.object(
                application,
                "_camera_undistortion_maps",
                return_value=(None, None, None, None, np.eye(3)),
            ),
            patch("src.application.cv2.remap", side_effect=lambda image, *_args, **_kwargs: image),
            patch("src.application.find_target_plane", return_value=plane),
            patch.object(
                application,
                "_direct_rectify_frames",
                return_value=[work_image, work_image, work_image],
            ) as direct_rectify,
            patch("src.application.estimate_plane_center_depth", return_value=distance),
            patch("src.application.measure_normal_low_resolution", return_value=task) as measure,
        ):
            result = application.process_camera_frames(raw_frames, "normal")

        direct_rectify.assert_called_once()
        self.assertEqual(len(measure.call_args.args[0]), 3)
        self.assertIsNone(measure.call_args.args[1])
        self.assertEqual(result.x_mm, 60.0)
        self.assertIsNone(result.distance_mm)
        self.assertIn("发挥（4）倾斜物面", result.summary)
        self.assertIn("目标距离：发挥（4）不要求输出", result.summary)
        self.assertEqual(result.rectified_a4.shape[:2], (594, 420))
        np.testing.assert_array_equal(result.input_image, raw_frames[-1])

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
