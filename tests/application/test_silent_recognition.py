from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from src.application import CameraSceneAnalysis, CameraSceneSignature
from src.silent_recognition import SilentRecognitionManager, scene_signatures_similar
from src.vision.measure import Measurement, MeasurementResult


def signature(value: int = 0) -> CameraSceneSignature:
    corners = np.asarray(
        [[0.2, 0.1], [0.8, 0.1], [0.8, 0.9], [0.2, 0.9]],
        dtype=np.float32,
    )
    mask = np.zeros((60, 40), dtype=bool)
    mask[15:45, 10:30] = True
    if value:
        mask[5:10, 5 + value : 10 + value] = True
    return CameraSceneSignature(corners, mask)


def scene_analysis(numbered: object | None = None) -> CameraSceneAnalysis:
    shape = Measurement(
        "square",
        80.0,
        (105.0, 148.5),
        0.0,
        1.0,
        "test",
    )
    geometry = MeasurementResult(
        shapes=(shape,),
        layout="single",
        pixels_per_mm=2.0,
        contour_foreground_coverage=1.0,
        measurement_image=np.zeros((594, 420, 3), dtype=np.uint8),
    )
    numbered_squares = None if numbered is None else (numbered,)
    return CameraSceneAnalysis(
        geometry=geometry,
        numbered_squares=numbered_squares,
        corrected_distance_mm=1000.0,
        reprojection_error_px=1.0,
        plane_axis_angle_deg=90.0,
        input_image=np.zeros((8, 12, 3), dtype=np.uint8),
        undistorted_image=np.zeros((8, 12, 3), dtype=np.uint8),
        a4_corners=np.zeros((4, 2), dtype=np.float64),
        rectified_a4=np.zeros((594, 420, 3), dtype=np.uint8),
    )


class SilentRecognitionTest(unittest.TestCase):
    def test_signature_comparison_tolerates_small_edge_jitter(self) -> None:
        first = signature()
        shifted = CameraSceneSignature(
            first.normalized_corners + 0.005,
            np.roll(first.content_mask, 1, axis=1),
        )
        moved = CameraSceneSignature(
            first.normalized_corners + 0.05,
            first.content_mask,
        )
        changed_mask = first.content_mask.copy()
        changed_mask[5:14, 4:20] = True
        changed = CameraSceneSignature(first.normalized_corners, changed_mask)
        self.assertTrue(scene_signatures_similar(first, shifted))
        self.assertFalse(scene_signatures_similar(first, moved))
        self.assertFalse(scene_signatures_similar(first, changed))

    def test_stable_scene_is_recognized_once_and_returned_from_cache(self) -> None:
        frames = tuple(np.full((8, 12, 3), value, dtype=np.uint8) for value in (1, 2, 3))
        analysis = object()
        expected = SimpleNamespace(summary="cached")
        application = Mock()
        application.buffered_camera_frames.side_effect = lambda **_kwargs: tuple(
            frame.copy() for frame in frames
        )
        application.probe_camera_frame.return_value = signature()
        application.analyze_camera_frames.return_value = analysis
        application.result_from_camera_analysis.return_value = expected
        manager = SilentRecognitionManager(application, stable_observations=3)
        manager.set_request("normal")

        manager.poll_once()
        manager.poll_once()
        self.assertIsNone(manager.cached_result("normal"))
        manager.poll_once()

        self.assertIs(manager.cached_result("normal"), expected)
        self.assertIs(manager.measure("normal"), expected)
        application.analyze_camera_frames.assert_called_once()
        application.process_camera_frames.assert_not_called()

    def test_same_camera_frame_is_not_counted_twice(self) -> None:
        frames = tuple(np.zeros((8, 12, 3), dtype=np.uint8) for _ in range(3))
        application = Mock()
        application.buffered_camera_frames.return_value = frames
        application.probe_camera_frame.return_value = signature()
        application.analyze_camera_frames.return_value = object()
        manager = SilentRecognitionManager(application, stable_observations=2)

        manager.poll_once()
        manager.poll_once()

        application.analyze_camera_frames.assert_not_called()

    def test_mode_and_digit_changes_reuse_one_scene_analysis(self) -> None:
        frames = tuple(np.zeros((8, 12, 3), dtype=np.uint8) for _ in range(3))
        analysis = object()
        application = Mock()
        application.buffered_camera_frames.side_effect = lambda **_kwargs: tuple(
            frame.copy() for frame in frames
        )
        application.probe_camera_frame.return_value = signature()
        application.analyze_camera_frames.return_value = analysis
        application.result_from_camera_analysis.side_effect = (
            lambda _analysis, mode, digit=None: (mode, digit)
        )
        manager = SilentRecognitionManager(application, stable_observations=1)
        manager.poll_once()

        self.assertEqual(manager.measure("normal"), ("normal", None))
        self.assertEqual(manager.measure("numbered", 3), ("numbered", 3))
        self.assertEqual(manager.measure("numbered", 4), ("numbered", 4))

        application.analyze_camera_frames.assert_called_once()
        application.process_camera_frames.assert_not_called()

    def test_background_builds_geometry_cache_before_digit_cache(self) -> None:
        frames = tuple(np.zeros((8, 12, 3), dtype=np.uint8) for _ in range(3))
        geometry_only = scene_analysis()
        with_digits = scene_analysis(numbered=object())
        application = Mock()
        application.buffered_camera_frames.side_effect = lambda **_kwargs: tuple(
            frame.copy() for frame in frames
        )
        application.probe_camera_frame.return_value = signature()
        application.analyze_camera_frames.return_value = geometry_only
        application.can_recognize_digits_from_analysis.return_value = True
        application.recognize_digits_for_analysis.return_value = with_digits
        application.result_from_camera_analysis.side_effect = (
            lambda analysis, mode, digit=None: (analysis, mode, digit)
        )
        manager = SilentRecognitionManager(application, stable_observations=1)

        manager.poll_once()

        application.analyze_camera_frames.assert_called_once()
        analyze_args, analyze_kwargs = application.analyze_camera_frames.call_args
        self.assertEqual(len(analyze_args[0]), len(frames))
        self.assertEqual(analyze_kwargs, {"recognize_digits": False})
        application.recognize_digits_for_analysis.assert_called_once_with(geometry_only)
        self.assertEqual(
            manager.cached_result("numbered", 3),
            (with_digits, "numbered", 3),
        )

    def test_numbered_button_completes_digit_cache_without_rebuilding_geometry(self) -> None:
        frames = tuple(np.zeros((8, 12, 3), dtype=np.uint8) for _ in range(3))
        geometry_only = scene_analysis()
        with_digits = scene_analysis(numbered=object())
        application = Mock()
        application.buffered_camera_frames.return_value = frames
        application.probe_camera_frame.return_value = signature()
        application.can_recognize_digits_from_analysis.return_value = True
        application.recognize_digits_for_analysis.return_value = with_digits
        application.result_from_camera_analysis.side_effect = (
            lambda analysis, mode, digit=None: (analysis, mode, digit)
        )
        manager = SilentRecognitionManager(application, stable_observations=1)
        manager._scene = signature()
        manager._stable_count = 1
        manager._trusted = True
        manager._analysis = geometry_only

        self.assertEqual(
            manager.measure("numbered", 4),
            (with_digits, "numbered", 4),
        )

        application.recognize_digits_for_analysis.assert_called_once_with(geometry_only)
        application.process_camera_frames.assert_not_called()
        application.analyze_camera_frames.assert_not_called()

    def test_button_revalidates_cached_scene_before_returning(self) -> None:
        frames = tuple(np.zeros((8, 12, 3), dtype=np.uint8) for _ in range(3))
        cached = object()
        direct = object()
        analysis = object()
        application = Mock()
        application.buffered_camera_frames.side_effect = lambda **_kwargs: tuple(
            frame.copy() for frame in frames
        )
        application.probe_camera_frame.return_value = signature()
        application.analyze_camera_frames.return_value = analysis
        application.result_from_camera_analysis.return_value = cached
        application.process_camera_frames.return_value = direct
        manager = SilentRecognitionManager(application, stable_observations=1)
        manager.poll_once()
        self.assertIs(manager.cached_result("normal"), cached)

        application.probe_camera_frame.return_value = CameraSceneSignature(
            signature().normalized_corners + 0.08,
            signature().content_mask,
        )
        self.assertIs(manager.measure("normal"), direct)
        application.analyze_camera_frames.assert_called_once()
        application.process_camera_frames.assert_called_once()

    def test_first_miss_blocks_cache_and_second_miss_clears_it(self) -> None:
        frames = tuple(np.zeros((8, 12, 3), dtype=np.uint8) for _ in range(3))
        expected = object()
        analysis = object()
        application = Mock()
        application.buffered_camera_frames.side_effect = lambda **_kwargs: tuple(
            frame.copy() for frame in frames
        )
        application.probe_camera_frame.return_value = signature()
        application.analyze_camera_frames.return_value = analysis
        application.result_from_camera_analysis.return_value = expected
        manager = SilentRecognitionManager(
            application,
            stable_observations=1,
            clear_after_misses=2,
        )
        manager.poll_once()
        self.assertIs(manager.cached_result("normal"), expected)

        application.probe_camera_frame.side_effect = ValueError("No A4")
        manager.poll_once()
        self.assertIsNone(manager.cached_result("normal"))
        manager.poll_once()

        application.probe_camera_frame.side_effect = None
        application.probe_camera_frame.return_value = signature()
        manager.poll_once()
        self.assertEqual(application.analyze_camera_frames.call_count, 2)

    def test_disabled_mode_uses_original_measurement_directly(self) -> None:
        frames = tuple(np.zeros((8, 12, 3), dtype=np.uint8) for _ in range(3))
        expected = object()
        application = Mock()
        application.buffered_camera_frames.return_value = frames
        application.process_camera_frames.return_value = expected
        manager = SilentRecognitionManager(application)
        manager.set_enabled(False)

        self.assertIs(manager.measure("numbered", 4), expected)
        application.process_camera_frames.assert_called_once_with(frames, "numbered", 4)

    def test_disabled_mode_does_not_wait_for_background_lock(self) -> None:
        frames = tuple(np.zeros((8, 12, 3), dtype=np.uint8) for _ in range(3))
        expected = object()
        application = Mock()
        application.buffered_camera_frames.return_value = frames
        application.process_camera_frames.return_value = expected
        manager = SilentRecognitionManager(application)
        manager.set_enabled(False)

        self.assertTrue(manager._process_lock.acquire(blocking=False))
        try:
            self.assertIs(manager.measure("normal"), expected)
        finally:
            manager._process_lock.release()

    def test_enabled_button_measurement_does_not_wait_for_background_lock(self) -> None:
        frames = tuple(np.zeros((8, 12, 3), dtype=np.uint8) for _ in range(3))
        expected = object()
        application = Mock()
        application.buffered_camera_frames.return_value = frames
        application.process_camera_frames.return_value = expected
        manager = SilentRecognitionManager(application)

        self.assertTrue(manager._process_lock.acquire(blocking=False))
        try:
            self.assertIs(manager.measure("numbered", 7), expected)
        finally:
            manager._process_lock.release()

        application.process_camera_frames.assert_called_once_with(frames, "numbered", 7)
        self.assertIsNone(manager.cached_result("normal"))

    def test_preempted_background_result_is_not_cached(self) -> None:
        frames = tuple(np.zeros((8, 12, 3), dtype=np.uint8) for _ in range(3))
        analysis = object()
        direct = object()
        application = Mock()
        application.buffered_camera_frames.side_effect = lambda **_kwargs: tuple(
            frame.copy() for frame in frames
        )
        application.probe_camera_frame.return_value = signature()
        application.process_camera_frames.return_value = direct
        manager = SilentRecognitionManager(application, stable_observations=1)

        def analyze_and_press_button(_frames, recognize_digits=True):
            self.assertIs(manager.measure("normal"), direct)
            return analysis

        application.analyze_camera_frames.side_effect = analyze_and_press_button
        application.result_from_camera_analysis.return_value = object()

        manager.poll_once()

        self.assertIsNone(manager.cached_result("normal"))
        application.process_camera_frames.assert_called_once()


if __name__ == "__main__":
    unittest.main()
