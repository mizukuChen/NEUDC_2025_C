from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from src.application import CameraSceneSignature
from src.silent_recognition import SilentRecognitionManager, scene_signatures_similar


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


if __name__ == "__main__":
    unittest.main()
