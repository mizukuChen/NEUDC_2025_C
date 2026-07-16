from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.display.camera_diagnostics import CachedPreviewFrame, DiagnosticProcessor
from src.hardware.camera_settings import CameraSettings


class CameraDiagnosticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = np.zeros((80, 100, 3), dtype=np.uint8)
        self.undistorted = np.full_like(self.raw, 220)
        self.settings = CameraSettings(
            roi_left=0.1,
            roi_top=0.2,
            roi_right=0.9,
            roi_bottom=0.8,
        )
        self.processor = DiagnosticProcessor(Path("fisheye.json"), Path("distance.json"))
        self.processor._preview_cache = CachedPreviewFrame(
            frame_id=7,
            size=(100, 80),
            undistorted=self.undistorted,
            dark_ratio=1.5,
            bright_ratio=2.5,
        )

    def test_preview_full_only_adds_roi_to_undistorted_image(self) -> None:
        preview = self.processor.preview_full(self.raw, self.settings, frame_id=7)
        self.assertEqual(preview.shape, self.raw.shape)
        self.assertGreater(int(np.count_nonzero(preview != self.undistorted)), 0)
        np.testing.assert_array_equal(self.processor._preview_cache.undistorted, self.undistorted)

    def test_failed_a4_location_returns_placeholder_diagnostics(self) -> None:
        with patch(
            "src.display.camera_diagnostics.find_target_plane",
            side_effect=ValueError("未找到A4"),
        ):
            diagnostics = self.processor.process(self.raw, self.settings, frame_id=7)
        self.assertIn("A4定位失败", diagnostics.message)
        self.assertIn("过暗 1.50%", diagnostics.message)
        self.assertEqual(diagnostics.rectified.shape, (594, 420, 3))
        self.assertEqual(diagnostics.probability.shape, (594, 420, 3))


if __name__ == "__main__":
    unittest.main()
