from __future__ import annotations

import unittest

from src.camera_auto_tune import auto_tune_candidates
from src.camera_settings import CameraSettings


class CameraAutoTuneTest(unittest.TestCase):
    def test_candidates_prioritize_low_sharpness_and_keep_roi(self) -> None:
        base = CameraSettings(
            auto_exposure=True,
            contrast=34,
            sharpness=6,
            roi_left=0.25,
            roi_top=0.1,
            roi_right=0.75,
            roi_bottom=0.9,
        )

        candidates = auto_tune_candidates(base)

        self.assertEqual(candidates[0].sharpness, 0)
        self.assertIn(6, [item.sharpness for item in candidates])
        self.assertTrue(all(item.auto_exposure for item in candidates))
        self.assertTrue(all(item.contrast == base.contrast for item in candidates))
        self.assertTrue(all(item.roi_left == base.roi_left for item in candidates))
        self.assertTrue(all(item.roi_bottom == base.roi_bottom for item in candidates))


if __name__ == "__main__":
    unittest.main()
