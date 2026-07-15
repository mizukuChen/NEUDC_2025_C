from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.camera_settings import (
    CameraDeviceLock,
    CameraSettings,
    apply_camera_controls,
    load_camera_settings,
    read_camera_controls,
    save_camera_settings,
)


class CameraSettingsTest(unittest.TestCase):
    def test_round_trip_and_missing_file_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.json"
            self.assertEqual(load_camera_settings(path), CameraSettings())
            expected = CameraSettings(False, 500, 12, -5, 38, 110, 7, 1)
            save_camera_settings(expected, path)
            self.assertEqual(load_camera_settings(path), expected)

    def test_out_of_range_value_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "gain"):
            CameraSettings(gain=64).validate()

    def test_shared_lock_rejects_second_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.lock"
            first = CameraDeviceLock(path)
            second = CameraDeviceLock(path)
            first.acquire()
            try:
                with self.assertRaisesRegex(RuntimeError, "另一个程序占用"):
                    second.acquire()
            finally:
                first.release()
            second.acquire()
            second.release()

    def test_v4l2_apply_and_readback(self) -> None:
        capture = Mock()
        settings = CameraSettings(False, 420, 8, -3, 36, 105, 5, 1)
        with tempfile.NamedTemporaryFile() as device:
            completed = subprocess.CompletedProcess(
                [],
                0,
                stdout=(
                    "brightness: -3\n"
                    "contrast: 36\n"
                    "auto_exposure: 3 (Aperture Priority Mode)\n"
                ),
                stderr="",
            )
            with patch("src.camera_settings.subprocess.run", return_value=completed) as run:
                apply_camera_controls(capture, settings, device.name)
                actual = read_camera_controls(device.name)
        self.assertEqual(
            actual,
            {"brightness": -3, "contrast": 36, "auto_exposure": 3},
        )
        set_command = run.call_args_list[0].args[0]
        self.assertIn("auto_exposure=1", set_command[-1])
        self.assertIn("exposure_time_absolute=420", set_command[-1])
        self.assertGreaterEqual(capture.set.call_count, 6)


if __name__ == "__main__":
    unittest.main()
