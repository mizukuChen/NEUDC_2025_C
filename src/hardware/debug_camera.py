"""相机调试软件使用的可动态应用参数的采集线程。"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from .camera_settings import (
    CameraSettings,
    apply_camera_controls,
    device_path,
    read_camera_controls,
)
from .camera_stream import configure_camera


class DebugCameraWorker:
    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
        settings: CameraSettings,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.settings = settings
        self._capture = None
        self._latest: np.ndarray | None = None
        self._error: str | None = None
        self._frame_id = 0
        self._mutex = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="camera-debug-capture",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def latest(self) -> tuple[np.ndarray | None, str | None, int]:
        with self._mutex:
            return (
                None if self._latest is None else self._latest.copy(),
                self._error,
                self._frame_id,
            )

    def apply(self, settings: CameraSettings) -> dict[str, int]:
        settings.validate()
        with self._mutex:
            if self._capture is None:
                raise RuntimeError(self._error or "找不到camera")
            apply_camera_controls(
                self._capture,
                settings,
                device_path(self.camera_index),
            )
            self.settings = settings
            self._latest = None
            self._error = None
        return read_camera_controls(device_path(self.camera_index))

    def _run(self) -> None:
        capture = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            with self._mutex:
                self._error = "找不到camera"
            return
        try:
            configure_camera(
                capture,
                self.width,
                self.height,
                self.settings,
                device_path(self.camera_index),
            )
            with self._mutex:
                self._capture = capture
            while not self._stop.is_set():
                with self._mutex:
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        self._error = "找不到camera"
                        break
                    self._frame_id += 1
                    self._latest = frame
                    self._error = None
        except Exception as exc:
            with self._mutex:
                self._error = str(exc)
        finally:
            with self._mutex:
                self._capture = None
            capture.release()
