"""V4L2 摄像头配置、单次采集与常驻帧缓冲。"""

from __future__ import annotations

from collections import deque
import threading
import time

import cv2
import numpy as np

from .camera_settings import (
    CameraSettings,
    apply_camera_controls,
    device_path,
    load_camera_settings,
)


def configure_camera(
    capture,
    width: int,
    height: int,
    settings: CameraSettings,
    camera_device: str | None = None,
) -> None:
    """应用树莓派测量使用的 YUYV、分辨率、帧率与现场参数。"""
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, 5)
    actual_device = (
        camera_device
        if type(capture).__module__.split(".", 1)[0] == "cv2"
        else None
    )
    apply_camera_controls(capture, settings, actual_device)


def capture_frame(
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
) -> np.ndarray:
    """打开摄像头并获取一张预热后的图像。"""
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("找不到camera")
    try:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        for _ in range(5):
            capture.grab()
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError("找不到camera")
    return frame


def capture_frames(
    camera_index: int = 0,
    width: int = 1920,
    height: int = 1080,
    frame_count: int = 3,
    warmup_frames: int = 20,
    settings: CameraSettings | None = None,
) -> tuple[np.ndarray, ...]:
    """按树莓派测量参数获取用于多帧融合的连续稳定帧。"""
    if frame_count < 1:
        raise ValueError("frame_count must be at least 1.")
    if warmup_frames < 0:
        raise ValueError("warmup_frames must not be negative.")
    capture = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("找不到camera")
    frames: list[np.ndarray] = []
    try:
        configure_camera(
            capture,
            width,
            height,
            settings or load_camera_settings(),
            device_path(camera_index),
        )
        for _ in range(warmup_frames):
            if not capture.grab():
                raise RuntimeError("找不到camera")
        for _ in range(frame_count):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("找不到camera")
            frames.append(frame)
    finally:
        capture.release()
    return tuple(frames)


class CameraFrameBuffer:
    """常驻 V4L2 相机线程，持续保存最新的稳定帧。"""

    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
        frame_count: int = 3,
        warmup_frames: int = 20,
        settings: CameraSettings | None = None,
    ) -> None:
        if frame_count < 1:
            raise ValueError("frame_count must be at least 1.")
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self.warmup_frames = warmup_frames
        self.settings = settings or CameraSettings()
        self._frames: deque[np.ndarray] = deque(maxlen=frame_count)
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._error: str | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="camera-frame-buffer",
            daemon=True,
        )

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def frames(self, timeout_s: float = 3.0) -> tuple[np.ndarray, ...]:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while len(self._frames) < self.frame_count and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            if len(self._frames) < self.frame_count:
                raise RuntimeError(self._error or "找不到camera")
            return tuple(self._frames)

    def _set_error(self, message: str) -> None:
        with self._condition:
            self._frames.clear()
            self._error = message
            self._condition.notify_all()

    def _run(self) -> None:
        while not self._stop.is_set():
            capture = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
            if not capture.isOpened():
                capture.release()
                self._set_error("找不到camera")
                self._stop.wait(0.5)
                continue
            try:
                configure_camera(
                    capture,
                    self.width,
                    self.height,
                    self.settings,
                    device_path(self.camera_index),
                )
                healthy = True
                for _ in range(self.warmup_frames):
                    if self._stop.is_set():
                        healthy = False
                        break
                    if not capture.grab():
                        self._set_error("找不到camera")
                        healthy = False
                        break
                while healthy and not self._stop.is_set():
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        self._set_error("找不到camera")
                        break
                    with self._condition:
                        self._frames.append(frame)
                        self._error = None
                        self._condition.notify_all()
            except Exception as exc:
                self._set_error(str(exc))
            finally:
                capture.release()
            if not self._stop.is_set():
                self._stop.wait(0.5)
