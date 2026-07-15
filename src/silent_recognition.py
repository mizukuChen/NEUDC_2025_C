from __future__ import annotations

from collections.abc import Callable
import threading
from typing import Any

import cv2
import numpy as np

from src.application import CameraSceneSignature, MeasurementApplication


RequestKey = tuple[str, int | None]
StatusCallback = Callable[[str], None]


def scene_signatures_similar(
    first: CameraSceneSignature,
    second: CameraSceneSignature,
    corner_tolerance: float = 0.025,
    content_tolerance: float = 0.06,
) -> bool:
    """比较A4位置和纸内黑色图案，并容忍二值边缘约1像素的抖动。"""
    first_corners = np.asarray(first.normalized_corners, dtype=np.float32)
    second_corners = np.asarray(second.normalized_corners, dtype=np.float32)
    if first_corners.shape != (4, 2) or second_corners.shape != (4, 2):
        return False
    if float(np.max(np.abs(first_corners - second_corners))) > corner_tolerance:
        return False

    first_mask = np.asarray(first.content_mask, dtype=bool)
    second_mask = np.asarray(second.content_mask, dtype=bool)
    if first_mask.shape != second_mask.shape or first_mask.ndim != 2:
        return False
    kernel = np.ones((3, 3), dtype=np.uint8)
    first_u8 = first_mask.astype(np.uint8)
    second_u8 = second_mask.astype(np.uint8)
    first_dilated = cv2.dilate(first_u8, kernel).astype(bool)
    second_dilated = cv2.dilate(second_u8, kernel).astype(bool)
    unmatched = (first_mask & ~second_dilated) | (second_mask & ~first_dilated)
    foreground = first_mask | second_mask
    foreground_count = int(np.count_nonzero(foreground))
    if foreground_count == 0:
        return True
    return float(np.count_nonzero(unmatched)) / foreground_count <= content_tolerance


class SilentRecognitionManager:
    """在GUI空闲时验证稳定场景并提前调用现有相机识别流程。"""

    def __init__(
        self,
        application: MeasurementApplication,
        interval_s: float = 0.5,
        stable_observations: int = 3,
        clear_after_misses: int = 2,
        status_callback: StatusCallback | None = None,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive.")
        if stable_observations < 1:
            raise ValueError("stable_observations must be at least 1.")
        if clear_after_misses < 1:
            raise ValueError("clear_after_misses must be at least 1.")
        self.application = application
        self.interval_s = interval_s
        self.stable_observations = stable_observations
        self.clear_after_misses = clear_after_misses
        self._status_callback = status_callback
        self._condition = threading.Condition()
        self._process_lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._enabled = True
        self._desired: RequestKey = ("normal", None)
        self._scene: CameraSceneSignature | None = None
        self._stable_count = 0
        self._miss_count = 0
        self._trusted = False
        self._cache: dict[RequestKey, Any] = {}
        self._recognizing: RequestKey | None = None
        self._last_frame: np.ndarray | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="silent-recognition",
            daemon=True,
        )

    @property
    def enabled(self) -> bool:
        with self._condition:
            return self._enabled

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        with self._condition:
            self._status_callback = None
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def set_enabled(self, enabled: bool) -> None:
        with self._condition:
            self._enabled = bool(enabled)
            self._stable_count = 0
            self._miss_count = 0
            self._trusted = False
            if not self._enabled:
                self._scene = None
                self._cache.clear()
            self._condition.notify_all()
        self._publish("静默识别：正在寻找 A4" if enabled else "静默识别：已关闭")
        self._wake.set()

    def set_request(self, mode: str, digit: int | None = None) -> None:
        if mode not in {"normal", "numbered"}:
            raise ValueError(f"未知测量模式：{mode}")
        if mode == "numbered" and (digit is None or not 0 <= digit <= 9):
            return
        key = (mode, int(digit) if mode == "numbered" else None)
        with self._condition:
            changed = key != self._desired
            self._desired = key
            cached = self._trusted and key in self._cache
            self._condition.notify_all()
        if cached:
            self._publish("静默识别：缓存已就绪")
        elif changed:
            self._publish("静默识别：等待当前模式结果")
        self._wake.set()

    def cached_result(self, mode: str, digit: int | None = None) -> Any | None:
        key = (mode, int(digit) if mode == "numbered" and digit is not None else None)
        with self._condition:
            if not self._enabled or not self._trusted:
                return None
            return self._cache.get(key)

    def measure(self, mode: str, digit: int | None = None) -> Any:
        """优先返回可信缓存；没有缓存时串行调用原相机识别流程。"""
        self.set_request(mode, digit)
        key = (mode, int(digit) if mode == "numbered" and digit is not None else None)
        with self._process_lock:
            cached = self.cached_result(mode, digit)
            frames = self.application.buffered_camera_frames()
            if cached is not None:
                try:
                    current = self.application.probe_camera_frame(frames[-1])
                except Exception:
                    self._record_miss()
                else:
                    with self._condition:
                        still_valid = (
                            self._enabled
                            and self._trusted
                            and self._scene is not None
                            and scene_signatures_similar(self._scene, current)
                        )
                        if still_valid:
                            self._scene = current
                            self._miss_count = 0
                    if still_valid:
                        return cached
                    self._record_signature(current)
            result = self.application.process_camera_frames(frames, key[0], key[1])
            return result

    def run_exclusive(self, function: Callable[[], Any]) -> Any:
        """使文件识别不会与后台完整识别同时修改OpenCV全局线程设置。"""
        with self._process_lock:
            return function()

    def poll_once(self) -> None:
        """执行一次轻量场景检查；公开此入口便于确定性测试。"""
        with self._condition:
            if not self._enabled:
                return
        if not self._process_lock.acquire(blocking=False):
            return
        try:
            try:
                frames = self.application.buffered_camera_frames(timeout_s=1.0)
                with self._condition:
                    if frames[-1] is self._last_frame:
                        return
                    self._last_frame = frames[-1]
                signature = self.application.probe_camera_frame(frames[-1])
            except Exception:
                self._record_miss()
                return

            should_recognize, key = self._record_signature(signature)
            if not should_recognize:
                return
            self._publish("静默识别：正在后台识别")
            with self._condition:
                self._recognizing = key
            try:
                result = self.application.process_camera_frames(frames, key[0], key[1])
                latest = self.application.buffered_camera_frames(timeout_s=1.0)
                latest_signature = self.application.probe_camera_frame(latest[-1])
            except Exception:
                self._record_miss()
                return

            with self._condition:
                same_scene = (
                    self._enabled
                    and self._scene is not None
                    and scene_signatures_similar(self._scene, latest_signature)
                )
                if same_scene:
                    self._scene = latest_signature
                    self._cache[key] = result
                    self._trusted = True
                    self._miss_count = 0
                else:
                    self._clear_locked()
                self._recognizing = None
                self._condition.notify_all()
            self._publish(
                "静默识别：缓存已就绪" if same_scene else "静默识别：画面已变化"
            )
        finally:
            with self._condition:
                if self._recognizing is not None:
                    self._recognizing = None
                    self._condition.notify_all()
            self._process_lock.release()

    def _record_signature(self, signature: CameraSceneSignature) -> tuple[bool, RequestKey]:
        with self._condition:
            if self._scene is None or not scene_signatures_similar(self._scene, signature):
                self._scene = signature
                self._stable_count = 1
                self._miss_count = 0
                self._trusted = self._stable_count >= self.stable_observations
                self._cache.clear()
                status = "静默识别：A4 稳定中 1/{}".format(self.stable_observations)
            else:
                self._scene = signature
                self._stable_count = min(self.stable_observations, self._stable_count + 1)
                self._miss_count = 0
                self._trusted = self._stable_count >= self.stable_observations
                status = "静默识别：A4 稳定中 {}/{}".format(
                    self._stable_count,
                    self.stable_observations,
                )
            key = self._desired
            cached = key in self._cache
            should_recognize = self._trusted and not cached and self._recognizing is None
        self._publish("静默识别：缓存已就绪" if self._trusted and cached else status)
        return should_recognize, key

    def _record_miss(self) -> None:
        with self._condition:
            self._miss_count += 1
            self._stable_count = 0
            self._trusted = False
            if self._miss_count >= self.clear_after_misses:
                self._clear_locked()
                status = "静默识别：正在寻找 A4"
            else:
                status = "静默识别：A4 暂时丢失"
            self._recognizing = None
            self._condition.notify_all()
        self._publish(status)

    def _clear_locked(self) -> None:
        self._scene = None
        self._stable_count = 0
        self._miss_count = 0
        self._trusted = False
        self._cache.clear()

    def _publish(self, status: str) -> None:
        if self._status_callback is not None:
            self._status_callback(status)

    def _run(self) -> None:
        self._publish("静默识别：正在寻找 A4")
        while not self._stop.is_set():
            self.poll_once()
            self._wake.wait(self.interval_s)
            self._wake.clear()
