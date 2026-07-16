from __future__ import annotations

from collections.abc import Callable
import threading
from typing import Any

import cv2
import numpy as np

from src.application import (
    CameraSceneAnalysis,
    CameraSceneSignature,
    MeasurementApplication,
)


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
        self._scene: CameraSceneSignature | None = None
        self._stable_count = 0
        self._miss_count = 0
        self._trusted = False
        self._analysis: CameraSceneAnalysis | None = None
        self._recognizing = False
        self._foreground_generation = 0
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
                self._analysis = None
            self._condition.notify_all()
        self._publish("静默识别：正在寻找 A4" if enabled else "静默识别：已关闭")
        self._wake.set()

    def set_request(self, mode: str, digit: int | None = None) -> None:
        """仅校验界面输入；场景分析不再依赖模式或预选数字。"""
        if mode not in {"normal", "numbered"}:
            raise ValueError(f"未知测量模式：{mode}")
        if mode == "numbered" and (digit is None or not 0 <= digit <= 9):
            return
        with self._condition:
            cached = self._trusted and self._analysis is not None
        if cached:
            self._publish(self._cache_status(self._analysis))

    def cached_result(self, mode: str, digit: int | None = None) -> Any | None:
        """从同一份可信场景分析即时派生当前界面请求。"""
        with self._condition:
            if not self._enabled or not self._trusted or self._analysis is None:
                return None
            analysis = self._analysis
        return self.application.result_from_camera_analysis(analysis, mode, digit)

    def measure(self, mode: str, digit: int | None = None) -> Any:
        """优先返回可信缓存；后台正忙时前台按钮不等待，直接识别当前帧。"""
        self.set_request(mode, digit)
        with self._condition:
            enabled = self._enabled
        if not enabled:
            frames = self.application.buffered_camera_frames()
            return self.application.process_camera_frames(frames, mode, digit)
        acquired = self._process_lock.acquire(blocking=False)
        if not acquired:
            with self._condition:
                self._foreground_generation += 1
                self._trusted = False
                self._analysis = None
                self._condition.notify_all()
            self._publish("静默识别：按钮测量优先，正在跳过后台缓存")
            frames = self.application.buffered_camera_frames()
            return self.application.process_camera_frames(frames, mode, digit)
        try:
            with self._condition:
                analysis = (
                    self._analysis
                    if self._enabled and self._trusted
                    else None
                )
            frames = self.application.buffered_camera_frames()
            if analysis is not None:
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
                        if mode == "numbered" and isinstance(analysis, CameraSceneAnalysis):
                            analysis = self._ensure_digit_cache(analysis)
                        return self.application.result_from_camera_analysis(
                            analysis,
                            mode,
                            digit,
                        )
                    self._record_signature(current)
            result = self.application.process_camera_frames(frames, mode, digit)
            return result
        finally:
            self._process_lock.release()

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

            should_recognize = self._record_signature(signature)
            if not should_recognize:
                return
            self._publish("静默识别：正在生成几何缓存")
            with self._condition:
                self._recognizing = True
                foreground_generation = self._foreground_generation
            try:
                analysis = self.application.analyze_camera_frames(
                    frames,
                    recognize_digits=False,
                )
                latest = self.application.buffered_camera_frames(timeout_s=1.0)
                latest_signature = self.application.probe_camera_frame(latest[-1])
            except Exception:
                self._record_miss()
                return

            with self._condition:
                preempted = foreground_generation != self._foreground_generation
                same_scene = (
                    self._enabled
                    and not preempted
                    and self._scene is not None
                    and scene_signatures_similar(self._scene, latest_signature)
                )
                if same_scene:
                    self._scene = latest_signature
                    self._analysis = analysis
                    self._trusted = True
                    self._miss_count = 0
                else:
                    self._clear_locked()
                self._recognizing = False
                self._condition.notify_all()
            self._publish(
                "静默识别：几何缓存已就绪"
                if same_scene
                else (
                    "静默识别：已让路给按钮测量"
                    if preempted
                    else "静默识别：画面已变化"
                )
            )
            if same_scene and isinstance(analysis, CameraSceneAnalysis):
                self._fill_digit_cache_if_stable(analysis, foreground_generation)
        finally:
            with self._condition:
                if self._recognizing:
                    self._recognizing = False
                    self._condition.notify_all()
            self._process_lock.release()

    def _record_signature(self, signature: CameraSceneSignature) -> bool:
        with self._condition:
            if self._scene is None or not scene_signatures_similar(self._scene, signature):
                self._scene = signature
                self._stable_count = 1
                self._miss_count = 0
                self._trusted = self._stable_count >= self.stable_observations
                self._analysis = None
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
            cached = self._analysis is not None
            should_recognize = self._trusted and not cached and not self._recognizing
            analysis = self._analysis
        self._publish(self._cache_status(analysis) if self._trusted and cached else status)
        return should_recognize

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
            self._recognizing = False
            self._condition.notify_all()
        self._publish(status)

    def _clear_locked(self) -> None:
        self._scene = None
        self._stable_count = 0
        self._miss_count = 0
        self._trusted = False
        self._analysis = None

    def _cache_status(self, analysis: CameraSceneAnalysis | None) -> str:
        if isinstance(analysis, CameraSceneAnalysis) and analysis.numbered_squares is not None:
            return "静默识别：数字缓存已就绪"
        return "静默识别：几何缓存已就绪"

    def _ensure_digit_cache(self, analysis: CameraSceneAnalysis) -> CameraSceneAnalysis:
        if analysis.numbered_squares is not None:
            return analysis
        if not self.application.can_recognize_digits_from_analysis(analysis):
            return analysis
        self._publish("静默识别：正在补充数字缓存")
        updated = self.application.recognize_digits_for_analysis(analysis)
        with self._condition:
            if self._analysis is analysis and self._enabled and self._trusted:
                self._analysis = updated
        self._publish(self._cache_status(updated))
        return updated

    def _fill_digit_cache_if_stable(
        self,
        analysis: CameraSceneAnalysis,
        foreground_generation: int,
    ) -> None:
        if (
            analysis.numbered_squares is not None
            or not self.application.can_recognize_digits_from_analysis(analysis)
        ):
            return
        self._publish("静默识别：正在补充数字缓存")
        with self._condition:
            self._recognizing = True
        try:
            updated = self.application.recognize_digits_for_analysis(analysis)
            latest = self.application.buffered_camera_frames(timeout_s=1.0)
            latest_signature = self.application.probe_camera_frame(latest[-1])
        except Exception:
            with self._condition:
                self._recognizing = False
                self._condition.notify_all()
            self._publish("静默识别：数字缓存暂未就绪")
            return

        with self._condition:
            preempted = foreground_generation != self._foreground_generation
            same_scene = (
                self._enabled
                and not preempted
                and self._trusted
                and self._scene is not None
                and self._analysis is analysis
                and scene_signatures_similar(self._scene, latest_signature)
            )
            if same_scene:
                self._scene = latest_signature
                self._analysis = updated
                self._miss_count = 0
            self._recognizing = False
            self._condition.notify_all()
        self._publish(
            "静默识别：数字缓存已就绪"
            if same_scene
            else (
                "静默识别：已让路给按钮测量"
                if preempted
                else "静默识别：画面已变化，保留几何缓存"
            )
        )

    def _publish(self, status: str) -> None:
        if self._status_callback is not None:
            self._status_callback(status)

    def _run(self) -> None:
        self._publish("静默识别：正在寻找 A4")
        while not self._stop.is_set():
            self.poll_once()
            self._wake.wait(self.interval_s)
            self._wake.clear()
