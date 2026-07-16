"""生成相机环境调试界面所需的诊断图和状态文本。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.application import MeasurementApplication
from src.hardware.camera_settings import CameraSettings
from src.vision.lowres_preprocess import estimate_black_probability
from src.vision.rectify import rectify_a4
from src.vision.target_plane import find_target_plane

from .measurement import draw_a4_location


@dataclass(frozen=True)
class CameraDiagnostics:
    full: np.ndarray
    located: np.ndarray
    rectified: np.ndarray
    probability: np.ndarray
    message: str


@dataclass(frozen=True)
class CachedPreviewFrame:
    frame_id: int
    size: tuple[int, int]
    undistorted: np.ndarray
    dark_ratio: float
    bright_ratio: float


class DiagnosticProcessor:
    """缓存去畸变结果并生成四幅现场诊断图。"""

    def __init__(self, fisheye_path: Path, distance_path: Path) -> None:
        self.application = MeasurementApplication(fisheye_path, distance_path)
        self._fisheye = None
        self._map_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._preview_cache: CachedPreviewFrame | None = None

    def _undistortion_maps(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        key = (width, height)
        cached = self._map_cache.get(key)
        if cached is not None:
            return cached
        if self._fisheye is None:
            self._fisheye, _ = self.application._load()
        fixed, interpolation, _, _, _ = self.application._camera_undistortion_maps(
            self._fisheye,
            key,
        )
        self._map_cache[key] = (fixed, interpolation)
        return fixed, interpolation

    def _preprocess(self, raw: np.ndarray, frame_id: int | None) -> CachedPreviewFrame:
        height, width = raw.shape[:2]
        size = (width, height)
        if (
            frame_id is not None
            and self._preview_cache is not None
            and self._preview_cache.frame_id == frame_id
            and self._preview_cache.size == size
        ):
            return self._preview_cache
        fixed, interpolation = self._undistortion_maps(width, height)
        undistorted = cv2.remap(
            raw,
            fixed,
            interpolation,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        preview = CachedPreviewFrame(
            frame_id=-1 if frame_id is None else frame_id,
            size=size,
            undistorted=undistorted,
            dark_ratio=float(np.mean(gray <= 5)) * 100.0,
            bright_ratio=float(np.mean(gray >= 250)) * 100.0,
        )
        if frame_id is not None:
            self._preview_cache = preview
        return preview

    def process(
        self,
        raw: np.ndarray,
        settings: CameraSettings,
        frame_id: int | None = None,
    ) -> CameraDiagnostics:
        preview = self._preprocess(raw, frame_id)
        undistorted = preview.undistorted
        height, width = raw.shape[:2]
        left, top, right, bottom = settings.roi_pixels(width, height)
        full = self.preview_full(raw, settings, frame_id)
        roi = undistorted[top:bottom, left:right]
        try:
            plane = find_target_plane(roi)
            full_corners = plane.corners.astype(np.float64)
            full_corners += np.asarray([left, top], dtype=np.float64)
            located = draw_a4_location(undistorted, full_corners)
            cv2.rectangle(
                located,
                (left, top),
                (right - 1, bottom - 1),
                (255, 180, 0),
                3,
                cv2.LINE_AA,
            )
            rectified = rectify_a4(roi, plane.corners, 420, 594)
            probability, black, white = estimate_black_probability(rectified)
            probability_u8 = np.rint(probability * 255.0).astype(np.uint8)
            probability_bgr = cv2.cvtColor(probability_u8, cv2.COLOR_GRAY2BGR)
            message = (
                f"A4定位成功｜黑框灰度 {black:.1f}｜白纸灰度 {white:.1f}｜"
                f"黑白差 {white - black:.1f}｜ROI {right-left}×{bottom-top}px｜"
                f"过暗 {preview.dark_ratio:.2f}%｜过曝 {preview.bright_ratio:.2f}%"
            )
        except Exception as exc:
            located = undistorted.copy()
            cv2.rectangle(
                located,
                (left, top),
                (right - 1, bottom - 1),
                (255, 180, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                located,
                "A4 NOT FOUND",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (0, 0, 255),
                4,
                cv2.LINE_AA,
            )
            rectified = np.full((594, 420, 3), 235, dtype=np.uint8)
            probability_bgr = np.full_like(rectified, 235)
            message = (
                f"A4定位失败：{exc}｜过暗 {preview.dark_ratio:.2f}%｜"
                f"过曝 {preview.bright_ratio:.2f}%"
            )
        return CameraDiagnostics(full, located, rectified, probability_bgr, message)

    def preview_full(
        self,
        raw: np.ndarray,
        settings: CameraSettings,
        frame_id: int | None = None,
    ) -> np.ndarray:
        preview = self._preprocess(raw, frame_id)
        undistorted = preview.undistorted
        height, width = raw.shape[:2]
        left, top, right, bottom = settings.roi_pixels(width, height)
        full = undistorted.copy()
        cv2.rectangle(
            full,
            (left, top),
            (right - 1, bottom - 1),
            (255, 180, 0),
            3,
            cv2.LINE_AA,
        )
        return full
