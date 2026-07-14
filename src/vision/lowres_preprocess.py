from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
BORDER_WIDTH_MM = 20.0
WORK_PIXELS_PER_MM = 2.0
WORK_WIDTH_PX = round(A4_WIDTH_MM * WORK_PIXELS_PER_MM)
WORK_HEIGHT_PX = round(A4_HEIGHT_MM * WORK_PIXELS_PER_MM)


@dataclass(frozen=True)
class LowResolutionA4:
    """低分辨率A4工作图及由靶纸自身黑白区域标定出的黑色概率。"""

    image: np.ndarray
    black_probability: np.ndarray
    black_level: float
    white_level: float
    pixels_per_mm: float


def fuse_rectified_frames(frames: tuple[np.ndarray, ...] | list[np.ndarray]) -> np.ndarray:
    """对已经对齐到同一A4坐标系的帧取逐像素中位数。"""
    if not frames:
        raise ValueError("At least one rectified A4 frame is required.")
    arrays = [np.asarray(frame) for frame in frames]
    shape = arrays[0].shape
    if arrays[0].size == 0 or any(frame.shape != shape for frame in arrays):
        raise ValueError("All rectified A4 frames must be non-empty and have the same shape.")
    return np.median(np.stack(arrays), axis=0).astype(np.uint8)


def resize_working_a4(image: np.ndarray) -> np.ndarray:
    """把已校正A4统一到2 px/mm工作网格；不声称增加真实分辨率。"""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")
    if image.shape[:2] == (WORK_HEIGHT_PX, WORK_WIDTH_PX):
        return image.copy()
    interpolation = (
        cv2.INTER_AREA
        if image.shape[1] > WORK_WIDTH_PX or image.shape[0] > WORK_HEIGHT_PX
        else cv2.INTER_LINEAR
    )
    return cv2.resize(image, (WORK_WIDTH_PX, WORK_HEIGHT_PX), interpolation=interpolation)


def estimate_black_probability(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    """用20 mm黑框和内部白纸的稳健分位数把灰度转换为0～1黑色概率。"""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = np.asarray(gray, dtype=np.uint8)
    height, width = gray.shape
    scale_x = width / A4_WIDTH_MM
    scale_y = height / A4_HEIGHT_MM
    border_x = max(1, round(BORDER_WIDTH_MM * scale_x))
    border_y = max(1, round(BORDER_WIDTH_MM * scale_y))
    inner_x = max(border_x + 1, round((BORDER_WIDTH_MM + 5.0) * scale_x))
    inner_y = max(border_y + 1, round((BORDER_WIDTH_MM + 5.0) * scale_y))
    if width <= 2 * inner_x or height <= 2 * inner_y:
        raise ValueError("A4 work image is too small for black/white reference regions.")

    border = np.zeros(gray.shape, dtype=bool)
    border[:border_y, :] = True
    border[-border_y:, :] = True
    border[:, :border_x] = True
    border[:, -border_x:] = True
    inner = gray[inner_y : height - inner_y, inner_x : width - inner_x]
    black_level = float(np.percentile(gray[border], 20.0))
    white_level = float(np.percentile(inner, 85.0))
    if white_level - black_level < 20.0:
        raise ValueError("A4 black/white contrast is too low.")

    probability = np.clip(
        (white_level - gray.astype(np.float32)) / (white_level - black_level),
        0.0,
        1.0,
    ).astype(np.float32)
    return probability, black_level, white_level


def prepare_low_resolution_a4(
    frames: tuple[np.ndarray, ...] | list[np.ndarray],
) -> LowResolutionA4:
    """统一工作尺寸、融合多帧并生成黑色概率图。"""
    working = [resize_working_a4(frame) for frame in frames]
    fused = fuse_rectified_frames(working)
    probability, black_level, white_level = estimate_black_probability(fused)
    return LowResolutionA4(
        image=fused,
        black_probability=probability,
        black_level=black_level,
        white_level=white_level,
        pixels_per_mm=WORK_PIXELS_PER_MM,
    )
