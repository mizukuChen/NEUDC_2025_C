"""两套重叠识别引擎共享的几何与候选验证函数。"""

from __future__ import annotations

from math import hypot

import cv2
import numpy as np


def edge_support(
    distance: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    tolerance: float,
) -> float:
    """返回候选边上落在真实边缘容差内的采样点比例。"""
    length = hypot(*(end - start))
    sample_count = max(20, round(length / 2.0))
    points = np.linspace(start, end, sample_count)
    xs = np.clip(np.rint(points[:, 0]).astype(int), 0, distance.shape[1] - 1)
    ys = np.clip(np.rint(points[:, 1]).astype(int), 0, distance.shape[0] - 1)
    return float(np.mean(distance[ys, xs] <= tolerance))


def outer_band_mean(
    values: np.ndarray,
    polygon: np.ndarray,
    inner_fraction: float = 0.68,
) -> float:
    """计算候选多边形外圈带内的平均值，忽略中央数字区域。"""
    integer = np.rint(polygon).astype(np.int32)
    x, y, width, height = cv2.boundingRect(integer)
    x0, y0 = max(0, x), max(0, y)
    x1 = min(values.shape[1], x + width)
    y1 = min(values.shape[0], y + height)
    if x1 <= x0 or y1 <= y0:
        return 0.0

    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    shifted = integer - np.array([x0, y0], dtype=np.int32)
    cv2.fillConvexPoly(mask, shifted, 255)
    center = shifted.astype(np.float32).mean(axis=0)
    inner = center + (shifted.astype(np.float32) - center) * inner_fraction
    cv2.fillConvexPoly(mask, np.rint(inner).astype(np.int32), 0)
    active = mask > 0
    if not np.any(active):
        return 0.0
    return float(np.mean(values[y0:y1, x0:x1][active]))


def polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    """计算两个凸多边形的交并比。"""
    first_float = first.astype(np.float32)
    second_float = second.astype(np.float32)
    area_first = abs(cv2.contourArea(first_float))
    area_second = abs(cv2.contourArea(second_float))
    intersection, _ = cv2.intersectConvexConvex(first_float, second_float)
    union = area_first + area_second - intersection
    return float(intersection / union) if union > 0 else 0.0


def square_angle_error(first_deg: float, second_deg: float) -> float:
    """返回正方形旋转角在 90° 周期下的最小差值。"""
    return abs((first_deg - second_deg + 45.0) % 90.0 - 45.0)
