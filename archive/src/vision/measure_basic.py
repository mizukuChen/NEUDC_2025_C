from __future__ import annotations

"""已归档的旧版单图形测量实现。"""

"""
历史遗留/已归档的基础形状测量模块。

该模块用于早期版本的基础图形（圆、等边三角形、正方形）测量开发与测试。
目前主线算法已迁移至 `measure_shapes.py`，本文件仅保留用于历史版本兼容与复现。
"""

from dataclasses import dataclass
from math import pi, sqrt

import cv2
import numpy as np

from src.vision.rectify import PX_PER_MM

# A4 纸张黑框宽度（单位：mm），电赛标准黑边固定宽 20mm
BORDER_WIDTH_MM = 20.0


@dataclass(frozen=True)
class BasicMeasurement:
    """
    单图形基础测量结果的数据结构。

    属性:
        shape_type: 识别到的形状类别 ("circle", "triangle", "square" 或 "unknown")。
        x_mm: 特征物理尺寸（直径或边长，单位：mm）。
        area_px: 图形在图像中的像素面积。
        contour: 图形在 A4 全图像素坐标系下的完整轮廓坐标。
    """
    shape_type: str
    x_mm: float
    area_px: float
    contour: np.ndarray


def _content_roi(rectified: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """
    裁剪出 A4 纸内侧的有效测量区域（ROI），剔除 20mm 黑框和额外的 5mm 保护边缘。

    参数:
        rectified: 透视校正拉平后的标准 A4 图像。

    返回:
        roi: 裁剪后的子图像区域。
        (x0, y0): ROI 区域左上角在标准 A4 图像中的像素坐标，以便后续将局部轮廓还原到全局。
    """
    border_px = int(BORDER_WIDTH_MM * PX_PER_MM)
    margin_px = int(5 * PX_PER_MM)
    x0 = border_px + margin_px
    y0 = border_px + margin_px
    x1 = rectified.shape[1] - border_px - margin_px
    y1 = rectified.shape[0] - border_px - margin_px
    return rectified[y0:y1, x0:x1], (x0, y0)


def _shape_mask(image: np.ndarray) -> np.ndarray:
    """
    对输入 ROI 图像进行二值化处理，生成前景黑色几何图形的掩膜。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 大津法（OTSU）自动阈值反二值化，使黑色图形成为白色前景（255），背景为黑色（0）
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 形态学开运算去除噪点，闭运算平滑边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _classify_shape(contour: np.ndarray) -> str:
    """
    基于轮廓的多边形逼近顶点数和圆度进行简单的形状分类。

    返回:
        "triangle" / "square" / "circle" / "unknown"
    """
    perimeter = cv2.arcLength(contour, True)
    # 用周长的 3.5% 作为逼近精度
    approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
    vertices = len(approx)

    area = cv2.contourArea(contour)
    # 圆度指标 = 4 * pi * Area / Perimeter^2，越接近 1 说明越像正圆
    circularity = 4.0 * pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0

    if vertices == 3:
        return "triangle"
    if vertices == 4:
        return "square"
    if circularity > 0.78:
        return "circle"
    return "unknown"


def _size_from_area(shape_type: str, area_px: float) -> float:
    """
    根据形状类别和像素面积反推其真实的特征物理尺寸（单位：mm）。

    公式:
        - 圆: 直径 = 2 * sqrt(面积_mm2 / pi)
        - 三角形: 边长 = sqrt(4 * 面积_mm2 / sqrt(3))
        - 正方形: 边长 = sqrt(面积_mm2)
    """
    area_mm2 = area_px / (PX_PER_MM * PX_PER_MM)
    if shape_type == "circle":
        return 2.0 * sqrt(area_mm2 / pi)
    if shape_type == "triangle":
        return sqrt(4.0 * area_mm2 / sqrt(3.0))
    if shape_type == "square":
        return sqrt(area_mm2)
    return sqrt(area_mm2)


def measure_basic_shape(rectified: np.ndarray) -> BasicMeasurement:
    """
    对标准拉平的 A4 图像中的单个核心图形进行物理尺寸测量。

    参数:
        rectified: 已经过透视矫正拉平的 A4 图像。

    返回:
        BasicMeasurement: 图形特征尺寸和轮廓等信息。
    """
    # 1. 裁剪有效测量区域
    roi, (x0, y0) = _content_roi(rectified)
    # 2. 提取二值化前景图形
    mask = _shape_mask(roi)
    # 3. 寻找图形轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No inner shape contour found.")

    # 4. 过滤小面积噪声轮廓（必须大于 ROI 面积的 1%）
    roi_area = roi.shape[0] * roi.shape[1]
    valid = [c for c in contours if cv2.contourArea(c) > roi_area * 0.01]
    if not valid:
        raise ValueError("No valid inner shape contour found.")

    # 5. 取面积最大的主要图形进行分类与计算
    contour = max(valid, key=cv2.contourArea)
    area_px = cv2.contourArea(contour)
    shape_type = _classify_shape(contour)
    x_mm = _size_from_area(shape_type, area_px)

    # 6. 将裁剪区域局部轮廓坐标还原到 A4 全图像素坐标系
    contour_full = contour.copy()
    contour_full[:, 0, 0] += x0
    contour_full[:, 0, 1] += y0

    return BasicMeasurement(
        shape_type=shape_type,
        x_mm=x_mm,
        area_px=area_px,
        contour=contour_full,
    )


def draw_basic_measurement_debug(rectified: np.ndarray, measurement: BasicMeasurement) -> np.ndarray:
    """
    在 A4 图像上绘制测量的图形轮廓和检测结果文字，用于调试可视化。
    """
    debug = rectified.copy()
    # 用红色线宽为 4 的线条勾勒轮廓
    cv2.drawContours(debug, [measurement.contour], -1, (0, 0, 255), 4)
    label = f"{measurement.shape_type}: {measurement.x_mm:.1f} mm"
    # 在左上角绘制标注文本
    cv2.putText(
        debug,
        label,
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (0, 0, 255),
        4,
        cv2.LINE_AA,
    )
    return debug
