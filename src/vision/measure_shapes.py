from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import cv2
import numpy as np

# A4 纸的标准物理尺寸（毫米）
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0

# 纸面黑边框的宽度（单位：mm），题目中靶纸黑色边框固定宽 20mm
BORDER_WIDTH_MM = 20.0

# 内部图形的保护页边距（单位：mm），避免二值化边界效应
CONTENT_MARGIN_MM = 5.0


@dataclass(frozen=True)
class ShapeMeasurement:
    """
    单个几何图形的测量结果数据结构。

    属性:
        shape_type: 识别到的形状类别 ("circle", "triangle", "square")。
        x_mm: 测量的图形特征尺寸（单位：mm），圆为直径，三角形/正方形为边长。
        area_mm2: 形状占用的实际物理面积（平方毫米）。
        center_mm: 图形质心在 A4 坐标系下的物理位置 (x, y)，单位：mm。
        angle_deg: 旋转角度（主要针对正方形，正数顺时针，负数逆时针），单位：度。
        confidence: 分类置信度评分 (0.0 ~ 1.0)。
        contour: 图形在全图坐标系下的完整轮廓像素坐标。
    """
    shape_type: str
    x_mm: float
    area_mm2: float
    center_mm: tuple[float, float]
    angle_deg: float
    confidence: float
    contour: np.ndarray


@dataclass(frozen=True)
class ShapeMeasurementResult:
    """
    整张 A4 靶纸的整体图形测量结果。

    属性:
        shapes: 所有识别到的图形测量结果元组，已按空间位置从上到下、从左到右排序。
        layout: 图形排版布局："single"（单图形）或 "separate"（分离多图形）。
        pixels_per_mm: 当前图像拉直后的平均比例尺（像素/毫米）。
    """
    shapes: tuple[ShapeMeasurement, ...]
    layout: str
    pixels_per_mm: float


def _pixel_scale(image: np.ndarray) -> tuple[float, float]:
    """
    计算当前矫正后图像在横向 and 纵向上的比例尺（像素/毫米）。
    """
    height, width = image.shape[:2]
    return width / A4_WIDTH_MM, height / A4_HEIGHT_MM


def _content_roi(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int], tuple[float, float]]:
    """
    裁剪出 A4 纸内侧有几何图形的有效区域（ROI），滤除最外侧 20mm 黑框和 5mm 边距。

    返回:
        roi: 裁剪出的子图像区域。
        (x0, y0): ROI 左上角在 A4 图像中的像素偏移坐标。
        (scale_x, scale_y): 横纵像素比例尺（像素/毫米）。
    """
    scale_x, scale_y = _pixel_scale(image)
    inset_mm = BORDER_WIDTH_MM + CONTENT_MARGIN_MM  # 25mm 保护边距
    x0 = round(inset_mm * scale_x)
    y0 = round(inset_mm * scale_y)
    x1 = image.shape[1] - x0
    y1 = image.shape[0] - y0
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Image is too small for an A4 target ROI.")
    return image[y0:y1, x0:x1], (x0, y0), (scale_x, scale_y)


def _shape_mask(image: np.ndarray, pixels_per_mm: float) -> np.ndarray:
    """
    对 ROI 图像进行滤波、自适应阈值分割和开运算去噪，获得干净的黑色实心图形掩膜。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # 高斯模糊的卷积核大小随比例尺自适应调整（约 0.5 mm）
    blur_size = max(3, round(pixels_per_mm * 0.5))
    if blur_size % 2 == 0:
        blur_size += 1
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    # OTSU 全局二值化反转，将黑色图形转化为白色前景（255），背景为黑色（0）
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 进行形态学开运算（先腐蚀后膨胀），滤除微小的灰尘或纸张反光毛刺
    kernel_size = max(3, round(pixels_per_mm * 0.5))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)


def _polygon_features(contour: np.ndarray) -> tuple[int, float, float]:
    """
    计算轮廓的多边形特征参数。

    返回:
        approx_len: 经过多边形逼近后的顶点数量（3 代表三角形，4 代表四边形等）。
        circularity: 圆度指标。公式 = (4 * pi * 面积) / (周长 * 周长)。越接近 1.0 说明越接近正圆。
        perimeter: 轮廓的真实像素周长。
    """
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return 0, 0.0, 0.0
    # 用 2.5% 周长作为精度的多边形逼近
    approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
    area = cv2.contourArea(contour)
    circularity = 4.0 * pi * area / (perimeter * perimeter)
    return len(approx), circularity, perimeter


def _square_quality(contour: np.ndarray) -> float:
    """
    计算轮廓与正方形的匹配质量。

    指标:
        - 宽高比分数（aspect_score）：横纵宽高之比，接近 1.0 说明接近正方形。
        - 填充率分数（fill_score）：实际面积 / 最小外接矩形面积，接近 1.0 说明实心程度高、无缺失。
        - 质量 = 宽高比 * 填充率。
    """
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    if min(width, height) <= 0:
        return 0.0
    aspect_score = min(width, height) / max(width, height)
    box_area = width * height
    fill_score = min(1.0, cv2.contourArea(contour) / box_area) if box_area > 0 else 0.0
    return aspect_score * fill_score


def _classify(contour: np.ndarray) -> tuple[str, float]:
    """
    对候选图形轮廓进行分类，判定其是圆形、等边三角形还是正方形。

    返回:
        shape_type: 类别名 ("circle", "triangle", "square" 或 "unknown")。
        confidence: 该分类的置信度得分 (0.0 ~ 1.0)。
    """
    vertices, circularity, _ = _polygon_features(contour)
    # 顶点数为 3，判定为等边三角形
    if vertices == 3:
        # 等边三角形的理想圆度约为 0.604 (pi / (3 * sqrt(3)))，以此为基准算置信度
        return "triangle", max(0.0, min(1.0, circularity / 0.61))

    # 顶点数为 4，进一步校验其是否为良好的实心正方形
    if vertices == 4:
        quality = _square_quality(contour)
        if quality >= 0.72:
            return "square", quality

    # 如果圆度大于 0.78，则归类为圆形
    if circularity >= 0.78:
        return "circle", min(1.0, circularity)

    return "unknown", 0.0


def _size_from_area(shape_type: str, area_mm2: float) -> float:
    """
    根据识别的形状类别和实际物理面积，数学反推其几何特征尺寸 x。

    计算公式:
        - 圆形 (直径 d): d = 2 * sqrt(面积 / pi)
        - 等边三角形 (边长 a): a = sqrt(4 * 面积 / sqrt(3))
        - 正方形 (边长 a): a = sqrt(面积)
    """
    if shape_type == "circle":
        return 2.0 * sqrt(area_mm2 / pi)
    if shape_type == "triangle":
        return sqrt(4.0 * area_mm2 / sqrt(3.0))
    return sqrt(area_mm2)


def _measurement(
    contour: np.ndarray,
    offset: tuple[int, int],
    scale: tuple[float, float],
) -> ShapeMeasurement | None:
    """
    对单个轮廓进行特征提取、分类、物理尺寸反推、中心物理坐标和旋转角度的计算。
    """
    shape_type, confidence = _classify(contour)
    if shape_type == "unknown":
        return None

    scale_x, scale_y = scale
    # 计算物理面积（平方毫米）：像素面积 / (scale_x * scale_y)
    area_mm2 = cv2.contourArea(contour) / (scale_x * scale_y)

    # 计算图像矩以获取准确的几何质心 (Centroid)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    cx = moments["m10"] / moments["m00"] + offset[0]
    cy = moments["m01"] / moments["m00"] + offset[1]

    angle = 0.0
    # 如果是正方形，计算其旋转角度（单位：度）
    if shape_type == "square":
        rect = cv2.minAreaRect(contour)
        # 将角度限制规范化到 -45 到 45 度之间
        angle = (float(rect[2]) + 45.0) % 90.0 - 45.0

    # 还原轮廓坐标到 A4 标准图的全图像素坐标系中
    contour_full = contour.copy()
    contour_full[:, 0, 0] += offset[0]
    contour_full[:, 0, 1] += offset[1]

    return ShapeMeasurement(
        shape_type=shape_type,
        x_mm=_size_from_area(shape_type, area_mm2),
        area_mm2=area_mm2,
        center_mm=(cx / scale_x, cy / scale_y),
        angle_deg=angle,
        confidence=confidence,
        contour=contour_full,
    )


def measure_shapes(rectified_a4: np.ndarray) -> ShapeMeasurementResult:
    """
    对拉平的 A4 纸面图像进行多目标几何尺寸检测的外部核心接口。

    参数:
        rectified_a4: 2100x2970 像素的拉直 A4 图像。

    返回:
        ShapeMeasurementResult: 包含所有检测到的形状、排版模式和比例尺的结构体。
    """
    if rectified_a4 is None or rectified_a4.size == 0:
        raise ValueError("Input image is empty.")

    # 1. 提取除去黑边框的图形 ROI 区域
    roi, offset, scale = _content_roi(rectified_a4)
    pixels_per_mm = (scale[0] + scale[1]) / 2.0

    # 2. 生成前景二值掩膜
    mask = _shape_mask(roi, pixels_per_mm)

    # 3. 提取所有的外部轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 排除极小的杂色噪声轮廓（最小边长约 20mm x 20mm 的范围）
    min_area_mm2 = 20.0 * 20.0
    measurements: list[ShapeMeasurement] = []

    for contour in contours:
        area_mm2 = cv2.contourArea(contour) / (scale[0] * scale[1])
        if area_mm2 < min_area_mm2:
            continue
        item = _measurement(contour, offset, scale)
        if item is not None:
            measurements.append(item)

    # 4. 对检测到的多个图形按物理中心坐标进行排序（主 Y 轴自上而下，次 X 轴自左而右）
    measurements.sort(key=lambda item: (item.center_mm[1], item.center_mm[0]))
    if not measurements:
        raise ValueError("No valid inner shapes found.")

    # 确定靶纸布局是单图形（single）还是分离多图形（separate）
    layout = "single" if len(measurements) == 1 else "separate"
    return ShapeMeasurementResult(tuple(measurements), layout, pixels_per_mm)


def draw_shape_measurements(image: np.ndarray, result: ShapeMeasurementResult) -> np.ndarray:
    """
    在图像上绘制图形轮廓和测量标注信息（形状名称 + 真实物理尺寸），用于调试可视化。
    """
    debug = image.copy()
    # 预设不同形状的绘制颜色：圆形为蓝色，等边三角形为橙黄色，正方形为红色
    colors = {"circle": (255, 0, 0), "triangle": (0, 165, 255), "square": (0, 0, 255)}
    for index, shape in enumerate(result.shapes, start=1):
        color = colors[shape.shape_type]
        # 1. 绘制图形的实心轮廓边界线
        cv2.drawContours(debug, [shape.contour], -1, color, 4)
        # 2. 计算文本放置点（图形包围盒的左上角附近）
        point = tuple(shape.contour.reshape(-1, 2).min(axis=0).astype(int))
        label = f"{index} {shape.shape_type} {shape.x_mm:.1f} mm"
        cv2.putText(debug, label, point, cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    return debug
