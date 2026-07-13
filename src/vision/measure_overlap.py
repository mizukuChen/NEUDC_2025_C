from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import atan2, cos, degrees, hypot, pi, sin

import cv2
import numpy as np

# A4 纸的标准物理尺寸（毫米）
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0

# 边缘安全收缩边距（单位：mm），避免将外圈黑色边框误识别为正方形线条
INSET_MM = 25.0

# 允许识别的正方形物理边长范围限制（单位：mm），过滤异常组合
MIN_SIDE_MM = 50.0
MAX_SIDE_MM = 125.0


@dataclass(frozen=True)
class ReconstructedSquare:
    """
    重建出的重叠正方形测量结果。

    属性:
        corners: 4x2 的 NumPy 数组，保存该正方形的 4 个角点像素坐标。
        center_mm: 在 A4 坐标系下的几何中心物理坐标 (x, y)，单位：mm。
        side_mm: 重建出的正方形物理边长（单位：mm）。
        angle_deg: 正方形的旋转偏角（单位：度，范围 -45 ~ 45）。
        confidence: 重建置信度得分 (0.0 ~ 1.0)。
    """
    corners: np.ndarray
    center_mm: tuple[float, float]
    side_mm: float
    angle_deg: float
    confidence: float


@dataclass(frozen=True)
class OverlapMeasurementResult:
    """
    重叠正方形检测的最终结果。

    属性:
        squares: 包含所有重建出的正方形结果的元组。
        edge_image: Canny 边缘检测图，用于可视化调试。
    """
    squares: tuple[ReconstructedSquare, ...]
    edge_image: np.ndarray


@dataclass(frozen=True)
class _Line:
    """
    内部解析直线的极坐标表示参数。

    属性:
        rho: 直线到原点的垂直距离（像素）。
        theta: 直线法向量的夹角（弧度）。
        support: 支撑该直线的全部线段累加物理长度（像素），用作权重评分。
    """
    rho: float
    theta: float
    support: float


def _angle_distance(a: float, b: float) -> float:
    """
    计算两个角度（弧度）之间的最小夹角距离（考虑周期性）。
    """
    diff = abs(a - b) % pi
    return min(diff, pi - diff)


def _canonical_line(x1: float, y1: float, x2: float, y2: float) -> _Line | None:
    """
    将 HoughLinesP 提取的线段 (x1, y1) -> (x2, y2) 转换为法线式（Hough 空间）极坐标形式。
    法线方程为：x * cos(theta) + y * sin(theta) = rho
    """
    length = hypot(x2 - x1, y2 - y1)
    if length <= 0:
        return None
    direction = atan2(y2 - y1, x2 - x1)
    # 计算法线角度 theta（垂直于线段方向）
    theta = (direction + pi / 2.0) % pi
    rho = x1 * cos(theta) + y1 * sin(theta)
    return _Line(rho, theta, length)


def _merge_lines(lines: list[_Line], angle_tol: float, rho_tol: float) -> list[_Line]:
    """
    将空间位置极其接近、方向几乎平行的多条线段合并（融合）为一条全局直线。
    使用线段的长度（support）作为合并权重，使得长线段在融合中占主导地位。
    """
    merged: list[_Line] = []
    # 优先处理长线段
    for line in sorted(lines, key=lambda item: item.support, reverse=True):
        match = next(
            (
                index
                for index, current in enumerate(merged)
                if _angle_distance(line.theta, current.theta) <= angle_tol
                and abs(line.rho - current.rho) <= rho_tol
            ),
            None,
        )
        if match is None:
            merged.append(line)
            continue
        # 找到匹配直线，进行加权平均融合
        current = merged[match]
        total = current.support + line.support
        merged[match] = _Line(
            rho=(current.rho * current.support + line.rho * line.support) / total,
            theta=(current.theta * current.support + line.theta * line.support) / total,
            support=total,
        )
    return merged


def _intersection(first: _Line, second: _Line) -> np.ndarray | None:
    """
    求解两条直线的交点。
    利用极坐标方程联立二元一次方程组求解：
    cos(theta1)*x + sin(theta1)*y = rho1
    cos(theta2)*x + sin(theta2)*y = rho2
    """
    matrix = np.array(
        [[cos(first.theta), sin(first.theta)], [cos(second.theta), sin(second.theta)]],
        dtype=np.float64,
    )
    determinant = np.linalg.det(matrix)
    if abs(determinant) < 1e-6:
        return None # 两直线几乎平行，无交点
    return np.linalg.solve(matrix, np.array([first.rho, second.rho], dtype=np.float64))


def _edge_support(distance: np.ndarray, start: np.ndarray, end: np.ndarray, tolerance_px: float) -> float:
    """
    计算正方形某条边在线条距离变换图中的边缘支持率。

    原理:
        - 在边 start -> end 上均匀采样多个像素点。
        - 检查这些采样点在距离变换图（Distance Transform）中的值是否小于设定的容差（在边缘附近）。
        - 支持率 = 落在真实图像边缘附近的点比例。支持率越高，说明图里这条边越真实存在。
    """
    length = hypot(*(end - start))
    sample_count = max(20, round(length / 2.0))
    points = np.linspace(start, end, sample_count)
    xs = np.clip(np.rint(points[:, 0]).astype(int), 0, distance.shape[1] - 1)
    ys = np.clip(np.rint(points[:, 1]).astype(int), 0, distance.shape[0] - 1)
    return float(np.mean(distance[ys, xs] <= tolerance_px))


def _interior_support(foreground: np.ndarray, polygon: np.ndarray) -> float:
    """
    验证重建的多边形区域内部是否为“纯黑实心”的。

    原理:
        - 创建一个多边形局部的掩膜画布。
        - 用 `fillConvexPoly` 填充多边形区域。
        - 统计在前景色二值图中，该多边形覆盖范围内的黑色（前景）像素占比。
        - 占比接近 1.0 说明该多边形内全是实心黑色，符合靶纸黑色实心正方形的特性。
    """
    integer = np.rint(polygon).astype(np.int32)
    x, y, width, height = cv2.boundingRect(integer)
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(foreground.shape[1], x + width)
    y1 = min(foreground.shape[0], y + height)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    region = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    shifted = integer - np.array([x0, y0], dtype=np.int32)
    cv2.fillConvexPoly(region, shifted, 255)
    # 数字位于方块中央，会在黑色内部形成白色孔洞。只验证外圈黑色带，
    # 忽略中央 68% 区域，避免数字降低实心支持率。
    center = shifted.astype(np.float32).mean(axis=0)
    inner = center + (shifted.astype(np.float32) - center) * 0.68
    cv2.fillConvexPoly(region, np.rint(inner).astype(np.int32), 0)
    pixels = region > 0
    if not np.any(pixels):
        return 0.0
    return float(np.mean(foreground[y0:y1, x0:x1][pixels]))


def _polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    """
    计算两个凸多边形之间的交并比（IoU），用于非极大值抑制（Deduplication）。
    """
    area_first = abs(cv2.contourArea(first.astype(np.float32)))
    area_second = abs(cv2.contourArea(second.astype(np.float32)))
    # 使用 OpenCV 内置的凸多边形相交面积计算函数
    intersection, _ = cv2.intersectConvexConvex(first.astype(np.float32), second.astype(np.float32))
    union = area_first + area_second - intersection
    return float(intersection / union) if union > 0 else 0.0


def _parallel_pair_lines(first: _Line, second: _Line) -> tuple[_Line, _Line]:
    """
    将两条方向极其接近的直线，强行修正为方向完全一致、法向量平行的理想平行线对。
    融合法向量时按直线长度权重平均，使得拟合更抗噪。
    """
    normal_first = np.array([cos(first.theta), sin(first.theta)])
    normal_second = np.array([cos(second.theta), sin(second.theta)])
    rho_second = second.rho
    # 如果两个法向量方向相反，将其反转对齐
    if float(normal_first @ normal_second) < 0.0:
        normal_second = -normal_second
        rho_second = -rho_second
    normal = normal_first * first.support + normal_second * second.support
    normal /= np.linalg.norm(normal)
    theta = atan2(normal[1], normal[0])
    return _Line(first.rho, theta, first.support), _Line(rho_second, theta, second.support)


def _square_from_pairs(
    pair_a: tuple[_Line, _Line, float],
    pair_b: tuple[_Line, _Line, float],
) -> np.ndarray | None:
    """
    通过相交两组相互垂直的平行线对，求出四个角点，重建出一个正方形。
    """
    first_a, second_a = _parallel_pair_lines(pair_a[0], pair_a[1])
    first_b, second_b = _parallel_pair_lines(pair_b[0], pair_b[1])
    corners = []
    # 交叉联立求四条边的四个交点
    for line_a, line_b in (
        (first_a, first_b),
        (second_a, first_b),
        (second_a, second_b),
        (first_a, second_b),
    ):
        point = _intersection(line_a, line_b)
        if point is None:
            return None
        corners.append(point)
    return np.asarray(corners, dtype=np.float32)


def _same_square(first: ReconstructedSquare, second: ReconstructedSquare) -> bool:
    """
    判断两个重建出的正方形是否为同一个目标（判定是否重合）。
    """
    center_distance = hypot(
        first.center_mm[0] - second.center_mm[0],
        first.center_mm[1] - second.center_mm[1],
    )
    angle_error = abs((first.angle_deg - second.angle_deg + 45.0) % 90.0 - 45.0)
    # 若中心物理距离小于 3mm 且边长差小于 3mm 且偏角小于 1.5 度，或者 IoU 大于 0.82，则认为是同一正方形
    return (
        center_distance <= 3.0
        and abs(first.side_mm - second.side_mm) <= 3.0
        and angle_error <= 1.5
    ) or _polygon_iou(first.corners, second.corners) >= 0.82


def _deduplicate_squares(candidates: list[ReconstructedSquare]) -> list[ReconstructedSquare]:
    """
    非极大值抑制（NMS）。对重叠正方形候选进行去重，保留置信度最高的目标。
    """
    kept: list[ReconstructedSquare] = []
    # 按置信度降序排列
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        if any(_same_square(candidate, current) for current in kept):
            continue
        kept.append(candidate)
    return kept


def reconstruct_overlapping_squares(rectified_a4: np.ndarray) -> OverlapMeasurementResult:
    """
    发挥题核心算法：从重叠遮挡的正方形目标中重建并测量每个正方形的尺寸和姿态。

    参数:
        rectified_a4: 拉平成标准尺寸的 A4 纸面图像。

    返回:
        OverlapMeasurementResult: 重建的所有正方形列表和调试用边缘图像。
    """
    if rectified_a4 is None or rectified_a4.size == 0:
        raise ValueError("Input image is empty.")

    height, width = rectified_a4.shape[:2]
    scale_x = width / A4_WIDTH_MM
    scale_y = height / A4_HEIGHT_MM
    scale = (scale_x + scale_y) / 2.0

    # 裁剪内圈区域，避开四周 25mm 宽的外层 A4 黑色框架线干扰
    x0 = round(INSET_MM * scale_x)
    y0 = round(INSET_MM * scale_y)
    roi = rectified_a4[y0 : height - y0, x0 : width - x0]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi

    # 1. 边缘检测
    edges = cv2.Canny(gray, 40, 120, apertureSize=3)

    # 2. 概率霍夫变换提取断裂的线段
    detected = cv2.HoughLinesP(
        edges,
        1.0,
        np.pi / 720.0,
        threshold=max(20, round(4.0 * scale)),
        minLineLength=max(20, round(8.0 * scale)),
        maxLineGap=max(4, round(0.8 * scale)),
    )
    if detected is None:
        raise ValueError("No line segments found.")

    segments = np.asarray(detected).reshape(-1, 4)
    # 转换为极坐标法线式表示并剔除零长线段
    raw_lines = [
        line
        for segment in segments
        if (line := _canonical_line(*(float(value) for value in segment))) is not None
    ]

    # 3. 将碎片线段融合成干净的全局长线，最多保留前 48 条长线
    lines = _merge_lines(raw_lines, angle_tol=np.deg2rad(0.5), rho_tol=0.7 * scale)
    lines = sorted(lines, key=lambda item: item.support, reverse=True)[:48]

    # 4. 对边缘图像进行距离变换，计算每个像素到最近边缘的距离图，用于边缘匹配验证
    distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    # 5. 二值化以获取前景实心掩膜，用于验证正方形内部实心度
    _, foreground_u8 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    foreground = foreground_u8 > 0

    # 6. 两两组合寻找方向平行且间距在正方形边长范围内的“平行线对”
    parallel_pairs: list[tuple[_Line, _Line, float]] = []
    for first, second in combinations(lines, 2):
        if _angle_distance(first.theta, second.theta) > np.deg2rad(1.0):
            continue
        separation_mm = abs(first.rho - second.rho) / scale
        if MIN_SIDE_MM <= separation_mm <= MAX_SIDE_MM:
            parallel_pairs.append((first, second, separation_mm))

    # 7. 两两组合两对垂直的平行线对，交叉求点，生成四边形正方形候选框
    candidates: list[ReconstructedSquare] = []
    geometries: list[tuple[np.ndarray, float, float, float]] = []
    for pair_a, pair_b in combinations(parallel_pairs, 2):
        perpendicular_error = abs(_angle_distance(pair_a[0].theta, pair_b[0].theta) - pi / 2.0)
        # 夹角需在 90 度垂直范围附近
        if perpendicular_error > np.deg2rad(1.25):
            continue
        # 两个平行线对确定的边长必须一致（边长差小于 2.5 mm）
        if abs(pair_a[2] - pair_b[2]) > 2.5:
            continue
        geometry = _square_from_pairs(pair_a, pair_b)
        if geometry is not None:
            geometries.append((geometry, pair_a[2], pair_b[2], perpendicular_error))

    # 8. 严格的双重验证（边缘吻合度与内部实心度）
    for polygon, side_a, side_b, perpendicular_error in geometries:
        # 过滤超出 A4 有效内容范围的越界候选
        if np.any(polygon[:, 0] < -7 * scale) or np.any(polygon[:, 1] < -7 * scale):
            continue
        if np.any(polygon[:, 0] >= roi.shape[1] + 7 * scale) or np.any(polygon[:, 1] >= roi.shape[0] + 7 * scale):
            continue

        # 校验 4 条边的边缘吻合率
        supports = [
            _edge_support(distance, polygon[index], polygon[(index + 1) % 4], 0.8 * scale)
            for index in range(4)
        ]
        mean_support = float(np.mean(supports))
        min_support = float(np.min(supports))
        # 校验多边形内部黑色填充率
        interior = _interior_support(foreground, polygon)

        # 阈值卡扣：单边支持度不小于 40%，平均支持度不小于 55%，实心黑色度不小于 94%
        if min_support < 0.40 or mean_support < 0.55 or interior < 0.94:
            continue

        # 映射回全图像素坐标下
        polygon[:, 0] += x0
        polygon[:, 1] += y0

        center = polygon.mean(axis=0)
        side_mm = (side_a + side_b) / 2.0
        # 计算旋转角度
        direction = atan2(*(polygon[1] - polygon[0])[::-1])
        angle = (degrees(direction) + 45.0) % 90.0 - 45.0

        # 计算打分权重，融合出置信度得分
        geometry_score = max(0.0, 1.0 - abs(side_a - side_b) / 2.5)
        perpendicular_score = max(0.0, 1.0 - perpendicular_error / np.deg2rad(1.25))
        confidence = (
            0.30 * mean_support
            + 0.25 * min_support
            + 0.20 * interior
            + 0.15 * geometry_score
            + 0.10 * perpendicular_score
        )
        candidates.append(
            ReconstructedSquare(
                corners=polygon,
                center_mm=(float(center[0] / scale_x), float(center[1] / scale_y)),
                side_mm=side_mm,
                angle_deg=angle,
                confidence=confidence,
            )
        )

    # 9. 对候选框进行非极大值抑制去重并排序输出
    squares = _deduplicate_squares(candidates)
    squares.sort(key=lambda item: (item.center_mm[1], item.center_mm[0]))
    return OverlapMeasurementResult(tuple(squares), edges)


def draw_overlap_measurements(image: np.ndarray, result: OverlapMeasurementResult) -> np.ndarray:
    """
    可视化调试绘制函数：用红色边框勾勒出重建的每一个正方形，并标注序号和物理边长尺寸。
    """
    debug = image.copy()
    for index, square in enumerate(result.squares, start=1):
        corners = np.rint(square.corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(debug, [corners], True, (0, 0, 255), 4, cv2.LINE_AA)
        origin = tuple(corners.reshape(-1, 2).min(axis=0))
        cv2.putText(
            debug,
            f"{index}: {square.side_mm:.1f} mm",
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
    return debug
