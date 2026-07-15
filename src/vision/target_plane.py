from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TargetPlane:
    """
    用于保存检测到的目标平面（A4 纸）信息的数据结构。

    属性:
        corners: 4x2 的 NumPy 数组，按 [左上, 右上, 右下, 左下] 顺序排列的角点像素坐标 (x, y)。
        contour_area: 检测到的 A4 黑边框轮廓的像素面积，用于辅助过滤背景噪声。
    """
    corners: np.ndarray
    contour_area: float
    strong_perspective: bool = False


def order_corners(points: np.ndarray) -> np.ndarray:
    """
    将任意顺序的 4 个角点坐标重新排序为标准顺序：[左上(TL), 右上(TR), 右下(BR), 左下(BL)]。
    这对于后续的透视变换（warpPerspective）和姿态估计（solvePnP）至关重要。

    原理:
        - 左上角点 (x, y) 靠近原点，其和 (x + y) 最小。
        - 右下角点 (x, y) 远离原点，其和 (x + y) 最大。
        - 右上角点 (x, y) 中 x 很大且 y 很小，其差 (y - x) 最小。
        - 左下角点 (x, y) 中 x 很小且 y 很大，其差 (y - x) 最大。
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(4)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]   # 左上角 TL (和最小)
    ordered[2] = pts[np.argmax(sums)]   # 右下角 BR (和最大)
    ordered[1] = pts[np.argmin(diffs)]  # 右上角 TR (差最小)
    ordered[3] = pts[np.argmax(diffs)]  # 左下角 BL (差最大)
    return ordered


def dark_mask(image: np.ndarray, close_iterations: int = 2) -> np.ndarray:
    """
    生成图像中的暗色（黑色）前景遮罩，用于寻找 A4 纸的黑色边框。
    结合了大津法（OTSU）、固定阈值和自适应阈值，以应对复杂的阴影和光照变化。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. 大津法：全局自动二值化，对强反差场景非常有效
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 2. 固定低阈值：滤除高亮度区域，强制判定极暗区域为黑色前景
    _, fixed = cv2.threshold(blurred, 95, 255, cv2.THRESH_BINARY_INV)
    # 3. 自适应阈值：局部邻域对比，能有效保留光照不均或阴影下的黑边线
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        9,
    )
    # 融合三者：既保留全局黑框，又能抗局部阴影
    mask = cv2.bitwise_or(fixed, cv2.bitwise_and(otsu, adaptive))

    # 使用形态学闭运算连接断裂的黑框线，核大小自适应于图像分辨率（约 1.5%）
    height, width = mask.shape[:2]
    kernel_size = max(3, round(min(height, width) * 0.015))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if close_iterations < 0:
        raise ValueError("close_iterations must not be negative.")
    if close_iterations:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=close_iterations,
        )
    return mask


def _contour_to_quad(contour: np.ndarray) -> np.ndarray | None:
    """
    尝试将一个闭合轮廓近似拟合为一个凸四边形。
    如果多边形逼近失败，则退而求其次使用最小外接矩形的 4 个角点。
    """
    perimeter = cv2.arcLength(contour, True)
    # 使用不同精度比例进行多边形逼近 (approxPolyDP)
    for ratio in (0.02, 0.03, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(contour, ratio * perimeter, True)
        # 如果逼近出 4 个顶点且是凸多边形，则认为是合格的四边形
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)

    # 兜底方案：使用最小外接矩形
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    if cv2.contourArea(box.astype(np.float32)) <= 0:
        return None
    return box.astype(np.float32)


def _border_score(mask: np.ndarray, quad: np.ndarray) -> float:
    """
    对候选四边形进行“A4 黑边框”相似度评分。

    原理:
        - 将候选四边形透视矫正为标准的 210 x 297 像素图像。
        - 统计图像外圈（宽 20 像素的边界区域）的平均灰度值（应该是黑色前景，占比大则均值高）。
        - 统计图像内部（避开黑框的白色空白区域）的平均灰度值（应该是白色背景，占比高则惩罚）。
        - 得分 = 边界区黑色占比 - 0.35 * 内部黑色占比。分值越高，越符合外黑内白的 A4 纸特征。
    """
    ordered = order_corners(quad)
    width, height = 210, 297
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    # 进行临时透视变换拉平，以方便在标准网格下计算评分
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(mask, matrix, (width, height))

    # 定义 20 像素宽的外圈边界区域掩膜
    border = 20
    border_mask = np.zeros((height, width), dtype=np.uint8)
    border_mask[:border, :] = 1
    border_mask[-border:, :] = 1
    border_mask[:, :border] = 1
    border_mask[:, -border:] = 1

    # 定义内部检测掩膜，避开黑框边缘干扰
    inner_margin = 35
    inner = warped[inner_margin : height - inner_margin, inner_margin : width - inner_margin]
    border_mean = float(warped[border_mask == 1].mean()) / 255.0
    inner_mean = float(inner.mean()) / 255.0 if inner.size else 0.0

    # 综合得分：边界越黑（值大）越好，内部越白（黑像素少，即值小）越好
    return border_mean - 0.35 * inner_mean


def _find_target_plane_once(
    image: np.ndarray,
    min_score: float = 0.35,
    close_iterations: int = 2,
    min_quad_fraction: float = 0.006,
    min_width_fraction: float = 0.08,
    max_aspect_ratio: float = 2.4,
) -> TargetPlane:
    """
    在单张图像（或裁剪区域）中搜索最佳的 A4 纸黑框平面。
    通过轮廓过滤、几何宽高比限制以及边界评分进行层层筛选。
    """
    mask = dark_mask(image, close_iterations=close_iterations)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No dark contour found.")

    image_area = image.shape[0] * image.shape[1]
    candidates: list[tuple[float, float, np.ndarray, float]] = []
    for contour in contours:
        # 面积太小则直接排除（小于图像面积的 0.3%）
        area = cv2.contourArea(contour)
        if area < image_area * 0.003:
            continue

        quad = _contour_to_quad(contour)
        if quad is None:
            continue

        # 限制候选四边形的像素面积范围 (0.6% ~ 90%)
        quad_area = abs(cv2.contourArea(quad))
        if quad_area < image_area * min_quad_fraction or quad_area > image_area * 0.90:
            continue

        # 检查候选框是否触碰图像边缘（触碰边界可能导致角点缺失）
        x, y, w, h = cv2.boundingRect(quad.astype(np.int32))
        edge_margin = max(2, round(min(image.shape[:2]) * 0.015))
        if (
            x <= edge_margin
            or y <= edge_margin
            or x + w >= image.shape[1] - edge_margin
            or y + h >= image.shape[0] - edge_margin
        ):
            continue

        # 尺寸限制：不能太小，且高宽比需在合理范围（0.8 ~ 2.4），过滤条状干扰
        if w < image.shape[1] * min_width_fraction or h < image.shape[0] * 0.08:
            continue
        if h / max(w, 1) < 0.8 or h / max(w, 1) > max_aspect_ratio:
            continue

        # 进行黑边框匹配度打分
        score = _border_score(mask, quad)
        if score < min_score:
            continue

        candidates.append((score, quad_area, quad, area))

    if not candidates:
        raise ValueError("No valid A4 border candidate found.")

    # 筛选出得分最高且面积最大的最佳候选框
    _, _, corners, contour_area = max(candidates, key=lambda item: (item[0], item[1]))
    return TargetPlane(corners=order_corners(corners), contour_area=contour_area)


def _candidate_regions(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    提供不同的候选裁切区域 [x, y, width, height]，以便在全局搜索失败时进行局部区域搜索。
    """
    height, width = image.shape[:2]
    return [
        (0, 0, width, height),                               # 1. 全局区域
        (0, height // 3, width, height - height // 3),       # 2. 下部区域（适合近处机位）
        (width // 4, height // 3, width // 2, height // 2),  # 3. 中心偏下区域
        (width // 5, height // 4, width * 3 // 5, height * 3 // 5), # 4. 中心缩放区域
    ]


def find_target_plane(image: np.ndarray) -> TargetPlane:
    """
    视觉定位的对外总接口。
    在全图及多个预设子区域中依次尝试定位 A4 平面，直到定位成功并返回排序后的角点。
    """
    errors: list[str] = []
    attempts = (
        (False, {}),
        (
            True,
            {
                # 实心暗块的理论得分约为0.65；强透视兜底必须高于它，
                # 防止把纸内正方形误当成A4黑框。
                "min_score": 0.70,
                "close_iterations": 1,
                "min_quad_fraction": 0.003,
                "min_width_fraction": 0.045,
                "max_aspect_ratio": 4.0,
            },
        ),
    )
    # 每个区域先常规、再强透视；全局强透视成功时无需等待全部局部常规尝试失败。
    for x, y, width, height in _candidate_regions(image):
        for strong_perspective, parameters in attempts:
            crop = image[y : y + height, x : x + width]
            try:
                plane = _find_target_plane_once(crop, **parameters)
            except ValueError as exc:
                errors.append(str(exc))
                continue

            # 映射回全图的全局像素坐标系下
            corners = plane.corners.copy()
            corners[:, 0] += x
            corners[:, 1] += y
            return TargetPlane(
                corners=order_corners(corners),
                contour_area=plane.contour_area,
                strong_perspective=strong_perspective,
            )

    raise ValueError(errors[-1] if errors else "No valid A4 border candidate found.")


def draw_target_plane_debug(image: np.ndarray, plane: TargetPlane) -> np.ndarray:
    """
    绘制定位调试图，标出 A4 纸绿色边缘轮廓和四个角点 (TL, TR, BR, BL)。
    """
    debug = image.copy()
    corners = plane.corners.astype(np.int32)
    cv2.polylines(debug, [corners], isClosed=True, color=(0, 255, 0), thickness=3)

    labels = ("TL", "TR", "BR", "BL")
    for label, (x, y) in zip(labels, corners):
        cv2.circle(debug, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(
            debug,
            label,
            (int(x) + 6, int(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return debug
