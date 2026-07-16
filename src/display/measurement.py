"""把测量数据转换成中文状态、目标量和标注图。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.vision.measure import draw_measurements
from src.vision.measurement_tasks import NumberedMeasurementResult, NormalMeasurementResult

from .models import MeasurementView

if TYPE_CHECKING:
    from src.application import ApplicationResult


LAYOUT_NAMES = {
    "single": "单个目标",
    "separate": "多个分离目标",
    "overlap": "多个重叠目标",
}
SHAPE_NAMES = {
    "circle": "圆形",
    "triangle": "三角形",
    "square": "正方形",
}


def draw_a4_location(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """在完整去畸变画面上描出 A4 黑框位置。"""
    debug = image.copy()
    points = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(debug, [points], True, (0, 200, 0), 4, cv2.LINE_AA)
    center = tuple(np.rint(corners.mean(axis=0)).astype(int))
    cv2.circle(debug, center, 7, (0, 200, 0), -1, cv2.LINE_AA)
    origin = tuple(points.reshape(-1, 2).min(axis=0))
    cv2.putText(
        debug,
        "A4",
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 200, 0),
        3,
        cv2.LINE_AA,
    )
    return debug


def normal_summary(
    task: NormalMeasurementResult,
    reprojection_error_px: float,
    plane_axis_angle_deg: float | None = None,
    tilted_plane: bool = False,
) -> str:
    """生成普通测量的中文识别状态。"""
    counts: dict[str, int] = {}
    for shape in task.geometry.shapes:
        name = SHAPE_NAMES.get(shape.shape_type, shape.shape_type)
        counts[name] = counts.get(name, 0) + 1
    detected = "、".join(f"{count} 个{name}" for name, count in counts.items())
    details = [
        f"  {index}. {SHAPE_NAMES.get(shape.shape_type, shape.shape_type)}，"
        f"尺寸 {shape.x_mm:.2f} mm"
        for index, shape in enumerate(task.geometry.shapes, 1)
    ]
    lines = ["识别成功", "模式：普通识别"]
    if tilted_plane:
        lines.extend(
            [
                "场景：发挥（4）倾斜物面",
                f"物面与轴线夹角：{plane_axis_angle_deg:.1f}°",
            ]
        )
    lines.extend(
        [
            f"布局：{LAYOUT_NAMES.get(task.geometry.layout, task.geometry.layout)}",
            f"检测结果：{detected}",
            f"测量输出：{task.x_mm:.2f} mm",
            (
                "目标距离：发挥（4）不要求输出"
                if tilted_plane
                else f"目标距离：{task.distance_mm:.1f} mm"
                if task.distance_mm is not None
                else "目标距离：未测量"
            ),
            f"A4 定位误差：{reprojection_error_px:.2f} px",
            "全部目标：",
            *details,
        ]
    )
    return "\n".join(lines)


def numbered_summary(
    task: NumberedMeasurementResult,
    reprojection_error_px: float,
) -> str:
    """生成数字测量的中文识别状态。"""
    details = []
    for index, square in enumerate(task.squares, 1):
        selected = "，已选中" if square is task.selected_square else ""
        details.append(
            f"  {index}. 数字 {square.digit}，尺寸 {square.geometry.x_mm:.2f} mm，"
            f"置信度 {square.digit_confidence:.3f}{selected}"
        )
    return "\n".join(
        [
            "识别成功",
            "模式：数字识别",
            f"输入数字：{task.requested_digit}",
            f"检测结果：{len(task.squares)} 个带数字正方形",
            f"选中方块尺寸：{task.x_mm:.2f} mm",
            f"目标距离：{task.distance_mm:.1f} mm"
            if task.distance_mm is not None
            else "目标距离：未测量",
            f"A4 定位误差：{reprojection_error_px:.2f} px",
            "全部目标：",
            *details,
        ]
    )


def target_text(result: ApplicationResult) -> str:
    """生成题目要求的目标量显示文本。"""
    distance = (
        f"{result.distance_mm:.1f} mm"
        if result.distance_mm is not None
        else "不要求输出/未测量"
    )
    if result.mode == "numbered":
        task = result.task
        assert isinstance(task, NumberedMeasurementResult)
        lines = [
            "目标量：识别成功",
            f"模式：数字识别，目标数字 {task.requested_digit}",
            f"目标方块尺寸 x：{result.x_mm:.2f} mm",
            f"目标距离 D：{distance}",
            f"A4 定位误差：{result.reprojection_error_px:.2f} px",
        ]
    else:
        lines = [
            "目标量：识别成功",
            "模式：普通识别",
            f"目标尺寸 x：{result.x_mm:.2f} mm",
            f"目标距离 D：{distance}",
            f"A4 定位误差：{result.reprojection_error_px:.2f} px",
        ]
    return "\n".join(lines)


def build_measurement_view(result: ApplicationResult) -> MeasurementView:
    """生成 GUI 可直接使用的完整展示结果。"""
    if result.mode == "normal":
        task = result.task
        assert isinstance(task, NormalMeasurementResult)
        summary = normal_summary(
            task,
            result.reprojection_error_px,
            result.plane_axis_angle_deg,
            result.tilted_plane,
        )
    else:
        task = result.task
        assert isinstance(task, NumberedMeasurementResult)
        summary = numbered_summary(task, result.reprojection_error_px)

    return MeasurementView(
        mode=result.mode,
        x_mm=result.x_mm,
        distance_mm=result.distance_mm,
        reprojection_error_px=result.reprojection_error_px,
        summary=summary,
        target_text=target_text(result),
        input_image=result.input_image,
        located_a4=draw_a4_location(result.undistorted_image, result.a4_corners),
        rectified_a4=result.rectified_a4,
        annotated_a4=draw_measurements(
            result.rectified_a4,
            result.task.geometry,
            result.selected_index,
        ),
    )
