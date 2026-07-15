from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.vision.distance import (
    estimate_plane_center_depth,
    load_distance_correction,
    undistorted_camera_matrix,
)
from src.vision.fisheye import load_fisheye_calibration, undistort_fisheye_image
from src.vision.measure import Measurement, MeasurementResult, draw_measurements
from src.vision.lowres_preprocess import WORK_HEIGHT_PX, WORK_WIDTH_PX
from src.vision.measurement_tasks import (
    NumberedMeasurementResult,
    NormalMeasurementResult,
    measure_normal,
    measure_normal_low_resolution,
    measure_numbered,
    measure_numbered_low_resolution,
)
from src.vision.rectify import rectify_a4
from src.vision.target_plane import find_target_plane


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FISHEYE = ROOT / "runtime/calibration/fisheye_1280x720.json"
DEFAULT_DISTANCE = ROOT / "runtime/calibration/distance_linear.json"

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
TILTED_PLANE_MIN_DEG = 25.0
TILTED_PLANE_MAX_DEG = 65.0


@dataclass(frozen=True)
class ApplicationResult:
    mode: str
    x_mm: float
    distance_mm: float | None
    reprojection_error_px: float
    summary: str
    input_image: np.ndarray
    located_a4: np.ndarray
    rectified_a4: np.ndarray
    annotated_a4: np.ndarray


def _selected_index(result: MeasurementResult, selected: Measurement) -> int:
    return next(index for index, shape in enumerate(result.shapes) if shape is selected)


def _plane_axis_angle_deg(rvec: np.ndarray) -> float:
    """由A4姿态计算物面与相机光轴之间的锐角。"""
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    plane_normal = rotation[:, 2]
    return float(
        np.degrees(
            np.arcsin(np.clip(abs(float(plane_normal[2])), 0.0, 1.0))
        )
    )


def _is_tilted_plane(angle_deg: float) -> bool:
    # 给题目30°～60°范围各留5°姿态估计余量。
    return TILTED_PLANE_MIN_DEG <= angle_deg <= TILTED_PLANE_MAX_DEG


def _normal_summary(
    task: NormalMeasurementResult,
    reprojection_error_px: float,
    plane_axis_angle_deg: float | None = None,
) -> str:
    counts: dict[str, int] = {}
    for shape in task.geometry.shapes:
        name = SHAPE_NAMES.get(shape.shape_type, shape.shape_type)
        counts[name] = counts.get(name, 0) + 1
    detected = "、".join(f"{count} 个{name}" for name, count in counts.items())
    details = [
        f"  {index}. {SHAPE_NAMES.get(shape.shape_type, shape.shape_type)}，尺寸 {shape.x_mm:.2f} mm"
        for index, shape in enumerate(task.geometry.shapes, 1)
    ]
    tilted = (
        plane_axis_angle_deg is not None
        and _is_tilted_plane(plane_axis_angle_deg)
    )
    lines = ["识别成功", "模式：普通识别"]
    if tilted:
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
                if tilted
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


def _numbered_summary(task: NumberedMeasurementResult, reprojection_error_px: float) -> str:
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
            f"目标距离：{task.distance_mm:.1f} mm" if task.distance_mm is not None else "目标距离：未测量",
            f"A4 定位误差：{reprojection_error_px:.2f} px",
            "全部目标：",
            *details,
        ]
    )


def _draw_a4_location(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    debug = image.copy()
    points = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(debug, [points], True, (0, 200, 0), 4, cv2.LINE_AA)
    center = tuple(np.rint(corners.mean(axis=0)).astype(int))
    cv2.circle(debug, center, 7, (0, 200, 0), -1, cv2.LINE_AA)
    origin = tuple(points.reshape(-1, 2).min(axis=0))
    cv2.putText(debug, "A4", origin, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3, cv2.LINE_AA)
    return debug


class MeasurementApplication:
    def __init__(
        self,
        fisheye_path: Path = DEFAULT_FISHEYE,
        distance_path: Path = DEFAULT_DISTANCE,
        strip_fraction: float = 1 / 3,
        max_reprojection_px: float = 2.0,
    ) -> None:
        if not 0 < strip_fraction <= 1:
            raise ValueError("strip_fraction must be in (0, 1].")
        self.fisheye_path = Path(fisheye_path)
        self.distance_path = Path(distance_path)
        self.strip_fraction = strip_fraction
        self.max_reprojection_px = max_reprojection_px
        self._fisheye = None
        self._correction = None

    def _load(self):
        if self._fisheye is None:
            if not self.fisheye_path.exists():
                raise FileNotFoundError(f"缺少鱼眼标定文件：{self.fisheye_path}")
            self._fisheye = load_fisheye_calibration(self.fisheye_path)
            self._correction = (
                load_distance_correction(self.distance_path)
                if self.distance_path.exists()
                else None
            )
        return self._fisheye, self._correction

    @staticmethod
    def capture_frame(camera_index: int = 0, width: int = 1280, height: int = 720) -> np.ndarray:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("找不到camera")
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            for _ in range(5):
                cap.grab()
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            raise RuntimeError("找不到camera")
        return frame

    @staticmethod
    def capture_frames(
        camera_index: int = 0,
        width: int = 1920,
        height: int = 1080,
        frame_count: int = 3,
        warmup_frames: int = 20,
        brightness: float = -20.0,
    ) -> tuple[np.ndarray, ...]:
        """用树莓派已验证的YUYV参数采集用于融合测量的稳定帧。"""
        if frame_count < 1:
            raise ValueError("frame_count must be at least 1.")
        if warmup_frames < 0:
            raise ValueError("warmup_frames must not be negative.")
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("找不到camera")
        frames: list[np.ndarray] = []
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 5)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
            for _ in range(warmup_frames):
                if not cap.grab():
                    raise RuntimeError("找不到camera")
            for _ in range(frame_count):
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("找不到camera")
                frames.append(frame)
        finally:
            cap.release()
        return tuple(frames)

    def process_raw(self, raw: np.ndarray, mode: str, digit: int | None = None) -> ApplicationResult:
        if raw is None or raw.size == 0:
            raise ValueError("输入图像为空")
        if mode not in {"normal", "numbered"}:
            raise ValueError(f"未知测量模式：{mode}")
        if mode == "numbered" and (digit is None or not 0 <= digit <= 9):
            raise ValueError("数字识别模式必须输入 0～9")

        fisheye, correction = self._load()
        undistorted = undistort_fisheye_image(raw, fisheye)
        height, width = undistorted.shape[:2]
        strip_width = round(width * self.strip_fraction)
        left = (width - strip_width) // 2
        strip = undistorted[:, left : left + strip_width]
        plane = find_target_plane(strip)
        full_corners = plane.corners.astype(np.float64)
        full_corners[:, 0] += left
        distance = estimate_plane_center_depth(
            full_corners,
            undistorted_camera_matrix(fisheye, (width, height)),
            correction,
        )
        if distance.reprojection_error_px > self.max_reprojection_px:
            raise ValueError(
                f"A4 定位重投影误差过大：{distance.reprojection_error_px:.2f} px"
            )

        rectified = rectify_a4(strip, plane.corners)
        if mode == "normal":
            plane_angle = _plane_axis_angle_deg(distance.rvec)
            tilted = _is_tilted_plane(plane_angle)
            task = measure_normal(
                rectified,
                None if tilted else distance.corrected_distance_mm,
            )
            selected = task.selected_shape
            summary = _normal_summary(
                task,
                distance.reprojection_error_px,
                plane_angle,
            )
            output_distance = None if tilted else distance.corrected_distance_mm
        else:
            task = measure_numbered(rectified, int(digit), distance.corrected_distance_mm)
            selected = task.selected_square.geometry
            summary = _numbered_summary(task, distance.reprojection_error_px)
            output_distance = distance.corrected_distance_mm

        selected_index = _selected_index(task.geometry, selected)
        return ApplicationResult(
            mode=mode,
            x_mm=task.x_mm,
            distance_mm=output_distance,
            reprojection_error_px=distance.reprojection_error_px,
            summary=summary,
            input_image=raw.copy(),
            located_a4=_draw_a4_location(undistorted, full_corners),
            rectified_a4=rectified,
            annotated_a4=draw_measurements(rectified, task.geometry, selected_index),
        )

    def process_camera_frames(
        self,
        frames: tuple[np.ndarray, ...] | list[np.ndarray],
        mode: str,
        digit: int | None = None,
    ) -> ApplicationResult:
        """定位并拉直多帧相机图像，再走低分辨率融合测量流程。"""
        if not frames:
            raise ValueError("输入相机帧为空")
        arrays = [np.asarray(frame) for frame in frames]
        shape = arrays[0].shape
        if arrays[0].size == 0 or any(frame.size == 0 or frame.shape != shape for frame in arrays):
            raise ValueError("输入相机帧必须非空且尺寸一致")
        if mode not in {"normal", "numbered"}:
            raise ValueError(f"未知测量模式：{mode}")
        if mode == "numbered" and (digit is None or not 0 <= digit <= 9):
            raise ValueError("数字识别模式必须输入 0～9")

        fisheye, correction = self._load()
        undistorted_frames: list[np.ndarray] = []
        rectified_frames: list[np.ndarray] = []
        full_corners_per_frame: list[np.ndarray] = []
        for raw in arrays:
            undistorted = undistort_fisheye_image(raw, fisheye)
            height, width = undistorted.shape[:2]
            strip_width = round(width * self.strip_fraction)
            left = (width - strip_width) // 2
            strip = undistorted[:, left : left + strip_width]
            plane = find_target_plane(strip)
            full_corners = plane.corners.astype(np.float64)
            full_corners[:, 0] += left
            undistorted_frames.append(undistorted)
            full_corners_per_frame.append(full_corners)
            rectified_frames.append(
                rectify_a4(strip, plane.corners, WORK_WIDTH_PX, WORK_HEIGHT_PX)
            )

        height, width = undistorted_frames[-1].shape[:2]
        median_corners = np.median(np.stack(full_corners_per_frame), axis=0)
        distance = estimate_plane_center_depth(
            median_corners,
            undistorted_camera_matrix(fisheye, (width, height)),
            correction,
        )
        if distance.reprojection_error_px > self.max_reprojection_px:
            raise ValueError(
                f"A4 定位重投影误差过大：{distance.reprojection_error_px:.2f} px"
            )

        if mode == "normal":
            plane_angle = _plane_axis_angle_deg(distance.rvec)
            tilted = _is_tilted_plane(plane_angle)
            task = measure_normal_low_resolution(
                rectified_frames,
                None if tilted else distance.corrected_distance_mm,
            )
            selected = task.selected_shape
            summary = _normal_summary(
                task,
                distance.reprojection_error_px,
                plane_angle,
            )
            output_distance = None if tilted else distance.corrected_distance_mm
        else:
            task = measure_numbered_low_resolution(
                rectified_frames,
                int(digit),
                distance.corrected_distance_mm,
            )
            selected = task.selected_square.geometry
            summary = _numbered_summary(task, distance.reprojection_error_px)
            output_distance = distance.corrected_distance_mm

        measurement_image = task.geometry.measurement_image
        if measurement_image is None:
            raise ValueError("低分辨率测量没有生成融合后的A4图像")
        selected_index = _selected_index(task.geometry, selected)
        return ApplicationResult(
            mode=mode,
            x_mm=task.x_mm,
            distance_mm=output_distance,
            reprojection_error_px=distance.reprojection_error_px,
            summary=summary,
            input_image=arrays[-1].copy(),
            located_a4=_draw_a4_location(undistorted_frames[-1], median_corners),
            rectified_a4=measurement_image,
            annotated_a4=draw_measurements(
                measurement_image,
                task.geometry,
                selected_index,
            ),
        )
