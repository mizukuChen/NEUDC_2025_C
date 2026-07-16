from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from src.vision.distance import (
    estimate_plane_center_depth,
    load_distance_correction,
    undistorted_camera_matrix,
)
from src.vision.fisheye import load_fisheye_calibration, undistort_fisheye_image
from src.vision.measure import (
    Measurement,
    MeasurementResult,
    measure_low_resolution,
)
from src.vision.lowres_preprocess import WORK_HEIGHT_PX, WORK_WIDTH_PX
from src.vision.measurement_tasks import (
    NumberedMeasurementResult,
    NumberedSquareMeasurement,
    NormalMeasurementResult,
    measure_normal,
    measure_numbered,
    normal_measurement_from_geometry,
    numbered_measurement_from_recognitions,
    recognize_numbered_squares,
)
from src.vision.rectify import rectify_a4
from src.vision.target_plane import find_target_plane
from src.hardware.camera_settings import (
    DEFAULT_CAMERA_SETTINGS_PATH,
    CameraDeviceLock,
    CameraSettings,
    apply_camera_controls,
    device_path,
    load_camera_settings,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FISHEYE = ROOT / "runtime/calibration/fisheye_1280x720.json"
DEFAULT_DISTANCE = ROOT / "runtime/calibration/distance_linear.json"

TILTED_PLANE_MIN_DEG = 25.0
TILTED_PLANE_MAX_DEG = 65.0


@dataclass(frozen=True)
class ApplicationResult:
    mode: str
    x_mm: float
    distance_mm: float | None
    reprojection_error_px: float
    plane_axis_angle_deg: float
    tilted_plane: bool
    task: NormalMeasurementResult | NumberedMeasurementResult
    selected_index: int
    input_image: np.ndarray
    undistorted_image: np.ndarray
    a4_corners: np.ndarray
    rectified_a4: np.ndarray


@dataclass(frozen=True)
class CameraSceneSignature:
    """供静默预识别比较的归一化A4位置与纸内二值图案。"""

    normalized_corners: np.ndarray
    content_mask: np.ndarray


@dataclass(frozen=True)
class CameraSceneAnalysis:
    """一次相机完整分析的模式无关结果，可派生普通或任意数字输出。"""

    geometry: MeasurementResult
    numbered_squares: tuple[NumberedSquareMeasurement, ...] | None
    corrected_distance_mm: float
    reprojection_error_px: float
    plane_axis_angle_deg: float
    input_image: np.ndarray
    undistorted_image: np.ndarray
    a4_corners: np.ndarray
    rectified_a4: np.ndarray


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


def _configure_measurement_camera(
    capture,
    width: int,
    height: int,
    settings: CameraSettings,
    camera_device: str | None = None,
) -> None:
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, 5)
    actual_device = (
        camera_device
        if type(capture).__module__.split(".", 1)[0] == "cv2"
        else None
    )
    apply_camera_controls(capture, settings, actual_device)


class CameraFrameBuffer:
    """常驻V4L2相机线程，持续保存最新的三张稳定帧。"""

    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
        frame_count: int = 3,
        warmup_frames: int = 20,
        settings: CameraSettings | None = None,
    ) -> None:
        if frame_count < 1:
            raise ValueError("frame_count must be at least 1.")
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self.warmup_frames = warmup_frames
        self.settings = settings or CameraSettings()
        self._frames: deque[np.ndarray] = deque(maxlen=frame_count)
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._error: str | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="camera-frame-buffer",
            daemon=True,
        )

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def frames(self, timeout_s: float = 3.0) -> tuple[np.ndarray, ...]:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while len(self._frames) < self.frame_count and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            if len(self._frames) < self.frame_count:
                raise RuntimeError(self._error or "找不到camera")
            return tuple(self._frames)

    def _set_error(self, message: str) -> None:
        with self._condition:
            self._frames.clear()
            self._error = message
            self._condition.notify_all()

    def _run(self) -> None:
        while not self._stop.is_set():
            capture = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
            if not capture.isOpened():
                capture.release()
                self._set_error("找不到camera")
                self._stop.wait(0.5)
                continue
            try:
                _configure_measurement_camera(
                    capture,
                    self.width,
                    self.height,
                    self.settings,
                    device_path(self.camera_index),
                )
                healthy = True
                for _ in range(self.warmup_frames):
                    if self._stop.is_set():
                        healthy = False
                        break
                    if not capture.grab():
                        self._set_error("找不到camera")
                        healthy = False
                        break
                while healthy and not self._stop.is_set():
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        self._set_error("找不到camera")
                        break
                    with self._condition:
                        self._frames.append(frame)
                        self._error = None
                        self._condition.notify_all()
            except Exception as exc:
                self._set_error(str(exc))
            finally:
                capture.release()
            if not self._stop.is_set():
                self._stop.wait(0.5)


class MeasurementApplication:
    def __init__(
        self,
        fisheye_path: Path = DEFAULT_FISHEYE,
        distance_path: Path = DEFAULT_DISTANCE,
        strip_fraction: float | None = None,
        max_reprojection_px: float = 2.0,
        camera_settings_path: Path = DEFAULT_CAMERA_SETTINGS_PATH,
    ) -> None:
        if strip_fraction is not None and not 0 < strip_fraction <= 1:
            raise ValueError("strip_fraction must be in (0, 1].")
        self.fisheye_path = Path(fisheye_path)
        self.distance_path = Path(distance_path)
        self.strip_fraction = strip_fraction
        self.max_reprojection_px = max_reprojection_px
        self.camera_settings_path = Path(camera_settings_path)
        self._fisheye = None
        self._correction = None
        self._camera_buffer: CameraFrameBuffer | None = None
        self._camera_lock = CameraDeviceLock()
        self._undistortion_cache: dict[
            tuple[int, int],
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ] = {}

    def _camera_roi_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """读取现场配置的矩形ROI；显式旧参数仅用于兼容历史调用。"""
        if self.strip_fraction is not None:
            roi_width = round(width * self.strip_fraction)
            left = (width - roi_width) // 2
            return left, 0, left + roi_width, height
        settings = load_camera_settings(self.camera_settings_path)
        return settings.roi_pixels(width, height)

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

    def _camera_undistortion_maps(
        self,
        fisheye,
        image_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """缓存全分辨率定点映射、浮点映射和去畸变后内参。"""
        cached = self._undistortion_cache.get(image_size)
        if cached is not None:
            return cached
        width, height = image_size
        source_width, source_height = fisheye.image_size
        scale_x = width / source_width
        scale_y = height / source_height
        if abs(scale_x - scale_y) > 1e-6:
            raise ValueError("Input aspect ratio does not match the fisheye calibration.")
        camera_matrix = fisheye.camera_matrix.copy()
        camera_matrix[0, 0] *= scale_x
        camera_matrix[1, 1] *= scale_y
        camera_matrix[0, 2] *= scale_x
        camera_matrix[1, 2] *= scale_y
        new_camera_matrix = undistorted_camera_matrix(fisheye, image_size)
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix,
            fisheye.distortion,
            np.eye(3),
            new_camera_matrix,
            image_size,
            cv2.CV_32FC1,
        )
        fixed_map, interpolation_map = cv2.convertMaps(
            map_x,
            map_y,
            cv2.CV_16SC2,
        )
        cached = (
            fixed_map,
            interpolation_map,
            map_x,
            map_y,
            new_camera_matrix,
        )
        self._undistortion_cache[image_size] = cached
        return cached

    @staticmethod
    def _direct_rectify_frames(
        frames: list[np.ndarray],
        full_corners: np.ndarray,
        map_x: np.ndarray,
        map_y: np.ndarray,
    ) -> list[np.ndarray]:
        """组合鱼眼映射与透视变换，原始帧直接输出420×594 A4。"""
        destination = np.asarray(
            [
                [0, 0],
                [WORK_WIDTH_PX - 1, 0],
                [WORK_WIDTH_PX - 1, WORK_HEIGHT_PX - 1],
                [0, WORK_HEIGHT_PX - 1],
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(
            np.asarray(full_corners, dtype=np.float32),
            destination,
        )
        size = (WORK_WIDTH_PX, WORK_HEIGHT_PX)
        direct_x = cv2.warpPerspective(
            map_x,
            transform,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1,
        )
        direct_y = cv2.warpPerspective(
            map_y,
            transform,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1,
        )
        return [
            cv2.remap(
                frame,
                direct_x,
                direct_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            for frame in frames
        ]

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
        settings: CameraSettings | None = None,
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
            _configure_measurement_camera(
                cap,
                width,
                height,
                settings or load_camera_settings(),
                device_path(camera_index),
            )
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

    def start_camera(
        self,
        camera_index: int = 0,
        width: int = 1920,
        height: int = 1080,
    ) -> None:
        """在GUI启动时异步预热相机；重复调用不会重复创建线程。"""
        if self._camera_buffer is not None:
            return
        self._camera_lock.acquire()
        try:
            fisheye, _ = self._load()
            self._camera_undistortion_maps(fisheye, (width, height))
            self._camera_buffer = CameraFrameBuffer(
                camera_index,
                width,
                height,
                settings=load_camera_settings(self.camera_settings_path),
            )
            self._camera_buffer.start()
        except Exception:
            self._camera_buffer = None
            self._camera_lock.release()
            raise

    def stop_camera(self) -> None:
        if self._camera_buffer is not None:
            self._camera_buffer.stop()
            self._camera_buffer = None
        self._camera_lock.release()

    def buffered_camera_frames(self, timeout_s: float = 6.0) -> tuple[np.ndarray, ...]:
        if self._camera_buffer is None:
            raise RuntimeError("找不到camera")
        return self._camera_buffer.frames(timeout_s)

    def probe_camera_frame(self, raw: np.ndarray) -> CameraSceneSignature:
        """轻量定位一帧中的A4，并生成抗光照变化的纸内内容特征。"""
        if raw is None or raw.size == 0:
            raise ValueError("输入图像为空")
        fisheye, _ = self._load()
        height, width = raw.shape[:2]
        fixed_map, interpolation_map, _, _, _ = self._camera_undistortion_maps(
            fisheye,
            (width, height),
        )
        undistorted = cv2.remap(
            raw,
            fixed_map,
            interpolation_map,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        left, top, right, bottom = self._camera_roi_pixels(width, height)
        roi = undistorted[top:bottom, left:right]
        plane = find_target_plane(roi)

        preview = rectify_a4(roi, plane.corners, 80, 112)
        gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
        margin_x = max(2, round(gray.shape[1] * 0.07))
        margin_y = max(2, round(gray.shape[0] * 0.07))
        inner = gray[margin_y:-margin_y, margin_x:-margin_x]
        blurred = cv2.GaussianBlur(inner, (3, 3), 0)
        _, binary = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        normalized = plane.corners.astype(np.float32) / np.asarray(
            [roi.shape[1], roi.shape[0]],
            dtype=np.float32,
        )
        return CameraSceneSignature(normalized, binary > 0)

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
        left, top, right, bottom = self._camera_roi_pixels(width, height)
        roi = undistorted[top:bottom, left:right]
        plane = find_target_plane(roi)
        full_corners = plane.corners.astype(np.float64)
        full_corners += np.asarray([left, top], dtype=np.float64)
        distance = estimate_plane_center_depth(
            full_corners,
            undistorted_camera_matrix(fisheye, (width, height)),
            correction,
        )
        if distance.reprojection_error_px > self.max_reprojection_px:
            raise ValueError(
                f"A4 定位重投影误差过大：{distance.reprojection_error_px:.2f} px"
            )

        rectified = rectify_a4(roi, plane.corners)
        plane_angle = _plane_axis_angle_deg(distance.rvec)
        if mode == "normal":
            tilted = _is_tilted_plane(plane_angle)
            task = measure_normal(
                rectified,
                None if tilted else distance.corrected_distance_mm,
            )
            selected = task.selected_shape
            output_distance = None if tilted else distance.corrected_distance_mm
        else:
            tilted = False
            task = measure_numbered(rectified, int(digit), distance.corrected_distance_mm)
            selected = task.selected_square.geometry
            output_distance = distance.corrected_distance_mm

        selected_index = _selected_index(task.geometry, selected)
        return ApplicationResult(
            mode=mode,
            x_mm=task.x_mm,
            distance_mm=output_distance,
            reprojection_error_px=distance.reprojection_error_px,
            plane_axis_angle_deg=plane_angle,
            tilted_plane=tilted,
            task=task,
            selected_index=selected_index,
            input_image=raw.copy(),
            undistorted_image=undistorted,
            a4_corners=full_corners,
            rectified_a4=rectified,
        )

    def process_camera_frames(
        self,
        frames: tuple[np.ndarray, ...] | list[np.ndarray],
        mode: str,
        digit: int | None = None,
    ) -> ApplicationResult:
        """保留原按键入口：分析当前三帧，再生成所请求模式的结果。"""
        if mode not in {"normal", "numbered"}:
            raise ValueError(f"未知测量模式：{mode}")
        if mode == "numbered" and (digit is None or not 0 <= digit <= 9):
            raise ValueError("数字识别模式必须输入 0～9")
        analysis = self.analyze_camera_frames(
            frames,
            recognize_digits=mode == "numbered",
        )
        return self.result_from_camera_analysis(analysis, mode, digit)

    def analyze_camera_frames(
        self,
        frames: tuple[np.ndarray, ...] | list[np.ndarray],
        recognize_digits: bool = True,
    ) -> CameraSceneAnalysis:
        """只做一次A4定位、几何检测，并按需一次性识别全部正方形数字。"""
        if not frames:
            raise ValueError("输入相机帧为空")
        arrays = [np.asarray(frame) for frame in frames]
        shape = arrays[0].shape
        if arrays[0].size == 0 or any(frame.size == 0 or frame.shape != shape for frame in arrays):
            raise ValueError("输入相机帧必须非空且尺寸一致")

        fisheye, correction = self._load()
        height, width = arrays[-1].shape[:2]
        fixed_map, interpolation_map, map_x, map_y, camera_matrix = (
            self._camera_undistortion_maps(fisheye, (width, height))
        )
        undistorted_latest = cv2.remap(
            arrays[-1],
            fixed_map,
            interpolation_map,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        left, top, right, bottom = self._camera_roi_pixels(width, height)
        roi = undistorted_latest[top:bottom, left:right]
        plane = find_target_plane(roi)
        full_corners = plane.corners.astype(np.float64)
        full_corners += np.asarray([left, top], dtype=np.float64)
        rectified_frames = self._direct_rectify_frames(
            arrays,
            full_corners,
            map_x,
            map_y,
        )
        distance = estimate_plane_center_depth(
            full_corners,
            camera_matrix,
            correction,
        )
        if distance.reprojection_error_px > self.max_reprojection_px:
            raise ValueError(
                f"A4 定位重投影误差过大：{distance.reprojection_error_px:.2f} px"
            )

        geometry = measure_low_resolution(rectified_frames)
        measurement_image = geometry.measurement_image
        if measurement_image is None:
            raise ValueError("低分辨率测量没有生成融合后的A4图像")
        numbered_squares = None
        if recognize_digits and geometry.shapes and all(
            item.shape_type == "square" for item in geometry.shapes
        ):
            numbered_squares = recognize_numbered_squares(
                measurement_image,
                geometry,
            )
        return CameraSceneAnalysis(
            geometry=geometry,
            numbered_squares=numbered_squares,
            corrected_distance_mm=distance.corrected_distance_mm,
            reprojection_error_px=distance.reprojection_error_px,
            plane_axis_angle_deg=_plane_axis_angle_deg(distance.rvec),
            input_image=arrays[-1].copy(),
            undistorted_image=undistorted_latest,
            a4_corners=full_corners,
            rectified_a4=measurement_image,
        )

    def can_recognize_digits_from_analysis(self, analysis: CameraSceneAnalysis) -> bool:
        """判断已有几何缓存是否足以补做数字识别。"""
        return (
            analysis.numbered_squares is None
            and bool(analysis.geometry.shapes)
            and all(item.shape_type == "square" for item in analysis.geometry.shapes)
        )

    def recognize_digits_for_analysis(
        self,
        analysis: CameraSceneAnalysis,
    ) -> CameraSceneAnalysis:
        """在不重跑A4定位/几何重建的情况下，为几何缓存补充数字识别结果。"""
        if not self.can_recognize_digits_from_analysis(analysis):
            return analysis
        numbered_squares = recognize_numbered_squares(
            analysis.rectified_a4,
            analysis.geometry,
        )
        return replace(analysis, numbered_squares=numbered_squares)

    def result_from_camera_analysis(
        self,
        analysis: CameraSceneAnalysis,
        mode: str,
        digit: int | None = None,
    ) -> ApplicationResult:
        """从同一场景分析快速派生普通输出或任意指定数字输出。"""
        if mode not in {"normal", "numbered"}:
            raise ValueError(f"未知测量模式：{mode}")
        if mode == "normal":
            tilted = _is_tilted_plane(analysis.plane_axis_angle_deg)
            task = normal_measurement_from_geometry(
                analysis.geometry,
                None if tilted else analysis.corrected_distance_mm,
            )
            selected = task.selected_shape
            output_distance = None if tilted else analysis.corrected_distance_mm
        else:
            tilted = False
            if digit is None or not 0 <= digit <= 9:
                raise ValueError("数字识别模式必须输入 0～9")
            if analysis.numbered_squares is None:
                raise ValueError("当前场景没有可用的数字正方形识别结果")
            task = numbered_measurement_from_recognitions(
                analysis.geometry,
                analysis.numbered_squares,
                int(digit),
                analysis.corrected_distance_mm,
            )
            selected = task.selected_square.geometry
            output_distance = analysis.corrected_distance_mm

        selected_index = _selected_index(task.geometry, selected)
        return ApplicationResult(
            mode=mode,
            x_mm=task.x_mm,
            distance_mm=output_distance,
            reprojection_error_px=analysis.reprojection_error_px,
            plane_axis_angle_deg=analysis.plane_axis_angle_deg,
            tilted_plane=tilted,
            task=task,
            selected_index=selected_index,
            input_image=analysis.input_image,
            undistorted_image=analysis.undistorted_image,
            a4_corners=analysis.a4_corners,
            rectified_a4=analysis.rectified_a4,
        )
