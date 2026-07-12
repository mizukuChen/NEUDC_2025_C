from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class FisheyeCalibration:
    """
    相机鱼眼畸变标定参数数据结构。

    属性:
        camera_matrix: 3x3 相机内参矩阵（包含焦距 fx, fy 和光心坐标 cx, cy）。
        distortion: 4x1 鱼眼畸变系数矩阵 (k1, k2, k3, k4)。
        image_size: 标定所使用的原始图片分辨率 (width, height)。
        rms_px: 重投影均方根误差（单位：像素），越小说明标定精度越高。
        used_images: 成功用于标定的棋盘格照片名称列表。
        rejected_images: 未能检测到棋盘格而被排除的照片名称列表。
        pattern_size: 棋盘格内角点尺寸 (columns, rows)。
        square_size_mm: 棋盘格中单个方格的实际物理边长（mm）。
    """
    camera_matrix: np.ndarray
    distortion: np.ndarray
    image_size: tuple[int, int]
    rms_px: float
    used_images: tuple[str, ...]
    rejected_images: tuple[str, ...]
    pattern_size: tuple[int, int]
    square_size_mm: float


def _fisheye_flag(name: str) -> int:
    """
    兼容 OpenCV 4 和 5 版本的鱼眼标定标志位加载函数。
    由于 OpenCV 5 调整了部分标志位的位置，该辅助函数能自动适配底层版本。
    """
    value = getattr(cv2.fisheye, name, None)
    if value is None:
        value = getattr(cv2, name, None)
    if value is None:
        raise RuntimeError(f"OpenCV does not expose the fisheye flag {name}.")
    return int(value)


def _find_corners(gray: np.ndarray, pattern_size: tuple[int, int]) -> np.ndarray | None:
    """
    在灰度图中寻找棋盘格内角点。
    优先尝试 OpenCV 4+ 中精度更高的 SB 棋盘格角点检测算法，失败则退回到经典算法并进行亚像素精修。
    """
    # 尝试使用高级的 findChessboardCornersSB
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags)
        if found:
            return corners.reshape(1, -1, 2).astype(np.float64)

    # 兜底：使用经典的 findChessboardCorners + 亚像素精修 (cornerSubPix)
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return refined.reshape(1, -1, 2).astype(np.float64)


def calibrate_fisheye(
    image_paths: Iterable[Path],
    pattern_size: tuple[int, int] = (8, 12),
    square_size_mm: float = 29.0,
) -> FisheyeCalibration:
    """
    从一组棋盘格照片中，标定出 OpenCV 鱼眼镜头模型（KANNALA-BRANDT）的相机内参和畸变参数。

    参数:
        image_paths: 标定棋盘格照片的文件路径迭代器。
        pattern_size: 棋盘格内角点行列数，例如 (8, 12) 代表 8 列 12 行。
        square_size_mm: 标定板上方格真实的物理边长（mm），默认 29mm。
    """
    paths = [Path(path) for path in image_paths]
    if not paths:
        raise ValueError("No calibration images were provided.")

    corner_count = pattern_size[0] * pattern_size[1]

    # 1. 建立标定板三维世界坐标系的虚拟角点三维世界坐标模板
    object_template = np.zeros((1, corner_count, 3), dtype=np.float64)
    object_template[0, :, :2] = (
        np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
        * square_size_mm
    )

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used_images: list[str] = []
    rejected_images: list[str] = []
    image_size: tuple[int, int] | None = None

    # 2. 遍历所有照片，查找 2D 角点
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            rejected_images.append(path.name)
            continue
        size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = size
        if size != image_size:
            rejected_images.append(path.name) # 分辨率不一致的图片直接排除
            continue

        corners = _find_corners(gray, pattern_size)
        if corners is None:
            rejected_images.append(path.name) # 未检出角点的图片排除
            continue
        object_points.append(object_template.copy())
        image_points.append(corners)
        used_images.append(path.name)

    # 鱼眼标定通常要求至少 8 张高质量的棋盘格角度照片
    if image_size is None or len(used_images) < 8:
        raise ValueError(f"Need at least 8 usable chessboard images; found {len(used_images)}.")

    # 3. 运行 OpenCV 鱼眼相机标定算法
    camera_matrix = np.zeros((3, 3), dtype=np.float64)
    distortion = np.zeros((4, 1), dtype=np.float64)
    # 标定参数配置：重新计算外参、固定偏斜角（Skew）为 0
    flags = _fisheye_flag("CALIB_RECOMPUTE_EXTRINSIC") | _fisheye_flag("CALIB_FIX_SKEW")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7)

    rms_px, camera_matrix, distortion, _, _ = cv2.fisheye.calibrate(
        object_points,
        image_points,
        image_size,
        camera_matrix,
        distortion,
        None,
        None,
        flags,
        criteria,
    )
    return FisheyeCalibration(
        camera_matrix=camera_matrix,
        distortion=distortion,
        image_size=image_size,
        rms_px=float(rms_px),
        used_images=tuple(used_images),
        rejected_images=tuple(rejected_images),
        pattern_size=pattern_size,
        square_size_mm=float(square_size_mm),
    )


def save_fisheye_calibration(calibration: FisheyeCalibration, path: Path) -> None:
    """
    将标定出的相机内参和畸变参数序列化保存为规范的 JSON 文件。
    """
    payload = {
        "model": "opencv_fisheye",
        "image_size": {"width": calibration.image_size[0], "height": calibration.image_size[1]},
        "camera_matrix": calibration.camera_matrix.tolist(),
        "distortion_coefficients": calibration.distortion.reshape(-1).tolist(),
        "rms_px": calibration.rms_px,
        "pattern": {
            "inner_corners": list(calibration.pattern_size),
            "square_size_mm": calibration.square_size_mm,
        },
        "used_images": list(calibration.used_images),
        "rejected_images": list(calibration.rejected_images),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_fisheye_calibration(path: Path) -> FisheyeCalibration:
    """
    从 JSON 文件中加载并解析鱼眼相机标定参数。
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model") != "opencv_fisheye":
        raise ValueError(f"Unsupported calibration model in {path}: {payload.get('model')!r}")
    image_size = payload["image_size"]
    pattern = payload["pattern"]
    return FisheyeCalibration(
        camera_matrix=np.asarray(payload["camera_matrix"], dtype=np.float64),
        distortion=np.asarray(payload["distortion_coefficients"], dtype=np.float64).reshape(4, 1),
        image_size=(int(image_size["width"]), int(image_size["height"])),
        rms_px=float(payload["rms_px"]),
        used_images=tuple(payload.get("used_images", [])),
        rejected_images=tuple(payload.get("rejected_images", [])),
        pattern_size=tuple(int(value) for value in pattern["inner_corners"]),
        square_size_mm=float(pattern["square_size_mm"]),
    )


def undistort_fisheye_image(
    image: np.ndarray,
    calibration: FisheyeCalibration,
    balance: float = 0.0,
) -> np.ndarray:
    """
    应用标定参数，对由于鱼眼镜头导致的弯曲图片进行去畸变处理（还原为理想的透视投影直线图像）。

    参数:
        image: 需要校正的畸变图像。
        calibration: 已载入的 FisheyeCalibration 标定参数对象。
        balance: 边缘裁剪与拉伸平衡因子 (0.0 ~ 1.0)。
            - 0.0: 完全裁剪去畸变后的空白边缘，保证画面填充，但视角会稍微收缩。
            - 1.0: 保留所有的原始传感器像素，去畸变后四周会包含大面积黑框。
    """
    if not 0.0 <= balance <= 1.0:
        raise ValueError("balance must be between 0.0 and 1.0.")

    height, width = image.shape[:2]
    source_width, source_height = calibration.image_size
    scale_x, scale_y = width / source_width, height / source_height
    # 校验输入的分辨率宽高比是否与标定时一致
    if abs(scale_x - scale_y) > 1e-6:
        raise ValueError(f"Input aspect ratio does not match calibration size {source_width}x{source_height}.")

    # 1. 对应调整图像尺寸缩放时的内参矩阵
    camera_matrix = calibration.camera_matrix.copy()
    camera_matrix[0, 0] *= scale_x
    camera_matrix[1, 1] *= scale_y
    camera_matrix[0, 2] *= scale_x
    camera_matrix[1, 2] *= scale_y
    size = (width, height)

    # 2. 估计去畸变后的无畸变相机矩阵
    new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        camera_matrix, calibration.distortion, size, np.eye(3), balance=balance
    )

    # 3. 计算畸变坐标到目标去畸变坐标的像素重映射图 (Map)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix, calibration.distortion, np.eye(3), new_camera_matrix, size, cv2.CV_16SC2
    )

    # 4. 执行像素插值映射，输出完美横平竖直的矫正图片
    return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
