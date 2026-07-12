from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.vision.fisheye import FisheyeCalibration


@dataclass(frozen=True)
class DistanceEstimate:
    """
    距离估计结果数据结构。

    属性:
        raw_distance_mm: 原始 PnP 解算出的目标平面中心距离相机光心的深度距离（Z轴分量，单位：mm）。
        corrected_distance_mm: 经过线性标定方程修正后的真实物理距离（单位：mm），即题目中要求的 D。
        reprojection_error_px: 2D 角点重投影误差（单位：像素），小于 2 像素通常说明姿态估计可靠。
        rvec: PnP 计算得到的 3x1 旋转向量（Rodrigues 格式）。
        tvec: PnP 计算得到的 3x1 平移向量（相对于相机坐标系，单位：mm）。
    """
    raw_distance_mm: float
    corrected_distance_mm: float
    reprojection_error_px: float
    rvec: np.ndarray
    tvec: np.ndarray


@dataclass(frozen=True)
class LinearDistanceCorrection:
    """
    线性测距误差修正模型参数 (D_corrected = D_raw * Scale + Offset)。
    通过实验拟合物理标定曲线以提升测距精度。
    """
    scale: float
    offset_mm: float
    fit_mae_mm: float              # 拟合时的平均绝对误差（mm）
    fit_rmse_mm: float             # 拟合时的均方根误差（mm）
    fit_max_error_mm: float        # 拟合时的最大绝对误差（mm）
    inlier_count: int              # 参与拟合的有效图片样本数
    excluded_images: tuple[str, ...] # 被剔除的异常标定图片列表

    def apply(self, raw_distance_mm: float) -> float:
        """
        将线性标定公式应用到原始测距数据上，获取修正后的准确距离。
        """
        return self.scale * raw_distance_mm + self.offset_mm


def undistorted_camera_matrix(
    calibration: FisheyeCalibration,
    image_size: tuple[int, int],
    balance: float = 0.0,
) -> np.ndarray:
    """
    计算鱼眼去畸变后对应的新相机内参矩阵。
    由于去畸变（undistort）会改变图像像素的投影中心和焦距，必须重新估算对应的理想相机内参矩阵。
    """
    source_width, source_height = calibration.image_size
    width, height = image_size
    scale_x, scale_y = width / source_width, height / source_height
    # 确保图像宽高比与标定尺寸比例一致
    if abs(scale_x - scale_y) > 1e-6:
        raise ValueError("Input aspect ratio does not match the fisheye calibration.")

    # 根据当前图像分辨率缩放原始相机矩阵
    camera_matrix = calibration.camera_matrix.copy()
    camera_matrix[0, 2] *= scale_x
    camera_matrix[1, 2] *= scale_y
    camera_matrix[0, 0] *= scale_x
    camera_matrix[1, 1] *= scale_y
    # 计算并返回最适合去畸变后图像的理想投影相机矩阵
    return cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        camera_matrix,
        calibration.distortion,
        image_size,
        np.eye(3),
        balance=balance,
    )


def estimate_plane_center_depth(
    corners: np.ndarray,
    camera_matrix: np.ndarray,
    correction: LinearDistanceCorrection | None = None,
    a4_width_mm: float = 210.0,
    a4_height_mm: float = 297.0,
) -> DistanceEstimate:
    """
    使用 OpenCV 的 solvePnP 解算 A4 纸面在相机坐标系下的三维姿态，计算中心深度（距离 D）。

    参数:
        corners: 2D 像素坐标系下 A4 纸的 4 个角点 [TL, TR, BR, BL]。
        camera_matrix: 理想（去畸变后）的相机内参矩阵。
        correction: 线性误差修正参数，为 None 时不执行修正。
        a4_width_mm: A4 纸的真实物理宽度（210 mm）。
        a4_height_mm: A4 纸的真实物理高度（297 mm）。
    """
    image_points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    # 定义 A4 纸在自身三维局部坐标系下的世界坐标点（Z轴为 0 的平面）
    object_points = np.array(
        [
            [0.0, 0.0, 0.0],                     # 1. TL (0, 0, 0)
            [a4_width_mm, 0.0, 0.0],              # 2. TR (W, 0, 0)
            [a4_width_mm, a4_height_mm, 0.0],     # 3. BR (W, H, 0)
            [0.0, a4_height_mm, 0.0],             # 4. BL (0, H, 0)
        ],
        dtype=np.float64,
    )
    # 调用 PnP 解算：求世界坐标 object_points 到像素坐标 image_points 的外参映射旋转与平移向量
    # 这里使用经典的 SOLVEPNP_IPPE（针对平面四点姿态估计算法，计算速度极快且在小倾角下最稳定）
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        None,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not ok:
        raise ValueError("PnP pose estimation failed.")

    # 1. 将旋转向量转换成 3x3 的旋转矩阵 R
    rotation, _ = cv2.Rodrigues(rvec)

    # 2. 计算 A4 平面中心在局部坐标系下的 3D 坐标：(W/2, H/2, 0)
    plane_center = np.array([a4_width_mm / 2.0, a4_height_mm / 2.0, 0.0])

    # 3. 映射到相机三维坐标系下：P_camera = R * P_world + T
    center_camera = rotation @ plane_center + tvec.reshape(3)

    # 获取相机光轴方向的垂直深度距离 Z 轴坐标分量（单位：mm）
    raw_distance_mm = float(center_camera[2])
    if raw_distance_mm <= 0:
        raise ValueError("PnP produced a target plane behind the camera.")

    # 4. 计算 2D 像素重投影误差以评估解算的可信度
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
    residual = projected.reshape(-1, 2) - image_points
    reprojection_error_px = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))

    # 5. 应用误差修正公式
    corrected = correction.apply(raw_distance_mm) if correction else raw_distance_mm

    return DistanceEstimate(
        raw_distance_mm=raw_distance_mm,
        corrected_distance_mm=corrected,
        reprojection_error_px=reprojection_error_px,
        rvec=rvec,
        tvec=tvec,
    )


def save_distance_correction(correction: LinearDistanceCorrection, path: Path) -> None:
    """
    将拟合好的线性距离修正参数保存到 JSON 文件中，供实时调用。
    """
    payload = {
        "model": "linear_pnp_plane_center_depth",
        "scale": correction.scale,
        "offset_mm": correction.offset_mm,
        "fit_mae_mm": correction.fit_mae_mm,
        "fit_rmse_mm": correction.fit_rmse_mm,
        "fit_max_error_mm": correction.fit_max_error_mm,
        "inlier_count": correction.inlier_count,
        "excluded_images": list(correction.excluded_images),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_distance_correction(path: Path) -> LinearDistanceCorrection:
    """
    从 JSON 文件中读取线性距离修正模型参数。
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model") != "linear_pnp_plane_center_depth":
        raise ValueError(f"Unsupported distance model in {path}: {payload.get('model')!r}")
    return LinearDistanceCorrection(
        scale=float(payload["scale"]),
        offset_mm=float(payload["offset_mm"]),
        fit_mae_mm=float(payload["fit_mae_mm"]),
        fit_rmse_mm=float(payload["fit_rmse_mm"]),
        fit_max_error_mm=float(payload["fit_max_error_mm"]),
        inlier_count=int(payload["inlier_count"]),
        excluded_images=tuple(payload.get("excluded_images", [])),
    )
