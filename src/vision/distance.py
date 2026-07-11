from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.vision.fisheye import FisheyeCalibration


@dataclass(frozen=True)
class DistanceEstimate:
    raw_distance_mm: float
    corrected_distance_mm: float
    reprojection_error_px: float
    rvec: np.ndarray
    tvec: np.ndarray


@dataclass(frozen=True)
class LinearDistanceCorrection:
    scale: float
    offset_mm: float
    fit_mae_mm: float
    fit_rmse_mm: float
    fit_max_error_mm: float
    inlier_count: int
    excluded_images: tuple[str, ...]

    def apply(self, raw_distance_mm: float) -> float:
        return self.scale * raw_distance_mm + self.offset_mm


def undistorted_camera_matrix(
    calibration: FisheyeCalibration,
    image_size: tuple[int, int],
    balance: float = 0.0,
) -> np.ndarray:
    source_width, source_height = calibration.image_size
    width, height = image_size
    scale_x, scale_y = width / source_width, height / source_height
    if abs(scale_x - scale_y) > 1e-6:
        raise ValueError("Input aspect ratio does not match the fisheye calibration.")

    camera_matrix = calibration.camera_matrix.copy()
    camera_matrix[0, 0] *= scale_x
    camera_matrix[1, 1] *= scale_y
    camera_matrix[0, 2] *= scale_x
    camera_matrix[1, 2] *= scale_y
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
    image_points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    object_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [a4_width_mm, 0.0, 0.0],
            [a4_width_mm, a4_height_mm, 0.0],
            [0.0, a4_height_mm, 0.0],
        ],
        dtype=np.float64,
    )
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        None,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not ok:
        raise ValueError("PnP pose estimation failed.")

    rotation, _ = cv2.Rodrigues(rvec)
    plane_center = np.array([a4_width_mm / 2.0, a4_height_mm / 2.0, 0.0])
    center_camera = rotation @ plane_center + tvec.reshape(3)
    raw_distance_mm = float(center_camera[2])
    if raw_distance_mm <= 0:
        raise ValueError("PnP produced a target plane behind the camera.")

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
    residual = projected.reshape(-1, 2) - image_points
    reprojection_error_px = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    corrected = correction.apply(raw_distance_mm) if correction else raw_distance_mm
    return DistanceEstimate(
        raw_distance_mm=raw_distance_mm,
        corrected_distance_mm=corrected,
        reprojection_error_px=reprojection_error_px,
        rvec=rvec,
        tvec=tvec,
    )


def save_distance_correction(correction: LinearDistanceCorrection, path: Path) -> None:
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
