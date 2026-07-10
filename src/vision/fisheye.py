from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class FisheyeCalibration:
    camera_matrix: np.ndarray
    distortion: np.ndarray
    image_size: tuple[int, int]
    rms_px: float
    used_images: tuple[str, ...]
    rejected_images: tuple[str, ...]
    pattern_size: tuple[int, int]
    square_size_mm: float


def _fisheye_flag(name: str) -> int:
    """Support OpenCV 4 and the OpenCV 5 package used by this project."""
    value = getattr(cv2.fisheye, name, None)
    if value is None:
        value = getattr(cv2, name, None)
    if value is None:
        raise RuntimeError(f"OpenCV does not expose the fisheye flag {name}.")
    return int(value)


def _find_corners(gray: np.ndarray, pattern_size: tuple[int, int]) -> np.ndarray | None:
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags)
        if found:
            return corners.reshape(1, -1, 2).astype(np.float64)

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
    """Calibrate an OpenCV fisheye model from chessboard photos."""
    paths = [Path(path) for path in image_paths]
    if not paths:
        raise ValueError("No calibration images were provided.")

    corner_count = pattern_size[0] * pattern_size[1]
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

    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            rejected_images.append(path.name)
            continue
        size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = size
        if size != image_size:
            rejected_images.append(path.name)
            continue

        corners = _find_corners(gray, pattern_size)
        if corners is None:
            rejected_images.append(path.name)
            continue
        object_points.append(object_template.copy())
        image_points.append(corners)
        used_images.append(path.name)

    if image_size is None or len(used_images) < 8:
        raise ValueError(f"Need at least 8 usable chessboard images; found {len(used_images)}.")

    camera_matrix = np.zeros((3, 3), dtype=np.float64)
    distortion = np.zeros((4, 1), dtype=np.float64)
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
    """Remove fisheye distortion before target-plane detection."""
    if not 0.0 <= balance <= 1.0:
        raise ValueError("balance must be between 0.0 and 1.0.")

    height, width = image.shape[:2]
    source_width, source_height = calibration.image_size
    scale_x, scale_y = width / source_width, height / source_height
    if abs(scale_x - scale_y) > 1e-6:
        raise ValueError(f"Input aspect ratio does not match calibration size {source_width}x{source_height}.")

    camera_matrix = calibration.camera_matrix.copy()
    camera_matrix[0, 0] *= scale_x
    camera_matrix[1, 1] *= scale_y
    camera_matrix[0, 2] *= scale_x
    camera_matrix[1, 2] *= scale_y
    size = (width, height)
    new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        camera_matrix, calibration.distortion, size, np.eye(3), balance=balance
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix, calibration.distortion, np.eye(3), new_camera_matrix, size, cv2.CV_16SC2
    )
    return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
