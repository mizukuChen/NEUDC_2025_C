from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.distance import (
    LinearDistanceCorrection,
    estimate_plane_center_depth,
    save_distance_correction,
    undistorted_camera_matrix,
)
from src.vision.fisheye import load_fisheye_calibration, undistort_fisheye_image
from src.vision.target_plane import find_target_plane


DISTANCE_PATTERN = re.compile(r"^d(?P<distance_mm>\d+)mm_")


def robust_linear_fit(raw: np.ndarray, truth: np.ndarray, threshold_mm: float) -> tuple[np.ndarray, np.ndarray]:
    slopes = []
    for i in range(len(raw)):
        for j in range(i + 1, len(raw)):
            delta = raw[j] - raw[i]
            if abs(delta) > 1e-6:
                slopes.append((truth[j] - truth[i]) / delta)
    if not slopes:
        raise ValueError("Distance samples do not span enough range for a linear fit.")
    robust_scale = float(np.median(slopes))
    robust_offset = float(np.median(truth - robust_scale * raw))
    inliers = np.abs(robust_scale * raw + robust_offset - truth) <= threshold_mm
    if int(inliers.sum()) < 5:
        raise ValueError("Too few inlier samples remain after robust filtering.")
    return np.polyfit(raw[inliers], truth[inliers], 1), inliers


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate PnP optical-axis distance correction.")
    parser.add_argument("input_dir", nargs="?", type=Path, default=ROOT / "runtime" / "captures" / "rpi_distance")
    parser.add_argument("--fisheye-calibration", type=Path, default=ROOT / "runtime" / "calibration" / "fisheye_1280x720.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runtime" / "calibration" / "distance_linear.json")
    parser.add_argument("--report", type=Path, default=ROOT / "runtime" / "debug" / "distance_calibration.csv")
    parser.add_argument("--strip-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--outlier-mm", type=float, default=50.0)
    parser.add_argument("--max-reprojection-px", type=float, default=2.0)
    args = parser.parse_args()

    calibration = load_fisheye_calibration(args.fisheye_calibration)
    camera_matrix = undistorted_camera_matrix(calibration, calibration.image_size)
    rows: list[dict[str, object]] = []
    for path in sorted(args.input_dir.glob("*.jpg")):
        match = DISTANCE_PATTERN.match(path.name)
        if not match:
            continue
        row: dict[str, object] = {"image": path.name, "truth_mm": int(match.group("distance_mm")), "raw_mm": "", "reprojection_px": "", "inlier": False, "error": ""}
        try:
            raw_image = cv2.imread(str(path))
            if raw_image is None:
                raise ValueError("cannot read image")
            undistorted = undistort_fisheye_image(raw_image, calibration)
            height, width = undistorted.shape[:2]
            strip_width = round(width * args.strip_fraction)
            left = (width - strip_width) // 2
            strip = undistorted[:, left : left + strip_width]
            plane = find_target_plane(strip)
            corners = plane.corners.astype(np.float64)
            corners[:, 0] += left
            estimate = estimate_plane_center_depth(corners, camera_matrix)
            if estimate.reprojection_error_px > args.max_reprojection_px:
                raise ValueError(f"reprojection error {estimate.reprojection_error_px:.3f} px exceeds limit")
            row["raw_mm"] = estimate.raw_distance_mm
            row["reprojection_px"] = estimate.reprojection_error_px
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)

    usable_indices = [index for index, row in enumerate(rows) if row["raw_mm"] != ""]
    if len(usable_indices) < 5:
        print("ERROR: fewer than five usable distance samples", file=sys.stderr)
        return 2
    raw = np.array([float(rows[index]["raw_mm"]) for index in usable_indices])
    truth = np.array([float(rows[index]["truth_mm"]) for index in usable_indices])
    coefficients, inliers = robust_linear_fit(raw, truth, args.outlier_mm)
    corrected = np.polyval(coefficients, raw[inliers])
    residual = corrected - truth[inliers]
    excluded: list[str] = []
    for local_index, row_index in enumerate(usable_indices):
        rows[row_index]["inlier"] = bool(inliers[local_index])
        if not inliers[local_index]:
            excluded.append(str(rows[row_index]["image"]))

    correction = LinearDistanceCorrection(
        scale=float(coefficients[0]),
        offset_mm=float(coefficients[1]),
        fit_mae_mm=float(np.mean(np.abs(residual))),
        fit_rmse_mm=float(np.sqrt(np.mean(residual * residual))),
        fit_max_error_mm=float(np.max(np.abs(residual))),
        inlier_count=int(inliers.sum()),
        excluded_images=tuple(excluded),
    )
    save_distance_correction(correction, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=("image", "truth_mm", "raw_mm", "reprojection_px", "inlier", "error"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"output: {args.output}")
    print(f"samples: {correction.inlier_count}/{len(rows)} inliers")
    print(f"scale: {correction.scale:.9f}")
    print(f"offset_mm: {correction.offset_mm:.4f}")
    print(f"fit_mae_mm: {correction.fit_mae_mm:.3f}")
    print(f"fit_max_error_mm: {correction.fit_max_error_mm:.3f}")
    print(f"excluded: {', '.join(correction.excluded_images) or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
