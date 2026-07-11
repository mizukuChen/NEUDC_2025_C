from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.distance import estimate_plane_center_depth, load_distance_correction, undistorted_camera_matrix
from src.vision.fisheye import load_fisheye_calibration, undistort_fisheye_image
from src.vision.target_plane import draw_target_plane_debug, find_target_plane


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure optical-axis distance to the A4 target plane.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--fisheye-calibration", type=Path, default=ROOT / "runtime" / "calibration" / "fisheye_1280x720.json")
    parser.add_argument("--distance-calibration", type=Path, default=ROOT / "runtime" / "calibration" / "distance_linear.json")
    parser.add_argument("--strip-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--max-reprojection-px", type=float, default=2.0)
    parser.add_argument("--debug-dir", type=Path, default=ROOT / "runtime" / "debug" / "distance")
    args = parser.parse_args()

    raw = cv2.imread(str(args.image))
    if raw is None:
        print(f"ERROR: cannot read image: {args.image}", file=sys.stderr)
        return 2
    fisheye = load_fisheye_calibration(args.fisheye_calibration)
    correction = load_distance_correction(args.distance_calibration)
    undistorted = undistort_fisheye_image(raw, fisheye)
    height, width = undistorted.shape[:2]
    strip_width = round(width * args.strip_fraction)
    left = (width - strip_width) // 2
    strip = undistorted[:, left : left + strip_width]
    plane = find_target_plane(strip)
    full_corners = plane.corners.astype(np.float64)
    full_corners[:, 0] += left
    camera_matrix = undistorted_camera_matrix(fisheye, (width, height))
    estimate = estimate_plane_center_depth(full_corners, camera_matrix, correction)
    if estimate.reprojection_error_px > args.max_reprojection_px:
        print(
            f"ERROR: reprojection error {estimate.reprojection_error_px:.3f} px exceeds "
            f"{args.max_reprojection_px:.3f} px",
            file=sys.stderr,
        )
        return 3

    debug_dir = args.debug_dir / args.image.stem
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / "00_undistorted.jpg"), undistorted)
    cv2.imwrite(str(debug_dir / "00_center_strip.jpg"), strip)
    cv2.imwrite(str(debug_dir / "01_plane_corners.jpg"), draw_target_plane_debug(strip, plane))
    print(f"image: {args.image}")
    print(f"D_mm: {estimate.corrected_distance_mm:.1f}")
    print(f"D_cm: {estimate.corrected_distance_mm / 10.0:.2f}")
    print(f"raw_pnp_mm: {estimate.raw_distance_mm:.1f}")
    print(f"reprojection_px: {estimate.reprojection_error_px:.3f}")
    print(f"debug_dir: {debug_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
