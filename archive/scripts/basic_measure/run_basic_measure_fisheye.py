from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.fisheye import load_fisheye_calibration, undistort_fisheye_image
from src.vision.measure_basic import draw_basic_measurement_debug, measure_basic_shape
from src.vision.rectify import rectify_a4
from src.vision.target_plane import draw_target_plane_debug, find_target_plane


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure a basic target after fisheye correction.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--calibration", type=Path, default=ROOT / "runtime" / "calibration" / "fisheye_1280x720.json")
    parser.add_argument("--balance", type=float, default=0.0)
    parser.add_argument("--debug-dir", type=Path, default=ROOT / "runtime" / "debug")
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        print(f"ERROR: cannot read image: {args.image}", file=sys.stderr)
        return 2

    image = undistort_fisheye_image(image, load_fisheye_calibration(args.calibration), args.balance)
    debug_dir = args.debug_dir / args.image.stem
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / "00_undistorted.jpg"), image)

    plane = find_target_plane(image)
    rectified = rectify_a4(image, plane.corners)
    measurement = measure_basic_shape(rectified)
    cv2.imwrite(str(debug_dir / "01_plane_corners.jpg"), draw_target_plane_debug(image, plane))
    cv2.imwrite(str(debug_dir / "02_rectified.jpg"), rectified)
    cv2.imwrite(str(debug_dir / "03_measurement.jpg"), draw_basic_measurement_debug(rectified, measurement))
    print(f"shape: {measurement.shape_type}")
    print(f"x_mm: {measurement.x_mm:.1f}")
    print(f"x_cm: {measurement.x_mm / 10.0:.2f}")
    print(f"debug_dir: {debug_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
