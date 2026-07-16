from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from archive.src.vision.measure_basic import draw_basic_measurement_debug, measure_basic_shape
from src.vision.rectify import rectify_a4
from src.vision.target_plane import draw_target_plane_debug, find_target_plane


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure a basic NUEDC C target image.")
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=ROOT / "runtime" / "debug",
        help="Directory for debug images.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = args.image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: cannot read image: {image_path}", file=sys.stderr)
        return 2

    debug_dir = args.debug_dir / image_path.stem
    debug_dir.mkdir(parents=True, exist_ok=True)

    plane = find_target_plane(image)
    plane_debug = draw_target_plane_debug(image, plane)
    rectified = rectify_a4(image, plane.corners)
    measurement = measure_basic_shape(rectified)
    measurement_debug = draw_basic_measurement_debug(rectified, measurement)

    plane_debug_path = debug_dir / "01_plane_corners.jpg"
    rectified_path = debug_dir / "02_rectified.jpg"
    measurement_debug_path = debug_dir / "03_measurement.jpg"
    cv2.imwrite(str(plane_debug_path), plane_debug)
    cv2.imwrite(str(rectified_path), rectified)
    cv2.imwrite(str(measurement_debug_path), measurement_debug)

    print(f"image: {image_path}")
    print(f"shape: {measurement.shape_type}")
    print(f"x_mm: {measurement.x_mm:.1f}")
    print(f"x_cm: {measurement.x_mm / 10.0:.2f}")
    print(f"debug_plane: {plane_debug_path}")
    print(f"debug_rectified: {rectified_path}")
    print(f"debug_measurement: {measurement_debug_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
