from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.fisheye import load_fisheye_calibration, undistort_fisheye_image


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove fisheye distortion from one image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--calibration", type=Path, default=ROOT / "runtime" / "calibration" / "fisheye_1280x720.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--balance", type=float, default=0.0)
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        print(f"ERROR: cannot read image: {args.image}", file=sys.stderr)
        return 2
    corrected = undistort_fisheye_image(image, load_fisheye_calibration(args.calibration), args.balance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), corrected):
        print(f"ERROR: cannot write image: {args.output}", file=sys.stderr)
        return 3
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
