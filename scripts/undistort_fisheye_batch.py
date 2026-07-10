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
    parser = argparse.ArgumentParser(description="Batch-remove fisheye distortion from calibration images.")
    parser.add_argument("input_dir", nargs="?", type=Path, default=ROOT / "runtime" / "captures" / "eye_distortion")
    parser.add_argument("--calibration", type=Path, default=ROOT / "runtime" / "calibration" / "fisheye_1280x720.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runtime" / "debug" / "eye_distortion_undistorted")
    parser.add_argument("--balance", type=float, default=0.0)
    args = parser.parse_args()

    paths = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        print(f"ERROR: no images found in {args.input_dir}", file=sys.stderr)
        return 2

    calibration = load_fisheye_calibration(args.calibration)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"SKIP: cannot read {path}", file=sys.stderr)
            continue
        corrected = undistort_fisheye_image(image, calibration, args.balance)
        output_path = args.output_dir / path.name
        if not cv2.imwrite(str(output_path), corrected):
            print(f"SKIP: cannot write {output_path}", file=sys.stderr)
            continue
        written += 1
    print(f"output_dir: {args.output_dir}")
    print(f"written: {written}/{len(paths)}")
    return 0 if written == len(paths) else 3


if __name__ == "__main__":
    raise SystemExit(main())
