from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.fisheye import calibrate_fisheye, save_fisheye_calibration


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate an OpenCV fisheye model from chessboard images.")
    parser.add_argument("image_dir", nargs="?", type=Path, default=ROOT / "runtime" / "captures" / "eye_distortion")
    parser.add_argument("--output", type=Path, default=ROOT / "runtime" / "calibration" / "fisheye_1280x720.json")
    args = parser.parse_args()
    image_paths = sorted(path for path in args.image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    calibration = calibrate_fisheye(image_paths)
    save_fisheye_calibration(calibration, args.output)
    print(f"output: {args.output}")
    print(f"used_images: {len(calibration.used_images)}")
    print(f"rejected_images: {len(calibration.rejected_images)}")
    print(f"rms_px: {calibration.rms_px:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
