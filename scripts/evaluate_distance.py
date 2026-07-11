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

from src.vision.distance import estimate_plane_center_depth, load_distance_correction, undistorted_camera_matrix
from src.vision.fisheye import load_fisheye_calibration, undistort_fisheye_image
from src.vision.target_plane import find_target_plane


DISTANCE_PATTERN = re.compile(r"^d(?P<distance_mm>\d+)mm_")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate corrected PnP distances.")
    parser.add_argument("input_dir", nargs="?", type=Path, default=ROOT / "runtime" / "captures" / "rpi_distance")
    parser.add_argument("--fisheye-calibration", type=Path, default=ROOT / "runtime" / "calibration" / "fisheye_1280x720.json")
    parser.add_argument("--distance-calibration", type=Path, default=ROOT / "runtime" / "calibration" / "distance_linear.json")
    parser.add_argument("--report", type=Path, default=ROOT / "runtime" / "debug" / "distance_evaluation.csv")
    parser.add_argument("--strip-fraction", type=float, default=1.0 / 3.0)
    args = parser.parse_args()

    fisheye = load_fisheye_calibration(args.fisheye_calibration)
    correction = load_distance_correction(args.distance_calibration)
    camera_matrix = undistorted_camera_matrix(fisheye, fisheye.image_size)
    rows: list[dict[str, object]] = []
    for path in sorted(args.input_dir.glob("*.jpg")):
        match = DISTANCE_PATTERN.match(path.name)
        if not match:
            continue
        truth_mm = int(match.group("distance_mm"))
        row: dict[str, object] = {"image": path.name, "truth_mm": truth_mm, "raw_mm": "", "corrected_mm": "", "error_mm": "", "reprojection_px": "", "status": "FAIL"}
        try:
            raw_image = cv2.imread(str(path))
            if raw_image is None:
                raise ValueError("cannot read image")
            undistorted = undistort_fisheye_image(raw_image, fisheye)
            height, width = undistorted.shape[:2]
            strip_width = round(width * args.strip_fraction)
            left = (width - strip_width) // 2
            plane = find_target_plane(undistorted[:, left : left + strip_width])
            corners = plane.corners.astype(np.float64)
            corners[:, 0] += left
            estimate = estimate_plane_center_depth(corners, camera_matrix, correction)
            error_mm = estimate.corrected_distance_mm - truth_mm
            row.update(raw_mm=f"{estimate.raw_distance_mm:.3f}", corrected_mm=f"{estimate.corrected_distance_mm:.3f}", error_mm=f"{error_mm:.3f}", reprojection_px=f"{estimate.reprojection_error_px:.4f}", status="PASS" if abs(error_mm) <= 50.0 else "FAIL")
        except Exception as exc:
            row["error_mm"] = str(exc)
        rows.append(row)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=("image", "truth_mm", "raw_mm", "corrected_mm", "error_mm", "reprojection_px", "status"))
        writer.writeheader()
        writer.writerows(rows)
    numeric_errors = np.array([float(row["error_mm"]) for row in rows if row["status"] in {"PASS", "FAIL"} and isinstance(row["error_mm"], str) and re.fullmatch(r"-?\d+(?:\.\d+)?", row["error_mm"])])
    passed = sum(row["status"] == "PASS" for row in rows)
    print(f"summary: {passed}/{len(rows)} within 50 mm")
    if numeric_errors.size:
        print(f"mae_mm: {np.mean(np.abs(numeric_errors)):.3f}")
        print(f"max_error_mm: {np.max(np.abs(numeric_errors)):.3f}")
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
