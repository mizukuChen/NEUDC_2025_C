from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.fisheye import load_fisheye_calibration, undistort_fisheye_image
from archive.src.vision.measure_basic import draw_basic_measurement_debug, measure_basic_shape
from src.vision.rectify import rectify_a4
from src.vision.target_plane import draw_target_plane_debug, find_target_plane


def expected_shape(path: Path) -> str | None:
    for shape in ("circle", "square", "triangle"):
        if shape in path.stem.lower():
            return shape
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-evaluate basic targets after fisheye correction.")
    parser.add_argument("input_dir", nargs="?", type=Path, default=ROOT / "runtime" / "captures" / "rpi")
    parser.add_argument("--calibration", type=Path, default=ROOT / "runtime" / "calibration" / "fisheye_1280x720.json")
    parser.add_argument("--debug-dir", type=Path, default=ROOT / "runtime" / "debug" / "rpi_fisheye_measurements")
    parser.add_argument("--report", type=Path, default=ROOT / "runtime" / "debug" / "rpi_fisheye_measurements.csv")
    parser.add_argument("--balance", type=float, default=0.0)
    parser.add_argument("--min-mm", type=float, default=110.0)
    parser.add_argument("--max-mm", type=float, default=130.0)
    args = parser.parse_args()

    paths = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        print(f"ERROR: no images found in {args.input_dir}", file=sys.stderr)
        return 2

    calibration = load_fisheye_calibration(args.calibration)
    args.debug_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for path in paths:
        expected = expected_shape(path)
        row = {"image": path.name, "expected": expected or "", "shape": "", "x_mm": "", "status": "FAIL", "error": ""}
        try:
            raw = cv2.imread(str(path))
            if raw is None:
                raise ValueError("cannot read image")
            image = undistort_fisheye_image(raw, calibration, args.balance)
            plane = find_target_plane(image)
            rectified = rectify_a4(image, plane.corners)
            measurement = measure_basic_shape(rectified)
            debug_dir = args.debug_dir / path.stem
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / "00_undistorted.jpg"), image)
            cv2.imwrite(str(debug_dir / "01_plane_corners.jpg"), draw_target_plane_debug(image, plane))
            cv2.imwrite(str(debug_dir / "02_rectified.jpg"), rectified)
            cv2.imwrite(str(debug_dir / "03_measurement.jpg"), draw_basic_measurement_debug(rectified, measurement))
            shape_ok = expected is None or measurement.shape_type == expected
            size_ok = args.min_mm <= measurement.x_mm <= args.max_mm
            row.update(shape=measurement.shape_type, x_mm=f"{measurement.x_mm:.2f}", status="PASS" if shape_ok and size_ok else "FAIL")
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)
        print(f"{row['status']:4s} {row['image']:28s} expected={row['expected'] or '-':8s} shape={row['shape'] or '-':8s} x={row['x_mm'] or '-':>7s} {row['error']}")

    with args.report.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=("image", "expected", "shape", "x_mm", "status", "error"))
        writer.writeheader()
        writer.writerows(rows)
    passed = sum(row["status"] == "PASS" for row in rows)
    print(f"summary: {passed}/{len(rows)} passed")
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
