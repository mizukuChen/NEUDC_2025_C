from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from archive.src.vision.measure_basic import measure_basic_shape
from src.vision.rectify import rectify_a4
from src.vision.target_plane import find_target_plane


@dataclass(frozen=True)
class ExpectedCase:
    keyword: str
    shape_type: str


EXPECTED_CASES = (
    ExpectedCase("circle", "circle"),
    ExpectedCase("square", "square"),
    ExpectedCase("triangle", "triangle"),
)


def expected_shape(path: Path) -> str | None:
    name = path.stem.lower()
    for case in EXPECTED_CASES:
        if case.keyword in name:
            return case.shape_type
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate basic target measurements for a folder.")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=ROOT / "runtime" / "captures" / "phone",
        help="Image file or directory to evaluate.",
    )
    parser.add_argument("--min-mm", type=float, default=110.0)
    parser.add_argument("--max-mm", type=float, default=130.0)
    return parser.parse_args()


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    patterns = ("*.jpg", "*.jpeg", "*.png")
    images: list[Path] = []
    for pattern in patterns:
        images.extend(path.glob(pattern))
    return sorted(images)


def measure_image(path: Path) -> tuple[str, float]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("cannot read image")
    plane = find_target_plane(image)
    rectified = rectify_a4(image, plane.corners)
    measurement = measure_basic_shape(rectified)
    return measurement.shape_type, measurement.x_mm


def main() -> int:
    args = parse_args()
    images = iter_images(args.input)
    if not images:
        print(f"No images found: {args.input}", file=sys.stderr)
        return 2

    failed = 0
    for path in images:
        expected = expected_shape(path)
        try:
            shape_type, x_mm = measure_image(path)
            shape_ok = expected is None or expected == shape_type
            size_ok = args.min_mm <= x_mm <= args.max_mm
            ok = shape_ok and size_ok
            status = "PASS" if ok else "FAIL"
            if not ok:
                failed += 1
            expected_text = expected if expected is not None else "-"
            print(f"{status} {path.name:28s} expected={expected_text:8s} shape={shape_type:8s} x={x_mm:6.1f} mm")
        except Exception as exc:
            failed += 1
            expected_text = expected if expected is not None else "-"
            print(f"FAIL {path.name:28s} expected={expected_text:8s} error={exc}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
