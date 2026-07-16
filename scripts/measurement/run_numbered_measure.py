from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.measurement_tasks import measure_numbered


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure the square selected by an input digit.")
    parser.add_argument("image", type=Path, help="已经完成透视拉平的 A4 图像。")
    parser.add_argument("digit", type=int, choices=range(10), help="IO 输入的目标数字编号 0~9。")
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        print(f"ERROR: cannot read image: {args.image}", file=sys.stderr)
        return 2
    result = measure_numbered(image, args.digit)
    print(f"requested_digit: {result.requested_digit}")
    print(f"recognized_squares: {len(result.squares)}")
    for index, square in enumerate(result.squares, start=1):
        print(
            f"square[{index}]: digit={square.digit} digit_confidence={square.digit_confidence:.3f} "
            f"x_mm={square.geometry.x_mm:.2f} center_mm={square.geometry.center_mm}"
        )
    print(f"selected_x_mm: {result.x_mm:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
