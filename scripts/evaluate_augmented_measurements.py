from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.measurement_tasks import measure_normal, measure_numbered


def main() -> int:
    parser = argparse.ArgumentParser(description="评估扰动 PNG 上的普通与数字测量任务。")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tolerance-mm", type=float, default=1.0)
    args = parser.parse_args()

    normal_total = normal_ok = numbered_total = numbered_ok = 0
    for image_path in sorted(args.directory.glob("*.png")):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"cannot read {image_path}")

        # 先检测 PNG，再读取 JSON 评分，防止真值泄漏进测量流程。
        normal = measure_normal(image)
        truth = json.loads(image_path.with_suffix(".json").read_text(encoding="utf-8"))
        key = "squares" if "squares" in truth else "shapes"
        expected = min(float(item["size_mm"]) for item in truth[key])
        normal_error = abs(normal.x_mm - expected)
        normal_passed = normal_error <= args.tolerance_mm
        normal_total += 1
        normal_ok += int(normal_passed)
        print(
            f"{'PASS' if normal_passed else 'FAIL'} normal {image_path.name} "
            f"error={normal_error:.3f}mm"
        )

        if truth.get("target_type") != "numbered_squares":
            continue
        for item in truth[key]:
            result = measure_numbered(image, int(item["digit"]))
            error = abs(result.x_mm - float(item["size_mm"]))
            passed = error <= args.tolerance_mm
            numbered_total += 1
            numbered_ok += int(passed)
            print(
                f"{'PASS' if passed else 'FAIL'} numbered {image_path.name} "
                f"digit={item['digit']} error={error:.3f}mm"
            )

    print(
        f"SUMMARY normal={normal_ok}/{normal_total} "
        f"numbered={numbered_ok}/{numbered_total}"
    )
    return int(normal_ok != normal_total or numbered_ok != numbered_total)


if __name__ == "__main__":
    raise SystemExit(main())
