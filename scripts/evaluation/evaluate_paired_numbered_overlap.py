from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import permutations
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.measure import measure
from src.vision.measurement_tasks import measure_numbered


DEFAULT_SOURCE = ROOT / "assets/targets/generated/paired_numbered_overlap"


def angle_error(first: float, second: float) -> float:
    return abs((first - second + 45.0) % 90.0 - 45.0)


def geometry_passes(image, truth: list[dict]) -> bool:
    try:
        actual = list(measure(image).shapes)
    except (RuntimeError, ValueError):
        return False
    if len(actual) != len(truth) or not actual:
        return False
    ordering = min(
        permutations(range(len(truth))),
        key=lambda order: sum(
            abs(actual[index].x_mm - float(truth[target]["size_mm"]))
            + math.hypot(
                actual[index].center_mm[0] - float(truth[target]["cx_mm"]),
                actual[index].center_mm[1] - float(truth[target]["cy_mm"]),
            )
            + angle_error(actual[index].angle_deg, float(truth[target]["angle_deg"]))
            for index, target in enumerate(order)
        ),
    )
    return all(
        abs(actual[index].x_mm - float(truth[target]["size_mm"])) <= 1.0
        and math.hypot(
            actual[index].center_mm[0] - float(truth[target]["cx_mm"]),
            actual[index].center_mm[1] - float(truth[target]["cy_mm"]),
        ) <= 1.0
        and angle_error(actual[index].angle_deg, float(truth[target]["angle_deg"])) <= 0.6
        for index, target in enumerate(ordering)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="成对评估数字对重叠几何和数字匹配的影响。")
    parser.add_argument("source", type=Path, nargs="?", default=DEFAULT_SOURCE)
    args = parser.parse_args()

    pair_total = plain_ok = numbered_ok = digit_total = digit_ok = 0
    digit_failures = []
    plain_only = []
    for metadata_path in sorted(args.source.glob("pair_*.json")):
        truth = json.loads(metadata_path.read_text(encoding="utf-8"))["squares"]
        stem = metadata_path.stem
        plain = cv2.imread(str(args.source / "plain" / f"{stem}.png"))
        numbered = cv2.imread(str(args.source / "numbered" / f"{stem}.png"))
        if plain is None or numbered is None:
            raise RuntimeError(f"missing image for {stem}")

        plain_passed = geometry_passes(plain, truth)
        numbered_passed = geometry_passes(numbered, truth)
        pair_total += 1
        plain_ok += int(plain_passed)
        numbered_ok += int(numbered_passed)
        if plain_passed and not numbered_passed:
            plain_only.append(stem)

        for item in truth:
            digit_total += 1
            try:
                result = measure_numbered(numbered, int(item["digit"]))
                center = result.selected_square.geometry.center_mm
                passed = (
                    abs(result.x_mm - float(item["size_mm"])) <= 1.0
                    and math.hypot(
                        center[0] - float(item["cx_mm"]),
                        center[1] - float(item["cy_mm"]),
                    ) <= 1.0
                )
            except (RuntimeError, ValueError):
                passed = False
            digit_ok += int(passed)
            if not passed:
                digit_failures.append(f"{stem}:digit={item['digit']}")

    print(f"geometry plain={plain_ok}/{pair_total} numbered={numbered_ok}/{pair_total}")
    print(f"digit selection={digit_ok}/{digit_total}")
    print(f"plain-pass/numbered-fail={len(plain_only)} {plain_only}")
    print(f"digit failures={len(digit_failures)} {digit_failures}")
    return int(numbered_ok != pair_total or digit_ok != digit_total)


if __name__ == "__main__":
    raise SystemExit(main())
