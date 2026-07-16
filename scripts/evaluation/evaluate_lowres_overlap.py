from __future__ import annotations

import argparse
import json
import sys
from itertools import permutations
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.overlap.lowres import (
    draw_low_resolution_squares,
    reconstruct_low_resolution_squares,
)
from src.vision.recognize_digits import HogSvmDigitRecognizer


DEFAULT_INPUT = ROOT / "runtime" / "debug" / "lowres_overlap_dataset"
DEFAULT_MODEL = ROOT / "assets" / "models" / "digit_hog_linear_svm.npz"


def _angle_error(first: float, second: float) -> float:
    return abs((first - second + 45.0) % 90.0 - 45.0)


def _match(actual, expected: list[dict]) -> list[tuple[int, int, float, float, float]]:
    if len(actual) != len(expected):
        return []
    ordering = min(
        permutations(range(len(expected))),
        key=lambda order: sum(
            abs(actual[index].side_mm - expected[target]["size_mm"])
            + np.hypot(
                actual[index].center_mm[0] - expected[target]["cx_mm"],
                actual[index].center_mm[1] - expected[target]["cy_mm"],
            )
            + _angle_error(actual[index].angle_deg, expected[target]["angle_deg"])
            for index, target in enumerate(order)
        ),
    )
    return [
        (
            index,
            target,
            abs(actual[index].side_mm - expected[target]["size_mm"]),
            float(
                np.hypot(
                    actual[index].center_mm[0] - expected[target]["cx_mm"],
                    actual[index].center_mm[1] - expected[target]["cy_mm"],
                )
            ),
            _angle_error(actual[index].angle_deg, expected[target]["angle_deg"]),
        )
        for index, target in enumerate(ordering)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="评估低分辨率LSD重叠正方形检测。")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--debug-dir", type=Path, default=ROOT / "runtime" / "debug" / "lowres_overlap_eval")
    parser.add_argument("--size-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--center-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--angle-tolerance-deg", type=float, default=5.0)
    args = parser.parse_args()

    recognizer = HogSvmDigitRecognizer.load(args.model)
    args.debug_dir.mkdir(parents=True, exist_ok=True)
    totals: dict[tuple[str, int], list[int]] = {}
    geometry_pass = 0
    sample_count = 0
    digit_correct = 0
    digit_total = 0
    errors: list[tuple[float, float, float]] = []

    for metadata_path in sorted(args.input.glob("*/*.json")):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        frames = [cv2.imread(str(metadata_path.parent / name), cv2.IMREAD_GRAYSCALE) for name in payload["frames"]]
        if any(frame is None for frame in frames):
            print(f"FAIL {metadata_path.name}: cannot read all frames")
            return 2
        result = reconstruct_low_resolution_squares(frames)
        expected = payload["squares"]
        matches = _match(result.squares, expected)
        ok = bool(matches) and all(
            size <= args.size_tolerance_mm
            and center <= args.center_tolerance_mm
            and angle <= args.angle_tolerance_deg
            for _, _, size, center, angle in matches
        )
        errors.extend((size, center, angle) for _, _, size, center, angle in matches)
        sample_count += 1
        geometry_pass += int(ok)
        key = (payload["family"], int(payload["native_a4_width_px"]))
        bucket = totals.setdefault(key, [0, 0])
        bucket[0] += int(ok)
        bucket[1] += 1

        if payload["family"] == "numbered_overlap" and matches:
            for actual_index, truth_index, *_ in matches:
                square = result.squares[actual_index]
                recognition = recognizer.recognize_square(
                    result.prepared.image,
                    (
                        square.center_mm[0] * result.prepared.pixels_per_mm,
                        square.center_mm[1] * result.prepared.pixels_per_mm,
                    ),
                    square.side_mm * result.prepared.pixels_per_mm,
                    square.angle_deg,
                )
                digit_correct += int(recognition.digit == int(expected[truth_index]["digit"]))
                digit_total += 1

        if not ok:
            debug = draw_low_resolution_squares(result.prepared.image, result)
            cv2.imwrite(str(args.debug_dir / f"{metadata_path.stem}.jpg"), debug)

    for (family, width), (passed, total) in sorted(totals.items()):
        print(f"{family:18s} A4_width={width:3d}px geometry={passed}/{total} ({passed / total:.1%})")
    if errors:
        columns = tuple(zip(*errors))
        print(
            "matched_error mean/max (size_mm, center_mm, angle_deg): "
            f"{tuple(round(float(np.mean(column)), 3) for column in columns)} / "
            f"{tuple(round(max(column), 3) for column in columns)}"
        )
    print(f"geometry_exact_and_within_tolerance: {geometry_pass}/{sample_count} ({geometry_pass / sample_count:.1%})")
    print(f"numbered_digit_accuracy_on_exact_geometry: {digit_correct}/{digit_total} ({digit_correct / digit_total:.1%})" if digit_total else "numbered_digit_accuracy_on_exact_geometry: n/a")
    return 0 if geometry_pass == sample_count and digit_correct == digit_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
