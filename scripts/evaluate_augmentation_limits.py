from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.targets.augmentation import AugmentationConfig, augment
from src.vision.measurement_tasks import measure_normal, measure_numbered


DATASETS = (
    ("basic", ROOT / "assets/targets/generated/modular/basic"),
    ("separate", ROOT / "assets/targets/generated/modular/separate"),
    ("overlap", ROOT / "assets/targets/generated/overlap_controlled"),
    ("numbered", ROOT / "assets/targets/generated/numbered_controlled"),
)

DEFAULT_LEVELS = {
    "blur": (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0),
    "illumination": (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60),
    "perspective": (0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0),
}


@dataclass
class Score:
    passed: int = 0
    total: int = 0
    max_error_mm: float = 0.0
    exceptions: int = 0
    mismatches: int = 0

    def add(self, error_mm: float | None, tolerance_mm: float) -> None:
        self.total += 1
        if error_mm is None:
            self.exceptions += 1
            return
        self.max_error_mm = max(self.max_error_mm, error_mm)
        self.passed += int(error_mm <= tolerance_mm)

    def add_mismatch(self) -> None:
        self.total += 1
        self.mismatches += 1


def config_for(factor: str, level: float) -> AugmentationConfig:
    zero = (0.0, 0.0)
    return AugmentationConfig(
        perspective_px=(level, level) if factor == "perspective" else zero,
        blur_sigma_px=(level, level) if factor == "blur" else zero,
        illumination_strength=(level, level) if factor == "illumination" else zero,
        noise_stddev=zero,
    )


def load_cases() -> list[tuple[str, Path, object, list[dict], bool]]:
    cases = []
    for category, directory in DATASETS:
        for image_path in sorted(directory.glob("*.png")):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"cannot read {image_path}")
            truth = json.loads(image_path.with_suffix(".json").read_text(encoding="utf-8"))
            key = "squares" if "squares" in truth else "shapes"
            cases.append((category, image_path, image, truth[key], category == "numbered"))
    return cases


def evaluate_level(
    cases: list[tuple[str, Path, object, list[dict], bool]],
    factor: str,
    level: float,
    variants: int,
    seed: int,
    tolerance_mm: float,
    center_tolerance_mm: float,
) -> dict[str, Score]:
    scores = {name: Score() for name, _ in DATASETS}
    scores["digit"] = Score()
    config = config_for(factor, level)
    for case_index, (category, _path, image, truth, numbered) in enumerate(cases):
        expected = min(float(item["size_mm"]) for item in truth)
        for variant in range(variants):
            sample_seed = seed + case_index * variants + variant
            disturbed, _ = augment(image, sample_seed, config)
            bgr = cv2.cvtColor(disturbed, cv2.COLOR_GRAY2BGR)
            try:
                result = measure_normal(bgr)
                scores[category].add(abs(result.x_mm - expected), tolerance_mm)
            except (RuntimeError, ValueError):
                scores[category].add(None, tolerance_mm)

            if not numbered:
                continue
            for item in truth:
                try:
                    result = measure_numbered(bgr, int(item["digit"]))
                    center = result.selected_square.geometry.center_mm
                    center_error = math.hypot(
                        center[0] - float(item["cx_mm"]),
                        center[1] - float(item["cy_mm"]),
                    )
                    size_error = abs(result.x_mm - float(item["size_mm"]))
                    # 核对中心，避免误选尺寸相近的另一个编号仍被判为通过。
                    if center_error <= center_tolerance_mm:
                        scores["digit"].add(size_error, tolerance_mm)
                    else:
                        scores["digit"].add_mismatch()
                except (RuntimeError, ValueError):
                    scores["digit"].add(None, tolerance_mm)
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description="逐项扫描图像扰动的测量耐受边界。")
    parser.add_argument("--factor", choices=(*DEFAULT_LEVELS, "all"), default="all")
    parser.add_argument("--levels", type=float, nargs="+", help="覆盖所选单项的默认强度")
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--tolerance-mm", type=float, default=1.0)
    parser.add_argument("--center-tolerance-mm", type=float, default=3.0)
    parser.add_argument("--csv", type=Path, help="可选的明细 CSV 输出路径")
    args = parser.parse_args()
    if args.variants < 1:
        parser.error("--variants must be at least 1")
    if args.levels and args.factor == "all":
        parser.error("--levels requires one specific --factor")

    cases = load_cases()
    factors = DEFAULT_LEVELS if args.factor == "all" else (args.factor,)
    rows = []
    for factor in factors:
        levels = tuple(args.levels) if args.levels else DEFAULT_LEVELS[factor]
        unit = {"blur": "sigma_px", "illumination": "strength", "perspective": "max_px"}[factor]
        print(f"\n[{factor}] unit={unit}", flush=True)
        for level in levels:
            scores = evaluate_level(
                cases,
                factor,
                level,
                args.variants,
                args.seed,
                args.tolerance_mm,
                args.center_tolerance_mm,
            )
            parts = []
            for category in ("basic", "separate", "overlap", "numbered", "digit"):
                score = scores[category]
                parts.append(
                    f"{category}={score.passed}/{score.total} "
                    f"max={score.max_error_mm:.3f}mm exc={score.exceptions} "
                    f"mismatch={score.mismatches}"
                )
                rows.append(
                    {
                        "factor": factor,
                        "unit": unit,
                        "level": level,
                        "category": category,
                        "passed": score.passed,
                        "total": score.total,
                        "max_error_mm": score.max_error_mm,
                        "exceptions": score.exceptions,
                        "mismatches": score.mismatches,
                    }
                )
            print(f"level={level:g} " + " | ".join(parts), flush=True)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
