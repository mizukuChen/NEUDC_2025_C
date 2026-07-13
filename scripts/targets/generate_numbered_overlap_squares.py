from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.targets.generation.constraints import validate_numbered_overlap
from scripts.targets.generation.page import DEFAULT_PAGE
from scripts.targets.generation.renderer import render
from scripts.targets.presets.numbered_overlap import PRESETS


DEFAULT_OUTPUT = ROOT / "assets/targets/generated/numbered_overlap_controlled"
MAX_EDGE_COVERAGE = 0.5
DIGIT_SAFE_RADIUS_RATIO = 0.30


def generate_case(case: str, shapes: tuple, output: Path) -> None:
    ratios, clearances = validate_numbered_overlap(
        shapes,
        max_coverage=MAX_EDGE_COVERAGE,
        digit_safe_radius_ratio=DIGIT_SAFE_RADIUS_RATIO,
    )
    image, items = render(shapes)
    for item, ratio, clearance in zip(items, ratios, clearances):
        item["edge_coverage"] = ratio
        item["digit_clearance_mm"] = clearance
    payload = {
        "page_mm": [210, 297],
        "dpi": DEFAULT_PAGE.dpi,
        "target_type": "numbered_squares",
        "layout": "overlap",
        "max_edge_coverage": MAX_EDGE_COVERAGE,
        "digit_safe_radius_ratio": DIGIT_SAFE_RADIUS_RATIO,
        "squares": items,
    }
    output.mkdir(parents=True, exist_ok=True)
    stem = f"numbered_overlap_{case}_A4"
    Image.fromarray(image).save(output / f"{stem}.png")
    (output / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"generated {stem}.png digits={[item.digit for item in shapes]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="生成带数字的受控重叠正方形目标。")
    parser.add_argument("--case", choices=(*PRESETS, "all"), default="all")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cases = PRESETS.items() if args.case == "all" else ((args.case, PRESETS[args.case]),)
    for case, shapes in cases:
        generate_case(case, shapes, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
