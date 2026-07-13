from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.targets.generation.constraints import validate_numbered_overlap
from scripts.targets.generation.renderer import render
from scripts.targets.presets.numbered_overlap import PRESETS


DEFAULT_OUTPUT = ROOT / "assets/targets/generated/paired_numbered_overlap"


def perturb(base: tuple, rng: np.random.Generator) -> tuple:
    return tuple(
        replace(
            item,
            cx_mm=item.cx_mm + float(rng.uniform(-4.0, 4.0)),
            cy_mm=item.cy_mm + float(rng.uniform(-4.0, 4.0)),
            size_mm=item.size_mm + float(rng.uniform(-3.0, 3.0)),
            angle_deg=float(np.clip(item.angle_deg + rng.uniform(-4.0, 4.0), -44.5, 44.5)),
        )
        for item in base
    )


def valid_variant(base: tuple, rng: np.random.Generator, attempts: int = 200) -> tuple:
    for _ in range(attempts):
        candidate = perturb(base, rng)
        try:
            validate_numbered_overlap(candidate)
        except ValueError:
            continue
        return candidate
    raise RuntimeError("could not sample a valid numbered-overlap variant")


def main() -> int:
    parser = argparse.ArgumentParser(description="生成同几何的无数字/带数字重叠方块成对数据。")
    parser.add_argument("--variants", type=int, default=10, help="每个基础布局的随机变体数")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.variants < 1:
        parser.error("--variants must be at least 1")

    rng = np.random.default_rng(args.seed)
    plain_dir = args.output / "plain"
    numbered_dir = args.output / "numbered"
    plain_dir.mkdir(parents=True, exist_ok=True)
    numbered_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for base_name, base in PRESETS.items():
        for variant in range(1, args.variants + 1):
            shapes = valid_variant(base, rng)
            numbered, items = render(shapes)
            plain_shapes = tuple(replace(item, digit=None) for item in shapes)
            plain, _ = render(plain_shapes)
            stem = f"pair_{base_name}_{variant:02d}"
            Image.fromarray(plain).save(plain_dir / f"{stem}.png")
            Image.fromarray(numbered).save(numbered_dir / f"{stem}.png")
            metadata = {
                "pair_id": stem,
                "seed": args.seed,
                "base_preset": base_name,
                "variant": variant,
                "squares": items,
            }
            (args.output / f"{stem}.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            total += 1
    print(f"generated {total} plain/numbered pairs in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
