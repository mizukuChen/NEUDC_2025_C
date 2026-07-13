from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.targets.augmentation import AugmentationConfig, augment


DEFAULT_OUTPUT = ROOT / "assets" / "targets" / "generated" / "augmented"


def _sources(path: Path) -> list[Path]:
    return sorted(path.rglob("*.png")) if path.is_dir() else [path]


def main() -> int:
    parser = argparse.ArgumentParser(description="为标准目标 PNG 生成轻微相机扰动样本。")
    parser.add_argument("input", type=Path, help="单张 PNG 或包含 PNG/JSON 的目录")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants", type=int, default=5, help="每张标准图的扰动版本数")
    parser.add_argument("--seed", type=int, default=2025, help="数据集随机种子")
    parser.add_argument("--noise", type=float, default=0.0, help="可选噪声标准差，默认关闭")
    args = parser.parse_args()

    if args.variants < 1:
        parser.error("--variants must be at least 1")
    if args.noise < 0:
        parser.error("--noise cannot be negative")

    config = AugmentationConfig(noise_stddev=(0.0, args.noise))
    sources = _sources(args.input)
    if not sources:
        parser.error(f"no PNG files found under {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    generated = 0
    for source_index, image_path in enumerate(sources):
        truth_path = image_path.with_suffix(".json")
        if not truth_path.exists():
            parser.error(f"missing ground truth: {truth_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            parser.error(f"cannot read image: {image_path}")
        truth = json.loads(truth_path.read_text(encoding="utf-8"))

        for variant in range(args.variants):
            sample_seed = args.seed + source_index * args.variants + variant
            disturbed, metadata = augment(image, sample_seed, config)
            stem = f"{image_path.stem}__aug_{variant + 1:02d}"
            cv2.imwrite(str(args.output / f"{stem}.png"), disturbed)
            payload = dict(truth)
            payload["augmentation"] = metadata
            payload["source_image"] = str(image_path.resolve())
            (args.output / f"{stem}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            generated += 1
    print(f"generated {generated} augmented PNG/JSON pairs in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
