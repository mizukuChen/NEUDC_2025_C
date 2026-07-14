from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.targets.augmentation.low_resolution import (
    LowResolutionConfig,
    simulate_low_resolution_a4,
)
from scripts.targets.generation.renderer import render
from scripts.targets.presets.numbered_overlap import PRESETS as NUMBERED_PRESETS
from scripts.targets.presets.overlap import PRESETS as OVERLAP_PRESETS


DEFAULT_OUTPUT = ROOT / "runtime" / "debug" / "lowres_overlap_dataset"
DEFAULT_WIDTHS = (82, 100, 130, 164)


def _sample_config(rng: np.random.Generator, native_width_px: int) -> LowResolutionConfig:
    return LowResolutionConfig(
        native_width_px=native_width_px,
        defocus_sigma_native_px=float(rng.uniform(0.20, 0.55)),
        black_level=float(rng.uniform(10.0, 25.0)),
        white_level=float(rng.uniform(155.0, 195.0)),
        illumination_strength=float(rng.uniform(0.0, 0.08)),
        illumination_angle_deg=float(rng.uniform(-180.0, 180.0)),
        noise_stddev=float(rng.uniform(0.3, 1.2)),
    )


def generate_dataset(
    output: Path,
    widths: tuple[int, ...] = DEFAULT_WIDTHS,
    variants: int = 3,
    frames_per_variant: int = 3,
    seed: int = 202507,
) -> int:
    if variants < 1 or frames_per_variant < 1:
        raise ValueError("variants and frames-per-variant must be positive")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    count = 0

    for family, presets in (("overlap", OVERLAP_PRESETS), ("numbered_overlap", NUMBERED_PRESETS)):
        family_dir = output / family
        family_dir.mkdir(parents=True, exist_ok=True)
        for case_name, specs in presets.items():
            source, shapes = render(specs)
            for native_width_px in widths:
                for variant in range(variants):
                    base = _sample_config(rng, native_width_px)
                    stem = f"{family}_{case_name}_w{native_width_px:03d}_v{variant + 1:02d}"
                    frame_names: list[str] = []
                    frame_metadata: list[dict] = []
                    for frame_index in range(frames_per_variant):
                        config = replace(
                            base,
                            phase_x_px=float(rng.uniform(-0.5, 0.5)),
                            phase_y_px=float(rng.uniform(-0.5, 0.5)),
                        )
                        frame, metadata = simulate_low_resolution_a4(
                            source,
                            config,
                            seed=seed + count * frames_per_variant + frame_index,
                        )
                        frame_name = f"{stem}_f{frame_index + 1:02d}.png"
                        if not cv2.imwrite(str(family_dir / frame_name), frame):
                            raise OSError(f"cannot write {family_dir / frame_name}")
                        frame_names.append(frame_name)
                        frame_metadata.append(metadata)

                    payload = {
                        "family": family,
                        "case": case_name,
                        "native_a4_width_px": native_width_px,
                        "variant": variant + 1,
                        "frames": frame_names,
                        "frame_metadata": frame_metadata,
                        "page_mm": [210.0, 297.0],
                        "squares": shapes,
                        "base_config": asdict(base),
                    }
                    (family_dir / f"{stem}.json").write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="生成低像素密度的重叠正方形模拟数据集。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--widths", type=int, nargs="+", default=list(DEFAULT_WIDTHS))
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--frames-per-variant", type=int, default=3)
    parser.add_argument("--seed", type=int, default=202507)
    args = parser.parse_args()
    count = generate_dataset(
        args.output,
        tuple(args.widths),
        args.variants,
        args.frames_per_variant,
        args.seed,
    )
    print(f"generated {count} samples in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
