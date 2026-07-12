from __future__ import annotations

"""
标准 A4 靶标图形尺寸测量精度批量评估工具。

本脚本读取生成的基础与高级 A4 靶纸图像（PNG 格式），
调用 `measure_shapes` 模块检测圆形、等边三角形和正方形。
根据图片名称（如 basic_circle_A4.png 等）解析其预期的图形个数和类型标签，
并比对判定识别数量和形状类别是否正确，最后输出详细的边长/直径测试数据。
"""

import argparse
import re
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.measure_shapes import measure_shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate complete-contour shape detection on standard targets.")
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[ROOT / "assets" / "targets" / "generated" / "png", ROOT / "assets" / "targets" / "generated" / "advanced" / "png"],
        help="输入图像或包含测试图像的目录列表。",
    )
    return parser.parse_args()


def iter_images(paths: list[Path]) -> list[Path]:
    """
    遍历输入路径列表，搜集并返回所有 PNG 格式的图像文件路径。
    """
    images: list[Path] = []
    for path in paths:
        if path.is_file():
            images.append(path)
        elif path.is_dir():
            images.extend(path.glob("*.png"))
    return sorted(images)


def expected(path: Path) -> tuple[int, str] | None:
    """
    通过文件名规则解析预期检测到的几何图形数量和形状类别。

    例如:
        - basic_circle_A4.png -> (1, "circle")
        - advanced_single_square_80mm_A4.png -> (1, "square")
        - advanced_separate_squares_02_A4.png -> (4, "square")
    """
    name = path.stem.lower()
    if name.startswith("basic_"):
        return 1, name.split("_")[1]
    if "single_square" in name:
        return 1, "square"
    match = re.search(r"separate_squares_(\d+)", name)
    counts = {"01": 3, "02": 4, "03": 5, "04": 4}
    if match:
        return counts[match.group(1)], "square"
    return None


def main() -> int:
    args = parse_args()
    images = [(path, expected(path)) for path in iter_images(args.inputs)]
    images = [(path, value) for path, value in images if value is not None]
    if not images:
        print("No supported standard target PNGs found.", file=sys.stderr)
        return 2

    failures = 0
    # 逐张读取图像进行测量评估
    for path, (expected_count, expected_type) in images:
        try:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError("cannot read image")

            # 执行多形状检测与测量
            result = measure_shapes(image)

            # 校验判定：所有提取到的形状类别必须匹配，且数量必须吻合
            types_ok = all(item.shape_type == expected_type for item in result.shapes)
            count_ok = len(result.shapes) == expected_count
            ok = types_ok and count_ok
            failures += int(not ok)

            # 格式化输出每一个靶标图像的实测特征尺寸数据
            sizes = ",".join(f"{item.x_mm:.1f}" for item in result.shapes)
            print(f"{'PASS' if ok else 'FAIL'} {path.name:40s} count={len(result.shapes)}/{expected_count} sizes_mm=[{sizes}]")
        except Exception as exc:
            failures += 1
            print(f"FAIL {path.name:40s} error={exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
