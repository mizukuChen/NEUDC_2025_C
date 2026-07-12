from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# 将项目根目录添加到系统搜索路径中，确保能成功导入 src 中的模块
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.measure_shapes import draw_shape_measurements, measure_shapes


def parse_args() -> argparse.Namespace:
    """
    配置并解析单图形状测量脚本参数。
    """
    parser = argparse.ArgumentParser(description="Measure shapes in a rectified A4 target PNG.")
    parser.add_argument("image", type=Path, help="已经完成透视拉平的 A4 标准图像路径。")
    parser.add_argument("--output", type=Path, default=ROOT / "runtime" / "debug" / "shape_measurement.jpg", help="标注后调试输出图的保存路径。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        print(f"ERROR: cannot read image: {args.image}", file=sys.stderr)
        return 2

    # 1. 调用 measure_shapes 进行多目标形状测量与分类
    result = measure_shapes(image)

    # 2. 绘制检测结果轮廓与文字标签，并保存图像
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), draw_shape_measurements(image, result))

    # 3. 在控制台格式化输出识别到的排版、图形个数和各图形参数
    print(f"layout: {result.layout}")
    print(f"count: {len(result.shapes)}")
    for index, shape in enumerate(result.shapes, start=1):
        print(
            f"shape[{index}]: type={shape.shape_type} x_mm={shape.x_mm:.2f} "
            f"center_mm=({shape.center_mm[0]:.1f},{shape.center_mm[1]:.1f}) "
            f"angle_deg={shape.angle_deg:.1f} confidence={shape.confidence:.3f}"
        )
    print(f"debug: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
