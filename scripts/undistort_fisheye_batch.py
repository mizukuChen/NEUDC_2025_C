from __future__ import annotations

"""
批量图像鱼眼畸变去畸变校正工具。

本脚本批量读取指定输入文件夹下的所有照片，加载相机鱼眼标定 JSON 参数，
进行去畸变平铺校正，并将校正后的图像统一保存输出到指定的调试目录中，
常用于批量评估相机标定效果或预处理测试图像数据集。
"""

import argparse
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.fisheye import load_fisheye_calibration, undistort_fisheye_image


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-remove fisheye distortion from calibration images.")
    parser.add_argument("input_dir", nargs="?", type=Path, default=ROOT / "runtime" / "captures" / "eye_distortion", help="包含待批量校正源图片的输入文件夹目录。")
    parser.add_argument("--calibration", type=Path, default=ROOT / "runtime" / "calibration" / "fisheye_1280x720.json", help="相机鱼眼标定参数 JSON 文件路径。")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runtime" / "debug" / "eye_distortion_undistorted", help="输出去畸变后图像保存的目标目录。")
    parser.add_argument("--balance", type=float, default=0.0, help="边缘保留平衡因子 (0.0~1.0)。")
    args = parser.parse_args()

    # 筛选并排序目录下所有的合法图片路径
    paths = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        print(f"ERROR: no images found in {args.input_dir}", file=sys.stderr)
        return 2

    # 加载鱼眼参数配置
    calibration = load_fisheye_calibration(args.calibration)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    # 循环去畸变并保存
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"SKIP: cannot read {path}", file=sys.stderr)
            continue
        corrected = undistort_fisheye_image(image, calibration, args.balance)
        output_path = args.output_dir / path.name
        if not cv2.imwrite(str(output_path), corrected):
            print(f"SKIP: cannot write {output_path}", file=sys.stderr)
            continue
        written += 1

    print(f"output_dir: {args.output_dir}")
    print(f"written: {written}/{len(paths)}")
    return 0 if written == len(paths) else 3


if __name__ == "__main__":
    raise SystemExit(main())
