from __future__ import annotations

"""
鱼眼相机畸变标定与参数生成工具。

本脚本读取指定棋盘格图片目录，利用 OpenCV 鱼眼标定算法计算相机内参矩阵
（包含 fx, fy, cx, cy）以及四阶鱼眼畸变系数（k1, k2, k3, k4），
并将标定好的参数以 JSON 格式输出，用于实时测距系统和去畸变预处理。
"""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.fisheye import calibrate_fisheye, save_fisheye_calibration


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate an OpenCV fisheye model from chessboard images.")
    parser.add_argument("image_dir", nargs="?", type=Path, default=ROOT / "runtime" / "captures" / "eye_distortion", help="包含棋盘格标定照片的输入目录。")
    parser.add_argument("--output", type=Path, default=ROOT / "runtime" / "calibration" / "fisheye_1280x720.json", help="输出鱼眼标定参数的 JSON 保存路径。")
    args = parser.parse_args()

    # 筛选并排序目录下所有的合法图片路径
    image_paths = sorted(path for path in args.image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})

    # 执行鱼眼畸变标定计算
    calibration = calibrate_fisheye(image_paths)

    # 保存参数到指定 JSON 配置文件中
    save_fisheye_calibration(calibration, args.output)

    # 在控制台输出标定结果统计指标，重投影误差越小说明标定越精准
    print(f"output: {args.output}")
    print(f"used_images: {len(calibration.used_images)}")
    print(f"rejected_images: {len(calibration.rejected_images)}")
    print(f"rms_px: {calibration.rms_px:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
