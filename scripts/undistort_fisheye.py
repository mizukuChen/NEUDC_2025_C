from __future__ import annotations

"""
单张图像鱼眼畸变去畸变测试脚本。

本脚本加载指定的鱼眼相机标定参数 JSON 文件，应用 OpenCV 鱼眼重映射模型，
对单张受鱼眼畸变弯曲的输入照片进行平铺拉直校正，并将无畸变的理想透视图像保存到指定路径。
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
    parser = argparse.ArgumentParser(description="Remove fisheye distortion from one image.")
    parser.add_argument("image", type=Path, help="待校正的畸变源图像路径。")
    parser.add_argument("--calibration", type=Path, default=ROOT / "runtime" / "calibration" / "fisheye_1280x720.json", help="相机鱼眼标定参数 JSON 文件路径。")
    parser.add_argument("--output", type=Path, required=True, help="校正去畸变后图像的保存路径。")
    parser.add_argument("--balance", type=float, default=0.0, help="去畸变边缘保留平衡因子 (0.0~1.0)。0 代表截掉空白边缘，1 代表保留全部像素。")
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        print(f"ERROR: cannot read image: {args.image}", file=sys.stderr)
        return 2

    # 应用标定参数去畸变
    corrected = undistort_fisheye_image(image, load_fisheye_calibration(args.calibration), args.balance)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), corrected):
        print(f"ERROR: cannot write image: {args.output}", file=sys.stderr)
        return 3
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
