from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# 将项目根目录添加到系统搜索路径中，确保能成功导入 src 中的模块
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.distance import estimate_plane_center_depth, load_distance_correction, undistorted_camera_matrix
from src.vision.fisheye import load_fisheye_calibration, undistort_fisheye_image
from src.vision.target_plane import draw_target_plane_debug, find_target_plane


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure optical-axis distance to the A4 target plane.")
    parser.add_argument("image", type=Path, help="输入的原始实拍图像。")
    parser.add_argument("--fisheye-calibration", type=Path, default=ROOT / "runtime" / "calibration" / "fisheye_1280x720.json", help="相机鱼眼标定参数路径。")
    parser.add_argument("--distance-calibration", type=Path, default=ROOT / "runtime" / "calibration" / "distance_linear.json", help="线性距离修正模型路径。")
    # 设定裁剪比例：默认只保留图像中央 1/3 的垂直区域，大大过滤了左右两侧的无关背景噪声
    parser.add_argument("--strip-fraction", type=float, default=1.0 / 3.0, help="垂直中心条带裁剪比例。")
    parser.add_argument("--max-reprojection-px", type=float, default=2.0, help="允许的最大重投影误差（像素），超限说明定位严重偏差。")
    parser.add_argument("--debug-dir", type=Path, default=ROOT / "runtime" / "debug" / "distance", help="调试输出图的保存目录。")
    args = parser.parse_args()

    raw = cv2.imread(str(args.image))
    if raw is None:
        print(f"ERROR: cannot read image: {args.image}", file=sys.stderr)
        return 2

    # 1. 载入标定参数
    fisheye = load_fisheye_calibration(args.fisheye_calibration)
    correction = load_distance_correction(args.distance_calibration)

    # 2. 图像鱼眼去畸变校正
    undistorted = undistort_fisheye_image(raw, fisheye)

    # 3. 截取图像中央 1/3 的垂直区域（Center Strip）
    height, width = undistorted.shape[:2]
    strip_width = round(width * args.strip_fraction)
    left = (width - strip_width) // 2
    strip = undistorted[:, left : left + strip_width]

    # 4. 在裁剪出的条带图像上定位 A4 黑边框
    plane = find_target_plane(strip)

    # 5. 将条带图像上的角点映射回原始大图的全局像素坐标系
    full_corners = plane.corners.astype(np.float64)
    full_corners[:, 0] += left

    # 6. 计算理想内参矩阵，并运用 solvePnP 进行三维空间深度测距
    camera_matrix = undistorted_camera_matrix(fisheye, (width, height))
    estimate = estimate_plane_center_depth(full_corners, camera_matrix, correction)

    # 7. 校验重投影误差，防止定位错误导致测距值完全失效
    if estimate.reprojection_error_px > args.max_reprojection_px:
        print(
            f"ERROR: reprojection error {estimate.reprojection_error_px:.3f} px exceeds "
            f"{args.max_reprojection_px:.3f} px",
            file=sys.stderr,
        )
        return 3

    # 保存各步骤调试图
    debug_dir = args.debug_dir / args.image.stem
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / "00_undistorted.jpg"), undistorted)
    cv2.imwrite(str(debug_dir / "00_center_strip.jpg"), strip)
    cv2.imwrite(str(debug_dir / "01_plane_corners.jpg"), draw_target_plane_debug(strip, plane))

    # 终端打印测距和校正值
    print(f"image: {args.image}")
    print(f"D_mm: {estimate.corrected_distance_mm:.1f}")
    print(f"D_cm: {estimate.corrected_distance_mm / 10.0:.2f}")
    print(f"raw_pnp_mm: {estimate.raw_distance_mm:.1f}")
    print(f"reprojection_px: {estimate.reprojection_error_px:.3f}")
    print(f"debug_dir: {debug_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
