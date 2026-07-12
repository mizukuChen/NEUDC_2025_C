from __future__ import annotations

"""
光轴深度距离（测距）标定修正系数计算脚本。

本脚本读取指定目录下不同标定距离（通过文件名解析，如 d1200mm_*.jpg）拍摄的 A4 靶纸图像，
利用鱼眼去畸变和 solvePnP 算法求解靶纸的原始相机坐标系深度（Z轴分量）。
随后，使用鲁棒线性回归拟合“原始测量深度 -> 真实距离”的修正系数（Scale & Offset），
并将标定系数输出为 JSON 配置文件，同时生成 CSV 校准分析报告。
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np

# 将项目根目录添加到系统搜索路径中，确保能成功导入 src 中的模块
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.distance import (
    LinearDistanceCorrection,
    estimate_plane_center_depth,
    save_distance_correction,
    undistorted_camera_matrix,
)
from src.vision.fisheye import load_fisheye_calibration, undistort_fisheye_image
from src.vision.target_plane import find_target_plane

# 文件名正则表达式，用于提取图片对应的真实物理距离（例如：d1200mm_rpi_1.jpg 提取出 1200）
DISTANCE_PATTERN = re.compile(r"^d(?P<distance_mm>\d+)mm_")


def robust_linear_fit(raw: np.ndarray, truth: np.ndarray, threshold_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """
    使用鲁棒的 Theil-Sen 估计器计算拟合斜率和截距，剔除由于检测失误或畸变产生的异常值（Outliers）。

    参数:
        raw: solvePnP 解算出的原始深度距离数组（mm）。
        truth: 实际摆放的真实物理距离数组（mm）。
        threshold_mm: 误差容差阈值，偏差大于此值的样本在二次拟合时被标记为离群点排除。

    返回:
        poly_coefficients: np.polyfit 得到的 [Scale, Offset] 拟合参数。
        inliers: 布尔掩膜数组，指示哪些样本点被判定为有效内点。
    """
    slopes = []
    # 1. 计算任意两点对之间的斜率
    for i in range(len(raw)):
        for j in range(i + 1, len(raw)):
            delta = raw[j] - raw[i]
            if abs(delta) > 1e-6:
                slopes.append((truth[j] - truth[i]) / delta)
    if not slopes:
        raise ValueError("Distance samples do not span enough range for a linear fit.")

    # 2. 取斜率中位数作为鲁棒的 Scale 估计值
    robust_scale = float(np.median(slopes))
    # 3. 取截距中位数作为鲁棒的 Offset 估计值
    robust_offset = float(np.median(truth - robust_scale * raw))

    # 4. 筛选误差在设定门限以内的内点（Inliers）
    inliers = np.abs(robust_scale * raw + robust_offset - truth) <= threshold_mm
    if int(inliers.sum()) < 5:
        raise ValueError("Too few inlier samples remain after robust filtering.")

    # 5. 对所有内点进行标准最小二乘一次多项式拟合，以获取高精度的最终系数
    return np.polyfit(raw[inliers], truth[inliers], 1), inliers


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate PnP optical-axis distance correction.")
    parser.add_argument("input_dir", nargs="?", type=Path, default=ROOT / "runtime" / "captures" / "rpi_distance", help="包含测距标定照片的输入目录。")
    parser.add_argument("--fisheye-calibration", type=Path, default=ROOT / "runtime" / "calibration" / "fisheye_1280x720.json", help="已保存的鱼眼标定 JSON 文件路径。")
    parser.add_argument("--output", type=Path, default=ROOT / "runtime" / "calibration" / "distance_linear.json", help="输出的线性距离修正模型保存路径。")
    parser.add_argument("--report", type=Path, default=ROOT / "runtime" / "debug" / "distance_calibration.csv", help="输出的 CSV 分析报告保存路径。")
    parser.add_argument("--strip-fraction", type=float, default=1.0 / 3.0, help="垂直中心条带裁剪比例。")
    parser.add_argument("--outlier-mm", type=float, default=50.0, help="鲁棒拟合中判断离群值（异常测距点）的最大容差阈值（mm）。")
    parser.add_argument("--max-reprojection-px", type=float, default=2.0, help="解算允许的最大重投影误差（像素）。")
    args = parser.parse_args()

    # 加载已有的相机鱼眼标定参数
    calibration = load_fisheye_calibration(args.fisheye_calibration)
    # 获取去畸变后的理想相机内参矩阵
    camera_matrix = undistorted_camera_matrix(calibration, calibration.image_size)
    rows: list[dict[str, object]] = []

    # 遍历输入目录下的标定照片
    for path in sorted(args.input_dir.glob("*.jpg")):
        match = DISTANCE_PATTERN.match(path.name)
        if not match:
            continue

        # 提取文件名中的真实距离（Ground Truth）
        truth_mm = int(match.group("distance_mm"))
        row: dict[str, object] = {
            "image": path.name,
            "truth_mm": truth_mm,
            "raw_mm": "",
            "reprojection_px": "",
            "inlier": False,
            "error": "",
        }

        try:
            raw_image = cv2.imread(str(path))
            if raw_image is None:
                raise ValueError("cannot read image")

            # 去鱼眼畸变
            undistorted = undistort_fisheye_image(raw_image, calibration)
            height, width = undistorted.shape[:2]

            # 截取垂直中央条带区域，避免左右背景杂质干扰
            strip_width = round(width * args.strip_fraction)
            left = (width - strip_width) // 2
            strip = undistorted[:, left : left + strip_width]

            # 寻找 A4 平面黑框
            plane = find_target_plane(strip)
            corners = plane.corners.astype(np.float64)
            # 映射回原图的 X 像素坐标
            corners[:, 0] += left

            # PnP 求解姿态，获取原始测距值
            estimate = estimate_plane_center_depth(corners, camera_matrix)
            if estimate.reprojection_error_px > args.max_reprojection_px:
                raise ValueError(f"reprojection error {estimate.reprojection_error_px:.3f} px exceeds limit")

            row["raw_mm"] = estimate.raw_distance_mm
            row["reprojection_px"] = estimate.reprojection_error_px
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)

    # 提取所有解算成功的测量样本
    usable_indices = [index for index, row in enumerate(rows) if row["raw_mm"] != ""]
    if len(usable_indices) < 5:
        print("ERROR: fewer than five usable distance samples", file=sys.stderr)
        return 2

    raw = np.array([float(rows[index]["raw_mm"]) for index in usable_indices])
    truth = np.array([float(rows[index]["truth_mm"]) for index in usable_indices])

    # 鲁棒多项式回归拟合修正公式： D_corrected = D_raw * Scale + Offset
    coefficients, inliers = robust_linear_fit(raw, truth, args.outlier_mm)
    corrected = np.polyval(coefficients, raw[inliers])
    residual = corrected - truth[inliers]
    excluded: list[str] = []

    for local_index, row_index in enumerate(usable_indices):
        rows[row_index]["inlier"] = bool(inliers[local_index])
        if not inliers[local_index]:
            excluded.append(str(rows[row_index]["image"]))

    # 创建并保存线性纠偏模型参数
    correction = LinearDistanceCorrection(
        scale=float(coefficients[0]),
        offset_mm=float(coefficients[1]),
        fit_mae_mm=float(np.mean(np.abs(residual))),
        fit_rmse_mm=float(np.sqrt(np.mean(residual * residual))),
        fit_max_error_mm=float(np.max(np.abs(residual))),
        inlier_count=int(inliers.sum()),
        excluded_images=tuple(excluded),
    )
    save_distance_correction(correction, args.output)

    # 导出 CSV 校准报告
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=("image", "truth_mm", "raw_mm", "reprojection_px", "inlier", "error"))
        writer.writeheader()
        writer.writerows(rows)

    # 控制台格式化输出标定拟合结果指标
    print(f"output: {args.output}")
    print(f"samples: {correction.inlier_count}/{len(rows)} inliers")
    print(f"scale: {correction.scale:.9f}")
    print(f"offset_mm: {correction.offset_mm:.4f}")
    print(f"fit_mae_mm: {correction.fit_mae_mm:.3f}")
    print(f"fit_max_error_mm: {correction.fit_max_error_mm:.3f}")
    print(f"excluded: {', '.join(correction.excluded_images) or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
