from __future__ import annotations

"""
相机测距误差批量评估与测试分析工具。

本脚本读取指定输入照片目录下的测距图片（要求图片文件名中包含真实摆放距离，如 d800mm_*.jpg），
批量执行以下测距工作流：
1. 鱼眼去畸变矫正
2. 截取图像中央 1/3 竖直条带（过滤环境噪声）
3. 定位 A4 纸四角并计算透视矫正
4. 利用 PnP 解算和线性校正系数，估计靶纸与相机的真实距离（D）
5. 计算测量误差（Error = D_estimated - D_groundtruth）

最后，将每个图像的测量数据详细记录并导出到 CSV 评估报告中，
同时在控制台打印平均绝对误差（MAE）、最大测量误差以及符合国家电赛测距精度的样本通过率（误差在 50mm 以内）。
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.distance import estimate_plane_center_depth, load_distance_correction, undistorted_camera_matrix
from src.vision.fisheye import load_fisheye_calibration, undistort_fisheye_image
from src.vision.target_plane import find_target_plane

# 文件名正则：匹配真实摆放距离以作比对（如 d1500mm_rpi.jpg 提取出 1500）
DISTANCE_PATTERN = re.compile(r"^d(?P<distance_mm>\d+)mm_")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate corrected PnP distances.")
    parser.add_argument("input_dir", nargs="?", type=Path, default=ROOT / "runtime" / "captures" / "rpi_distance", help="需要进行测距评估的输入图像目录。")
    parser.add_argument("--fisheye-calibration", type=Path, default=ROOT / "runtime" / "calibration" / "fisheye_1280x720.json", help="已保存的鱼眼参数配置文件路径。")
    parser.add_argument("--distance-calibration", type=Path, default=ROOT / "runtime" / "calibration" / "distance_linear.json", help="已保存的线性距离纠偏配置文件路径。")
    parser.add_argument("--report", type=Path, default=ROOT / "runtime" / "debug" / "distance_evaluation.csv", help="输出的评估分析 CSV 报告保存路径。")
    parser.add_argument("--strip-fraction", type=float, default=1.0 / 3.0, help="垂直中心条带裁剪比例。")
    args = parser.parse_args()

    # 1. 载入鱼眼标定和线性测距修正参数
    fisheye = load_fisheye_calibration(args.fisheye_calibration)
    correction = load_distance_correction(args.distance_calibration)

    # 2. 计算理想无畸变相机投影内参矩阵
    camera_matrix = undistorted_camera_matrix(fisheye, fisheye.image_size)
    rows: list[dict[str, object]] = []

    # 遍历目录下所有实拍标定照片
    for path in sorted(args.input_dir.glob("*.jpg")):
        match = DISTANCE_PATTERN.match(path.name)
        if not match:
            continue

        # 提取真实距离标签
        truth_mm = int(match.group("distance_mm"))
        row: dict[str, object] = {
            "image": path.name,
            "truth_mm": truth_mm,
            "raw_mm": "",
            "corrected_mm": "",
            "error_mm": "",
            "reprojection_px": "",
            "status": "FAIL",
        }

        try:
            raw_image = cv2.imread(str(path))
            if raw_image is None:
                raise ValueError("cannot read image")

            # 去除鱼眼弯曲畸变
            undistorted = undistort_fisheye_image(raw_image, fisheye)
            height, width = undistorted.shape[:2]

            # 截取垂直中心条带
            strip_width = round(width * args.strip_fraction)
            left = (width - strip_width) // 2

            # 在条带上寻找 A4 黑边框
            plane = find_target_plane(undistorted[:, left : left + strip_width])
            corners = plane.corners.astype(np.float64)
            # 还原到原图像素横坐标
            corners[:, 0] += left

            # PnP 解算三维距离并进行线性斜率/截距校准
            estimate = estimate_plane_center_depth(corners, camera_matrix, correction)
            error_mm = estimate.corrected_distance_mm - truth_mm

            # 记录数据行，误差在 50mm 以内判定为通过（PASS）
            row.update(
                raw_mm=f"{estimate.raw_distance_mm:.3f}",
                corrected_mm=f"{estimate.corrected_distance_mm:.3f}",
                error_mm=f"{error_mm:.3f}",
                reprojection_px=f"{estimate.reprojection_error_px:.4f}",
                status="PASS" if abs(error_mm) <= 50.0 else "FAIL",
            )
        except Exception as exc:
            row["error_mm"] = str(exc)
        rows.append(row)

    # 3. 写入 CSV 分析报告
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=("image", "truth_mm", "raw_mm", "corrected_mm", "error_mm", "reprojection_px", "status"))
        writer.writeheader()
        writer.writerows(rows)

    # 4. 统计并计算平均绝对误差与最大偏差值
    numeric_errors = np.array([
        float(row["error_mm"])
        for row in rows
        if row["status"] in {"PASS", "FAIL"}
        and isinstance(row["error_mm"], str)
        and re.fullmatch(r"-?\d+(?:\.\d+)?", row["error_mm"])
    ])
    passed = sum(row["status"] == "PASS" for row in rows)

    print(f"summary: {passed}/{len(rows)} within 50 mm")
    if numeric_errors.size:
        print(f"mae_mm: {np.mean(np.abs(numeric_errors)):.3f}")
        print(f"max_error_mm: {np.max(np.abs(numeric_errors)):.3f}")
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
