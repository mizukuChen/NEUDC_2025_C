from __future__ import annotations

"""
受控重叠正方形数据集精度评估脚本。

本脚本遍历 `assets/targets/generated/overlap_controlled/` 目录下的所有受控多正方形重叠靶纸图像，
调用 `reconstruct_overlapping_squares` 算法重建各重叠正方形。
随后，读取同名的 JSON ground truth 元数据，并使用 permutations 进行多对多匹配，
找出与真实正方形偏差最小的最佳对应，比对并统计：
- 识别到的正方形个数是否准确；
- 物理边长尺寸误差（size_mm）；
- 中心物理坐标误差（cx_mm, cy_mm）；
- 旋转倾斜角误差（angle_deg）。
"""

import json
import sys
from itertools import permutations
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.overlap.standard import draw_overlap_measurements, reconstruct_overlapping_squares


def _angle_error(first: float, second: float) -> float:
    """
    计算两个角度之间的最小夹角误差（限制在 -45 到 45 度的周期范围内）。
    """
    return abs((first - second + 45.0) % 90.0 - 45.0)


def main() -> int:
    source = ROOT / "assets" / "targets" / "generated" / "overlap_controlled"
    output = ROOT / "runtime" / "debug" / "overlap_controlled"
    output.mkdir(parents=True, exist_ok=True)
    failed = 0
    all_errors: list[tuple[float, float, float]] = []

    # 遍历所有 JSON 标注元文件
    for metadata_path in sorted(source.glob("overlap_controlled_*_A4.json")):
        image_path = metadata_path.with_suffix(".png")
        image = cv2.imread(str(image_path))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = metadata["squares"]

        if image is None:
            print(f"FAIL {image_path.name}: cannot read image")
            failed += 1
            continue

        # 1. 运行重叠重建检测算法
        result = reconstruct_overlapping_squares(image)
        actual = list(result.squares)
        matches = []

        # 2. 如果检测个数与标注数一致，使用全排列对齐匹配
        if len(actual) == len(expected):
            # 遍历所有可能的对应关系排列组合，挑选出累计误差最小的最优排布
            ordering = min(
                permutations(range(len(expected))),
                key=lambda order: sum(
                    abs(actual[index].side_mm - expected[target]["size_mm"])
                    + abs(actual[index].center_mm[0] - expected[target]["cx_mm"])
                    + abs(actual[index].center_mm[1] - expected[target]["cy_mm"])
                    + _angle_error(actual[index].angle_deg, expected[target]["angle_deg"])
                    for index, target in enumerate(order)
                ),
            )

            # 计算最佳对应排布下的误差明细
            for index, target in enumerate(ordering):
                measured = actual[index]
                truth = expected[target]
                matches.append(
                    (
                        abs(measured.side_mm - truth["size_mm"]), # 边长绝对误差 (mm)
                        ((measured.center_mm[0] - truth["cx_mm"]) ** 2 + (measured.center_mm[1] - truth["cy_mm"]) ** 2) ** 0.5, # 中心欧氏距离误差 (mm)
                        _angle_error(measured.angle_deg, truth["angle_deg"]), # 角度绝对误差 (deg)
                    )
                )

        # 3. 校验判定：所有匹配正方形的边长误差 <= 1.0mm 且中心偏离 <= 1.0mm 且偏角偏差 <= 0.6度
        ok = bool(matches) and all(size <= 1.0 and center <= 1.0 and angle <= 0.6 for size, center, angle in matches)
        all_errors.extend(matches)
        failed += int(not ok)

        # 4. 保存可视化调试图
        cv2.imwrite(str(output / f"{image_path.stem}.jpg"), draw_overlap_measurements(image, result))
        print(
            f"{'PASS' if ok else 'FAIL'} {image_path.name} "
            f"count={len(actual)}/{len(expected)} "
            f"max_error(size,center,angle)="
            f"{tuple(round(max(values), 3) for values in zip(*matches)) if matches else 'n/a'}"
        )

    # 5. 输出汇总评估报告
    if all_errors:
        columns = tuple(zip(*all_errors))
        maxima = tuple(max(values) for values in columns)
        means = tuple(sum(values) / len(values) for values in columns)
        print(
            f"SUMMARY squares={len(all_errors)} "
            f"mean_error(size_mm,center_mm,angle_deg)={tuple(round(value, 3) for value in means)} "
            f"max_error={tuple(round(value, 3) for value in maxima)}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
