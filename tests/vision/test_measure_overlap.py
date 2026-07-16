from __future__ import annotations

"""
重叠正方形重建与测量单元测试套件。

本测试套件包含：
1. 校验生成器预设用例中正方形的单边最大遮挡比率不超过 50%；
2. 动态渲染这些重叠用例，调用主线 `reconstruct_overlapping_squares` 重建算法，
   比对并断言重建出的各正方形物理边长与预期设计尺寸完全吻合（误差在 1.0mm 内）。
"""

import unittest

import cv2

from scripts.targets.generate_overlap_squares import (
    MAX_EDGE_COVERAGE,
    PRESETS,
    edge_coverage_ratios,
    render,
)
from src.vision.overlap.standard import reconstruct_overlapping_squares


class OverlapMeasurementTest(unittest.TestCase):
    """
    重叠正方形重建算法测试类。
    """

    def test_presets_keep_every_edge_more_than_half_visible(self) -> None:
        """
        验证预设的所有重叠测试用例中，每个正方形的每条边被其他正方形遮挡的比例是否低于 50%。
        """
        for case, squares in PRESETS.items():
            with self.subTest(case=case):
                ratios = edge_coverage_ratios(squares)
                self.assertLess(max(max(values) for values in ratios), MAX_EDGE_COVERAGE)

    def test_controlled_overlap_sizes(self) -> None:
        """
        动态将预设的多正方形重叠参数渲染为 PNG 图片，调用重叠识别算法，
        比对解算出的边长尺寸与 Ground Truth，断言误差在 1.0mm 以内。
        """
        for case, squares in PRESETS.items():
            with self.subTest(case=case):
                gray = render(squares)
                image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                # 运行重叠正方形重建算法
                result = reconstruct_overlapping_squares(image)

                actual = sorted(item.side_mm for item in result.squares)
                expected = sorted(item.size_mm for item in squares)
                self.assertEqual(len(actual), len(expected))

                # 逐个断言比对正方形物理尺寸
                for measured, target in zip(actual, expected):
                    self.assertAlmostEqual(measured, target, delta=1.0)


if __name__ == "__main__":
    unittest.main()
