from __future__ import annotations

"""
标准 A4 靶标形状与尺寸测量单元测试套件。

本测试套件覆盖单图形及多分离图形测量功能，通过当前模块化预设即时渲染标准靶图，
断言验证 `measure_shapes` 算法对以下情形的分类与特征边长计算精度：
- 基础单图形（圆形 120mm、等边三角形 120mm、正方形 120mm）
- 不同尺寸的单正方形（60mm、80mm、100mm）
- 各种分离排版的多正方形（边长 46mm~78mm 不等）
"""

import unittest

import cv2

from scripts.targets.generation.renderer import render
from scripts.targets.generation.shapes import ShapeSpec
from scripts.targets.presets.basic import PRESETS as BASIC_PRESETS
from scripts.targets.presets.separate import PRESETS as SEPARATE_PRESETS
from src.vision.measure_shapes import measure_shapes


class MeasureShapesTest(unittest.TestCase):
    """
    几何图形尺寸测量算法测试类。
    """

    def measure(self, shapes: tuple[ShapeSpec, ...]):
        gray, _ = render(shapes)
        return measure_shapes(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    def test_basic_shapes(self) -> None:
        """
        测试基础圆形、三角形、正方形的单目标分类与 120mm 物理尺寸测量精度。
        要求实测特征边长/直径误差在 1.0mm 以内。
        """
        for shape_type in ("circle", "triangle", "square"):
            with self.subTest(shape_type=shape_type):
                result = self.measure(BASIC_PRESETS[shape_type])
                self.assertEqual(result.layout, "single")
                self.assertEqual(len(result.shapes), 1)
                self.assertEqual(result.shapes[0].shape_type, shape_type)
                self.assertAlmostEqual(result.shapes[0].x_mm, 120.0, delta=1.0)

    def test_single_squares(self) -> None:
        """
        测试 60mm、80mm、100mm 各种物理大小的单正方形边长测量。
        """
        for size_mm in (60, 80, 100):
            with self.subTest(size_mm=size_mm):
                result = self.measure((ShapeSpec("square", 105, 148.5, size_mm),))
                self.assertEqual(len(result.shapes), 1)
                self.assertEqual(result.shapes[0].shape_type, "square")
                self.assertAlmostEqual(result.shapes[0].x_mm, size_mm, delta=1.0)

    def test_separate_squares(self) -> None:
        """
        测试多正方形分离布局下，各个正方形的检测数量与独立尺寸测量的精确度。
        """
        for case, shapes in SEPARATE_PRESETS.items():
            with self.subTest(case=case):
                result = self.measure(shapes)
                sizes = tuple(item.size_mm for item in shapes)
                self.assertEqual(result.layout, "separate")
                self.assertEqual(len(result.shapes), len(sizes))
                self.assertTrue(all(item.shape_type == "square" for item in result.shapes))
                measured = sorted(item.x_mm for item in result.shapes)
                # 一一比对尺寸，要求误差均在 1.0mm 内
                for actual, target in zip(measured, sorted(sizes)):
                    self.assertAlmostEqual(actual, target, delta=1.0)


if __name__ == "__main__":
    unittest.main()
