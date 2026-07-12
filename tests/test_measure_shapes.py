from __future__ import annotations

"""
标准 A4 靶标形状与尺寸测量单元测试套件。

本测试套件覆盖单图形及多分离图形测量功能，通过加载生成的 A4 PNG 标准靶图，
断言验证 `measure_shapes` 算法对以下情形的分类与特征边长计算精度：
- 基础单图形（圆形 120mm、等边三角形 120mm、正方形 120mm）
- 不同尺寸的单正方形（60mm、80mm、100mm）
- 各种分离排版的多正方形（边长 46mm~78mm 不等）
"""

import unittest
from pathlib import Path

import cv2

from src.vision.measure_shapes import measure_shapes


ROOT = Path(__file__).resolve().parents[1]
BASIC = ROOT / "assets" / "targets" / "generated" / "png"
ADVANCED = ROOT / "assets" / "targets" / "generated" / "advanced" / "png"


class MeasureShapesTest(unittest.TestCase):
    """
    几何图形尺寸测量算法测试类。
    """

    def measure(self, path: Path):
        """
        读取图像并调用 measure_shapes 返回测量结果。
        """
        image = cv2.imread(str(path))
        self.assertIsNotNone(image, str(path))
        return measure_shapes(image)

    def test_basic_shapes(self) -> None:
        """
        测试基础圆形、三角形、正方形的单目标分类与 120mm 物理尺寸测量精度。
        要求实测特征边长/直径误差在 1.0mm 以内。
        """
        for shape_type in ("circle", "triangle", "square"):
            with self.subTest(shape_type=shape_type):
                result = self.measure(BASIC / f"basic_{shape_type}_A4.png")
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
                result = self.measure(ADVANCED / f"advanced_single_square_{size_mm}mm_A4.png")
                self.assertEqual(len(result.shapes), 1)
                self.assertEqual(result.shapes[0].shape_type, "square")
                self.assertAlmostEqual(result.shapes[0].x_mm, size_mm, delta=1.0)

    def test_separate_squares(self) -> None:
        """
        测试多正方形分离布局下，各个正方形的检测数量与独立尺寸测量的精确度。
        """
        # 预设各个测试用例对应的预期正方形实际尺寸 (mm)
        expected = {
            "01": (60, 68, 78),
            "02": (54, 58, 62, 70),
            "03": (46, 52, 56, 64, 68),
            "04": (54, 60, 64, 72),
        }
        for case, sizes in expected.items():
            with self.subTest(case=case):
                result = self.measure(ADVANCED / f"advanced_separate_squares_{case}_A4.png")
                self.assertEqual(result.layout, "separate")
                self.assertEqual(len(result.shapes), len(sizes))
                self.assertTrue(all(item.shape_type == "square" for item in result.shapes))
                measured = sorted(item.x_mm for item in result.shapes)
                # 一一比对尺寸，要求误差均在 1.0mm 内
                for actual, target in zip(measured, sorted(sizes)):
                    self.assertAlmostEqual(actual, target, delta=1.0)


if __name__ == "__main__":
    unittest.main()
