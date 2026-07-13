from __future__ import annotations

import unittest

import cv2

from scripts.targets.generate_overlap_squares import PRESETS, render
from scripts.targets.generation.renderer import render as render_shapes
from scripts.targets.presets.basic import PRESETS as BASIC_PRESETS
from scripts.targets.presets.separate import PRESETS as SEPARATE_PRESETS
from src.vision.measure import measure


class UnifiedMeasurementTest(unittest.TestCase):
    def test_basic_shapes_use_contours(self) -> None:
        for shape_type in ("circle", "triangle", "square"):
            with self.subTest(shape_type=shape_type):
                gray, _ = render_shapes(BASIC_PRESETS[shape_type])
                image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                result = measure(image)
                self.assertEqual(result.shapes[0].source, "contour")
                self.assertGreaterEqual(result.contour_foreground_coverage, 0.92)

    def test_separate_squares_use_contours(self) -> None:
        shapes = SEPARATE_PRESETS["02"]
        gray, _ = render_shapes(shapes)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        result = measure(image)
        self.assertEqual(result.layout, "separate")
        self.assertEqual(len(result.shapes), len(shapes))
        self.assertTrue(all(item.source == "contour" for item in result.shapes))

    def test_controlled_overlap_uses_line_reconstruction(self) -> None:
        for case, squares in PRESETS.items():
            with self.subTest(case=case):
                gray = render(squares)
                image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                result = measure(image)
                self.assertEqual(result.layout, "overlap")
                self.assertEqual(len(result.shapes), len(squares))
                self.assertTrue(all(item.source == "overlap_lines" for item in result.shapes))


if __name__ == "__main__":
    unittest.main()
