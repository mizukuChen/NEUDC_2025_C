from __future__ import annotations

import unittest

import numpy as np

from src.vision.overlap.geometry import (
    edge_support,
    outer_band_mean,
    polygon_iou,
    square_angle_error,
)


class OverlapGeometryTest(unittest.TestCase):
    def test_square_angle_error_wraps_every_ninety_degrees(self) -> None:
        self.assertAlmostEqual(square_angle_error(44.0, -44.0), 2.0)
        self.assertAlmostEqual(square_angle_error(10.0, 100.0), 0.0)

    def test_polygon_iou_handles_equal_and_disjoint_squares(self) -> None:
        first = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float32)
        second = first + np.array([20, 0], np.float32)
        self.assertAlmostEqual(polygon_iou(first, first), 1.0)
        self.assertAlmostEqual(polygon_iou(first, second), 0.0)

    def test_outer_band_mean_ignores_polygon_center(self) -> None:
        values = np.ones((30, 30), np.float32)
        values[10:20, 10:20] = 0.0
        polygon = np.array([[2, 2], [27, 2], [27, 27], [2, 27]], np.float32)
        self.assertAlmostEqual(outer_band_mean(values, polygon), 1.0)

    def test_edge_support_samples_candidate_edge(self) -> None:
        distance = np.full((20, 20), 10.0, np.float32)
        distance[5, 2:18] = 0.0
        support = edge_support(
            distance,
            np.array([2.0, 5.0]),
            np.array([17.0, 5.0]),
            tolerance=0.5,
        )
        self.assertAlmostEqual(support, 1.0)


if __name__ == "__main__":
    unittest.main()
