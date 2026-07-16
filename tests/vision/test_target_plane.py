from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.vision.target_plane import find_target_plane


def _strong_perspective_scene() -> tuple[np.ndarray, np.ndarray]:
    """生成黑框接近地面暗线、二次闭运算会发生粘连的侧视A4。"""
    page_width, page_height = 420, 594
    page = np.full((page_height, page_width, 3), 235, dtype=np.uint8)
    cv2.rectangle(page, (0, 0), (page_width - 1, page_height - 1), (20, 20, 20), -1)
    cv2.rectangle(page, (40, 40), (page_width - 41, page_height - 41), (235, 235, 235), -1)

    expected = np.asarray(
        [[378, 344], [433, 335], [426, 469], [372, 466]],
        dtype=np.float32,
    )
    source = np.asarray(
        [[0, 0], [page_width - 1, 0], [page_width - 1, page_height - 1], [0, page_height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, expected)
    canvas = np.full((1080, 640, 3), 170, dtype=np.uint8)
    warped = cv2.warpPerspective(page, transform, (640, 1080), borderValue=(170, 170, 170))
    page_mask = cv2.warpPerspective(
        np.full((page_height, page_width), 255, dtype=np.uint8),
        transform,
        (640, 1080),
    )
    canvas[page_mask > 0] = warped[page_mask > 0]

    # 模拟真实画面中的地砖缝：距黑框很近，但并未真正接触靶纸。
    cv2.line(canvas, (370, 475), (370, 1079), (30, 30, 30), 3)
    cv2.line(canvas, (0, 520), (639, 520), (60, 60, 60), 5)
    return canvas, expected


class TargetPlanePerspectiveTest(unittest.TestCase):
    def test_strong_perspective_fallback_recovers_a4(self) -> None:
        image, expected = _strong_perspective_scene()
        plane = find_target_plane(image)
        self.assertTrue(plane.strong_perspective)
        np.testing.assert_allclose(plane.corners, expected, atol=3.0)


if __name__ == "__main__":
    unittest.main()
