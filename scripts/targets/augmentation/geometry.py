from __future__ import annotations

import cv2
import numpy as np


def residual_perspective(
    image: np.ndarray,
    corner_offsets_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """模拟 A4 透视校正后仍残留的轻微四角定位误差。"""
    height, width = image.shape[:2]
    source = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    destination = source + np.asarray(corner_offsets_px, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return warped, matrix
