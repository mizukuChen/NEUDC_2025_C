from __future__ import annotations

import cv2
import numpy as np


A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
PX_PER_MM = 10
RECTIFIED_WIDTH_PX = int(A4_WIDTH_MM * PX_PER_MM)
RECTIFIED_HEIGHT_PX = int(A4_HEIGHT_MM * PX_PER_MM)


def rectify_a4(
    image: np.ndarray,
    corners: np.ndarray,
    width_px: int = RECTIFIED_WIDTH_PX,
    height_px: int = RECTIFIED_HEIGHT_PX,
) -> np.ndarray:
    src = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    dst = np.array(
        [
            [0, 0],
            [width_px - 1, 0],
            [width_px - 1, height_px - 1],
            [0, height_px - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (width_px, height_px))
