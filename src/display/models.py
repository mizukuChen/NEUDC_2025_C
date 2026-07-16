"""与具体 GUI 框架无关的测量展示结果。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeasurementView:
    """UI 可以直接展示的文字与图像集合。"""

    mode: str
    x_mm: float
    distance_mm: float | None
    reprojection_error_px: float
    summary: str
    target_text: str
    input_image: np.ndarray
    located_a4: np.ndarray
    rectified_a4: np.ndarray
    annotated_a4: np.ndarray
