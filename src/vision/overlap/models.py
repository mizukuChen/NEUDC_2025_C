"""重叠正方形识别的公共结果模型。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.vision.lowres_preprocess import LowResolutionA4


@dataclass(frozen=True)
class ReconstructedSquare:
    """高分辨率标准图中重建出的正方形。"""

    corners: np.ndarray
    center_mm: tuple[float, float]
    side_mm: float
    angle_deg: float
    confidence: float


@dataclass(frozen=True)
class OverlapMeasurementResult:
    """高分辨率重叠正方形检测结果。"""

    squares: tuple[ReconstructedSquare, ...]
    edge_image: np.ndarray


@dataclass(frozen=True)
class LowResolutionSquare:
    """低分辨率多帧输入中重建出的正方形。"""

    corners: np.ndarray
    center_mm: tuple[float, float]
    side_mm: float
    angle_deg: float
    confidence: float
    edge_supports: tuple[float, float, float, float]
    source_hits: int = 1


@dataclass(frozen=True)
class LowResolutionOverlapResult:
    """低分辨率多帧重叠正方形检测结果。"""

    squares: tuple[LowResolutionSquare, ...]
    prepared: LowResolutionA4
    edge_image: np.ndarray
    line_count: int
    candidate_count: int
