from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.vision.hog import HOG_FEATURE_SIZE, PATCH_SIZE, extract_hog
from src.vision.linear_svm import LinearSvmModel, load_linear_svm

SQUARE_INNER_FRACTION = 0.84


def extract_digit_patch(
    image: np.ndarray,
    center_px: tuple[float, float],
    side_px: float,
    angle_deg: float,
) -> np.ndarray:
    """按检测到的方块中心和角度转正，提取中央白字黑底图。"""
    if side_px <= 0:
        raise ValueError("square side must be positive")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    matrix = cv2.getRotationMatrix2D(center_px, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        gray, matrix, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR, borderValue=255
    )
    crop_size = max(8, round(side_px * SQUARE_INNER_FRACTION))
    crop = cv2.getRectSubPix(rotated, (crop_size, crop_size), center_px)
    patch = cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
    _, patch = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return patch


def hog_features(patch: np.ndarray) -> np.ndarray:
    return extract_hog(patch)


@dataclass(frozen=True)
class DigitRecognition:
    digit: int
    confidence: float
    patch: np.ndarray


class HogSvmDigitRecognizer:
    def __init__(self, model: LinearSvmModel) -> None:
        if model.feature_count != HOG_FEATURE_SIZE:
            raise ValueError(
                f"digit model expects {model.feature_count} HOG features; "
                f"runtime produces {HOG_FEATURE_SIZE}"
            )
        if not np.array_equal(model.classes, np.arange(10, dtype=np.int32)):
            raise ValueError("digit model classes must be 0 through 9")
        self.model = model

    @classmethod
    def load(cls, model_path: Path) -> "HogSvmDigitRecognizer":
        return cls(load_linear_svm(model_path))

    def recognize_square(
        self,
        image: np.ndarray,
        center_px: tuple[float, float],
        side_px: float,
        angle_deg: float,
    ) -> DigitRecognition:
        patch = extract_digit_patch(image, center_px, side_px, angle_deg)
        features = hog_features(patch)
        scores = self.model.decision_function(features)
        digit = int(self.model.classes[int(np.argmax(scores))])
        best = np.partition(scores, -2)[-2:]
        margin = float(best.max() - best.min())
        confidence = float(1.0 / (1.0 + np.exp(-margin)))
        return DigitRecognition(digit, confidence, patch)
