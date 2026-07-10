from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import cv2
import numpy as np

from src.vision.rectify import PX_PER_MM


BORDER_WIDTH_MM = 20.0


@dataclass(frozen=True)
class BasicMeasurement:
    shape_type: str
    x_mm: float
    area_px: float
    contour: np.ndarray


def _content_roi(rectified: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    border_px = int(BORDER_WIDTH_MM * PX_PER_MM)
    margin_px = int(5 * PX_PER_MM)
    x0 = border_px + margin_px
    y0 = border_px + margin_px
    x1 = rectified.shape[1] - border_px - margin_px
    y1 = rectified.shape[0] - border_px - margin_px
    return rectified[y0:y1, x0:x1], (x0, y0)


def _shape_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _classify_shape(contour: np.ndarray) -> str:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
    vertices = len(approx)

    area = cv2.contourArea(contour)
    circularity = 4.0 * pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0

    if vertices == 3:
        return "triangle"
    if vertices == 4:
        return "square"
    if circularity > 0.78:
        return "circle"
    return "unknown"


def _size_from_area(shape_type: str, area_px: float) -> float:
    area_mm2 = area_px / (PX_PER_MM * PX_PER_MM)
    if shape_type == "circle":
        return 2.0 * sqrt(area_mm2 / pi)
    if shape_type == "triangle":
        return sqrt(4.0 * area_mm2 / sqrt(3.0))
    if shape_type == "square":
        return sqrt(area_mm2)
    return sqrt(area_mm2)


def measure_basic_shape(rectified: np.ndarray) -> BasicMeasurement:
    roi, (x0, y0) = _content_roi(rectified)
    mask = _shape_mask(roi)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No inner shape contour found.")

    roi_area = roi.shape[0] * roi.shape[1]
    valid = [c for c in contours if cv2.contourArea(c) > roi_area * 0.01]
    if not valid:
        raise ValueError("No valid inner shape contour found.")

    contour = max(valid, key=cv2.contourArea)
    area_px = cv2.contourArea(contour)
    shape_type = _classify_shape(contour)
    x_mm = _size_from_area(shape_type, area_px)

    contour_full = contour.copy()
    contour_full[:, 0, 0] += x0
    contour_full[:, 0, 1] += y0

    return BasicMeasurement(
        shape_type=shape_type,
        x_mm=x_mm,
        area_px=area_px,
        contour=contour_full,
    )


def draw_basic_measurement_debug(rectified: np.ndarray, measurement: BasicMeasurement) -> np.ndarray:
    debug = rectified.copy()
    cv2.drawContours(debug, [measurement.contour], -1, (0, 0, 255), 4)
    label = f"{measurement.shape_type}: {measurement.x_mm:.1f} mm"
    cv2.putText(
        debug,
        label,
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (0, 0, 255),
        4,
        cv2.LINE_AA,
    )
    return debug
