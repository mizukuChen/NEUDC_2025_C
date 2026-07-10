from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TargetPlane:
    corners: np.ndarray
    contour_area: float


def order_corners(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(4)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def dark_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, fixed = cv2.threshold(blurred, 95, 255, cv2.THRESH_BINARY_INV)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        9,
    )
    mask = cv2.bitwise_or(fixed, cv2.bitwise_and(otsu, adaptive))

    height, width = mask.shape[:2]
    kernel_size = max(3, round(min(height, width) * 0.015))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _contour_to_quad(contour: np.ndarray) -> np.ndarray | None:
    perimeter = cv2.arcLength(contour, True)
    for ratio in (0.02, 0.03, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(contour, ratio * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    if cv2.contourArea(box.astype(np.float32)) <= 0:
        return None
    return box.astype(np.float32)


def _border_score(mask: np.ndarray, quad: np.ndarray) -> float:
    ordered = order_corners(quad)
    width, height = 210, 297
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(mask, matrix, (width, height))

    border = 20
    border_mask = np.zeros((height, width), dtype=np.uint8)
    border_mask[:border, :] = 1
    border_mask[-border:, :] = 1
    border_mask[:, :border] = 1
    border_mask[:, -border:] = 1

    inner_margin = 35
    inner = warped[inner_margin : height - inner_margin, inner_margin : width - inner_margin]
    border_mean = float(warped[border_mask == 1].mean()) / 255.0
    inner_mean = float(inner.mean()) / 255.0 if inner.size else 0.0
    return border_mean - 0.35 * inner_mean


def _find_target_plane_once(image: np.ndarray, min_score: float = 0.35) -> TargetPlane:
    mask = dark_mask(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No dark contour found.")

    image_area = image.shape[0] * image.shape[1]
    candidates: list[tuple[float, float, np.ndarray, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.003:
            continue

        quad = _contour_to_quad(contour)
        if quad is None:
            continue

        quad_area = abs(cv2.contourArea(quad))
        if quad_area < image_area * 0.006 or quad_area > image_area * 0.90:
            continue

        x, y, w, h = cv2.boundingRect(quad.astype(np.int32))
        edge_margin = max(2, round(min(image.shape[:2]) * 0.015))
        if (
            x <= edge_margin
            or y <= edge_margin
            or x + w >= image.shape[1] - edge_margin
            or y + h >= image.shape[0] - edge_margin
        ):
            continue
        if w < image.shape[1] * 0.08 or h < image.shape[0] * 0.08:
            continue
        if h / max(w, 1) < 0.8 or h / max(w, 1) > 2.4:
            continue

        score = _border_score(mask, quad)
        if score < min_score:
            continue

        candidates.append((score, quad_area, quad, area))

    if not candidates:
        raise ValueError("No valid A4 border candidate found.")

    _, _, corners, contour_area = max(candidates, key=lambda item: (item[0], item[1]))
    return TargetPlane(corners=order_corners(corners), contour_area=contour_area)


def _candidate_regions(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    return [
        (0, 0, width, height),
        (0, height // 3, width, height - height // 3),
        (width // 4, height // 3, width // 2, height // 2),
        (width // 5, height // 4, width * 3 // 5, height * 3 // 5),
    ]


def find_target_plane(image: np.ndarray) -> TargetPlane:
    errors: list[str] = []
    for x, y, width, height in _candidate_regions(image):
        crop = image[y : y + height, x : x + width]
        try:
            plane = _find_target_plane_once(crop)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        corners = plane.corners.copy()
        corners[:, 0] += x
        corners[:, 1] += y
        return TargetPlane(corners=order_corners(corners), contour_area=plane.contour_area)

    raise ValueError(errors[-1] if errors else "No valid A4 border candidate found.")


def draw_target_plane_debug(image: np.ndarray, plane: TargetPlane) -> np.ndarray:
    debug = image.copy()
    corners = plane.corners.astype(np.int32)
    cv2.polylines(debug, [corners], isClosed=True, color=(0, 255, 0), thickness=3)

    labels = ("TL", "TR", "BR", "BL")
    for label, (x, y) in zip(labels, corners):
        cv2.circle(debug, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(
            debug,
            label,
            (int(x) + 6, int(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return debug
