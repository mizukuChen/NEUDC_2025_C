from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from math import atan2, degrees, hypot

import cv2
import numpy as np

from src.vision.lowres_preprocess import (
    LowResolutionA4,
    prepare_low_resolution_a4,
)


MIN_SIDE_MM = 50.0
MAX_SIDE_MM = 125.0
CONTENT_INSET_MM = 20.0


@dataclass(frozen=True)
class LowResolutionSquare:
    corners: np.ndarray
    center_mm: tuple[float, float]
    side_mm: float
    angle_deg: float
    confidence: float
    edge_supports: tuple[float, float, float, float]
    source_hits: int = 1


@dataclass(frozen=True)
class LowResolutionOverlapResult:
    squares: tuple[LowResolutionSquare, ...]
    prepared: LowResolutionA4
    edge_image: np.ndarray
    line_count: int
    candidate_count: int


@dataclass(frozen=True)
class _Line:
    normal: np.ndarray
    rho: float
    support: float
    start: np.ndarray
    end: np.ndarray


def _line_from_segment(segment: np.ndarray) -> _Line | None:
    x1, y1, x2, y2 = (float(value) for value in segment)
    dx, dy = x2 - x1, y2 - y1
    length = hypot(dx, dy)
    if length <= 0:
        return None
    normal = np.array([-dy / length, dx / length], dtype=np.float64)
    rho = float(normal @ np.array([x1, y1], dtype=np.float64))
    dominant = int(np.argmax(np.abs(normal)))
    if normal[dominant] < 0:
        normal = -normal
        rho = -rho
    return _Line(
        normal,
        rho,
        length,
        np.array([x1, y1], dtype=np.float64),
        np.array([x2, y2], dtype=np.float64),
    )


def _align_line(reference: _Line, line: _Line) -> _Line:
    """统一两条近似平行线的法向；翻转法向时必须同时翻转rho。"""
    if float(reference.normal @ line.normal) >= 0:
        return line
    return _Line(-line.normal, -line.rho, line.support, line.start, line.end)


def _fit_lines(lines: tuple[_Line, ...] | list[_Line]) -> _Line:
    points = np.concatenate(
        [np.stack([line.start, line.end]) for line in lines],
        axis=0,
    ).astype(np.float32)
    vx, vy, x0, y0 = (
        float(value)
        for value in cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    )
    direction = np.array([vx, vy], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    dominant = int(np.argmax(np.abs(normal)))
    if normal[dominant] < 0:
        normal = -normal
        direction = -direction
    origin = np.array([x0, y0], dtype=np.float64)
    projections = (points.astype(np.float64) - origin) @ direction
    start = origin + direction * float(np.min(projections))
    end = origin + direction * float(np.max(projections))
    return _Line(
        normal,
        float(normal @ origin),
        float(sum(line.support for line in lines)),
        start,
        end,
    )


def _projection_gap(first: _Line, second: _Line) -> float:
    direction = np.array([-first.normal[1], first.normal[0]], dtype=np.float64)
    first_values = sorted((float(first.start @ direction), float(first.end @ direction)))
    second_values = sorted((float(second.start @ direction), float(second.end @ direction)))
    return max(first_values[0], second_values[0]) - min(first_values[1], second_values[1])


def _bridge_fragmented_lines(lines: list[_Line], scale: float) -> list[_Line]:
    """为同一物理边两端的短线段补充联合拟合，不删除原始线段。"""
    bridged: list[_Line] = []
    for first, second in combinations(lines, 2):
        if _angle_error(first, second) > np.deg2rad(20.0):
            continue
        second = _align_line(first, second)
        if abs(first.rho - second.rho) > 35.0 * scale:
            continue
        gap = _projection_gap(first, second)
        if not -2.0 * scale <= gap <= 65.0 * scale:
            continue
        fitted = _fit_lines((first, second))
        points = np.stack([first.start, first.end, second.start, second.end])
        residual = np.max(np.abs(points @ fitted.normal - fitted.rho))
        if fitted.support >= 12.0 * scale and residual <= 2.0 * scale:
            bridged.append(fitted)
    return lines + bridged


def _angle_error(first: _Line, second: _Line) -> float:
    dot = float(np.clip(abs(first.normal @ second.normal), 0.0, 1.0))
    return float(np.arccos(dot))


def _merge_lines(lines: list[_Line], angle_tolerance: float, rho_tolerance: float) -> list[_Line]:
    merged: list[_Line] = []
    for line in sorted(lines, key=lambda item: item.support, reverse=True):
        match = next(
            (
                index
                for index, current in enumerate(merged)
                if _angle_error(line, current) <= angle_tolerance
                and abs(line.rho - current.rho) <= rho_tolerance
            ),
            None,
        )
        if match is None:
            merged.append(line)
            continue
        current = merged[match]
        merged[match] = _fit_lines((current, _align_line(current, line)))
    return merged


def _intersection(first: _Line, second: _Line) -> np.ndarray | None:
    matrix = np.stack([first.normal, second.normal])
    if abs(float(np.linalg.det(matrix))) < 1e-6:
        return None
    return np.linalg.solve(matrix, np.array([first.rho, second.rho], dtype=np.float64))


def _common_parallel_pair(first: _Line, second: _Line) -> tuple[_Line, _Line]:
    second = _align_line(first, second)
    normal = first.normal * first.support + second.normal * second.support
    normal /= np.linalg.norm(normal)

    def project(line: _Line) -> _Line:
        midpoint = (line.start + line.end) / 2.0
        return _Line(
            normal,
            float(normal @ midpoint),
            line.support,
            line.start,
            line.end,
        )

    return project(first), project(second)


def _square_from_pairs(first_pair, second_pair) -> np.ndarray | None:
    first_a, second_a, _ = first_pair
    first_b, second_b, _ = second_pair
    second_a = _align_line(first_a, second_a)
    second_b = _align_line(first_b, second_b)
    normal_a = first_a.normal * first_a.support + second_a.normal * second_a.support
    normal_b = first_b.normal * first_b.support + second_b.normal * second_b.support
    normal_a /= np.linalg.norm(normal_a)
    normal_b /= np.linalg.norm(normal_b)
    lines_a = (
        _Line(
            normal_a,
            float(normal_a @ ((first_a.start + first_a.end) / 2.0)),
            first_a.support,
            first_a.start,
            first_a.end,
        ),
        _Line(
            normal_a,
            float(normal_a @ ((second_a.start + second_a.end) / 2.0)),
            second_a.support,
            second_a.start,
            second_a.end,
        ),
    )
    lines_b = (
        _Line(
            normal_b,
            float(normal_b @ ((first_b.start + first_b.end) / 2.0)),
            first_b.support,
            first_b.start,
            first_b.end,
        ),
        _Line(
            normal_b,
            float(normal_b @ ((second_b.start + second_b.end) / 2.0)),
            second_b.support,
            second_b.start,
            second_b.end,
        ),
    )
    corners = []
    for line_a, line_b in (
        (lines_a[0], lines_b[0]),
        (lines_a[1], lines_b[0]),
        (lines_a[1], lines_b[1]),
        (lines_a[0], lines_b[1]),
    ):
        point = _intersection(line_a, line_b)
        if point is None:
            return None
        corners.append(point)
    return np.asarray(corners, dtype=np.float32)


def _edge_support(distance: np.ndarray, start: np.ndarray, end: np.ndarray, tolerance: float) -> float:
    length = float(np.linalg.norm(end - start))
    count = max(20, round(length / 2.0))
    points = np.linspace(start, end, count)
    xs = np.clip(np.rint(points[:, 0]).astype(int), 0, distance.shape[1] - 1)
    ys = np.clip(np.rint(points[:, 1]).astype(int), 0, distance.shape[0] - 1)
    return float(np.mean(distance[ys, xs] <= tolerance))


def _outer_band_probability(probability: np.ndarray, polygon: np.ndarray) -> float:
    integer = np.rint(polygon).astype(np.int32)
    x, y, width, height = cv2.boundingRect(integer)
    x0, y0 = max(0, x), max(0, y)
    x1 = min(probability.shape[1], x + width)
    y1 = min(probability.shape[0], y + height)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    shifted = integer - np.array([x0, y0], dtype=np.int32)
    cv2.fillConvexPoly(mask, shifted, 255)
    center = shifted.astype(np.float32).mean(axis=0)
    inner = center + (shifted.astype(np.float32) - center) * 0.68
    cv2.fillConvexPoly(mask, np.rint(inner).astype(np.int32), 0)
    active = mask > 0
    if not np.any(active):
        return 0.0
    return float(np.mean(probability[y0:y1, x0:x1][active]))


def _polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    area_first = abs(cv2.contourArea(first.astype(np.float32)))
    area_second = abs(cv2.contourArea(second.astype(np.float32)))
    intersection, _ = cv2.intersectConvexConvex(first.astype(np.float32), second.astype(np.float32))
    union = area_first + area_second - intersection
    return float(intersection / union) if union > 0 else 0.0


def _same_square(first: LowResolutionSquare, second: LowResolutionSquare) -> bool:
    center_distance = hypot(
        first.center_mm[0] - second.center_mm[0],
        first.center_mm[1] - second.center_mm[1],
    )
    angle_error = abs((first.angle_deg - second.angle_deg + 45.0) % 90.0 - 45.0)
    return (
        center_distance <= 5.0
        and abs(first.side_mm - second.side_mm) <= 5.0
        and angle_error <= 5.0
    ) or _polygon_iou(first.corners, second.corners) >= 0.78


def _deduplicate(candidates: list[LowResolutionSquare]) -> list[LowResolutionSquare]:
    kept: list[LowResolutionSquare] = []
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        if any(_same_square(candidate, current) for current in kept):
            continue
        kept.append(candidate)
    return kept


def _candidate_from_geometry(
    polygon: np.ndarray,
    side_a: float,
    side_b: float,
    prepared: LowResolutionA4,
    edges: np.ndarray,
    inset: int,
) -> LowResolutionSquare | None:
    scale = prepared.pixels_per_mm
    roi_shape = edges.shape
    margin = 3.0 * scale
    if np.any(polygon < -margin):
        return None
    if np.any(polygon[:, 0] >= roi_shape[1] + margin) or np.any(
        polygon[:, 1] >= roi_shape[0] + margin
    ):
        return None
    distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    supports = tuple(
        _edge_support(distance, polygon[index], polygon[(index + 1) % 4], 1.75 * scale)
        for index in range(4)
    )
    mean_support = float(np.mean(supports))
    min_support = float(np.min(supports))
    probability_roi = prepared.black_probability[inset:-inset, inset:-inset]
    interior = _outer_band_probability(probability_roi, polygon)
    if min_support < 0.08 or mean_support < 0.24 or interior < 0.45:
        return None

    full_polygon = polygon.copy()
    full_polygon += inset
    center = full_polygon.mean(axis=0)
    side_mm = (side_a + side_b) / 2.0
    direction = atan2(*(full_polygon[1] - full_polygon[0])[::-1])
    angle = (degrees(direction) + 45.0) % 90.0 - 45.0
    geometry_score = max(0.0, 1.0 - abs(side_a - side_b) / 7.0)
    confidence = (
        0.42 * mean_support
        + 0.23 * min_support
        + 0.20 * interior
        + 0.15 * geometry_score
    )
    return LowResolutionSquare(
        corners=full_polygon,
        center_mm=(float(center[0] / scale), float(center[1] / scale)),
        side_mm=side_mm,
        angle_deg=angle,
        confidence=confidence,
        edge_supports=supports,
    )


def _extract_candidates(
    prepared: LowResolutionA4,
) -> tuple[list[LowResolutionSquare], np.ndarray, int]:
    scale = prepared.pixels_per_mm
    probability_u8 = np.rint(prepared.black_probability * 255.0).astype(np.uint8)
    inset = round(CONTENT_INSET_MM * scale)
    roi = probability_u8[inset:-inset, inset:-inset]
    edges = cv2.Canny(roi, 35, 90)

    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(roi)[0]
    segments = np.empty((0, 4), dtype=np.float32) if detected is None else detected.reshape(-1, 4)
    minimum_length = max(8.0, 5.0 * scale)
    raw_lines = []
    for segment in segments:
        line = _line_from_segment(segment)
        if line is not None and line.support >= minimum_length:
            raw_lines.append(line)
    base_lines = _merge_lines(raw_lines, np.deg2rad(2.0), 1.5 * scale)
    lines = _bridge_fragmented_lines(base_lines, scale)
    lines = _merge_lines(lines, np.deg2rad(1.0), 0.75 * scale)
    lines = sorted(lines, key=lambda item: item.support, reverse=True)[:100]

    parallel_pairs = []
    for first, second in combinations(lines, 2):
        if _angle_error(first, second) > np.deg2rad(5.0):
            continue
        first, second = _common_parallel_pair(first, second)
        separation_mm = abs(first.rho - second.rho) / scale
        if MIN_SIDE_MM <= separation_mm <= MAX_SIDE_MM:
            parallel_pairs.append((first, second, separation_mm))

    candidates: list[LowResolutionSquare] = []
    for pair_a, pair_b in combinations(parallel_pairs, 2):
        perpendicular_error = abs(_angle_error(pair_a[0], pair_b[0]) - np.pi / 2.0)
        if perpendicular_error > np.deg2rad(7.0):
            continue
        if abs(pair_a[2] - pair_b[2]) > 9.0:
            continue
        polygon = _square_from_pairs(pair_a, pair_b)
        if polygon is None:
            continue
        candidate = _candidate_from_geometry(
            polygon,
            pair_a[2],
            pair_b[2],
            prepared,
            edges,
            inset,
        )
        if candidate is not None:
            candidates.append(candidate)

    edge_full = np.zeros_like(probability_u8)
    edge_full[inset:-inset, inset:-inset] = edges
    return _deduplicate(candidates), edge_full, len(lines)


def _verify_on_fused(
    candidate: LowResolutionSquare,
    prepared: LowResolutionA4,
    edge_image: np.ndarray,
    source_hits: int,
) -> LowResolutionSquare | None:
    scale = prepared.pixels_per_mm
    inset = round(CONTENT_INSET_MM * scale)
    polygon = candidate.corners - inset
    edges = edge_image[inset:-inset, inset:-inset]
    side_lengths = [
        float(np.linalg.norm(polygon[(index + 1) % 4] - polygon[index]) / scale)
        for index in range(4)
    ]
    side_a = (side_lengths[0] + side_lengths[2]) / 2.0
    side_b = (side_lengths[1] + side_lengths[3]) / 2.0
    verified = _candidate_from_geometry(polygon, side_a, side_b, prepared, edges, inset)
    if verified is None:
        return None
    supports = verified.edge_supports
    ordered_supports = sorted(supports, reverse=True)
    probability_roi = prepared.black_probability[inset:-inset, inset:-inset]
    inner_boundary_probability = _outer_band_probability(probability_roi, polygon)
    if not MIN_SIDE_MM <= verified.side_mm <= MAX_SIDE_MM:
        return None
    if (
        min(supports) < 0.48
        or float(np.mean(supports)) < 0.36
        or ordered_supports[0] < 0.78
        or ordered_supports[1] < 0.65
        or inner_boundary_probability < 0.75
    ):
        return None
    return LowResolutionSquare(
        corners=verified.corners,
        center_mm=verified.center_mm,
        side_mm=verified.side_mm,
        angle_deg=verified.angle_deg,
        confidence=verified.confidence,
        edge_supports=verified.edge_supports,
        source_hits=source_hits,
    )


def reconstruct_low_resolution_squares(
    frames: tuple[np.ndarray, ...] | list[np.ndarray],
    confidence_threshold: float = 0.70,
) -> LowResolutionOverlapResult:
    """逐帧宽松生成候选，再在融合图上严格验证重叠正方形。"""
    prepared = prepare_low_resolution_a4(frames)
    frame_list = list(frames)

    def extract_frame(frame: np.ndarray):
        return _extract_candidates(prepare_low_resolution_a4([frame]))

    # 四次LSD互不依赖：融合图留在主线程，三张单帧同时使用其余CPU核心。
    # OpenCV在树莓派默认还会为每次调用开4个内部线程；并行区临时限制为
    # 每路1线程，避免4路任务膨胀成16个计算线程并造成严重调度波动。
    previous_cv_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        with ThreadPoolExecutor(max_workers=max(1, min(3, len(frame_list)))) as executor:
            frame_futures = [executor.submit(extract_frame, frame) for frame in frame_list]
            fused_candidates, edge_full, fused_line_count = _extract_candidates(prepared)
            frame_results = [future.result() for future in frame_futures]
    finally:
        cv2.setNumThreads(previous_cv_threads)

    sources = [(0, candidate) for candidate in fused_candidates]
    for source_index, (frame_candidates, _, _) in enumerate(frame_results, 1):
        sources.extend((source_index, candidate) for candidate in frame_candidates)

    groups: list[tuple[list[LowResolutionSquare], set[int]]] = []
    for source_index, candidate in sorted(
        sources,
        key=lambda item: item[1].confidence,
        reverse=True,
    ):
        match = next(
            (
                index
                for index, (current, _) in enumerate(groups)
                if any(_same_square(candidate, item) for item in current)
            ),
            None,
        )
        if match is None:
            groups.append(([candidate], {source_index}))
        else:
            groups[match][0].append(candidate)
            groups[match][1].add(source_index)

    verified_candidates = []
    for candidates, source_indexes in groups:
        verified_options = [
            verified
            for candidate in candidates
            if (
                verified := _verify_on_fused(
                    candidate,
                    prepared,
                    edge_full,
                    len(source_indexes),
                )
            )
            is not None
        ]
        if not verified_options:
            continue
        verified = max(verified_options, key=lambda item: item.confidence)
        if len(source_indexes) == 1 and verified.confidence < 0.78:
            continue
        if verified.confidence >= confidence_threshold:
            verified_candidates.append(verified)

    squares = _deduplicate(verified_candidates)
    squares.sort(key=lambda item: (item.center_mm[1], item.center_mm[0]))
    return LowResolutionOverlapResult(
        squares=tuple(squares),
        prepared=prepared,
        edge_image=edge_full,
        line_count=fused_line_count,
        candidate_count=len(groups),
    )


def draw_low_resolution_squares(
    image: np.ndarray,
    result: LowResolutionOverlapResult,
) -> np.ndarray:
    debug = image.copy()
    if debug.ndim == 2:
        debug = cv2.cvtColor(debug, cv2.COLOR_GRAY2BGR)
    for index, square in enumerate(result.squares, 1):
        corners = np.rint(square.corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(debug, [corners], True, (0, 200, 0), 2, cv2.LINE_AA)
        origin = tuple(corners.reshape(-1, 2).min(axis=0))
        cv2.putText(
            debug,
            f"{index}: {square.side_mm:.1f} mm",
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )
    return debug
