from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from src.application import MeasurementApplication
from src.camera_settings import CameraSettings
from src.vision.lowres_preprocess import estimate_black_probability
from src.vision.measure_overlap_lowres import reconstruct_low_resolution_squares
from src.vision.rectify import rectify_a4
from src.vision.target_plane import find_target_plane


@dataclass(frozen=True)
class AutoTuneMetrics:
    ok: bool
    score: float
    message: str
    black_gray: float | None = None
    white_gray: float | None = None
    edge_pct: float | None = None
    line_count: int | None = None
    candidate_count: int | None = None
    square_count: int | None = None
    elapsed_s: float = 0.0


@dataclass(frozen=True)
class AutoTuneEvaluation:
    settings: CameraSettings
    metrics: AutoTuneMetrics


def auto_tune_candidates(base: CameraSettings) -> tuple[CameraSettings, ...]:
    """生成少量现场调参候选，优先处理锐化导致的伪边缘问题。"""
    sharpness_values = [0, 2, 4]
    if int(base.sharpness) not in sharpness_values:
        sharpness_values.append(int(base.sharpness))
    candidates: list[CameraSettings] = []
    for sharpness in sharpness_values:
        candidates.append(
            CameraSettings(
                auto_exposure=True,
                exposure_time_absolute=base.exposure_time_absolute,
                gain=base.gain,
                brightness=base.brightness,
                contrast=base.contrast,
                gamma=base.gamma,
                sharpness=sharpness,
                power_line_frequency=base.power_line_frequency,
                roi_left=base.roi_left,
                roi_top=base.roi_top,
                roi_right=base.roi_right,
                roi_bottom=base.roi_bottom,
            )
        )
    return tuple(dict.fromkeys(candidates))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def evaluate_camera_health(
    raw: np.ndarray,
    settings: CameraSettings,
    application: MeasurementApplication,
) -> AutoTuneMetrics:
    start = perf_counter()
    try:
        fisheye, _ = application._load()
        height, width = raw.shape[:2]
        fixed, interpolation, _, _, _ = application._camera_undistortion_maps(
            fisheye,
            (width, height),
        )
        undistorted = cv2.remap(
            raw,
            fixed,
            interpolation,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        left, top, right, bottom = settings.roi_pixels(width, height)
        roi = undistorted[top:bottom, left:right]
        plane = find_target_plane(roi)
        rectified = rectify_a4(roi, plane.corners, 420, 594)
        probability, black, white = estimate_black_probability(rectified)
        overlap = reconstruct_low_resolution_squares([rectified])
        edge_pct = float(np.mean(overlap.edge_image > 0) * 100.0)
        elapsed = perf_counter() - start
    except Exception as exc:
        return AutoTuneMetrics(
            ok=False,
            score=-1.0,
            message=f"诊断失败：{exc}",
            elapsed_s=perf_counter() - start,
        )

    # 经验目标来自现场验证：sharpness=0 时 edge_pct≈1.35%、line_count≈67，
    # 而 sharpness=6 时 line_count打满100且误检。评分只用于候选排序。
    line_health = _clamp01(1.0 - max(0.0, overlap.line_count - 70.0) / 35.0)
    edge_health = _clamp01(1.0 - abs(edge_pct - 1.35) / 1.15)
    contrast_health = _clamp01((float(white - black) - 25.0) / 80.0)
    candidate_health = _clamp01(1.0 - max(0.0, overlap.candidate_count - 18.0) / 25.0)
    time_health = _clamp01(1.0 - max(0.0, elapsed - 3.0) / 8.0)
    score = (
        35.0 * line_health
        + 25.0 * edge_health
        + 20.0 * contrast_health
        + 10.0 * candidate_health
        + 10.0 * time_health
    )
    message = (
        f"score {score:.1f}｜线 {overlap.line_count}｜候选 {overlap.candidate_count}｜"
        f"边缘 {edge_pct:.2f}%｜方块 {len(overlap.squares)}｜耗时 {elapsed:.1f}s"
    )
    return AutoTuneMetrics(
        ok=True,
        score=score,
        message=message,
        black_gray=float(black),
        white_gray=float(white),
        edge_pct=edge_pct,
        line_count=overlap.line_count,
        candidate_count=overlap.candidate_count,
        square_count=len(overlap.squares),
        elapsed_s=elapsed,
    )
