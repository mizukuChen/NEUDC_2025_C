from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from src.vision.lowres_preprocess import WORK_HEIGHT_PX, WORK_WIDTH_PX


@dataclass(frozen=True)
class LowResolutionConfig:
    """模拟A4在原始相机画面中的真实像素密度和静态成像退化。"""

    native_width_px: int
    phase_x_px: float = 0.0
    phase_y_px: float = 0.0
    defocus_sigma_native_px: float = 0.35
    black_level: float = 18.0
    white_level: float = 178.0
    illumination_strength: float = 0.04
    illumination_angle_deg: float = 0.0
    noise_stddev: float = 0.8

    def __post_init__(self) -> None:
        if self.native_width_px < 40:
            raise ValueError("native A4 width must be at least 40 pixels")
        if not -0.5 <= self.phase_x_px <= 0.5 or not -0.5 <= self.phase_y_px <= 0.5:
            raise ValueError("sampling phase must be within [-0.5, 0.5] pixels")
        if self.defocus_sigma_native_px < 0 or self.noise_stddev < 0:
            raise ValueError("blur and noise must be non-negative")
        if not 0 <= self.black_level < self.white_level <= 255:
            raise ValueError("black/white levels must satisfy 0 <= black < white <= 255")
        if not 0 <= self.illumination_strength <= 0.5:
            raise ValueError("illumination strength must be within [0, 0.5]")


def _sample_sensor_grid(
    image: np.ndarray,
    width: int,
    height: int,
    phase_x: float,
    phase_y: float,
) -> np.ndarray:
    source_height, source_width = image.shape
    xs = ((np.arange(width, dtype=np.float32) + 0.5 + phase_x) * source_width / width) - 0.5
    ys = ((np.arange(height, dtype=np.float32) + 0.5 + phase_y) * source_height / height) - 0.5
    map_x, map_y = np.meshgrid(xs, ys)
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def simulate_low_resolution_a4(
    image: np.ndarray,
    config: LowResolutionConfig,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """先形成低分辨率传感器图，再重采样到420×594算法工作网格。"""
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("low-resolution simulation input must be an 8-bit grayscale A4 image")
    native_height = round(config.native_width_px * 297.0 / 210.0)
    source_scale = image.shape[1] / config.native_width_px
    sigma_source = config.defocus_sigma_native_px * source_scale
    optical = (
        cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_source, sigmaY=sigma_source)
        if sigma_source > 0
        else image
    )
    sensor = _sample_sensor_grid(
        optical,
        config.native_width_px,
        native_height,
        config.phase_x_px,
        config.phase_y_px,
    ).astype(np.float32)

    normalized = sensor / 255.0
    sensor = config.black_level + normalized * (config.white_level - config.black_level)
    yy, xx = np.mgrid[-1.0:1.0:complex(native_height), -1.0:1.0:complex(config.native_width_px)]
    angle = np.deg2rad(config.illumination_angle_deg)
    coordinate = (xx * np.cos(angle) + yy * np.sin(angle)) / np.sqrt(2.0)
    sensor *= 1.0 + config.illumination_strength * coordinate
    if config.noise_stddev > 0:
        rng = np.random.default_rng(seed)
        sensor += rng.normal(0.0, config.noise_stddev, sensor.shape)
    sensor_u8 = np.clip(sensor, 0, 255).astype(np.uint8)

    working = cv2.resize(
        sensor_u8,
        (WORK_WIDTH_PX, WORK_HEIGHT_PX),
        interpolation=cv2.INTER_LINEAR,
    )
    metadata = {
        "seed": int(seed),
        "native_height_px": native_height,
        "working_size_px": [WORK_WIDTH_PX, WORK_HEIGHT_PX],
        "config": asdict(config),
    }
    return working, metadata
