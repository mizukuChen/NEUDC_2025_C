from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .geometry import residual_perspective
from .imaging import gaussian_defocus, illumination_gradient, sensor_noise


@dataclass(frozen=True)
class AugmentationConfig:
    """每个参数的范围均在上下界内均匀采样。"""

    perspective_px: tuple[float, float] = (0.3, 2.5)
    blur_sigma_px: tuple[float, float] = (0.25, 1.15)
    illumination_strength: tuple[float, float] = (0.01, 0.10)
    noise_stddev: tuple[float, float] = (0.0, 0.0)


def _sample_range(rng: np.random.Generator, limits: tuple[float, float]) -> float:
    low, high = limits
    if low < 0 or high < low:
        raise ValueError(f"invalid augmentation range: {limits}")
    return float(rng.uniform(low, high))


def augment(
    image: np.ndarray,
    seed: int,
    config: AugmentationConfig = AugmentationConfig(),
) -> tuple[np.ndarray, dict]:
    """返回扰动图和完整参数记录；相同输入、seed 与配置可完全复现。"""
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("augmentation input must be an 8-bit grayscale image")

    rng = np.random.default_rng(seed)
    perspective = _sample_range(rng, config.perspective_px)
    corner_offsets = rng.uniform(-perspective, perspective, size=(4, 2)).astype(np.float32)
    blur_sigma = _sample_range(rng, config.blur_sigma_px)
    light_strength = _sample_range(rng, config.illumination_strength)
    light_angle = float(rng.uniform(0.0, 360.0))
    noise_stddev = _sample_range(rng, config.noise_stddev)

    result, matrix = residual_perspective(image, corner_offsets)
    result = illumination_gradient(result, light_strength, light_angle)
    result = gaussian_defocus(result, blur_sigma)
    result = sensor_noise(result, noise_stddev, rng)

    metadata = {
        "seed": int(seed),
        "config": asdict(config),
        "sampled": {
            "corner_offsets_px": corner_offsets.tolist(),
            "perspective_matrix": matrix.tolist(),
            "blur_sigma_px": blur_sigma,
            "illumination_strength": light_strength,
            "illumination_angle_deg": light_angle,
            "noise_stddev": noise_stddev,
        },
    }
    return result, metadata
