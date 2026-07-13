from __future__ import annotations

import cv2
import numpy as np


def illumination_gradient(
    image: np.ndarray,
    strength: float,
    angle_deg: float,
) -> np.ndarray:
    """加入缓慢变化的线性光照梯度；strength 是两端最大相对变化量。"""
    height, width = image.shape[:2]
    yy, xx = np.mgrid[-1.0:1.0:complex(height), -1.0:1.0:complex(width)]
    angle = np.deg2rad(angle_deg)
    coordinate = (xx * np.cos(angle) + yy * np.sin(angle)) / np.sqrt(2.0)
    gain = 1.0 + strength * coordinate
    return np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def gaussian_defocus(image: np.ndarray, sigma_px: float) -> np.ndarray:
    """模拟静态场景中的轻微失焦，不产生方向性拖影。"""
    if sigma_px <= 0:
        return image.copy()
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_px, sigmaY=sigma_px)


def sensor_noise(image: np.ndarray, stddev: float, rng: np.random.Generator) -> np.ndarray:
    """可选的极弱像素读出噪声；默认配置关闭。"""
    if stddev <= 0:
        return image.copy()
    noise = rng.normal(0.0, stddev, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
