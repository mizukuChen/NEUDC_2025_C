from __future__ import annotations

import cv2
import numpy as np

PATCH_SIZE = 64
CELL_SIZE = 8
BLOCK_CELLS = 2
BINS = 9
HOG_FEATURE_SIZE = (
    (PATCH_SIZE // CELL_SIZE - BLOCK_CELLS + 1) ** 2
    * BLOCK_CELLS
    * BLOCK_CELLS
    * BINS
)


def extract_hog(image: np.ndarray) -> np.ndarray:
    """手工实现的 64×64、1764 维 unsigned HOG 特征。"""
    if image.shape != (PATCH_SIZE, PATCH_SIZE):
        raise ValueError(f"HOG input must be {PATCH_SIZE}x{PATCH_SIZE}")
    gray = image.astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle %= 180.0
    bin_position = angle / (180.0 / BINS)
    lower = np.floor(bin_position).astype(np.int32) % BINS
    upper = (lower + 1) % BINS
    upper_weight = bin_position - np.floor(bin_position)

    cells = PATCH_SIZE // CELL_SIZE
    hist = np.zeros((cells, cells, BINS), dtype=np.float32)
    for cy in range(cells):
        for cx in range(cells):
            ys = slice(cy * CELL_SIZE, (cy + 1) * CELL_SIZE)
            xs = slice(cx * CELL_SIZE, (cx + 1) * CELL_SIZE)
            mag = magnitude[ys, xs].ravel()
            lo = lower[ys, xs].ravel()
            hi = upper[ys, xs].ravel()
            weight = upper_weight[ys, xs].ravel()
            np.add.at(hist[cy, cx], lo, mag * (1.0 - weight))
            np.add.at(hist[cy, cx], hi, mag * weight)

    features = []
    epsilon = 1e-6
    for cy in range(cells - BLOCK_CELLS + 1):
        for cx in range(cells - BLOCK_CELLS + 1):
            block = hist[cy : cy + BLOCK_CELLS, cx : cx + BLOCK_CELLS].ravel()
            block = block / np.sqrt(float(np.dot(block, block)) + epsilon * epsilon)
            block = np.minimum(block, 0.2)
            block = block / np.sqrt(float(np.dot(block, block)) + epsilon * epsilon)
            features.append(block)
    return np.concatenate(features).astype(np.float32)
