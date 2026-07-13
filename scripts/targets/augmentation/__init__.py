"""对已透视校正的标准目标图添加可复现的轻微成像扰动。"""

from .pipeline import AugmentationConfig, augment

__all__ = ["AugmentationConfig", "augment"]
