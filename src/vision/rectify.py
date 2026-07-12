from __future__ import annotations

import cv2
import numpy as np

# A4 纸的标准物理尺寸（毫米）
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0

# 设定每毫米对应的像素数：10 px/mm，即 1 像素代表 0.1 毫米
PX_PER_MM = 10

# 变换后的标准 A4 纸图像像素大小：2100 x 2970 像素
RECTIFIED_WIDTH_PX = int(A4_WIDTH_MM * PX_PER_MM)
RECTIFIED_HEIGHT_PX = int(A4_HEIGHT_MM * PX_PER_MM)


def rectify_a4(
    image: np.ndarray,
    corners: np.ndarray,
    width_px: int = RECTIFIED_WIDTH_PX,
    height_px: int = RECTIFIED_HEIGHT_PX,
) -> np.ndarray:
    """
    对检测到的 A4 纸面进行透视变换（Perspective Warp）拉平矫正，输出正视视角的标准图像。

    参数:
        image: 输入的原始图像（可以是全图或去畸变后的图）。
        corners: 4 个定位角点，按 [左上, 右上, 右下, 左下] 顺序排列。
        width_px: 矫正后目标图像的像素宽度，默认 2100。
        height_px: 矫正后目标图像的像素高度，默认 2970。

    返回:
        变换拉直后的 A4 纸面图像（分辨率 2100 x 2970），该图上的几何比例尺为 10 px = 1 mm。
    """
    # 将输入的 4 个原图角点作为源控制点 (src)
    src = np.asarray(corners, dtype=np.float32).reshape(4, 2)

    # 定义拉直后标准 A4 矩形对应的 4 个目标像素位置 (dst)
    dst = np.array(
        [
            [0, 0],                            # 左上角TL -> 像素原点 (0, 0)
            [width_px - 1, 0],                  # 右上角TR -> 像素点 (W-1, 0)
            [width_px - 1, height_px - 1],      # 右下角BR -> 像素点 (W-1, H-1)
            [0, height_px - 1],                 # 左下角BL -> 像素点 (0, H-1)
        ],
        dtype=np.float32,
    )

    # 1. 计算 3x3 的单应性透视变换矩阵
    matrix = cv2.getPerspectiveTransform(src, dst)

    # 2. 执行透视变换投影，将纸面拉平并剪裁输出
    return cv2.warpPerspective(image, matrix, (width_px, height_px))
