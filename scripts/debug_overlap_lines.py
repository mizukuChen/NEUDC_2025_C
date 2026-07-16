from __future__ import annotations

"""
重叠正方形测量算法的多阶段可视化调试工具。

本脚本载入指定的重叠靶纸图像，执行重叠图形重建算法的完整工作流。
提取各核心处理阶段的图像输出，包括：
1. 原始透视矫正 A4 输入图像
2. 大津法（Otsu）提取的黑色实心前景图
3. Canny 算子边缘检测图
4. 霍夫概率线变换（HoughLinesP）提取并合并后的长直线图
5. 平行线垂直交叉组合后的所有正方形候选框（含虚警）
6. 非极大值抑制（NMS）去重过滤后的最终正方形重建结果

最后，将 6 个阶段的图像按 2x3 的网格拼接，保存为一张超大调试版面板图（Debug Board）。
"""

import argparse
import sys
from math import cos, sin
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.vision.overlap.standard as overlap
from scripts.targets.generate_overlap_squares import PRESETS

# 调试标注使用的调色盘（BGR 格式），用于区分绘制不同的直线和正方形候选
COLORS = (
    (0, 0, 255),      # 红
    (0, 180, 0),      # 绿
    (255, 80, 0),     # 蓝
    (180, 0, 180),    # 紫
    (0, 170, 255),    # 橙黄
    (255, 0, 150),    # 粉红
)


def _label(image: np.ndarray, text: str) -> np.ndarray:
    """
    在图像顶部绘制一块灰白色矩形横条，并添加阶段文字说明标签。
    """
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 76), (245, 245, 245), -1)
    cv2.putText(output, text, (24, 51), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (20, 20, 20), 3, cv2.LINE_AA)
    return output


def _fit_panel(image: np.ndarray, width: int = 720, height: int = 900) -> np.ndarray:
    """
    将输入图像等比例缩放，并以居中对齐、灰色背景填充的方式嵌入到标准的 720x900 画布板中，
    便于多图像拼图时边界完全对齐。
    """
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    panel = np.full((height, width, 3), 238, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel


def _draw_infinite_line(image: np.ndarray, line: overlap._Line, color: tuple[int, int, int]) -> None:
    """
    根据极坐标法线式直线方程（rho, theta），在图像中绘制一条延伸至无穷远的直线。
    """
    normal = np.array([cos(line.theta), sin(line.theta)])
    direction = np.array([-normal[1], normal[0]])
    center = normal * line.rho
    extent = 2.0 * max(image.shape[:2])
    start = np.rint(center - direction * extent).astype(int)
    end = np.rint(center + direction * extent).astype(int)
    cv2.line(image, tuple(start), tuple(end), color, 2, cv2.LINE_AA)


def _detect_lines(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[overlap._Line], tuple[int, int]]:
    """
    运行前期的线条边缘检测与 Hough 直线提取融合逻辑，为拼版图准备数据。
    """
    height, width = image.shape[:2]
    scale_x = width / overlap.A4_WIDTH_MM
    scale_y = height / overlap.A4_HEIGHT_MM
    scale = (scale_x + scale_y) / 2.0
    x0 = round(overlap.INSET_MM * scale_x)
    y0 = round(overlap.INSET_MM * scale_y)
    # 裁剪内圈，避开 A4 黑边框
    roi = image[y0 : height - y0, x0 : width - x0]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120, apertureSize=3)

    raw: list[overlap._Line] = []
    probabilistic = cv2.HoughLinesP(
        edges,
        1.0,
        np.pi / 720.0,
        threshold=max(20, round(4.0 * scale)),
        minLineLength=max(20, round(8.0 * scale)),
        maxLineGap=max(4, round(0.8 * scale)),
    )
    if probabilistic is not None:
        for segment in np.asarray(probabilistic).reshape(-1, 4):
            line = overlap._canonical_line(*(float(value) for value in segment))
            if line is not None:
                raw.append(line)
    # 合并极坐标位置和角度非常接近的直线
    lines = overlap._merge_lines(raw, np.deg2rad(0.5), 0.7 * scale)
    lines = sorted(lines, key=lambda item: item.support, reverse=True)[:48]
    _, foreground = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return roi, edges, lines, (x0, y0)


def create_debug_board(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    运行重叠正方形重建逻辑，截获中间生成的候选框数据，拼装 2x3 的多阶段面板大图。

    返回:
        board: 拼装好的大调试图。
        candidate_count: 原始四边形候选数。
        selected_count: 最终通过去重和实心校验筛选出的正方形数。
    """
    roi, edges, lines, offset = _detect_lines(image)
    captured: list[overlap.ReconstructedSquare] = []
    original_deduplicate = overlap._deduplicate_squares

    # 通过猴子补丁（Monkey Patch）截获中间阶段生成的正方形候选框
    def capture(candidates):
        captured.extend(candidates)
        return original_deduplicate(candidates)

    overlap._deduplicate_squares = capture
    try:
        result = overlap.reconstruct_overlapping_squares(image)
    finally:
        overlap._deduplicate_squares = original_deduplicate

    mask = cv2.threshold(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    mask_view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    edge_view = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 绘制霍夫变换合并后的直线
    line_view = roi.copy()
    for index, line in enumerate(lines):
        _draw_infinite_line(line_view, line, COLORS[index % len(COLORS)])

    # 绘制筛选前所有的正方形候选框（虚警）
    candidate_view = image.copy()
    for index, candidate in enumerate(captured):
        points = np.rint(candidate.corners).astype(np.int32).reshape(-1, 1, 2)
        color = COLORS[index % len(COLORS)]
        cv2.polylines(candidate_view, [points], True, color, 3, cv2.LINE_AA)
        center = tuple(np.rint(candidate.corners.mean(axis=0)).astype(int))
        cv2.putText(candidate_view, f"{candidate.side_mm:.1f}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

    # 绘制去重并进行支持率筛选后的最终结果
    final_view = overlap.draw_overlap_measurements(image, result)

    # 拼装面板网格
    panels = [
        _label(_fit_panel(image), "1  Rectified A4 input"),
        _label(_fit_panel(mask_view), "2  Otsu black foreground"),
        _label(_fit_panel(edge_view), "3  Canny visible edges"),
        _label(_fit_panel(line_view), f"4  Filtered HoughLinesP lines ({len(lines)})"),
        _label(_fit_panel(candidate_view), f"5  Four-edge candidates ({len(captured)})"),
        _label(_fit_panel(final_view), f"6  Deduplicated final ({len(result.squares)})"),
    ]
    board = np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:])))
    return board, len(captured), len(result.squares)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize overlap-square detection stages.")
    parser.add_argument("case", nargs="?", choices=tuple(PRESETS), help="指定运行的重叠图形预设用例 ID。若不指定，则遍历运行所有用例。")
    parser.add_argument("--dataset", choices=("controlled", "legacy"), default="controlled", help="选择数据集：controlled（新受控数据集）或 legacy（老版数据集）。")
    args = parser.parse_args()

    cases = [args.case] if args.case else list(PRESETS)
    output = ROOT / "runtime" / "debug" / "overlap_process"
    output.mkdir(parents=True, exist_ok=True)

    for case in cases:
        if args.dataset == "controlled":
            path = ROOT / "assets" / "targets" / "generated" / "overlap_controlled" / f"overlap_controlled_{case}_A4.png"
        else:
            path = ROOT / "assets" / "targets" / "generated" / "advanced" / "png" / f"advanced_overlap_squares_{case}_A4.png"

        image = cv2.imread(str(path))
        if image is None:
            print(f"FAIL {path}: cannot read image")
            return 1

        board, candidate_count, selected_count = create_debug_board(image)
        destination = output / f"overlap_{args.dataset}_{case}_process.jpg"
        cv2.imwrite(str(destination), board, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"case={case} candidates={candidate_count} selected={selected_count} output={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
