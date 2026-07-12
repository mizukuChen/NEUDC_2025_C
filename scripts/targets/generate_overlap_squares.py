from __future__ import annotations

"""
受控实心重叠多正方形靶纸图像（PNG/JSON 元数据）渲染生成工具。

本脚本为发挥题测试算法生成标准的多正方形重叠 A4 靶图。
主要功能包括：
1. 预设 9 组倾斜/位置不同的正方形重叠参数（PRESETS）；
2. 验证各个正方形是否越界超出 A4 有效内容区域；
3. 计算每个正方形的每条边被其他正方形覆盖的比例（使用 `edge_coverage_ratios` 线采样法），
   保证单边覆盖率严格小于 50%（MAX_EDGE_COVERAGE），确保图形测量的物理可见性；
4. 渲染 300 DPI 分辨率下带有 20mm 黑边框、下方定位标志圆点和实心黑色多正方形的 PNG 图片，
   并导出同名的 JSON ground truth 配置文件。
"""

import argparse
import json
from dataclasses import asdict, dataclass
from math import cos, radians, sin
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
# 默认生成的受控重叠图形输出目录
DEFAULT_OUTPUT = ROOT / "assets" / "targets" / "generated" / "overlap_controlled"

# 物理排版常数定义
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
BORDER_WIDTH_MM = 20.0
MARKER_DIAMETER_MM = 6.0
DPI = 300
PX_PER_MM = DPI / 25.4
# 单条边被遮挡的最大允许物理比例（50%）
MAX_EDGE_COVERAGE = 0.50


@dataclass(frozen=True)
class SquareSpec:
    """
    单个正方形的几何配置规范数据结构。

    属性:
        cx_mm: 几何中心物理 X 坐标 (mm)。
        cy_mm: 几何中心物理 Y 坐标 (mm，以左上角为原点)。
        size_mm: 正方形物理边长 (mm)。
        angle_deg: 正方形旋转偏角 (度)。
    """
    cx_mm: float
    cy_mm: float
    size_mm: float
    angle_deg: float


# 9 组精心设计的受控重叠正方形排版用例
PRESETS: dict[str, tuple[SquareSpec, ...]] = {
    "01": (SquareSpec(68, 140, 72, -8), SquareSpec(145, 155, 68, 13)),
    "02": (
        SquareSpec(54, 100, 58, 10),
        SquareSpec(110, 150, 68, 0),
        SquareSpec(156, 210, 56, -14),
    ),
    "03": (
        SquareSpec(53, 80, 54, -12),
        SquareSpec(105, 125, 58, 8),
        SquareSpec(155, 175, 60, -6),
        SquareSpec(115, 230, 52, 15),
    ),
    "04": (
        SquareSpec(70, 125, 80, 0),
        SquareSpec(140, 155, 66, 12),
        SquareSpec(105, 215, 54, -15),
    ),
    "05": (SquareSpec(72, 137, 76, 18), SquareSpec(144, 165, 72, -14)),
    "06": (
        SquareSpec(58, 104, 60, -16),
        SquareSpec(112, 141, 76, 7),
        SquareSpec(153, 194, 58, 15),
    ),
    "07": (
        SquareSpec(60, 198, 68, 11),
        SquareSpec(113, 148, 64, -10),
        SquareSpec(153, 95, 56, 18),
    ),
    "08": (
        SquareSpec(52, 74, 52, 0),
        SquareSpec(99, 113, 60, 15),
        SquareSpec(148, 153, 68, -14),
        SquareSpec(114, 213, 56, 7),
    ),
    "09": (
        SquareSpec(70, 85, 80, 0),
        SquareSpec(140, 115, 52, 18),
        SquareSpec(105, 180, 76, -12),
    ),
}


def square_points_mm(square: SquareSpec) -> np.ndarray:
    """
    根据正方形的中心点、尺寸和倾斜偏角，计算其在 A4 物理空间下的 4 个旋转后角点坐标 (mm)。
    """
    half = square.size_mm / 2.0
    angle = radians(square.angle_deg)
    rotation = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
    local = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    return local @ rotation.T + np.array([square.cx_mm, square.cy_mm])


def edge_coverage_ratios(squares: tuple[SquareSpec, ...], samples: int = 2001) -> list[list[float]]:
    """
    计算用例中每个正方形 4 条边被其他正方形覆盖遮挡的可见性比例。

    原理:
        - 在正方形的每条边上均匀进行样本点采样（默认 2000 个采样点，避开两端顶点）。
        - 使用 OpenCV 的 `pointPolygonTest` 测试每个点是否落入其他正方形内部。
        - 统计被遮挡的采样点比例，作为该条边的遮挡率。
    """
    polygons = [square_points_mm(square).astype(np.float32) for square in squares]
    ratios: list[list[float]] = []
    for square_index, polygon in enumerate(polygons):
        square_ratios = []
        for edge_index in range(4):
            # 避开精确端点，因为顶点接触并不算作边线遮挡长度
            weights = np.linspace(0.0, 1.0, samples)[1:-1, None]
            points = polygon[edge_index] * (1.0 - weights) + polygon[(edge_index + 1) % 4] * weights
            covered = np.zeros(len(points), dtype=bool)
            for other_index, other in enumerate(polygons):
                if other_index == square_index:
                    continue
                # 检测采样点是否在其他多边形内（> 0 代表在多边形内部，0 在边缘，< 0 在外部）
                covered |= np.fromiter(
                    (cv2.pointPolygonTest(other, tuple(map(float, point)), False) > 0 for point in points),
                    dtype=bool,
                    count=len(points),
                )
            square_ratios.append(float(np.mean(covered)))
        ratios.append(square_ratios)
    return ratios


def validate(squares: tuple[SquareSpec, ...]) -> list[list[float]]:
    """
    对用例进行合法性约束校验。

    要求:
        1. 正方形不能超出 A4 20mm 黑框的内部白区域。
        2. 任意正方形的任意一条边被遮挡率必须严格小于 50%，以保证霍夫线变换有足够的未遮挡线段重建角点。
    """
    if len(squares) < 2:
        raise ValueError("At least two squares are required.")
    for index, square in enumerate(squares):
        polygon = square_points_mm(square)
        if np.any(polygon[:, 0] < BORDER_WIDTH_MM) or np.any(polygon[:, 0] > A4_WIDTH_MM - BORDER_WIDTH_MM):
            raise ValueError(f"square[{index}] exceeds the horizontal content area")
        if np.any(polygon[:, 1] < BORDER_WIDTH_MM) or np.any(polygon[:, 1] > A4_HEIGHT_MM - BORDER_WIDTH_MM):
            raise ValueError(f"square[{index}] exceeds the vertical content area")

    ratios = edge_coverage_ratios(squares)
    for square_index, values in enumerate(ratios):
        for edge_index, ratio in enumerate(values):
            if ratio >= MAX_EDGE_COVERAGE:
                raise ValueError(
                    f"square[{square_index}] edge[{edge_index}] coverage={ratio:.3f} must be < {MAX_EDGE_COVERAGE:.2f}"
                )
    return ratios


def render(squares: tuple[SquareSpec, ...]) -> np.ndarray:
    """
    在 300 DPI 分辨率下将正方形和 A4 黑边框渲染为灰度（8位）图像。
    """
    width = round(A4_WIDTH_MM * PX_PER_MM)
    height = round(A4_HEIGHT_MM * PX_PER_MM)
    # 背景默认为白色 (255)
    image = np.full((height, width), 255, dtype=np.uint8)

    # 1. 绘制四周 20mm 黑边框
    border = round(BORDER_WIDTH_MM * PX_PER_MM)
    image[:border] = 0
    image[-border:] = 0
    image[:, :border] = 0
    image[:, -border:] = 0

    # 2. 绘制下底边中点处的定位圆点
    marker_radius = round(MARKER_DIAMETER_MM * PX_PER_MM / 2.0)
    cv2.circle(image, (width // 2, height - border - marker_radius), marker_radius, 0, -1, cv2.LINE_AA)

    # 3. 按顺序将正方形填充为纯黑 (0)
    for square in squares:
        points = np.rint(square_points_mm(square) * PX_PER_MM).astype(np.int32)
        cv2.fillConvexPoly(image, points, 0, cv2.LINE_AA)
    return image


def generate_case(case: str, squares: tuple[SquareSpec, ...], output: Path) -> None:
    """
    验证、渲染并保存单个测试用例的 PNG 和标注 JSON 文件。
    """
    ratios = validate(squares)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"overlap_controlled_{case}_A4"
    cv2.imwrite(str(output / f"{stem}.png"), render(squares))

    # 导出 JSON 格式的 Ground Truth 配置文件
    metadata = {
        "page_mm": [A4_WIDTH_MM, A4_HEIGHT_MM],
        "dpi": DPI,
        "max_edge_coverage": MAX_EDGE_COVERAGE,
        "squares": [
            {**asdict(square), "edge_coverage": ratios[index]}
            for index, square in enumerate(squares)
        ],
    }
    (output / f"{stem}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    maximum = max(max(values) for values in ratios)
    print(f"generated {stem}.png squares={len(squares)} max_edge_coverage={maximum:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate controlled overlapping solid-square PNG targets.")
    parser.add_argument("--case", choices=(*PRESETS.keys(), "all"), default="all", help="指定生成某一个用例 ID。默认 'all' 生成所有用例。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="文件保存的目标目录。")
    args = parser.parse_args()

    cases = PRESETS.items() if args.case == "all" else ((args.case, PRESETS[args.case]),)
    for case, squares in cases:
        generate_case(case, squares, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
