from __future__ import annotations

"""
基础与高级靶纸 A3 排版拼版 PDF 生成器。

本脚本用于生成 A3 大幅面打印测试的 PDF 靶纸文件，将原来的 A4 靶纸内容按照规定的平移偏差量，
垂直居中拼版到标准的 A3 (297mm x 420mm) 画布中央。
主要功能包括：
1. 预设基础形状与高级形状靶纸，并平移映射其在 A3 上的物理坐标；
2. 绘制标准的 A3 页面底板，并在 A3 中央勾勒出 A4 黑框边界以指导物理打印对齐；
3. 输出每个用例的独立 A3 PDF，并生成一个 18 页合并的大 PDF 靶纸合集。
"""

import math
from dataclasses import dataclass
from pathlib import Path

from reportlab.lib.colors import black, white
from reportlab.lib.pagesizes import A3
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
TARGET_ROOT = ROOT / "assets" / "targets" / "generated"
OUT_DIR = TARGET_ROOT / "a3" / "pdf"
MERGED_OUTPUT = OUT_DIR / "all_targets_A3_18pages.pdf"

# 物理与排版比例常数
MM_TO_PT = 72.0 / 25.4

# 标准尺寸规范（mm）
A3_WIDTH_MM = 297.0
A3_HEIGHT_MM = 420.0
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
BORDER_WIDTH_MM = 20.0
MARKER_DIAMETER_MM = 6.0

# 建立 A3 PDF 页面尺寸
A3_SIZE = (A3_WIDTH_MM * MM_TO_PT, A3_HEIGHT_MM * MM_TO_PT)
# 计算 A4 区域居中于 A3 时的 X 与 Y 物理偏移量 (mm)
A4_OFFSET_X_MM = (A3_WIDTH_MM - A4_WIDTH_MM) / 2.0
A4_OFFSET_Y_MM = (A3_HEIGHT_MM - A4_HEIGHT_MM) / 2.0


@dataclass(frozen=True)
class Square:
    """
    正方形物理布局定义。
    """
    cx_mm: float
    cy_mm_from_top: float
    size_mm: float
    digit: str | None = None
    angle_deg: float = 0.0


@dataclass(frozen=True)
class Target:
    """
    高级多图形靶标定义。
    """
    name: str
    squares: tuple[Square, ...]


# A3 基础题形状配置列表
BASIC_TARGETS = (
    ("basic_circle_on_A3", "circle", 120.0),
    ("basic_triangle_on_A3", "triangle", 120.0),
    ("basic_square_on_A3", "square", 120.0),
)

# A3 发挥题复杂多图形与重叠正方形配置列表
ADVANCED_TARGETS: tuple[Target, ...] = (
    Target("advanced_single_square_60mm_on_A3", (Square(105, 148.5, 60),)),
    Target("advanced_single_square_80mm_on_A3", (Square(105, 148.5, 80),)),
    Target("advanced_single_square_100mm_on_A3", (Square(105, 148.5, 100),)),
    Target("advanced_separate_squares_01_on_A3", (Square(70, 92, 60, angle_deg=-12), Square(142, 118, 68, angle_deg=8), Square(92, 197, 78))),
    Target("advanced_separate_squares_02_on_A3", (Square(65, 85, 54, angle_deg=14), Square(145, 90, 62, angle_deg=-10), Square(66, 198, 70), Square(145, 198, 58, angle_deg=9))),
    Target("advanced_separate_squares_03_on_A3", (Square(55, 75, 46, angle_deg=-16), Square(152, 80, 56, angle_deg=11), Square(105, 145, 64), Square(58, 220, 52, angle_deg=8), Square(150, 215, 68, angle_deg=-7))),
    Target("advanced_separate_squares_04_on_A3", (Square(58, 88, 64), Square(145, 116, 72, angle_deg=12), Square(68, 214, 60, angle_deg=-9), Square(150, 210, 54, angle_deg=16))),
    Target("advanced_overlap_squares_01_on_A3", (Square(80, 143, 78, angle_deg=-8), Square(134, 156, 72, angle_deg=13))),
    Target("advanced_overlap_squares_02_on_A3", (Square(60, 122, 68, angle_deg=10), Square(112, 148, 82), Square(152, 186, 62, angle_deg=-14))),
    Target("advanced_overlap_squares_03_on_A3", (Square(56, 120, 60, angle_deg=-12), Square(98, 145, 66, angle_deg=8), Square(140, 170, 70, angle_deg=-6), Square(152, 218, 56, angle_deg=15))),
    Target("advanced_overlap_squares_04_on_A3", (Square(78, 134, 90), Square(137, 156, 74, angle_deg=12), Square(102, 215, 58, angle_deg=-15))),
    Target("advanced_numbered_squares_01_on_A3", (Square(68, 104, 62, "1", -10), Square(142, 112, 70, "2", 8), Square(105, 197, 78, "3"))),
    Target("advanced_numbered_squares_02_on_A3", (Square(64, 88, 56, "4", 13), Square(146, 92, 64, "5", -9), Square(66, 199, 68, "6"), Square(146, 202, 60, "7", 11))),
    Target("advanced_numbered_squares_03_on_A3", (Square(54, 76, 48, "0", -14), Square(152, 78, 54, "1", 10), Square(105, 145, 64, "3"), Square(58, 220, 52, "5", 9), Square(150, 216, 66, "8", -8))),
    Target("advanced_numbered_squares_04_on_A3", (Square(60, 110, 72, "2"), Square(146, 112, 58, "4", 15), Square(72, 208, 56, "6", -10), Square(144, 198, 78, "9", 7))),
)


def mm_to_pt(value: float) -> float:
    return value * MM_TO_PT


def a4_x(x_mm: float) -> float:
    """
    将 A4 纸面局部 X 坐标物理位置映射为 A3 PDF 页面上的绝对 X 磅数位置。
    """
    return mm_to_pt(A4_OFFSET_X_MM + x_mm)


def a4_y(y_mm_from_bottom: float) -> float:
    """
    将 A4 纸面局部自下而上 Y 坐标映射为 A3 PDF 页面上的绝对 Y 磅数位置。
    """
    return mm_to_pt(A4_OFFSET_Y_MM + y_mm_from_bottom)


def a4_y_from_top(y_mm_from_top: float) -> float:
    """
    将 A4 纸面局部自上而下 Y 坐标映射为 A3 PDF 页面上的绝对 Y 磅数位置。
    """
    return a4_y(A4_HEIGHT_MM - y_mm_from_top)


def rotated_square_points_mm(square: Square) -> list[tuple[float, float]]:
    """
    计算正方形在 A4 上的 4 个旋转后物理角点位置。
    """
    half = square.size_mm / 2.0
    angle = math.radians(square.angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    points = [(-half, -half), (half, -half), (half, half), (-half, half)]
    return [
        (square.cx_mm + x * cos_a - y * sin_a, square.cy_mm_from_top + x * sin_a + y * cos_a)
        for x, y in points
    ]


def polygon_area(points: list[tuple[float, float]]) -> float:
    """
    使用鞋带定理计算多边形面积。
    """
    if len(points) < 3:
        return 0.0
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]))) / 2.0


def is_inside(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> bool:
    """
    判定点是否在多边形内侧。
    """
    px, py = point
    sx, sy = start
    ex, ey = end
    return (ex - sx) * (py - sy) - (ey - sy) * (px - sx) >= -1e-9


def line_intersection(p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float], p4: tuple[float, float]) -> tuple[float, float]:
    """
    两线交点求解。
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return p2
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return px, py


def polygon_clip(subject: list[tuple[float, float]], clip: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    裁剪计算两个多边形的相交重叠多边形。
    """
    output = subject
    for idx, start in enumerate(clip):
        end = clip[(idx + 1) % len(clip)]
        input_list = output
        output = []
        if not input_list:
            break
        prev = input_list[-1]
        for current in input_list:
            current_inside = is_inside(current, start, end)
            prev_inside = is_inside(prev, start, end)
            if current_inside:
                if not prev_inside:
                    output.append(line_intersection(prev, current, start, end))
                output.append(current)
            elif prev_inside:
                output.append(line_intersection(prev, current, start, end))
            prev = current
    return output


def validate_advanced_layouts() -> None:
    """
    对所有 A3 发挥题布局进行安全验证，防止越界或过大的重叠（重合面积 <= 30%）。
    """
    left, right = BORDER_WIDTH_MM, A4_WIDTH_MM - BORDER_WIDTH_MM
    top = BORDER_WIDTH_MM
    bottom = A4_HEIGHT_MM - BORDER_WIDTH_MM - MARKER_DIAMETER_MM - 2.0
    for target in ADVANCED_TARGETS:
        polygons = [rotated_square_points_mm(square) for square in target.squares]
        for square, polygon in zip(target.squares, polygons):
            for x, y in polygon:
                if x < left or x > right or y < top or y > bottom:
                    raise ValueError(f"{target.name}: square out of A4 safe area: {square}")
        for i, poly_a in enumerate(polygons):
            for j in range(i + 1, len(polygons)):
                overlap = polygon_area(polygon_clip(poly_a, polygons[j]))
                smaller = min(target.squares[i].size_mm**2, target.squares[j].size_mm**2)
                if smaller and overlap / smaller > 0.30:
                    raise ValueError(f"{target.name}: pair overlap exceeds 30%")
        for i, poly_a in enumerate(polygons):
            for j in range(i + 1, len(polygons)):
                pair = polygon_clip(poly_a, polygons[j])
                for k in range(j + 1, len(polygons)):
                    if polygon_area(polygon_clip(pair, polygons[k])) > 1e-6:
                        raise ValueError(f"{target.name}: three squares overlap in one area")


def draw_a4_frame(c: canvas.Canvas) -> None:
    """
    在 A3 纸张正中央绘制 A4 物理黑边框界线及下方定位标志黑圆点，用于指引视觉矫正。
    """
    border = mm_to_pt(BORDER_WIDTH_MM)
    a4_w = mm_to_pt(A4_WIDTH_MM)
    a4_h = mm_to_pt(A4_HEIGHT_MM)
    ox = mm_to_pt(A4_OFFSET_X_MM)
    oy = mm_to_pt(A4_OFFSET_Y_MM)
    marker_r = mm_to_pt(MARKER_DIAMETER_MM / 2.0)

    # A3 页面白底填充
    c.setFillColor(white)
    c.rect(0, 0, A3_SIZE[0], A3_SIZE[1], stroke=0, fill=1)

    # 绘制居中的 A4 20mm 黑框线和定位圆点
    c.setFillColor(black)
    c.rect(ox, oy, a4_w, border, stroke=0, fill=1)
    c.rect(ox, oy + a4_h - border, a4_w, border, stroke=0, fill=1)
    c.rect(ox, oy, border, a4_h, stroke=0, fill=1)
    c.rect(ox + a4_w - border, oy, border, a4_h, stroke=0, fill=1)
    c.circle(ox + a4_w / 2.0, oy + border + marker_r, marker_r, stroke=0, fill=1)


def draw_basic(c: canvas.Canvas, shape: str, size_mm: float) -> None:
    """
    绘制基础 A3 靶纸形状（圆、三角形、正方形）。
    """
    cx = a4_x(A4_WIDTH_MM / 2.0)
    cy = a4_y(A4_HEIGHT_MM / 2.0)
    size = mm_to_pt(size_mm)
    c.setFillColor(black)
    if shape == "circle":
        c.circle(cx, cy, size / 2.0, stroke=0, fill=1)
    elif shape == "triangle":
        height = size * math.sqrt(3.0) / 2.0
        path = c.beginPath()
        path.moveTo(cx, cy + 2.0 * height / 3.0)
        path.lineTo(cx - size / 2.0, cy - height / 3.0)
        path.lineTo(cx + size / 2.0, cy - height / 3.0)
        path.close()
        c.drawPath(path, stroke=0, fill=1)
    elif shape == "square":
        c.rect(cx - size / 2.0, cy - size / 2.0, size, size, stroke=0, fill=1)
    else:
        raise ValueError(f"unknown basic shape: {shape}")


def draw_square(c: canvas.Canvas, square: Square) -> None:
    """
    在 A3 居中的 A4 画布区域绘制倾斜/带编号的高级正方形目标。
    """
    size = mm_to_pt(square.size_mm)
    c.saveState()
    # 先平移到正方形对应在 A3 画布上的物理坐标系中心
    c.translate(a4_x(square.cx_mm), a4_y_from_top(square.cy_mm_from_top))
    c.rotate(-square.angle_deg)
    c.setFillColor(black)
    c.rect(-size / 2.0, -size / 2.0, size, size, stroke=0, fill=1)

    # 绘制带编号的白色字体并对其作旋转平移
    if square.digit is not None:
        font_size = mm_to_pt(square.size_mm * 0.52)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", font_size)
        width = stringWidth(square.digit, "Helvetica-Bold", font_size)
        c.drawString(-width / 2.0, -font_size * 0.36, square.digit)
    c.restoreState()


def draw_advanced(c: canvas.Canvas, target: Target) -> None:
    """
    绘制高级靶标正方形组件集。
    """
    for square in target.squares:
        draw_square(c, square)


def draw_page(c: canvas.Canvas, item: tuple[str, str, float] | Target) -> None:
    """
    根据靶标类别绘制完整的 A3 画布页面（A4 边框 + 核心图形）。
    """
    draw_a4_frame(c)
    if isinstance(item, Target):
        draw_advanced(c, item)
    else:
        _, shape, size_mm = item
        draw_basic(c, shape, size_mm)


def write_single_pdf(item: tuple[str, str, float] | Target) -> None:
    """
    输出单页 A3 靶标 PDF 文件。
    """
    name = item.name if isinstance(item, Target) else item[0]
    path = OUT_DIR / f"{name}.pdf"
    c = canvas.Canvas(str(path), pagesize=A3_SIZE)
    draw_page(c, item)
    c.showPage()
    c.save()


def write_merged_pdf(items: list[tuple[str, str, float] | Target]) -> None:
    """
    将所有 18 页 A3 靶标合并生成多页 PDF 大文件，方便一次性批量打印。
    """
    c = canvas.Canvas(str(MERGED_OUTPUT), pagesize=A3_SIZE)
    for item in items:
        draw_page(c, item)
        c.showPage()
    c.save()


def main() -> None:
    # 1. 安全布局校验
    validate_advanced_layouts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    items: list[tuple[str, str, float] | Target] = [*BASIC_TARGETS, *ADVANCED_TARGETS]

    # 2. 循环输出单页 PDF 靶标
    for item in items:
        write_single_pdf(item)

    # 3. 批量合并输出大合集文件
    write_merged_pdf(items)

    print(f"Generated {len(items)} individual A3 PDFs in {OUT_DIR}")
    print(f"Generated merged A3 PDF: {MERGED_OUTPUT}")
    print(f"A4 content offset on A3: x={A4_OFFSET_X_MM} mm, y={A4_OFFSET_Y_MM} mm")


if __name__ == "__main__":
    main()
