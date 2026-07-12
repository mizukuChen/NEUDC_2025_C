from __future__ import annotations

"""
高级 A4 靶标（单正方形、分离多正方形、旋转正方形及带数字编号正方形）生成器。

本脚本主要用于发挥题的靶纸渲染生成，支持：
1. 单正方形和分离多正方形；
2. 正方形可设定任意倾斜偏角；
3. 支持正方形内部绘制白色识别数字，并保证数字在旋转倾斜下的水平垂直居中对齐；
4. 运用 Sutherland-Hodgman 多边形裁剪算法进行重叠面积检测，自动校验多正方形的布局合法性
   （确保在 A4 白区域以内，且任意两正方形重叠率不大于 30%）。
"""

import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.colors import black, white
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
TARGET_ROOT = ROOT / "assets" / "targets" / "generated"
PDF_DIR = TARGET_ROOT / "advanced" / "pdf"
PNG_DIR = TARGET_ROOT / "advanced" / "png"

# 单位换算比例系数
MM_TO_PT = 72.0 / 25.4
DPI = 300
MM_TO_PX = DPI / 25.4

# 物理布局常量
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
BORDER_WIDTH_MM = 20.0
MARKER_DIAMETER_MM = 6.0


@dataclass(frozen=True)
class Square:
    """
    单个正方形的几何与标注属性。

    属性:
        cx_mm: 中心 X 物理坐标 (mm)。
        cy_mm_from_top: 从 A4 纸顶边算起的中心 Y 物理坐标 (mm)。
        size_mm: 正方形的物理边长 (mm)。
        digit: 填充的白色数字字符（如 "1", "2" 等，为 None 代表不带编号）。
        angle_deg: 顺时针旋转偏角 (度)。
    """
    cx_mm: float
    cy_mm_from_top: float
    size_mm: float
    digit: str | None = None
    angle_deg: float = 0.0


@dataclass(frozen=True)
class Target:
    """
    一个完整的 A4 发挥题靶纸预设。
    """
    name: str
    squares: tuple[Square, ...]


# 预设的 A4 发挥题测试靶纸排版配置
TARGETS: tuple[Target, ...] = (
    Target("advanced_single_square_60mm_A4", (Square(105, 148.5, 60),)),
    Target("advanced_single_square_80mm_A4", (Square(105, 148.5, 80),)),
    Target("advanced_single_square_100mm_A4", (Square(105, 148.5, 100),)),
    Target(
        "advanced_separate_squares_01_A4",
        (
            Square(70, 92, 60, angle_deg=-12),
            Square(142, 118, 68, angle_deg=8),
            Square(92, 197, 78, angle_deg=0),
        ),
    ),
    Target(
        "advanced_separate_squares_02_A4",
        (
            Square(65, 85, 54, angle_deg=14),
            Square(145, 90, 62, angle_deg=-10),
            Square(66, 198, 70, angle_deg=0),
            Square(145, 198, 58, angle_deg=9),
        ),
    ),
    Target(
        "advanced_separate_squares_03_A4",
        (
            Square(55, 75, 46, angle_deg=-16),
            Square(152, 80, 56, angle_deg=11),
            Square(105, 145, 64, angle_deg=0),
            Square(58, 220, 52, angle_deg=8),
            Square(150, 215, 68, angle_deg=-7),
        ),
    ),
    Target(
        "advanced_separate_squares_04_A4",
        (
            Square(58, 88, 64, angle_deg=0),
            Square(145, 116, 72, angle_deg=12),
            Square(68, 214, 60, angle_deg=-9),
            Square(150, 210, 54, angle_deg=16),
        ),
    ),
    Target(
        "advanced_numbered_squares_01_A4",
        (
            Square(68, 104, 62, "1", -10),
            Square(142, 112, 70, "2", 8),
            Square(105, 197, 78, "3", 0),
        ),
    ),
    Target(
        "advanced_numbered_squares_02_A4",
        (
            Square(64, 88, 56, "4", 13),
            Square(146, 92, 64, "5", -9),
            Square(66, 199, 68, "6", 0),
            Square(146, 202, 60, "7", 11),
        ),
    ),
    Target(
        "advanced_numbered_squares_03_A4",
        (
            Square(54, 76, 48, "0", -14),
            Square(152, 78, 54, "1", 10),
            Square(105, 145, 64, "3", 0),
            Square(58, 220, 52, "5", 9),
            Square(150, 216, 66, "8", -8),
        ),
    ),
    Target(
        "advanced_numbered_squares_04_A4",
        (
            Square(60, 110, 72, "2", 0),
            Square(146, 112, 58, "4", 15),
            Square(72, 208, 56, "6", -10),
            Square(144, 198, 78, "9", 7),
        ),
    ),
)


def mm_to_pt(value: float) -> float:
    return value * MM_TO_PT


def mm_to_px(value: float) -> int:
    return round(value * MM_TO_PX)


def top_mm_to_pdf_y(top_mm: float) -> float:
    """
    转换 Y 轴坐标系：从顶边起算 (mm) 转为 PDF 的左下角起算 (points)。
    """
    return mm_to_pt(A4_HEIGHT_MM - top_mm)


def rotated_square_points_mm(square: Square) -> list[tuple[float, float]]:
    """
    计算旋转倾斜后正方形的 4 个物理顶点坐标 (mm)。
    """
    half = square.size_mm / 2.0
    angle = math.radians(square.angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    points = [(-half, -half), (half, -half), (half, half), (-half, half)]
    return [
        (
            square.cx_mm + x * cos_a - y * sin_a,
            square.cy_mm_from_top + x * sin_a + y * cos_a,
        )
        for x, y in points
    ]


def polygon_area(points: list[tuple[float, float]]) -> float:
    """
    使用鞋带公式（Shoelace Formula）计算任意多边形面积。
    """
    if len(points) < 3:
        return 0.0
    total = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def is_inside(point: tuple[float, float], edge_start: tuple[float, float], edge_end: tuple[float, float]) -> bool:
    """
    判断点是否落在裁剪边界边的内侧（利用向量叉乘判别方向）。
    """
    px, py = point
    sx, sy = edge_start
    ex, ey = edge_end
    return (ex - sx) * (py - sy) - (ey - sy) * (px - sx) >= -1e-9


def line_intersection(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> tuple[float, float]:
    """
    求解两条直线 p1->p2 与 p3->p4 的交点坐标。
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


def polygon_clip(
    subject: list[tuple[float, float]],
    clip: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    Sutherland-Hodgman 多边形裁剪算法，计算两个凸多边形（正方形）重合的交集区域多边形。
    """
    output = subject
    for idx, clip_start in enumerate(clip):
        clip_end = clip[(idx + 1) % len(clip)]
        input_list = output
        output = []
        if not input_list:
            break
        prev = input_list[-1]
        for current in input_list:
            current_inside = is_inside(current, clip_start, clip_end)
            prev_inside = is_inside(prev, clip_start, clip_end)
            if current_inside:
                if not prev_inside:
                    output.append(line_intersection(prev, current, clip_start, clip_end))
                output.append(current)
            elif prev_inside:
                output.append(line_intersection(prev, current, clip_start, clip_end))
            prev = current
    return output


def validate_layouts() -> None:
    """
    验证所有预设靶纸的几何合理性：
    1. 每个正方形不能越界碰到 20mm 的 A4 黑色外边框和下方的标记圆点；
    2. 分离正方形之间的两两重叠率必须严格低于 30%，且绝对不允许有三图形重合交集。
    """
    left = BORDER_WIDTH_MM
    right = A4_WIDTH_MM - BORDER_WIDTH_MM
    top = BORDER_WIDTH_MM
    bottom = A4_HEIGHT_MM - BORDER_WIDTH_MM - MARKER_DIAMETER_MM - 2.0

    for target in TARGETS:
        polygons = [rotated_square_points_mm(square) for square in target.squares]
        # 1. 越界验证
        for square, polygon in zip(target.squares, polygons):
            for x, y in polygon:
                if x < left or x > right or y < top or y > bottom:
                    raise ValueError(f"{target.name}: square out of white area: {square}")

        # 2. 两两重叠率验证（重合面积占小正方形面积比例 <= 30%）
        for idx_a, poly_a in enumerate(polygons):
            for idx_b in range(idx_a + 1, len(polygons)):
                poly_b = polygons[idx_b]
                overlap = polygon_area(polygon_clip(poly_a, poly_b))
                if overlap == 0:
                    continue
                smaller = min(target.squares[idx_a].size_mm**2, target.squares[idx_b].size_mm**2)
                ratio = overlap / smaller
                if ratio > 0.30:
                    raise ValueError(f"{target.name}: pair overlap {ratio:.3f} exceeds 30%")

        # 3. 三者重叠验证
        for idx_a, poly_a in enumerate(polygons):
            for idx_b in range(idx_a + 1, len(polygons)):
                pair = polygon_clip(poly_a, polygons[idx_b])
                if not pair:
                    continue
                for idx_c in range(idx_b + 1, len(polygons)):
                    triple = polygon_clip(pair, polygons[idx_c])
                    if polygon_area(triple) > 1e-6:
                        raise ValueError(f"{target.name}: three squares overlap in one area")


def draw_pdf_common(c: canvas.Canvas) -> None:
    width_pt, height_pt = A4
    border_pt = mm_to_pt(BORDER_WIDTH_MM)
    marker_r_pt = mm_to_pt(MARKER_DIAMETER_MM / 2.0)

    c.setFillColor(white)
    c.rect(0, 0, width_pt, height_pt, stroke=0, fill=1)

    c.setFillColor(black)
    c.rect(0, 0, width_pt, border_pt, stroke=0, fill=1)
    c.rect(0, height_pt - border_pt, width_pt, border_pt, stroke=0, fill=1)
    c.rect(0, 0, border_pt, height_pt, stroke=0, fill=1)
    c.rect(width_pt - border_pt, 0, border_pt, height_pt, stroke=0, fill=1)
    c.circle(width_pt / 2.0, border_pt + marker_r_pt, marker_r_pt, stroke=0, fill=1)


def draw_png_common(draw: ImageDraw.ImageDraw, width_px: int, height_px: int) -> None:
    border_px = mm_to_px(BORDER_WIDTH_MM)
    marker_r_px = mm_to_px(MARKER_DIAMETER_MM / 2.0)

    draw.rectangle([0, 0, width_px, border_px - 1], fill=0)
    draw.rectangle([0, height_px - border_px, width_px, height_px], fill=0)
    draw.rectangle([0, 0, border_px - 1, height_px], fill=0)
    draw.rectangle([width_px - border_px, 0, width_px, height_px], fill=0)

    cx = width_px // 2
    cy = height_px - border_px - marker_r_px
    draw.ellipse([cx - marker_r_px, cy - marker_r_px, cx + marker_r_px, cy + marker_r_px], fill=0)


def draw_pdf_square(c: canvas.Canvas, square: Square) -> None:
    """
    在 ReportLab 画布上绘制实心黑色正方形。
    利用仿射变换矩阵（translate + rotate）实现任意角度倾斜。
    """
    size_pt = mm_to_pt(square.size_mm)
    c.saveState()
    # 平移至正方形物理中心
    c.translate(mm_to_pt(square.cx_mm), top_mm_to_pdf_y(square.cy_mm_from_top))
    # 顺时针旋转指定角度
    c.rotate(-square.angle_deg)
    # 以原点 (0,0) 为正方形几何中心进行对称绘制
    c.rect(-size_pt / 2.0, -size_pt / 2.0, size_pt, size_pt, stroke=0, fill=1)
    c.restoreState()


def square_png_points(square: Square) -> list[tuple[int, int]]:
    return [(mm_to_px(x), mm_to_px(y)) for x, y in rotated_square_points_mm(square)]


def draw_pdf_digit(c: canvas.Canvas, square: Square) -> None:
    """
    在旋转的正方形正中心绘制白色的编号字符。
    """
    if square.digit is None:
        return

    cx = mm_to_pt(square.cx_mm)
    cy = top_mm_to_pdf_y(square.cy_mm_from_top)
    font_size = mm_to_pt(square.size_mm * 0.52) # 字体大小随正方形大小等比例缩放
    text = square.digit

    c.setFillColor(white)
    c.saveState()
    c.translate(cx, cy)
    c.rotate(-square.angle_deg)
    c.setFont("Helvetica-Bold", font_size)
    width = stringWidth(text, "Helvetica-Bold", font_size)
    # Helvetica 基准线稍微偏下，向下偏移 36% 的字体高度以确保视觉垂直完全居中
    c.drawString(-width / 2.0, -font_size * 0.36, text)
    c.restoreState()


def get_digit_font(size_px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    搜寻 Windows 系统的 Arial-Bold 字体，用以在 PNG 图上渲染数字。
    """
    font_size = round(size_px * 0.62)
    candidates = (
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\Arialbd.ttf",
        r"C:\Windows\Fonts\calibrib.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), font_size)
    return ImageFont.load_default()


def draw_png_digit(image: Image.Image, square: Square) -> None:
    """
    在 PIL 图像上渲染倾斜的白色数字：
    - 创建一个单独的临时透明黑板遮罩图层渲染白色字符；
    - 旋转该临时遮罩图层以对齐正方形角度；
    - 最终居中贴合粘贴回原始大图像。
    """
    if square.digit is None:
        return

    size_px = mm_to_px(square.size_mm)
    font = get_digit_font(size_px)
    text = square.digit
    cx = mm_to_px(square.cx_mm)
    cy = mm_to_px(square.cy_mm_from_top)

    mask_size = round(size_px * 1.5)
    mask = Image.new("L", (mask_size, mask_size), 0)
    mask_draw = ImageDraw.Draw(mask)
    bbox = mask_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # 居中渲染
    mask_draw.text(
        (mask_size / 2.0 - text_w / 2.0 - bbox[0], mask_size / 2.0 - text_h / 2.0 - bbox[1]),
        text,
        fill=255,
        font=font,
    )
    # 按正方形偏角旋转数字遮罩层
    rotated = mask.rotate(square.angle_deg, resample=Image.Resampling.BICUBIC, expand=True)
    x = round(cx - rotated.width / 2.0)
    y = round(cy - rotated.height / 2.0)
    # 贴图回主图，白色前景色填充
    image.paste(255, (x, y), rotated)


def generate_pdf(target: Target) -> None:
    """
    生成高级 A4 PDF 矢量文件。
    """
    path = PDF_DIR / f"{target.name}.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    draw_pdf_common(c)

    c.setFillColor(black)
    for square in target.squares:
        draw_pdf_square(c, square)

    # 再次绘制，确保白色数字始终贴在黑色正方形之上层
    for square in target.squares:
        draw_pdf_digit(c, square)

    c.showPage()
    c.save()


def generate_png(target: Target) -> None:
    """
    生成高级 A4 PNG 300 DPI 像素图像。
    """
    path = PNG_DIR / f"{target.name}.png"
    width_px = mm_to_px(A4_WIDTH_MM)
    height_px = mm_to_px(A4_HEIGHT_MM)

    image = Image.new("L", (width_px, height_px), 255)
    draw = ImageDraw.Draw(image)
    draw_png_common(draw, width_px, height_px)

    for square in target.squares:
        draw.polygon(square_png_points(square), fill=0)

    for square in target.squares:
        draw_png_digit(image, square)

    image.save(path)


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    # 运行布局合理性安全验证
    validate_layouts()

    for target in TARGETS:
        generate_pdf(target)
        generate_png(target)

    print(f"Generated {len(TARGETS)} advanced PDFs in {PDF_DIR}")
    print(f"Generated {len(TARGETS)} advanced PNGs in {PNG_DIR}")


if __name__ == "__main__":
    main()
