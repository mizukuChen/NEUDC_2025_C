from __future__ import annotations

"""
基础 A4 靶标（圆形、正方形、等边三角形）生成器。

本脚本利用 ReportLab 库生成矢量格式的 PDF 靶纸，同时利用 PIL 库生成 300 DPI 像素分辨率的 PNG 图像。
所有靶纸均带有 20mm 黑色大边框，下方中点处标有直径为 6mm 的定位黑点。
生成的物理靶标图形边长/直径默认为 120.0mm，位于 A4 纸面中央。
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw
from reportlab.lib.colors import black, white
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
TARGET_ROOT = ROOT / "assets" / "targets" / "generated"
PDF_DIR = TARGET_ROOT / "pdf"
PNG_DIR = TARGET_ROOT / "png"

# 物理与像素/排版单位换算常数
MM_TO_PT = 72.0 / 25.4 # 毫米转 PDF 磅数 (points)
DPI = 300
MM_TO_PX = DPI / 25.4 # 毫米转 300DPI 像素数

# 题目规定的 A4 靶标物理排版标准参数
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
BORDER_WIDTH_MM = 20.0
TARGET_SIZE_MM = 120.0
MARKER_DIAMETER_MM = 6.0


def mm_to_pt(value: float) -> float:
    """
    毫米物理尺寸转 PDF 磅数单位。
    """
    return value * MM_TO_PT


def mm_to_px(value: float) -> int:
    """
    毫米物理尺寸转像素单位。
    """
    return round(value * MM_TO_PX)


def pdf_path(name: str) -> Path:
    return PDF_DIR / f"basic_{name}_A4.pdf"


def png_path(name: str) -> Path:
    return PNG_DIR / f"basic_{name}_A4.png"


def draw_pdf_common(c: canvas.Canvas) -> None:
    """
    在 ReportLab 矢量画布上绘制 A4 靶纸的公共框架：
    - 填充白色背景
    - 绘制四周 20mm 的黑色边框
    - 绘制下方中心直径为 6mm 的定位标志圆点
    """
    width_pt, height_pt = A4
    border_pt = mm_to_pt(BORDER_WIDTH_MM)
    marker_r_pt = mm_to_pt(MARKER_DIAMETER_MM / 2.0)

    # 1. 填白底
    c.setFillColor(white)
    c.rect(0, 0, width_pt, height_pt, stroke=0, fill=1)

    # 2. 画黑边
    c.setFillColor(black)
    c.rect(0, 0, width_pt, border_pt, stroke=0, fill=1)
    c.rect(0, height_pt - border_pt, width_pt, border_pt, stroke=0, fill=1)
    c.rect(0, 0, border_pt, height_pt, stroke=0, fill=1)
    c.rect(width_pt - border_pt, 0, border_pt, height_pt, stroke=0, fill=1)

    # 3. 画下侧圆点
    c.circle(width_pt / 2.0, border_pt + marker_r_pt, marker_r_pt, stroke=0, fill=1)


def draw_png_common(draw: ImageDraw.ImageDraw, width_px: int, height_px: int) -> None:
    """
    在 PIL 图像画布上绘制 A4 靶标公共黑边框和下方定位黑点。
    """
    border_px = mm_to_px(BORDER_WIDTH_MM)
    marker_r_px = mm_to_px(MARKER_DIAMETER_MM / 2.0)

    # 绘制四周矩形黑色边框
    draw.rectangle([0, 0, width_px, border_px - 1], fill=0)
    draw.rectangle([0, height_px - border_px, width_px, height_px], fill=0)
    draw.rectangle([0, 0, border_px - 1, height_px], fill=0)
    draw.rectangle([width_px - border_px, 0, width_px, height_px], fill=0)

    # 绘制定位圆点
    cx = width_px // 2
    cy = height_px - border_px - marker_r_px
    draw.ellipse(
        [cx - marker_r_px, cy - marker_r_px, cx + marker_r_px, cy + marker_r_px],
        fill=0,
    )


def triangle_points_pdf(cx: float, cy: float, side: float) -> list[tuple[float, float]]:
    """
    根据给定的重心物理位置和边长，计算等边三角形在 PDF 画布坐标系（以左下角为原点）下的 3 个顶点坐标。
    """
    height = side * math.sqrt(3.0) / 2.0
    return [
        (cx, cy + (2.0 * height / 3.0)),       # 顶角
        (cx - side / 2.0, cy - height / 3.0),  # 左底角
        (cx + side / 2.0, cy - height / 3.0),  # 右底角
    ]


def triangle_points_png(cx: int, cy: int, side: int) -> list[tuple[int, int]]:
    """
    根据重心像素位置和像素边长，计算等边三角形在 PIL 图像坐标系（以左上角为原点）下的 3 个顶点坐标。
    """
    height = side * math.sqrt(3.0) / 2.0
    return [
        (cx, round(cy - (2.0 * height / 3.0))),  # 顶角（y轴向下）
        (round(cx - side / 2.0), round(cy + height / 3.0)),
        (round(cx + side / 2.0), round(cy + height / 3.0)),
    ]


def generate_pdf(name: str) -> None:
    """
    生成单图形基础 A4 PDF 矢量文件。
    """
    path = pdf_path(name)
    width_pt, height_pt = A4
    target_pt = mm_to_pt(TARGET_SIZE_MM)
    cx = width_pt / 2.0
    cy = height_pt / 2.0

    c = canvas.Canvas(str(path), pagesize=A4)
    draw_pdf_common(c)
    c.setFillColor(black)
    c.setStrokeColor(black)

    # 在页面重心处绘制指定形状
    if name == "circle":
        c.circle(cx, cy, target_pt / 2.0, stroke=0, fill=1)
    elif name == "triangle":
        path_obj = c.beginPath()
        points = triangle_points_pdf(cx, cy, target_pt)
        path_obj.moveTo(*points[0])
        path_obj.lineTo(*points[1])
        path_obj.lineTo(*points[2])
        path_obj.close()
        c.drawPath(path_obj, stroke=0, fill=1)
    elif name == "square":
        c.rect(cx - target_pt / 2.0, cy - target_pt / 2.0, target_pt, target_pt, stroke=0, fill=1)
    else:
        raise ValueError(f"unknown target name: {name}")

    c.showPage()
    c.save()


def generate_png(name: str) -> None:
    """
    生成单图形基础 A4 PNG 300 DPI 像素图像。
    """
    path = png_path(name)
    width_px = mm_to_px(A4_WIDTH_MM)
    height_px = mm_to_px(A4_HEIGHT_MM)
    target_px = mm_to_px(TARGET_SIZE_MM)
    cx = width_px // 2
    cy = height_px // 2

    # 创建白色灰度底图 (8位)
    image = Image.new("L", (width_px, height_px), 255)
    draw = ImageDraw.Draw(image)
    draw_png_common(draw, width_px, height_px)

    if name == "circle":
        r = target_px // 2
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=0)
    elif name == "triangle":
        draw.polygon(triangle_points_png(cx, cy, target_px), fill=0)
    elif name == "square":
        half = target_px // 2
        draw.rectangle([cx - half, cy - half, cx + half, cy + half], fill=0)
    else:
        raise ValueError(f"unknown target name: {name}")

    image.save(path)


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    for name in ("circle", "triangle", "square"):
        generate_pdf(name)
        generate_png(name)

    print(f"Generated PDFs in {PDF_DIR}")
    print(f"Generated PNGs in {PNG_DIR}")


if __name__ == "__main__":
    main()
