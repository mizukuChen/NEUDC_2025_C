from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import black, white
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "assets" / "calibration" / "fisheye_chessboard_A3_8x12_29mm.pdf"

MM_TO_PT = 72.0 / 25.4
A3_WIDTH_MM = 297.0
A3_HEIGHT_MM = 420.0

# OpenCV's (9, 6) pattern size describes inner corners, not drawn squares.
INNER_CORNERS_X = 8
INNER_CORNERS_Y = 12
SQUARE_SIZE_MM = 29.0
SQUARES_X = INNER_CORNERS_X + 1
SQUARES_Y = INNER_CORNERS_Y + 1


def mm_to_pt(value: float) -> float:
    return value * MM_TO_PT


def draw_chessboard(c: canvas.Canvas) -> None:
    board_width_mm = SQUARES_X * SQUARE_SIZE_MM
    board_height_mm = SQUARES_Y * SQUARE_SIZE_MM
    origin_x_mm = (A3_WIDTH_MM - board_width_mm) / 2.0
    origin_y_mm = (A3_HEIGHT_MM - board_height_mm) / 2.0
    square_size_pt = mm_to_pt(SQUARE_SIZE_MM)

    c.setFillColor(white)
    c.rect(0, 0, mm_to_pt(A3_WIDTH_MM), mm_to_pt(A3_HEIGHT_MM), stroke=0, fill=1)
    c.setFillColor(black)
    for row in range(SQUARES_Y):
        for column in range(SQUARES_X):
            if (row + column) % 2 == 0:
                x = mm_to_pt(origin_x_mm + column * SQUARE_SIZE_MM)
                # ReportLab uses a bottom-left origin; row 0 is visually at the top.
                y = mm_to_pt(origin_y_mm + (SQUARES_Y - 1 - row) * SQUARE_SIZE_MM)
                c.rect(x, y, square_size_pt, square_size_pt, stroke=0, fill=1)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    page_size = (mm_to_pt(A3_WIDTH_MM), mm_to_pt(A3_HEIGHT_MM))
    pdf = canvas.Canvas(str(OUTPUT_PATH), pagesize=page_size)
    pdf.setTitle("Fisheye calibration chessboard - 9x6 inner corners - 25 mm squares")
    pdf.setAuthor("NUEDC 2025 C target tools")
    draw_chessboard(pdf)
    pdf.showPage()
    pdf.save()
    print(f"Generated: {OUTPUT_PATH}")
    print(f"Page: A3 portrait ({A3_WIDTH_MM:.0f} x {A3_HEIGHT_MM:.0f} mm)")
    print(f"Pattern: {INNER_CORNERS_X} x {INNER_CORNERS_Y} inner corners")
    print(f"Grid: {SQUARES_X} x {SQUARES_Y} squares, {SQUARE_SIZE_MM:.1f} mm per square")


if __name__ == "__main__":
    main()
