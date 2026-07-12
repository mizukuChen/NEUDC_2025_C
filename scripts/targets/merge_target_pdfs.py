from __future__ import annotations

"""
多页 A4 靶标 PDF 拼接合并脚本。

本脚本搜集生成的基础形状与高级复杂形状的 A4 PDF 文件，
使用 `pypdf` 库将这些独立的单页 PDF 合并为一个包含了所有 18 页测试靶标的多页大 PDF 文件，
以便于用户进行一次性的双面/单面批量打印。
"""

from pathlib import Path

from pypdf import PdfReader, PdfWriter


ROOT = Path(__file__).resolve().parents[2]
TARGET_ROOT = ROOT / "assets" / "targets" / "generated"
OUTPUT = TARGET_ROOT / "pdf" / "all_targets_A4_18pages.pdf"

# 待合并的所有 18 个单页 PDF 文件列表
PDFS = [
    TARGET_ROOT / "pdf" / "basic_circle_A4.pdf",
    TARGET_ROOT / "pdf" / "basic_triangle_A4.pdf",
    TARGET_ROOT / "pdf" / "basic_square_A4.pdf",
    TARGET_ROOT / "advanced" / "pdf" / "advanced_single_square_60mm_A4.pdf",
    TARGET_ROOT / "advanced" / "pdf" / "advanced_single_square_80mm_A4.pdf",
    TARGET_ROOT / "advanced" / "pdf" / "advanced_single_square_100mm_A4.pdf",
    TARGET_ROOT / "advanced" / "pdf" / "advanced_separate_squares_01_A4.pdf",
    TARGET_ROOT / "advanced" / "pdf" / "advanced_separate_squares_02_A4.pdf",
    TARGET_ROOT / "advanced" / "pdf" / "advanced_separate_squares_03_A4.pdf",
    TARGET_ROOT / "advanced" / "pdf" / "advanced_separate_squares_04_A4.pdf",
    TARGET_ROOT / "advanced" / "pdf" / "advanced_overlap_squares_01_A4.pdf",
    TARGET_ROOT / "advanced" / "pdf" / "advanced_overlap_squares_02_A4.pdf",
    TARGET_ROOT / "advanced" / "pdf" / "advanced_overlap_squares_03_A4.pdf",
    TARGET_ROOT / "advanced" / "pdf" / "advanced_overlap_squares_04_A4.pdf",
    TARGET_ROOT / "advanced" / "pdf" / "advanced_numbered_squares_01_A4.pdf",
    TARGET_ROOT / "advanced" / "pdf" / "advanced_numbered_squares_02_A4.pdf",
    TARGET_ROOT / "advanced" / "pdf" / "advanced_numbered_squares_03_A4.pdf",
    TARGET_ROOT / "advanced" / "pdf" / "advanced_numbered_squares_04_A4.pdf",
]


def main() -> None:
    # 校验所有输入的单页 PDF 是否存在
    missing = [path for path in PDFS if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing PDF files:\n" + "\n".join(str(path) for path in missing))

    writer = PdfWriter()
    # 逐一读取并拼接
    for path in PDFS:
        reader = PdfReader(str(path))
        if len(reader.pages) != 1:
            raise ValueError(f"{path} should contain exactly one page")
        writer.add_page(reader.pages[0])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("wb") as file:
        writer.write(file)

    print(f"Merged {len(PDFS)} PDFs into {OUTPUT}")


if __name__ == "__main__":
    main()
