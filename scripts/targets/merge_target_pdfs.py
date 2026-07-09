from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader, PdfWriter


ROOT = Path(__file__).resolve().parents[2]
TARGET_ROOT = ROOT / "assets" / "targets" / "generated"
OUTPUT = TARGET_ROOT / "pdf" / "all_targets_A4_18pages.pdf"

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
    missing = [path for path in PDFS if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing PDF files:\n" + "\n".join(str(path) for path in missing))

    writer = PdfWriter()
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
