from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "assets" / "targets" / "generated"
DEFAULT_OUTPUT = GENERATED / "print" / "representative_targets_A3_portrait.pdf"

MM_TO_PT = 72.0 / 25.4
A3_WIDTH_MM = 297.0
A3_HEIGHT_MM = 420.0
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
A3_SIZE = (A3_WIDTH_MM * MM_TO_PT, A3_HEIGHT_MM * MM_TO_PT)
A4_SIZE = (A4_WIDTH_MM * MM_TO_PT, A4_HEIGHT_MM * MM_TO_PT)
A4_OFFSET = (
    (A3_WIDTH_MM - A4_WIDTH_MM) * MM_TO_PT / 2.0,
    (A3_HEIGHT_MM - A4_HEIGHT_MM) * MM_TO_PT / 2.0,
)


@dataclass(frozen=True)
class PrintTarget:
    category: str
    name: str
    source: Path


SELECTION = (
    PrintTarget("basic", "单圆形 120 mm", GENERATED / "modular/basic/basic_circle_A4.png"),
    PrintTarget("basic", "单三角形 120 mm", GENERATED / "modular/basic/basic_triangle_A4.png"),
    PrintTarget("basic", "单正方形 120 mm", GENERATED / "modular/basic/basic_square_A4.png"),
    PrintTarget("single_square", "单正方形 60 mm", GENERATED / "modular/single_square/single_square_60mm_A4.png"),
    PrintTarget("single_square", "单正方形 80 mm", GENERATED / "modular/single_square/single_square_80mm_A4.png"),
    PrintTarget("single_square", "单正方形 100 mm", GENERATED / "modular/single_square/single_square_100mm_A4.png"),
    PrintTarget("single_square", "单正方形 90 mm、旋转 27 度", GENERATED / "modular/single_square/single_square_90mm_rotated_A4.png"),
    PrintTarget("separate", "分离方块 3 个", GENERATED / "modular/separate/separate_squares_01_A4.png"),
    PrintTarget("separate", "分离方块 4 个", GENERATED / "modular/separate/separate_squares_02_A4.png"),
    PrintTarget("separate", "分离方块 5 个", GENERATED / "modular/separate/separate_squares_03_A4.png"),
    PrintTarget("separate", "分离方块 4 个-布局 B", GENERATED / "modular/separate/separate_squares_04_A4.png"),
    PrintTarget("overlap", "重叠方块 2 个", GENERATED / "overlap_controlled/overlap_controlled_01_A4.png"),
    PrintTarget("overlap", "重叠方块 3 个-链式 A", GENERATED / "overlap_controlled/overlap_controlled_02_A4.png"),
    PrintTarget("overlap", "重叠方块 4 个-链式", GENERATED / "overlap_controlled/overlap_controlled_03_A4.png"),
    PrintTarget("overlap", "重叠方块 3 个-布局 A", GENERATED / "overlap_controlled/overlap_controlled_04_A4.png"),
    PrintTarget("overlap", "重叠方块 2 个-布局 B", GENERATED / "overlap_controlled/overlap_controlled_05_A4.png"),
    PrintTarget("overlap", "重叠方块 3 个-布局 B", GENERATED / "overlap_controlled/overlap_controlled_06_A4.png"),
    PrintTarget("overlap", "重叠方块 3 个-布局 C", GENERATED / "overlap_controlled/overlap_controlled_07_A4.png"),
    PrintTarget("overlap", "重叠方块 4 个", GENERATED / "overlap_controlled/overlap_controlled_08_A4.png"),
    PrintTarget("overlap", "重叠方块 3 个-尺寸差异", GENERATED / "overlap_controlled/overlap_controlled_09_A4.png"),
    PrintTarget("numbered", "分离编号 0、1", GENERATED / "numbered_controlled/numbered_controlled_01_A4.png"),
    PrintTarget("numbered", "分离编号 2、3", GENERATED / "numbered_controlled/numbered_controlled_02_A4.png"),
    PrintTarget("numbered", "分离编号 4、5", GENERATED / "numbered_controlled/numbered_controlled_03_A4.png"),
    PrintTarget("numbered", "分离编号 6、7", GENERATED / "numbered_controlled/numbered_controlled_04_A4.png"),
    PrintTarget("numbered", "分离编号 8、9", GENERATED / "numbered_controlled/numbered_controlled_05_A4.png"),
    PrintTarget("numbered_overlap", "重叠编号 0、1", GENERATED / "numbered_overlap_controlled/numbered_overlap_01_A4.png"),
    PrintTarget("numbered_overlap", "重叠编号 2、3、4", GENERATED / "numbered_overlap_controlled/numbered_overlap_02_A4.png"),
    PrintTarget("numbered_overlap", "重叠编号 5、6、7", GENERATED / "numbered_overlap_controlled/numbered_overlap_03_A4.png"),
    PrintTarget("numbered_overlap", "重叠编号 8、9", GENERATED / "numbered_overlap_controlled/numbered_overlap_04_A4.png"),
    PrintTarget("single_numbered", "正向单编号方块 1、45 mm", GENERATED / "modular/single_numbered/single_numbered_digit_1_45mm_upright_A4.png"),
    PrintTarget("single_numbered", "正向单编号方块 3、55 mm", GENERATED / "modular/single_numbered/single_numbered_digit_3_55mm_upright_A4.png"),
    PrintTarget("single_numbered", "正向单编号方块 5、70 mm", GENERATED / "modular/single_numbered/single_numbered_digit_5_upright_A4.png"),
    PrintTarget("single_numbered", "正向单编号方块 7、85 mm", GENERATED / "modular/single_numbered/single_numbered_digit_7_85mm_upright_A4.png"),
    PrintTarget("single_numbered", "正向单编号方块 9、105 mm", GENERATED / "modular/single_numbered/single_numbered_digit_9_105mm_upright_A4.png"),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="将代表性 A4 目标以真实尺寸居中排入竖版 A3 多页 PDF。"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    missing = [item.source for item in SELECTION if not item.source.exists()]
    if missing:
        parser.error("missing source PNG:\n" + "\n".join(str(path) for path in missing))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    document = canvas.Canvas(str(args.output), pagesize=A3_SIZE, pageCompression=1)
    manifest = []
    for page_number, item in enumerate(SELECTION, start=1):
        # A4 与 A3 均为竖版；不缩放内容，只按 A4 的真实物理尺寸嵌入。
        document.drawImage(
            ImageReader(str(item.source)),
            A4_OFFSET[0],
            A4_OFFSET[1],
            width=A4_SIZE[0],
            height=A4_SIZE[1],
            preserveAspectRatio=True,
            mask=None,
        )
        document.showPage()
        manifest.append(
            {
                "page": page_number,
                "category": item.category,
                "name": item.name,
                "source_png": str(item.source.relative_to(ROOT)),
                "a3_mm": [A3_WIDTH_MM, A3_HEIGHT_MM],
                "a4_mm": [A4_WIDTH_MM, A4_HEIGHT_MM],
                "a4_offset_mm": [
                    (A3_WIDTH_MM - A4_WIDTH_MM) / 2.0,
                    (A3_HEIGHT_MM - A4_HEIGHT_MM) / 2.0,
                ],
            }
        )
    document.save()

    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(
        json.dumps({"pdf": args.output.name, "pages": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"generated {len(SELECTION)} pages: {args.output}")
    print(f"page manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
