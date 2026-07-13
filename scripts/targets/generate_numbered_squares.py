from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.targets.generation.output import generate_png_json
from scripts.targets.generation.renderer import render
from scripts.targets.generation.digits import FONT_PATH, FONT_RATIO as FONT_HEIGHT_RATIO
from scripts.targets.generation.shapes import ShapeSpec as NumberedSquareSpec
from scripts.targets.presets.numbered import PRESETS

DEFAULT_OUTPUT=ROOT/"assets"/"targets"/"generated"/"numbered_controlled"
DPI=300; PX_PER_MM=DPI/25.4

def main()->int:
    p=argparse.ArgumentParser(description="Generate numbered-square PNG/JSON targets.")
    p.add_argument("--case",choices=(*PRESETS,"all"),default="all"); p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); a=p.parse_args()
    cases=PRESETS.items() if a.case=="all" else ((a.case,PRESETS[a.case]),)
    for case,shapes in cases:
        generate_png_json(f"numbered_controlled_{case}_A4",shapes,a.output,"numbered_squares","squares")
        print(f"generated numbered_controlled_{case}_A4.png digits={[s.digit for s in shapes]}")
    return 0
if __name__=="__main__": raise SystemExit(main())
