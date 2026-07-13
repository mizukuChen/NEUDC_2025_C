from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from PIL import Image
from scripts.targets.generation.constraints import edge_coverage_ratios,validate_overlap
from scripts.targets.generation.page import DEFAULT_PAGE
from scripts.targets.generation.renderer import render as _render
from scripts.targets.generation.shapes import ShapeSpec as SquareSpec
from scripts.targets.presets.overlap import PRESETS
DEFAULT_OUTPUT=ROOT/"assets"/"targets"/"generated"/"overlap_controlled"; MAX_EDGE_COVERAGE=.5

def render(shapes): return _render(shapes)[0]
def validate(shapes): return validate_overlap(shapes,MAX_EDGE_COVERAGE)
def generate_case(case,shapes,output):
    ratios=validate(shapes); image,items=_render(shapes); output.mkdir(parents=True,exist_ok=True); stem=f"overlap_controlled_{case}_A4"
    Image.fromarray(image).save(output/f"{stem}.png")
    for item,ratio in zip(items,ratios): item["edge_coverage"]=ratio
    (output/f"{stem}.json").write_text(json.dumps({"page_mm":[210,297],"dpi":DEFAULT_PAGE.dpi,"max_edge_coverage":MAX_EDGE_COVERAGE,"squares":items},indent=2),encoding="utf-8")
def main():
    p=argparse.ArgumentParser();p.add_argument("--case",choices=(*PRESETS,"all"),default="all");p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);a=p.parse_args()
    for case,shapes in (PRESETS.items() if a.case=="all" else ((a.case,PRESETS[a.case]),)): generate_case(case,shapes,a.output)
    return 0
if __name__=="__main__": raise SystemExit(main())
