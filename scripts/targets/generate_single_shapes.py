from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.targets.generation.output import generate_png_json
from scripts.targets.presets.basic import PRESETS
OUTPUT=ROOT/"assets"/"targets"/"generated"/"modular"/"basic"
if __name__=="__main__":
    for name,shapes in PRESETS.items(): generate_png_json(f"basic_{name}_A4",shapes,OUTPUT,"single_shape")
