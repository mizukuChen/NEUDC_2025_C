from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.targets.generation.output import generate_png_json
from scripts.targets.presets.single_numbered import PRESETS

OUTPUT = ROOT / "assets/targets/generated/modular/single_numbered"

if __name__ == "__main__":
    for name, shapes in PRESETS.items():
        generate_png_json(
            f"single_numbered_{name}_A4", shapes, OUTPUT, "numbered_squares", "squares"
        )
