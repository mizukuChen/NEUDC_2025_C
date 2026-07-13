import json
from pathlib import Path
from PIL import Image
from .constraints import validate
from .page import DEFAULT_PAGE
from .renderer import render

def generate_png_json(name: str, shapes: tuple, output: Path, target_type: str, collection_key: str="shapes") -> None:
    validate(shapes); image,items=render(shapes); output.mkdir(parents=True,exist_ok=True)
    Image.fromarray(image).save(output/f"{name}.png")
    payload={"page_mm":[DEFAULT_PAGE.width_mm,DEFAULT_PAGE.height_mm],"dpi":DEFAULT_PAGE.dpi,"target_type":target_type,collection_key:items}
    (output/f"{name}.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
