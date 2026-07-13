from scripts.targets.generation.shapes import ShapeSpec as S

PRESETS = {
    "60mm": (S("square", 105, 148.5, 60, 0),),
    "80mm": (S("square", 105, 148.5, 80, 0),),
    "100mm": (S("square", 105, 148.5, 100, 0),),
    "90mm_rotated": (S("square", 105, 148.5, 90, 27),),
}
