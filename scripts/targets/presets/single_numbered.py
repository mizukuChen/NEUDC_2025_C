from scripts.targets.generation.shapes import ShapeSpec as S

PRESETS = {
    "digit_1_45mm_upright": (S("square", 105, 148.5, 45, 0, 1),),
    "digit_3_55mm_upright": (S("square", 105, 148.5, 55, 0, 3),),
    "digit_5_upright": (S("square", 105, 148.5, 70, 0, 5),),
    "digit_7_85mm_upright": (S("square", 105, 148.5, 85, 0, 7),),
    "digit_9_105mm_upright": (S("square", 105, 148.5, 105, 0, 9),),
}
