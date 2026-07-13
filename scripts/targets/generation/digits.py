from pathlib import Path
from PIL import ImageDraw, ImageFont

FONT_PATH=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
FONT_RATIO=0.52

def draw_centered_digit(draw: ImageDraw.ImageDraw, size_px: int, digit: int) -> int:
    font_size=round(size_px*FONT_RATIO); font=ImageFont.truetype(str(FONT_PATH),font_size)
    draw.text((size_px/2,size_px/2),str(digit),font=font,fill=255,anchor="mm")
    return font_size
