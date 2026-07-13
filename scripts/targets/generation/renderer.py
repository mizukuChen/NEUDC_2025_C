from PIL import Image, ImageDraw
import numpy as np
from .page import DEFAULT_PAGE, PageSpec, create_page
from .shapes import ShapeSpec, polygon_mm
from .digits import draw_centered_digit

def render(shapes: tuple[ShapeSpec,...], page: PageSpec=DEFAULT_PAGE) -> tuple[np.ndarray,list[dict]]:
    image=create_page(page).convert("RGBA"); s=page.px_per_mm; meta=[]
    for item in shapes:
        if item.shape_type == "circle":
            d=ImageDraw.Draw(image); r=item.size_mm*s/2; cx=item.cx_mm*s; cy=item.cy_mm*s
            d.ellipse((cx-r,cy-r,cx+r,cy+r),fill=(0,0,0,255)); font_size=None
        elif item.digit is None:
            polygon=np.rint(polygon_mm(item)*s).astype(int)
            ImageDraw.Draw(image).polygon([tuple(point) for point in polygon],fill=(0,0,0,255)); font_size=None
        else:
            size=round(item.size_mm*s); component=Image.new("RGBA",(size,size),(0,0,0,0)); d=ImageDraw.Draw(component)
            d.rectangle((0,0,size-1,size-1),fill=(0,0,0,255))
            font_size=draw_centered_digit(d,size,item.digit) if item.digit is not None else None
            rotated=component.rotate(-item.angle_deg,resample=Image.Resampling.BICUBIC,expand=True)
            image.alpha_composite(rotated,(round(item.cx_mm*s-rotated.width/2),round(item.cy_mm*s-rotated.height/2)))
        data=item.__dict__.copy(); data["font_size_px"]=font_size; meta.append(data)
    return np.asarray(image.convert("L")),meta
