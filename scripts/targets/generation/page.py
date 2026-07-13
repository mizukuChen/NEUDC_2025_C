from dataclasses import dataclass
from PIL import Image, ImageDraw

@dataclass(frozen=True)
class PageSpec:
    width_mm: float = 210.0
    height_mm: float = 297.0
    dpi: int = 300
    border_mm: float = 20.0
    marker_mm: float = 6.0

    @property
    def px_per_mm(self) -> float: return self.dpi / 25.4

DEFAULT_PAGE = PageSpec()

def create_page(spec: PageSpec = DEFAULT_PAGE) -> Image.Image:
    s=spec.px_per_mm; w=round(spec.width_mm*s); h=round(spec.height_mm*s); b=round(spec.border_mm*s)
    image=Image.new("L",(w,h),255); draw=ImageDraw.Draw(image)
    draw.rectangle((0,0,w,b-1),fill=0); draw.rectangle((0,h-b,w,h),fill=0)
    draw.rectangle((0,0,b-1,h),fill=0); draw.rectangle((w-b,0,w,h),fill=0)
    r=round(spec.marker_mm*s/2); cx=w//2; cy=h-b-r; draw.ellipse((cx-r,cy-r,cx+r,cy+r),fill=0)
    return image
