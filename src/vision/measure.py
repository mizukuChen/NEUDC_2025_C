from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.vision.measure_overlap import reconstruct_overlapping_squares
from src.vision.measure_shapes import (
    A4_HEIGHT_MM, A4_WIDTH_MM, BORDER_WIDTH_MM, CONTENT_MARGIN_MM,
    ShapeMeasurementResult, measure_shapes,
)


@dataclass(frozen=True)
class Measurement:
    shape_type: str
    x_mm: float
    center_mm: tuple[float, float]
    angle_deg: float
    confidence: float
    source: str
    contour: np.ndarray | None = None
    corners: np.ndarray | None = None
    area_mm2: float | None = None


@dataclass(frozen=True)
class MeasurementResult:
    shapes: tuple[Measurement, ...]
    layout: str
    pixels_per_mm: float
    contour_foreground_coverage: float


def _foreground_mask(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    h,w=image.shape[:2]; sx=w/A4_WIDTH_MM; sy=h/A4_HEIGHT_MM
    inset=BORDER_WIDTH_MM+CONTENT_MARGIN_MM; x0=round(inset*sx); y0=round(inset*sy)
    roi=image[y0:h-y0,x0:w-x0]; gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY) if roi.ndim==3 else roi
    _,mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return mask,(x0,y0)


def _coverage(image: np.ndarray, result: ShapeMeasurementResult) -> float:
    foreground,(x0,y0)=_foreground_mask(image); active=foreground>0; total=int(np.count_nonzero(active))
    if not total:return 0.0
    explained=np.zeros_like(foreground)
    for shape in result.shapes:
        contour=shape.contour.copy(); contour[:,0,0]-=x0; contour[:,0,1]-=y0
        cv2.drawContours(explained,[contour],-1,255,cv2.FILLED)
    return float(np.count_nonzero((explained>0)&active)/total)


def measure(image: np.ndarray, contour_coverage_threshold: float=0.92) -> MeasurementResult:
    if image is None or image.size==0: raise ValueError("Input image is empty.")
    contour_result=None; coverage=0.0
    try:
        contour_result=measure_shapes(image); coverage=_coverage(image,contour_result)
    except ValueError: pass
    if contour_result is not None and coverage>=contour_coverage_threshold:
        shapes=tuple(Measurement(x.shape_type,x.x_mm,x.center_mm,x.angle_deg,x.confidence,"contour",contour=x.contour,area_mm2=x.area_mm2) for x in contour_result.shapes)
        return MeasurementResult(shapes,contour_result.layout,contour_result.pixels_per_mm,coverage)
    result=reconstruct_overlapping_squares(image); sx=image.shape[1]/A4_WIDTH_MM; sy=image.shape[0]/A4_HEIGHT_MM
    shapes=tuple(Measurement("square",x.side_mm,x.center_mm,x.angle_deg,x.confidence,"overlap_lines",corners=x.corners,area_mm2=x.side_mm*x.side_mm) for x in result.squares)
    if not shapes: raise ValueError("No valid shapes found by either measurement module.")
    return MeasurementResult(shapes,"overlap",(sx+sy)/2,coverage)


def draw_measurements(
    image: np.ndarray,
    result: MeasurementResult,
    selected_index: int | None = None,
) -> np.ndarray:
    debug=image.copy(); colors={"circle":(255,0,0),"triangle":(0,165,255),"square":(0,0,255)}
    for i,shape in enumerate(result.shapes,1):
        color=(0,180,0) if selected_index==i-1 else colors[shape.shape_type]
        if shape.contour is not None:
            cv2.drawContours(debug,[shape.contour],-1,color,4,cv2.LINE_AA); origin=shape.contour.reshape(-1,2).min(axis=0)
        elif shape.corners is not None:
            corners=np.rint(shape.corners).astype(np.int32).reshape(-1,1,2); cv2.polylines(debug,[corners],True,color,4,cv2.LINE_AA); origin=corners.reshape(-1,2).min(axis=0)
        else: continue
        marker="*" if selected_index==i-1 else ""
        cv2.putText(debug,f"{marker}{i} {shape.shape_type} {shape.x_mm:.1f} mm",tuple(np.rint(origin).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,1.1,color,3,cv2.LINE_AA)
    return debug
