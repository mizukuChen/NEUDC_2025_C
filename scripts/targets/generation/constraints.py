import numpy as np
import cv2
from .shapes import ShapeSpec, polygon_mm

def validate(shapes: tuple[ShapeSpec,...], inset_mm: float=25.0) -> None:
    for i,item in enumerate(shapes):
        if not -45<=item.angle_deg<=45: raise ValueError(f"shape[{i}] angle out of range")
        if item.digit is not None and not 0<=item.digit<=9: raise ValueError(f"shape[{i}] digit out of range")
        if item.shape_type != "circle":
            p=polygon_mm(item)
            if np.any(p[:,0]<inset_mm) or np.any(p[:,0]>210-inset_mm) or np.any(p[:,1]<inset_mm) or np.any(p[:,1]>297-inset_mm):
                raise ValueError(f"shape[{i}] outside measurement ROI")
        elif item.cx_mm-item.size_mm/2<inset_mm or item.cx_mm+item.size_mm/2>210-inset_mm or item.cy_mm-item.size_mm/2<inset_mm or item.cy_mm+item.size_mm/2>297-inset_mm:
            raise ValueError(f"shape[{i}] outside measurement ROI")

def edge_coverage_ratios(shapes: tuple[ShapeSpec,...], samples: int=2001) -> list[list[float]]:
    polygons=[polygon_mm(s).astype(np.float32) for s in shapes]; result=[]
    for i,p in enumerate(polygons):
        values=[]
        for edge in range(4):
            t=np.linspace(0,1,samples)[1:-1,None]; points=p[edge]*(1-t)+p[(edge+1)%4]*t; covered=np.zeros(len(points),bool)
            for j,other in enumerate(polygons):
                if i!=j: covered|=np.fromiter((cv2.pointPolygonTest(other,tuple(map(float,q)),False)>0 for q in points),bool,len(points))
            values.append(float(np.mean(covered)))
        result.append(values)
    return result

def validate_overlap(shapes: tuple[ShapeSpec,...], max_coverage: float=0.5) -> list[list[float]]:
    validate(shapes,inset_mm=20.0); ratios=edge_coverage_ratios(shapes)
    if max(map(max,ratios))>=max_coverage: raise ValueError("square edge coverage exceeds limit")
    return ratios

def validate_numbered_overlap(
    shapes: tuple[ShapeSpec,...],
    max_coverage: float=0.5,
    digit_safe_radius_ratio: float=0.30,
    min_overlap_area_mm2: float=20.0,
) -> tuple[list[list[float]],list[float]]:
    """验证编号重叠图：边可见、数字安全区无遮挡，且每个方块都参与重叠。"""
    ratios=validate_overlap(shapes,max_coverage)
    digits=[item.digit for item in shapes]
    if any(digit is None for digit in digits) or len(set(digits))!=len(digits):
        raise ValueError("numbered overlap digits must exist and be unique")
    polygons=[polygon_mm(item).astype(np.float32) for item in shapes]
    clearances=[]; overlaps=[False]*len(shapes)
    for i,item in enumerate(shapes):
        clearance=float("inf")
        for j,other in enumerate(polygons):
            if i==j: continue
            signed=cv2.pointPolygonTest(other,(float(item.cx_mm),float(item.cy_mm)),True)
            clearance=min(clearance,-float(signed))
            area=float(cv2.intersectConvexConvex(polygons[i],other)[0])
            if area>=min_overlap_area_mm2: overlaps[i]=True
        required=item.size_mm*digit_safe_radius_ratio
        if clearance<required:
            raise ValueError(f"digit {item.digit} safe area is covered")
        clearances.append(clearance)
    if not all(overlaps): raise ValueError("every numbered square must overlap another square")
    return ratios,clearances
