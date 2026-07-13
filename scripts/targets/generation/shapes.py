from dataclasses import dataclass
from math import cos, radians, sin, sqrt
import numpy as np

@dataclass(frozen=True)
class ShapeSpec:
    shape_type: str; cx_mm: float; cy_mm: float; size_mm: float; angle_deg: float = 0.0; digit: int|None = None

def polygon_mm(shape: ShapeSpec) -> np.ndarray:
    if shape.shape_type == "square":
        h=shape.size_mm/2; points=np.array([[-h,-h],[h,-h],[h,h],[-h,h]])
    elif shape.shape_type == "triangle":
        h=shape.size_mm*sqrt(3)/2; points=np.array([[0,-2*h/3],[shape.size_mm/2,h/3],[-shape.size_mm/2,h/3]])
    else: raise ValueError("polygon_mm only supports square and triangle")
    a=radians(shape.angle_deg); r=np.array([[cos(a),-sin(a)],[sin(a),cos(a)]])
    return points@r.T+np.array([shape.cx_mm,shape.cy_mm])
