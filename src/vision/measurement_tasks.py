from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.vision.measure import Measurement, MeasurementResult, measure
from src.vision.recognize_digits import DigitRecognition, HogSvmDigitRecognizer

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_DIGIT_MODEL=ROOT/"assets/models/digit_hog_svm.joblib"
DEFAULT_DIGIT_METADATA=ROOT/"assets/models/digit_hog_svm_centroids.npz"
A4_WIDTH_MM=210.0; A4_HEIGHT_MM=297.0

@dataclass(frozen=True)
class NormalMeasurementResult:
    distance_mm: float|None
    geometry: MeasurementResult
    selected_shape: Measurement
    x_mm: float

@dataclass(frozen=True)
class NumberedSquareMeasurement:
    geometry: Measurement
    digit: int
    digit_confidence: float
    digit_patch: np.ndarray

@dataclass(frozen=True)
class NumberedMeasurementResult:
    distance_mm: float|None
    requested_digit: int
    geometry: MeasurementResult
    squares: tuple[NumberedSquareMeasurement,...]
    selected_square: NumberedSquareMeasurement
    x_mm: float

def _select_normal_shape(geometry: MeasurementResult)->Measurement:
    if len(geometry.shapes)==1:return geometry.shapes[0]
    if not geometry.shapes:raise ValueError("No geometry was detected.")
    if not all(x.shape_type=="square" for x in geometry.shapes):raise ValueError("A multi-target normal measurement must contain only squares.")
    return min(geometry.shapes,key=lambda x:x.x_mm)

def measure_normal(rectified_a4:np.ndarray,distance_mm:float|None=None)->NormalMeasurementResult:
    geometry=measure(rectified_a4); selected=_select_normal_shape(geometry)
    return NormalMeasurementResult(distance_mm,geometry,selected,selected.x_mm)

def measure_numbered(rectified_a4:np.ndarray,requested_digit:int,distance_mm:float|None=None,recognizer:HogSvmDigitRecognizer|None=None)->NumberedMeasurementResult:
    if not 0<=requested_digit<=9:raise ValueError("requested_digit must be between 0 and 9.")
    geometry=measure(rectified_a4)
    if not geometry.shapes or not all(x.shape_type=="square" for x in geometry.shapes):raise ValueError("Numbered measurement requires square targets only.")
    digit_recognizer=recognizer or HogSvmDigitRecognizer.load(DEFAULT_DIGIT_MODEL,DEFAULT_DIGIT_METADATA)
    sx=rectified_a4.shape[1]/A4_WIDTH_MM; sy=rectified_a4.shape[0]/A4_HEIGHT_MM; scale=(sx+sy)/2
    numbered=[]
    for shape in geometry.shapes:
        recognition:DigitRecognition=digit_recognizer.recognize_square(rectified_a4,(shape.center_mm[0]*sx,shape.center_mm[1]*sy),shape.x_mm*scale,shape.angle_deg)
        numbered.append(NumberedSquareMeasurement(shape,recognition.digit,recognition.confidence,recognition.patch))
    matches=[x for x in numbered if x.digit==requested_digit]
    if not matches:raise ValueError(f"Requested digit {requested_digit} was not recognized in the image.")
    if len(matches)>1:raise ValueError(f"Requested digit {requested_digit} is ambiguous in the image.")
    selected=matches[0]
    return NumberedMeasurementResult(distance_mm,requested_digit,geometry,tuple(numbered),selected,selected.geometry.x_mm)
