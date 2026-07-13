from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2,numpy as np
from src.vision.distance import estimate_plane_center_depth,load_distance_correction,undistorted_camera_matrix
from src.vision.fisheye import load_fisheye_calibration,undistort_fisheye_image
from src.vision.measure import draw_measurements
from src.vision.measurement_tasks import measure_normal,measure_numbered
from src.vision.rectify import rectify_a4
from src.vision.target_plane import find_target_plane

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_FISHEYE=ROOT/"runtime/calibration/fisheye_1280x720.json"
DEFAULT_DISTANCE=ROOT/"runtime/calibration/distance_linear.json"

@dataclass(frozen=True)
class ApplicationResult:
    mode:str; x_mm:float; distance_mm:float; reprojection_error_px:float
    summary:str; rectified_a4:np.ndarray; annotated_a4:np.ndarray

class MeasurementApplication:
    def __init__(self,fisheye_path:Path=DEFAULT_FISHEYE,distance_path:Path=DEFAULT_DISTANCE,strip_fraction:float=1/3,max_reprojection_px:float=2.0):
        if not 0<strip_fraction<=1:raise ValueError("strip_fraction must be in (0, 1].")
        self.fisheye_path=Path(fisheye_path);self.distance_path=Path(distance_path);self.strip_fraction=strip_fraction;self.max_reprojection_px=max_reprojection_px;self._fisheye=None;self._correction=None
    def _load(self):
        if self._fisheye is None:
            if not self.fisheye_path.exists():raise FileNotFoundError(f"缺少鱼眼标定文件：{self.fisheye_path}")
            self._fisheye=load_fisheye_calibration(self.fisheye_path)
            self._correction=load_distance_correction(self.distance_path) if self.distance_path.exists() else None
        return self._fisheye,self._correction
    @staticmethod
    def capture_frame(camera_index:int=0,width:int=1280,height:int=720)->np.ndarray:
        cap=cv2.VideoCapture(camera_index)
        if not cap.isOpened():raise RuntimeError(f"无法打开相机 /dev/video{camera_index}")
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,width);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
            for _ in range(5):cap.grab()
            ok,frame=cap.read()
        finally:cap.release()
        if not ok or frame is None:raise RuntimeError("相机已打开，但未能获取图像")
        return frame
    def process_raw(self,raw:np.ndarray,mode:str,digit:int|None=None)->ApplicationResult:
        if raw is None or raw.size==0:raise ValueError("输入图像为空")
        if mode not in {"normal","numbered"}:raise ValueError(f"未知测量模式：{mode}")
        if mode=="numbered" and (digit is None or not 0<=digit<=9):raise ValueError("数字识别模式必须输入 0～9")
        fisheye,correction=self._load(); undistorted=undistort_fisheye_image(raw,fisheye); h,w=undistorted.shape[:2]
        sw=round(w*self.strip_fraction);left=(w-sw)//2;strip=undistorted[:,left:left+sw];plane=find_target_plane(strip)
        corners=plane.corners.astype(np.float64);corners[:,0]+=left
        distance=estimate_plane_center_depth(corners,undistorted_camera_matrix(fisheye,(w,h)),correction)
        if distance.reprojection_error_px>self.max_reprojection_px:raise ValueError(f"A4 定位重投影误差过大：{distance.reprojection_error_px:.2f} px")
        rectified=rectify_a4(strip,plane.corners)
        if mode=="normal":
            task=measure_normal(rectified,distance.corrected_distance_mm);summary=f"普通识别\n布局：{task.geometry.layout}\n目标数：{len(task.geometry.shapes)}\n尺寸 x：{task.x_mm:.2f} mm\n距离 D：{distance.corrected_distance_mm:.1f} mm"
        else:
            task=measure_numbered(rectified,int(digit),distance.corrected_distance_mm);summary=f"数字识别\n输入数字：{digit}\n识别方块数：{len(task.squares)}\n尺寸 x：{task.x_mm:.2f} mm\n距离 D：{distance.corrected_distance_mm:.1f} mm\n数字置信度：{task.selected_square.digit_confidence:.3f}"
        return ApplicationResult(mode,task.x_mm,distance.corrected_distance_mm,distance.reprojection_error_px,summary,rectified,draw_measurements(rectified,task.geometry))
