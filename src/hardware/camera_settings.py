from __future__ import annotations

"""树莓派摄像头配置、V4L2控制与跨进程设备锁。"""

from dataclasses import asdict, dataclass
import fcntl
import json
from pathlib import Path
import subprocess
from typing import TextIO

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMERA_SETTINGS_PATH = ROOT / "config/camera_runtime.json"
DEFAULT_CAMERA_LOCK_PATH = Path("/tmp/nuedc-camera.lock")


@dataclass(frozen=True)
class CameraSettings:
    auto_exposure: bool = True
    exposure_time_absolute: int = 313
    gain: int = 0
    brightness: int = 0
    contrast: int = 34
    gamma: int = 100
    sharpness: int = 6
    power_line_frequency: int = 1
    roi_left: float = 1.0 / 3.0
    roi_top: float = 0.0
    roi_right: float = 2.0 / 3.0
    roi_bottom: float = 1.0

    def validate(self) -> None:
        ranges = {
            "exposure_time_absolute": (1, 5000),
            "gain": (0, 63),
            "brightness": (-128, 127),
            "contrast": (0, 64),
            "gamma": (1, 500),
            "sharpness": (0, 25),
            "power_line_frequency": (0, 2),
        }
        for name, (minimum, maximum) in ranges.items():
            value = int(getattr(self, name))
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}.")
        boundaries = {
            "roi_left": float(self.roi_left),
            "roi_top": float(self.roi_top),
            "roi_right": float(self.roi_right),
            "roi_bottom": float(self.roi_bottom),
        }
        if any(not 0.0 <= value <= 1.0 for value in boundaries.values()):
            raise ValueError("ROI 边界必须位于画面范围 0～1 内")
        if not self.roi_left < self.roi_right or not self.roi_top < self.roi_bottom:
            raise ValueError("ROI 必须是左边界小于右边界、上边界小于下边界的正矩形")

    def roi_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """返回半开区间形式的像素 ROI：(left, top, right, bottom)。"""
        if width < 1 or height < 1:
            raise ValueError("图像尺寸必须为正数")
        self.validate()
        left = min(width - 1, max(0, round(self.roi_left * width)))
        top = min(height - 1, max(0, round(self.roi_top * height)))
        right = min(width, max(left + 1, round(self.roi_right * width)))
        bottom = min(height, max(top + 1, round(self.roi_bottom * height)))
        return left, top, right, bottom


def load_camera_settings(path: Path = DEFAULT_CAMERA_SETTINGS_PATH) -> CameraSettings:
    path = Path(path)
    if not path.exists():
        return CameraSettings()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("相机配置必须是JSON对象")
    known = {field for field in CameraSettings.__dataclass_fields__}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"相机配置包含未知字段：{', '.join(sorted(unknown))}")
    settings = CameraSettings(**payload)
    settings.validate()
    return settings


def save_camera_settings(
    settings: CameraSettings,
    path: Path = DEFAULT_CAMERA_SETTINGS_PATH,
) -> None:
    settings.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(asdict(settings), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def device_path(camera_index: int) -> str:
    return f"/dev/video{camera_index}"


def _v4l2_values(settings: CameraSettings) -> dict[str, int]:
    values = {
        "auto_exposure": 3 if settings.auto_exposure else 1,
        "gain": settings.gain,
        "brightness": settings.brightness,
        "contrast": settings.contrast,
        "gamma": settings.gamma,
        "sharpness": settings.sharpness,
        "power_line_frequency": settings.power_line_frequency,
    }
    if not settings.auto_exposure:
        values["exposure_time_absolute"] = settings.exposure_time_absolute
    return values


def apply_camera_controls(
    capture,
    settings: CameraSettings,
    camera_device: str | None = None,
) -> None:
    """同时走OpenCV和V4L2控制；V4L2不可用时仍可在其他平台运行。"""
    settings.validate()
    capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if settings.auto_exposure else 1)
    if not settings.auto_exposure:
        capture.set(cv2.CAP_PROP_EXPOSURE, settings.exposure_time_absolute)
    for prop, value in (
        (cv2.CAP_PROP_GAIN, settings.gain),
        (cv2.CAP_PROP_BRIGHTNESS, settings.brightness),
        (cv2.CAP_PROP_CONTRAST, settings.contrast),
        (cv2.CAP_PROP_GAMMA, settings.gamma),
        (cv2.CAP_PROP_SHARPNESS, settings.sharpness),
    ):
        capture.set(prop, value)

    if camera_device is None or not Path(camera_device).exists():
        return
    controls = ",".join(f"{name}={value}" for name, value in _v4l2_values(settings).items())
    try:
        completed = subprocess.run(
            ["v4l2-ctl", "-d", camera_device, "--set-ctrl", controls],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"无法设置相机参数：{exc}") from exc
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"无法设置相机参数：{message}")


def read_camera_controls(camera_device: str) -> dict[str, int]:
    names = tuple(_v4l2_values(CameraSettings())) + ("exposure_time_absolute",)
    try:
        completed = subprocess.run(
            ["v4l2-ctl", "-d", camera_device, "--get-ctrl", ",".join(names)],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"无法读取相机参数：{exc}") from exc
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"无法读取相机参数：{message}")
    values: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        # 菜单类控件会返回 ``3 (Aperture Priority Mode)``，数值位于首字段。
        value_text = raw.strip().split(maxsplit=1)[0]
        values[name.strip()] = int(value_text)
    return values


class CameraDeviceLock:
    """正式测量和调试软件共用的非阻塞相机进程锁。"""

    def __init__(self, path: Path = DEFAULT_CAMERA_LOCK_PATH) -> None:
        self.path = Path(path)
        self._handle: TextIO | None = None

    def acquire(self) -> None:
        if self._handle is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError("相机正被另一个程序占用，请先关闭测量或调试软件") from exc
        self._handle = handle

    def release(self) -> None:
        if self._handle is None:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None

    def __enter__(self) -> CameraDeviceLock:
        self.acquire()
        return self

    def __exit__(self, *_args) -> None:
        self.release()
