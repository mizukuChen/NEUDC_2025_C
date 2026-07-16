from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from PIL import Image

from src.application import DEFAULT_DISTANCE, DEFAULT_FISHEYE, MeasurementApplication, _draw_a4_location
from src.camera_auto_tune import (
    AutoTuneEvaluation,
    AutoTuneMetrics,
    auto_tune_candidates,
    evaluate_camera_health,
)
from src.camera_settings import (
    DEFAULT_CAMERA_SETTINGS_PATH,
    CameraDeviceLock,
    CameraSettings,
    apply_camera_controls,
    device_path,
    load_camera_settings,
    read_camera_controls,
    save_camera_settings,
)
from src.vision.lowres_preprocess import estimate_black_probability
from src.vision.rectify import rectify_a4
from src.vision.target_plane import find_target_plane


VIEW_FULL = "去畸变完整画面"
VIEW_LOCATION = "A4 定位"
VIEW_RECTIFIED = "拉直后的 A4"
VIEW_BINARY = "黑色概率图"

HELP_TEXT = """相机环境调试使用说明

1. 使用前先关闭正式测量软件。两个软件不能同时占用摄像头。
2. 在“ROI”页签独立调整上、下、左、右边界；“去畸变完整画面”中的蓝色矩形必须完整包住A4黑框。
3. 查看“A4定位”，绿色四边形应准确贴合A4黑框；定位失败时调整ROI、曝光或照明。
4. 重点查看“拉直后的A4”和“黑色概率图”：黑框及目标边缘应连续，白纸不应变黑，数字笔画不能粘连或消失。
5. 建议优先使用自动曝光。需要固定画面时关闭自动曝光，再调曝光时间；画面仍暗时才提高增益。
6. 对比度过高会损失边缘层次，锐度过高会产生双边和LSD伪线。通常保持默认值即可。
7. “自动调参”会尝试少量候选参数，重点抑制数字边缘和双边缘；调参期间进度条会显示当前阶段，完成后自动应用并保存最高分配置。

按钮说明
• 临时应用：立即作用于当前相机，本次退出后不改变正式测量配置。
• 保存为测量配置：应用参数并写入 config/camera_runtime.json，正式测量下次启动时读取。
• 恢复默认：恢复当前已验证的基准值，但仍需点击“保存为测量配置”才会永久保存。

诊断指标
• 黑框灰度越低越黑，白纸灰度越高越白。
• 黑白差值建议明显大于20；小于20时识别会判定对比度不足。
• 过暗/过曝比例过高表示灰度已经剪切，应优先调整曝光而不是修改识别阈值。
"""


@dataclass(frozen=True)
class CameraDiagnostics:
    full: np.ndarray
    located: np.ndarray
    rectified: np.ndarray
    probability: np.ndarray
    message: str


class DiagnosticProcessor:
    def __init__(self, fisheye_path: Path, distance_path: Path) -> None:
        self.application = MeasurementApplication(fisheye_path, distance_path)

    def process(self, raw: np.ndarray, settings: CameraSettings) -> CameraDiagnostics:
        fisheye, _ = self.application._load()
        height, width = raw.shape[:2]
        fixed, interpolation, _, _, _ = self.application._camera_undistortion_maps(
            fisheye,
            (width, height),
        )
        undistorted = cv2.remap(
            raw,
            fixed,
            interpolation,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        left, top, right, bottom = settings.roi_pixels(width, height)
        # ROI是在去畸变坐标系中裁剪的，因此调试预览也必须使用同一图像，
        # 否则鱼眼原图上的直线框并不对应算法真正搜索的区域。
        full = undistorted.copy()
        cv2.rectangle(
            full,
            (left, top),
            (right - 1, bottom - 1),
            (255, 180, 0),
            3,
            cv2.LINE_AA,
        )
        roi = undistorted[top:bottom, left:right]
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        dark_ratio = float(np.mean(gray <= 5)) * 100.0
        bright_ratio = float(np.mean(gray >= 250)) * 100.0
        try:
            plane = find_target_plane(roi)
            full_corners = plane.corners.astype(np.float64)
            full_corners += np.asarray([left, top], dtype=np.float64)
            located = _draw_a4_location(undistorted, full_corners)
            cv2.rectangle(
                located,
                (left, top),
                (right - 1, bottom - 1),
                (255, 180, 0),
                3,
                cv2.LINE_AA,
            )
            rectified = rectify_a4(roi, plane.corners, 420, 594)
            probability, black, white = estimate_black_probability(rectified)
            probability_u8 = np.rint(probability * 255.0).astype(np.uint8)
            probability_bgr = cv2.cvtColor(probability_u8, cv2.COLOR_GRAY2BGR)
            message = (
                f"A4定位成功｜黑框灰度 {black:.1f}｜白纸灰度 {white:.1f}｜"
                f"黑白差 {white - black:.1f}｜ROI {right-left}×{bottom-top}px｜"
                f"过暗 {dark_ratio:.2f}%｜过曝 {bright_ratio:.2f}%"
            )
        except Exception as exc:
            located = undistorted.copy()
            cv2.rectangle(
                located,
                (left, top),
                (right - 1, bottom - 1),
                (255, 180, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                located,
                "A4 NOT FOUND",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (0, 0, 255),
                4,
                cv2.LINE_AA,
            )
            rectified = np.full((594, 420, 3), 235, dtype=np.uint8)
            probability_bgr = np.full_like(rectified, 235)
            message = (
                f"A4定位失败：{exc}｜过暗 {dark_ratio:.2f}%｜过曝 {bright_ratio:.2f}%"
            )
        return CameraDiagnostics(full, located, rectified, probability_bgr, message)


class DebugCameraWorker:
    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
        settings: CameraSettings,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.settings = settings
        self._capture = None
        self._latest: np.ndarray | None = None
        self._error: str | None = None
        self._mutex = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera-debug-capture", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def latest(self) -> tuple[np.ndarray | None, str | None]:
        with self._mutex:
            return (None if self._latest is None else self._latest.copy(), self._error)

    def apply(self, settings: CameraSettings) -> dict[str, int]:
        settings.validate()
        with self._mutex:
            if self._capture is None:
                raise RuntimeError(self._error or "找不到camera")
            apply_camera_controls(
                self._capture,
                settings,
                device_path(self.camera_index),
            )
            self.settings = settings
        return read_camera_controls(device_path(self.camera_index))

    def _run(self) -> None:
        capture = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            with self._mutex:
                self._error = "找不到camera"
            return
        try:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            capture.set(cv2.CAP_PROP_FPS, 5)
            apply_camera_controls(capture, self.settings, device_path(self.camera_index))
            with self._mutex:
                self._capture = capture
            while not self._stop.is_set():
                with self._mutex:
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        self._error = "找不到camera"
                        break
                    self._latest = frame
                    self._error = None
        except Exception as exc:
            with self._mutex:
                self._error = str(exc)
        finally:
            with self._mutex:
                self._capture = None
            capture.release()


class CameraDebugGui:
    def __init__(
        self,
        root,
        camera_index: int,
        width: int,
        height: int,
        fisheye_path: Path,
        distance_path: Path,
        settings_path: Path,
    ) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.settings_path = settings_path
        self.settings = load_camera_settings(settings_path)
        self.device_lock = CameraDeviceLock()
        self.device_lock.acquire()
        self.worker = DebugCameraWorker(camera_index, width, height, self.settings)
        self.processor = DiagnosticProcessor(fisheye_path, distance_path)
        self._photo = None
        self._diagnostics: CameraDiagnostics | None = None
        self._actual_loaded = False
        self._auto_tuning = False

        root.title("2025 电赛 C 题相机环境调试")
        root.geometry("1180x720")
        root.minsize(960, 600)
        controls = ttk.Frame(root, padding=12, width=340)
        controls.pack(side=tk.LEFT, fill=tk.Y)
        controls.pack_propagate(False)
        preview = ttk.Frame(root, padding=(4, 8, 8, 8))
        preview.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        title = ttk.Frame(controls)
        title.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(title, text="相机环境调试", font=("sans", 18, "bold")).pack(side=tk.LEFT)
        help_canvas = tk.Canvas(title, width=34, height=34, highlightthickness=0, cursor="hand2")
        help_canvas.pack(side=tk.RIGHT)
        help_canvas.create_oval(4, 4, 30, 30, outline="#205493", width=2)
        help_canvas.create_text(17, 17, text="?", fill="#205493", font=("sans", 15, "bold"))
        help_canvas.bind("<Button-1>", lambda _event: self._show_help())

        settings_tabs = ttk.Notebook(controls, height=255)
        settings_tabs.pack(fill=tk.X)
        camera_panel = ttk.Frame(settings_tabs, padding=(4, 6))
        roi_panel = ttk.Frame(settings_tabs, padding=(4, 6))
        settings_tabs.add(camera_panel, text="相机参数")
        settings_tabs.add(roi_panel, text="ROI")

        self.auto_exposure = tk.BooleanVar(value=self.settings.auto_exposure)
        ttk.Checkbutton(
            camera_panel,
            text="自动曝光",
            variable=self.auto_exposure,
            command=self._sync_exposure_state,
        ).pack(anchor=tk.W, pady=(0, 8))

        self.variables: dict[str, object] = {}
        self.widgets: dict[str, tuple[object, object]] = {}
        specifications = (
            ("exposure_time_absolute", "曝光时间", 1, 2000),
            ("gain", "增益", 0, 63),
            ("brightness", "亮度", -128, 127),
            ("contrast", "对比度", 0, 64),
            ("gamma", "Gamma", 1, 500),
            ("sharpness", "锐度", 0, 25),
        )
        for name, label, minimum, maximum in specifications:
            variable = tk.IntVar(value=int(getattr(self.settings, name)))
            row = ttk.Frame(camera_panel)
            row.pack(fill=tk.X, pady=3)
            ttk.Label(row, text=label, width=9).pack(side=tk.LEFT)
            scale = ttk.Scale(
                row,
                from_=minimum,
                to=maximum,
                variable=variable,
                orient=tk.HORIZONTAL,
            )
            scale.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
            spin = ttk.Spinbox(row, from_=minimum, to=maximum, textvariable=variable, width=6)
            spin.pack(side=tk.RIGHT)
            self.variables[name] = variable
            self.widgets[name] = (scale, spin)

        self.roi_variables: dict[str, object] = {}
        for name, label in (
            ("roi_left", "左边界"),
            ("roi_right", "右边界"),
            ("roi_top", "上边界"),
            ("roi_bottom", "下边界"),
        ):
            variable = tk.DoubleVar(value=float(getattr(self.settings, name)) * 100.0)
            row = ttk.Frame(roi_panel)
            row.pack(fill=tk.X, pady=7)
            ttk.Label(row, text=label, width=9).pack(side=tk.LEFT)
            ttk.Scale(
                row,
                from_=0.0,
                to=100.0,
                variable=variable,
                orient=tk.HORIZONTAL,
            ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
            ttk.Spinbox(
                row,
                from_=0.0,
                to=100.0,
                increment=0.1,
                textvariable=variable,
                width=6,
                format="%.1f",
            ).pack(side=tk.RIGHT)
            self.roi_variables[name] = variable
        ttk.Label(
            roi_panel,
            text="四条边界均按完整画面的百分比保存；只要求左<右、上<下。",
            wraplength=290,
        ).pack(anchor=tk.W, pady=(8, 0))

        button_row = ttk.Frame(controls)
        button_row.pack(fill=tk.X, pady=(12, 4))
        ttk.Button(button_row, text="临时应用", command=self._apply).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3)
        )
        ttk.Button(button_row, text="保存为测量配置", command=self._save).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0)
        )
        ttk.Button(controls, text="恢复默认值", command=self._reset).pack(fill=tk.X, pady=4)
        self.auto_tune_button = ttk.Button(
            controls,
            text="自动调参",
            command=self._start_auto_tune,
        )
        self.auto_tune_button.pack(fill=tk.X, pady=(4, 2))
        self.auto_tune_progress = tk.DoubleVar(value=0.0)
        ttk.Progressbar(
            controls,
            variable=self.auto_tune_progress,
            maximum=100.0,
        ).pack(fill=tk.X, pady=(0, 2))
        self.auto_tune_status = tk.StringVar(value="自动调参：待启动")
        ttk.Label(
            controls,
            textvariable=self.auto_tune_status,
            wraplength=310,
            foreground="#555555",
        ).pack(anchor=tk.W, fill=tk.X)

        ttk.Label(controls, text="相机实际值", font=("sans", 12, "bold")).pack(
            anchor=tk.W, pady=(12, 4)
        )
        self.actual = tk.StringVar(value="正在读取……")
        ttk.Label(controls, textvariable=self.actual, wraplength=310).pack(anchor=tk.W, fill=tk.X)
        ttk.Label(controls, text="图像诊断", font=("sans", 12, "bold")).pack(
            anchor=tk.W, pady=(12, 4)
        )
        self.status = tk.StringVar(value="正在启动相机……")
        ttk.Label(controls, textvariable=self.status, wraplength=310).pack(anchor=tk.W, fill=tk.X)

        self.notebook = ttk.Notebook(preview)
        self.notebook.pack(expand=True, fill=tk.BOTH)
        self.labels = {}
        for name in (VIEW_FULL, VIEW_LOCATION, VIEW_RECTIFIED, VIEW_BINARY):
            frame = ttk.Frame(self.notebook)
            label = ttk.Label(frame, text="等待图像", anchor=tk.CENTER)
            label.pack(expand=True, fill=tk.BOTH)
            self.notebook.add(frame, text=name)
            self.labels[name] = label
        self.notebook.bind("<<NotebookTabChanged>>", lambda _event: self._render_selected())

        self._sync_exposure_state()
        self.worker.start()
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.after(300, self._tick)

    def _settings_from_ui(self) -> CameraSettings:
        settings = CameraSettings(
            auto_exposure=bool(self.auto_exposure.get()),
            power_line_frequency=1,
            **{
                name: int(round(variable.get()))
                for name, variable in self.variables.items()
            },
            **{
                name: float(variable.get()) / 100.0
                for name, variable in self.roi_variables.items()
            },
        )
        settings.validate()
        return settings

    def _sync_exposure_state(self) -> None:
        state = "disabled" if self.auto_exposure.get() else "normal"
        for widget in self.widgets.get("exposure_time_absolute", ()):
            widget.configure(state=state)

    def _apply(self) -> bool:
        from tkinter import messagebox

        try:
            settings = self._settings_from_ui()
            actual = self.worker.apply(settings)
        except Exception as exc:
            messagebox.showerror("参数应用失败", str(exc))
            return False
        self.settings = settings
        self.actual.set("｜".join(f"{name}={value}" for name, value in actual.items()))
        return True

    def _save(self) -> None:
        from tkinter import messagebox

        if not self._apply():
            return
        try:
            save_camera_settings(self._settings_from_ui(), self.settings_path)
        except Exception as exc:
            messagebox.showerror("配置保存失败", str(exc))
            return
        messagebox.showinfo("保存成功", "已保存。正式测量软件下次启动时使用这组相机参数。")

    def _reset(self) -> None:
        defaults = CameraSettings()
        self.auto_exposure.set(defaults.auto_exposure)
        for name, variable in self.variables.items():
            variable.set(getattr(defaults, name))
        for name, variable in self.roi_variables.items():
            variable.set(float(getattr(defaults, name)) * 100.0)
        self._sync_exposure_state()
        self._apply()

    def _set_ui_settings(self, settings: CameraSettings) -> None:
        self.auto_exposure.set(settings.auto_exposure)
        for name, variable in self.variables.items():
            variable.set(getattr(settings, name))
        for name, variable in self.roi_variables.items():
            variable.set(float(getattr(settings, name)) * 100.0)
        self._sync_exposure_state()

    def _set_auto_tune_progress(self, percent: float, message: str) -> None:
        self.auto_tune_progress.set(percent)
        self.auto_tune_status.set(message)

    def _finish_auto_tune(
        self,
        best: AutoTuneEvaluation | None,
        evaluations: list[AutoTuneEvaluation],
        error: str | None = None,
    ) -> None:
        self._auto_tuning = False
        self.auto_tune_button.configure(state="normal")
        if error is not None:
            self._set_auto_tune_progress(0.0, f"自动调参失败：{error}")
            return
        if best is None:
            self._set_auto_tune_progress(0.0, "自动调参失败：没有可用候选")
            return
        self._set_ui_settings(best.settings)
        self.settings = best.settings
        try:
            actual = self.worker.apply(best.settings)
            save_camera_settings(best.settings, self.settings_path)
        except Exception as exc:
            self._set_auto_tune_progress(100.0, f"最佳参数保存失败：{exc}")
            return
        self.actual.set("｜".join(f"{name}={value}" for name, value in actual.items()))
        detail = "；".join(
            f"锐度{item.settings.sharpness}: {item.metrics.score:.0f}"
            for item in evaluations
            if item.metrics.ok
        )
        self._set_auto_tune_progress(
            100.0,
            f"自动调参完成：锐度 {best.settings.sharpness}｜{best.metrics.message}｜{detail}",
        )

    def _start_auto_tune(self) -> None:
        if self._auto_tuning:
            return
        self._auto_tuning = True
        self.auto_tune_button.configure(state="disabled")
        self._set_auto_tune_progress(0.0, "自动调参：准备候选参数")
        threading.Thread(target=self._run_auto_tune, name="camera-auto-tune", daemon=True).start()

    def _run_auto_tune(self) -> None:
        try:
            base = self._settings_from_ui()
            candidates = auto_tune_candidates(base)
            evaluations: list[AutoTuneEvaluation] = []
            total = max(1, len(candidates))
            for index, settings in enumerate(candidates, 1):
                prefix = f"自动调参：{index}/{total} 锐度 {settings.sharpness}"
                self.root.after(
                    0,
                    self._set_auto_tune_progress,
                    (index - 1) / total * 100.0,
                    f"{prefix}｜正在应用参数",
                )
                self.worker.apply(settings)
                self.root.after(
                    0,
                    self._set_auto_tune_progress,
                    (index - 0.7) / total * 100.0,
                    f"{prefix}｜等待相机稳定",
                )
                time.sleep(1.6)
                frame, error = self.worker.latest()
                if error is not None:
                    metrics = AutoTuneMetrics(
                        ok=False,
                        score=-1.0,
                        message=f"相机错误：{error}",
                    )
                elif frame is None:
                    metrics = AutoTuneMetrics(
                        ok=False,
                        score=-1.0,
                        message="相机暂时没有画面",
                    )
                else:
                    self.root.after(
                        0,
                        self._set_auto_tune_progress,
                        (index - 0.35) / total * 100.0,
                        f"{prefix}｜计算算法健康度",
                    )
                    metrics = evaluate_camera_health(frame, settings, self.processor.application)
                evaluation = AutoTuneEvaluation(settings, metrics)
                evaluations.append(evaluation)
                self.root.after(
                    0,
                    self._set_auto_tune_progress,
                    index / total * 100.0,
                    f"{prefix}｜{metrics.message}",
                )
            usable = [item for item in evaluations if item.metrics.ok]
            best = max(usable, key=lambda item: item.metrics.score) if usable else None
            self.root.after(0, self._finish_auto_tune, best, evaluations, None)
        except Exception as exc:
            self.root.after(0, self._finish_auto_tune, None, [], str(exc))

    def _tick(self) -> None:
        frame, error = self.worker.latest()
        if error:
            self.status.set(error)
        elif frame is not None:
            try:
                if not self._actual_loaded:
                    actual = read_camera_controls(device_path(self.worker.camera_index))
                    self.actual.set(
                        "｜".join(f"{name}={value}" for name, value in actual.items())
                    )
                    self._actual_loaded = True
                self._diagnostics = self.processor.process(frame, self._settings_from_ui())
                self.status.set(self._diagnostics.message)
                self._render_selected()
            except Exception as exc:
                self.status.set(f"诊断失败：{exc}")
        self.root.after(500, self._tick)

    def _render_selected(self) -> None:
        if self._diagnostics is None:
            return
        selected = self.notebook.tab(self.notebook.select(), "text")
        label = self.labels[selected]
        image = {
            VIEW_FULL: self._diagnostics.full,
            VIEW_LOCATION: self._diagnostics.located,
            VIEW_RECTIFIED: self._diagnostics.rectified,
            VIEW_BINARY: self._diagnostics.probability,
        }[selected]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rendered = Image.fromarray(rgb)
        # 预览区会随屏幕分辨率及窗口大小变化。固定使用 810×650 会让
        # 1024×600 的树莓派屏幕直接裁掉图片；按标签实际尺寸缩放才能
        # 保证完整画面和竖版 A4 都全部可见。
        viewport_width = label.winfo_width()
        viewport_height = label.winfo_height()
        if viewport_width <= 2 or viewport_height <= 2:
            return
        rendered.thumbnail(
            (max(1, viewport_width - 4), max(1, viewport_height - 4)),
            Image.Resampling.BILINEAR,
        )
        from PIL import ImageTk

        self._photo = ImageTk.PhotoImage(rendered)
        label.configure(image=self._photo, text="")

    def _show_help(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        window = tk.Toplevel(self.root)
        window.title("相机环境调试使用说明")
        window.geometry("720x560")
        text = tk.Text(window, wrap=tk.WORD, padx=14, pady=14, font=("sans", 11))
        scrollbar = ttk.Scrollbar(window, command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        text.insert("1.0", HELP_TEXT)
        text.configure(state=tk.DISABLED)
        text.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _close(self) -> None:
        self.worker.stop()
        self.device_lock.release()
        self.root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description="树莓派相机环境调试 GUI")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fisheye-calibration", type=Path, default=DEFAULT_FISHEYE)
    parser.add_argument("--distance-calibration", type=Path, default=DEFAULT_DISTANCE)
    parser.add_argument("--camera-settings", type=Path, default=DEFAULT_CAMERA_SETTINGS_PATH)
    args = parser.parse_args()
    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError as exc:
        raise SystemExit("缺少 Tkinter，请安装系统包 python3-tk") from exc
    root = tk.Tk()
    try:
        CameraDebugGui(
            root,
            args.camera_index,
            args.width,
            args.height,
            args.fisheye_calibration,
            args.distance_calibration,
            args.camera_settings,
        )
    except Exception as exc:
        messagebox.showerror("相机调试启动失败", str(exc))
        root.destroy()
        return 1
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
