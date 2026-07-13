from __future__ import annotations

import argparse
import threading
from pathlib import Path

import cv2
from PIL import Image

from src.application import DEFAULT_DISTANCE, DEFAULT_FISHEYE, MeasurementApplication


VIEW_INPUT = "输入图像"
VIEW_LOCATION = "A4 定位"
VIEW_RECTIFIED = "校正后的 A4"
VIEW_RESULT = "识别结果"


class MeasurementGui:
    def __init__(
        self,
        root,
        application: MeasurementApplication,
        camera_index: int,
        width: int,
        height: int,
    ) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.application = application
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._photos: dict[str, object] = {}
        self._busy = False

        root.title("2025 电赛 C 题视觉测量")
        root.geometry("1024x600")
        root.minsize(800, 480)

        controls = ttk.Frame(root, padding=12, width=310)
        controls.pack(side=tk.LEFT, fill=tk.Y)
        controls.pack_propagate(False)
        preview = ttk.Frame(root, padding=(4, 8, 8, 8))
        preview.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        ttk.Label(controls, text="测量模式", font=("sans", 18, "bold")).pack(
            anchor=tk.W, pady=(0, 8)
        )
        self.mode = tk.StringVar(value="normal")
        ttk.Radiobutton(
            controls,
            text="普通识别",
            value="normal",
            variable=self.mode,
            command=self._sync_mode,
        ).pack(anchor=tk.W, pady=4)
        ttk.Radiobutton(
            controls,
            text="数字识别",
            value="numbered",
            variable=self.mode,
            command=self._sync_mode,
        ).pack(anchor=tk.W, pady=4)

        row = ttk.Frame(controls)
        row.pack(fill=tk.X, pady=(12, 6))
        ttk.Label(row, text="目标数字：").pack(side=tk.LEFT)
        self.digit = tk.StringVar(value="0")
        self.entry = ttk.Entry(
            row,
            textvariable=self.digit,
            width=4,
            justify=tk.CENTER,
            font=("sans", 18),
        )
        self.entry.pack(side=tk.LEFT)

        self.camera_button = ttk.Button(
            controls, text="相机测量", command=self.measure_camera
        )
        self.camera_button.pack(fill=tk.X, pady=(12, 4), ipady=7)
        self.file_button = ttk.Button(
            controls, text="选择图片测试", command=self.measure_file
        )
        self.file_button.pack(fill=tk.X, pady=4, ipady=5)

        ttk.Label(controls, text="识别状态", font=("sans", 13, "bold")).pack(
            anchor=tk.W, pady=(12, 4)
        )
        status_frame = ttk.Frame(controls)
        status_frame.pack(expand=True, fill=tk.BOTH)
        self.status = tk.Text(
            status_frame,
            wrap=tk.WORD,
            width=30,
            height=12,
            font=("sans", 11),
            relief=tk.FLAT,
            padx=6,
            pady=6,
        )
        scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.status.yview)
        self.status.configure(yscrollcommand=scrollbar.set)
        self.status.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.notebook = ttk.Notebook(preview)
        self.notebook.pack(expand=True, fill=tk.BOTH)
        self.image_labels: dict[str, object] = {}
        self.tab_frames: dict[str, object] = {}
        for title in (VIEW_INPUT, VIEW_LOCATION, VIEW_RECTIFIED, VIEW_RESULT):
            frame = ttk.Frame(self.notebook)
            label = ttk.Label(frame, text="等待图像", anchor=tk.CENTER)
            label.pack(expand=True, fill=tk.BOTH)
            self.notebook.add(frame, text=title)
            self.tab_frames[title] = frame
            self.image_labels[title] = label

        self._set_status("请选择识别模式，然后开始测量。")
        self._sync_mode()

    def _set_status(self, message: str) -> None:
        import tkinter as tk

        self.status.configure(state=tk.NORMAL)
        self.status.delete("1.0", tk.END)
        self.status.insert("1.0", message)
        self.status.configure(state=tk.DISABLED)

    def _sync_mode(self) -> None:
        self.entry.configure(state="normal" if self.mode.get() == "numbered" else "disabled")

    def _clear_views(self) -> None:
        self._photos.clear()
        for label in self.image_labels.values():
            label.configure(image="", text="等待图像")

    def _show_image(self, title: str, image_bgr) -> None:
        from PIL import ImageTk

        if image_bgr.ndim == 2:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((680, 500), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self._photos[title] = photo
        self.image_labels[title].configure(image=photo, text="")

    def _start(self, loader) -> None:
        from tkinter import messagebox

        if self._busy:
            return
        mode = self.mode.get()
        value = self.digit.get().strip()
        if mode == "numbered" and (len(value) != 1 or value not in "0123456789"):
            messagebox.showerror("输入错误", "请输入一个 0～9 的数字")
            return

        digit = int(value) if mode == "numbered" else None
        self._busy = True
        self.camera_button.configure(state="disabled")
        self.file_button.configure(state="disabled")
        self._clear_views()
        self._set_status("正在获取输入图像……")

        def worker() -> None:
            try:
                raw = loader()
                self.root.after(0, self._input_ready, raw)
                result = self.application.process_raw(raw, mode, digit)
            except Exception as exc:
                self.root.after(0, self._error, str(exc))
            else:
                self.root.after(0, self._success, result)

        threading.Thread(target=worker, daemon=True).start()

    def _input_ready(self, raw) -> None:
        self._show_image(VIEW_INPUT, raw)
        self.notebook.select(self.tab_frames[VIEW_INPUT])
        self._set_status("已获取输入图像，正在定位 A4 并识别目标……")

    def measure_camera(self) -> None:
        self._start(
            lambda: self.application.capture_frame(
                self.camera_index, self.width, self.height
            )
        )

    def measure_file(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            filetypes=(("图像", "*.png *.jpg *.jpeg *.bmp"), ("全部文件", "*"))
        )
        if not path:
            return

        def load():
            image = cv2.imread(path)
            if image is None:
                raise ValueError(f"无法读取图片：{path}")
            return image

        self._start(load)

    def _idle(self) -> None:
        self._busy = False
        self.camera_button.configure(state="normal")
        self.file_button.configure(state="normal")

    def _error(self, message: str) -> None:
        from tkinter import messagebox

        self._idle()
        if message == "找不到camera":
            self._set_status("找不到camera")
            messagebox.showerror("相机错误", "找不到camera")
            return
        self._set_status(f"识别失败\n原因：{message}")
        messagebox.showerror("识别失败", message)

    def _success(self, result) -> None:
        self._idle()
        self._show_image(VIEW_INPUT, result.input_image)
        self._show_image(VIEW_LOCATION, result.located_a4)
        self._show_image(VIEW_RECTIFIED, result.rectified_a4)
        self._show_image(VIEW_RESULT, result.annotated_a4)
        self.notebook.select(self.tab_frames[VIEW_RESULT])
        self._set_status(result.summary)


def main() -> int:
    parser = argparse.ArgumentParser(description="树莓派视觉测量 GUI")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fisheye-calibration", type=Path, default=DEFAULT_FISHEYE)
    parser.add_argument("--distance-calibration", type=Path, default=DEFAULT_DISTANCE)
    parser.add_argument("--strip-fraction", type=float, default=1 / 3)
    args = parser.parse_args()
    try:
        import tkinter as tk
    except ImportError as exc:
        raise SystemExit("缺少 Tkinter，请安装系统包 python3-tk") from exc
    root = tk.Tk()
    MeasurementGui(
        root,
        MeasurementApplication(
            args.fisheye_calibration,
            args.distance_calibration,
            args.strip_fraction,
        ),
        args.camera_index,
        args.width,
        args.height,
    )
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
