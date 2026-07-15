from __future__ import annotations

import argparse
import threading
from pathlib import Path

import cv2
from PIL import Image

from src.application import DEFAULT_DISTANCE, DEFAULT_FISHEYE, MeasurementApplication
from src.silent_recognition import SilentRecognitionManager


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
        self._pending_views: dict[str, object] = {}
        self._busy = False
        self.silent_recognition = SilentRecognitionManager(
            application,
            status_callback=lambda message: self.root.after(
                0,
                self._set_silent_status,
                message,
            ),
        )

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
        self.digit.trace_add("write", lambda *_args: self._sync_silent_request())
        self.entry = ttk.Entry(
            row,
            textvariable=self.digit,
            width=4,
            justify=tk.CENTER,
            font=("sans", 18),
        )
        self.entry.pack(side=tk.LEFT)

        digit_pad = ttk.Frame(controls)
        digit_pad.pack(fill=tk.X, pady=(2, 8))
        self.digit_buttons = []
        for digit in range(10):
            button = ttk.Button(
                digit_pad,
                text=str(digit),
                width=3,
                command=lambda value=digit: self._select_digit(value),
            )
            button.grid(row=digit // 5, column=digit % 5, padx=2, pady=2, sticky="ew")
            digit_pad.columnconfigure(digit % 5, weight=1)
            self.digit_buttons.append(button)

        self.camera_button = ttk.Button(
            controls, text="相机测量", command=self.measure_camera
        )
        self.camera_button.pack(fill=tk.X, pady=(12, 4), ipady=7)
        self.file_button = ttk.Button(
            controls, text="选择图片测试", command=self.measure_file
        )
        self.file_button.pack(fill=tk.X, pady=4, ipady=5)

        self.silent_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls,
            text="静默预识别（默认开启）",
            variable=self.silent_enabled,
            command=self._toggle_silent_recognition,
        ).pack(anchor=tk.W, pady=(10, 2))
        self.silent_status = tk.StringVar(value="静默识别：正在启动")
        ttk.Label(
            controls,
            textvariable=self.silent_status,
            wraplength=280,
            foreground="#555555",
        ).pack(anchor=tk.W, fill=tk.X, pady=(0, 2))

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
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self._set_status("请选择识别模式，然后开始测量。")
        self._sync_mode()
        self.application.start_camera(camera_index, width, height)
        self.silent_recognition.start()
        root.protocol("WM_DELETE_WINDOW", self._close)

    def _set_status(self, message: str) -> None:
        import tkinter as tk

        self.status.configure(state=tk.NORMAL)
        self.status.delete("1.0", tk.END)
        self.status.insert("1.0", message)
        self.status.configure(state=tk.DISABLED)

    def _sync_mode(self) -> None:
        state = "normal" if self.mode.get() == "numbered" else "disabled"
        self.entry.configure(state=state)
        for button in self.digit_buttons:
            button.configure(state=state)
        if state == "normal":
            self.entry.focus_set()
            self.entry.selection_range(0, "end")
        self._sync_silent_request()

    def _select_digit(self, digit: int) -> None:
        self.digit.set(str(digit))
        self.entry.focus_set()
        self.entry.selection_range(0, "end")
        self._sync_silent_request()

    def _sync_silent_request(self) -> None:
        if not hasattr(self, "silent_recognition"):
            return
        mode = self.mode.get()
        value = self.digit.get().strip()
        if mode == "numbered" and (len(value) != 1 or value not in "0123456789"):
            return
        self.silent_recognition.set_request(
            mode,
            int(value) if mode == "numbered" else None,
        )

    def _toggle_silent_recognition(self) -> None:
        enabled = bool(self.silent_enabled.get())
        self.silent_recognition.set_enabled(enabled)
        if enabled:
            self._sync_silent_request()

    def _set_silent_status(self, message: str) -> None:
        self.silent_status.set(message)

    def _clear_views(self) -> None:
        self._photos.clear()
        self._pending_views.clear()
        for label in self.image_labels.values():
            label.configure(image="", text="等待图像")

    def _show_image(self, title: str, image_bgr) -> None:
        from PIL import ImageTk

        if image_bgr.ndim == 2:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((680, 500), Image.Resampling.BILINEAR)
        photo = ImageTk.PhotoImage(image)
        self._photos[title] = photo
        self.image_labels[title].configure(image=photo, text="")

    def _start(self, loader, camera_input: bool = False) -> None:
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
        self._set_status("正在读取相机缓存……" if camera_input else "正在获取输入图像……")

        def worker() -> None:
            try:
                if camera_input:
                    result = self.silent_recognition.measure(mode, digit)
                else:
                    loaded = loader()
                    self.root.after(0, self._input_ready, loaded)
                    result = self.silent_recognition.run_exclusive(
                        lambda: self.application.process_raw(loaded, mode, digit)
                    )
            except Exception as exc:
                self.root.after(0, self._error, str(exc))
            else:
                self.root.after(0, self._success, result)

        threading.Thread(target=worker, daemon=True).start()

    def _input_ready(self, raw) -> None:
        self._pending_views[VIEW_INPUT] = raw
        self._set_status("已获取输入图像，正在定位 A4 并识别目标……")

    def measure_camera(self) -> None:
        self._start(
            None,
            camera_input=True,
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
        self._pending_views.update(
            {
                VIEW_INPUT: result.input_image,
                VIEW_LOCATION: result.located_a4,
                VIEW_RECTIFIED: result.rectified_a4,
            }
        )
        self._show_image(VIEW_RESULT, result.annotated_a4)
        self.notebook.select(self.tab_frames[VIEW_RESULT])
        self._set_status(result.summary)

    def _on_tab_changed(self, _event=None) -> None:
        selected = self.notebook.select()
        if not selected:
            return
        title = self.notebook.tab(selected, "text")
        image = self._pending_views.pop(title, None)
        if image is not None and title not in self._photos:
            self._show_image(title, image)

    def _close(self) -> None:
        self.silent_recognition.stop()
        self.application.stop_camera()
        self.root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description="树莓派视觉测量 GUI")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fisheye-calibration", type=Path, default=DEFAULT_FISHEYE)
    parser.add_argument("--distance-calibration", type=Path, default=DEFAULT_DISTANCE)
    parser.add_argument(
        "--strip-fraction",
        type=float,
        default=None,
        help="兼容旧版：临时覆盖现场ROI为居中的指定宽度比例",
    )
    args = parser.parse_args()
    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError as exc:
        raise SystemExit("缺少 Tkinter，请安装系统包 python3-tk") from exc
    root = tk.Tk()
    try:
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
    except Exception as exc:
        messagebox.showerror("测量软件启动失败", str(exc))
        root.destroy()
        return 1
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
