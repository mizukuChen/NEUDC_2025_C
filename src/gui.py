from __future__ import annotations
import argparse,threading
from pathlib import Path
import cv2
from PIL import Image
from src.application import DEFAULT_DISTANCE,DEFAULT_FISHEYE,MeasurementApplication

class MeasurementGui:
    def __init__(self,root,application,camera_index:int,width:int,height:int):
        import tkinter as tk
        from tkinter import ttk
        self.root=root;self.application=application;self.camera_index=camera_index;self.width=width;self.height=height;self._photo=None;self._busy=False
        root.title("2025 电赛 C 题视觉测量");root.geometry("1024x600");root.minsize(800,480)
        controls=ttk.Frame(root,padding=16);controls.pack(side=tk.LEFT,fill=tk.Y);preview=ttk.Frame(root,padding=10);preview.pack(side=tk.RIGHT,expand=True,fill=tk.BOTH)
        ttk.Label(controls,text="测量模式",font=("sans",18,"bold")).pack(anchor=tk.W,pady=(0,10))
        self.mode=tk.StringVar(value="normal")
        ttk.Radiobutton(controls,text="普通识别",value="normal",variable=self.mode,command=self._sync).pack(anchor=tk.W,pady=6)
        ttk.Radiobutton(controls,text="数字识别",value="numbered",variable=self.mode,command=self._sync).pack(anchor=tk.W,pady=6)
        row=ttk.Frame(controls);row.pack(fill=tk.X,pady=(18,8));ttk.Label(row,text="目标数字：").pack(side=tk.LEFT);self.digit=tk.StringVar(value="0")
        self.entry=ttk.Entry(row,textvariable=self.digit,width=4,justify=tk.CENTER,font=("sans",18));self.entry.pack(side=tk.LEFT)
        self.camera_button=ttk.Button(controls,text="相机测量",command=self.measure_camera);self.camera_button.pack(fill=tk.X,pady=(18,6),ipady=8)
        self.file_button=ttk.Button(controls,text="选择图片测试",command=self.measure_file);self.file_button.pack(fill=tk.X,pady=6,ipady=5)
        self.status=tk.StringVar(value="请选择模式，然后开始测量");ttk.Label(controls,textvariable=self.status,wraplength=280,justify=tk.LEFT,font=("sans",13)).pack(anchor=tk.W,pady=(22,0))
        self.image_label=ttk.Label(preview,anchor=tk.CENTER);self.image_label.pack(expand=True,fill=tk.BOTH);self._sync()
    def _sync(self):self.entry.configure(state="normal" if self.mode.get()=="numbered" else "disabled")
    def _start(self,loader):
        from tkinter import messagebox
        if self._busy:return
        mode=self.mode.get();value=self.digit.get().strip()
        if mode=="numbered" and (len(value)!=1 or value not in "0123456789"):messagebox.showerror("输入错误","请输入一个 0～9 的数字");return
        digit=int(value) if mode=="numbered" else None;self._busy=True;self.camera_button.configure(state="disabled");self.file_button.configure(state="disabled");self.status.set("正在采集和识别，请稍候……")
        def worker():
            try:result=self.application.process_raw(loader(),mode,digit)
            except Exception as exc:self.root.after(0,self._error,str(exc))
            else:self.root.after(0,self._success,result)
        threading.Thread(target=worker,daemon=True).start()
    def measure_camera(self):self._start(lambda:self.application.capture_frame(self.camera_index,self.width,self.height))
    def measure_file(self):
        from tkinter import filedialog
        path=filedialog.askopenfilename(filetypes=(("图像","*.png *.jpg *.jpeg *.bmp"),("全部文件","*")))
        if path:
            def load():
                image=cv2.imread(path)
                if image is None:raise ValueError(f"无法读取图片：{path}")
                return image
            self._start(load)
    def _idle(self):self._busy=False;self.camera_button.configure(state="normal");self.file_button.configure(state="normal")
    def _error(self,message):
        from tkinter import messagebox
        self._idle();self.status.set("测量失败，请检查目标和相机");messagebox.showerror("测量失败",message)
    def _success(self,result):
        from PIL import ImageTk
        self._idle();self.status.set(result.summary+f"\n重投影误差：{result.reprojection_error_px:.2f} px");rgb=cv2.cvtColor(result.annotated_a4,cv2.COLOR_BGR2RGB);image=Image.fromarray(rgb);image.thumbnail((650,560),Image.Resampling.LANCZOS);self._photo=ImageTk.PhotoImage(image);self.image_label.configure(image=self._photo)

def main()->int:
    p=argparse.ArgumentParser(description="树莓派视觉测量 GUI");p.add_argument("--camera-index",type=int,default=0);p.add_argument("--width",type=int,default=1280);p.add_argument("--height",type=int,default=720);p.add_argument("--fisheye-calibration",type=Path,default=DEFAULT_FISHEYE);p.add_argument("--distance-calibration",type=Path,default=DEFAULT_DISTANCE);p.add_argument("--strip-fraction",type=float,default=1/3);a=p.parse_args()
    try:import tkinter as tk
    except ImportError as exc:raise SystemExit("缺少 Tkinter，请安装系统包 python3-tk") from exc
    root=tk.Tk();MeasurementGui(root,MeasurementApplication(a.fisheye_calibration,a.distance_calibration,a.strip_fraction),a.camera_index,a.width,a.height);root.mainloop();return 0
if __name__=="__main__":raise SystemExit(main())
