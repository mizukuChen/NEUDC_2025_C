# 2025 电赛 C 题：基于单目视觉的目标物测量装置

本项目实现 2025 年全国大学生电子设计竞赛 C 题的视觉测量部分，运行平台为树莓派与 USB 摄像头。

当前主线已经覆盖：

- 鱼眼标定、去畸变与可调矩形 ROI；
- A4 黑框定位、透视校正、距离估计及线性修正；
- 圆形、等边三角形、单个或多个分离正方形测量；
- 实心重叠多正方形的标准图和低分辨率多帧识别；
- HOG + 线性 SVM 数字识别；
- 普通识别与指定数字识别两套任务；
- 发挥（4）绕竖直轴倾斜物面的检测；
- 正式触摸 GUI、静默预识别缓存和独立相机调试 GUI。

## 快速开始

### WSL 开发环境

```bash
cd ~/NUEDC_2025_C
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

运行完整测试：

```bash
.venv/bin/python -m unittest discover -v
```

### 树莓派环境

树莓派需要 Python、OpenCV、NumPy、Pillow 和 Tkinter。Python 依赖见 `requirements-rpi.txt`；Tkinter 由系统包管理器安装：

```bash
sudo apt install python3-tk v4l-utils
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-rpi.txt
```

正式测量 GUI：

```bash
scripts/start_rpi_gui.sh
```

相机环境调试 GUI：

```bash
scripts/start_camera_debug.sh
```

两个程序通过 `/tmp/nuedc-camera.lock` 互斥占用摄像头，不能同时运行。启动日志分别写入 `runtime/logs/gui.log` 和 `runtime/logs/camera_debug.log`。

## 正式测量流程

正式 GUI 提供两种模式：

1. 普通识别：识别圆形、三角形、单/多正方形和重叠正方形；正方形中的数字只作为内部纹理，不改变几何类型。
2. 数字识别：输入一个画面中确实存在的 `0～9` 数字，寻找相同数字的正方形并输出其边长。

相机主流程：

```text
1920×1080 YUYV 多帧
  → 鱼眼去畸变
  → 现场矩形 ROI
  → A4 黑框定位与距离估计
  → 420×594 A4 工作图
  → 完整轮廓检测
  → 必要时启动低分辨率重叠重建
  → 普通结果或数字匹配结果
  → display 生成中文状态与标注图
  → ui 显示
```

GUI 默认启用静默预识别。后台先生成与模式无关的几何缓存，场景稳定且目标均为正方形时再补充数字缓存；切换普通/数字模式或修改目标数字不会重复定位 A4 和重建几何。按钮测量优先于后台任务。

## 相机现场调试

调试 GUI 提供四幅图：

- 去畸变完整画面：蓝色矩形表示实际搜索 ROI；
- A4 定位：绿色四边形应贴合黑框；
- 拉直后的 A4：检查目标边缘和数字笔画；
- 黑色概率图：检查白纸、黑框和目标之间的分离程度。

ROI 可独立调整上、下、左、右四条边界，唯一几何约束是它必须构成非空矩形。曝光、增益、亮度、对比度、Gamma、锐度和防频闪参数可临时应用，也可保存到：

```text
config/camera_runtime.json
```

正式测量程序下次启动时自动读取该配置。

## 目录结构

```text
src/
├── application.py          测量编排和场景分析
├── silent_recognition.py   静默识别与两级缓存
├── hardware/               摄像头配置、设备锁和采集线程
├── display/                中文摘要、目标量和诊断可视化
├── ui/                     两套 Tkinter 触摸界面
└── vision/                 标定、A4定位、几何与数字识别

scripts/
├── measurement/            离线单图测量
├── evaluation/             批量准确率与误差评估
├── calibration/            鱼眼和距离标定
├── training/               数字模型训练
├── debug/                  算法中间过程调试
└── targets/                靶纸与扰动数据生成

tests/
├── application/
├── display/
├── hardware/
├── targets/
└── vision/
```

`assets/` 保存模型、标定图案和标准靶图；`runtime/` 保存现场配置、日志和可再生成的调试输出；`archive/` 保存已经退出主线但需要历史复现的程序与照片。

## 常用离线命令

测量已经拉直的 A4 图像：

```bash
python scripts/measurement/run_shape_measure.py <A4图像>
python scripts/measurement/run_numbered_measure.py <A4图像> <数字>
```

运行核心评估：

```bash
python scripts/evaluation/evaluate_shape_measure.py
python scripts/evaluation/evaluate_controlled_overlap.py
python scripts/evaluation/evaluate_measurement_tasks.py
python scripts/evaluation/evaluate_lowres_overlap.py
```

生成标准靶图和 A3 打印 PDF：

```bash
python scripts/targets/generate_single_shapes.py
python scripts/targets/generate_single_squares.py
python scripts/targets/generate_separate_squares.py
python scripts/targets/generate_overlap_squares.py
python scripts/targets/generate_numbered_squares.py
python scripts/targets/generate_numbered_overlap_squares.py
python scripts/targets/generate_print_selection_a3.py
```

各脚本的输入、输出和用途见 `scripts/README.md`。

## 文档

- [工程架构](docs/design/architecture.md)
- [测量思路](docs/design/measurement_notes.md)
- [C 题原题](docs/problem/C题_基于单目视觉的目标物测量装置.pdf)
