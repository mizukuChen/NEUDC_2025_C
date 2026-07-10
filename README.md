# 2025 电赛 C 题：基于单目视觉的目标物测量装置

本工程用于在树莓派上实现 2025 年全国大学生电子设计竞赛 C 题“基于单目视觉的目标物测量装置”。

目标功能包括：

- 使用单个摄像头测量目标物距离 `D`。
- 测量目标物平面上几何图形的边长或直径 `x`。
- 支持基本题的圆形、等边三角形、正方形识别。
- 后续扩展发挥题的多个正方形、重叠正方形、带编号正方形和斜放目标物测量。
- 接入按键、显示屏、电流采样和功耗计算。

## 目录结构

- `src/`：树莓派端运行代码。
- `src/vision/`：图像处理、相机标定、姿态估计和目标尺寸测量模块。
- `src/hardware/`：GPIO 按键、电流采样和功耗测量相关代码。
- `src/display/`：屏幕显示适配代码。
- `config/`：运行配置示例。
- `scripts/targets/`：A4/A3 调试靶标生成脚本。
- `assets/targets/generated/`：已生成的 PDF/PNG 调试靶标。
- `docs/problem/`：竞赛原题 PDF。
- `docs/design/`：工程设计说明和测量思路笔记。
- `runtime/`：树莓派运行时产生的照片、标定文件和日志。
- `tests/`：单元测试和图像处理回归测试。

## 开发机使用

建议在开发机上使用独立 conda 环境，避免污染默认 Python 环境：

```powershell
conda create -n opencv-nuedc python=3.12 -y
conda activate opencv-nuedc
uv pip install -r requirements-dev.txt
```

也可以不用激活环境，直接指定环境里的 Python：

```powershell
uv pip install --python D:\work_office\resource\anaconda\envs\opencv-nuedc\python.exe -r requirements-dev.txt
```

安装依赖后，可以重新生成测试靶标：

```powershell
python scripts/targets/generate_basic_targets.py
python scripts/targets/generate_advanced_targets.py
python scripts/targets/merge_target_pdfs.py
python scripts/targets/generate_targets_a3.py
python scripts/calibration/generate_chessboard_a3.py
```

生成后的合并版靶标在：

```text
assets/targets/generated/pdf/all_targets_A4_18pages.pdf
assets/targets/generated/a3/pdf/all_targets_A3_18pages.pdf
assets/calibration/fisheye_chessboard_A3_8x12_29mm.pdf
```

## 树莓派使用

在树莓派上安装运行依赖：

```bash
python3 -m pip install -r requirements-rpi.txt
```

先测试摄像头是否能正常打开并抓图：

```bash
python3 src/camera_test.py
```

测试照片默认保存到：

```text
runtime/captures/rpi_test_shot.jpg
```

也可以在树莓派项目目录中直接采集一张图到 `runtime/captures/rpi/`：

```bash
python3 src/capture.py --name d100_square_01.jpg
```

如果树莓派目录已经通过 WSL 挂载到 Windows：

```text
\\wsl.localhost\Ubuntu\home\mizuku\nailong
```

可以在 Windows 开发机上导入并处理树莓派拍摄的图像：

```powershell
conda activate opencv-nuedc
python scripts/process_rpi_mount.py --latest
```

默认会从挂载目录下的 `runtime/captures/rpi/` 寻找图片，复制到本地 `runtime/captures/rpi/`，然后生成识别结果和 `runtime/debug/` 调试图。

## 离线图像识别

准备好手机或树莓派拍摄的目标图后，可以先在开发机上运行基础题离线识别：

```powershell
python scripts/run_basic_measure.py runtime/captures/phone/basic_square_01.jpg
```

脚本会输出图形类型、尺寸 `x`，并在 `runtime/debug/` 下保存调试图：

```text
01_plane_corners.jpg   黑框四角点检测结果
02_rectified.jpg       透视矫正后的标准 A4 图
03_measurement.jpg     内部图形轮廓和尺寸结果
```

也可以批量评估一个目录下的基础题图片：

```powershell
python scripts/evaluate_basic_measure.py runtime/captures/phone
```

批量脚本会根据文件名中的 `circle`、`square`、`triangle` 判断期望类别，并检查尺寸是否落在默认 `110 mm` 到 `130 mm` 范围内。

## 实现路线

1. 完成摄像头采集和相机标定。
2. 检测 A4 目标纸的黑色边框和底边中点标记。
3. 使用单目视觉方法估计目标物距离 `D`。
4. 对目标物平面做透视矫正。
5. 在矫正后的纸面图像上测量图形尺寸 `x`。
6. 集成按键、显示、电流采样和功耗计算。

更详细的交接规划见：

```text
docs/design/handoff_plan.md
```
