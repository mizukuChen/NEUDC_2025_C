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

在开发机上安装调试依赖，并重新生成测试靶标：

```powershell
python -m pip install -r requirements-dev.txt
python scripts/targets/generate_basic_targets.py
python scripts/targets/generate_advanced_targets.py
python scripts/targets/merge_target_pdfs.py
python scripts/targets/generate_targets_a3.py
```

生成后的合并版靶标在：

```text
assets/targets/generated/pdf/all_targets_A4_18pages.pdf
assets/targets/generated/a3/pdf/all_targets_A3_18pages.pdf
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
