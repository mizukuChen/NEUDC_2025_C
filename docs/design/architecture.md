# 工程架构

## 依赖方向

```text
vision ─────┐
hardware ───┼─→ application ─→ display ─→ ui ─→ main
            └─→ silent_recognition ────────┘
```

依赖只允许向右：视觉算法和硬件层不导入显示或界面；`display` 不依赖 Tkinter；`ui` 负责触摸交互和把展示结果放入窗口。

## 源码层次

### `src/vision`

纯视觉算法，包括鱼眼标定、距离估计、A4 平面定位、透视校正、完整轮廓测量、重叠正方形重建和数字识别。

重叠识别位于 `src/vision/overlap/`：

- `geometry.py`：两套引擎共享的边支持、外圈概率、IoU 和角度计算；
- `models.py`：标准图与低分辨率结果模型；
- `standard.py`：高分辨率单帧标准图引擎；
- `lowres.py`：低分辨率多帧融合引擎。

两套引擎共享数学能力，但保留各自的线段来源、阈值、候选生成和严格验证策略。

### `src/hardware`

- `camera_settings.py`：现场配置、V4L2 参数读写和跨进程设备锁；
- `camera_stream.py`：YUYV 配置、单次采集和正式测量常驻帧缓冲；
- `debug_camera.py`：调试软件使用的动态参数采集线程。

### `src/application.py`

编排去畸变、ROI、A4 定位、距离估计、几何识别和数字识别，返回不含 Tkinter 和中文排版逻辑的测量数据。

### `src/silent_recognition.py`

比较轻量场景签名，维护几何缓存和数字缓存。后台任务不得阻塞按钮测量；场景丢失或变化时使旧缓存失效。

### `src/display`

把应用结果转换为中文识别状态、目标量文本、A4 描边、目标标注图和相机诊断图。输出是 NumPy 图像和普通数据类，不创建窗口。

### `src/ui`

- `measurement_gui.py`：正式普通/数字识别触摸界面；
- `camera_debug_gui.py`：ROI、曝光与图像诊断界面。

### `src/main.py`

稳定的正式启动入口。树莓派始终通过 `python -m src.main` 或 `scripts/start_rpi_gui.sh` 启动。

## 数据目录

- `assets/targets/generated/`：标准生成靶图及真值；
- `assets/models/`：树莓派运行所需的 NumPy 数字模型；
- `runtime/calibration/`：当前鱼眼与距离修正参数；
- `runtime/logs/`：GUI 日志；
- `runtime/debug/`、`runtime/reports/`：可删除并重新生成；
- `archive/`：退出主线的历史程序、模型和采集照片。

## 测试与评估

`tests/` 镜像主要源码层次，用单元测试保护接口和核心算法。`scripts/evaluation/` 使用生成图和 JSON 真值执行批量准确率评估。结构重构必须同时通过两者。
