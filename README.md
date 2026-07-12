# 2025 电赛 C 题：基于单目视觉的目标物测量装置

本仓库用于实现 2025 年全国大学生电子设计竞赛 C 题。当前已完成 USB 摄像头采集、树莓派网页/HDMI 预览、A4 黑框定位、平面透视矫正，以及圆形、正方形、等边三角形的基础尺寸测量原型。

当前主开发目录为 WSL：

```bash
~/NUEDC_2025_C
```

树莓派通过 SSHFS 挂载到 WSL 的：

```bash
~/nailong
```

它对应树莓派的 `/home/nailong`。该目录中的 `pi_cv` 是树莓派环境，不能在 WSL 中激活或重建。

## 目录说明

- `src/`：树莓派采集、预览和视觉算法。
- `src/vision/`：A4 黑框定位、透视矫正、基础图形测量。
- `scripts/`：离线测量、批量评估、数据导入、靶纸生成脚本。
- `config/`：配置示例。
- `assets/`：生成的靶纸和标定图案。
- `docs/`：题目、设计说明和交接记录。
- `runtime/`：运行时照片、调试图、标定文件。该目录多数内容被 Git 忽略。
- `tests/`：回归测试。

## WSL 开发环境

首次进入 WSL 工程目录后：

```bash
cd ~/NUEDC_2025_C
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

当前迁移目录中的 `.venv` 已安装 Python 3.12 和 OpenCV；日常使用只需：

```bash
cd ~/NUEDC_2025_C
source .venv/bin/activate
```

确认环境：

```bash
python --version
python -c "import cv2; print(cv2.__version__)"
```

## 树莓派环境与部署

树莓派当前地址为 `192.168.112.67`，用户名为 `nailong`，USB 摄像头使用 `/dev/video0`。

进入树莓派终端：

```bash
ssh nailong@192.168.112.67
source ~/pi_cv/bin/activate
cd ~
```

旧版树莓派预览脚本已归档在 `archive/src/`。树莓派端如需继续使用，脚本放在家目录 `~`，即：

```text
~/camera_preview.py
~/capture.py
```

修改采集脚本后，在 WSL 同步到树莓派：

```bash
cp ~/NUEDC_2025_C/archive/src/camera_preview.py ~/nailong/camera_preview.py
cp ~/NUEDC_2025_C/src/capture.py ~/nailong/capture.py
```

不要在 WSL 中执行 `source ~/nailong/pi_cv/bin/activate`，因为它会把树莓派虚拟环境的符号链接解析为 WSL 本机路径。

## 摄像头程序

### 网页预览与电脑拍摄

这是当前推荐的采集方式。树莓派端启动：

```bash
source ~/pi_cv/bin/activate
cd ~
python camera_preview.py --preview-mode web --port 8080
```

在电脑浏览器打开：

```text
http://192.168.112.67:8080
```

网页显示实时预览，点击 `Capture` 会将当前原始帧自动以时间戳命名，保存到：

```text
~/runtime/captures/rpi/
```

若热点不允许直接访问端口，在电脑或 WSL 中建立 SSH 隧道：

```bash
ssh -N -L 8080:127.0.0.1:8080 nailong@192.168.112.67
```

然后浏览器打开 `http://localhost:8080`。

### 预览模式

`camera_preview.py` 只打开一次摄像头，可用 `--preview-mode` 选择输出方式：

```bash
# 树莓派 HDMI 屏幕预览
DISPLAY=:0 XAUTHORITY=$HOME/.Xauthority python camera_preview.py --preview-mode hdmi --fullscreen

# 电脑浏览器预览和拍摄
python camera_preview.py --preview-mode web --port 8080

# HDMI 与电脑浏览器同时预览
DISPLAY=:0 XAUTHORITY=$HOME/.Xauthority python camera_preview.py --preview-mode both --fullscreen --port 8080

# 无预览，仅保留文件触发拍摄
python camera_preview.py --preview-mode none
```

运行预览服务时，不要同时运行 `capture.py` 或另一份预览服务，否则会争用 `/dev/video0`。在 SSH 终端按 `Ctrl+C` 停止服务。

### 单张拍摄

不启动预览服务时，可直接拍一张：

```bash
source ~/pi_cv/bin/activate
cd ~
python capture.py --name sample.jpg
```

### 旧的文件触发方式

当预览服务正在运行时，也会监听：

```text
~/runtime/capture_request.txt
```

在 WSL 中写入空请求即可自动命名拍摄：

```bash
: > ~/nailong/runtime/capture_request.txt
```

原 Windows 侧请求封装已经删除；如继续使用旧预览服务，可直接写入上述请求文件。

## 图像处理程序

以下命令在 WSL 工程根目录、且已激活 `.venv` 时执行。

### 单图形状测量

```bash
python scripts/run_shape_measure.py <透视校正后的A4图像>
```

该入口检测圆形、三角形和完整轮廓正方形，输入应当是已经完成 A4 透视校正的图像。旧版完整处理入口已归档到 `archive/scripts/basic_measure/`。

调试图默认写入：

```text
runtime/debug/shape_measurement.jpg
```

### 批量评估

```bash
python scripts/evaluate_shape_measure.py
python scripts/evaluate_controlled_overlap.py
```

前者评估基础和分离图形，后者使用 JSON ground truth 评估受控重叠正方形。脚本的完整用途见 `scripts/README.md`。

## 靶纸生成程序

在 WSL 开发环境执行：

```bash
python scripts/targets/generate_basic_targets.py
python scripts/targets/generate_advanced_targets.py
python scripts/targets/merge_target_pdfs.py
python scripts/targets/generate_targets_a3.py
python scripts/calibration/generate_chessboard_a3.py
```

生成物位于 `assets/targets/generated/` 和 `assets/calibration/`。

## 当前算法状态

基础题原型的流程为：

```text
原图 -> A4 黑框定位 -> 透视矫正 -> 内部黑色图形分割 -> 类别与尺寸计算
```

已验证部分手机和树莓派样本可正确识别圆形、正方形、等边三角形，120 mm 目标的测量值约为 119 到 122 mm。

USB 摄像头存在明显鱼眼畸变，当前用户决定改用新的方法处理该问题。因此后续修改黑框定位前，应先确定并实现畸变校正方案。地砖照片已用于评估直线约束方案，详细现状见 [当前交接记录](docs/design/current_handoff.md)。

## 交接资料

- [当前交接记录](docs/design/current_handoff.md)
- [总体架构](docs/design/architecture.md)
- [测量思路](docs/design/measurement_notes.md)
- [历史交接规划](docs/design/handoff_plan.md)
