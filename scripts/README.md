# scripts 目录说明

本目录存放标定、评估、单次运行、调试以及测试图生成工具。正式视觉算法位于 `src/vision/`，本目录中的程序主要负责调用算法、准备数据和输出报告。

## 当前测量与评估主线

### `run_shape_measure.py`

读取一张已经完成 A4 透视校正的图像，调用 `src/vision/measure_shapes.py` 检测圆形、三角形和完整轮廓正方形，并把标注结果写入 `runtime/debug/`。

该脚本不负责鱼眼去畸变和 A4 定位。

### `evaluate_shape_measure.py`

在生成的基础图、单正方形和分离多正方形标准图上批量评估 `measure_shapes.py`，检查形状类型、数量和尺寸误差。

### `run_distance_measure.py`

对单张实拍图执行鱼眼去畸变、中央竖直区域裁剪、A4 平面定位、PnP 光轴距离估计和线性距离修正，并输出调试图。

### `evaluate_distance.py`

批量读取文件名中包含真实距离的标定照片，评估距离修正后的误差，并输出 CSV 报告。

## 重叠正方形主线

### `targets/generate_overlap_squares.py`

生成受控的实心重叠多正方形 PNG，同时保存同名 JSON ground truth。生成前会检查每个正方形每条边被其他正方形覆盖的区间并集，确保单边覆盖率严格小于 50%。

### `evaluate_controlled_overlap.py`

调用 `src/vision/measure_overlap.py` 检测所有受控重叠图，并与 JSON ground truth 一一匹配，比较：

- 正方形数量；
- 边长误差；
- 中心位置误差；
- 角度误差。

### `debug_overlap_lines.py`

生成重叠检测的六阶段调试图，包括：

1. A4 输入图；
2. Otsu 黑色前景；
3. Canny 可见边缘；
4. 筛选后的 HoughLinesP 直线；
5. 四边正方形候选；
6. 去重后的最终结果。

调试输出属于可再生成文件，统一写入 `runtime/debug/`。

## 标定工具

### `calibrate_fisheye.py`

使用棋盘照片进行 OpenCV 鱼眼模型标定，默认输出 `runtime/calibration/fisheye_1280x720.json`。

旧标定照片已经归档到 `archive/runtime/captures/eye_distortion/`，需要重新使用旧照片时应显式传入该目录。

### `calibrate_distance.py`

根据不同真实距离下拍摄的 A4 目标照片，拟合 PnP 原始距离到真实距离的鲁棒线性修正，默认输出 `runtime/calibration/distance_linear.json` 和 CSV 报告。

旧距离照片位于 `archive/runtime/captures/rpi_distance/`。

### `calibration/generate_chessboard_a3.py`

生成鱼眼标定使用的 A3 棋盘 PDF。

### `undistort_fisheye.py`

读取一张图像和鱼眼标定 JSON，输出去畸变后的单张图像。

### `undistort_fisheye_batch.py`

批量去畸变指定目录中的图片，适合检查标定效果。

## 标准目标生成与打印

### `targets/generate_basic_targets.py`

生成基础圆形、三角形和正方形的 A4 PDF 与 PNG。

### `targets/generate_advanced_targets.py`

生成单正方形、分离多正方形和编号正方形的 A4 PDF 与 PNG。旧版重叠正方形预设已经移除，新重叠图统一由 `generate_overlap_squares.py` 生成。

### `targets/generate_targets_a3.py`

把基础及高级 A4 目标排版到 A3 页面中央并生成 PDF。该脚本目前仍包含旧版 overlap 预设，使用前应更新或将其归档。

### `targets/merge_target_pdfs.py`

把多张 A4 目标 PDF 合并为一个便于打印的多页 PDF。当前列表仍引用旧 overlap PDF，使用前应更新。

## 已归档脚本

旧版基础测量脚本已经移动到：

```text
archive/scripts/basic_measure/
```

包括：

- `evaluate_basic_measure.py`：旧版基础形状批量评估；
- `evaluate_basic_measure_fisheye.py`：旧版鱼眼校正后批量评估；
- `evaluate_basic_measure_fisheye_auto.py`：带中心裁剪回退的实验版本；
- `evaluate_basic_measure_fisheye_center_strip.py`：中央 1/3 竖直区域实验版本；
- `run_basic_measure.py`：旧版单图基础测量入口；
- `run_basic_measure_fisheye.py`：旧版带鱼眼去畸变的单图入口。

它们依赖旧模块 `src/vision/measure_basic.py`，仅用于历史复现，不属于当前主线。

旧树莓派预览脚本位于：

```text
archive/src/
```

旧采集照片位于：

```text
archive/runtime/captures/
```

## 已删除的旧工具

- `request_rpi_capture.py`：向旧预览服务写入拍摄请求；
- `process_rpi_mount.py`：从旧树莓派挂载目录复制并处理照片；
- `evaluate_overlap_measure.py`：评估不符合新单边覆盖约束的旧 overlap 图。
