# scripts 目录说明

本目录存放标定、评估、单次运行、调试以及测试图生成工具。正式视觉算法位于 `src/vision/`，本目录中的程序主要负责调用算法、准备数据和输出报告。

目录按开发流程分类：

- `measurement/`：离线处理单张图像，不是树莓派正式入口；
- `evaluation/`：批量读取标准数据并统计准确率与误差；
- `calibration/`：鱼眼、距离标定与去畸变检查；
- `training/`：数字识别模型训练；
- `debug/`：算法中间过程可视化；
- `targets/`：标准靶纸、扰动数据与打印文件生成。

正式测量和相机调试的两个 Shell 启动脚本保留在本目录根部。

树莓派正式 GUI 入口为 `python -m src.main`。界面支持普通识别、手动输入 0～9 的数字识别、相机抓拍测量和选择原始图片测试，并通过页签展示输入图像、A4 定位、校正后的 A4 和标出全部目标的识别结果；完整应用 pipeline 位于 `src/application.py`。GUI 启动后会常驻打开 1920×1080 YUYV 相机并维护最新三帧，鱼眼映射按分辨率缓存，只在现场配置的矩形ROI中定位一次A4，随后三张原始帧直接映射为 420×594 A4 工作图。ROI的四条边界由相机调试软件独立调整并保存；ROI内定位失败时不会自动扩大。默认开启的静默预识别调度器位于 `src/silent_recognition.py`：它每 0.5 秒检查 A4 位置与纸内二值内容，三次稳定后生成一份与界面模式无关的场景分析，缓存全部几何，并在全部目标为正方形时一次性识别所有数字。普通/数字模式和预选数字只负责从这份缓存生成展示，不触发重新检测；一次漏检立即禁止使用旧缓存、连续两次漏检清空缓存，按钮命中缓存时仍用最新帧复核。关闭开关或缓存无效时继续使用原按键流程。输入页仍可查看完整相机画面，其他页签在用户切换时再生成预览。图片测试保留单图高分辨率流程。若系统没有 Tkinter，需要通过系统包管理器安装 `python3-tk`，它不是 pip 依赖。

树莓派相机环境调试入口为 `python -m src.ui.camera_debug_gui`，正式启动脚本为 `start_camera_debug.sh`。它是与测量GUI分离的软件，提供去畸变完整画面、A4定位、拉直A4、黑色概率图、V4L2参数实际读回、图像灰度诊断以及内置圆形 `?` 中文帮助。左侧ROI页签可独立调整上下左右四条边界，蓝色矩形实时显示算法实际搜索范围。临时应用不会写文件，“保存为测量配置”会把相机参数和ROI共同更新到 `config/camera_runtime.json`，正式测量下次启动自动读取。`src/hardware/camera_settings.py` 负责两个软件共用的配置校验、V4L2读写和 `/tmp/nuedc-camera.lock` 互斥锁，因此调试与测量不能同时打开摄像头。

普通识别同时覆盖发挥（4）：常规 A4 定位失败后，程序会用较小的形态学闭运算和更宽的透视宽高比范围搜索强侧视黑框，并通过更高黑框评分及 PnP 重投影误差验证。由姿态估计得到的物面—轴线夹角落在题目范围附近时，界面标记为“发挥（4）倾斜物面”，只输出正方形边长 `x`，不要求输出距离 `D`。

## 当前测量与评估主线

### `measurement/run_shape_measure.py`

读取一张已经完成 A4 透视校正的图像，调用统一入口 `src/vision/measure.py`。统一入口根据完整轮廓对黑色前景的解释率，自动选择：

- `measure_shapes.py`：圆形、三角形和完整轮廓正方形；
- `overlap/standard.py`：高分辨率标准图中的实心重叠多正方形。
- `overlap/lowres.py`：相机低分辨率、多帧融合的重叠多正方形。

终端输出中的 `source` 会标明结果来自 `contour` 或 `overlap_lines`，标注图写入 `runtime/debug/`。

该脚本不负责鱼眼去畸变和 A4 定位。

该脚本对应普通检测任务：内部白色数字只视为方块纹理；单目标输出该目标尺寸，多正方形输出最小面积正方形的边长。

### `measurement/run_numbered_measure.py`

发挥（3）的开发入口。第二个位置参数模拟 IO 输入数字，程序复用普通几何检测，识别所有正方形编号，找到指定数字并输出对应正方形边长。实际按键、串口或触摸屏输入适配尚未决定。

### `evaluation/evaluate_shape_measure.py`

在生成的基础图、单正方形和分离多正方形标准图上批量评估 `measure_shapes.py`，检查形状类型、数量和尺寸误差。

### `measurement/run_distance_measure.py`

对单张实拍图执行鱼眼去畸变、中央竖直区域裁剪、A4 平面定位、PnP 光轴距离估计和线性距离修正，并输出调试图。

### `evaluation/evaluate_distance.py`

批量读取文件名中包含真实距离的标定照片，评估距离修正后的误差，并输出 CSV 报告。

## 重叠正方形主线

### `targets/generate_overlap_squares.py`

调用 `presets/overlap.py`，使用公共正方形渲染器生成受控重叠多正方形 PNG/JSON；生成前检查每条边被其他正方形覆盖的区间并集，确保单边覆盖率严格小于 50%。

### `evaluation/evaluate_controlled_overlap.py`

调用 `src/vision/overlap/standard.py` 检测所有受控重叠图，并与 JSON ground truth 一一匹配，比较：

- 正方形数量；
- 边长误差；
- 中心位置误差；
- 角度误差。

### `targets/generate_lowres_overlap_dataset.py`

从无数字和带数字的重叠预设出发，先按指定的相机原生 A4 像素宽度进行光学模糊和传感器采样，再恢复到算法使用的 420×594 工作网格。每组默认生成三帧，并记录采样相位、静态失焦、黑白电平、弱光照梯度和弱噪声参数。默认数据写入 `runtime/debug/lowres_overlap_dataset/`，不会进入 Git。

### `evaluation/evaluate_lowres_overlap.py`

在上述低分辨率三帧数据集上评估独立的 `overlap/lowres.py`：统计方块数量、尺寸、中心和角度误差，并在几何数量正确时继续检查带数字方块的 HOG+SVM 识别。失败样本标注图写入 `runtime/debug/lowres_overlap_eval/`。

低分辨率重叠检测采用“宽松生成、严格验证”：分别从每一帧和三帧融合图提取 LSD 线段，用联合拟合残差桥接同一物理边两端的短线段，并在共同法向量下计算平行边距；候选合并后，再回到融合图检查四边支持、至少两条强边、边界内侧黑色概率、多帧来源和 `0.70` 总置信度。融合图和三张单帧的 LSD 候选提取并行执行，但候选仍按固定来源顺序合并，阈值与最终选择逻辑不变。算法仍保存在独立模块中，但已由正式相机测量入口按“完整轮廓优先、重叠重建兜底”的方式调用。

### `debug/debug_overlap_lines.py`

生成重叠检测的六阶段调试图，包括：

1. A4 输入图；
2. Otsu 黑色前景；
3. Canny 可见边缘；
4. 筛选后的 HoughLinesP 直线；
5. 四边正方形候选；
6. 去重后的最终结果。

调试输出属于可再生成文件，统一写入 `runtime/debug/`。

## 独立数字识别模块

### `targets/generate_numbered_squares.py`

调用 `presets/numbered.py`，在公共正方形组件上使用 Pillow 和 `DejaVuSans-Bold.ttf` 添加白色数字，并保存 PNG/JSON。字号固定为方块像素边长的 52%，同尺寸的 0～9 使用完全相同字号；数字与方块作为整体在 `[-45°, 45°]` 内旋转。

### `targets/generate_numbered_overlap_squares.py`

调用 `presets/numbered_overlap.py` 生成带数字的受控重叠方块。除单边覆盖率必须小于 50% 外，还要求每个方块参与真实重叠、数字唯一，并保证以数字中心为圆心、方块边长 30% 为半径的安全区不被其他方块覆盖。

### `targets/generate_paired_numbered_overlap.py`

围绕编号重叠预设随机改变中心、尺寸和角度，并为同一几何分别输出无数字图与带数字图。`evaluation/evaluate_paired_numbered_overlap.py` 成对比较几何重建结果，再检查每个输入数字是否选择了正确方块，用于区分重叠布局失败、数字干扰几何和纯数字识别失败。

### `training/train_digit_svm.py`

生成带轻微旋转、平移、模糊和笔画变化的数字训练图，调用独立 HOG 子模块提取特征，并训练 scikit-learn `LinearSVC`。训练完成后仅把权重、偏置、类别和 HOG 类中心导出为 NumPy NPZ；运行时不依赖 scikit-learn。

### `archive/scripts/convert_digit_svm.py`（已归档）

把 `archive/assets/models/legacy_joblib/` 中的旧 scikit-learn Joblib 数字模型无损转换为纯 NumPy 线性 SVM 模型，并在生成样本上比较两者的决策分数和预测结果。该脚本只用于开发和迁移，不参与树莓派运行。

### `evaluation/evaluate_digit_recognition.py`

使用编号方块 JSON 中的真实几何参数裁剪和转正方块，独立评估 HOG+SVM 的数字准确率、置信度和混淆矩阵，并把数字小图写入 `runtime/debug/digits/`。

### `evaluation/evaluate_measurement_tasks.py`

对两套任务入口进行生成图端到端评估。普通模式只把 PNG 交给 `measure_normal()`，检查基础图形、分离方块、重叠方块及带数字方块的最终尺寸；数字模式仅把保证存在的目标数字交给 `measure_numbered()`，检查匹配方块及其尺寸。JSON ground truth 只在检测完成后用于评分，不参与检测。

数字识别算法仍由独立的 `recognize_digits.py` 和 `hog.py` 实现，任务层通过 `measurement_tasks.py` 将其与公共几何检测组合。

## 标定工具

### `calibration/calibrate_fisheye.py`

使用棋盘照片进行 OpenCV 鱼眼模型标定，默认输出 `runtime/calibration/fisheye_1280x720.json`。

旧标定照片已经归档到 `archive/runtime/captures/eye_distortion/`，需要重新使用旧照片时应显式传入该目录。

### `calibration/calibrate_distance.py`

根据不同真实距离下拍摄的 A4 目标照片，拟合 PnP 原始距离到真实距离的鲁棒线性修正，默认输出 `runtime/calibration/distance_linear.json` 和 CSV 报告。

旧距离照片位于 `archive/runtime/captures/rpi_distance/`。

### `calibration/generate_chessboard_a3.py`

生成鱼眼标定使用的 A3 棋盘 PDF。

### `calibration/undistort_fisheye.py`

读取一张图像和鱼眼标定 JSON，输出去畸变后的单张图像。

### `calibration/undistort_fisheye_batch.py`

批量去畸变指定目录中的图片，适合检查标定效果。

## 标准目标生成与打印

PNG/JSON 测试图生成采用三层结构：

```text
targets/generation/    页面、几何、数字、渲染、约束和输出能力
        ↑
targets/presets/       basic、separate、overlap、numbered 上层布局
        ↑
targets/generate_*.py  薄命令行入口
```

`generation/` 不包含具体案例，`presets/` 不包含绘制实现。数字是正方形的可选装饰，不另写一套正方形生成逻辑。

### `targets/generate_augmented_dataset.py`

读取现有标准 PNG/JSON，为已经完成 A4 透视校正的图像生成可复现的轻微扰动版本。独立的 `targets/augmentation/` 子模块包含残余透视、静态失焦、弱光照梯度和可选的极弱传感器噪声；默认关闭噪声，不模拟运动模糊、JPEG 压缩或打印误差。每张输出 JSON 会保留原真值，并记录随机种子、实际采样参数和源图片路径。

例如，为编号标准图各生成 5 个版本：

```bash
python scripts/targets/generate_augmented_dataset.py \
  assets/targets/generated/numbered_controlled --variants 5
```

生成后可用 `evaluation/evaluate_augmented_measurements.py <扰动目录>` 对普通任务和编号任务进行端到端评分。

### `evaluation/evaluate_augmentation_limits.py`

逐项关闭其他扰动，分别扫描高斯模糊、光照梯度和残余透视的强度。报告基础图形、分离方块、重叠方块、带数字方块普通测量以及数字匹配的通过率、最大尺寸误差和异常数；可用 `--csv` 保存明细。该工具只评估现有算法，不自动调整算法参数。

### `targets/generate_single_shapes.py`

调用 `presets/basic.py` 生成单圆形、单三角形和单正方形 PNG/JSON。

### `targets/generate_separate_squares.py`

调用 `presets/separate.py` 生成多个彼此分离的正方形 PNG/JSON。

### `targets/generate_single_squares.py` / `targets/generate_single_numbered_square.py`

分别生成不同尺寸/角度的单正方形，以及用于打印检查的正向单编号方块；两者都复用公共方块与数字渲染模块。

### `targets/generate_print_selection_a3.py`

从当前模块化生成系统的标准 A4 PNG 中选取少量代表性目标，以真实 210×297 mm 尺寸居中嵌入竖版 A3（297×420 mm），输出一个多页打印 PDF 和对应页码 JSON 清单。A4 四周因此分别保留 43.5 mm 与 61.5 mm 的 A3 余量，避免打印机不可打印边缘裁掉 A4 黑框。

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

它们依赖归档模块 `archive/src/vision/measure_basic.py`，仅用于历史复现，不属于当前主线。

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
