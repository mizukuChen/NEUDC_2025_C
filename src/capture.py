from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2


SCRIPT_PATH = Path(__file__).resolve()
# 自动向上溯源，找到项目的物理根目录
ROOT = SCRIPT_PATH.parents[1] if SCRIPT_PATH.parent.name == "src" else SCRIPT_PATH.parent
# 默认拍摄照片的输出保存目录
DEFAULT_OUTPUT_DIR = ROOT / "runtime" / "captures" / "rpi"


def parse_args() -> argparse.Namespace:
    """
    配置并解析拍照命令行参数。
    """
    parser = argparse.ArgumentParser(description="Capture a single frame from the camera.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV 摄像头设备索引，默认 /dev/video0。")
    parser.add_argument("--width", type=int, default=1280, help="设定的图像像素宽度（默认 1280）。")
    parser.add_argument("--height", type=int, default=720, help="设定的图像像素高度（默认 720）。")
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="保存的文件名。如果为空，则使用当前时间戳自动命名。格式为 .jpg 格式。",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出存储目录。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 规范化输出文件名，若没有指定名称，按时间戳自动命名为 rpi_年月日_时分秒_毫秒
    name = args.name.strip()
    if not name:
        name = datetime.now().strftime("rpi_%Y%m%d_%H%M%S_%f")
    if not name.lower().endswith((".jpg", ".jpeg", ".png")):
        name += ".jpg"
    output_path = args.output_dir / Path(name).name

    # 1. 初始化相机通道
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera index {args.camera_index}", file=sys.stderr)
        return 1

    # 设置拍摄的分辨率尺寸
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # 2. 预热相机：丢弃前 5 帧图像
    # 原因：摄像头刚上电启动时，传感器的自动曝光（AE）和自动白平衡（AWB）需要一定帧数来稳定，直接抓拍前两帧可能过黑或过亮。
    for _ in range(5):
        cap.grab()

    # 3. 读取并捕获一帧高质量图像
    ok, frame = cap.read()
    cap.release() # 立即释放对摄像头硬件资源的占用，防止设备争用

    if not ok:
        print("ERROR: camera opened, but no frame was captured.", file=sys.stderr)
        return 2

    # 4. 写入存储文件
    if not cv2.imwrite(str(output_path), frame):
        print(f"ERROR: failed to save image to: {output_path}", file=sys.stderr)
        return 3

    print(f"saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
