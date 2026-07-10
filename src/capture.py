from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1] if SCRIPT_PATH.parent.name == "src" else SCRIPT_PATH.parent
DEFAULT_OUTPUT_DIR = ROOT / "runtime" / "captures" / "rpi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one image from a Raspberry Pi camera.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height.")
    parser.add_argument("--warmup-frames", type=int, default=8, help="Frames to discard before capture.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", type=str, default=None, help="Output file name. Defaults to timestamp.")
    return parser.parse_args()


def capture_image(
    camera_index: int,
    width: int,
    height: int,
    warmup_frames: int,
) -> tuple[bool, object]:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return False, f"cannot open camera index {camera_index}"

    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        frame = None
        ok = False
        for _ in range(max(1, warmup_frames)):
            ok, frame = cap.read()
        if not ok or frame is None:
            return False, "camera opened, but no frame was captured"
        return True, frame
    finally:
        cap.release()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    file_name = args.name
    if file_name is None:
        file_name = datetime.now().strftime("rpi_%Y%m%d_%H%M%S.jpg")
    if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
        file_name += ".jpg"

    ok, result = capture_image(args.camera_index, args.width, args.height, args.warmup_frames)
    if not ok:
        print(f"ERROR: {result}")
        return 1

    output_path = output_dir / file_name
    cv2.imwrite(str(output_path), result)
    print(f"saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
