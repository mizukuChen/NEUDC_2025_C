from __future__ import annotations

from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
CAPTURE_DIR = ROOT / "runtime" / "captures"


def main() -> int:
    print("====================================")
    print("OpenCV is installed.")
    print("OpenCV version:", cv2.__version__)
    print("====================================")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open camera index 0.")
        print("Check the USB/CSI camera connection and Raspberry Pi camera settings.")
        return 1

    try:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: camera opened, but no frame was captured.")
            return 2

        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        output_path = CAPTURE_DIR / "rpi_test_shot.jpg"
        cv2.imwrite(str(output_path), frame)
        print(f"Camera test passed. Saved: {output_path}")
        return 0
    finally:
        cap.release()


if __name__ == "__main__":
    raise SystemExit(main())
