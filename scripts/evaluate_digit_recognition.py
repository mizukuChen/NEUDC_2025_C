from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.recognize_digits import HogSvmDigitRecognizer


DEFAULT_INPUT = ROOT / "assets" / "targets" / "generated" / "numbered_controlled"
DEFAULT_MODEL = ROOT / "assets" / "models" / "digit_hog_linear_svm.npz"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate independent HOG + SVM digit recognition.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--debug-dir", type=Path, default=ROOT / "runtime" / "debug" / "digits")
    args = parser.parse_args()

    recognizer = HogSvmDigitRecognizer.load(args.model)
    confusion = np.zeros((10, 10), dtype=np.int32)
    confidences: list[float] = []
    total = 0
    correct = 0
    args.debug_dir.mkdir(parents=True, exist_ok=True)

    for metadata_path in sorted(args.input.glob("numbered_controlled_*_A4.json")):
        image_path = metadata_path.with_suffix(".png")
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"FAIL {image_path.name}: cannot read image")
            return 2
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        scale_x = image.shape[1] / float(payload["page_mm"][0])
        scale_y = image.shape[0] / float(payload["page_mm"][1])
        scale = (scale_x + scale_y) / 2.0
        case_correct = 0
        debug = image.copy()

        for index, square in enumerate(payload["squares"]):
            truth = int(square["digit"])
            result = recognizer.recognize_square(
                image,
                (float(square["cx_mm"]) * scale_x, float(square["cy_mm"]) * scale_y),
                float(square["size_mm"]) * scale,
                float(square["angle_deg"]),
            )
            prediction = result.digit
            confusion[truth, prediction] += 1
            total += 1
            case_correct += int(prediction == truth)
            correct += int(prediction == truth)
            confidences.append(result.confidence)
            cv2.imwrite(str(args.debug_dir / f"{metadata_path.stem}_{index}_truth{truth}_pred{prediction}.png"), result.patch)
            point = (round(float(square["cx_mm"]) * scale_x), round(float(square["cy_mm"]) * scale_y))
            color = (0, 160, 0) if prediction == truth else (0, 0, 255)
            cv2.putText(debug, f"{prediction} ({result.confidence:.2f})", point, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
        cv2.imwrite(str(args.debug_dir / f"{metadata_path.stem}_result.jpg"), debug)
        print(f"{'PASS' if case_correct == len(payload['squares']) else 'FAIL'} {image_path.name} accuracy={case_correct}/{len(payload['squares'])}")

    accuracy = correct / total if total else 0.0
    print("confusion_matrix rows=true columns=predicted")
    print("    " + " ".join(f"{digit:2d}" for digit in range(10)))
    for digit, row in enumerate(confusion):
        print(f"{digit:2d}: " + " ".join(f"{value:2d}" for value in row))
    print(f"accuracy: {correct}/{total} = {accuracy:.4f}")
    print(f"confidence_mean: {float(np.mean(confidences)) if confidences else 0.0:.4f}")
    print(f"confidence_min: {float(np.min(confidences)) if confidences else 0.0:.4f}")
    return 0 if correct == total and total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
