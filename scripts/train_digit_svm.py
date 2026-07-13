from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sklearn
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.targets.generate_numbered_squares import FONT_HEIGHT_RATIO, FONT_PATH
from src.vision.recognize_digits import PATCH_SIZE, SQUARE_INNER_FRACTION, hog_features
from src.vision.linear_svm import LinearSvmModel, save_linear_svm


DEFAULT_MODEL = ROOT / "assets" / "models" / "digit_hog_linear_svm.npz"
DEFAULT_CONFIG = ROOT / "assets" / "models" / "digit_hog_linear_svm.json"


def canonical_patch(digit: int, canvas_size: int = 96) -> np.ndarray:
    font_size = round(canvas_size * FONT_HEIGHT_RATIO)
    font = ImageFont.truetype(str(FONT_PATH), font_size)
    image = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(image)
    draw.text((canvas_size / 2.0, canvas_size / 2.0), str(digit), fill=255, font=font, anchor="mm")
    inner = round(canvas_size * SQUARE_INNER_FRACTION)
    left = (canvas_size - inner) // 2
    cropped = np.asarray(image.crop((left, left, left + inner, left + inner)))
    return cv2.resize(cropped, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)


def augment(base: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle = float(rng.uniform(-3.0, 3.0))
    scale = float(rng.uniform(0.94, 1.06))
    tx = float(rng.uniform(-3.0, 3.0))
    ty = float(rng.uniform(-3.0, 3.0))
    matrix = cv2.getRotationMatrix2D((PATCH_SIZE / 2.0, PATCH_SIZE / 2.0), angle, scale)
    matrix[:, 2] += (tx, ty)
    patch = cv2.warpAffine(base, matrix, (PATCH_SIZE, PATCH_SIZE), flags=cv2.INTER_LINEAR, borderValue=0)
    if rng.random() < 0.35:
        patch = cv2.GaussianBlur(patch, (3, 3), float(rng.uniform(0.3, 0.9)))
    if rng.random() < 0.20:
        kernel = np.ones((2, 2), dtype=np.uint8)
        patch = cv2.dilate(patch, kernel) if rng.random() < 0.5 else cv2.erode(patch, kernel)
    _, patch = cv2.threshold(patch, 127, 255, cv2.THRESH_BINARY)
    return patch


def build_dataset(samples_per_digit: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features: list[np.ndarray] = []
    labels: list[int] = []
    for digit in range(10):
        base = canonical_patch(digit)
        for sample_index in range(samples_per_digit):
            patch = base if sample_index == 0 else augment(base, rng)
            features.append(hog_features(patch))
            labels.append(digit)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an OpenCV HOG + linear SVM digit classifier.")
    parser.add_argument("--samples-per-digit", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20250712)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    features, labels = build_dataset(args.samples_per_digit, args.seed)
    svm = LinearSVC(C=2.5, dual="auto", max_iter=10000, random_state=args.seed)
    svm.fit(features, labels)

    centroids = np.stack([features[labels == digit].mean(axis=0) for digit in range(10)])
    for path in (args.model, args.config):
        path.parent.mkdir(parents=True, exist_ok=True)
    model = LinearSvmModel(svm.coef_, svm.intercept_, svm.classes_)
    numpy_predictions = np.asarray(model.predict(features))
    sklearn_predictions = svm.predict(features)
    if not np.array_equal(numpy_predictions, sklearn_predictions):
        raise RuntimeError("NumPy export does not match scikit-learn predictions")
    save_linear_svm(args.model, model, centroids=centroids)
    args.config.write_text(
        json.dumps(
            {
                "font": FONT_PATH.name,
                "font_path": str(FONT_PATH),
                "font_height_ratio": FONT_HEIGHT_RATIO,
                "samples_per_digit": args.samples_per_digit,
                "seed": args.seed,
                "patch_size": PATCH_SIZE,
                "square_inner_fraction": SQUARE_INNER_FRACTION,
                "model_format": "numpy-linear-svm-v1",
                "feature_count": model.feature_count,
                "class_count": len(model.classes),
                "training_implementation": "sklearn.svm.LinearSVC",
                "training_sklearn_version": sklearn.__version__,
                "svm_kernel": "linear",
                "svm_c": 2.5,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    accuracy = float(np.mean(numpy_predictions.astype(int) == labels))
    print(f"samples: {len(labels)}")
    print(f"training_accuracy: {accuracy:.4f}")
    print(f"model: {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
