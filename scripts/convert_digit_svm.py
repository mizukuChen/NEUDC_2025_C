from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_digit_svm import build_dataset
from src.vision.linear_svm import LinearSvmModel, save_linear_svm


LEGACY_MODEL_DIR = ROOT / "archive" / "assets" / "models" / "legacy_joblib"
DEFAULT_INPUT = LEGACY_MODEL_DIR / "digit_hog_svm.joblib"
DEFAULT_METADATA = LEGACY_MODEL_DIR / "digit_hog_svm_centroids.npz"
DEFAULT_OUTPUT = ROOT / "assets" / "models" / "digit_hog_linear_svm.npz"
DEFAULT_SOURCE_CONFIG = LEGACY_MODEL_DIR / "digit_hog_svm.json"
DEFAULT_OUTPUT_CONFIG = ROOT / "assets" / "models" / "digit_hog_linear_svm.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a scikit-learn LinearSVC Joblib model to a NumPy deployment model."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--output-config", type=Path, default=DEFAULT_OUTPUT_CONFIG)
    parser.add_argument("--samples-per-digit", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20250712)
    args = parser.parse_args()

    source = joblib.load(args.input)
    for attribute in ("coef_", "intercept_", "classes_", "decision_function", "predict"):
        if not hasattr(source, attribute):
            raise TypeError(f"input model does not provide {attribute}")
    model = LinearSvmModel(source.coef_, source.intercept_, source.classes_)

    with np.load(args.metadata, allow_pickle=False) as payload:
        centroids = payload["centroids"]

    features, _ = build_dataset(args.samples_per_digit, args.seed)
    source_scores = np.asarray(source.decision_function(features))
    numpy_scores = model.decision_function(features)
    max_score_error = float(np.max(np.abs(source_scores - numpy_scores)))
    source_predictions = np.asarray(source.predict(features))
    numpy_predictions = np.asarray(model.predict(features))
    if not np.array_equal(source_predictions, numpy_predictions):
        raise RuntimeError("converted model predictions differ from the source model")

    save_linear_svm(args.output, model, centroids=centroids)
    config = json.loads(args.source_config.read_text(encoding="utf-8"))
    config.pop("svm_implementation", None)
    config.update(
        {
            "model_format": "numpy-linear-svm-v1",
            "feature_count": model.feature_count,
            "class_count": len(model.classes),
            "converted_from": args.input.name,
        }
    )
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"samples_checked: {len(features)}")
    print(f"max_score_error: {max_score_error:.12g}")
    print("predictions_equal: true")
    print(f"model: {args.output}")
    print(f"config: {args.output_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
