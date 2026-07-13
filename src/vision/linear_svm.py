from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


MODEL_FORMAT_VERSION = 1


@dataclass(frozen=True)
class LinearSvmModel:
    """只包含线性 SVM 推理所需数值参数的部署模型。"""

    coef: np.ndarray
    intercept: np.ndarray
    classes: np.ndarray

    def __post_init__(self) -> None:
        coef = np.asarray(self.coef, dtype=np.float64)
        intercept = np.asarray(self.intercept, dtype=np.float64)
        classes = np.asarray(self.classes)
        if coef.ndim != 2 or coef.shape[0] < 2 or coef.shape[1] < 1:
            raise ValueError("coef must have shape (classes, features)")
        if intercept.shape != (coef.shape[0],):
            raise ValueError("intercept length must match coef rows")
        if classes.shape != (coef.shape[0],):
            raise ValueError("classes length must match coef rows")
        if not np.issubdtype(classes.dtype, np.integer):
            raise ValueError("classes must contain integers")
        if len(np.unique(classes)) != len(classes):
            raise ValueError("classes must be unique")
        if not np.isfinite(coef).all() or not np.isfinite(intercept).all():
            raise ValueError("model parameters must be finite")
        object.__setattr__(self, "coef", coef)
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "classes", classes.astype(np.int32, copy=False))

    @property
    def feature_count(self) -> int:
        return int(self.coef.shape[1])

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features)
        if values.ndim not in (1, 2):
            raise ValueError("features must be a vector or a matrix")
        if values.shape[-1] != self.feature_count:
            raise ValueError(
                f"expected {self.feature_count} features, got {values.shape[-1]}"
            )
        if not np.isfinite(values).all():
            raise ValueError("features must be finite")
        return values @ self.coef.T + self.intercept

    def predict(self, features: np.ndarray) -> int | np.ndarray:
        scores = self.decision_function(features)
        if scores.ndim == 1:
            return int(self.classes[int(np.argmax(scores))])
        return self.classes[np.argmax(scores, axis=1)]


def save_linear_svm(
    path: Path,
    model: LinearSvmModel,
    *,
    centroids: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "format_version": np.asarray(MODEL_FORMAT_VERSION, dtype=np.int32),
        "coef": model.coef,
        "intercept": model.intercept,
        "classes": model.classes,
    }
    if centroids is not None:
        values = np.asarray(centroids, dtype=np.float32)
        if values.shape != model.coef.shape:
            raise ValueError("centroids must have the same shape as coef")
        if not np.isfinite(values).all():
            raise ValueError("centroids must be finite")
        payload["centroids"] = values
    np.savez_compressed(path, **payload)


def load_linear_svm(path: Path) -> LinearSvmModel:
    with np.load(path, allow_pickle=False) as payload:
        required = {"format_version", "coef", "intercept", "classes"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"model file is missing: {', '.join(sorted(missing))}")
        version = int(np.asarray(payload["format_version"]).item())
        if version != MODEL_FORMAT_VERSION:
            raise ValueError(
                f"unsupported model format version {version}; expected {MODEL_FORMAT_VERSION}"
            )
        return LinearSvmModel(
            coef=payload["coef"],
            intercept=payload["intercept"],
            classes=payload["classes"],
        )
