from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.vision.linear_svm import LinearSvmModel, load_linear_svm, save_linear_svm


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "assets" / "models" / "digit_hog_linear_svm.npz"


class LinearSvmModelTest(unittest.TestCase):
    def test_predict_uses_highest_linear_score(self) -> None:
        model = LinearSvmModel(
            coef=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            intercept=np.asarray([0.1, -0.1]),
            classes=np.asarray([3, 7]),
        )
        self.assertEqual(model.predict(np.asarray([2.0, 1.0])), 3)
        np.testing.assert_array_equal(
            model.predict(np.asarray([[2.0, 1.0], [0.0, 4.0]])),
            np.asarray([3, 7]),
        )

    def test_round_trip_and_format_validation(self) -> None:
        model = LinearSvmModel(
            coef=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            intercept=np.asarray([0.1, -0.1]),
            classes=np.asarray([3, 7]),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            save_linear_svm(path, model)
            loaded = load_linear_svm(path)
            np.testing.assert_array_equal(loaded.coef, model.coef)
            np.testing.assert_array_equal(loaded.intercept, model.intercept)
            np.testing.assert_array_equal(loaded.classes, model.classes)

            invalid = Path(directory) / "invalid.npz"
            np.savez(invalid, format_version=np.asarray(999), coef=model.coef)
            with self.assertRaises(ValueError):
                load_linear_svm(invalid)

    def test_rejects_wrong_feature_count_and_nonfinite_values(self) -> None:
        model = LinearSvmModel(
            coef=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            intercept=np.asarray([0.1, -0.1]),
            classes=np.asarray([3, 7]),
        )
        with self.assertRaises(ValueError):
            model.decision_function(np.asarray([1.0, 2.0, 3.0]))
        with self.assertRaises(ValueError):
            model.decision_function(np.asarray([np.nan, 1.0]))

    def test_runtime_imports_without_sklearn_or_joblib(self) -> None:
        code = f"""
import importlib.abc
import pathlib
import sys
class BlockTrainingDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'sklearn', 'joblib'}}:
            raise ImportError('training dependency imported at runtime: ' + fullname)
        return None
sys.meta_path.insert(0, BlockTrainingDependencies())
sys.path.insert(0, {str(ROOT)!r})
from src.vision.recognize_digits import HogSvmDigitRecognizer
HogSvmDigitRecognizer.load(pathlib.Path({str(MODEL)!r}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
