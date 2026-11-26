"""Exercise orchestration with a fake estimator; no models are fitted in tests."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brand_momentum_transfer.baselines import prediction_table, run_baselines
from brand_momentum_transfer.evaluation import compare_prediction_tables


class BaselineTests(unittest.TestCase):
    def test_baseline_and_neural_outcomes_roundtrip_identically(self):
        actual = np.array([1.23456789, 9.87654321], dtype=np.float32)
        values = {"brand_id": ["a", "b"], "input_week_start": ["2026-01-05"] * 2,
                  "target_period": [""] * 2, "y_raw": actual}
        baseline = prediction_table(values, np.zeros(2), "training_mean")
        neural = baseline.copy()
        neural["bmi_actual"] = actual.astype(np.float64)
        neural["model"] = "ttm"
        with tempfile.TemporaryDirectory() as tmp:
            baseline.to_csv(Path(tmp) / "baseline.csv", index=False)
            neural.to_csv(Path(tmp) / "neural.csv", index=False)
            compare_prediction_tables({
                "baseline": pd.read_csv(Path(tmp) / "baseline.csv"),
                "neural": pd.read_csv(Path(tmp) / "neural.csv"),
            })

    def test_selection_and_scaling_receive_no_test_data(self):
        def arrays(date, features, labels):
            return {"X": np.full((2, 2, 1), features, dtype=np.float32),
                    "y_raw": np.array(labels, dtype=np.float32),
                    "brand_id": np.array(["a", "b"]),
                    "input_week_start": np.array([date] * 2),
                    "target_period": np.array([""] * 2)}
        data = {"train": arrays("2026-01-05", 1, [2, 4]),
                "validation": arrays("2026-01-12", 5, [10, 10]),
                "test": arrays("2026-01-19", 99, [1000, 1000])}
        manifest = {"sample_counts": {name: 2 for name in data},
                    "split_basis": "input_week_start", "feature_names": ["x"],
                    "context_length": 2}
        fitted = []

        class FakePipeline:
            def __init__(self, alpha):
                self.alpha = alpha
                self.steps = [("scaler", SimpleNamespace(mean_=np.zeros(3), scale_=np.ones(3))),
                              ("ridge", SimpleNamespace(coef_=np.zeros(3), intercept_=alpha))]

            def fit(self, x, y):
                fitted.append((self.alpha, x.copy(), y.copy()))
                return self

            def predict(self, x):
                return np.full(len(x), self.alpha)

        def pipeline(scaler, estimator):
            return FakePipeline(estimator.alpha)

        with tempfile.TemporaryDirectory() as tmp, patch(
            "brand_momentum_transfer.baselines.load_saved_splits", return_value=(data, manifest, {}, {})
        ), patch("sklearn.pipeline.make_pipeline", side_effect=pipeline):
            output = Path(tmp)
            results = run_baselines(output, {"baselines": {"ridge_alphas": [1, 10, 1000]}})
            saved = json.loads((output / "baselines/ridge/model.json").read_text())
            self.assertEqual(saved["alpha"], 10)  # 1000 would win on test, but must not be selected.
            self.assertEqual(len(fitted), 3)
            for _, x, y in fitted:
                np.testing.assert_array_equal(x, [[1, 1, 0], [1, 1, 0]])
                np.testing.assert_array_equal(y, [2, 4])
            predicted = pd.read_csv(output / "baselines/training_mean/test_predictions.csv")
            np.testing.assert_array_equal(predicted.bmi_predicted, [3, 3])
            self.assertEqual({row["model"] for row in results}, {"training_mean", "ridge"})


if __name__ == "__main__":
    unittest.main()
