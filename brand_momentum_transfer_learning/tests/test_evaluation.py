from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brand_momentum_transfer.baselines import summarize_windows
from brand_momentum_transfer.data import select_feature_set
from brand_momentum_transfer.evaluation import (
    build_watchlist, compare_prediction_tables, evaluate_predictions,
    export_evaluation, regression_metrics, validate_predictions,
)


def example():
    return pd.DataFrame({
        "brand_id": ["a", "b", "a", "b"],
        "input_week_start": ["2026-01-05"] * 2 + ["2026-01-12"] * 2,
        "target_period": ["2026-01-12"] * 2 + ["2026-01-19"] * 2,
        "bmi_actual": [1., 3., 4., 2.],
        "bmi_predicted": [1., 3., 4., 2.],
    })


class EvaluationTests(unittest.TestCase):
    def test_perfect_predictions_and_date_rankings(self):
        metrics, brands, dates = evaluate_predictions(example())
        self.assertEqual(metrics["mae"], 0)
        self.assertAlmostEqual(metrics["mean_date_spearman"], 1)
        self.assertEqual(metrics["ranked_date_count"], 2)
        self.assertEqual(len(brands), 2)
        self.assertEqual(len(dates), 2)

    def test_date_ranking_does_not_use_pooled_correlation(self):
        frame = example()
        frame["bmi_actual"] = [1., 2., 101., 102.]
        frame["bmi_predicted"] = [2., 1., 102., 101.]
        metrics, _, _ = evaluate_predictions(frame)
        self.assertGreater(metrics["spearman"], 0)
        self.assertAlmostEqual(metrics["mean_date_spearman"], -1)

    def test_constant_predictions_have_undefined_correlation(self):
        metrics = regression_metrics([1, 2, 3], [2, 2, 2])
        self.assertTrue(np.isnan(metrics["spearman"]))
        frame = example()
        frame["bmi_predicted"] = 2.
        metrics, _, _ = evaluate_predictions(frame)
        self.assertEqual(metrics["ranked_date_count"], 0)

    def test_invalid_predictions_fail(self):
        for column, value in [("bmi_predicted", np.inf), ("input_week_start", "bad"), ("brand_id", "")]:
            frame = example()
            frame.loc[0, column] = value
            with self.assertRaises(ValueError):
                validate_predictions(frame)
        with self.assertRaises(ValueError):
            validate_predictions(pd.concat([example(), example().iloc[:1]]))

    def test_target_horizons_cannot_be_mixed_in_rankings(self):
        frame = example()
        frame.loc[0, "target_period"] = "2026-01-19"
        with self.assertRaisesRegex(ValueError, "common target_period"):
            evaluate_predictions(frame)

    def test_watchlist_changes_and_rank_improvement(self):
        watch = build_watchlist(example(), 0.5).set_index("brand_id")
        self.assertEqual(watch.loc["a", "momentum_status"], "Gaining traction")
        self.assertEqual(watch.loc["a", "predicted_score_change"], 3)
        self.assertEqual(watch.loc["a", "rank_improvement"], 1)
        self.assertEqual(watch.loc["b", "momentum_status"], "Losing momentum")
        self.assertNotIn("bmi_actual", watch)
        self.assertEqual(len(watch), 2)

    def test_first_snapshot_does_not_invent_change(self):
        watch = build_watchlist(example().iloc[:2])
        self.assertTrue(watch.predicted_score_change.isna().all())
        self.assertEqual(set(watch.momentum_status), {"Insufficient history"})

    def test_different_portfolios_do_not_imply_rank_improvement(self):
        frame = example()
        frame.loc[3, "brand_id"] = "c"
        watch = build_watchlist(frame)
        self.assertTrue(watch.rank_improvement.isna().all())
        self.assertFalse(watch.rank_change_comparable.any())

    def test_different_forecast_horizons_do_not_imply_change(self):
        frame = example()
        frame.loc[2:, "target_period"] = "2026-01-26"
        watch = build_watchlist(frame)
        self.assertEqual(len(watch), 4)
        self.assertTrue(watch.predicted_score_change.isna().all())

    def test_threshold_and_missing_labels(self):
        frame = example().drop(columns=["bmi_actual", "target_period"])
        self.assertEqual(len(build_watchlist(frame)), 2)
        watch = build_watchlist(frame, 3)
        self.assertEqual(set(watch.momentum_status), {"Stable"})
        with self.assertRaises(ValueError):
            build_watchlist(frame, -1)

    def test_comparison_requires_identical_samples_and_outcomes(self):
        other = example().iloc[::-1].copy()
        other["bmi_predicted"] += 1
        compare_prediction_tables({"a": example(), "b": other})
        other.loc[0, "bmi_actual"] += 1
        with self.assertRaises(ValueError):
            compare_prediction_tables({"a": example(), "b": other})
        with self.assertRaises(ValueError):
            compare_prediction_tables({"a": example(), "b": example().iloc[:2]})

    def test_summary_preserves_temporal_order(self):
        values = np.array([[[1., 2.], [3., 6.]]])
        np.testing.assert_array_equal(summarize_windows(values), [[3, 6, 2, 4, 2, 4]])
        with self.assertRaises(ValueError):
            summarize_windows(np.zeros((1, 0, 2)))

    def test_ablation_selection(self):
        names = ["google_trends", "gt_lag_1", "trends_available", "title_bge__mean__pca_001", "post_count__scaled"]
        self.assertEqual(select_feature_set(names, "search_only"), names[:3])
        self.assertEqual(select_feature_set(names, "social_only"), names[3:])
        self.assertEqual(select_feature_set(names, "multimodal"), names)
        with self.assertRaises(ValueError):
            select_feature_set(names[3:], "search_only")

    def test_json_export_uses_null_for_undefined_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = example()
            frame["bmi_predicted"] = 2.
            export_evaluation(frame, Path(tmp), "test")
            raw = (Path(tmp) / "test_evaluation.json").read_text()
            self.assertNotIn("NaN", raw)
            self.assertIsNone(json.loads(raw)["spearman"])

    def test_offline_cli_uses_only_saved_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for name in ("ttm", "chronos2"):
                directory = output / "models" / name
                directory.mkdir(parents=True)
                example().to_csv(directory / "test_predictions.csv", index=False)
            subprocess.run([
                sys.executable, str(ROOT / "evaluate_results.py"),
                "--output-root", str(output),
            ], check=True, capture_output=True, text=True)
            self.assertTrue((output / "comparison/test_evaluation.csv").is_file())
            self.assertEqual(len(pd.read_csv(output / "models/ttm/test_watchlist.csv")), 2)


if __name__ == "__main__":
    unittest.main()
