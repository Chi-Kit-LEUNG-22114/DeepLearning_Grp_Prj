"""Training-mean and Ridge references on Andy's shared prepared splits."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_saved_splits
from .evaluation import export_evaluation, write_json


def summarize_windows(values: np.ndarray) -> np.ndarray:
    """Keep the last value, history mean, and end-to-start change per channel."""
    values = np.asarray(values)
    if values.ndim != 3 or min(values.shape) == 0 or not np.isfinite(values).all():
        raise ValueError("Baseline inputs must be finite non-empty [samples, weeks, channels] arrays.")
    return np.concatenate(
        [values[:, -1], values.mean(axis=1), values[:, -1] - values[:, 0]], axis=1
    )


def prediction_table(arrays: dict, predictions: np.ndarray, model: str) -> pd.DataFrame:
    # Match the neural export's float64 conversion so CSV serialization does not
    # round float32 outcomes differently and invalidate exact sample comparisons.
    actual = np.asarray(arrays["y_raw"], dtype=np.float64)
    return pd.DataFrame({
        "brand_id": arrays["brand_id"],
        "input_week_start": arrays["input_week_start"],
        "target_period": arrays["target_period"],
        "bmi_actual": actual,
        "bmi_predicted": predictions,
        "residual": np.asarray(predictions) - actual,
        "model": model,
    })


def run_baselines(output_dir: Path, config: dict) -> list[dict]:
    """Fit only when explicitly invoked by the runner or training comparison.

    Ridge alpha is selected on validation MAE; test outcomes never select it.
    Baselines use original-scale BMI and the exact prepared neural-model samples.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    arrays, manifest, _, _ = load_saved_splits(output_dir)
    settings = config.get("baselines", {})
    alphas = [float(value) for value in settings.get("ridge_alphas", [0.1, 1.0, 10.0, 100.0])]
    if not alphas or any(not np.isfinite(value) or value <= 0 for value in alphas):
        raise ValueError("baselines.ridge_alphas must contain positive finite values.")
    x = {name: summarize_windows(data["X"]) for name, data in arrays.items()}
    train_y = np.asarray(arrays["train"]["y_raw"], dtype=float)
    if not len(train_y) or not np.isfinite(train_y).all():
        raise ValueError("Training BMI values must be non-empty and finite.")
    training_mean = float(train_y.mean())
    best = None
    best_mae = float("inf")
    selected_alpha = None
    candidates = []
    for alpha in alphas:
        candidate = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        candidate.fit(x["train"], train_y)
        prediction = candidate.predict(x["validation"])
        mae = float(np.abs(prediction - arrays["validation"]["y_raw"]).mean())
        candidates.append({"alpha": alpha, "validation_mae": mae})
        if np.isfinite(mae) and mae < best_mae:
            best, best_mae, selected_alpha = candidate, mae, alpha
    if best is None:
        raise ValueError("Ridge validation did not produce a finite score.")
    results = []
    threshold = float(config.get("reporting", {}).get("change_threshold", 0.0))
    for name in ("training_mean", "ridge"):
        directory = output_dir / "baselines" / name
        result = {"model": name, "split_counts": manifest["sample_counts"], "split_basis": manifest["split_basis"]}
        for split in ("validation", "test"):
            predicted = np.full(len(arrays[split]["y_raw"]), training_mean) if name == "training_mean" else best.predict(x[split])
            table = prediction_table(arrays[split], predicted, name)
            result[split] = export_evaluation(table, directory, split, threshold)
            table.to_csv(directory / f"{split}_predictions.csv", index=False)
        write_json(directory / "metrics.json", result)
        results.append(result)
    # Persist the linear model as inspectable numeric coefficients, not an opaque pickle.
    scaler, ridge = best.steps[0][1], best.steps[1][1]
    write_json(output_dir / "baselines/ridge/model.json", {
        "feature_names": manifest["feature_names"],
        "summary_order": ["last", "mean", "last_minus_first"],
        "context_length": manifest["context_length"],
        "alpha": selected_alpha,
        "selection": "minimum validation MAE; no test-based selection",
        "candidates": candidates,
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coefficients": ridge.coef_.tolist(), "intercept": float(ridge.intercept_),
    })
    write_json(output_dir / "baselines/training_mean/model.json", {"training_mean": training_mean})
    return results
