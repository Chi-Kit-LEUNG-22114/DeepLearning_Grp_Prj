"""Offline evaluation and portfolio reporting. No model loading or training."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) < 2 or np.std(actual) <= 1e-12 or np.std(predicted) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(actual, predicted)[0, 1])


def regression_metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if actual.shape != predicted.shape or not len(actual):
        raise ValueError("Metrics require non-empty, equally sized actual/predicted arrays.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite actual and predicted values.")
    error = predicted - actual
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "n": len(actual),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(error.mean()),
        "r2": float(1 - np.sum(error ** 2) / denominator)
        if denominator > 1e-12 else float("nan"),
        "pearson": _correlation(actual, predicted),
        "spearman": _correlation(
            pd.Series(actual).rank().to_numpy(), pd.Series(predicted).rank().to_numpy()
        ),
    }


def validate_predictions(frame: pd.DataFrame, require_actual: bool = True) -> pd.DataFrame:
    required = {"brand_id", "input_week_start", "bmi_predicted"}
    if require_actual:
        required.add("bmi_actual")
    if not required.issubset(frame):
        raise ValueError(f"Missing prediction columns: {sorted(required - set(frame))}")
    if frame.empty:
        raise ValueError("Prediction table is empty.")
    if "model" in frame and frame.model.nunique(dropna=False) != 1:
        raise ValueError("Evaluate one model per prediction table.")
    result = frame.copy()
    if result.brand_id.isna().any() or result.brand_id.astype(str).str.strip().eq("").any():
        raise ValueError("Prediction brand IDs must be populated.")
    result["brand_id"] = result.brand_id.astype(str)
    for column in ("input_week_start", "target_period"):
        if column not in result:
            result[column] = ""
        raw = result[column].fillna("").astype(str)
        dates = pd.to_datetime(raw.replace("", None), errors="coerce", utc=True, format="mixed")
        if (raw.ne("") & dates.isna()).any() or (
            column == "input_week_start" and dates.isna().any()
        ):
            raise ValueError(f"Invalid dates in {column}.")
        result[column] = dates.dt.strftime("%Y-%m-%d").fillna("")
    known = result.target_period.ne("")
    if (result.loc[known, "target_period"] <= result.loc[known, "input_week_start"]).any():
        raise ValueError("target_period must be later than input_week_start.")
    if result.duplicated(["brand_id", "input_week_start"]).any():
        raise ValueError("Duplicate brand/input-week predictions; evaluate each model separately.")
    # Mixing known and unknown horizons in one cohort makes ranks ambiguous.
    if result.groupby("input_week_start").target_period.nunique().gt(1).any():
        raise ValueError("Each input date must have a common target_period for ranking.")
    for column in ("bmi_predicted", "bmi_actual"):
        if column not in result:
            continue
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if not np.isfinite(result[column]).all():
            raise ValueError(f"Non-finite values in {column}.")
    return result.sort_values(["input_week_start", "brand_id"]).reset_index(drop=True)


def rank_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    result = validate_predictions(frame, require_actual="bmi_actual" in frame)
    groups = result.groupby("input_week_start")
    # Higher BMI is treated as stronger momentum. Ties receive average ranks.
    result["predicted_rank"] = groups.bmi_predicted.rank(ascending=False, method="average")
    result["portfolio_size"] = groups.brand_id.transform("size")
    if "bmi_actual" in result:
        result["actual_rank"] = groups.bmi_actual.rank(ascending=False, method="average")
        result["absolute_rank_error"] = (result.predicted_rank - result.actual_rank).abs()
    return result


def evaluate_predictions(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    ranked = rank_predictions(validate_predictions(frame))
    by_date = []
    for date, group in ranked.groupby("input_week_start", sort=True):
        by_date.append({
            "input_week_start": date,
            "target_period": group.target_period.iloc[0],
            "n_brands": len(group),
            "spearman": _correlation(group.actual_rank.to_numpy(), group.predicted_rank.to_numpy()),
            "mean_absolute_rank_error": float(group.absolute_rank_error.mean()),
        })
    ranking = pd.DataFrame(by_date)
    metrics = regression_metrics(ranked.bmi_actual, ranked.bmi_predicted)
    metrics.update({
        "mean_date_spearman": float(ranking.spearman.mean()),
        "ranked_date_count": int(ranking.spearman.notna().sum()),
        "date_count": len(ranking),
        "mean_absolute_rank_error": float(ranked.absolute_rank_error.mean()),
    })
    brands = pd.DataFrame([
        {"brand_id": brand, **regression_metrics(group.bmi_actual, group.bmi_predicted)}
        for brand, group in ranked.groupby("brand_id", sort=True)
    ])
    return metrics, brands, ranking


def build_watchlist(frame: pd.DataFrame, change_threshold: float = 0.0) -> pd.DataFrame:
    """Latest common snapshot versus the preceding snapshot at the same horizon.

    Changes are prediction revisions across weeks, not observed growth or sales.
    Rank changes are reported only when the two portfolios contain the same brands.
    """
    if not np.isfinite(change_threshold) or change_threshold < 0:
        raise ValueError("Watchlist change_threshold must be finite and non-negative.")
    ranked = rank_predictions(frame)
    dates = pd.to_datetime(ranked.input_week_start)
    targets = pd.to_datetime(ranked.target_period.replace("", None))
    ranked["horizon_days"] = (targets - dates).dt.days.astype("Int64")
    snapshots = []
    for _, history in ranked.groupby("horizon_days", dropna=False):
        periods = sorted(history.input_week_start.unique())
        current = history.loc[history.input_week_start.eq(periods[-1])].copy()
        previous = history.loc[history.input_week_start.eq(periods[-2])] if len(periods) > 1 else history.iloc[:0]
        comparable = len(previous) > 0 and set(current.brand_id) == set(previous.brand_id)
        previous = previous.set_index("brand_id")
        current["previous_input_week_start"] = periods[-2] if len(periods) > 1 else ""
        current["previous_bmi_predicted"] = current.brand_id.map(previous.bmi_predicted)
        current["predicted_score_change"] = current.bmi_predicted - current.previous_bmi_predicted
        current["previous_rank"] = current.brand_id.map(previous.predicted_rank) if comparable else np.nan
        current["rank_improvement"] = current.previous_rank - current.predicted_rank
        current["rank_change_comparable"] = comparable
        current["momentum_status"] = np.select(
            [current.predicted_score_change.isna(),
             current.predicted_score_change > change_threshold,
             current.predicted_score_change < -change_threshold],
            ["Insufficient history", "Gaining traction", "Losing momentum"], default="Stable",
        )
        current["watchlist_action"] = current.momentum_status.map({
            "Insufficient history": "Collect a comparable prior prediction",
            "Gaining traction": "Review for feature or promotion",
            "Losing momentum": "Investigate seasonality and competitive context",
            "Stable": "Monitor",
        })
        # Watchlists must not imply that evaluation outcomes are available at prediction time.
        current = current.drop(columns=["bmi_actual", "residual", "actual_rank", "absolute_rank_error"], errors="ignore")
        snapshots.append(current)
    return pd.concat(snapshots, ignore_index=True).sort_values(
        ["input_week_start", "predicted_rank", "brand_id"]
    ).reset_index(drop=True)


def write_json(path: Path, value) -> None:
    """Write strict JSON: undefined metrics are null rather than NaN."""
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return None
        if isinstance(item, np.generic):
            return item.item()
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def export_evaluation(frame: pd.DataFrame, directory: Path, split: str, change_threshold: float = 0.0) -> dict:
    metrics, brands, ranking = evaluate_predictions(frame)
    directory.mkdir(parents=True, exist_ok=True)
    brands.to_csv(directory / f"{split}_per_brand_metrics.csv", index=False)
    ranking.to_csv(directory / f"{split}_ranking_metrics.csv", index=False)
    ranked = rank_predictions(frame)
    ranked.to_csv(directory / f"{split}_ranked_predictions.csv", index=False)
    errors = ranked.assign(
        residual=ranked.bmi_predicted - ranked.bmi_actual,
        absolute_error=(ranked.bmi_predicted - ranked.bmi_actual).abs(),
    ).sort_values(["absolute_error", "brand_id"], ascending=[False, True]).head(20)
    errors.to_csv(directory / f"{split}_error_cases.csv", index=False)
    build_watchlist(frame, change_threshold).to_csv(directory / f"{split}_watchlist.csv", index=False)
    write_json(directory / f"{split}_evaluation.json", metrics)
    return metrics


def compare_prediction_tables(tables: dict[str, pd.DataFrame]) -> None:
    """Prevent comparing models evaluated on different samples or actual outcomes."""
    reference = None
    for name, frame in tables.items():
        current = validate_predictions(frame)[["brand_id", "input_week_start", "target_period", "bmi_actual"]]
        if reference is None:
            reference = current
        elif not (
            current.drop(columns="bmi_actual").equals(reference.drop(columns="bmi_actual"))
            and np.array_equal(current.bmi_actual.to_numpy(), reference.bmi_actual.to_numpy())
        ):
            raise ValueError(f"Model {name!r} does not use the same samples, target periods, and actual values.")
