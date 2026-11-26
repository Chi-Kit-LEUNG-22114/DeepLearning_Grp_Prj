from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PreparedSplits:
    arrays: dict[str, dict[str, np.ndarray]]
    feature_names: list[str]
    feature_groups: dict[str, list[int]]
    preprocessor: dict[str, Any]
    split_manifest: dict[str, Any]
    target_scaler: dict[str, float]


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input table does not exist: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    raise ValueError(f"Supported table formats are CSV, TSV, and Parquet: {path}")


def _as_dates(values: pd.Series, name: str) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    if dates.isna().any():
        examples = values[dates.isna()].head(5).tolist()
        raise ValueError(f"Invalid dates in {name}; examples: {examples}")
    return dates


def _feature_group(name: str) -> str:
    if name == "google_trends" or name.startswith(("gt_", "trends_available")):
        return "search_trends"
    if name.startswith(("comment_mean__", "comment_weighted_mean__")):
        return "comments_bge"
    if name.startswith("image_embedding__"):
        return "images_clip"
    if name.startswith("title_bge__"):
        return "titles_bge"
    if name.startswith("title_clip__"):
        return "titles_clip"
    if name.startswith((
        "image_posts_available",
        "title_posts_available",
        "caption_posts_available",
        "tag_posts_available",
        "comment_posts_available",
        "image_text_alignment_",
        "social_data_available",
    )):
        return "availability_quality"
    return "numeric_context"


def select_feature_set(feature_names: list[str], feature_set: str) -> list[str]:
    """Select ablation inputs without changing samples, labels, or split dates."""
    if feature_set not in {"multimodal", "search_only", "social_only"}:
        raise ValueError(f"Unknown data.feature_set: {feature_set!r}")
    selected = [name for name in feature_names if (
        feature_set == "multimodal"
        or (feature_set == "search_only" and _feature_group(name) == "search_trends")
        or (feature_set == "social_only" and _feature_group(name) != "search_trends")
    )]
    if not selected:
        raise ValueError(f"No eligible input columns for feature set {feature_set!r}.")
    return selected


def _feature_columns(
    features: pd.DataFrame,
    metadata: set[str],
    exclude_patterns: list[str],
) -> list[str]:
    patterns = [re.compile(pattern, flags=re.IGNORECASE) for pattern in exclude_patterns]
    cols = [
        name for name in features.columns
        if name not in metadata
        and pd.api.types.is_numeric_dtype(features[name])
        and not any(pattern.search(name) for pattern in patterns)
    ]
    if not cols:
        raise ValueError("No numeric feature columns remain after applying exclusions.")
    return cols


def _date_splits(dates: list[pd.Timestamp], train: float, validation: float, test: float):
    if not np.isclose(train + validation + test, 1.0, atol=1e-8):
        raise ValueError("train_fraction + validation_fraction + test_fraction must equal 1.")
    unique = sorted(pd.Timestamp(value) for value in set(dates))
    if len(unique) < 3:
        raise ValueError("At least three distinct input-week dates are needed for the split.")
    train_count = max(1, int(round(len(unique) * train)))
    validation_count = max(1, int(round(len(unique) * validation)))
    if train_count + validation_count >= len(unique):
        validation_count = max(1, len(unique) - train_count - 1)
    train_dates = unique[:train_count]
    validation_dates = unique[train_count:train_count + validation_count]
    test_dates = unique[train_count + validation_count:]
    if not train_dates or not validation_dates or not test_dates:
        raise ValueError("The available input-week dates cannot form non-empty splits.")
    return {
        "train": set(train_dates),
        "validation": set(validation_dates),
        "test": set(test_dates),
    }, {
        "train_start": train_dates[0].date().isoformat(),
        "train_end": train_dates[-1].date().isoformat(),
        "validation_start": validation_dates[0].date().isoformat(),
        "validation_end": validation_dates[-1].date().isoformat(),
        "test_start": test_dates[0].date().isoformat(),
        "test_end": test_dates[-1].date().isoformat(),
        "date_counts": {
            "train": len(train_dates),
            "validation": len(validation_dates),
            "test": len(test_dates),
        },
    }


def _load_labels(config: dict[str, Any], project_root: Path):
    columns = config["columns"]
    label_path = (project_root / config["paths"]["labels"]).resolve()
    labels = _read_table(label_path)
    brand_col = columns["label_brand_id"]
    input_week_col = columns["input_week_start"]
    target_col = columns["target"]
    target_period_col = columns.get("target_period")
    required = {brand_col, input_week_col, target_col}
    if not required.issubset(labels.columns):
        raise ValueError(
            f"BMI label table must contain {sorted(required)}. See README.md for its schema."
        )
    labels[input_week_col] = _as_dates(labels[input_week_col], input_week_col)
    labels[brand_col] = labels[brand_col].astype(str)
    labels[target_col] = pd.to_numeric(labels[target_col], errors="coerce")
    labels = labels.loc[np.isfinite(labels[target_col])].copy()
    if labels.empty:
        raise ValueError("No finite numeric BMI labels were found.")
    if labels.duplicated([brand_col, input_week_col]).any():
        raise ValueError("BMI labels must be unique by brand_id and input_week_start.")

    if target_period_col and target_period_col in labels.columns:
        raw_periods = labels[target_period_col]
        parsed = pd.to_datetime(raw_periods, errors="coerce", utc=True)
        invalid = raw_periods.notna() & parsed.isna()
        if invalid.any():
            raise ValueError(f"Invalid dates in optional {target_period_col}.")
        labels[target_period_col] = parsed.dt.tz_convert(None).dt.normalize()
        known = labels[target_period_col].notna()
        if (
            labels.loc[known, target_period_col]
            <= labels.loc[known, input_week_col]
        ).any():
            raise ValueError(
                f"{target_period_col} must be later than {input_week_col} when supplied."
            )
    else:
        target_period_col = None
    return labels, label_path, brand_col, input_week_col, target_col, target_period_col


def get_shared_split_plan(config: dict[str, Any], project_root: Path):
    """Calculate the one input-week split used by both models and encoder PCA."""
    labels, label_path, _, input_week_col, _, _ = _load_labels(config, project_root)
    data_cfg = config["data"]
    split_sets, summary = _date_splits(
        labels[input_week_col].tolist(),
        float(data_cfg["train_fraction"]),
        float(data_cfg["validation_fraction"]),
        float(data_cfg["test_fraction"]),
    )
    return split_sets, summary, label_path


def validate_encoder_preprocessing(feature_path: Path, expected_train_end: str) -> dict[str, Any]:
    """Reject PCA/scalers fitted on a different cutoff or on all observed weeks."""
    artifacts = feature_path.parent.parent / "artifacts"
    reducer_path = artifacts / "reducer_manifest.json"
    preprocessing_path = artifacts / "preprocessing_manifest.json"
    if not reducer_path.is_file() or not preprocessing_path.is_file():
        raise FileNotFoundError(
            "Encoder preprocessing manifests are missing. Run the encoder's "
            "07_fit_feature_reducers.py and 08_build_weekly_vectors.py first."
        )
    reducers = json.loads(reducer_path.read_text(encoding="utf-8"))
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    if preprocessing.get("fit_scope") != "training_dates":
        raise ValueError("Encoder numeric/trend preprocessing was not fit on training dates.")
    if preprocessing.get("fit_through") != expected_train_end:
        raise ValueError(
            "Encoder preprocessing cutoff does not match the shared split: "
            f"expected {expected_train_end}, found {preprocessing.get('fit_through')}."
        )
    mismatched = []
    unsafe = []
    for column, spec in reducers.items():
        if int(spec.get("components", 0)) <= 0:
            continue
        if spec.get("fit_scope") != "training_dates":
            unsafe.append(column)
        elif spec.get("fit_through") != expected_train_end:
            mismatched.append(column)
    if unsafe:
        raise ValueError(
            "Encoder PCA used a non-training fallback for: "
            f"{unsafe[:8]}. Disable the all-weeks fallback and refit."
        )
    if mismatched:
        raise ValueError(
            "Encoder PCA cutoff differs from the shared transfer split for: "
            f"{mismatched[:8]}."
        )
    return {
        "fit_scope": preprocessing["fit_scope"],
        "fit_through": preprocessing["fit_through"],
        "reducer_manifest": str(reducer_path),
        "preprocessing_manifest": str(preprocessing_path),
        "reduced_blocks": sum(int(item.get("components", 0)) > 0 for item in reducers.values()),
    }


def _fit_feature_scaler(
    frame: pd.DataFrame,
    feature_names: list[str],
    fit_dates: set[pd.Timestamp],
    date_column: str,
) -> dict[str, Any]:
    fit_rows = frame.loc[frame[date_column].isin(fit_dates), feature_names]
    if fit_rows.empty:
        raise ValueError("No training-period feature rows are available for model scaling.")
    values = fit_rows.to_numpy(dtype=np.float64)
    values[~np.isfinite(values)] = np.nan
    medians = np.nanmedian(values, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    imputed = np.where(np.isfinite(values), values, medians[None, :])
    means = imputed.mean(axis=0)
    scales = imputed.std(axis=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-8), scales, 1.0)
    return {
        "feature_names": feature_names,
        "median": medians.tolist(),
        "mean": means.tolist(),
        "scale": scales.tolist(),
    }


def _transform_features(frame: pd.DataFrame, preprocessor: dict[str, Any]) -> np.ndarray:
    names = preprocessor["feature_names"]
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ValueError(f"Reduced encoder table is missing trained features: {missing[:12]}")
    values = frame[names].to_numpy(dtype=np.float64)
    values[~np.isfinite(values)] = np.nan
    medians = np.asarray(preprocessor["median"], dtype=np.float64)
    means = np.asarray(preprocessor["mean"], dtype=np.float64)
    scales = np.asarray(preprocessor["scale"], dtype=np.float64)
    values = np.where(np.isfinite(values), values, medians[None, :])
    return ((values - means[None, :]) / scales[None, :]).astype(np.float32)


def _make_windows(
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
    transformed: np.ndarray,
    feature_names: list[str],
    split_assignments: dict[pd.Timestamp, str],
    context_length: int,
    brand_col: str,
    week_col: str,
    input_week_col: str,
    target_col: str,
    target_period_col: str | None,
):
    row_by_brand_date: dict[tuple[str, pd.Timestamp], int] = {}
    brand_rows: dict[str, list[tuple[pd.Timestamp, int]]] = {}
    for row_idx, (brand, week) in enumerate(zip(feature_df[brand_col], feature_df[week_col])):
        brand = str(brand)
        row_by_brand_date[(brand, week)] = row_idx
        brand_rows.setdefault(brand, []).append((week, row_idx))
    position_by_brand_row: dict[tuple[str, int], int] = {}
    for brand, rows in brand_rows.items():
        rows.sort(key=lambda item: item[0])
        for position, (_, row_idx) in enumerate(rows):
            position_by_brand_row[(brand, row_idx)] = position

    result = {
        split: {
            "X": [], "y": [], "brand_id": [], "input_week_start": [],
            "split_date": [], "target_period": [],
        }
        for split in ("train", "validation", "test")
    }
    skipped_context = 0
    for label in label_df.to_dict(orient="records"):
        brand = str(label[brand_col])
        input_date = label[input_week_col]
        split = split_assignments.get(input_date)
        anchor = row_by_brand_date.get((brand, input_date))
        if split is None or anchor is None:
            skipped_context += 1
            continue
        rows = brand_rows[brand]
        position = position_by_brand_row[(brand, anchor)]
        if position < context_length - 1:
            skipped_context += 1
            continue
        selected = rows[position - context_length + 1:position + 1]
        dates = [week for week, _ in selected]
        if len(selected) != context_length or any(
            (dates[i] - dates[i - 1]).days != 7 for i in range(1, len(dates))
        ):
            skipped_context += 1
            continue
        indices = [idx for _, idx in selected]
        x = transformed[indices]
        y = float(label[target_col])
        if not np.isfinite(x).all() or not np.isfinite(y):
            continue
        result[split]["X"].append(x)
        result[split]["y"].append(y)
        result[split]["brand_id"].append(brand)
        result[split]["input_week_start"].append(input_date.date().isoformat())
        result[split]["split_date"].append(input_date.date().isoformat())
        target_period = label.get(target_period_col) if target_period_col else None
        result[split]["target_period"].append(
            target_period.date().isoformat() if pd.notna(target_period) else ""
        )

    arrays = {}
    for split, values in result.items():
        if not values["y"]:
            arrays[split] = {
                "X": np.empty((0, context_length, len(feature_names)), dtype=np.float32),
                "y_raw": np.empty((0,), dtype=np.float32),
                "brand_id": np.asarray([], dtype=str),
                "input_week_start": np.asarray([], dtype=str),
                "split_date": np.asarray([], dtype=str),
                "target_period": np.asarray([], dtype=str),
            }
        else:
            arrays[split] = {
                "X": np.stack(values["X"]).astype(np.float32),
                "y_raw": np.asarray(values["y"], dtype=np.float32),
                "brand_id": np.asarray(values["brand_id"], dtype=str),
                "input_week_start": np.asarray(values["input_week_start"], dtype=str),
                "split_date": np.asarray(values["split_date"], dtype=str),
                "target_period": np.asarray(values["target_period"], dtype=str),
            }
    return arrays, skipped_context


def prepare_splits(config: dict[str, Any], project_root: Path) -> PreparedSplits:
    columns = config["columns"]
    data_cfg = config["data"]
    feature_path = (project_root / config["paths"]["weekly_features"]).resolve()
    feature_df = _read_table(feature_path)
    labels, label_path, label_brand_col, input_week_col, target_col, target_period_col = (
        _load_labels(config, project_root)
    )
    brand_col = columns["brand_id"]
    week_col = columns["week_start"]
    if not {brand_col, week_col}.issubset(feature_df.columns):
        raise ValueError(f"Reduced encoder output must contain {brand_col} and {week_col}.")
    feature_df[week_col] = _as_dates(feature_df[week_col], week_col)
    feature_df[brand_col] = feature_df[brand_col].astype(str)
    labels = labels.rename(columns={label_brand_col: brand_col})
    if feature_df.duplicated([brand_col, week_col]).any():
        raise ValueError("Reduced encoder table has duplicate brand/week rows.")
    unknown_brands = sorted(set(labels[brand_col]) - set(feature_df[brand_col]))
    if unknown_brands:
        raise ValueError(f"BMI labels contain brands absent from encoder output: {unknown_brands[:10]}")

    split_sets, split_summary = _date_splits(
        labels[input_week_col].tolist(),
        float(data_cfg["train_fraction"]),
        float(data_cfg["validation_fraction"]),
        float(data_cfg["test_fraction"]),
    )
    split_assignments = {date: name for name, dates in split_sets.items() for date in dates}
    encoder_manifest = validate_encoder_preprocessing(
        feature_path, split_summary["train_end"]
    )

    feature_df = feature_df.sort_values([brand_col, week_col]).reset_index(drop=True)
    patterns = list(data_cfg.get("exclude_feature_patterns", []))
    metadata = {
        brand_col, week_col, "date", "target_date", "target_gt_t_plus_1",
        "geo", "query", "source_name", "source_file",
    }
    feature_names = _feature_columns(feature_df, metadata, patterns)
    feature_set = data_cfg.get("feature_set", "multimodal")
    feature_names = select_feature_set(feature_names, feature_set)
    train_end = pd.Timestamp(split_summary["train_end"])
    fit_dates = set(feature_df.loc[feature_df[week_col] <= train_end, week_col])
    # Fit model-input normalization only on rows inside the shared train dates.
    feature_scaler = _fit_feature_scaler(feature_df, feature_names, fit_dates, week_col)
    transformed = _transform_features(feature_df, feature_scaler)
    arrays, skipped_context = _make_windows(
        feature_df=feature_df,
        label_df=labels,
        transformed=transformed,
        feature_names=feature_names,
        split_assignments=split_assignments,
        context_length=int(data_cfg["context_length"]),
        brand_col=brand_col,
        week_col=week_col,
        input_week_col=input_week_col,
        target_col=target_col,
        target_period_col=target_period_col,
    )
    if any(len(arrays[name]["y_raw"]) == 0 for name in ("train", "validation", "test")):
        counts = {name: len(values["y_raw"]) for name, values in arrays.items()}
        raise ValueError(f"At least one split has no complete labelled windows: {counts}.")

    training_y = arrays["train"]["y_raw"].astype(np.float64)
    y_mean = float(training_y.mean())
    y_scale = float(training_y.std())
    if not np.isfinite(y_scale) or y_scale < 1e-8:
        y_scale = 1.0
    target_scaler = {"mean": y_mean, "scale": y_scale}
    for split in arrays:
        arrays[split]["y"] = (
            (arrays[split]["y_raw"] - y_mean) / y_scale
        ).astype(np.float32)

    feature_groups: dict[str, list[int]] = {}
    for index, name in enumerate(feature_names):
        feature_groups.setdefault(_feature_group(name), []).append(index)
    feature_groups = {name: indices for name, indices in feature_groups.items() if indices}
    manifest = {
        "split_basis": input_week_col,
        "ratios": {
            "train": float(data_cfg["train_fraction"]),
            "validation": float(data_cfg["validation_fraction"]),
            "test": float(data_cfg["test_fraction"]),
        },
        **split_summary,
        "sample_counts": {
            name: int(len(values["y_raw"])) for name, values in arrays.items()
        },
        "skipped_incomplete_context_count": int(skipped_context),
        "context_length": int(data_cfg["context_length"]),
        "input_week_column": input_week_col,
        "target_column": target_col,
        "target_period_column": target_period_col,
        "feature_path": str(feature_path),
        "label_path": str(label_path),
        "feature_count": len(feature_names),
        "feature_set": feature_set,
        "feature_names": feature_names,
        "feature_groups": {name: len(indices) for name, indices in feature_groups.items()},
        "feature_groups_indices": feature_groups,
        "encoder_preprocessing": encoder_manifest,
    }
    return PreparedSplits(
        arrays=arrays,
        feature_names=feature_names,
        feature_groups=feature_groups,
        preprocessor=feature_scaler,
        split_manifest=manifest,
        target_scaler=target_scaler,
    )


def save_prepared_splits(prepared: PreparedSplits, output_dir: Path) -> None:
    shared_dir = output_dir / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    for split, arrays in prepared.arrays.items():
        np.savez_compressed(shared_dir / f"{split}.npz", **arrays)
    (shared_dir / "feature_preprocessor.json").write_text(
        json.dumps(prepared.preprocessor, indent=2), encoding="utf-8"
    )
    (shared_dir / "target_scaler.json").write_text(
        json.dumps(prepared.target_scaler, indent=2), encoding="utf-8"
    )
    (shared_dir / "split_manifest.json").write_text(
        json.dumps(prepared.split_manifest, indent=2), encoding="utf-8"
    )


def load_saved_splits(output_dir: Path):
    shared_dir = output_dir / "shared"
    manifest = json.loads((shared_dir / "split_manifest.json").read_text(encoding="utf-8"))
    preprocessor = json.loads(
        (shared_dir / "feature_preprocessor.json").read_text(encoding="utf-8")
    )
    target_scaler = json.loads(
        (shared_dir / "target_scaler.json").read_text(encoding="utf-8")
    )
    arrays = {}
    for split in ("train", "validation", "test"):
        with np.load(shared_dir / f"{split}.npz", allow_pickle=False) as data:
            arrays[split] = {key: data[key] for key in data.files}
    return arrays, manifest, preprocessor, target_scaler


def transform_feature_table(
    feature_df: pd.DataFrame,
    preprocessor: dict[str, Any],
    brand_col: str,
    week_col: str,
    context_length: int,
):
    if brand_col not in feature_df or week_col not in feature_df:
        raise ValueError(f"Prediction input requires {brand_col!r} and {week_col!r}.")
    feature_df = feature_df.copy()
    feature_df[brand_col] = feature_df[brand_col].astype(str)
    feature_df[week_col] = _as_dates(feature_df[week_col], week_col)
    names = preprocessor["feature_names"]
    missing = sorted(set(names) - set(feature_df.columns))
    if missing:
        raise ValueError(f"Reduced encoder table is missing trained features: {missing[:12]}")
    if feature_df.duplicated([brand_col, week_col]).any():
        raise ValueError("Prediction encoder table has duplicate brand/week rows.")
    feature_df = feature_df.sort_values([brand_col, week_col]).reset_index(drop=True)
    transformed = _transform_features(feature_df, preprocessor)
    windows, brands, ends = [], [], []
    for brand, indices in feature_df.groupby(brand_col, sort=False).indices.items():
        indices = list(indices)
        for end in range(context_length - 1, len(indices)):
            selected = indices[end - context_length + 1:end + 1]
            dates = feature_df.loc[selected, week_col].tolist()
            if any((dates[i] - dates[i - 1]).days != 7 for i in range(1, len(dates))):
                continue
            windows.append(transformed[selected])
            brands.append(str(brand))
            ends.append(dates[-1].date().isoformat())
    if not windows:
        raise ValueError("No complete consecutive-week prediction windows were found.")
    return (
        np.stack(windows).astype(np.float32),
        np.asarray(brands, dtype=str),
        np.asarray(ends, dtype=str),
    )
