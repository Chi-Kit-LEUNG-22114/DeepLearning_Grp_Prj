#!/usr/bin/env python3
"""Fit modality reducers and numeric scalers on training dates only."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from brand_encoder.common import load_yaml, ensure_dir, write_json
from brand_encoder.temporal import build_trend_features, TREND_FEATURES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.add_argument(
        "--train-end",
        help="Optional training cutoff override used to align preprocessing with a downstream split.",
    )
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out = Path(cfg["project"]["output_root"])
    df = pd.read_parquet(out / "weekly/brand_week_unreduced.parquet")
    df["week_start"] = pd.to_datetime(df.week_start)
    train_end = pd.Timestamp(args.train_end or cfg["splits"]["train_end"])
    train = df[df.week_start <= train_end]
    if train.empty:
        raise ValueError("No training-period brand-weeks found; adjust splits.train_end or provide earlier data.")

    artifacts = ensure_dir(out / "artifacts")
    manifest = {}
    mappings = {
        "image_embedding": cfg["reduction"]["image_components"],
        "title_clip": cfg["reduction"]["clip_text_components"],
        "caption_clip": cfg["reduction"]["clip_text_components"],
        "title_bge": cfg["reduction"]["title_components"],
        "caption_bge": cfg["reduction"]["caption_components"],
        "comment_mean": cfg["reduction"]["comment_components"],
        "comment_weighted_mean": cfg["reduction"]["comment_components"],
        "tags_bge": cfg["reduction"]["tag_components"],
    }
    for prefix, requested_components in mappings.items():
        for column in [c for c in train if c.startswith(prefix + "__")]:
            vectors = [
                np.asarray(value, dtype=np.float32)
                for value in train[column]
                if isinstance(value, (list, np.ndarray))
            ]
            fit_scope = "training_dates"
            if len(vectors) < 2 and cfg["reduction"].get("allow_unlabeled_all_weeks_fallback", False):
                vectors = [
                    np.asarray(value, dtype=np.float32)
                    for value in df[column]
                    if isinstance(value, (list, np.ndarray))
                ]
                fit_scope = "all_observed_weeks_unlabeled_fallback"
            if len(vectors) < 2:
                continue
            matrix = np.vstack(vectors)
            if not np.isfinite(matrix).all():
                raise ValueError(f"Non-finite training embeddings in {column}")
            if float(np.var(matrix, axis=0).sum()) <= 1e-12:
                manifest[column] = {"components": 0, "reason": "zero_variance"}
                continue
            components = max(1, min(int(requested_components), matrix.shape[1], matrix.shape[0] - 1))
            reducer = PCA(components, random_state=cfg["project"]["random_seed"]).fit(matrix)
            joblib.dump(reducer, artifacts / f"{column}.joblib")
            manifest[column] = {
                "components": components,
                "explained_variance": float(np.nansum(reducer.explained_variance_ratio_)),
                "fit_scope": fit_scope,
                "fit_through": train_end.date().isoformat() if fit_scope == "training_dates" else None,
            }

    summary_suffixes = ("_total", "_observed_posts", "_mean", "_median", "_min", "_max", "_p90")
    numeric_dtypes = set(train.select_dtypes(include=[np.number]).columns)
    numeric_columns = [
        c for c in train.columns if c in numeric_dtypes
        and (c in {"post_count", "engagement_weight_total", "metrics_snapshot_late_posts"}
        or (c.endswith(summary_suffixes) and not c.startswith("image_text_alignment_"))
        )
    ]
    numeric_scaler = StandardScaler().fit(
        np.log1p(train[numeric_columns].fillna(0).clip(lower=0))
    )
    joblib.dump((numeric_scaler, numeric_columns), artifacts / "numeric_scaler.joblib")

    trends = pd.read_parquet(out / "canonical/trends.parquet")
    trend_features = build_trend_features(trends, cfg["project"]["timezone"])
    trend_train = trend_features[
        trend_features.week_start <= train_end
    ]
    trend_columns = [
        c for c in TREND_FEATURES
        if c in trend_train and trend_train[c].notna().any()
    ]
    if trend_columns:
        trend_scaler = StandardScaler().fit(trend_train[trend_columns])
        joblib.dump((trend_scaler, trend_columns), artifacts / "trend_scaler.joblib")
    else:
        (artifacts / "trend_scaler.joblib").unlink(missing_ok=True)

    write_json(artifacts / "reducer_manifest.json", manifest)
    write_json(
        artifacts / "preprocessing_manifest.json",
        {
            "fit_scope": "training_dates",
            "fit_through": train_end.date().isoformat(),
            "numeric_scaler_columns": numeric_columns,
            "trend_scaler_columns": trend_columns if trend_columns else [],
        },
    )
    print(artifacts)


if __name__ == "__main__":
    main()
