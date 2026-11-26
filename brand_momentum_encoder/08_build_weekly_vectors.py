#!/usr/bin/env python3
"""Build the fixed-width brand-week model input table."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from brand_encoder.common import load_yaml, ensure_dir
from brand_encoder.temporal import build_trend_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/pipeline.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out = Path(cfg["project"]["output_root"])
    pooled = pd.read_parquet(out / "weekly/brand_week_unreduced.parquet")
    if pooled.empty:
        raise ValueError("No pooled brand-weeks found; check canonical posts and timestamps.")

    pooled["week_start"] = pd.to_datetime(pooled.week_start).dt.tz_localize(None)
    # Use one shared calendar range and the full brand registry, including brands
    # with no posts, so every brand-week has the same temporal coverage.
    brand_path = out / "canonical/brands.parquet"
    brand_ids = (
        pd.read_parquet(brand_path).brand_id.dropna().astype(str).unique().tolist()
        if brand_path.exists() else pooled.brand_id.dropna().astype(str).unique().tolist()
    )
    brand_ids = sorted(brand_ids)
    dates = pd.date_range(pooled.week_start.min(), pooled.week_start.max(), freq="W-MON")
    grids = [
        {"brand_id": brand, "week_start": date}
        for brand in brand_ids
        for date in dates
    ]
    weekly = pd.DataFrame(grids).merge(pooled, on=["brand_id", "week_start"], how="left")
    for col in ["post_count", "likes_total", "comments_total", "shares_total", "engagement_weight_total"]:
        if col in weekly:
            weekly[col] = weekly[col].fillna(0)
    for col in ["image_posts", "title_posts", "caption_posts", "tag_posts", "comment_posts"]:
        if col in weekly:
            weekly[col] = weekly[col].fillna(0).astype("int64")

    rows = {
        "brand_id": weekly["brand_id"].to_numpy(),
        "week_start": weekly["week_start"].to_numpy(),
    }
    art = out / "artifacts"
    reducer_manifest_path = art / "reducer_manifest.json"
    reducer_manifest = json.loads(reducer_manifest_path.read_text(encoding="utf-8")) if reducer_manifest_path.exists() else {}
    for path in sorted(art.glob("*.joblib")):
        if path.name in {"numeric_scaler.joblib", "trend_scaler.joblib"}:
            continue
        col = path.stem
        if col not in weekly or reducer_manifest.get(col, {}).get("components", 0) <= 0:
            continue
        model = joblib.load(path)
        vectors = []
        present = []
        for value in weekly[col]:
            vec = np.asarray(value, dtype=np.float32) if isinstance(value, (list, np.ndarray)) else None
            if vec is None or vec.shape != (model.n_features_in_,):
                vectors.append(np.zeros(model.n_features_in_, dtype=np.float32))
                present.append(False)
            else:
                vectors.append(vec)
                present.append(True)
        reduced = model.transform(np.vstack(vectors))
        reduced[~np.asarray(present, dtype=bool)] = 0.0
        for i in range(reduced.shape[1]):
            rows[f"{col}__pca_{i + 1:03d}"] = reduced[:, i]

    scaler_path = art / "numeric_scaler.joblib"
    if scaler_path.exists():
        scaler, numeric_cols = joblib.load(scaler_path)
        if numeric_cols:
            values = weekly[numeric_cols].fillna(0).clip(lower=0)
            transformed = scaler.transform(np.log1p(values))
            for i, col in enumerate(numeric_cols):
                rows[f"{col}__scaled"] = transformed[:, i]

    for col in ["post_count", "image_posts", "title_posts", "caption_posts", "tag_posts", "comment_posts"]:
        if col in weekly:
            rows[f"{col}_log1p" if col == "post_count" else f"{col}_available"] = (
                np.log1p(weekly[col].clip(lower=0)) if col == "post_count"
                else (weekly[col] > 0).astype("int8")
            )
    for col in ["image_text_alignment_mean", "image_text_alignment_std", "image_text_alignment_available"]:
        if col in weekly:
            rows[col] = weekly[col].fillna(0).to_numpy()

    rows = pd.DataFrame(rows)

    # Optional weekly Google Trends observations. The current aggregated JSON
    # contains query metadata only, so no trend values or labels are fabricated.
    trend_path = out / "canonical/trends.parquet"
    if trend_path.exists():
        trends = pd.read_parquet(trend_path)
        trend_features = build_trend_features(trends, cfg["project"]["timezone"])
        if not trend_features.empty:
            rows = rows.merge(trend_features, on=["brand_id", "week_start"], how="left")
            trend_scaler_path = art / "trend_scaler.joblib"
            if trend_scaler_path.exists():
                scaler, trend_cols = joblib.load(trend_scaler_path)
                rows[trend_cols] = scaler.transform(rows[trend_cols])
            rows["trends_available"] = rows["google_trends"].notna().astype("int8")

    # Availability is independent from whether the corresponding embedding
    # happens to contain zeros.
    rows["social_data_available"] = (weekly.post_count.to_numpy() > 0).astype("int8")
    rows = rows.sort_values(["brand_id", "week_start"]).reset_index(drop=True)
    dest = ensure_dir(out / "weekly") / "brand_week_features.parquet"
    rows.to_parquet(dest, index=False)
    feature_columns = [
        c for c in rows.select_dtypes(include=[np.number]).columns
        if not c.startswith("target_")
    ]
    (art / "feature_manifest.json").write_text(
        json.dumps({
            "unit": "brand_id x Monday-start week",
            "timezone": cfg["project"]["timezone"],
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "target_column": "target_gt_t_plus_1" if "target_gt_t_plus_1" in rows else None,
        }, indent=2),
        encoding="utf-8",
    )
    print(dest, rows.shape)


if __name__ == "__main__":
    main()
