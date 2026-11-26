#!/usr/bin/env python3
"""Validate weekly vectors and optional next-week labels."""
import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from brand_encoder.common import load_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/pipeline.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out = Path(cfg["project"]["output_root"])
    df = pd.read_parquet(out / "weekly/brand_week_features.parquet")
    issues = []

    if df.duplicated(["brand_id", "week_start"]).any():
        issues.append("duplicate brand-week rows")
    if not df.sort_values(["brand_id", "week_start"]).index.equals(df.index):
        issues.append("weekly table is not sorted by brand and week")

    numeric = df.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        issues.append("infinite numerical values")
    feature_cols = [c for c in numeric.columns if not c.startswith("target_")]
    if not feature_cols:
        issues.append("no numeric input feature channels")

    if "target_date" in df and "week_start" in df:
        target_date = pd.to_datetime(df["target_date"], errors="coerce")
        week_start = pd.to_datetime(df["week_start"], errors="coerce")
        invalid = target_date.notna() & ((target_date - week_start).dt.days != 7)
        if invalid.any():
            issues.append(f"{int(invalid.sum())} labels are not exactly one week ahead")

    input_counts = {}
    post_vectors_path = out / "features/post_vectors.parquet"
    if post_vectors_path.exists():
        post_vectors = pd.read_parquet(post_vectors_path)
        input_counts["posts"] = len(post_vectors)
        late_metrics = post_vectors["metrics_snapshot_late"] if "metrics_snapshot_late" in post_vectors else pd.Series(0, index=post_vectors.index)
        input_counts["late_metric_snapshot_posts"] = int(
            pd.to_numeric(late_metrics, errors="coerce")
            .fillna(0).sum()
        )
    image_path = out / "embeddings/images.parquet"
    if image_path.exists():
        images = pd.read_parquet(image_path)
        input_counts["image_files_encoded"] = len(images)
        image_available = images["available"] if "available" in images else pd.Series(0, index=images.index)
        input_counts["image_files_available"] = int(
            pd.to_numeric(image_available, errors="coerce").fillna(0).sum()
        )
    text_path = out / "embeddings/post_text.parquet"
    if text_path.exists():
        text = pd.read_parquet(text_path)
        input_counts["post_text_rows"] = len(text)
        input_counts["titles_encoded"] = int(text.get("title_available", pd.Series(dtype=int)).sum())
        input_counts["captions_encoded"] = int(text.get("caption_available", pd.Series(dtype=int)).sum())
    comment_path = out / "embeddings/comments_by_post.parquet"
    if comment_path.exists():
        comments = pd.read_parquet(comment_path)
        input_counts["posts_with_comments_encoded"] = len(comments)
        comment_count = comments["comment_count_encoded"] if "comment_count_encoded" in comments else pd.Series(0, index=comments.index)
        input_counts["comments_encoded"] = int(
            pd.to_numeric(comment_count, errors="coerce")
            .fillna(0).sum()
        )

    report = {
        "rows": len(df),
        "brands": int(df.brand_id.nunique()),
        "weeks": int(df.week_start.nunique()),
        "numeric_features": len(numeric.columns),
        "feature_channels": len(feature_cols),
        "input_counts": input_counts,
        "trends_values_present": bool("google_trends" in df and df.google_trends.notna().any()),
        "target_labels_present": bool("target_gt_t_plus_1" in df and df.target_gt_t_plus_1.notna().any()),
        "issues": issues,
        "missing_rate_top": df.isna().mean().sort_values(ascending=False).head(20).to_dict(),
    }
    write_json(out / "feature_validation_report.json", report)
    pipeline = Path(args.config).resolve()
    input_json = cfg["project"].get("input_json")
    input_path = Path(input_json) if input_json else None
    if input_path is not None and not input_path.is_absolute():
        input_path = (pipeline.parent / input_path).resolve()
    packages = {}
    for package in ["numpy", "pandas", "scikit-learn", "torch", "transformers", "sentence-transformers"]:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    write_json(out / "run_manifest.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_config": str(pipeline),
        "input_json": str(input_path) if input_path else None,
        "input_json_exists": bool(input_path and input_path.exists()),
        "timezone": cfg["project"]["timezone"],
        "week_frequency": "Monday-start; weekly observations",
        "models": cfg.get("models", {}),
        "pooling": cfg.get("pooling", {}),
        "splits": cfg.get("splits", {}),
        "reducer_manifest": json.loads(
            (out / "artifacts/reducer_manifest.json").read_text(encoding="utf-8")
        ) if (out / "artifacts/reducer_manifest.json").exists() else {},
        "feature_columns": feature_cols,
        "packages": packages,
        "validation": report,
    })
    print(json.dumps(report, indent=2, default=str))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
