#!/usr/bin/env python3
"""Pool variable-length post features into one record per brand-week."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from brand_encoder.common import load_yaml, ensure_dir, week_start
from brand_encoder.aggregation.pooling import pool_matrix


EMBEDDING_COLUMNS = [
    "image_embedding", "title_clip", "caption_clip",
    "title_bge", "caption_bge", "tags_bge",
    "comment_mean", "comment_weighted_mean",
]
NUMERIC_SUMMARIES = ("mean", "median", "min", "max", "p90")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/pipeline.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out = Path(cfg["project"]["output_root"])
    posts = pd.read_parquet(out / "features/post_vectors.parquet")
    posts["week_start"] = week_start(posts.created_at, cfg["project"]["timezone"])

    rows = []
    groups = posts.dropna(subset=["brand_id", "week_start"]).groupby(
        ["brand_id", "week_start"], sort=True
    )
    for (brand, week), group in groups:
        row = {
            "brand_id": brand,
            "week_start": week,
            "post_count": len(group),
            "engagement_weight_total": float(group.engagement_weight.sum()),
            "metrics_snapshot_late_posts": int(
                pd.to_numeric(group.get("metrics_snapshot_late", pd.Series(0, index=group.index)),
                              errors="coerce").fillna(0).sum()
            ),
        }
        if "comment_count_encoded" in group:
            row["comments_encoded_total"] = int(
                pd.to_numeric(group.comment_count_encoded, errors="coerce").fillna(0).sum()
            )

        for metric in ["likes", "comments_count", "shares", "views", "impressions"]:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_total"] = float(values.sum()) if len(values) else 0.0
            row[f"{metric}_observed_posts"] = int(len(values))
            for stat in NUMERIC_SUMMARIES:
                if len(values):
                    if stat == "p90":
                        value = values.quantile(0.9)
                    else:
                        value = getattr(values, stat)()
                    row[f"{metric}_{stat}"] = float(value)
                else:
                    row[f"{metric}_{stat}"] = 0.0

        availability = [
            ("image_posts", "image_available"),
            ("title_posts", "title_available"),
            ("caption_posts", "caption_available"),
            ("tag_posts", "tags_available"),
            ("comment_posts", "comment_count_encoded"),
        ]
        for feature, source in availability:
            if source in group:
                values = pd.to_numeric(group[source], errors="coerce").fillna(0)
                row[feature] = int((values > 0).sum())

        alignment = pd.to_numeric(
            group.get("image_text_alignment", pd.Series(index=group.index, dtype=float)),
            errors="coerce",
        ).dropna()
        row["image_text_alignment_mean"] = float(alignment.mean()) if len(alignment) else 0.0
        row["image_text_alignment_std"] = float(alignment.std()) if len(alignment) > 1 else 0.0
        row["image_text_alignment_available"] = int(len(alignment) > 0)

        for column in EMBEDDING_COLUMNS:
            if column not in group:
                continue
            vectors, weights = [], []
            for _, post in group.iterrows():
                value = post[column]
                if isinstance(value, (list, np.ndarray)):
                    vectors.append(np.asarray(value, dtype=np.float32))
                    weights.append(float(post.engagement_weight))
            if not vectors:
                continue
            pooled = pool_matrix(
                np.vstack(vectors),
                weights,
                use_max=cfg["pooling"]["max"],
                use_min=cfg["pooling"]["min"],
            )
            for statistic, vector in pooled.items():
                row[f"{column}__{statistic}"] = vector.tolist()

        rows.append(row)

    destination = ensure_dir(out / "weekly") / "brand_week_unreduced.parquet"
    pd.DataFrame(rows).to_parquet(destination, index=False)
    print(destination, "weeks", len(rows))


if __name__ == "__main__":
    main()
