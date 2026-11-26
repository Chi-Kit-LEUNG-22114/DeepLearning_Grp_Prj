#!/usr/bin/env python3
"""Encode eligible comments and pool them within each post."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from brand_encoder.common import load_yaml, ensure_dir, select_device, numeric
from brand_encoder.encoders.text_encoder import TextEncoder
from brand_encoder.aggregation.pooling import pool_matrix


OUTPUT_COLUMNS = [
    "post_id", "comment_count_encoded", "comment_mean",
    "comment_weighted_mean", "comment_std", "comment_likes_total", "encoder",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/pipeline.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out = Path(cfg["project"]["output_root"])
    comments = pd.read_parquet(out / "canonical/comments.parquet")
    posts = pd.read_parquet(out / "canonical/posts.parquet")
    rows = []

    if not comments.empty:
        comments["text"] = comments.text.fillna("").astype(str).str.strip()
        comments = comments[comments.text.ne("")].copy()

    # Exclude audience comments written after the observation post's week.
    if not comments.empty and "created_at" in comments and "created_at" in posts:
        post_dates = posts.set_index("post_id")["created_at"]
        post_local = pd.to_datetime(
            comments.post_id.map(post_dates), utc=True, errors="coerce"
        ).dt.tz_convert(cfg["project"]["timezone"])
        week_start = (
            post_local - pd.to_timedelta(post_local.dt.weekday, unit="D")
        ).dt.normalize()
        cutoff = (
            week_start + pd.Timedelta(days=7) - pd.Timedelta(nanoseconds=1)
        ).dt.tz_convert("UTC")
        comment_dates = pd.to_datetime(comments.created_at, utc=True, errors="coerce")
        comments = comments.loc[comment_dates.notna() & (comment_dates <= cutoff)].copy()

    if comments.empty:
        dest = ensure_dir(out / "embeddings") / "comments_by_post.parquet"
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_parquet(dest, index=False)
        print(dest, "no eligible comments")
        return

    comments = numeric(comments, ["likes"])
    max_comments = int(cfg["comments"]["max_comments_per_post"])
    samples = []
    for _, group in comments.groupby("post_id", sort=False):
        if len(group) > max_comments:
            # Current JSON comment likes are not historical snapshots, so sample
            # reproducibly rather than using likes to select future top comments.
            group = group.sample(n=max_comments, random_state=cfg["comments"]["sampling_seed"])
        samples.append(group)
    comments = pd.concat(samples, ignore_index=True)

    encoder = TextEncoder(
        cfg["models"]["text_encoder"],
        select_device(cfg["models"]["device"]),
        cfg["text"]["max_chars_per_chunk"],
        True,
        cfg["models"].get("offline", False),
    )
    vectors = encoder.encode_many(
        comments.text.tolist(), int(cfg["models"]["text_batch_size"])
    )
    comments["_embedding"] = vectors

    for post_id, group in tqdm(comments.groupby("post_id", sort=False), total=comments.post_id.nunique()):
        matrix = np.vstack(group["_embedding"].to_list())
        # Comment like counts are unavailable as historical snapshots in this input.
        weights = np.ones(len(group), dtype=np.float32)
        pooled = pool_matrix(matrix, weights)
        rows.append({
            "post_id": post_id,
            "comment_count_encoded": len(group),
            "comment_mean": pooled["mean"].tolist(),
            "comment_weighted_mean": pooled["weighted_mean"].tolist(),
            "comment_std": pooled["std"].tolist(),
            "comment_likes_total": None,
            "encoder": encoder.model_name,
        })

    dest = ensure_dir(out / "embeddings") / "comments_by_post.parquet"
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_parquet(dest, index=False)
    print(dest, "posts", len(rows), "comments", len(comments))


if __name__ == "__main__":
    main()
