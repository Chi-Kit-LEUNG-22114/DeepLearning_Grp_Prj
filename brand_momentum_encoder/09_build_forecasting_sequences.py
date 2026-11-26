#!/usr/bin/env python3
"""Create fixed-shape history windows and optional one-week-ahead labels."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from brand_encoder.common import load_yaml, ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/pipeline.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out = Path(cfg["project"]["output_root"])
    df = pd.read_parquet(out / "weekly/brand_week_features.parquet")
    df["week_start"] = pd.to_datetime(df.week_start)
    length = int(cfg["forecast"]["context_length"])
    metadata = {
        "brand_id", "week_start", "date", "target_date", "target_gt_t_plus_1",
        "geo", "query", "source_name", "source_file",
    }
    features = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in metadata and not c.startswith("target_")
    ]
    if not features:
        raise ValueError("No numeric feature channels found in brand_week_features.parquet")

    dest = ensure_dir(out / "sequences")
    bounds = [
        ("train", pd.Timestamp.min, pd.Timestamp(cfg["splits"]["train_end"])),
        ("validation", pd.Timestamp(cfg["splits"]["train_end"]), pd.Timestamp(cfg["splits"]["validation_end"])),
        ("test", pd.Timestamp(cfg["splits"]["validation_end"]), pd.Timestamp(cfg["splits"]["test_end"])),
    ]
    for name, lower, upper in bounds:
        xs, ys, brands, ends = [], [], [], []
        for brand, group in df.sort_values("week_start").groupby("brand_id"):
            group = group.sort_values("week_start").reset_index(drop=True)
            for i in range(length - 1, len(group)):
                row = group.iloc[i]
                if not (row.week_start > lower and row.week_start <= upper):
                    continue
                window = group.iloc[i - length + 1:i + 1]
                # Never create a model sample across a missing calendar week.
                if len(window) != length or (window.week_start.diff().dropna().dt.days != 7).any():
                    continue
                x = window[features].astype(float).to_numpy(np.float32)
                if not np.isfinite(x).all():
                    continue
                has_target = "target_gt_t_plus_1" in row.index and pd.notna(row.get("target_gt_t_plus_1"))
                if not has_target:
                    continue
                target_date = pd.to_datetime(row.get("target_date"), errors="coerce")
                if pd.isna(target_date) or (target_date - row.week_start).days != 7:
                    continue
                xs.append(x)
                ys.append(float(row.target_gt_t_plus_1))
                brands.append(brand)
                ends.append(str(row.week_start.date()))
        x_array = np.asarray(xs, dtype=np.float32).reshape((-1, length, len(features)))
        y_array = np.asarray(ys, dtype=np.float32)
        np.savez_compressed(
            dest / f"{name}.npz",
            X=x_array,
            y=y_array,
            label_available=np.isfinite(y_array),
            brand_id=np.asarray(brands),
            input_end_date=np.asarray(ends),
            feature_names=np.asarray(features),
        )

    # A model-neutral, unlabeled adapter remains useful when Trends labels have
    # not yet been supplied.
    all_x, all_brands, all_ends = [], [], []
    for brand, group in df.sort_values("week_start").groupby("brand_id"):
        group = group.sort_values("week_start").reset_index(drop=True)
        if len(group) < length:
            continue
        for i in range(length - 1, len(group)):
            window = group.iloc[i - length + 1:i + 1]
            if (window.week_start.diff().dropna().dt.days != 7).any():
                continue
            x = window[features].astype(float).to_numpy(np.float32)
            if np.isfinite(x).all():
                all_x.append(x)
                all_brands.append(brand)
                all_ends.append(str(window.week_start.iloc[-1].date()))
    np.savez_compressed(
        dest / "weekly_input_windows.npz",
        X=np.asarray(all_x, dtype=np.float32).reshape((-1, length, len(features))),
        brand_id=np.asarray(all_brands),
        input_end_date=np.asarray(all_ends),
        feature_names=np.asarray(features),
    )

    long = df.rename(columns={"brand_id": "item_id", "week_start": "timestamp"})
    long.to_parquet(dest / "forecast_long.parquet", index=False)
    print(dest, "channels", len(features))


if __name__ == "__main__":
    main()
