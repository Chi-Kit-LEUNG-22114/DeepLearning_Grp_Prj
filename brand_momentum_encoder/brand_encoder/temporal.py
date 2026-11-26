from __future__ import annotations

import numpy as np
import pandas as pd

from .common import week_start


TREND_FEATURES = [
    "google_trends", "gt_lag_1", "gt_lag_2", "gt_lag_4", "gt_lag_8",
    "gt_lag_13", "gt_lag_26", "gt_lag_52", "gt_diff_1", "gt_diff_4",
    "gt_pct_4", "gt_mean_4", "gt_mean_13", "gt_std_4", "gt_std_13",
    "gt_slope_4", "gt_slope_13", "gt_acceleration",
]


def build_trend_features(trends: pd.DataFrame, timezone: str) -> pd.DataFrame:
    required = {"brand_id", "date", "google_trends"}
    if trends.empty or not required.issubset(trends.columns):
        return pd.DataFrame(columns=["brand_id", "week_start", *TREND_FEATURES, "target_gt_t_plus_1", "target_date"])
    frame = trends[["brand_id", "date", "google_trends"]].copy()
    frame["week_start"] = week_start(frame["date"], timezone)
    frame["google_trends"] = pd.to_numeric(frame["google_trends"], errors="coerce")
    frame = frame.groupby(["brand_id", "week_start"], as_index=False)["google_trends"].mean()
    frame["week_start"] = pd.to_datetime(frame["week_start"]).dt.tz_localize(None)

    complete = []
    for brand, group in frame.groupby("brand_id"):
        dates = pd.date_range(group.week_start.min(), group.week_start.max(), freq="W-MON")
        complete.append(pd.DataFrame({"brand_id": brand, "week_start": dates}))
    frame = pd.concat(complete, ignore_index=True).merge(frame, on=["brand_id", "week_start"], how="left")
    frame = frame.sort_values(["brand_id", "week_start"]).reset_index(drop=True)
    grouped = frame.groupby("brand_id")["google_trends"]
    for lag in [1, 2, 4, 8, 13, 26, 52]:
        frame[f"gt_lag_{lag}"] = grouped.shift(lag)
    frame["gt_diff_1"] = grouped.diff(1)
    frame["gt_diff_4"] = grouped.diff(4)
    frame["gt_pct_4"] = grouped.pct_change(4).replace([np.inf, -np.inf], np.nan)
    frame["gt_mean_4"] = grouped.transform(lambda s: s.rolling(4, min_periods=1).mean())
    frame["gt_mean_13"] = grouped.transform(lambda s: s.rolling(13, min_periods=1).mean())
    frame["gt_std_4"] = grouped.transform(lambda s: s.rolling(4, min_periods=2).std())
    frame["gt_std_13"] = grouped.transform(lambda s: s.rolling(13, min_periods=2).std())

    def trailing_slope(values: np.ndarray, window: int) -> float:
        if not np.isfinite(values).all():
            return np.nan
        return float(np.polyfit(np.arange(window), values, 1)[0])

    frame["gt_slope_4"] = grouped.transform(lambda s: s.rolling(4, min_periods=4).apply(lambda x: trailing_slope(x, 4), raw=True))
    frame["gt_slope_13"] = grouped.transform(lambda s: s.rolling(13, min_periods=13).apply(lambda x: trailing_slope(x, 13), raw=True))
    frame["gt_acceleration"] = frame.groupby("brand_id")["gt_diff_1"].diff()
    frame["target_gt_t_plus_1"] = grouped.shift(-1)
    frame["target_date"] = frame.groupby("brand_id")["week_start"].shift(-1)
    return frame
