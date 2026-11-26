import pandas as pd

from brand_encoder.temporal import build_trend_features


def test_trend_target_uses_next_calendar_week() -> None:
    trends = pd.DataFrame({
        "brand_id": ["brand"] * 4,
        "date": pd.date_range("2025-01-06", periods=4, freq="W-MON"),
        "google_trends": [10, 20, 30, 40],
    })

    features = build_trend_features(trends, "Asia/Hong_Kong")
    first = features.iloc[0]

    assert first["google_trends"] == 10
    assert first["target_gt_t_plus_1"] == 20
    assert (first["target_date"] - first["week_start"]).days == 7
