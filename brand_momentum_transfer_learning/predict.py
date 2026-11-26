#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brand_momentum_transfer.data import transform_feature_table  # noqa: E402
from brand_momentum_transfer.models import (  # noqa: E402
    Chronos2MultimodalRegressor,
    TTMMultimodalRegressor,
)
from brand_momentum_transfer.training import resolve_device  # noqa: E402
from brand_momentum_transfer.evaluation import build_watchlist  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict future BMI values with a saved TTM or Chronos-2 checkpoint."
    )
    parser.add_argument("--model-dir", required=True, help="Path to the checkpoint's best/ directory.")
    parser.add_argument(
        "--features",
        required=True,
        help="New reduced weekly feature Parquet/CSV table from brand_momentum_encoder.",
    )
    parser.add_argument("--output", required=True, help="Destination CSV for BMI predictions.")
    parser.add_argument("--brand-column", default="brand_id")
    parser.add_argument("--week-column", default="week_start")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--watchlist-output", help="Optional CSV of latest predicted ranks and changes.")
    parser.add_argument("--change-threshold", type=float, default=0.0,
                        help="Minimum absolute BMI score change for gaining/losing status.")
    args = parser.parse_args()
    output_path = Path(args.output).expanduser().resolve()
    watchlist_path = Path(args.watchlist_output).expanduser().resolve() if args.watchlist_output else None
    if watchlist_path == output_path:
        raise ValueError("Watchlist and predictions must use different output paths.")
    if not np.isfinite(args.change_threshold) or args.change_threshold < 0:
        raise ValueError("change-threshold must be finite and non-negative.")

    checkpoint_dir = Path(args.model_dir).expanduser().resolve()
    model_meta = json.loads(
        (checkpoint_dir / "model_config.json").read_text(encoding="utf-8")
    )
    preprocessor = json.loads(
        (checkpoint_dir / "preprocessing.json").read_text(encoding="utf-8")
    )
    model_name = model_meta["model_name"]
    feature_groups = model_meta["feature_groups_indices"]
    model_config = dict(model_meta["model_config"])
    model_config["context_length"] = int(model_meta["context_length"])
    model_config["backbone_config"] = model_meta["backbone_config"]
    if model_name == "ttm":
        model = TTMMultimodalRegressor(
            feature_groups, model_config, dropout=float(model_meta["dropout"])
        )
    elif model_name == "chronos2":
        model = Chronos2MultimodalRegressor(
            feature_groups, model_config, dropout=float(model_meta["dropout"])
        )
    else:
        raise ValueError(f"Unsupported saved model: {model_name}")

    device = resolve_device(args.device)
    model.load_state_dict(
        load_file(str(checkpoint_dir / "model.safetensors"), device="cpu")
    )
    model.to(device)
    model.eval()

    feature_path = Path(args.features).expanduser().resolve()
    if feature_path.suffix.lower() in {".parquet", ".pq"}:
        feature_df = pd.read_parquet(feature_path)
    elif feature_path.suffix.lower() == ".csv":
        feature_df = pd.read_csv(feature_path)
    else:
        raise ValueError("Prediction input must be Parquet or CSV.")
    context_length = int(model_meta["context_length"])
    x, brands, input_dates = transform_feature_table(
        feature_df=feature_df,
        preprocessor=preprocessor,
        brand_col=args.brand_column,
        week_col=args.week_column,
        context_length=context_length,
    )

    batch_size = 2 if model_name == "chronos2" else 16
    output_scaled: list[float] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start:start + batch_size]).to(device)
            output_scaled.extend(
                model(batch).detach().cpu().numpy().reshape(-1).tolist()
            )
    target_scaler = model_meta["target_scaler"]
    predictions = (
        np.asarray(output_scaled, dtype=np.float64) * float(target_scaler["scale"])
        + float(target_scaler["mean"])
    )
    result = pd.DataFrame(
        {
            "brand_id": brands,
            "input_week_start": input_dates,
            "bmi_predicted": predictions,
            "model": model_name,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    if watchlist_path is not None:
        watchlist = build_watchlist(result, args.change_threshold)
        watchlist_path.parent.mkdir(parents=True, exist_ok=True)
        watchlist.to_csv(watchlist_path, index=False)
    print(f"Wrote {len(result)} BMI predictions to {output_path}")


if __name__ == "__main__":
    main()
