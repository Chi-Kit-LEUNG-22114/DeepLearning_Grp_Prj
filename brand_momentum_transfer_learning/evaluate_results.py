#!/usr/bin/env python3
"""Generate reports from saved predictions, without training or downloading models."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brand_momentum_transfer.config import load_config, resolve_path  # noqa: E402
from brand_momentum_transfer.evaluation import compare_prediction_tables, export_evaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/default.yaml"))
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--output-root", help="Override the configured directory containing models/ and baselines/.")
    args = parser.parse_args()
    config, root = load_config(args.config)
    output = Path(args.output_root).expanduser().resolve() if args.output_root else resolve_path(root, config["paths"]["output"])
    paths = sorted(output.glob(f"models/*/{args.split}_predictions.csv"))
    paths += sorted(output.glob(f"baselines/*/{args.split}_predictions.csv"))
    if not paths:
        parser.error(f"No saved {args.split} predictions under {output}. This command never runs training.")
    tables = {f"{path.parent.parent.name}/{path.parent.name}": pd.read_csv(
        path, dtype={"brand_id": str}, keep_default_na=False
    ) for path in paths}
    compare_prediction_tables(tables)
    threshold = float(config.get("reporting", {}).get("change_threshold", 0.0))
    rows = []
    for name, table in tables.items():
        directory = output / name
        metrics = export_evaluation(table, directory, args.split, threshold)
        rows.append({"model": name, "split": args.split, **metrics})
    comparison = output / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(comparison / f"{args.split}_evaluation.csv", index=False)
    print(f"Evaluated {len(rows)} saved prediction tables. Reports: {comparison}")


if __name__ == "__main__":
    main()
