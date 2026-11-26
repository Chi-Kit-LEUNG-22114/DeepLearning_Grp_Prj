#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brand_momentum_transfer.config import load_config  # noqa: E402
from brand_momentum_transfer.training import train_comparison  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train TTM and Chronos-2 using the same saved 70:10:20 split."
    )
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    config, project_root = load_config(PROJECT_ROOT / args.config)
    results = train_comparison(config, project_root)
    for result in results:
        print(
            f"{result['model']}: "
            f"validation MAE={result['validation']['mae']:.4f}, "
            f"test MAE={result['test']['mae']:.4f}, "
            f"best_epoch={result['best_epoch']}"
        )


if __name__ == "__main__":
    main()
