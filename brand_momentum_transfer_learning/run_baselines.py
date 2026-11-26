#!/usr/bin/env python3
"""Fit small reference models on an existing shared dataset; no neural training."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brand_momentum_transfer.baselines import run_baselines  # noqa: E402
from brand_momentum_transfer.config import load_config, resolve_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/default.yaml"))
    args = parser.parse_args()
    config, root = load_config(args.config)
    results = run_baselines(resolve_path(root, config["paths"]["output"]), config)
    for result in results:
        print(f"{result['model']}: test MAE={result['test']['mae']:.4f}")


if __name__ == "__main__":
    main()
