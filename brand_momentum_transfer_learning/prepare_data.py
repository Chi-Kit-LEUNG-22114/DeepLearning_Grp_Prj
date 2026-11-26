#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brand_momentum_transfer.config import load_config  # noqa: E402
from brand_momentum_transfer.training import prepare_shared_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one shared BMI dataset and chronological split for both models."
    )
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    config, project_root = load_config(PROJECT_ROOT / args.config)
    output = prepare_shared_dataset(config, project_root)
    print(f"Shared dataset and split artifacts written to: {output / 'shared'}")


if __name__ == "__main__":
    main()
