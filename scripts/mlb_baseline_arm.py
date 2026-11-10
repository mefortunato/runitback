"""Backtest one configuration of the MLB baseline experiment on one season.

Writes <experiment dir>/<config>/<season>.csv (see runitback.wager.experiments.run_experiments).
The fit runs in a spawned subprocess, which re-imports this file: it must be a file, not stdin.

    python scripts/mlb_baseline_arm.py prod 2024 --experiment experiments/mlb/baseline.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from runitback.wager.experiments import run_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config")
    parser.add_argument("season", type=int)
    parser.add_argument("--experiment", default="experiments/mlb/baseline.yaml")
    parser.add_argument("--games", default="data/mlb/games.csv")
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.experiment))
    out_dir = Path(args.experiment).with_suffix("")
    run_experiments(args.games, "mlb", {args.config: cfg["configs"][args.config]}, [args.season],
                    str(out_dir), n_jobs=1, threads=args.threads)


if __name__ == "__main__":
    main()
