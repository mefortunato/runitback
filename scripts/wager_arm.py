"""Backtest one configuration of an experiment's YAML on one season, for any sport.

One GitHub Actions matrix job per (configuration, season) runs this; it does what
`runitback-wager <sport> ablate` does for that cell and writes
<output-dir>/<config>/<season>.csv (skipped when it exists). With `--package sportsml` the same
cell runs with sportsml's code instead (`mefortunato/sportsml` installed in the environment), so
the two implementations can be compared on the same games table (experiments/<sport>/port.yaml).

    python scripts/wager_arm.py nba prod 2025 --experiment experiments/nba/port.yaml
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sport")
    ap.add_argument("config")
    ap.add_argument("season", type=int)
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--games", help="default: data/<sport>/games.csv")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--package", default="runitback.wager", choices=["runitback.wager", "sportsml"])
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    configs = yaml.safe_load(a.experiment.read_text())["configs"]
    if a.config not in configs:
        raise SystemExit(f"{a.config} is not in {a.experiment}: {', '.join(configs)}")
    run_experiments = importlib.import_module(f"{a.package}.experiments").run_experiments
    run_experiments(a.games or f"data/{a.sport}/games.csv", a.sport, {a.config: configs[a.config]},
                    [a.season], a.output_dir, n_jobs=1, threads=a.threads)
    print(Path(a.output_dir) / a.config / f"{a.season}.csv")


if __name__ == "__main__":
    main()
