"""Backtest one configuration of experiments/nfl/foundation.yaml on one season.

The GitHub Actions experiment (.github/workflows/foundation-experiment.yml) runs one job per
(configuration, season); this is that job. Same result as `runitback-wager nfl ablate` for that
cell: experiments/nfl/foundation/<config>/<season>.csv (skipped when it exists).

    python scripts/foundation_arm.py pbp32 2023 --threads 4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from runitback.wager.experiments import run_experiments


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config")
    ap.add_argument("season", type=int)
    ap.add_argument("--experiment", type=Path, default=Path("experiments/nfl/foundation.yaml"))
    ap.add_argument("--games", default="data/nfl/games.csv")
    ap.add_argument("--output-dir", default="experiments/nfl/foundation")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    configs = yaml.safe_load(a.experiment.read_text())["configs"]
    if a.config not in configs:
        raise SystemExit(f"{a.config} is not in {a.experiment}: {', '.join(configs)}")
    run_experiments(a.games, "nfl", {a.config: configs[a.config]}, [a.season], a.output_dir,
                    n_jobs=1, threads=a.threads)
    print(Path(a.output_dir) / a.config / f"{a.season}.csv")


if __name__ == "__main__":
    main()
