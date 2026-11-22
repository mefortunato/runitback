"""Backtest one configuration of a wager experiment on one season, with runitback.wager or sportsml.

The GitHub Actions experiments run one job per (configuration, season); this is that job. Same
result as `runitback-wager <sport> ablate` for that cell: <output-dir>/<name>/<season>.csv (skipped
when it exists). With `--package sportsml` the cell runs on mefortunato/sportsml's own code (the
package the wager model was ported from) and is written as `sportsml_<config>`, so a port can be
scored against its source on the same games:

    python scripts/wager_arm.py prod 2023 --sport cfb --experiment experiments/cfb/baseline.yaml
    python scripts/wager_arm.py prod 2023 --sport cfb --experiment experiments/cfb/baseline.yaml \
        --package sportsml --games data/cfb/games_sportsml.csv

An ensemble cell (``fit: {ensemble: N}``) can also train each member in a job of its own and then
assemble them (runitback.wager.web.split.arm_member / arm_season; the same CSV as the one-job cell):

    python scripts/wager_arm.py prod_e4 2025 --sport wnba --experiment experiments/wnba/wager.yaml \
        --member 2 --members-dir members          # members/member2.ckpt
    python scripts/wager_arm.py prod_e4 2025 --sport wnba --experiment experiments/wnba/wager.yaml \
        --members members                         # assemble members/member*.ckpt, replay the season
    python scripts/wager_arm.py prod_e4 2025 --sport wnba --experiment experiments/wnba/wager.yaml --size
                                                  # prints the number of members (for a job matrix)

Only needs PyYAML and the chosen package, so it runs where runitback is not installed (not the member modes).
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

import yaml

PACKAGES = {"runitback": "runitback.wager", "sportsml": "sportsml"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config")
    ap.add_argument("season", type=int)
    ap.add_argument("--sport", required=True)
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--package", choices=sorted(PACKAGES), default="runitback")
    ap.add_argument("--games", help="processed games table (default: data/<sport>/games.csv)")
    ap.add_argument("--output-dir", help="default: the experiment file's path without .yaml")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--member", type=int, help="train only this member of the cell's ensemble (to --members-dir)")
    ap.add_argument("--members-dir", type=Path, default=Path("members"))
    ap.add_argument("--members", type=Path, help="assemble the members in this directory and replay the season")
    ap.add_argument("--size", action="store_true", help="print the cell's ensemble size and exit")
    a = ap.parse_args()
    configs = yaml.safe_load(a.experiment.read_text())["configs"]
    if a.config not in configs:
        raise SystemExit(f"{a.config} is not in {a.experiment}: {', '.join(configs)}")
    name = a.config if a.package == "runitback" else f"{a.package}_{a.config}"
    games = a.games or f"data/{a.sport}/games.csv"
    output_dir = a.output_dir or str(a.experiment.with_suffix(""))
    if a.size or a.member is not None or a.members is not None:
        if a.package != "runitback":
            raise SystemExit("--size, --member and --members are runitback's only")
        import torch

        from runitback.wager.web import split

        config = configs[a.config]
        if a.size:
            print(split.arm_size(config, a.sport))
            return
        torch.set_num_threads(a.threads)
        if a.member is not None:
            print(split.arm_member(games, a.sport, config, a.season, a.member, a.members_dir))
            return
        out = Path(output_dir) / name / f"{a.season}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        split.arm_season(games, a.sport, config, a.season, a.members).to_csv(out, index=False)
        print(out)
        return
    experiments = importlib.import_module(f"{PACKAGES[a.package]}.experiments")
    experiments.run_experiments(games, a.sport, {name: configs[a.config]}, [a.season], output_dir,
                                n_jobs=1, threads=a.threads)
    print(Path(output_dir) / name / f"{a.season}.csv")


if __name__ == "__main__":
    main()
