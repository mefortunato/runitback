"""Leak-free weekly player values and per-game lineup values (runitback.nfl.player_values).

The experiment's copy of the production table (``runitback nfl player-values``): fits player values
every week from 2016 on, on the plays before that week, and sums each team's projected pre-game
lineup per position group. Writes <out>/pv.parquet (the wager model's ``_pv_*`` columns, see
``player_values.game_table``), lineups.parquet / slots.json (the lineup editor's) and
values_latest.csv (the most recent fit).

    python scripts/player_values.py --out experiments/nfl/playervalues
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from runitback.nfl import player_values as pv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data/nfl/processed"))
    ap.add_argument("--raw", type=Path, default=Path("data/nfl/raw"))
    ap.add_argument("--out", type=Path, default=Path("experiments/nfl/playervalues"))
    ap.add_argument("--first-season", type=int, default=2016)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    gv = pv.game_table(a.data, a.raw, a.first_season, a.jobs)
    pv.write_game_table(gv, a.out, pd.read_parquet(a.data / "players.parquet"))
    print(f"{len(gv.pv)} team-games -> {a.out / 'pv.parquet'} (latest fit: before {gv.latest.before})")


if __name__ == "__main__":
    main()
