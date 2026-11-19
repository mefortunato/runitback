"""Fail fast if the college wager experiment's per-team-game tables do not join the games table.

A table that keys games or teams differently makes every row "missing" with no error (N5's NBA ids did):
the model then trains on nothing. This joins the games table (``data/<gender>/games.csv``) to the
foundation fold's ``pbp_32.parquet`` and to the player-value tables the way the wager model does
(``runitback.wager.models.transformer.data``) and prints, per season, the share of Division I games
(neither team is the pooled ``Non-D1``) that found their row. Exits 1 below ``--min`` anywhere.

    python scripts/college_join_check.py cbb --pbp runs/foundation/fold2026/pbp_32.parquet \
        --pv player_values/cbb/pv.parquet --pv player_values/cbb_prev/pv.parquet
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

NON_D1 = "Non-D1"


def coverage(games: pd.DataFrame, schema, pbp: list[str], pv: list[str], gender: str, seasons: list[int]):
    """``{table: {season: share of Division I games whose team and opponent rows were found}}``."""
    from runitback.wager.models.transformer import data as tdata

    d1 = games[(games[schema.team] != NON_D1) & (games[schema.opp] != NON_D1)
               & games[schema.season].isin(seasons)]
    out = {}
    for path in pbp:
        miss = tdata.pbp_features(d1, schema, path, 32)[tdata.PBP_MISSING]
        out[path] = (1 - miss).groupby(d1[schema.season]).mean().to_dict()
    for path in pv:
        lu = tdata.lineup_features(d1, schema, path, "pvc", gender)
        found = 1 - lu[[tdata.LINEUP_MISSING, f"_opp{tdata.LINEUP_MISSING}"]].max(axis=1)
        out[path] = found.groupby(d1[schema.season]).mean().to_dict()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gender", choices=["cbb", "wcbb"])
    ap.add_argument("--games", help="default: data/<gender>/games.csv")
    ap.add_argument("--pbp", action="append", default=[], help="embedding table(s)")
    ap.add_argument("--pv", action="append", default=[], help="player-value table(s)")
    ap.add_argument("--seasons", type=int, nargs="*", default=[2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--min", type=float, default=0.8, help="lowest share of games joined (default 0.8)")
    a = ap.parse_args()
    from runitback.wager.core.io import read_table
    from runitback.wager.core.schema import played
    from runitback.wager.core.sport import get_sport

    schema = get_sport(a.gender).schema
    games = played(read_table(a.games or f"data/{a.gender}/games.csv"), schema)
    cov = coverage(games, schema, a.pbp, a.pv, a.gender, a.seasons)
    bad = False
    for path, by in cov.items():
        print(path, {int(s): round(float(v), 3) for s, v in sorted(by.items())})
        bad |= any(v < a.min for v in by.values()) or not by
    if bad:
        sys.exit(f"::error::a table joins fewer than {a.min:.0%} of a season's Division I games")


if __name__ == "__main__":
    main()
