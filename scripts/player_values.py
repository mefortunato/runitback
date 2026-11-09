"""Leak-free weekly player values and per-game lineup values (runitback.nfl.player_values).

The experiment's copy of the production table (``runitback nfl player-values``): fits player values
every week from 2016 on, on the plays before that week, and sums each team's projected pre-game
lineup per position group. Writes <out>/pv.parquet (the wager model's ``_pv_*`` columns, see
``player_values.game_table``), lineups.parquet / slots.json (the lineup editor's) and
values_latest.csv (the most recent fit).

With ``--season-values``, also writes, for each listed season S, ``values_<S>.csv`` (every
player's value from the fit on the plays before S: side, player_id, group, plays, value_play) and
``swap_refs_<S>.csv`` (each group's best and worst regular by points as a starter, for the
position-swap and lineup-sensitivity checks). A regular is a player in one of his group's starting
slots (:data:`STARTERS`, by snaps) in at least ``REGULAR_GAMES`` games of season S-1, with a value of
his own (enough plays for his own coefficient, or a kicker / punter with attempts).

    python scripts/player_values.py --out experiments/nfl/playervalues
    python scripts/player_values.py --out experiments/nfl/playervalues --download --season-values 2021 2022 2023 2024 2025
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from runitback.nfl import player_values as pv

STARTERS = {"QB": 1, "RB": 1, "WR": 3, "TE": 1, "OL": 5, "DL": 4, "LB": 3, "DB": 4, "K": 1, "P": 1}
REGULAR_GAMES = 8


def season_values(data_dir: Path, raw_dir: Path, seasons: list[int], out: Path) -> None:
    """values_<S>.csv and swap_refs_<S>.csv for each season S (see the module docstring)."""
    players = pd.read_parquet(data_dir / "players.parquet")
    games = pd.read_parquet(data_dir / "games.parquet")
    pg = pd.read_parquet(data_dir / "player_games.parquet")
    data = pv.play_table(data_dir, raw_dir, dict(zip(players.player_id, players.group)))
    raw_seasons = sorted(s for s in set(games.season) if (raw_dir / f"pbp_{s}.parquet").exists())
    special = pv.special_table(raw_dir, raw_seasons, games[["game_id", "season", "week"]])
    slots = pv.slot_multipliers(pv.slot_plays(pg[pg.season < 2016]))
    names = players.set_index("player_id").name
    out.mkdir(parents=True, exist_ok=True)
    for s in seasons:
        table = pv.fit_values(data, special, s * 100 + 1).table
        table.to_csv(out / f"values_{s}.csv", index=False, float_format="%.5f")
        prev = pg[(pg.season == s - 1) & (pg.depth < pg.group.map(STARTERS).fillna(0))]
        starts = prev.groupby(["group", "player_id"]).size()
        rows = []
        for g in STARTERS:
            side = "off" if g in pv.OFF_GROUPS else "def" if g in pv.DEF_GROUPS else g
            regular = set(starts.loc[g][starts.loc[g] >= REGULAR_GAMES].index) if g in starts.index else set()
            t = table[(table.side == side) & (table.group == g) & table.player_id.isin(regular)]
            if g in pv.OFF_GROUPS + pv.DEF_GROUPS:
                t = t[t.plays >= pv.MIN_PLAYS]
            if t.empty:
                continue
            t = t.assign(pts_as_starter=(t.value_play * slots[g][0]).round(2), n_regulars=len(t))
            t = t.sort_values("pts_as_starter")
            for kind, r in (("best", t.iloc[-1]), ("worst", t.iloc[0])):
                rows.append({"group": g, "kind": kind, "player_id": r.player_id, "name": names.get(r.player_id),
                             "plays": round(float(r.plays)), "pts_as_starter": r.pts_as_starter,
                             "n_regulars": r.n_regulars})
        pd.DataFrame(rows).to_csv(out / f"swap_refs_{s}.csv", index=False)
        print(f"season {s}: {len(table)} player values, {len(rows)} swap references -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data/nfl/processed"))
    ap.add_argument("--raw", type=Path, default=Path("data/nfl/raw"))
    ap.add_argument("--out", type=Path, default=Path("experiments/nfl/playervalues"))
    ap.add_argument("--first-season", type=int, default=2016)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--download", action="store_true",
                    help="fetch the depth charts, injury reports and rosters first (needed on a fresh checkout)")
    ap.add_argument("--season-values", type=int, nargs="*", default=[],
                    help="also write values_<S>.csv and swap_refs_<S>.csv for these seasons")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    gv = pv.game_table(a.data, a.raw, a.first_season, a.jobs, download=a.download)
    pv.write_game_table(gv, a.out, pd.read_parquet(a.data / "players.parquet"))
    print(f"{len(gv.pv)} team-games -> {a.out / 'pv.parquet'} (latest fit: before {gv.latest.before})")
    if a.season_values:
        season_values(a.data, a.raw, a.season_values, a.out)


if __name__ == "__main__":
    main()
