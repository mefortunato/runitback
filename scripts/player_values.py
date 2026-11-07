"""Leak-free weekly player values and per-game lineup values (runitback.nfl.player_values).

For every week from 2016 on, fits player values on the plays before that week, then sums each
team's projected pre-game lineup (published depth chart minus players ruled out, as the lineup
features use) and its previous game's actual lineup. Writes <out>/pv.parquet, one row per
(game_id, team) with the wager model's ``_pv_*`` columns, in points per game over replacement:

    _pv_0 .. _pv_8  QB, RB, WR, TE, OL, DL, LB, DB, kicker + punter (each group's sum)
    _pv_9  offense change vs the previous game's lineup   _pv_10 defense change

and <out>/values_latest.csv, the most recent fit (every player's value per play and as a starter).

    python scripts/player_values.py --out experiments/nfl/playervalues
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from runitback.nfl import player_values as pv

log = logging.getLogger("player_values")


def _fit(data, special, before):
    return pv.fit_values(data, special, before)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data/nfl/processed"))
    ap.add_argument("--raw", type=Path, default=Path("data/nfl/raw"))
    ap.add_argument("--out", type=Path, default=Path("experiments/nfl/playervalues"))
    ap.add_argument("--first-season", type=int, default=2016)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from runitback.nfl.lineup_features import load_availability, load_depth_charts, projected_lineups

    players = pd.read_parquet(a.data / "players.parquet")
    group_of = dict(zip(players.player_id, players.group))
    games = pd.read_parquet(a.data / "games.parquet")
    pg = pd.read_parquet(a.data / "player_games.parquet")
    data = pv.play_table(a.data, a.raw, group_of)
    seasons = sorted(set(games.season))
    special = pv.special_table(a.raw, [s for s in seasons if (a.raw / f"pbp_{s}.parquet").exists()],
                               games[["game_id", "season", "week"]])
    plays_at = pv.slot_plays(pg[pg.season < 2016])   # snap shares by slot (before every fit; stable)
    target = games[games.season >= a.first_season]
    weeks = sorted({s * 100 + w for s, w in zip(target.season, target.week)})
    weeks = [t for t in weeks if (data.plays.t < t).any()]
    log.info("%d plays; fitting %d weeks", len(data.plays), len(weeks))
    fits = Parallel(n_jobs=a.jobs)(delayed(_fit)(data, special, t) for t in weeks)
    vals = {f.before: {(s, p): v for s, p, v in zip(f.table.side, f.table.player_id, f.table.value_play)}
            for f in fits}

    GRP = list(pv.OFF_GROUPS + pv.DEF_GROUPS) + ["ST"]

    def side_values(rows, vp):
        out = dict.fromkeys(GRP, 0.0)
        for g, k, pid in rows:
            if g in pv.OFF_GROUPS:
                out[g] += vp.get(("off", pid), 0.0) * plays_at.get((g, k), 0.0)
            elif g in pv.DEF_GROUPS:
                out[g] += vp.get(("def", pid), 0.0) * plays_at.get((g, k), 0.0)
            elif g in ("K", "P") and k == 0:
                out["ST"] += vp.get((g, pid), 0.0) * (pv.FG_PER_GAME if g == "K" else pv.PUNTS_PER_GAME)
        return out

    off_sum = lambda v: sum(v[g] for g in pv.OFF_GROUPS)
    def_sum = lambda v: sum(v[g] for g in pv.DEF_GROUPS)

    out_players, active = load_availability(a.raw, sorted(set(target.season)), download=False)
    charts = load_depth_charts(a.raw, sorted(set(target.season)), download=False)
    proj = projected_lineups(pg, target, out_players, active, charts)
    proj_rows = {k: list(zip(d.group, d["rank"].astype(int), d.player_id)) for k, d in proj.groupby(["game_id", "team"])}
    order = pg[["game_id", "team", "season", "week"]].drop_duplicates().sort_values(["team", "season", "week"])
    prev_of = {}
    for team, d in order.groupby("team"):
        ids = d.game_id.tolist()
        prev_of |= {(g, team): p for p, g in zip([None] + ids[:-1], ids)}
    actual = {k: list(zip(d.group, d.depth.astype(int), d.player_id)) for k, d in pg.groupby(["game_id", "team"])}
    rows = []
    for g in target.itertuples():
        t = g.season * 100 + g.week
        vp = vals.get(t)
        if vp is None:
            continue
        for team in (g.home, g.away):
            pr = proj_rows.get((g.game_id, team))
            if pr is None:
                continue
            cur = side_values(pr, vp)
            prev = prev_of.get((g.game_id, team))
            pv_prev = side_values(actual[(prev, team)], vp) if prev and (prev, team) in actual else None
            row = {"game_id": g.game_id, "team": team} | {f"_pv_{i}": cur[k] for i, k in enumerate(GRP)}
            row["_pv_9"] = off_sum(cur) - off_sum(pv_prev) if pv_prev else 0.0
            row["_pv_10"] = def_sum(cur) - def_sum(pv_prev) if pv_prev else 0.0
            rows.append(row)
    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df[[c for c in df if c.startswith("_pv_")]] = df[[c for c in df if c.startswith("_pv_")]].astype(np.float32)
    df.to_parquet(a.out / "pv.parquet", index=False)
    last = fits[-1].table.copy()
    names = players.set_index("player_id").name
    last["name"] = last.player_id.map(names)
    starter = {g: plays_at.get((g, 0), 0.0) for g in pv.OFF_GROUPS + pv.DEF_GROUPS} | \
        {"K": pv.FG_PER_GAME, "P": pv.PUNTS_PER_GAME}
    last["pts_as_starter"] = last.value_play * last.group.map(starter).fillna(0.0)
    last.sort_values(["group", "pts_as_starter"], ascending=[True, False]).to_csv(
        a.out / "values_latest.csv", index=False, float_format="%.4f")
    log.info("%d team-games -> %s (latest fit: before %d)", len(df), a.out / "pv.parquet", fits[-1].before)


if __name__ == "__main__":
    main()
