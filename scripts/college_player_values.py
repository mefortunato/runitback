"""College basketball player values and rotations (work package C3): checks for the write-up.

    python scripts/college_player_values.py tune --gender cbb       # box-value settings, next-season fit
    python scripts/college_player_values.py report --gender cbb     # rotation check, values vs margin, top players

Reads data/<gender>/processed (`runitback college players` / `lineups`) and player_values/<gender>
(`runitback college player-values`). Prints markdown; writes nothing to the repository.
"""
from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd

from runitback.college import box_values as B
from runitback.college import lineups as L


def _dir(gender: str) -> Path:
    return Path(os.environ.get(f"RUNITBACK_{gender.upper()}_DATA", f"data/{gender}"))


def _table(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def tune(args):
    proc = _dir(args.gender) / "processed"
    pg, tg = pd.read_parquet(proc / "player_games.parquet"), pd.read_parquet(proc / "team_games.parquet")
    pg, tg = pg[pg.season >= args.since], tg[tg.season >= args.since]
    rows = []
    for w, pm, dec, mm in itertools.product(args.window, args.pseudo, args.decay, args.min_minutes):
        m = B.BoxModel.build(pg, tg, B.BoxConfig(window=w, pseudo_minutes=pm, decay=dec, min_minutes=mm))
        r = [B.holdout_score(m, s) for s in args.seasons]
        rows.append({"window": w, "pseudo min": pm, "decay": dec, "repl min": mm,
                     **{f"{x['season']} R²": round(x["r2"], 4) for x in r},
                     "mean R²": round(np.mean([x["r2"] for x in r]), 4),
                     "mean corr": round(np.mean([x["corr"] for x in r]), 4),
                     "slope": round(np.mean([x["slope"] for x in r]), 2)})
        print(rows[-1], flush=True)
    print(_table(pd.DataFrame(rows)))
    print("\nR²: team-game points per 100 possessions predicted from the players' values at the season's "
          "start (weighted by the minutes they played), vs a constant.")


def change_check(pv: pd.DataFrame, pg: pd.DataFrame, recent: int = 8) -> pd.DataFrame:
    """Does the lineup's change vs the team's recent rotations (``_pv_8 .. _pv_13``, the ``pvc``
    inputs) predict the margin beyond the team's recent results? Per season group, the least-squares
    fit of the final margin on home court, the two teams' mean margins over their previous ``recent``
    games, and the change difference (home - away): the change's coefficient and its standard error,
    overall and on the games with the biggest changes."""
    tm = pg.drop_duplicates(["game_id", "team_id"])[["game_id", "team_id", "game_date", "home", "team_score",
                                                     "opp_score"]].copy()
    tm["margin"] = tm.team_score - tm.opp_score
    tm = tm.sort_values(["team_id", "game_date", "game_id"], kind="stable")
    tm["form"] = tm.groupby("team_id").margin.transform(lambda s: s.shift(1).rolling(recent, min_periods=3).mean())
    chg = [f"_pv_{i}" for i in range(8, 14)]
    m = pv.assign(chg=pv[chg].sum(axis=1)).merge(tm, on=["game_id", "team_id"]).dropna(subset=["form"])
    j = m[m.home].merge(m[~m.home][["game_id", "form", "chg"]], on="game_id", suffixes=("", "_a"))
    j["dform"], j["dchg"] = j.form - j.form_a, j.chg - j.chg_a
    rows = []
    big = j.dchg.abs() >= j.dchg.abs().quantile(0.9)
    for name, d in [("all", j), ("top-decile change", j[big]), ("2022-2026", j[j.season.between(2022, 2026)])]:
        X = np.column_stack([np.ones(len(d)), d.dform, d.dchg])
        b, *_ = np.linalg.lstsq(X, d.margin.to_numpy(), rcond=None)
        sigma2 = ((d.margin.to_numpy() - X @ b) ** 2).sum() / (len(d) - 3)
        se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
        rows.append({"games": name, "n": len(d), "margin per pt of recent form": round(b[1], 3),
                     "margin per pt of lineup change": round(b[2], 2), "± 2 se": round(2 * se[2], 2)})
    return pd.DataFrame(rows)


def report(args):
    proc = _dir(args.gender) / "processed"
    out = Path(args.values or f"player_values/{args.gender}")
    pg = pd.read_parquet(proc / "player_games.parquet")
    print("## Rotation projection vs who played\n")
    for rule in L.RULES:
        path = proc / ("projected_lineups.parquet" if rule == "dnp" else f"projected_lineups_{rule}.parquet")
        proj = pd.read_parquet(path) if path.exists() else L.projected_lineups(pg, rule)
        ev = L.evaluate(proj, pg)
        ev = ev[ev.season >= args.first]
        print(f"\n`rule={rule}`\n")
        print(_table(ev.round(3)))

    pv = pd.read_parquet(out / "pv.parquet")
    tm = pg.drop_duplicates(["game_id", "team_id"])[["game_id", "team_id", "home", "team_score", "opp_score"]]
    lv = [f"_pv_{i}" for i in range(6)]
    pv["lvl"] = pv[lv].sum(axis=1)
    m = pv.merge(tm, on=["game_id", "team_id"])
    j = m[m.home].merge(m[~m.home][["game_id", "lvl", "_pv_6", "_pv_7"]], on="game_id", suffixes=("", "_a"))
    j["d"], j["margin"] = j.lvl - j.lvl_a, j.team_score - j.opp_score
    rows = []
    for s, d in [*j.groupby("season"), ("all", j)]:
        rows.append({"season": s, "games": len(d), "corr with margin": round(np.corrcoef(d.d, d.margin)[0, 1], 3),
                     "margin pts per value pt": round(np.polyfit(d.d, d.margin, 1)[0], 2),
                     "sd of difference": round(float(d.d.std()), 1)})
    print("\n## Lineup value difference (home - away) vs the final margin\n")
    print(_table(pd.DataFrame(rows)))
    print("\n" + _table(change_check(pv, pg)))

    v = pd.read_csv(out / "values_latest.csv")
    recent = set(pg.loc[pg.season >= pg.season.max() - 1, "player_id"])
    w = v.pivot_table(index=["player_id", "name", "team", "group"], columns="side", values="pts_per_40").reset_index()
    w["minutes"] = w.player_id.map(v.groupby("player_id").minutes.first())
    w = w[w.player_id.isin(recent) & (w.minutes >= args.min_minutes)]
    w["total"] = w["off"] + w["def"]
    w = w.sort_values("total", ascending=False)
    cols = ["name", "team", "group", "off", "def", "total"]
    print(f"\n## Top and bottom of the latest fit (points per 40 minutes over replacement, "
          f"{args.min_minutes}+ minutes in the window, played in the last two seasons)\n")
    print(_table(pd.concat([w.head(args.top), w.tail(3)])[cols].round(1)))
    print("\n" + _table(w.groupby("group").total.describe()[["count", "mean", "50%", "max"]].round(2).reset_index()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--gender", default="cbb")
    t.add_argument("--since", type=int, default=2014)
    t.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023, 2024, 2025, 2026])
    t.add_argument("--window", type=int, nargs="+", default=[2, 3])
    t.add_argument("--pseudo", type=float, nargs="+", default=[100, 300])
    t.add_argument("--decay", type=float, nargs="+", default=[0.7, 0.85])
    t.add_argument("--min-minutes", type=float, nargs="+", default=[150])
    r = sub.add_parser("report")
    r.add_argument("--gender", default="cbb")
    r.add_argument("--values", default=None)
    r.add_argument("--first", type=int, default=2022)
    r.add_argument("--min-minutes", type=float, default=600)
    r.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    {"tune": tune, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
