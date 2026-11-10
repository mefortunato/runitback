"""NBA player values (work package N2): tuning and sanity checks for `runitback.nba.player_values`.

    python scripts/nba_player_values.py tune      # ridge penalty / window on the next season's possessions
    python scripts/nba_player_values.py report    # possession checks, top players, pv vs margin and spread

Reads data/nba/processed (`runitback nba players`, then `runitback nba player-values` builds
possessions.parquet) and data/nba/raw. Prints markdown; writes nothing to the repository.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from runitback.nba import player_values as pv

DATA = Path(os.environ.get("RUNITBACK_NBA_DATA", "data/nba"))


def load():
    proc = DATA / "processed"
    poss, pg = pv.franchise_only(pd.read_parquet(proc / "possessions.parquet"),
                                 pd.read_parquet(proc / "player_games.parquet"))
    return poss, pg, pd.read_parquet(proc / "players.parquet")


def tune(args):
    poss, _, players = load()
    ev = pv.events_from(poss, pv.player_groups(players))
    print("| lambda | window | decay | min minutes | " + " | ".join(f"{s} MSE gain / corr" for s in args.seasons)
          + " | mean MSE gain |")
    print("|---" * (5 + len(args.seasons)) + "|")
    for lam in args.lam:
        for window in args.window:
            for mm in args.min_minutes:
                cfg = pv.apm_config(lam=lam, window=window, decay=args.decay,
                                    min_poss=int(round(mm * pv.POSS_PER_MIN)))
                res = [pv.holdout_score(ev, s, cfg) for s in args.seasons]
                cells = " | ".join(f"{1e4 * r['mse_gain']:.1f} / {r['corr']:.4f}" for r in res)
                mean = np.mean([r["mse_gain"] for r in res])
                print(f"| {lam:g} | {window} | {args.decay} | {mm} | {cells} | {1e4 * mean:.2f} |", flush=True)
    print("\nMSE gain: out-of-sample reduction in squared error per possession vs a constant, x 1e-4.")


def report(args):
    poss, pg, players = load()
    # possessions vs the box score
    team_pts = pg.groupby(["game_id", "home"]).points.sum()
    pp = poss.groupby(["game_id", "off_home"]).points.sum()
    j = pd.concat([team_pts.rename("box"), pp.rename("poss")], axis=1, join="inner")
    per = poss.groupby(["season", "game_id", "off_home"]).size().groupby("season").mean()
    ppp = poss.groupby("season").points.mean()
    print("| season | possessions per team-game | points per possession |\n|---|---|---|")
    for s in per.index:
        print(f"| {s} | {per[s]:.1f} | {ppp[s]:.3f} |")
    err = (j.poss - j.box).abs()
    print(f"\nPossession points vs box-score team points: exact {100 * (err == 0).mean():.1f}%, "
          f"within 3 {100 * (err <= 3).mean():.1f}%, MAE {err.mean():.2f} ({len(j)} team-games)")

    gv = pv.game_table(poss, pg, players, args.first_season, args.jobs)
    latest = gv.latest.assign(name=lambda d: d.player_id.map(players.set_index("player_id").name))
    starter = gv.slots[max(gv.slots)][0]
    tot = latest.pivot_table(index=["player_id", "name", "group"], columns="side", values="value_poss").fillna(0)
    tot["pts"] = (tot["off"] + tot["def"]) * starter
    tot = tot.reset_index().sort_values("pts", ascending=False)
    print(f"\nLatest fit ({len(latest)} rows; a starter plays {starter:.0f} possessions a side). "
          "Points per game over replacement as a starter:\n")
    print("| player | group | offense | defense | total |\n|---|---|---|---|---|")
    for r in pd.concat([tot.head(args.top), tot.tail(5)]).itertuples():
        print(f"| {r.name} | {r.group} | {r.off * starter:+.1f} | {r._4 * starter:+.1f} | {r.pts:+.1f} |")

    # game level: projected (previous game's) rotation value difference vs margin and the market
    home = pg.drop_duplicates(["game_id", "team"])[["game_id", "team", "home", "season"]]
    t = gv.pv.merge(home, on=["game_id", "team"])
    t["net"] = t[[f"_pv_{i}" for i in range(6)]].sum(1)
    h, a = t[t.home].set_index("game_id"), t[~t.home].set_index("game_id")
    g = pd.DataFrame({"diff": h.net - a.net, "season": h.season}).dropna().astype({"season": int})
    margin = pg.groupby(["game_id", "home"]).points.sum().unstack()
    g["margin"] = (margin[True] - margin[False]).reindex(g.index)
    spreads = []
    for s in sorted(g.season.unique()):
        p = DATA / "raw" / f"pbp_{s}.parquet"
        if p.exists():
            d = pd.read_parquet(p, columns=["game_id", "home_team_spread"])
            spreads.append(d.groupby(d.game_id.astype(str)).home_team_spread.first())
    # ESPN's home_team_spread is the home team's expected margin (positive: home favored)
    g["market"] = pd.concat(spreads).reindex(g.index) if spreads else np.nan
    print("\n| season | games | corr(pv diff, margin) | corr(pv diff, market) | corr(market, margin) | "
          "margin pts per pv pt |\n|---|---|---|---|---|---|")
    for s, d in [*g.groupby("season"), ("all", g)]:
        m = d.dropna(subset=["market"])
        slope = np.polyfit(d["diff"], d.margin, 1)[0]
        print(f"| {s} | {len(d)} | {d['diff'].corr(d.margin):.3f} | {m['diff'].corr(m.market):.3f} | "
              f"{m.market.corr(m.margin):.3f} | {slope:.2f} |")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    t.add_argument("--lam", type=float, nargs="+", default=[1000, 3000, 10000, 30000])
    t.add_argument("--window", type=int, nargs="+", default=[pv.WINDOW])
    t.add_argument("--min-minutes", type=int, nargs="+", default=[pv.MIN_MINUTES])
    t.add_argument("--decay", type=float, default=pv.DECAY)
    r = sub.add_parser("report")
    r.add_argument("--first-season", type=int, default=2016)
    r.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    r.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    {"tune": tune, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
