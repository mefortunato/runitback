"""MLB player values (work package M2): tuning and sanity checks for `runitback.mlb.player_values`.

    python scripts/mlb_player_values.py tune      # ridge penalty / window / threshold on the next season's PAs
    python scripts/mlb_player_values.py report    # top players, pv vs the game's margin, total and the market

Reads data/mlb/processed (`runitback mlb process`) and, for `report`, player_values/mlb/pv.parquet
(`runitback mlb player-values`) and optionally the wager games table (data/mlb/raw.csv, from
`runitback-wager mlb download`) for the market. Prints markdown; writes nothing to the repository.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from runitback.mlb import paths
from runitback.mlb import player_values as pv


def tune(args):
    tb = pv.load(paths.processed_dir(), list(range(args.start, max(args.seasons) + 1)))
    evs = {f: pv.events_from(tb.pa, tb.players, fielding=f) for f in set(args.fielding)}
    print("| lambda | window | decay | min PA | fielding | P scale | "
          + " | ".join(f"{s} MSE gain / corr" for s in args.seasons) + " | mean MSE gain |")
    print("|---" * (7 + len(args.seasons)) + "|")
    for fielding in args.fielding:
        for lam in args.lam:
            for window in args.window:
                for mp in args.min_pa:
                    for ps in args.p_scale:
                        cfg = pv.apm_config(lam=lam, window=window, decay=args.decay, min_pa=mp, fielding=fielding,
                                            group_scale={"P": ps})
                        res = [pv.holdout_score(evs[fielding], s, cfg) for s in args.seasons]
                        cells = " | ".join(f"{1e4 * r['mse_gain']:.2f} / {r['corr']:.4f}" for r in res)
                        mean = np.mean([r["mse_gain"] for r in res])
                        print(f"| {lam:g} | {window} | {args.decay} | {mp} | {fielding} | {ps:g} | {cells} | "
                              f"{1e4 * mean:.3f} |", flush=True)
    print("\nMSE gain: out-of-sample reduction in squared error per PA vs a constant, x 1e-4.")


def report(args):
    proc = paths.processed_dir()
    tb = pv.load(proc, list(range(args.start, args.end + 1)))
    out = Path(args.values)
    table = pd.read_parquet(out / "pv.parquet")
    latest = pd.read_csv(out / "values_latest.csv")

    pa = tb.pa
    print("| season | PAs | run value mean | run value sd |\n|---|---|---|---|")
    for s, d in pa.groupby("season"):
        print(f"| {s} | {len(d)} | {d.run_value.mean():+.4f} | {d.run_value.std():.3f} |")

    for side, label, unit in (("bat", "Batters", "per 600 PA"), ("pit", "Pitchers", "per 700 BF")):
        d = latest[latest.side == side].sort_values("value_pa", ascending=False)
        k = 600 if side == "bat" else 700
        print(f"\n{label} in the latest fit ({len(d)} with their own value; runs over replacement {unit}):\n")
        print("| player | group | PA in window | runs |\n|---|---|---|---|")
        for r in pd.concat([d.head(args.top), d.tail(3)]).itertuples():
            print(f"| {r.name} | {r.group} | {r.pa} | {r.value_pa * k:+.1f} |")

    # game level: projected value difference (home - away) vs margin, total and the market
    g = tb.games.set_index("game_pk")
    t = table.merge(tb.games[["game_pk", "season", "home_id"]], on="game_pk")
    t["home"] = t.team_id == t.home_id
    t["net"] = t[["_pv_0", "_pv_1", "_pv_2"]].sum(1)
    h, a = t[t.home].set_index("game_pk"), t[~t.home].set_index("game_pk")
    j = pd.DataFrame({"diff": h.net - a.net, "sum_bat": h._pv_0 + a._pv_0, "sum_pit": h._pv_1 + h._pv_2 + a._pv_1
                      + a._pv_2, "season": h.season}).dropna()
    j["margin"] = (g.home_runs - g.away_runs).reindex(j.index)
    j["total"] = (g.home_runs + g.away_runs).reindex(j.index)
    j["home_win"] = (j.margin > 0).astype(float)
    games_csv = Path(args.games)
    mkt = None
    if games_csv.exists():
        w = pd.read_csv(games_csv, usecols=["game_id", "location", "moneyline", "opp_moneyline", "total_line"],
                        low_memory=False)
        w = w[w.location.astype(str).isin(["1", "1.0", "home"])].dropna(subset=["moneyline", "opp_moneyline"])

        def implied(ml):
            ml = ml.astype(float)
            return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))

        ph, pa_ = implied(w.moneyline), implied(w.opp_moneyline)
        mkt = pd.DataFrame({"p_home": ph / (ph + pa_), "total_line": w.total_line.to_numpy()},
                           index=w.game_id.astype(int).to_numpy())
        mkt = mkt[~mkt.index.duplicated()]
        j = j.join(mkt)
        keyed = table.merge(w[["game_id"]].astype({"game_id": str}).drop_duplicates(), on="game_id")
        print(f"\npv.parquet rows whose game_id is in {games_csv.name}: {len(keyed)} of {len(table)}")
    print("\n| season | games | corr(pv diff, margin) | corr(pv diff, home win) | corr(pv diff, market p) | "
          "corr(market p, home win) | corr(bat - pit sum, total) | corr(bat - pit sum, total line) | "
          "runs of margin per pv run |\n|---|---|---|---|---|---|---|---|---|")
    j["tot_pv"] = j.sum_bat - j.sum_pit
    for s, d in [*j.groupby("season"), ("all", j)]:
        slope = np.polyfit(d["diff"], d.margin, 1)[0]
        cells = [f"{d['diff'].corr(d.margin):.3f}", f"{d['diff'].corr(d.home_win):.3f}"]
        if mkt is not None and d.p_home.notna().sum() > 50:
            m = d.dropna(subset=["p_home"])
            cells += [f"{m['diff'].corr(m.p_home):.3f}", f"{m.p_home.corr(m.home_win):.3f}"]
        else:
            cells += ["", ""]
        cells.append(f"{d.tot_pv.corr(d.total):.3f}")
        cells.append(f"{d.dropna(subset=['total_line']).tot_pv.corr(d.total_line.dropna()):.3f}"
                     if mkt is not None and d.total_line.notna().sum() > 50 else "")
        print(f"| {s} | {len(d)} | " + " | ".join(cells) + f" | {slope:.2f} |")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--start", type=int, default=2015)
    t.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    t.add_argument("--lam", type=float, nargs="+", default=[300, 1000, 3000, 10000])
    t.add_argument("--window", type=int, nargs="+", default=[pv.WINDOW])
    t.add_argument("--min-pa", type=int, nargs="+", default=[pv.MIN_PA])
    t.add_argument("--p-scale", type=float, nargs="+", default=[pv.P_SCALE])
    t.add_argument("--fielding", type=int, nargs="+", default=[0])
    t.add_argument("--decay", type=float, default=pv.DECAY)
    r = sub.add_parser("report")
    r.add_argument("--start", type=int, default=2015)
    r.add_argument("--end", type=int, default=2026)
    r.add_argument("--values", default="player_values/mlb")
    r.add_argument("--games", default=str(paths.data_dir() / "raw.csv"))
    r.add_argument("--top", type=int, default=10)
    r.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()
    if args.cmd == "tune":
        args.fielding = [bool(f) for f in args.fielding]
    {"tune": tune, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
