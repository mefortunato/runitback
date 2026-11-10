"""NHL player values (work package H2): tuning and sanity checks for `runitback.nhl.player_values`.

    python scripts/nhl_player_values.py tune      # ridge penalty / window on the next season's segments
    python scripts/nhl_player_values.py goalies   # goalie shrinkage on the next season's shots
    python scripts/nhl_player_values.py report    # segment checks, top players, pv vs margin and market

Reads data/nhl/processed (`runitback nhl events`, then `runitback nhl player-values` builds
segments.parquet and shots.parquet), player_values/nhl (`runitback nhl player-values`) and, for the
market comparison, data/nhl/wager_games.csv (`runitback-wager nhl download --first_season 2020
--output_file data/nhl/wager_games.csv`). Prints markdown; writes nothing to the repository.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from runitback.nhl import player_values as pv

DATA = Path(os.environ.get("RUNITBACK_NHL_DATA", "data/nhl"))


def load():
    proc = DATA / "processed"
    return (pd.read_parquet(proc / "segments.parquet"), pd.read_parquet(proc / "shots.parquet"),
            pd.read_parquet(proc / "games.parquet"), pd.read_parquet(proc / "players.parquet"))


def tune(args):
    seg, _, _, players = load()
    ev = pv.events_from(seg, pv.player_groups(players))
    print("| lambda | window | decay | min minutes | " + " | ".join(f"{s} MSE gain / corr" for s in args.seasons)
          + " | mean MSE gain |")
    print("|---" * (5 + len(args.seasons)) + "|")
    for lam in args.lam:
        for window in args.window:
            for mm in args.min_toi:
                for decay in args.decay:
                    cfg = pv.apm_config(lam=lam, window=window, decay=decay, min_toi=mm)
                    res = [pv.holdout_score(ev, s, cfg) for s in args.seasons]
                    cells = " | ".join(f"{r['mse_gain']:.3f} / {r['corr']:.4f}" for r in res)
                    mean = np.mean([r["mse_gain"] for r in res])
                    print(f"| {lam:g} | {window} | {decay} | {mm} | {cells} | {mean:.4f} |", flush=True)
    print("\nMSE gain: out-of-sample reduction in the minutes-weighted squared error of each segment's "
          "xG per 60 vs a constant.")


def goalies(args):
    _, shots, _, _ = load()
    print("| shrink (shots) | " + " | ".join(f"{s} MSE gain / corr" for s in args.seasons) + " | mean MSE gain |")
    print("|---" * (2 + len(args.seasons)) + "|")
    for shrink in args.shrink:
        res = [pv.goalie_holdout(shots, s, shrink) for s in args.seasons]
        cells = " | ".join(f"{r['mse_gain']:.3f} / {r['corr']:.3f}" for r in res)
        print(f"| {shrink:g} | {cells} | {np.mean([r['mse_gain'] for r in res]):.4f} |", flush=True)
    print("\nMSE gain: out-of-sample reduction in squared error of each shot's (xG - goal) vs the season "
          "mean, x 1e-4; corr: goalies' predicted value vs their realized rate (500+ test shots).")


def report(args):
    seg, shots, games, players = load()
    print("| season | games | segments | seconds each | fitted minutes per game | xG per team-game | "
          "goals / xG |\n|---|---|---|---|---|---|---|")
    for s, d in seg.groupby("season"):
        n = d.game_id.nunique()
        sh = shots[shots.season == s]
        print(f"| {s} | {n} | {len(d)} | {d.seconds.mean():.1f} | {d.seconds.sum() / 60 / n:.1f} | "
              f"{(d.xg_home.sum() + d.xg_away.sum()) / 2 / n:.2f} | {sh.goal.sum() / sh.xg.sum():.2f} |")

    # the outputs of `runitback nhl player-values`
    import json

    from runitback.wager.nhl.data.download import MOVES

    use = json.loads((args.values / "usage.json").read_text())
    vl = pd.read_csv(args.values / "values_latest.csv")
    pvt = pd.read_parquet(args.values / "pv.parquet")
    last = max(use["toi"], key=int)
    toi, shots_pg = use["toi"][last], use["shots"][last]
    top = {g: toi[g][0] for g in pv.GROUPS}
    print(f"\nUsage ({last}, from {int(last) - 1}): top forward {top['F']:.1f} min, top defenseman {top['D']:.1f} min, "
          f"12th forward {toi['F'][11]:.1f}, 6th defenseman {toi['D'][5]:.1f}; a starting goalie faces "
          f"{shots_pg:.1f} unblocked shots.")
    sk = vl[vl.side.isin(["off", "def"])]
    tab = sk.pivot_table(index=["player_id", "name", "group"], columns="side", values="value_60").fillna(0)
    tab = tab.reset_index().rename(columns={"off": "o", "def": "d"})
    tab["gpg"] = (tab.o + tab.d) * tab.group.map(top) / 60
    for g in pv.GROUPS:
        t = tab[tab.group == g].sort_values("gpg", ascending=False)
        print(f"\n**{g}**: latest fit, xG per 60 over replacement and goals per game in the top {g} slot\n")
        print("| player | offense / 60 | defense / 60 | goals per game |\n|---|---|---|---|")
        for r in pd.concat([t.head(args.top), t.tail(3)]).itertuples():
            print(f"| {r.name} | {r.o:+.3f} | {r.d:+.3f} | {r.gpg:+.3f} |")
    gk = vl[vl.side == "goalie"].sort_values("value_shot", ascending=False)
    print("\n**Goalies**: goals saved above expected per shot (vs the league), and per game\n")
    print("| goalie | shots in window | per shot | per game |\n|---|---|---|---|")
    for r in pd.concat([gk.head(args.top), gk.tail(3)]).itertuples():
        print(f"| {r.name} | {int(r.shots)} | {r.value_shot:+.4f} | {r.value_shot * shots_pg:+.3f} |")
    games = games.assign(game_id=games.game_id.astype(str), home=games.home.map(lambda t: MOVES.get(t, t)),
                         away=games.away.map(lambda t: MOVES.get(t, t)))

    g = games.set_index("game_id")
    m = None
    if args.market.exists():   # the wager games table: `runitback-wager nhl download`
        m = pd.read_csv(args.market, dtype={"game_id": str})
        m = m[(m.location == 1) & m.moneyline.abs().ge(100) & m.opp_moneyline.abs().ge(100)].set_index("game_id")
        imp = lambda o: np.where(o < 0, -o / (-o + 100), 100 / (o + 100))  # noqa: E731
        ph, pa = imp(m.moneyline.to_numpy()), imp(m.opp_moneyline.to_numpy())
        m["p_home"] = ph / (ph + pa)
        m["logit"] = np.log(m.p_home / (1 - m.p_home))
    else:
        print(f"\n({args.market} missing: no market comparison)")
    game_level(pvt, g, m, "(a) previous starter (pv.parquet)")
    act = args.values / "pv_actual.parquet"
    if act.exists():
        game_level(pd.read_parquet(act), g, m, "(b) actual starter, known at puck drop (pv_actual.parquet)")


def game_level(pvt: pd.DataFrame, g: pd.DataFrame, m: pd.DataFrame | None, label: str):
    """Projected lineup value difference (home - away) vs the goal margin and the market."""
    print(f"\n### {label}")
    pvt = pvt.merge(g[["home", "away", "season"]], left_on="game_id", right_index=True)
    lev = [f"_pv_{i}" for i in range(5)]
    pvt["net"] = pvt[lev].sum(axis=1)
    pvt["skaters"] = pvt[lev[:4]].sum(axis=1)
    h = pvt[pvt.team == pvt.home].set_index("game_id")
    a = pvt[pvt.team == pvt.away].set_index("game_id")
    d = pd.DataFrame({"diff": h.net - a.net, "sk": h.skaters - a.skaters, "gk": h._pv_4 - a._pv_4,
                      "season": h.season}).dropna()
    # the events' score: regulation and overtime goals (a shootout game is a 0 margin)
    d["margin"] = (g.home_score - g.away_score).reindex(d.index)
    d["win"] = (g.home_final > g.away_final).reindex(d.index).astype(float)
    print("\n| season | games | corr(pv diff, margin) | skaters | goalie | margin goals per pv goal | "
          "home win rate, pv diff top vs bottom third |\n|---|---|---|---|---|---|---|")
    for s, x in [*d.groupby("season"), ("all", d)]:
        q = pd.qcut(x["diff"], 3, labels=False)
        slope = np.polyfit(x["diff"], x.margin, 1)[0]
        print(f"| {s} | {len(x)} | {x['diff'].corr(x.margin):.3f} | {x.sk.corr(x.margin):.3f} | "
              f"{x.gk.corr(x.margin):.3f} | {slope:.2f} | {x.win[q == 2].mean():.3f} / {x.win[q == 0].mean():.3f} |")

    if m is None:
        return
    d.index = d.index.astype(str)
    tot = (h[["_pv_0", "_pv_1"]].sum(axis=1) + a[["_pv_0", "_pv_1"]].sum(axis=1) - h[["_pv_2", "_pv_3", "_pv_4"]].sum(axis=1)
           - a[["_pv_2", "_pv_3", "_pv_4"]].sum(axis=1))
    d["total_pv"] = tot.set_axis(tot.index.astype(str)).reindex(d.index)
    j = d.join(m[["logit", "p_home", "total_line", "total"]], how="inner").dropna(subset=["logit"])
    j = j.assign(goals=g.set_axis(g.index.astype(str)).eval("home_score + away_score").reindex(j.index))
    print("\n| season | games | corr(pv diff, market logit) | skaters | goalie | corr(market, margin) | "
          "corr(pv diff, margin) | corr(pv total, total line) | corr(pv total, goals) |\n|---|---|---|---|---|---|---|---|---|")
    for s, x in [*j.groupby("season"), ("all", j)]:
        tl = x.dropna(subset=["total_line"])
        print(f"| {s} | {len(x)} | {x['diff'].corr(x.logit):.3f} | {x.sk.corr(x.logit):.3f} | {x.gk.corr(x.logit):.3f} | "
              f"{x.logit.corr(x.margin):.3f} | {x['diff'].corr(x.margin):.3f} | "
              f"{tl.total_pv.corr(tl.total_line):.3f} | {x.total_pv.corr(x.goals):.3f} |")
    # does the lineup value add to the market? partial correlation with the margin, the market logit held fixed
    res = lambda y, x: y - np.polyval(np.polyfit(x, y, 1), x)  # noqa: E731
    pc = np.corrcoef(res(j["diff"].to_numpy(), j.logit.to_numpy()), res(j.margin.to_numpy(), j.logit.to_numpy()))[0, 1]
    print(f"\nPartial correlation of the pv difference with the margin, the market held fixed ({len(j)} games): "
          f"{pc:+.3f} +/- {1.96 / np.sqrt(len(j)):.3f}.")




def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    t.add_argument("--lam", type=float, nargs="+", default=[100, 300, 1000, 3000, 10000])
    t.add_argument("--window", type=int, nargs="+", default=[pv.WINDOW])
    t.add_argument("--min-toi", type=float, nargs="+", default=[pv.MIN_TOI])
    t.add_argument("--decay", type=float, nargs="+", default=[pv.DECAY])
    g = sub.add_parser("goalies")
    g.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023, 2024, 2025, 2026])
    g.add_argument("--shrink", type=float, nargs="+", default=[500, 1500, 3000, 6000, 12000])
    r = sub.add_parser("report")
    r.add_argument("--values", type=Path, default=Path("player_values/nhl"), help="`runitback nhl player-values` output")
    r.add_argument("--top", type=int, default=10)
    r.add_argument("--market", type=Path, default=DATA / "wager_games.csv",
                   help="the wager games table with moneylines (runitback-wager nhl download --output_file)")
    args = ap.parse_args()
    {"tune": tune, "goalies": goalies, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
