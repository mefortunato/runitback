"""College football QB values (work package F2): tuning and sanity checks for `runitback.cfb.player_values`.

    python scripts/cfb_player_values.py tune      # ridge penalty / replacement / defense prior on the next season
    python scripts/cfb_player_values.py report    # starters, projection hit rates, top QBs, values vs margin and line

Reads data/cfb/processed (`runitback cfb players`) and data/cfb/raw (`runitback cfb download`). Prints
markdown; writes nothing to the repository.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from runitback.cfb import player_values as pv
from runitback.cfb.data.download import load
from runitback.cfb.data.players import DROPBACK, game_days

DATA = Path(os.environ.get("RUNITBACK_CFB_DATA", "data/cfb"))


def load_processed(rushes: bool = True):
    proc = DATA / "processed"
    plays = pd.read_parquet(proc / "qb_plays.parquet")
    if not rushes:
        plays = plays[plays.kind == DROPBACK].reset_index(drop=True)
    return plays, pd.read_parquet(proc / "team_games.parquet"), pd.read_parquet(proc / "players.parquet")


def tune(args):
    plays, _, _ = load_processed(not args.no_rushes)
    ev = pv.events_from(plays)
    weekly = args.mode == "weekly"
    print(f"Holdout: {'each week from the fit before it' if weekly else 'each season from the fit before it'}.\n")
    print("| lambda | min plays | defense prior | decay | " + " | ".join(
        f"{s} MSE gain / corr{' / slope' if weekly else ''}" for s in args.seasons) + " | mean MSE gain | mean corr |")
    print("|---" * (6 + len(args.seasons)) + "|")
    for lam in args.lam:
        for mp in args.min_plays:
            for ds in args.def_scale:
                for dec in args.decay:
                    cfg = pv.apm_config(lam=lam, min_plays=mp, def_scale=ds, decay=dec)
                    res = [pv.holdout_weeks(ev, s, cfg, args.jobs) if weekly else pv.holdout_score(ev, s, cfg)
                           for s in args.seasons]
                    cells = " | ".join(f"{1e4 * r['mse_gain']:.1f} / {r['corr']:.4f}"
                                       + (f" / {r['slope']:.2f}" if weekly else "") for r in res)
                    print(f"| {lam:g} | {mp} | {ds:g} | {dec:g} | {cells} | "
                          f"{1e4 * np.mean([r['mse_gain'] for r in res]):.2f} | "
                          f"{np.mean([r['corr'] for r in res]):.4f} |", flush=True)
    print("\nMSE gain: out-of-sample reduction in squared error per quarterback play vs a constant, x 1e-4. "
          "Slope: EPA per point of predicted EPA (1 = calibrated).")


def games_frame(seasons: list[int]) -> pd.DataFrame:
    """One row per game: home / away ESPN ids, the final margin and ESPN's line (home expected margin)."""
    parts = []
    for s in seasons:
        p = DATA / "raw" / f"pbp_{s}.parquet"
        if p.exists():
            d = pd.read_parquet(p, columns=["game_id", "homeTeamId", "awayTeamId", "homeFinalScore",
                                            "awayFinalScore", "homeTeamSpread", "gameSpreadAvailable"])
            parts.append(d.groupby(d.game_id.astype(str)).first())
    g = pd.concat(parts)
    out = pd.DataFrame({"home_id": pd.to_numeric(g.homeTeamId), "away_id": pd.to_numeric(g.awayTeamId),
                        "margin": pd.to_numeric(g.homeFinalScore) - pd.to_numeric(g.awayFinalScore)})
    avail = g.gameSpreadAvailable.astype("boolean").fillna(False).astype(bool)
    out["market"] = pd.to_numeric(g.homeTeamSpread, errors="coerce").where(avail)
    return out


def ols(df: pd.DataFrame, y: str, xs: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    """OLS of ``y`` on ``xs`` with an intercept: coefficients and standard errors (``xs`` order)."""
    d = df.dropna(subset=[y, *xs])
    X = np.c_[np.ones(len(d)), d[xs].to_numpy(float)]
    beta, *_ = np.linalg.lstsq(X, d[y].to_numpy(float), rcond=None)
    resid = d[y].to_numpy(float) - X @ beta
    cov = np.linalg.inv(X.T @ X) * (resid @ resid) / (len(d) - X.shape[1])
    return beta[1:], np.sqrt(np.diag(cov))[1:], len(d)


def report(args):
    plays, tg, players = load_processed(not args.no_rushes)
    seasons = sorted(plays.season.unique())
    name = players.set_index("player_id").name

    # starters: the first-dropbacks rule vs the game's dropback leader
    print(f"Team-games with a starter: {len(tg)}; the starter is the game's dropback leader in "
          f"{100 * (tg.starter == tg.leader).mean():.1f}%, and has a median "
          f"{tg.starter_dropbacks.median():.0f} of the team's {tg.dropbacks.median():.0f} dropbacks.\n")

    schedule = load("schedule", [int(s) for s in seasons], DATA / "raw")
    days = game_days(schedule)
    games = pv.schedule_team_games(schedule, days, args.first_season)
    played = games.merge(tg[["game_id", "team_id", "starter"]].assign(game_id=lambda d: d.game_id.astype(str)),
                         on=["game_id", "team_id"])
    # projection hit rates
    first_game = played.sort_values("t").groupby(["team_id", "season"]).t.transform("min") == played.t
    print("Projected starter vs the actual starter (played team-games, "
          f"{args.first_season}-{max(seasons)}):\n")
    print("| rule | all | season openers | other games | games where the starter changed |\n|---|---|---|---|---|")
    prev = pv.project_starters(tg, played, "prev_starter").rename(columns={"qb": "prev"})
    for rule in pv.RULES:
        pr = pv.project_starters(tg, played, rule)
        m = played.merge(pr, on=["game_id", "team_id"], how="left").merge(prev, on=["game_id", "team_id"], how="left")
        hit = m.qb == m.starter
        changed = m.prev.notna() & (m.prev != m.starter)
        print(f"| {rule} | {100 * hit.mean():.1f}% | {100 * hit[first_game.to_numpy()].mean():.1f}% | "
              f"{100 * hit[~first_game.to_numpy()].mean():.1f}% | {100 * hit[changed.to_numpy()].mean():.1f}% "
              f"({changed.sum()}) |")

    gv = pv.game_table(plays, tg, games, args.jobs, rule=args.rule, cfg=pv.apm_config(def_scale=args.def_scale))
    starter_n = gv.usage[max(gv.usage)]
    latest = gv.latest.assign(name=lambda d: d.player_id.map(name)).sort_values("value_play", ascending=False)
    print(f"\nLatest fit: {len(latest)} quarterbacks with their own value; a starter makes {starter_n:.0f} "
          "quarterback plays a game. Points per game over replacement as a starter:\n")
    print("| quarterback | plays (weighted window) | EPA per play over replacement | points per game |\n"
          "|---|---|---|---|")
    for r in pd.concat([latest.head(args.top), latest.tail(5)]).itertuples():
        print(f"| {r.name} | {r.plays} | {r.value_play:+.3f} | {r.value_play * starter_n:+.1f} |")
    fits = sorted(gv.fits)
    pre = [w for w in fits if w < int(days.t[days.game_id.isin(games[games.season == max(seasons)].game_id)].min())]
    if pre:
        pf = gv.fits[pre[-1]].assign(name=lambda d: d.player_id.map(name)).sort_values("value_play", ascending=False)
        print(f"\nBefore the {max(seasons)} season, top {args.top}: "
              + ", ".join(f"{r.name} ({r.value_play * starter_n:+.1f})" for r in pf.head(args.top).itertuples()))

    # transfers: quarterbacks who started for two teams
    st = gv.starters.dropna(subset=["starter"])
    teams = st.groupby("starter").team_id.nunique()
    print(f"\n{(teams > 1).sum()} starters started for more than one team (their ESPN id carries over).")

    # game level
    g = games_frame([int(s) for s in seasons])
    s2 = gv.starters.assign(key=lambda d: d.season.astype(str) + ":" + d.team_id.astype(str))
    rows = []
    for gid, d in s2.groupby("game_id"):
        if gid not in g.index or len(d) != 2:
            continue
        gr = g.loc[gid]
        h, a = d[d.team_id == gr.home_id], d[d.team_id == gr.away_id]
        if len(h) != 1 or len(a) != 1:
            continue
        h, a = h.iloc[0], a.iloc[0]
        rows.append({"game_id": gid, "season": h.season, "proj": h._pv_0 - a._pv_0, "chg": h._pv_2 - a._pv_2,
                     "actual": h.starter_value - a.starter_value, "margin": gr.margin, "market": gr.market,
                     "h_key": h.key, "a_key": a.key,
                     "changed": any(pd.notna(x.starter) and x.starter != x.projected for x in (h, a))})
    gm = pd.DataFrame(rows)
    print("\nQB value difference (home - away, points per game) vs the game: `proj` uses the projected "
          "starters (leak-free), `actual` the starters who played (evaluation only; the starter is not "
          "known before the game in this data).\n")
    print("| season | games | corr(proj, margin) | corr(proj, market) | corr(actual, margin) | "
          "corr(actual, market) | corr(market, margin) | margin pts per QB pt (proj) |\n|---|---|---|---|---|---|---|---|")
    for s, d in [*gm.groupby("season"), ("all", gm)]:
        m = d.dropna(subset=["market"])
        a = d.dropna(subset=["actual"])
        slope = np.polyfit(d.proj, d.margin, 1)[0]
        print(f"| {s} | {len(d)} | {d.proj.corr(d.margin):.3f} | {m.proj.corr(m.market):.3f} | "
              f"{a.actual.corr(a.margin):.3f} | {a.dropna(subset=['market']).pipe(lambda z: z.actual.corr(z.market)):.3f} | "
              f"{m.market.corr(m.margin):.3f} | {slope:.2f} |")

    gm["surprise"] = gm.actual - gm.proj
    print("\nDoes a quarterback change move the line and the result by the value change? OLS over games "
          "(home - away), with `surprise` the actual starters' value minus the projected starters' (nonzero "
          "when a projection was wrong). Coefficients +- 1.96 SE:\n")
    print("| outcome | games | per projected QB pt | per surprise QB pt |\n|---|---|---|---|")
    for y in ("market", "margin"):
        b, se, n = ols(gm, y, ["proj", "surprise"])
        print(f"| {y} | {n} | {b[0]:.2f} +- {1.96 * se[0]:.2f} | {b[1]:.2f} +- {1.96 * se[1]:.2f} |")
    ch = gm[gm.changed]
    b, se, n = ols(ch, "margin", ["proj", "surprise"])
    bm, sem, _ = ols(ch, "market", ["proj", "surprise"])
    print(f"\nGames where a projected starter was wrong: {len(ch)} of {len(gm)} "
          f"({100 * len(ch) / len(gm):.1f}%). On them alone, per surprise QB pt: market "
          f"{bm[1]:.2f} +- {1.96 * sem[1]:.2f}, margin {b[1]:.2f} +- {1.96 * se[1]:.2f}.")
    b, se, n = ols(gm, "margin", ["market", "surprise"])
    print(f"\nMargin on the market line and the surprise: {b[1]:+.2f} +- {1.96 * se[1]:.2f} per surprise QB pt "
          "beyond the line (about 0 if the market prices the change fully).")
    gm["recent"] = gm.proj - gm.chg
    print("\nThe starting-QB-change term (`_pv_2`, leak-free), holding the team's recent starters' value "
          "(`recent`, a stand-in for the wager model's history) fixed:\n")
    print("| outcome | games | per change pt | per recent-starters pt |\n|---|---|---|---|")
    for y in ("market", "margin"):
        b, se, n = ols(gm, y, ["chg", "recent"])
        print(f"| {y} | {n} | {b[0]:.2f} +- {1.96 * se[0]:.2f} | {b[1]:.2f} +- {1.96 * se[1]:.2f} |")
    nz = gm[gm.chg.abs() > 1e-6]
    b, se, n = ols(nz, "margin", ["chg", "recent"])
    bm, sem, _ = ols(nz, "market", ["chg", "recent"])
    print(f"\nGames with a nonzero change term: {len(nz)}; on them, per change pt: market {bm[0]:.2f} +- "
          f"{1.96 * sem[0]:.2f}, margin {b[0]:.2f} +- {1.96 * se[0]:.2f}.")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rushes", action="store_true", help="dropbacks only (default: quarterback runs too)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--seasons", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025])
    t.add_argument("--lam", type=float, nargs="+", default=[300, 1000, 3000, 10000])
    t.add_argument("--min-plays", type=int, nargs="+", default=[pv.MIN_PLAYS])
    t.add_argument("--def-scale", type=float, nargs="+", default=[pv.DEF_SCALE])
    t.add_argument("--decay", type=float, nargs="+", default=[pv.DECAY])
    t.add_argument("--mode", choices=["weekly", "season"], default="weekly")
    t.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    r = sub.add_parser("report")
    r.add_argument("--first-season", type=int, default=2014)
    r.add_argument("--rule", default=pv.RULES[0], choices=pv.RULES)
    r.add_argument("--def-scale", type=float, default=pv.DEF_SCALE)
    r.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    r.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    {"tune": tune, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
