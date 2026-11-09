"""Does the wager model's prediction move the right way when one player changes, at every position?

Fits a wager configuration with the player values in its context (pbp32_pv, experiments/nfl/
playervalues.yaml) on the seasons before ``--season``, then predicts every game of that season
again with one change to the home team's projected lineup: that group's starter (QB1, RB1, WR1,
TE1, OL1, DL1, LB1, DB1, K, P) replaced by the best / worst regular (swap_refs.csv), by a
replacement-level player (value 0), or ruled out (the group moves up a slot). The change goes in
through the lineup values (the week's fit, scripts/player_values.py). Prints the change in the
predicted home margin and win probability, next to what the direct game model's coefficients
(scripts/direct_game_model.py) say it should be.

    python scripts/wager_lineup_sensitivity.py --season 2025
"""
from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger("sensitivity")
GROUPS = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB", "K", "P"]
PV_COL = {g: i for i, g in enumerate(["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB"])} | {"K": 8, "P": 8}
OFF = {"QB", "RB", "WR", "TE", "OL"}
DEF = {"DL", "LB", "DB"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--config", default="pbp32_pv")
    ap.add_argument("--experiment", type=Path, default=Path("experiments/nfl/playervalues.yaml"))
    ap.add_argument("--games", default="data/nfl/games.csv")
    ap.add_argument("--plays", type=Path, help="cached play table (pickle of player_values.play_table, special)")
    ap.add_argument("--refs", type=Path, default=Path("experiments/nfl/playervalues/swap_refs_2025.csv"))
    ap.add_argument("--out", type=Path, default=Path("experiments/nfl/playervalues/sensitivity.csv"))
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import torch

    torch.set_num_threads(a.threads)
    from runitback.wager.core.io import read_table
    from runitback.wager.core.schema import one_row_per_game, played
    from runitback.wager.core.splits import split_seasons
    from runitback.wager.core.sport import get_sport
    from runitback.nfl import player_values as pv
    from runitback.nfl.lineup_features import load_availability, load_depth_charts, projected_lineups
    from runitback.wager.backtest import transformer_fitter
    from runitback.wager.models.transformer import data as tdata

    D, R = Path("data/nfl/processed"), Path("data/nfl/raw")
    cfg = yaml.safe_load(a.experiment.read_text())["configs"][a.config]
    pv_path = cfg["model"]["lineup"]
    schema = get_sport("nfl").schema
    games = played(read_table(a.games), schema)
    history = games[games[schema.season] < a.season]
    split = split_seasons(history[schema.season].unique(), test=[])
    fitter = transformer_fitter("nfl", max_epochs=cfg.get("max_epochs", 100), patience=cfg.get("patience", 10),
                                model=cfg.get("model") or {}, **(cfg.get("fit") or {}))
    log.info("fitting %s on the seasons before %d", a.config, a.season)
    predictor = fitter(history, split)

    # this season's projected lineups and each week's values
    players = pd.read_parquet(D / "players.parquet")
    group_of = dict(zip(players.player_id, players.group))
    if a.plays and a.plays.exists():
        data, special = pickle.load(open(a.plays, "rb"))
    else:
        g_ = pd.read_parquet(D / "games.parquet", columns=["game_id", "season", "week"])
        data = pv.play_table(D, R, group_of)
        special = pv.special_table(R, list(range(a.season - 5, a.season + 1)), g_)
    pg = pd.read_parquet(D / "player_games.parquet")
    plays_at = pv.slot_plays(pg[pg.season < 2016])
    sched = pd.read_parquet(D / "games.parquet")
    sched = sched[sched.season == a.season]
    out_p, active = load_availability(R, [a.season], download=False)
    charts = load_depth_charts(R, [a.season], download=False)
    proj = projected_lineups(pg, sched, out_p, active, charts)
    proj = {k: d for k, d in proj.groupby(["game_id", "team"])}
    refs = pd.read_csv(a.refs)
    fits = {}

    def week_values(t):
        if t not in fits:
            f = pv.fit_values(data, special, t).table
            fits[t] = {(s, p): v for s, p, v in zip(f.side, f.player_id, f.value_play)}
        return fits[t]

    def pts(vp, grp, pid, rank):
        if grp in OFF:
            return vp.get(("off", pid), 0.0) * plays_at.get((grp, rank), 0.0)
        if grp in DEF:
            return vp.get(("def", pid), 0.0) * plays_at.get((grp, rank), 0.0)
        return vp.get((grp, pid), 0.0) * (pv.FG_PER_GAME if grp == "K" else pv.PUNTS_PER_GAME)

    def change(game_id, team, t, grp, kind) -> float | None:
        """Change in the team's group value (points per game) under one lineup change."""
        lu = proj.get((game_id, team))
        if lu is None:
            return None
        d = lu[lu.group == grp].sort_values("rank")
        if d.empty:
            return None
        vp = week_values(t)
        cur = [pts(vp, grp, p, int(k)) for p, k in zip(d.player_id, d["rank"])]
        if kind == "replacement":
            return -cur[0]
        if kind == "out":
            if grp in ("K", "P"):
                return -cur[0]
            ids = d.player_id.tolist()[1:]
            new = [pts(vp, grp, p, k) for k, p in enumerate(ids)]
            return sum(new) - sum(cur)
        r = refs[(refs.group == grp) & (refs.kind == kind)]
        if r.empty:
            return None
        return pts(vp, grp, r.player_id.iloc[0], 0) - cur[0]

    key = next((k for k in tdata._PBP_CACHE if str(k).endswith(Path(pv_path).name)), None)
    current = games[games[schema.season] == a.season]
    rows = []
    kinds = ["best", "worst", "replacement", "out"]
    for date in sorted(current[schema.date].unique()):
        known = pd.concat([history, current[current[schema.date] < date]])
        predictor.update_state(known)
        day = one_row_per_game(current[current[schema.date] == date], schema)
        teams = set(predictor.model.teams)
        day = day[day[schema.team].isin(teams) & day[schema.opp].isin(teams)]
        if day.empty:
            continue
        base = predictor.predict(day)
        if key is None:
            key = next((k for k in tdata._PBP_CACHE if str(k).endswith(Path(pv_path).name)), None)
        table = tdata._PBP_CACHE[key]
        wk = sched.set_index("game_id").week
        for grp in GROUPS:
            for kind in kinds:
                saved, deltas = {}, {}
                for gid, team in zip(day[schema.game_id], day[schema.team]):
                    if (gid, team) not in table.index or gid not in wk.index:
                        continue
                    dv = change(gid, team, a.season * 100 + int(wk[gid]), grp, kind)
                    if dv is None:
                        continue
                    idx = (gid, team)
                    saved[idx] = table.loc[idx].copy()
                    c = f"_pv_{PV_COL[grp]}"
                    table.loc[idx, c] += dv
                    if "_pv_19" in table.columns:          # pva: the group's change column too
                        table.loc[idx, f"_pv_{11 + PV_COL[grp]}"] += dv
                    if grp in OFF:
                        table.loc[idx, "_pv_9"] += dv
                    elif grp in DEF:
                        table.loc[idx, "_pv_10"] += dv
                    deltas[gid] = dv
                if not deltas:
                    continue
                pred = predictor.predict(day)
                for idx, v in saved.items():
                    table.loc[idx] = v
                for i in range(len(day)):
                    gid = day[schema.game_id].iloc[i]
                    if gid in deltas:
                        rows.append(dict(game_id=gid, group=grp, kind=kind, d_value=deltas[gid],
                                         d_margin=float(pred.pred_margin.iloc[i] - base.pred_margin.iloc[i]),
                                         d_win=float(pred.win_prob.iloc[i] - base.win_prob.iloc[i])))
        log.info("%s: %d rows", date, len(rows))
    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    summarize(df, refs)


def summarize(df: pd.DataFrame, refs: pd.DataFrame) -> None:
    names = {(r.group, r.kind): r["name"] for _, r in refs.iterrows()}
    coef = {**dict.fromkeys(OFF, 0.611), **dict.fromkeys(DEF, 0.46), "K": 0.98, "P": 0.98}  # direct model, 2025 fit
    print(f"{df.game_id.nunique()} games; one change to the home team's projected lineup:")
    print(f"  {'slot':4s} {'change':11s} {'player':22s} {'Δ value':>8s} {'Δ margin':>15s} {'direct':>7s} {'Δ win':>15s}  same sign")
    for g in GROUPS:
        for k in ["best", "worst", "replacement", "out"]:
            d = df[(df.group == g) & (df.kind == k)]
            if d.empty:
                continue
            ci = lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x))
            nz = d[d.d_value.abs() > 0.05]
            agree = (np.sign(nz.d_margin) == np.sign(nz.d_value)).mean() if len(nz) else np.nan
            print(f"  {g:4s} {k:11s} {names.get((g, k), '')[:22]:22s} {d.d_value.mean():+8.2f} "
                  f"{d.d_margin.mean():+7.2f} ± {ci(d.d_margin):.2f} {coef[g] * d.d_value.mean():+7.2f} "
                  f"{d.d_win.mean():+.3f} ± {ci(d.d_win):.3f}  {agree:.0%}")


if __name__ == "__main__":
    main()
