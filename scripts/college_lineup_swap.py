"""College basketball swap test: does the wager model's prediction move the right way when one starter
changes, at every position? (C4; the NBA's is scripts/nba_lineup_swap.py.)

Fits a wager configuration with the player values (default ``pbp32_pvc_s0``, experiments/<gender>/
wager.yaml) on the seasons before ``--season``, then predicts every game of that season again with one
change to the listed (home) team's projected rotation (C3, ``projected_lineups.parquet``): the team's
starter of one position group (its G, F or C with the most projected minutes) replaced by

- ``best`` / ``worst``: the group's best / worst regular in the week's values (offense plus defense per
  possession, at least ``--regular-minutes`` on the floor in the fit's window, and a player of the
  test season), for the starter's minutes;
- ``replacement``: a replacement-level player (value 0) for the starter's minutes;
- ``out``: ruled out, so the others' minutes are scaled up to the team's 200.

The change goes into the game's ``_pv_*`` row (levels, previous-game summaries and changes vs the team's
last 8 rotations) through the week's values (``runitback.college.player_values``: the same weekly fits
``pv.parquet`` uses, read from ``fits_<season>.parquet``). Writes one row per (game, group, change) with
the change in value, margin, win probability and total, and the summary table of
``scripts/nba_lineup_swap.py``: a position passes when the margin and the win probability move with the
value in every game where the value moved (|Δ| > 0.05 points).

    python scripts/college_lineup_swap.py --gender cbb --season 2026 --out swap/swap_2026.csv
    python scripts/college_lineup_swap.py --summary swap/swap_2026.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from nba_lineup_swap import KINDS, summarize  # noqa: E402

log = logging.getLogger("college_swap")


def regular_refs(fit: pd.DataFrame, pg: pd.DataFrame, season: int, min_minutes: float) -> dict:
    """:func:`runitback.college.player_values.swap_refs` among the players who play in ``season`` (so
    the best and worst regular are players still in college), valued by the season's first week's fit."""
    from runitback.college import player_values as pv

    active = set(pg.loc[(pg.season == season) & ~pg.dnp, "player_id"])
    return pv.swap_refs(fit, min_minutes, active)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gender", default="cbb", choices=["cbb", "wcbb"])
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--config", default="pbp32_pvc_s0")
    ap.add_argument("--experiment", type=Path, help="default: experiments/<gender>/wager.yaml")
    ap.add_argument("--games", help="default: data/<gender>/games.csv")
    ap.add_argument("--data", type=Path, help="default: data/<gender>/processed")
    ap.add_argument("--values", type=Path, help="default: player_values/<gender> (fits_<season>.parquet, ppg.json)")
    ap.add_argument("--regular-minutes", type=float, default=600.0)
    ap.add_argument("--day-stride", type=int, default=1, help="predict every n-th game day (the fit is the same)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--summary", type=Path, help="only summarize an earlier --out file")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    if a.summary:
        print(summarize(pd.read_csv(a.summary)))
        return
    g = a.gender
    experiment = a.experiment or Path(f"experiments/{g}/wager.yaml")
    data_dir = a.data or Path(f"data/{g}/processed")
    values = a.values or Path(f"player_values/{g}")
    out = a.out or Path(f"experiments/{g}/wager/swap_{a.season}.csv")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import torch

    torch.set_num_threads(a.threads)
    from runitback.college import player_values as pv
    from runitback.wager.backtest import transformer_fitter
    from runitback.wager.core.io import read_table
    from runitback.wager.core.schema import one_row_per_game, played
    from runitback.wager.core.splits import split_seasons
    from runitback.wager.core.sport import get_sport
    from runitback.wager.models.transformer import data as tdata

    cfg = yaml.safe_load(experiment.read_text())["configs"][a.config]
    model_cfg = cfg.get("model") or {}
    schema = get_sport(g).schema
    games = played(read_table(a.games or f"data/{g}/games.csv"), schema)
    history = games[games[schema.season] < a.season]
    fitter = transformer_fitter(g, max_epochs=cfg.get("max_epochs", 100), patience=cfg.get("patience", 10),
                                model=model_cfg, **(cfg.get("fit") or {}))
    log.info("fitting %s on the seasons before %d", a.config, a.season)
    predictor = fitter(history, split_seasons(history[schema.season].unique(), test=[]))
    from runitback.wager.web.lineups import additive_weights

    try:
        weights = additive_weights(predictor)   # what the fit learned: margin points per point of change
    except Exception as e:                      # a report line, not worth losing the fit for
        log.warning("could not read the learned lineup weights: %s", e)
        weights = None
    log.info("learned lineup weights (margin per point of value change, by group): %s", weights)
    table = tdata._read_pbp(tdata.pbp_fold_path(model_cfg["lineup"], a.season))
    cols_missing = set(f"_pv_{i}" for i in range(14)) - set(table.columns)
    if cols_missing:
        raise SystemExit(f"{model_cfg['lineup']} has no {sorted(cols_missing)}")

    # this season's weekly values, rotations and the possessions-per-game scale, as pv.parquet used them
    pg = pd.read_parquet(data_dir / "player_games.parquet")
    pg["game_id"] = pg.game_id.astype(str)
    proj = pd.read_parquet(data_dir / "projected_lineups.parquet")
    proj["game_id"] = proj.game_id.astype(str)
    fits = pd.read_parquet(values / f"fits_{a.season}.parquet")
    vals = {int(b): dict(zip(zip(f.side, f.player_id), f.value)) for b, f in fits.groupby("before")}
    first_week = int(fits.before.min())
    ppg = {int(k): v for k, v in json.loads((values / "ppg.json").read_text()).items()}
    rate = ppg.get(a.season) or ppg[max(ppg)]
    group_of = {int(p): grp for p, grp in zip(pg.player_id, pg.group)}
    names = dict(zip(pg.player_id, pg["name"]))
    rot = {k: list(zip(d.player_id.astype(int), d.proj_minutes)) for k, d in
           proj[proj.season == a.season].groupby(["game_id", "team_id"])}
    week_of = {(gid, tid): int(b) for gid, tid, b in pv.week_cutoffs(pg, a.season)[
        ["game_id", "team_id", "before"]].itertuples(index=False)}
    refs = regular_refs(fits[fits.before == first_week], pg, a.season, a.regular_minutes)
    for (grp, k), p in sorted(refs.items()):
        log.info("%s %s regular: %s", grp, k, names.get(p, p))
    team_id = table["team_id"] if "team_id" in table else None
    if team_id is None:
        raise SystemExit("the player-value table needs a team_id column")

    n = len(pv.GROUP_COLUMNS)

    def delta(game_id: str, team: str, group: str, kind: str) -> np.ndarray | None:
        """The swap's change to the team-game's 14 ``_pv_*`` columns."""
        tid = team_id.get((game_id, team))
        rows = rot.get((game_id, tid))
        week = week_of.get((game_id, tid))
        if rows is None or week is None or week not in vals:
            return None
        return pv.swap_delta(rows, vals[week], group_of, rate, group, kind, refs.get((group, kind)))

    current = games[games[schema.season] == a.season]
    pv_cols = [f"_pv_{i}" for i in range(2 * n + 2)]
    rows_out = []
    for date in sorted(current[schema.date].unique())[::a.day_stride]:
        known = pd.concat([history, current[current[schema.date] < date]])
        predictor.update_state(known)
        day = one_row_per_game(current[current[schema.date] == date], schema)
        teams = set(predictor.model.teams)
        day = day[day[schema.team].isin(teams) & day[schema.opp].isin(teams)].reset_index(drop=True)
        if day.empty:
            continue
        ids = tdata.game_keys(day[schema.game_id])
        base = predictor.predict(day)
        for group in pv.POSITIONS:
            for kind in KINDS:
                idx, ds, at = [], [], []
                for i, (gid, team) in enumerate(zip(ids, day[schema.team])):
                    d = delta(str(gid), team, group, kind)
                    if d is None or (gid, team) not in table.index:
                        continue
                    idx.append((gid, team))
                    ds.append(d)
                    at.append(i)
                if not idx:
                    continue
                ds = np.array(ds, dtype=np.float32)
                saved = table.loc[idx, pv_cols].copy()
                table.loc[idx, pv_cols] = saved.to_numpy() + ds
                try:
                    pred = predictor.predict(day)
                finally:
                    table.loc[idx, pv_cols] = saved.to_numpy()
                for (gid, _), d, i in zip(idx, ds, at):
                    rows_out.append(dict(
                        game_id=gid, date=date, group=group, kind=kind,
                        ref=names.get(refs.get((group, kind)), refs.get((group, kind), ""))
                        if kind in ("best", "worst") else "",
                        d_value=float(d[:n].sum()), d_off=float(d[n]), d_def=float(d[n + 1]),
                        d_margin=float(pred.pred_margin.iloc[i] - base.pred_margin.iloc[i]),
                        d_win=float(pred.win_prob.iloc[i] - base.win_prob.iloc[i]),
                        d_total=float(pred.pred_total.iloc[i] - base.pred_total.iloc[i])
                        if "pred_total" in pred else np.nan))
        log.info("%s: %d rows", date, len(rows_out))
    df = pd.DataFrame(rows_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    text = summarize(df)
    if weights:
        m = np.mean(weights["margin"])
        text += (f"\n\nLearned lineup weight in `{a.config}` (margin per point of value change; started at "
                 f"{model_cfg.get('additive_init', 0.598)}): {m:.2f}; on the total, per group "
                 f"{np.round(weights['total'], 2).tolist()}.")
    out.with_suffix(".md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
