"""NBA swap test: does the wager model's prediction move the right way when one starter changes, at
every position?

Fits a wager configuration with the player values (default ``pbp32_pvc_e4``, experiments/nba/
wager.yaml) on the seasons before ``--season``, then predicts every game of that season again with
one change to the home team's projected rotation (N3, ``projected_lineups.parquet``): the starter
of one position group (its top-ranked G, F or C) replaced by

- ``best`` / ``worst``: the group's best / worst regular in the fit before the season (offense plus
  defense value per possession, at least ``--regular-minutes`` minutes on court in the window),
  playing his slot;
- ``replacement``: a replacement-level player (value 0) in his slot;
- ``out``: ruled out, so everyone below him moves up a rotation slot.

The change goes into the game's ``_pv_*`` row (levels, previous-game summaries and changes vs the
team's last 10 rotations) through the week's values (``runitback.nba.player_values``, the same fit
``pv.parquet`` uses). Writes one row per (game, group, change) with the change in home value,
margin, win probability and total, and prints the summary table: a position passes when the margin
and the win probability move with the value in every game where the value moved (|Δ| > 0.05).

    python scripts/nba_lineup_swap.py --season 2026 --out swap/swap_2026.csv
    python scripts/nba_lineup_swap.py --summary swap/swap_2026.csv
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger("nba_swap")
KINDS = ("best", "worst", "replacement", "out")
MIN_CHANGE = 0.05      # points of value: smaller changes are not scored for direction


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--config", default="pbp32_pvc_e4")
    ap.add_argument("--experiment", type=Path, default=Path("experiments/nba/wager.yaml"))
    ap.add_argument("--games", default="data/nba/games.csv")
    ap.add_argument("--data", type=Path, default=Path("data/nba/processed"))
    ap.add_argument("--regular-minutes", type=float, default=2000.0)
    ap.add_argument("--out", type=Path, default=Path("experiments/nba/wager/swap.csv"))
    ap.add_argument("--summary", type=Path, help="only summarize an earlier --out file")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    if a.summary:
        print(summarize(pd.read_csv(a.summary)))
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import torch

    torch.set_num_threads(a.threads)
    from runitback.nba import player_values as pv
    from runitback.wager.backtest import transformer_fitter
    from runitback.wager.core.io import read_table
    from runitback.wager.core.schema import one_row_per_game, played
    from runitback.wager.core.splits import split_seasons
    from runitback.wager.core.sport import get_sport
    from runitback.wager.models.transformer import data as tdata

    cfg = yaml.safe_load(a.experiment.read_text())["configs"][a.config]
    schema = get_sport("nba").schema
    games = played(read_table(a.games), schema)
    history = games[games[schema.season] < a.season]
    fitter = transformer_fitter("nba", max_epochs=cfg.get("max_epochs", 100), patience=cfg.get("patience", 10),
                                model=cfg.get("model") or {}, **(cfg.get("fit") or {}))
    log.info("fitting %s on the seasons before %d", a.config, a.season)
    predictor = fitter(history, split_seasons(history[schema.season].unique(), test=[]))
    pv_path = tdata.pbp_fold_path(cfg["model"]["lineup"], a.season)
    table = tdata._read_pbp(pv_path)

    # this season's weekly values, rotations and usage, as runitback nba player-values builds them
    poss = pd.read_parquet(a.data / "possessions.parquet")
    poss, pg = pv.franchise_only(poss, pd.read_parquet(a.data / "player_games.parquet"))
    players = pd.read_parquet(a.data / "players.parquet")
    group_of = pv.player_groups(players)
    names = dict(zip(players.player_id, players.name))
    events = pv.events_from(poss, group_of)
    mult = pv.slot_possessions(poss, pg)[a.season]
    proj = pv.projected_rotation(pd.read_parquet(a.data / "projected_lineups.parquet"))
    sched = pg[pg.season == a.season][["game_id", "team", "game_date"]].drop_duplicates(["game_id", "team"])
    proj = proj.merge(sched, on=["game_id", "team"])
    rot = {k: list(zip(d["rank"].astype(int), d.player_id)) for k, d in proj.groupby(["game_id", "team"])}
    week_of = {g: pv.week_start(pv.day_number(d)) for g, d in zip(sched.game_id, sched.game_date)}
    weeks = sorted(set(week_of.values()))
    log.info("fitting %d weekly values for %d", len(weeks), a.season)
    from runitback.core.player_values import fit_weekly

    fits = fit_weekly(events, pv.apm_config(), weeks, lambda w: a.season, a.threads)
    fits = {w: f.rename(columns={"events": "possessions", "value": "value_poss"}) for w, f in zip(weeks, fits)}
    vals = {w: dict(zip(zip(f.side, f.player_id), f.value_poss)) for w, f in fits.items()}
    refs = pv.swap_refs(fits[weeks[0]], int(a.regular_minutes * pv.POSS_PER_MIN))
    for (g, k), p in sorted(refs.items()):
        log.info("%s %s regular: %s", g, k, names.get(p, p))
    # player_games' team keys (canonical franchise abbreviations) are the wager table's
    missing = set(sched.team) - set(table.index.get_level_values(1))
    if missing:
        raise SystemExit(f"teams of player_games not in {pv_path}: {sorted(missing)}")

    cols = list(pv.GROUP_COLUMNS)
    n = len(cols)

    def delta(game_id: str, team: str, group: str, kind: str) -> np.ndarray | None:
        """The swap's change to the team-game's 14 ``_pv_*`` columns."""
        rows = rot.get((game_id, team))
        if rows is None or game_id not in week_of:
            return None
        new = pv.swap_rotation(rows, group_of, group, kind, refs.get((group, kind)))
        if new is None:
            return None
        vp = vals[week_of[game_id]]
        cur, nxt = pv.side_values(rows, vp, group_of, mult), pv.side_values(new, vp, group_of, mult)
        d = np.array([nxt[c] - cur[c] for c in cols])
        off = sum(d[i] for i, c in enumerate(cols) if c.endswith("_OFF"))
        dfn = sum(d[i] for i, c in enumerate(cols) if c.endswith("_DEF"))
        return np.concatenate([d, [off, dfn], d])      # levels, summaries, changes vs recent

    current = games[games[schema.season] == a.season]
    pv_cols = [f"_pv_{i}" for i in range(2 * n + 2)]
    rows_out = []
    for date in sorted(current[schema.date].unique()):
        known = pd.concat([history, current[current[schema.date] < date]])
        predictor.update_state(known)
        day = one_row_per_game(current[current[schema.date] == date], schema)
        teams = set(predictor.model.teams)
        day = day[day[schema.team].isin(teams) & day[schema.opp].isin(teams)].reset_index(drop=True)
        if day.empty:
            continue
        ids = tdata.game_keys(day[schema.game_id])
        base = predictor.predict(day)
        for group in pv.GROUPS:
            for kind in KINDS:
                idx, ds, at = [], [], []
                for i, (gid, team) in enumerate(zip(ids, day[schema.team])):
                    d = delta(gid, team, group, kind)
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
                        ref=names.get(refs.get((group, kind)), "") if kind in ("best", "worst") else "",
                        d_value=float(d[:n].sum()), d_off=float(d[n]), d_def=float(d[n + 1]),
                        d_margin=float(pred.pred_margin.iloc[i] - base.pred_margin.iloc[i]),
                        d_win=float(pred.win_prob.iloc[i] - base.win_prob.iloc[i]),
                        d_total=float(pred.pred_total.iloc[i] - base.pred_total.iloc[i])
                        if "pred_total" in pred else np.nan))
        log.info("%s: %d rows", date, len(rows_out))
    df = pd.DataFrame(rows_out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    text = summarize(df)
    a.out.with_suffix(".md").write_text(text + "\n")
    print(text)


def summarize(df: pd.DataFrame) -> str:
    """Per (group, change): games, mean Δ value / margin / win / total with 95% CIs, and the share of
    games (with |Δ value| > MIN_CHANGE) where margin and win probability move with the value."""
    def ci(x):
        return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0

    lines = [f"{df.game_id.nunique()} games; one change to the home team's projected rotation.", "",
             "| group | change | regular | games | Δ value | Δ home margin | Δ home win | Δ total | right way |",
             "|---|---|---|---|---|---|---|---|---|"]
    ok = True
    for g in ("G", "F", "C"):
        for k in KINDS:
            d = df[(df.group == g) & (df.kind == k)]
            if d.empty:
                continue
            nz = d[d.d_value.abs() > MIN_CHANGE]
            right = ((np.sign(nz.d_margin) == np.sign(nz.d_value)) & (np.sign(nz.d_win) == np.sign(nz.d_value)))
            share = right.mean() if len(nz) else np.nan
            ok &= bool(len(nz) == 0 or right.all())
            ref = d.ref.dropna().iloc[0] if "ref" in d and d.ref.notna().any() else ""
            lines.append(f"| {g} | {k} | {ref} | {len(d)} | {d.d_value.mean():+.2f} | "
                         f"{d.d_margin.mean():+.2f} ± {ci(d.d_margin):.2f} | {d.d_win.mean():+.3f} ± {ci(d.d_win):.3f} | "
                         f"{d.d_total.mean():+.2f} ± {ci(d.d_total):.2f} | {share:.1%} |")
    lines += ["", f"**Every position moves the prediction the right way: {'yes' if ok else 'no'}.**"]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
