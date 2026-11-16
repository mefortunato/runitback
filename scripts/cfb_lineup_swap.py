"""CFB swap test: does the wager model's prediction move the right way when the home team's starting
quarterback changes?

College football has no personnel data beyond the quarterback (README "College football"), so the test
changes that one position. Fits a wager configuration with the QB values (default ``pbp32_qb_e4``,
experiments/cfb/wager.yaml) on the seasons before ``--season``, then predicts every game of that season
again with the home team's projected starter replaced by

- ``best`` / ``worst``: the best / worst regular quarterback in the fit at the season's start (value
  per play, at least ``--regular-plays`` weighted plays in the window);
- ``replacement``: a replacement-level player (value 0);
- ``out``: ruled out, so the team's backup plays: its most frequent other starter among its last
  12 (replacement level when it has had none).

The change goes into the game's ``_pv_*`` row (levels, change vs the previous starter and vs the
last 7) through the week's values (``runitback.cfb.player_values``, the fit ``pv.parquet`` uses).
Writes one row per (game, change) with the change in home value, margin, win probability and total,
and prints the summary table: the position passes when the margin and the win probability move with
the value in every game where the value moved (|Δ| > 0.05).

    python scripts/cfb_lineup_swap.py --season 2025 --out swap/swap_2025.csv
    python scripts/cfb_lineup_swap.py --summary swap/swap_2025.csv
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger("cfb_swap")
KINDS = ("best", "worst", "replacement", "out")
MIN_CHANGE = 0.05      # points of value: smaller changes are not scored for direction
PV_COLS = ["_pv_0", "_pv_1", "_pv_2"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--config", default="pbp32_qb_e4")
    ap.add_argument("--experiment", type=Path, default=Path("experiments/cfb/wager.yaml"))
    ap.add_argument("--games", default="data/cfb/games.csv")
    ap.add_argument("--data", type=Path, default=Path("data/cfb"))
    ap.add_argument("--start", type=int, default=2012, help="first season of plays")
    ap.add_argument("--regular-plays", type=float, default=400.0)
    ap.add_argument("--out", type=Path, default=Path("experiments/cfb/wager/swap.csv"))
    ap.add_argument("--summary", type=Path, help="only summarize an earlier --out file")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    if a.summary:
        print(summarize(pd.read_csv(a.summary)))
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import torch

    torch.set_num_threads(a.threads)
    from runitback.cfb import player_values as pv
    from runitback.cfb.data.download import load
    from runitback.cfb.data.players import game_days
    from runitback.wager.backtest import transformer_fitter
    from runitback.wager.core.io import read_table
    from runitback.wager.core.schema import one_row_per_game, played
    from runitback.wager.core.splits import split_seasons
    from runitback.wager.core.sport import get_sport
    from runitback.wager.models.transformer import data as tdata

    cfg = yaml.safe_load(a.experiment.read_text())["configs"][a.config]
    schema = get_sport("cfb").schema
    games = played(read_table(a.games), schema)
    history = games[games[schema.season] < a.season]
    fitter = transformer_fitter("cfb", max_epochs=cfg.get("max_epochs", 100), patience=cfg.get("patience", 10),
                                model=cfg.get("model") or {}, **(cfg.get("fit") or {}))
    log.info("fitting %s on the seasons before %d", a.config, a.season)
    predictor = fitter(history, split_seasons(history[schema.season].unique(), test=[]))
    table = tdata._read_pbp(cfg["model"]["lineup"])

    # the season's weekly values and starters, as runitback cfb player-values builds them
    proc, seasons = a.data / "processed", list(range(a.start, a.season + 1))
    plays = pd.read_parquet(proc / "qb_plays.parquet")
    plays = plays[plays.season.isin(seasons)]
    tg = pd.read_parquet(proc / "team_games.parquet")
    tg = tg[tg.season.isin(seasons)]
    schedule = load("schedule", seasons, a.data / "raw")
    gv = pv.game_table(plays, tg, pv.schedule_team_games(schedule, game_days(schedule), a.season),
                       a.threads)
    names = dict(zip(pd.read_parquet(proc / "players.parquet").player_id,
                     pd.read_parquet(proc / "players.parquet").name))
    st = gv.starters.dropna(subset=["projected"])
    st = st[st.team != "FCS"].assign(game_id=lambda d: d.game_id.astype(str))
    rows = {(g, t): r for g, t, r in zip(st.game_id, st.team, st.itertuples())}
    weeks = sorted({pv.week_start(int(t)) for t in st.t})
    refs = pv.regulars(gv.fits[weeks[0]], a.regular_plays)
    for k, p in refs.items():
        log.info("%s regular: %s (%+.2f / play)", k, names.get(p, p),
                 gv.fits[weeks[0]].set_index("player_id").value_play[p])
    mult = gv.usage.get(a.season) or gv.usage[max(gv.usage)]
    hist = {team: d.sort_values(["t", "game_id"], kind="stable") for team, d in tg.groupby("team_id")}

    def delta(game_id: str, team: str, kind: str) -> float | None:
        r = rows.get((game_id, team))
        if r is None or r.season != a.season:
            return None
        fit = gv.fits.get(pv.week_start(int(r.t)))
        if fit is None:
            return None
        vp = dict(zip(fit.player_id, fit.value_play))
        h = hist.get(r.team_id)
        past = h[h.t < r.t].starter.to_numpy() if h is not None else []
        return pv.swap_delta(kind, r.projected, vp, refs, pv.backup_of(past, r.projected), mult)

    current = games[games[schema.season] == a.season]
    out = []
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
        for kind in KINDS:
            idx, ds, at = [], [], []
            for i, (gid, team) in enumerate(zip(ids, day[schema.team])):
                d = delta(str(gid), team, kind)
                if d is None or (gid, team) not in table.index:
                    continue
                idx.append((gid, team))
                ds.append(d)
                at.append(i)
            if not idx:
                continue
            saved = table.loc[idx, PV_COLS].copy()
            table.loc[idx, PV_COLS] = saved.to_numpy() + np.array(ds, dtype=np.float32)[:, None]
            try:
                pred = predictor.predict(day)
            finally:
                table.loc[idx, PV_COLS] = saved.to_numpy()
            for (gid, _), d, i in zip(idx, ds, at):
                out.append(dict(
                    game_id=gid, date=date, group="QB", kind=kind,
                    ref=names.get(refs.get(kind), "") if kind in ("best", "worst") else "", d_value=d,
                    d_margin=float(pred.pred_margin.iloc[i] - base.pred_margin.iloc[i]),
                    d_win=float(pred.win_prob.iloc[i] - base.win_prob.iloc[i]),
                    d_total=float(pred.pred_total.iloc[i] - base.pred_total.iloc[i])
                    if "pred_total" in pred else np.nan))
        log.info("%s: %d rows", date, len(out))
    df = pd.DataFrame(out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    text = summarize(df)
    a.out.with_suffix(".md").write_text(text + "\n")
    print(text)


def summarize(df: pd.DataFrame) -> str:
    """Per change: games, mean Δ value / margin / win / total with 95% CIs, and the share of games
    (with |Δ value| > MIN_CHANGE) where margin and win probability move with the value."""
    def ci(x):
        return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0

    lines = [f"{df.game_id.nunique()} games; the home team's projected starting quarterback changed.", "",
             "| change | quarterback | games | Δ value (pts/game) | Δ home margin | Δ home win | Δ total | right way |",
             "|---|---|---|---|---|---|---|---|"]
    ok = True
    for k in KINDS:
        d = df[df.kind == k]
        if d.empty:
            continue
        nz = d[d.d_value.abs() > MIN_CHANGE]
        right = (np.sign(nz.d_margin) == np.sign(nz.d_value)) & (np.sign(nz.d_win) == np.sign(nz.d_value))
        share = right.mean() if len(nz) else np.nan
        ok &= bool(len(nz) == 0 or right.all())
        ref = d.ref.dropna().iloc[0] if "ref" in d and d.ref.notna().any() else ""
        lines.append(f"| {k} | {ref} | {len(d)} | {d.d_value.mean():+.2f} | "
                     f"{d.d_margin.mean():+.2f} ± {ci(d.d_margin):.2f} | {d.d_win.mean():+.3f} ± {ci(d.d_win):.3f} | "
                     f"{d.d_total.mean():+.2f} ± {ci(d.d_total):.2f} | {share:.1%} |")
    lines += ["", f"**The quarterback moves the prediction the right way in every game: {'yes' if ok else 'no'}.**"]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
