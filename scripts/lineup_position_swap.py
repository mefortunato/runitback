"""Does changing the lineup move the simulated games, at every position? The position-swap check.

For games of one season, roll out a foundation model with each team's previous-game lineup, then
again with one change to the home team's depth chart, on the same random seeds:
    best / worst  that group's first slot (QB1, RB1, WR1, TE1, OL1, DL1, LB1, DB1, K, P) given to
                  the best / worst regular by the direct player values fit before the season
                  (experiments/nfl/allpositions/swap_refs.csv)
    new           that slot given to the replacement-level "new player" token
    out           the slot's starter ruled out: the rest of the group moves up one slot
Prints the change in the home team's win probability and margin per condition.

    python scripts/lineup_position_swap.py --ckpt runs/ae/fold2025/model.pt --season 2025 --games 120
    python scripts/lineup_position_swap.py --summary out/ae/position_swap.*.csv
"""
from __future__ import annotations

import argparse
import copy
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

log = logging.getLogger("position_swap")
DATA = Path("data/nfl/processed")
GROUPS = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB", "K", "P"]
KINDS = ["best", "worst", "new", "out"]


def rollouts(sim, row, pre, lineups, sims: int, seed: int) -> np.ndarray:
    from runitback.nfl.engine.rules import new_game

    finals = []
    for k, n in ((0, sims // 2), (1, sims - sims // 2)):
        start = new_game(row.home, row.away, row.season, row.game_type, opening_kicker=k)
        f, _ = sim.rollout(start, n, seed=seed + k, ratings=pre, lineups=lineups)
        finals += f
    return np.array([f.score[0] - f.score[1] for f in finals], dtype=float)


def summarize(df: pd.DataFrame, refs: pd.DataFrame | None = None) -> None:
    names = {} if refs is None else {(r.group, r.kind): r["name"] for _, r in refs.iterrows()}
    print(f"{len(df)} games; one change to the home team's depth chart vs its previous-game lineup "
          "(Δ home win, Δ margin in points, 95% CI):")
    print(f"  {'slot':5s} {'change':6s} {'player':24s} {'Δ win':>16s} {'Δ margin':>16s}")
    for g in GROUPS:
        for k in KINDS:
            c = f"{g}_{k}"
            if f"win_{c}" not in df:
                continue
            d = (df[f"win_{c}"] - df.win_base).dropna()
            dm = (df[f"margin_{c}"] - df.margin_base).dropna()
            ci = lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x))
            print(f"  {g:5s} {k:6s} {names.get((g, k), '')[:24]:24s} {d.mean():+.3f} ± {ci(d):.3f}"
                  f"   {dm.mean():+.2f} ± {ci(dm):.2f}")


def main() -> None:
    from runitback.nfl.evaluate import game_lineups, load_lineup_data
    from runitback.nfl.model.lineups import SLOTS, SLOT_IDX, new_token
    from runitback.nfl.sim.sampler import Simulator, scale_ratings

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--season", type=int)
    ap.add_argument("--summary", type=Path, nargs="+", help="print saved runs (shards are concatenated)")
    ap.add_argument("--refs", type=Path, default=Path("experiments/nfl/allpositions/swap_refs.csv"))
    ap.add_argument("--groups", nargs="+", default=GROUPS)
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("out/position_swap.csv"))
    a = ap.parse_args()
    refs = pd.read_csv(a.refs)
    if a.summary:
        return summarize(pd.concat([pd.read_csv(f) for f in a.summary], ignore_index=True), refs)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if a.threads:
        torch.set_num_threads(a.threads)

    sim = Simulator.load(a.ckpt)
    ld = load_lineup_data(sim, DATA)
    games = pd.read_parquet(DATA / "games.parquet")
    ratings = pd.read_parquet(DATA / "ratings.parquet")
    g = games[(games.season == a.season) & (games.week > 1)].sample(a.games, random_state=0)
    career = ld.pg.groupby("player_id").career_games.max()
    tok = {}
    for r in refs.itertuples():
        t = ld.pvocab.token(r.player_id, r.group, float(career.get(r.player_id, 0)))
        if t != ld.pvocab.idx.get(r.player_id):
            log.warning("%s %s (%s) has no embedding in this model: skipped", r.group, r.kind, r.name)
            continue
        tok[(r.group, r.kind)] = t
    slots_of = {grp: [i for i, (gg, _) in enumerate(SLOTS) if gg == grp] for grp in a.groups}

    def changed(lu, grp, kind):
        x = copy.deepcopy(lu)
        row = x.tokens[0]
        s0 = SLOT_IDX[(grp, 0)]
        if row[s0] == 0:
            return None
        if kind in ("best", "worst"):
            t = tok.get((grp, kind))
            if t is None:
                return None
            if t in row[slots_of[grp]]:            # already on the team: swap him into the slot
                row[slots_of[grp]] = np.where(row[slots_of[grp]] == t, row[s0], row[slots_of[grp]])
            row[s0] = t
        elif kind == "new":
            row[s0] = new_token(grp)
        else:                                       # out: the group moves up one slot
            s = slots_of[grp]
            row[s] = np.concatenate([row[s][1:], [0]])
        return x

    rows, t0 = [], time.time()
    for i, row in enumerate(g.itertuples()):
        if i % a.shards != a.shard:
            continue
        rg = ratings[ratings.game_id == row.game_id].set_index("side")
        pre = tuple(scale_ratings({k: float(rg.at[s, f"pre_{k}"]) for k in ("off_epa", "def_epa", "elo")}, a.scale)
                    for s in ("home", "away"))
        lu = game_lineups(ld, row, "previous")
        out = dict(game_id=row.game_id, home=row.home)
        m = rollouts(sim, row, pre, lu, a.sims, 1000 * i)
        out["win_base"], out["margin_base"] = float((m > 0).mean() + 0.5 * (m == 0).mean()), float(m.mean())
        for grp in a.groups:
            for kind in KINDS:
                x = changed(lu, grp, kind)
                if x is None:
                    continue
                m = rollouts(sim, row, pre, x, a.sims, 1000 * i)       # same seeds in every condition
                out[f"win_{grp}_{kind}"] = float((m > 0).mean() + 0.5 * (m == 0).mean())
                out[f"margin_{grp}_{kind}"] = float(m.mean())
        rows.append(out)
        log.info("%d games (%.0fs)", len(rows), time.time() - t0)
    df = pd.DataFrame(rows)
    if a.shards > 1 and a.out.suffix == ".csv":
        a.out = a.out.with_name(f"{a.out.stem}.{a.shard}.csv")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    summarize(df, refs)


if __name__ == "__main__":
    main()
