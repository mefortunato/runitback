#!/usr/bin/env python
"""Compare simulated player stat lines from two runs of player_stats_eval.py (same players and games),
with a paired game-level bootstrap. Usage: python scripts/compare_samplers.py A.csv B.csv"""
import sys
import numpy as np
import pandas as pd

a, b = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
sim_cols = [c for c in b if c.startswith("sim_")]
m = b.merge(a[["game_id", "player_id"] + sim_cols], on=["game_id", "player_id"], suffixes=("", "_a"))
rng = np.random.default_rng(0)
print(f"{len(m)} player-games. RMSE: A -> B | form+opp | B-A MSE (95% CI)")
for g, ks in (("RB", ("carries", "rush_yds", "targets", "rec_yds")), ("WR", ("targets", "receptions", "rec_yds")),
              ("TE", ("targets", "receptions", "rec_yds"))):
    d = m[m.group == g]
    gids = d.game_id.to_numpy(); ug = np.unique(gids)
    for k in ks:
        ea, eb, ef = ((d[c] - d[k]) ** 2 for c in (f"sim_{k}_a", f"sim_{k}", f"formopp_{k}"))
        diff = (eb - ea).to_numpy()
        per = pd.Series(diff).groupby(gids).agg(["sum", "count"]).loc[ug]
        bs = [per["sum"].to_numpy()[i].sum() / per["count"].to_numpy()[i].sum()
              for i in (rng.integers(0, len(ug), len(ug)) for _ in range(1000))]
        print(f"  {g} {k:10s} {np.sqrt(ea.mean()):6.2f} -> {np.sqrt(eb.mean()):6.2f} | {np.sqrt(ef.mean()):6.2f} | "
              f"{diff.mean():+8.2f} ({np.percentile(bs, 2.5):+.2f}..{np.percentile(bs, 97.5):+.2f})")
print("\nmean by pre-game role: A / B / actual")
for g, ranks, k in (("RB", 2, "carries"), ("RB", 1, "rush_yds"), ("WR", 3, "targets"), ("TE", 2, "targets")):
    d = m[m.group == g]
    for r in range(ranks):
        y = d[d.depth == r]
        print(f"  {g}{r + 1} {k:9s} {y[f'sim_{k}_a'].mean():5.2f} / {y[f'sim_{k}'].mean():5.2f} / {y[k].mean():5.2f}")
