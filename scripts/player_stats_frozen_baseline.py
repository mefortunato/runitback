#!/usr/bin/env python
"""Apples-to-apples check for scripts/player_stats_eval.py: the model's player embeddings stop at the
training cutoff, while the 'form' baseline keeps updating through the held-out weeks. This recomputes
form (shrunk EWMA, same settings) frozen at the cutoff and compares it with the model's saved
predictions.

Usage: python scripts/player_stats_frozen_baseline.py runs/nfl/eval_who/player_stats_previous.csv [cutoff_season cutoff_week]
"""
import sys

import numpy as np
import pandas as pd

CSV = sys.argv[1]
CUT_SEASON, CUT_WEEK = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (2025, 10)
STATS = ["targets", "receptions", "rec_yds", "carries", "rush_yds"]
ALPHA, K = 0.2, 3.0
df = pd.read_csv(CSV)
pg = pd.read_parquet("data/nfl/processed/player_games.parquet")
pg = pg[pg.group.isin(["RB", "WR", "TE"])]
before = (pg.season < CUT_SEASON) | ((pg.season == CUT_SEASON) & (pg.week < CUT_WEEK))
frames = []
for s in range(2012, CUT_SEASON + 1):
    d = pd.read_parquet(f"data/nfl/raw/pbp_{s}.parquet",
                        columns=["game_id", "play_type", "receiver_player_id", "receiving_yards", "complete_pass",
                                 "rusher_player_id", "rushing_yards"])
    rec = d[(d.play_type == "pass") & d.receiver_player_id.notna()].groupby(["game_id", "receiver_player_id"]).agg(
        targets=("complete_pass", "size"), receptions=("complete_pass", "sum"),
        rec_yds=("receiving_yards", lambda x: x.fillna(0).sum())).reset_index().rename(columns={"receiver_player_id": "player_id"})
    rush = d[(d.play_type == "run") & d.rusher_player_id.notna()].groupby(["game_id", "rusher_player_id"]).agg(
        carries=("rushing_yards", "size"), rush_yds=("rushing_yards", lambda x: x.fillna(0).sum())).reset_index().rename(
        columns={"rusher_player_id": "player_id"})
    frames.append(rec.merge(rush, on=["game_id", "player_id"], how="outer"))
prod = pd.concat(frames).fillna(0)
hist = pg[before].merge(prod, on=["game_id", "player_id"], how="left").fillna({k: 0 for k in STATS})
hist = hist.sort_values(["season", "week"])
role = hist[hist.season >= 2016].groupby(["group", "depth"])[STATS].mean()
state = {}
for r in hist[["player_id", *STATS]].itertuples(index=False):
    m, n = state.get(r.player_id, (None, 0))
    y = np.array(r[1:], dtype=float)
    state[r.player_id] = (y if m is None else (1 - ALPHA) * m + ALPHA * y, n + 1)
ws = df.merge(pg[["game_id", "player_id", "group", "depth"]].drop_duplicates(["game_id", "player_id"]),
              on=["game_id", "player_id"], how="left", suffixes=("", "_pg"))
for j, k in enumerate(STATS):
    vals = []
    for pid, grp, dp in zip(ws.player_id, ws.group, ws.depth):
        rm = role.loc[(grp, min(dp, role.loc[grp].index.max()))][k]
        m, n = state.get(pid, (None, 0))
        vals.append((n * m[j] + K * rm) / (n + K) if n else rm)
    ws[f"frozen_{k}"] = vals
print(f"{CSV}: {len(ws)} player-games; RMSE (lower is better)")
for grp in ("RB", "WR", "TE"):
    d = ws[ws.group == grp]
    for k in STATS:
        if grp != "RB" and k in ("carries", "rush_yds"):
            continue
        rm = lambda c: np.sqrt(((d[c] - d[k]) ** 2).mean())
        print(f"  {grp} {k:10s} model {rm(f'sim_{k}'):6.2f} | form frozen at cutoff {rm(f'frozen_{k}'):6.2f} | "
              f"form updated weekly {rm(f'form_{k}'):6.2f} | role {rm(f'role_{k}'):6.2f}")
