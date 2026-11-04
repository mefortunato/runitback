#!/usr/bin/env python
"""Is the who-gets-the-ball head good when it knows who is on the field? On the held-out games' real
plays (teacher forcing, actual 22 on the field), sum each player's predicted target / carry
probabilities per game and compare with actual targets / carries. Baselines: an equal split among
eligible on-field players (RB/WR/TE for targets, RB/QB for carries) and the form baseline from
scripts/player_stats_eval.py (its saved CSV).

Usage: python scripts/who_head_teacher_forced.py runs/nfl/eval_who/model.pt runs/nfl/eval_who/player_stats_game.csv
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from runitback.nfl.evaluate import load_lineup_data
from runitback.nfl.model.dataset import collate, encode_games
from runitback.nfl.model.lineups import SIDE, WHO_NONE
from runitback.nfl.data.players import GROUP_IDX
from runitback.nfl.sim.sampler import Simulator
from runitback.nfl.train import split_games
from runitback.nfl.vocab import PT

torch.set_num_threads(4)
P = Path("data/nfl/processed")
sim = Simulator.load(sys.argv[1]); m = sim.model.eval(); ld = load_lineup_data(sim, P)
games, plays, ratings = (pd.read_parquet(P / f) for f in ("games.parquet", "plays.parquet", "ratings.parquet"))
_, val = split_games(games, sim.meta["val_season"], sim.meta.get("val_week", 10))
vp = plays[plays.game_id.isin(val.game_id)].sort_values(["game_id", "idx"])
items = encode_games(vp, val, sim.ts, 400, ratings, ld)
pe = ld.personnel.set_index(["game_id", "play_id"])
ELIG_T = {GROUP_IDX[g] + 1 for g in ("RB", "WR", "TE")}
ELIG_C = {GROUP_IDX[g] + 1 for g in ("RB", "QB")}
pred, naive = defaultdict(lambda: np.zeros(2)), defaultdict(lambda: np.zeros(2))
with torch.no_grad():
    for it in items:
        gid = it["game_id"]
        b = collate([it]); f = b["feats"]
        h = m.encode(f); p = m.personnel(f)
        t = b["type"]
        cand = m.who_candidates(f["pers"], f["pers_grp"])
        lg = m.who_logits(h, t, p, cand).masked_fill(~m.who_mask(t, f["pers"]), -1e9)
        pr = torch.softmax(lg, -1)[0].numpy()
        gp = vp[vp.game_id == gid].reset_index(drop=True)
        for i in range(len(it["type"])):
            ty = it["type"][i]
            if ty not in (PT["pass"], PT["run"]) or not (it["pers"][i] > 0).all():
                continue
            row = pe.loc[(gid, int(gp.play_id[i]))]
            ids = list(row.home_ids if gp.pos[i] == 0 else row.away_ids)[:SIDE]
            col = 0 if ty == PT["pass"] else 1
            grps = it["pers_grp"][i, :SIDE]
            elig = [j for j in range(len(ids)) if grps[j] in (ELIG_T if col == 0 else ELIG_C)]
            for j, pid in enumerate(ids):
                pred[(gid, pid)][col] += pr[i, j]
                if j in elig:
                    naive[(gid, pid)][col] += (1 - (pr[i, WHO_NONE] if col == 0 else 0)) / len(elig)
df = pd.read_csv(sys.argv[2])
df["tf_targets"] = [pred[(g, p)][0] for g, p in zip(df.game_id, df.player_id)]
df["tf_carries"] = [pred[(g, p)][1] for g, p in zip(df.game_id, df.player_id)]
df["naive_targets"] = [naive[(g, p)][0] for g, p in zip(df.game_id, df.player_id)]
df["naive_carries"] = [naive[(g, p)][1] for g, p in zip(df.game_id, df.player_id)]
rm = lambda d, c, k: np.sqrt(((d[c] - d[k]) ** 2).mean())
print("RMSE vs actual (teacher forcing = real plays and real on-field players, so all columns but 'form' know the play count)")
for grp, ks in (("RB", ("targets", "carries")), ("WR", ("targets",)), ("TE", ("targets",))):
    d = df[df.group == grp]
    for k in ks:
        print(f"  {grp} {k:8s} who head {rm(d, f'tf_{k}', k):5.2f} | equal split on field {rm(d, f'naive_{k}', k):5.2f} | "
              f"form+opp {rm(d, f'formopp_{k}', k):5.2f} | simulated model {rm(d, f'sim_{k}', k):5.2f}  (n={len(d)})")
