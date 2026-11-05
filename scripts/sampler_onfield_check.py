#!/usr/bin/env python
"""On the held-out games' real scrimmage plays: how often is each team's pre-game RB1/RB2, WR1-3,
TE1/TE2 actually on the field, vs. how often each personnel sampler puts them there in the same
situations (league rotation by slot, and player-specific usage)?
Usage: python scripts/sampler_onfield_check.py runs/nfl/eval_who/model.pt"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from runitback.nfl.data.players import DEPTH
from runitback.nfl.evaluate import load_lineup_data
from runitback.nfl.model.lineups import SLOT_IDX, role_share_table, slot_usage, usage_multipliers
from runitback.nfl.sim.sampler import Simulator
from runitback.nfl.train import split_games

P = Path("data/nfl/processed")
sim = Simulator.load(sys.argv[1]); ld = load_lineup_data(sim, P); smp = sim.lineups.sampler
games = pd.read_parquet(P / "games.parquet"); plays = pd.read_parquet(P / "plays.parquet")
train_g, val = split_games(games, 2025, 10)
pg = ld.pg
role = role_share_table(pg[pg.game_id.isin(train_g.game_id) & (pg.season >= 2016)])
mult_before, _ = usage_multipliers(pg, role)
rng = np.random.default_rng(0)

def pregame(gid, team):
    rows = pg[(pg.game_id == gid) & (pg.team == team)]
    prev = ld._prev_game.get((gid, team))
    prank = {} if prev is None else {(r.group, r.player_id): r.depth for r in pg[(pg.game_id == prev) & (pg.team == team)].itertuples()}
    toks, smap = np.zeros(len(SLOT_IDX), np.int64), {}
    for grp, d in rows.groupby("group"):
        for rank, r in enumerate(sorted(d.itertuples(), key=lambda r: (prank.get((grp, r.player_id), 99), -r.career_games))[:DEPTH[grp]]):
            sl = SLOT_IDX[(grp, rank)]; toks[sl] = ld.pvocab.token(r.player_id, grp, r.career_games); smap[sl] = r.player_id
    return toks, smap

roles = [("RB", 0), ("RB", 1), ("WR", 0), ("WR", 1), ("WR", 2), ("TE", 0), ("TE", 1)]
acc = {k: np.zeros(3) for k in roles}; cnt = {k: 0 for k in roles}
pe = ld.personnel.merge(plays[["game_id", "play_id", "play_type", "down", "ytg", "yl", "pos"]], on=["game_id", "play_id"])
pe = pe[pe.game_id.isin(val.game_id) & pe.play_type.isin(["run", "pass"])]
g2 = val.set_index("game_id")
charts = {}
for r in pe.itertuples():
    team = g2.at[r.game_id, "home"] if r.pos == 0 else g2.at[r.game_id, "away"]
    if (r.game_id, team) not in charts:
        t, m = pregame(r.game_id, team)
        u = slot_usage(t, role, lambda sl, m=m, g=r.game_id: mult_before.get((g, m.get(sl)), 1.0))
        charts[(r.game_id, team)] = (t, m, u)
    toks, smap, usage = charts[(r.game_id, team)]
    on_field = set(list(r.home_ids if r.pos == 0 else r.away_ids)[:11])
    _, _, c_league = smp.sample("off", r.play_type, int(r.down), int(r.ytg), int(r.yl), toks, rng, codes=True)
    _, _, c_usage = smp.sample("off", r.play_type, int(r.down), int(r.ytg), int(r.yl), toks, rng, codes=True, usage=usage)
    for k in roles:
        sl = SLOT_IDX[k]
        if sl not in smap:
            continue
        acc[k] += [smap[sl] in on_field, sl in c_league, sl in c_usage]; cnt[k] += 1
print(f"on-field rate on {len(pe)} held-out run/pass plays (pre-game depth chart ranks): actual | league rotation | player usage")
for k in roles:
    a = acc[k] / max(cnt[k], 1)
    print(f"  {k[0]}{k[1] + 1}: {a[0]:.3f} | {a[1]:.3f} | {a[2]:.3f}")
