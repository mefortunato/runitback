#!/usr/bin/env python
"""Does the lineup model's use of specific defenders carry real information? (teacher forcing, held-out games)"""
import sys, numpy as np, pandas as pd, torch
torch.set_num_threads(4)
from pathlib import Path
from runitback.nfl.data.players import GROUP_IDX
from runitback.nfl.evaluate import load_lineup_data
from runitback.nfl.model.boxscore import expected_play_stats, true_play_stats
from runitback.nfl.model.dataset import collate, encode_games
from runitback.nfl.model.lineups import SLOT_GROUP, other_token
from runitback.nfl.sim.sampler import Simulator
from runitback.nfl.train import split_games
from runitback.nfl.vocab import YARDS_MIN
P = Path("data/nfl/processed")
games, plays, ratings = (pd.read_parquet(P / f) for f in ("games.parquet", "plays.parquet", "ratings.parquet"))
_, val = split_games(games, 2025, 10)
sim = Simulator.load(sys.argv[1]); m = sim.model.eval(); ld = load_lineup_data(sim, P)
items = encode_games(plays[plays.game_id.isin(val.game_id)], val, sim.ts, 400, ratings, ld)

def generic(it, groups):
    it = dict(it); gi = [GROUP_IDX[g] for g in groups]
    lu = it["lineup"].copy()
    for g in gi:
        mask = (SLOT_GROUP == g)[None, :] & (lu > 0)
        lu = np.where(mask, other_token(g), lu)
    pe = it["pers"].copy(); gr = it["pers_grp"]
    for g in gi:
        pe = np.where((gr == g + 1) & (pe > 0), other_token(g), pe)
    it["lineup"], it["pers"] = lu, pe
    return it

def team_stats(its):
    rows = []
    with torch.no_grad():
        for i in range(0, len(its), 25):
            b = collate(its[i:i + 25]); f = b["feats"]
            h = m.encode(f); p = m.personnel(f)
            eo, _ = expected_play_stats(m, h, p, b["type_mask"], f["yl"])
            to, _ = true_play_stats(b["type"], b["result"], b["yards"] + YARDS_MIN, f["yl"])
            v, home = b["valid"], f["is_home"].float()
            for side, w in (("home", home), ("away", 1 - home)):
                ww = (v * w)[..., None]
                e, t = (eo * ww).sum(1), (to * ww).sum(1)
                for k in range(len(e)):
                    rows.append((its[i + k]["game_id"], side, *e[k, 1:3].tolist(), *t[k, 1:3].tolist()))
    return pd.DataFrame(rows, columns=["game_id", "side", "e_rush", "e_pass", "a_rush", "a_pass"])

real = team_stats(items)
for label, groups, stat in (("defensive backs", ["DB"], "pass"), ("pass defense (DB+LB)", ["DB", "LB"], "pass"),
                            ("run defense (DL+LB)", ["DL", "LB"], "rush")):
    gen = team_stats([generic(it, groups) for it in items])
    delta = real[f"e_{stat}"] - gen[f"e_{stat}"]          # what the specific defenders (and own players) change
    resid = real[f"a_{stat}"] - gen[f"e_{stat}"]          # what actually happened vs the generic prediction
    r = np.corrcoef(delta, resid)[0, 1]
    rng = np.random.default_rng(0); n = len(delta)
    bs = [np.corrcoef(delta.values[k], resid.values[k])[0, 1] for k in (rng.integers(0, n, n) for _ in range(2000))]
    mse_g = ((resid) ** 2).mean(); mse_r = ((real[f"a_{stat}"] - real[f"e_{stat}"]) ** 2).mean()
    print(f"{label:22s} {stat} yds: model shift sd {delta.std():5.1f} yds/team-game; corr(shift, actual - generic) "
          f"{r:+.3f} (95% CI {np.percentile(bs, 2.5):+.3f}..{np.percentile(bs, 97.5):+.3f}); "
          f"RMSE specific {np.sqrt(mse_r):.1f} vs generic {np.sqrt(mse_g):.1f}  (n={n})")
