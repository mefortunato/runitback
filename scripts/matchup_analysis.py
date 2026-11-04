#!/usr/bin/env python
"""Is there a WR-vs-CB matchup signal in the data, beyond receiver form and team pass defense?

Leak-free: every feature for a game uses only earlier games. Fit on 2013-2022, test on 2023-2025.
Usage: python matchup_data.py <offense group> <defense position set> e.g. WR CB
"""
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

OFF_GROUP = sys.argv[1] if len(sys.argv) > 1 else "WR"
DEF_POS = set((sys.argv[2] if len(sys.argv) > 2 else "CB").split(","))
RANK = int(sys.argv[3]) if len(sys.argv) > 3 else 0          # 0 = WR1/CB1
STAT = sys.argv[4] if len(sys.argv) > 4 else "rec_yds"
ALPHA, K = 0.2, 3.0
P = "data/nfl/processed"
pg = pd.read_parquet(f"{P}/player_games.parquet")
games = pd.read_parquet(f"{P}/games.parquet")[["game_id", "season", "week", "home", "away"]]
games = games[games.season.between(2012, 2025)]

# per-player per-game receiving / rushing production from play-by-play
frames = []
for s in range(2012, 2026):
    d = pd.read_parquet(f"data/nfl/raw/pbp_{s}.parquet",
                        columns=["game_id", "play_type", "receiver_player_id", "receiving_yards", "complete_pass",
                                 "rusher_player_id", "rushing_yards"])
    rec = d[d.receiver_player_id.notna()].groupby(["game_id", "receiver_player_id"]).agg(
        rec_yds=("receiving_yards", lambda x: x.fillna(0).sum()), rec=("complete_pass", "sum"),
        tgt=("complete_pass", "size")).reset_index().rename(columns={"receiver_player_id": "player_id"})
    rush = d[d.rusher_player_id.notna() & (d.play_type == "run")].groupby(["game_id", "rusher_player_id"]).agg(
        rush_yds=("rushing_yards", lambda x: x.fillna(0).sum()), car=("rushing_yards", "size")).reset_index().rename(
        columns={"rusher_player_id": "player_id"})
    frames.append(rec.merge(rush, on=["game_id", "player_id"], how="outer"))
prod = pd.concat(frames).fillna(0)

pg = pg[pg.game_id.isin(games.game_id)]
off = pg[(pg.group == OFF_GROUP) & (pg.depth == RANK) & (pg.offense_snaps > 0)]
off = off.merge(prod, on=["game_id", "player_id"], how="left").fillna({STAT: 0})
dfn = pg[pg.position.isin(DEF_POS) & (pg.defense_snaps > 0)].sort_values(
    ["game_id", "team", "defense_snaps"], ascending=[True, True, False])
dfn["crank"] = dfn.groupby(["game_id", "team"]).cumcount()
cb = dfn[dfn.crank == RANK][["game_id", "team", "player_id", "name"]].rename(
    columns={"team": "def_team", "player_id": "cb_id", "name": "cb_name"})
g2 = games.set_index("game_id")
off["def_team"] = [g2.at[g, "away"] if t == g2.at[g, "home"] else g2.at[g, "home"] for g, t in zip(off.game_id, off.team)]
df = off.merge(cb, on=["game_id", "def_team"], how="inner")
df = df.sort_values(["season", "week", "game_id"]).reset_index(drop=True)

# usual CB1 of each defense = its CB1 in its previous game; absent if he has no snaps in this game
played = set(zip(pg.game_id, pg.player_id))
last_cb1 = {}
usual_absent = []
for r in df.itertuples():
    u = last_cb1.get(r.def_team)
    usual_absent.append(u is not None and u != r.cb_id and (r.game_id, u) not in played)
cbs_by_game = {(r.game_id, r.def_team): r.cb_id for r in df.itertuples()}
# compute last_cb1 chronologically by defense team game order
order = df.drop_duplicates(["game_id", "def_team"])[["game_id", "def_team", "cb_id", "season", "week"]]
prev_cb1 = {}
last = {}
for r in order.sort_values(["season", "week"]).itertuples():
    prev_cb1[(r.game_id, r.def_team)] = last.get(r.def_team)
    last[r.def_team] = r.cb_id
df["usual_cb1"] = [prev_cb1.get((g, t)) for g, t in zip(df.game_id, df.def_team)]
df["usual_absent"] = [u is not None and u != c and (g, u) not in played for g, u, c in zip(df.game_id, df.usual_cb1, df.cb_id)]

# rolling leak-free features
role_mean = df[df.season <= 2022][STAT].mean()
wr_state, def_state, cb_state = {}, {}, {}

def get(st, k):
    v = st.get(k)
    return (v[0], v[1]) if v else (0.0, 0)

wr_f, def_f, cb_f, cb_n = [], [], [], []
for (season, week), wk in df.groupby(["season", "week"], sort=True):
    rows = []
    for r in wk.itertuples():
        m, n = get(wr_state, r.player_id)
        wr = (n * m + K * role_mean) / (n + K) if n else role_mean
        dm, dn = get(def_state, r.def_team)
        cm, cn = get(cb_state, r.cb_id)
        wr_f.append(wr); def_f.append(dm * dn / (dn + K)); cb_f.append(cm * cn / (cn + K)); cb_n.append(cn)
        rows.append((r, wr))
    for r, wr in rows:  # update after the whole week (games in a week are simultaneous)
        y = getattr(r, STAT)
        m, n = get(wr_state, r.player_id)
        wr_state[r.player_id] = ((1 - ALPHA) * m + ALPHA * y if n else y, n + 1)
        resid = y - wr
        for st, k in ((def_state, r.def_team), (cb_state, r.cb_id)):
            m, n = get(st, k)
            st[k] = ((1 - ALPHA) * m + ALPHA * resid if n else resid, n + 1)
df["wr_prior"], df["def_prior"], df["cb_prior"], df["cb_n"] = wr_f, def_f, cb_f, cb_n
df = df[df.season >= 2013]
train, test = df[df.season <= 2022], df[df.season >= 2023]

def fit(cols):
    X = np.c_[np.ones(len(train)), train[cols].to_numpy()]
    b = np.linalg.lstsq(X, train[STAT].to_numpy(), rcond=None)[0]
    return b, np.c_[np.ones(len(test)), test[cols].to_numpy()] @ b

y = test[STAT].to_numpy()
models = {"WR form only": ["wr_prior"], "+ opp team vs WR": ["wr_prior", "def_prior"],
          "+ opp CB track record": ["wr_prior", "def_prior", "cb_prior"]}
preds = {}
print(f"\n{OFF_GROUP}{RANK+1} {STAT} vs opponent {'/'.join(sorted(DEF_POS))}{RANK+1}: train {len(train)} team-games, test {len(test)} (2023-2025)")
print(f"  mean {y.mean():.1f}, sd {y.std():.1f}")
for name, cols in models.items():
    b, p = fit(cols)
    preds[name] = p
    print(f"  {name:24s} RMSE {np.sqrt(((p - y) ** 2).mean()):6.2f}  R2 {1 - ((p - y) ** 2).mean() / y.var():.4f}  coefs {np.round(b[1:], 3)}")
rng = np.random.default_rng(0)
gid = test.game_id.to_numpy(); ug = np.unique(gid); idx = {g: np.flatnonzero(gid == g) for g in ug}
def boot(a, b):
    d = (preds[a] - y) ** 2 - (preds[b] - y) ** 2
    pergame = np.array([d[idx[g]].sum() for g in ug]); cnt = np.array([len(idx[g]) for g in ug])
    bs = [ (pergame[k].sum() / cnt[k].sum()) for k in (rng.integers(0, len(ug), len(ug)) for _ in range(2000))]
    return d.mean(), np.percentile(bs, [2.5, 97.5])
for a, b in (("WR form only", "+ opp team vs WR"), ("+ opp team vs WR", "+ opp CB track record")):
    m, ci = boot(a, b)
    print(f"  MSE gain {b!r} over {a!r}: {m:+.2f} (95% CI {ci[0]:+.2f}..{ci[1]:+.2f})")
# CB identity effect, ridge-shrunk on training residuals, applied to test CBs seen in training
_, p1 = fit(["wr_prior", "def_prior"])
Xtr = np.c_[np.ones(len(train)), train[["wr_prior", "def_prior"]].to_numpy()]
btr = np.linalg.lstsq(Xtr, train[STAT].to_numpy(), rcond=None)[0]
res_tr = train[STAT].to_numpy() - Xtr @ btr
agg = pd.DataFrame({"cb": train.cb_id, "r": res_tr}).groupby("cb").r.agg(["sum", "count"])
for lam in (5, 20, 50):
    eff = (agg["sum"] / (agg["count"] + lam)).to_dict()
    p2 = p1 + np.array([eff.get(c, 0.0) for c in test.cb_id])
    seen = test.cb_id.isin(eff).to_numpy()
    print(f"  CB identity (ridge {lam:2d}): test RMSE {np.sqrt(((p2 - y) ** 2).mean()):6.2f} vs {np.sqrt(((p1 - y) ** 2).mean()):6.2f}"
          f" ({seen.mean():.0%} of test CBs seen); effect sd {np.std(list(eff.values())):.2f}")
# absence of the defense's usual CB1
r1 = y - p1
ab = test.usual_absent.to_numpy()
diff = r1[ab].mean() - r1[~ab].mean()
se = np.sqrt(r1[ab].var() / ab.sum() + r1[~ab].var() / (~ab).sum())
print(f"  usual CB1 absent ({ab.sum()} games): residual {r1[ab].mean():+.2f} vs present {r1[~ab].mean():+.2f}; diff {diff:+.2f} ± {1.96*se:.2f}")
# split-half reliability of CB track record: does a CB's allowed-residual in one season predict the next?
tr2 = df.assign(res=df[STAT] - (np.c_[np.ones(len(df)), df[["wr_prior", "def_prior"]].to_numpy()] @ btr))
cs = tr2.groupby(["cb_id", "season"]).res.agg(["mean", "count"]).reset_index()
cs = cs[cs["count"] >= 8]
nxt = cs.assign(season=cs.season - 1).rename(columns={"mean": "next", "count": "n2"})
pair = cs.merge(nxt, on=["cb_id", "season"])
print(f"  CB year-to-year correlation of allowed residual (>=8 games both seasons, {len(pair)} pairs): "
      f"{np.corrcoef(pair['mean'], pair['next'])[0, 1]:+.3f}")
tm = tr2.groupby(["def_team", "season"]).res.mean().reset_index()
tn = tm.assign(season=tm.season - 1).rename(columns={"res": "next"})
tp = tm.merge(tn, on=["def_team", "season"])
print(f"  team year-to-year correlation of allowed residual ({len(tp)} pairs): {np.corrcoef(tp.res, tp.next)[0, 1]:+.3f}")
wm = tr2.groupby(["player_id", "season"]).agg(m=(STAT, "mean"), n=(STAT, "size")).reset_index()
wm = wm[wm.n >= 8]; wn = wm.assign(season=wm.season - 1)
wp = wm.merge(wn, on=["player_id", "season"])
print(f"  {OFF_GROUP} year-to-year correlation of {STAT} ({len(wp)} pairs): {np.corrcoef(wp.m_x, wp.m_y)[0, 1]:+.3f}")
