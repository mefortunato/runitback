#!/usr/bin/env python
"""Player stat lines from the who-gets-the-ball model vs actual, on the held-out games.

For every held-out game, simulate N games from each team's depth chart, attribute each run (carry)
and pass target to the player in the chosen depth-chart slot, and average per player. Compare with
actual stat lines and with leak-free baselines:

  role       mean of the player's depth-chart role (e.g. WR1) in training seasons
  form       the player's own exponentially weighted average over his earlier games, shrunk to role
  form+opp   form, adjusted by the opponent's tendency to allow that stat to that position
             (linear fit on training seasons)

Usage: python scripts/player_stats_eval.py runs/nfl/eval_who/model.pt [sims=100] [out.json] [max_games|all]
                                          [sources=previous,pregame,game]

Depth-chart sources: previous = each team's previous game (ranked by its snaps); pregame = the players
active for this game (known ~90 min before kickoff) ranked by their previous-game role, newcomers by
career games; game = this game's own snap ranks (optimistic: knows how usage actually split).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from runitback.nfl.data.ratings import scale_ratings
from runitback.nfl.engine.rules import new_game
from runitback.nfl.evaluate import load_lineup_data
from runitback.nfl.model.lineups import (
    N_SLOTS, N_USAGE, SLOT_IDX, SLOTS, role_share_table, slot_usage, usage_multipliers,
)
from runitback.nfl.sim.sampler import GameLineups, Simulator
from runitback.nfl.train import split_games

torch.set_num_threads(4)
CKPT = Path(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else CKPT.parent / "player_stats.json"
MAX_GAMES = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "all" else None  # smoke tests
SOURCES = sys.argv[5].split(",") if len(sys.argv) > 5 else ["previous", "pregame", "game"]
# a "+usage" suffix (e.g. pregame+usage) picks on-field players by player-specific snap shares
# (league role share x the player's usage multiplier from his snap counts before the game)
P = Path("data/nfl/processed")
STATS = ["targets", "receptions", "rec_yds", "carries", "rush_yds"]
GROUPS = ["RB", "WR", "TE"]
ALPHA, K = 0.2, 3.0

games = pd.read_parquet(P / "games.parquet")
ratings = pd.read_parquet(P / "ratings.parquet")
sim = Simulator.load(CKPT)
assert sim.model.has_who, "checkpoint has no who-gets-the-ball head (train with --who)"
ld = load_lineup_data(sim, P)
train_g, val = split_games(games, sim.meta["val_season"], sim.meta.get("val_week", 10))
if MAX_GAMES:
    val = val.head(MAX_GAMES)
pg = ld.pg

# ------------------------------------------------------------------ actual per-player game stats
frames = []
for s in range(2012, int(val.season.max()) + 1):
    d = pd.read_parquet(f"data/nfl/raw/pbp_{s}.parquet",
                        columns=["game_id", "play_type", "receiver_player_id", "receiving_yards", "complete_pass",
                                 "rusher_player_id", "rushing_yards"])
    rec = d[(d.play_type == "pass") & d.receiver_player_id.notna()].groupby(["game_id", "receiver_player_id"]).agg(
        targets=("complete_pass", "size"), receptions=("complete_pass", "sum"),
        rec_yds=("receiving_yards", lambda x: x.fillna(0).sum())).reset_index().rename(
        columns={"receiver_player_id": "player_id"})
    rush = d[(d.play_type == "run") & d.rusher_player_id.notna()].groupby(["game_id", "rusher_player_id"]).agg(
        carries=("rushing_yards", "size"), rush_yds=("rushing_yards", lambda x: x.fillna(0).sum())).reset_index().rename(
        columns={"rusher_player_id": "player_id"})
    frames.append(rec.merge(rush, on=["game_id", "player_id"], how="outer"))
prod = pd.concat(frames).fillna(0)

# every game a player appeared in (snap counts), with his stats (0 if none) and role
app = pg[pg.group.isin(GROUPS)][["game_id", "season", "week", "team", "player_id", "name", "group", "depth"]]
app = app.merge(prod, on=["game_id", "player_id"], how="left").fillna({k: 0 for k in STATS})
g2 = games.set_index("game_id")
app = app[app.game_id.isin(g2.index)]
app["opp"] = [g2.at[g, "away"] if t == g2.at[g, "home"] else g2.at[g, "home"] for g, t in zip(app.game_id, app.team)]
app = app.sort_values(["season", "week", "game_id"]).reset_index(drop=True)

# ------------------------------------------------------------------ leak-free rolling baselines
is_train = app.game_id.isin(train_g.game_id)
role_mean = app[is_train & app.season.ge(2016)].groupby(["group", "depth"])[STATS].mean()
form = {k: np.zeros(len(app)) for k in STATS}
oppf = {k: np.zeros(len(app)) for k in STATS}
pstate, ostate = {}, {}
pid_a, opp_a, grp_a = app.player_id.to_numpy(), app.opp.to_numpy(), app.group.to_numpy()
max_depth = {gname: role_mean.loc[gname].index.max() for gname in GROUPS}
role_a = np.stack([role_mean.loc[(gname, min(dp, max_depth[gname]))].to_numpy()
                   for gname, dp in zip(grp_a, app.depth.to_numpy())])           # (rows, stats)
y_a = app[STATS].to_numpy(dtype=float)
week_key = (app.season * 100 + app.week).to_numpy()
bounds = np.flatnonzero(np.r_[True, week_key[1:] != week_key[:-1], True])
for a, b in zip(bounds[:-1], bounds[1:]):
    for i in range(a, b):
        for j, k in enumerate(STATS):
            m, n = pstate.get((pid_a[i], j), (0.0, 0))
            form[k][i] = (n * m + K * role_a[i, j]) / (n + K)
            om, on = ostate.get((opp_a[i], grp_a[i], j), (0.0, 0))
            oppf[k][i] = om * on / (on + K)
    for i in range(a, b):  # update after the whole week (its games are simultaneous)
        for j, k in enumerate(STATS):
            y = y_a[i, j]
            m, n = pstate.get((pid_a[i], j), (0.0, 0))
            pstate[(pid_a[i], j)] = ((1 - ALPHA) * m + ALPHA * y if n else y, n + 1)
            res = y - form[k][i]
            om, on = ostate.get((opp_a[i], grp_a[i], j), (0.0, 0))
            ostate[(opp_a[i], grp_a[i], j)] = ((1 - ALPHA) * om + ALPHA * res if on else res, on + 1)
for k in STATS:
    app[f"form_{k}"], app[f"opp_{k}"] = form[k], oppf[k]
fit_rows = app[is_train & app.season.ge(2016)]
coef = {}
for k in STATS:
    for gname in GROUPS:
        d = fit_rows[fit_rows.group == gname]
        X = np.c_[np.ones(len(d)), d[f"form_{k}"], d[f"opp_{k}"]]
        coef[(gname, k)] = np.linalg.lstsq(X, d[k].to_numpy(), rcond=None)[0]

# ------------------------------------------------------------------ simulate held-out games
_slot_maps: dict[tuple[str, str], dict[int, str]] = {}
for (gid_, team_), rows_ in ld.pg_tokens.groupby(["game_id", "team"]):
    _slot_maps[(gid_, team_)] = dict(zip(rows_.slot, rows_.player_id))


def slot_players(game_id: str, team: str) -> dict[int, str]:
    return _slot_maps.get((game_id, team), {})


def pregame_chart(game_id: str, team: str) -> tuple[np.ndarray, dict[int, str]]:
    """This game's active players, ordered by their role in the team's previous game."""
    from runitback.nfl.data.players import DEPTH
    rows = pg[(pg.game_id == game_id) & (pg.team == team)]
    prev = ld._prev_game.get((game_id, team))
    prank = {} if prev is None else {(r.group, r.player_id): r.depth for r in
                                     pg[(pg.game_id == prev) & (pg.team == team)].itertuples()}
    toks, smap = np.zeros(N_SLOTS, dtype=np.int32), {}
    for grp, d in rows.groupby("group"):
        order = sorted(d.itertuples(), key=lambda r: (prank.get((grp, r.player_id), 99), -r.career_games))
        for rank, r in enumerate(order[: DEPTH[grp]]):
            sl = SLOT_IDX[(grp, rank)]
            toks[sl] = ld.pvocab.token(r.player_id, grp, r.career_games)
            smap[sl] = r.player_id
    return toks, smap


ROLE = role_share_table(pg[pg.game_id.isin(train_g.game_id) & (pg.season >= 2016)])
MULT_BEFORE, _ = usage_multipliers(pg, ROLE)  # leak-free: each game's value uses earlier snap counts only


def simulate(source: str) -> pd.DataFrame:
    out, t0 = [], time.time()
    use_usage = source.endswith("+usage")
    source = source.removesuffix("+usage")
    for i, g in enumerate(val.itertuples()):
        maps, toks = [], []
        for team in (g.home, g.away):
            if source == "pregame":
                t_, m_ = pregame_chart(g.game_id, team)
                toks.append(t_), maps.append(m_)
                continue
            src_game = g.game_id if source == "game" else ld._prev_game.get((g.game_id, team))
            maps.append(slot_players(src_game, team) if src_game else {})
            toks.append(ld.lineup(g.game_id, team) if source == "game" else ld.previous_lineup(g.game_id, team))
        usage = None
        if use_usage:
            usage = [slot_usage(t_, ROLE, lambda sl, m_=m_: MULT_BEFORE.get((g.game_id, m_.get(sl)), 1.0))
                     for t_, m_ in zip(toks, maps)]
        # recent-usage features of each depth-chart player as of this game (earlier games only)
        feats = []
        for m_ in maps:
            f_ = np.zeros((N_SLOTS, N_USAGE), np.float32)
            for sl, pid in m_.items():
                v = ld.usage.get((g.game_id, pid))
                if v is not None:
                    f_[sl] = v
            feats.append(f_)
        lu = GameLineups(toks[0], toks[1], ld.coach(g.game_id, "home"), ld.coach(g.game_id, "away"), usage=usage,
                         features=feats)
        rg = ratings[ratings.game_id == g.game_id].set_index("side")
        pre = tuple(scale_ratings({k: float(rg.at[sd, f"pre_{k}"]) for k in ("off_epa", "def_epa", "elo")},
                                  sim.strength_scale) for sd in ("home", "away"))
        start = new_game(g.home, g.away, g.season, g.game_type, opening_kicker=int(g.opening_kicker))
        _, logs = sim.rollout(start, N, seed=i, keep_plays=N, ratings=pre, lineups=lu)
        acc: dict[str, np.ndarray] = {}
        for lg in logs:
            for s, o in lg:
                if o.who is None or o.who >= N_SLOTS or o.play_type not in ("run", "pass"):
                    continue
                pid = maps[s.pos].get(o.who)
                if pid is None:
                    continue
                a = acc.setdefault(pid, np.zeros(len(STATS)))
                if o.play_type == "run":
                    a[3] += 1
                    a[4] += o.yards if o.result == "run" else 0
                else:
                    a[0] += 1
                    if o.result == "complete":
                        a[1] += 1
                        a[2] += o.yards
        predicted_ids = {pid for m in maps for slot, pid in m.items() if SLOTS[slot][0] in GROUPS}
        for pid in predicted_ids:
            out.append({"game_id": g.game_id, "player_id": pid,
                        **{f"sim_{k}": v / N for k, v in zip(STATS, acc.get(pid, np.zeros(len(STATS))))}})
        if i % 25 == 0:
            print(f"  {source}: {i + 1}/{len(val)} games ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ compare
def metrics(df: pd.DataFrame, pred: str, k: str) -> dict:
    e = df[pred] - df[k]
    return {"rmse": float(np.sqrt((e ** 2).mean())), "mae": float(e.abs().mean()),
            "corr": float(np.corrcoef(df[pred], df[k])[0, 1]), "mean_pred": float(df[pred].mean()),
            "mean_actual": float(df[k].mean())}


report = {"checkpoint": str(CKPT), "sims_per_game": N, "val_games": len(val), "by_source": {}}
test = app[app.game_id.isin(val.game_id)].copy()
for j, k in enumerate(STATS):
    test[f"role_{k}"] = role_a[test.index.to_numpy(), j]
    test[f"formopp_{k}"] = [coef[(gname, k)] @ [1, f, o] for gname, f, o in
                            zip(test.group, test[f"form_{k}"], test[f"opp_{k}"])]
rng = np.random.default_rng(0)
for source in SOURCES:
    simdf = simulate(source)
    # evaluated: players on the simulated depth chart who actually played in the game
    df = test.merge(simdf, on=["game_id", "player_id"], how="inner")
    tot = prod[prod.game_id.isin(val.game_id)]
    cover = {k: float(df[k].sum() / tot[k].sum()) for k in ("targets", "carries")}
    res = {"player_games": len(df), "share_of_actual_volume_covered": cover, "groups": {}}
    for gname in GROUPS:
        d = df[df.group == gname]
        res["groups"][gname] = {"n": len(d)}
        for k in STATS:
            if gname != "RB" and k in ("carries", "rush_yds"):
                continue
            row = {name: metrics(d, f"{col}_{k}", k) for name, col in
                   (("model", "sim"), ("role", "role"), ("form", "form"), ("form+opp", "formopp"))}
            # paired bootstrap over games: model MSE minus form+opp MSE
            dd = (d[f"sim_{k}"] - d[k]) ** 2 - (d[f"formopp_{k}"] - d[k]) ** 2
            gids = d.game_id.to_numpy(); ug = np.unique(gids)
            per = pd.Series(dd.to_numpy()).groupby(gids).agg(["sum", "count"]).loc[ug]
            bs = []
            for _ in range(1000):
                kk = rng.integers(0, len(ug), len(ug))
                bs.append(per["sum"].to_numpy()[kk].sum() / per["count"].to_numpy()[kk].sum())
            row["model_minus_formopp_mse"] = [float(dd.mean()), float(np.percentile(bs, 2.5)),
                                              float(np.percentile(bs, 97.5))]
            # does the model know something the baseline doesn't? corr of (model - base) with (actual - base)
            a, b = d[f"sim_{k}"] - d[f"formopp_{k}"], d[k] - d[f"formopp_{k}"]
            row["corr_model_adjustment_vs_actual"] = float(np.corrcoef(a, b)[0, 1])
            # blend: y ~ w0 + w1 * model + w2 * form+opp, weights fit on half the games and scored on
            # the other half (and vice versa), so the comparison stays out of sample
            half = (pd.Series(gids).map({g: j % 2 for j, g in enumerate(ug)})).to_numpy()
            blend = np.zeros(len(d))
            X = np.c_[np.ones(len(d)), d[f"sim_{k}"], d[f"formopp_{k}"]]
            for h_ in (0, 1):
                w = np.linalg.lstsq(X[half != h_], d[k].to_numpy()[half != h_], rcond=None)[0]
                blend[half == h_] = X[half == h_] @ w
            row["blend"] = metrics(d.assign(blend_=blend), "blend_", k)
            dd = (blend - d[k]) ** 2 - (d[f"formopp_{k}"] - d[k]) ** 2
            per = pd.Series(dd.to_numpy()).groupby(gids).agg(["sum", "count"]).loc[ug]
            bs = []
            for _ in range(1000):
                kk = rng.integers(0, len(ug), len(ug))
                bs.append(per["sum"].to_numpy()[kk].sum() / per["count"].to_numpy()[kk].sum())
            row["blend_minus_formopp_mse"] = [float(dd.mean()), float(np.percentile(bs, 2.5)),
                                              float(np.percentile(bs, 97.5))]
            res["groups"][gname][k] = row
    report["by_source"][source] = res
    df.to_csv(OUT.with_name(f"{OUT.stem}_{source}.csv"), index=False)
    print(json.dumps({source: {g: {k: {m: round(v["rmse"], 2) for m, v in r.items() if isinstance(v, dict)}
                                   for k, r in gr.items() if k != "n"} for g, gr in res["groups"].items()}}, indent=0))
OUT.write_text(json.dumps(report, indent=1))
print(f"wrote {OUT}")
