"""Vectorized encoding of processed plays into padded per-game tensors."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..vocab import (
    FIRST_SEASON, KICKOFF, MAX_ELAPSED, MAX_SEASON, N_ELAPSED, N_YARDS, PAT, PLAY_TYPES, PT,
    RESULTS, RESULTS_FOR_TYPE, RS, SCRIMMAGE, TEAM_IDX, TYPES_FOR_PHASE, YARDS_MIN,
    NONE_RESULTS, SPOT_RESULTS,
)
from .features import CONT_FEATURES, FEATURES, TeamSeasonVocab

PHASE_TYPES = np.zeros((3, len(PLAY_TYPES)), dtype=bool)
for ph, types in TYPES_FOR_PHASE.items():
    PHASE_TYPES[ph, [PT[t] for t in types]] = True
TYPE_RESULTS = np.zeros((len(PLAY_TYPES), len(RESULTS)), dtype=bool)
for t, rs in RESULTS_FOR_TYPE.items():
    TYPE_RESULTS[PT[t], [RS[r] for r in rs]] = True
SPOT = np.array([r in SPOT_RESULTS for r in RESULTS])
NONE = np.array([r in NONE_RESULTS for r in RESULTS])
PAT_TYPES = np.array([t in ("extra_point", "two_point") for t in PLAY_TYPES])


def build_team_seasons(games: pd.DataFrame) -> TeamSeasonVocab:
    keys = [f"{t}_{s}" for t, s in zip(games.home, games.season)] + \
           [f"{t}_{s}" for t, s in zip(games.away, games.season)]
    return TeamSeasonVocab(keys)


def encode_games(plays: pd.DataFrame, games: pd.DataFrame, ts: TeamSeasonVocab, max_len: int,
                 ratings: pd.DataFrame | None = None):
    """Returns a list of dicts of numpy arrays, one per game. `ratings` adds the "cont" features."""
    g = games.set_index("game_id")
    p = plays.sort_values(["game_id", "idx"]).reset_index(drop=True)
    meta = g.loc[p.game_id]
    pos = p.pos.to_numpy()
    home, away = meta.home.to_numpy(), meta.away.to_numpy()
    season = meta.season.to_numpy()
    off = np.where(pos == 0, home, away)
    dfn = np.where(pos == 0, away, home)
    sh, sa = p.score_h.to_numpy(), p.score_a.to_numpy()
    toh, toa = p.to_h.to_numpy(), p.to_a.to_numpy()
    diff = np.where(pos == 0, sh - sa, sa - sh)
    t_idx = p.play_type.map(PT).to_numpy()
    r_idx = p.result.map(RS).to_numpy()
    y_idx = (np.clip(p.yards.to_numpy(), YARDS_MIN, YARDS_MIN + N_YARDS - 1) - YARDS_MIN)
    e_idx = np.clip(p.elapsed.to_numpy(), 0, MAX_ELAPSED)
    first = (p.game_id != p.game_id.shift()).to_numpy()

    def prev(a, fill=0):
        out = np.roll(a + 1, 1)
        out[first] = fill
        return out

    feats = dict(
        qtr=np.minimum(p.qtr.to_numpy(), 6),
        qsec=np.clip(p.qsec.to_numpy(), 0, 900) // 15,
        phase=p.phase.to_numpy(),
        down=np.minimum(p.down.to_numpy(), 4),
        ytg=np.clip(p.ytg.to_numpy(), 0, 30),
        yl=np.clip(p.yl.to_numpy(), 0, 100),
        diff=np.clip(diff, -30, 30) + 30,
        to_off=np.where(pos == 0, toh, toa), to_def=np.where(pos == 0, toa, toh),
        is_home=(pos == 0).astype(int),
        off_team=np.array([TEAM_IDX.get(t, -1) + 1 for t in off]),
        def_team=np.array([TEAM_IDX.get(t, -1) + 1 for t in dfn]),
        off_ts=np.array([ts.get(t, s) for t, s in zip(off, season)]),
        def_ts=np.array([ts.get(t, s) for t, s in zip(dfn, season)]),
        season=np.minimum(season - FIRST_SEASON, MAX_SEASON - FIRST_SEASON),
        game_type=(meta.game_type.to_numpy() == "POST").astype(int),
        prev_type=prev(t_idx), prev_result=prev(r_idx), prev_yards=prev(y_idx), prev_elapsed=prev(e_idx),
    )
    # legal masks, mirroring engine.rules.legal_* / yards_range
    to_off, to_def = feats["to_off"], feats["to_def"]
    tmask = PHASE_TYPES[feats["phase"]].copy()
    tmask[:, PT["timeout"]] &= (to_off > 0) | (to_def > 0)
    rmask = TYPE_RESULTS[t_idx].copy()
    is_to = t_idx == PT["timeout"]
    rmask[is_to, RS["timeout_offense"]] &= to_off[is_to] > 0
    rmask[is_to, RS["timeout_defense"]] &= to_def[is_to] > 0
    yl = feats["yl"]
    none = NONE[r_idx] | PAT_TYPES[t_idx]
    spot = SPOT[r_idx] & ~none
    lo = np.where(none, 0, np.where(spot, 0, yl - 100))
    hi = np.where(none, 0, np.where(spot, 99, yl))
    y_true = y_idx + YARDS_MIN
    assert ((y_true >= lo) & (y_true <= hi)).mean() > 0.999, "targets outside legal yards range"
    lo, hi = np.minimum(lo, y_true), np.maximum(hi, y_true)
    el_hi = np.maximum(np.minimum(p.qsec.to_numpy(), MAX_ELAPSED), e_idx)

    cont = np.zeros((len(p), len(CONT_FEATURES)), dtype=np.float32)
    if ratings is not None:
        side = {}
        for sd in ("home", "away"):
            r = ratings[ratings.side == sd].set_index("game_id").reindex(p.game_id)
            side[sd] = {k: r[f"pre_{k}"].to_numpy() for k in ("off_epa", "def_epa", "elo")}
        known = ~np.isnan(side["home"]["elo"]) & ~np.isnan(side["away"]["elo"])
        o = {k: np.where(pos == 0, side["home"][k], side["away"][k]) for k in side["home"]}
        d = {k: np.where(pos == 0, side["away"][k], side["home"][k]) for k in side["home"]}
        cont = np.stack([o["off_epa"] * 10, o["def_epa"] * 10, d["off_epa"] * 10, d["def_epa"] * 10,
                         (o["elo"] - d["elo"]) / 100], axis=1).astype(np.float32)
        cont[~known] = 0.0

    arrays = dict(**{f"f_{k}": v for k, v in feats.items()}, f_cont=cont, type=t_idx, result=r_idx, yards=y_idx,
                  elapsed=e_idx, type_mask=tmask, result_mask=rmask, y_lo=lo, y_hi=hi, el_hi=el_hi)
    bounds = np.flatnonzero(np.r_[first, True])
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        b = min(b, a + max_len)
        out.append({k: v[a:b] for k, v in arrays.items()} | {"game_id": p.game_id.iloc[a]})
    return out


def collate(games: list[dict], unk_ts_prob: float = 0.0, rng: np.random.Generator | None = None,
            unk_cont_prob: float = 0.0) -> dict:
    B = len(games)
    T = max(len(g["type"]) for g in games)

    def pad(key, dtype=torch.long, extra=()):
        arr = np.zeros((B, T, *extra), dtype=games[0][key].dtype)
        for i, g in enumerate(games):
            arr[i, : len(g[key])] = g[key]
        return torch.as_tensor(arr, dtype=dtype)

    feats = {f: pad(f"f_{f}") for f in FEATURES}
    if unk_ts_prob > 0 and rng is not None:
        drop = torch.as_tensor(rng.random(B) < unk_ts_prob)[:, None]
        feats["off_ts"] = feats["off_ts"].masked_fill(drop, 0)
        feats["def_ts"] = feats["def_ts"].masked_fill(drop, 0)
    feats["cont"] = pad("f_cont", torch.float32, (len(CONT_FEATURES),))
    if unk_cont_prob > 0 and rng is not None:
        keep = torch.as_tensor(rng.random(B) >= unk_cont_prob, dtype=torch.float32)[:, None, None]
        feats["cont"] = feats["cont"] * keep
    valid = torch.zeros(B, T)
    for i, g in enumerate(games):
        valid[i, : len(g["type"])] = 1
    tm = pad("type_mask", torch.bool, (len(PLAY_TYPES),))
    rm = pad("result_mask", torch.bool, (len(RESULTS),))
    tm[valid == 0] = True
    rm[valid == 0] = True
    ar = torch.arange(N_YARDS) + YARDS_MIN
    ym = (ar >= pad("y_lo")[..., None]) & (ar <= pad("y_hi")[..., None])
    ym[valid == 0] = True
    em = torch.arange(N_ELAPSED) <= pad("el_hi")[..., None]
    em[valid == 0] = True
    return dict(feats=feats, valid=valid, type=pad("type"), result=pad("result"), yards=pad("yards"),
                elapsed=pad("elapsed"), type_mask=tm, result_mask=rm, yards_mask=ym, elapsed_mask=em)
