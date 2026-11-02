"""Vectorized encoding of processed NBA events into padded per-game tensors."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..vocab import (
    FIRST_SEASON, MAX_ELAPSED, MAX_SEASON, N_ELAPSED, N_TO, PHASES, PLAY_TYPES, PT, RESULTS,
    RESULTS_FOR_TYPE, RS, TEAM_IDX, TYPES_FOR_PHASE,
)
from .features import CONT_FEATURES, FEATURES, TeamSeasonVocab

PHASE_TYPES = np.zeros((len(PHASES), len(PLAY_TYPES)), dtype=bool)
for ph, types in TYPES_FOR_PHASE.items():
    PHASE_TYPES[ph, [PT[t] for t in types]] = True
TYPE_RESULTS = np.zeros((len(PLAY_TYPES), len(RESULTS)), dtype=bool)
for t, rs in RESULTS_FOR_TYPE.items():
    TYPE_RESULTS[PT[t], [RS[r] for r in rs]] = True


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
    e_idx = np.clip(p.elapsed.to_numpy(), 0, MAX_ELAPSED)
    qsec = p.qsec.to_numpy()
    first = (p.game_id != p.game_id.shift()).to_numpy()

    def prev(a, fill=0):
        out = np.roll(a + 1, 1)
        out[first] = fill
        return out

    feats = dict(
        qtr=np.minimum(p.qtr.to_numpy(), 6),
        qsec=np.clip(qsec, 0, 720) // 15,
        phase=p.phase.to_numpy(),
        fts_left=np.clip(p.fts_left.to_numpy(), 0, 3),
        diff=np.clip(diff, -40, 40) + 40,
        to_off=np.minimum(np.where(pos == 0, toh, toa), N_TO - 1),
        to_def=np.minimum(np.where(pos == 0, toa, toh), N_TO - 1),
        is_home=(pos == 0).astype(int),
        off_team=np.array([TEAM_IDX.get(t, -1) + 1 for t in off]),
        def_team=np.array([TEAM_IDX.get(t, -1) + 1 for t in dfn]),
        off_ts=np.array([ts.get(t, s) for t, s in zip(off, season)]),
        def_ts=np.array([ts.get(t, s) for t, s in zip(dfn, season)]),
        season=np.clip(season - FIRST_SEASON, 0, MAX_SEASON - FIRST_SEASON),
        game_type=(meta.game_type.to_numpy() == "POST").astype(int),
        prev_type=prev(t_idx), prev_result=prev(r_idx), prev_elapsed=prev(e_idx),
    )
    # legal masks, mirroring engine.rules.legal_* / elapsed_range
    to_off, to_def = np.where(pos == 0, toh, toa), np.where(pos == 0, toa, toh)
    tmask = PHASE_TYPES[feats["phase"]].copy()
    tmask[:, PT["timeout"]] &= (to_off > 0) | (to_def > 0)
    tmask[:, PT["run_out"]] &= qsec <= MAX_ELAPSED
    rmask = TYPE_RESULTS[t_idx].copy()
    is_to = t_idx == PT["timeout"]
    rmask[is_to, RS["timeout_offense"]] &= to_off[is_to] > 0
    rmask[is_to, RS["timeout_defense"]] &= to_def[is_to] > 0
    is_ft = t_idx == PT["free_throw"]
    more = p.fts_left.to_numpy() > 1
    rmask[is_ft & more, RS["missed_off"]] = False
    rmask[is_ft & more, RS["missed_def"]] = False
    rmask[is_ft & ~more, RS["missed"]] = False
    hi = np.minimum(np.maximum(qsec, 0), MAX_ELAPSED)
    lo = np.where(t_idx == PT["run_out"], hi, 0)
    legal = tmask[np.arange(len(p)), t_idx] & rmask[np.arange(len(p)), r_idx] & (e_idx >= lo) & (e_idx <= hi)
    assert legal.mean() > 0.995, f"only {legal.mean():.4f} of targets are legal"
    # keep the rare illegal real target trainable rather than infinitely unlikely
    tmask[np.arange(len(p)), t_idx] = True
    rmask[np.arange(len(p)), r_idx] = True
    lo, hi = np.minimum(lo, e_idx), np.maximum(hi, e_idx)

    cont = np.zeros((len(p), len(CONT_FEATURES)), dtype=np.float32)
    if ratings is not None:
        side = {}
        for sd in ("home", "away"):
            r = ratings[ratings.side == sd].set_index("game_id").reindex(p.game_id)
            side[sd] = {k: r[f"pre_{k}"].to_numpy() for k in ("off_rtg", "def_rtg", "elo")}
        known = ~np.isnan(side["home"]["elo"]) & ~np.isnan(side["away"]["elo"])
        o = {k: np.where(pos == 0, side["home"][k], side["away"][k]) for k in side["home"]}
        d = {k: np.where(pos == 0, side["away"][k], side["home"][k]) for k in side["home"]}
        cont = np.stack([o["off_rtg"] / 5, o["def_rtg"] / 5, d["off_rtg"] / 5, d["def_rtg"] / 5,
                         (o["elo"] - d["elo"]) / 100], axis=1).astype(np.float32)
        cont[~known] = 0.0

    arrays = dict(**{f"f_{k}": v for k, v in feats.items()}, f_cont=cont, type=t_idx, result=r_idx,
                  elapsed=e_idx, type_mask=tmask, result_mask=rmask, el_lo=lo, el_hi=hi)
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
    ar = torch.arange(N_ELAPSED)
    em = (ar >= pad("el_lo")[..., None]) & (ar <= pad("el_hi")[..., None])
    em[valid == 0] = True
    return dict(feats=feats, valid=valid, type=pad("type"), result=pad("result"),
                elapsed=pad("elapsed"), type_mask=tm, result_mask=rm, elapsed_mask=em)
