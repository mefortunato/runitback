"""Vectorized encoding of the NHL plays table (`data.preprocess`) into padded per-game tensors."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..data.preprocess import STATE_COLUMNS
from ..vocab import MAX_ELAPSED, N_ELAPSED, PLAY_TYPES, PT, RESULTS, RS
from .features import CONT_FEATURES, FEATURES, N_SIDES, TeamSeasonVocab, encode_arrays


def build_team_seasons(games: pd.DataFrame) -> TeamSeasonVocab:
    keys = [f"{t}_{s}" for t, s in zip(games.home, games.season)] + \
           [f"{t}_{s}" for t, s in zip(games.away, games.season)]
    return TeamSeasonVocab(keys)


def unpack(mask: np.ndarray, n: int) -> np.ndarray:
    """Bit masks (int) -> (len, n) bool."""
    return ((np.asarray(mask, dtype=np.int64)[:, None] >> np.arange(n)) & 1).astype(bool)


def ratings_matrix(ratings: pd.DataFrame | None, game_ids: pd.Series) -> np.ndarray:
    """(len(game_ids), CONT_FEATURES) pre-game ratings per row; zeros where a team has none."""
    cont = np.zeros((len(game_ids), len(CONT_FEATURES)), dtype=np.float32)
    if ratings is None:
        return cont
    side = {}
    for sd in ("home", "away"):
        r = ratings[ratings.side == sd].drop_duplicates("game_id").set_index("game_id").reindex(game_ids)
        side[sd] = {k: r[f"pre_{k}"].to_numpy(float) for k in ("off_rtg", "def_rtg", "xg_off", "xg_def", "elo")}
    h, a = side["home"], side["away"]
    known = ~np.isnan(h["elo"]) & ~np.isnan(a["elo"])
    cont = np.stack([h["off_rtg"], h["def_rtg"], h["xg_off"], h["xg_def"],
                     a["off_rtg"], a["def_rtg"], a["xg_off"], a["xg_def"],
                     (h["elo"] - a["elo"]) / 100], axis=1).astype(np.float32)
    cont[~known] = 0.0
    return np.nan_to_num(cont)


def encode_games(plays: pd.DataFrame, games: pd.DataFrame, ts: TeamSeasonVocab, max_len: int,
                 ratings: pd.DataFrame | None = None) -> list[dict]:
    """One dict of numpy arrays per game (its first `max_len` events)."""
    if not len(plays):
        return []
    g = games.set_index("game_id")
    p = plays.sort_values(["game_id", "idx"]).reset_index(drop=True)
    meta = g.loc[p.game_id]
    first = (p.game_id != p.game_id.shift()).to_numpy()
    t_idx = p.play_type.map(PT).to_numpy()
    r_idx = p.result.map(RS).to_numpy()
    s_idx = p.side.to_numpy().astype(np.int64)
    e_idx = np.clip(p.elapsed.to_numpy(), 0, MAX_ELAPSED).astype(np.int64)

    def prev(a):
        out = np.roll(a, 1).astype(np.int64)
        out[first] = -1
        return out

    feats = encode_arrays({c: p[c].to_numpy() for c in STATE_COLUMNS}, meta.home.to_numpy(),
                          meta.away.to_numpy(), meta.season.to_numpy(), meta.game_type.to_numpy(),
                          dict(type=prev(t_idx), side=prev(s_idx), result=prev(r_idx), elapsed=prev(e_idx)), ts)
    arrays = dict(**{f"f_{k}": np.asarray(v, dtype=np.int64) for k, v in feats.items()},
                  f_cont=ratings_matrix(ratings, p.game_id), type=t_idx, side=s_idx, result=r_idx,
                  elapsed=e_idx, type_mask=unpack(p.type_mask, len(PLAY_TYPES)),
                  side_mask=unpack(p.side_mask, N_SIDES), result_mask=unpack(p.result_mask, len(RESULTS)),
                  el_lo=p.el_lo.to_numpy().astype(np.int64), el_hi=p.el_hi.to_numpy().astype(np.int64))
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
        feats["home_ts"] = feats["home_ts"].masked_fill(drop, 0)
        feats["away_ts"] = feats["away_ts"].masked_fill(drop, 0)
    feats["cont"] = pad("f_cont", torch.float32, (len(CONT_FEATURES),))
    if unk_cont_prob > 0 and rng is not None:
        keep = torch.as_tensor(rng.random(B) >= unk_cont_prob, dtype=torch.float32)[:, None, None]
        feats["cont"] = feats["cont"] * keep
    valid = torch.zeros(B, T)
    for i, g in enumerate(games):
        valid[i, : len(g["type"])] = 1
    tm = pad("type_mask", torch.bool, (len(PLAY_TYPES),))
    sm = pad("side_mask", torch.bool, (N_SIDES,))
    rm = pad("result_mask", torch.bool, (len(RESULTS),))
    for m in (tm, sm, rm):
        m[valid == 0] = True
    ar = torch.arange(N_ELAPSED)
    em = (ar >= pad("el_lo")[..., None]) & (ar <= pad("el_hi")[..., None])
    em[valid == 0] = True
    return dict(feats=feats, valid=valid, type=pad("type"), side=pad("side"), result=pad("result"),
                elapsed=pad("elapsed"), type_mask=tm, side_mask=sm, result_mask=rm, elapsed_mask=em)
