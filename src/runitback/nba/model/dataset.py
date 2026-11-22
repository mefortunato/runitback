"""Vectorized encoding of processed NBA events into padded per-game tensors."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..league import NBA, League
from ..vocab import FIRST_SEASON, MAX_SEASON, N_ELAPSED, N_TO, PHASES, PLAY_TYPES, PT, TYPES_FOR_PHASE
from .features import CONT_FEATURES, TeamSeasonVocab

PHASE_TYPES = np.zeros((len(PHASES), len(PLAY_TYPES)), dtype=bool)
for ph, types in TYPES_FOR_PHASE.items():
    PHASE_TYPES[ph, [PT[t] for t in types]] = True


def type_results(league: League) -> np.ndarray:
    """(play type, result) -> whether the result can follow that play type, in the league's vocabulary."""
    m = np.zeros((len(PLAY_TYPES), league.n_results), dtype=bool)
    for t, rs in league.results_for_type.items():
        m[PT[t], [league.rs[r] for r in rs]] = True
    return m


TYPE_RESULTS = type_results(NBA)


def build_team_seasons(games: pd.DataFrame, league: League | str | None = None) -> TeamSeasonVocab:
    """The (team, season) vocabulary of `games`. A league without a fixed team list (college) gets the
    teams of `games` too: a team that first appears later is UNK, like an unseen team-season."""
    keys = [f"{t}_{s}" for t, s in zip(games.home, games.season)] + \
           [f"{t}_{s}" for t, s in zip(games.away, games.season)]
    ts = TeamSeasonVocab(keys, league)
    if ts.league.dynamic_teams:
        ts = TeamSeasonVocab(keys, ts.league, sorted(set(games.home) | set(games.away)))
    return ts


def encode_games(plays: pd.DataFrame, games: pd.DataFrame, ts: TeamSeasonVocab, max_len: int,
                 ratings: pd.DataFrame | None = None):
    """Returns a list of dicts of numpy arrays, one per game. `ratings` adds the "cont" features.
    The league (the period length, the vocabulary, the team-foul features) is `ts.league`'s."""
    lg = ts.league
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
    r_idx = p.result.map(lg.rs).to_numpy()
    e_idx = np.clip(p.elapsed.to_numpy(), 0, lg.max_elapsed)
    qsec = p.qsec.to_numpy()
    first = (p.game_id != p.game_id.shift()).to_numpy()

    def prev(a, fill=0):
        out = np.roll(a + 1, 1)
        out[first] = fill
        return out

    fts = np.clip(p.fts_left.to_numpy(), 0, 3)
    feats = dict(
        qtr=np.minimum(p.qtr.to_numpy(), 6),
        qsec=np.clip(qsec, 0, lg.period_seconds) // 15,
        phase=p.phase.to_numpy(),
        fts_left=fts,
        diff=np.clip(diff, -40, 40) + 40,
        to_off=np.minimum(np.where(pos == 0, toh, toa), N_TO - 1),
        to_def=np.minimum(np.where(pos == 0, toa, toh), N_TO - 1),
        is_home=(pos == 0).astype(int),
        off_team=np.array([ts.team(t) for t in off]),
        def_team=np.array([ts.team(t) for t in dfn]),
        off_ts=np.array([ts.get(t, s) for t, s in zip(off, season)]),
        def_ts=np.array([ts.get(t, s) for t, s in zip(dfn, season)]),
        season=np.clip(season - FIRST_SEASON, 0, MAX_SEASON - FIRST_SEASON),
        game_type=(meta.game_type.to_numpy() == "POST").astype(int),
        prev_type=prev(t_idx), prev_result=prev(r_idx), prev_elapsed=prev(e_idx),
    )
    one_and_one = np.zeros(len(p), dtype=bool)
    fouls_off = fouls_def = None
    if lg.bonus:
        one_and_one = p.oo.to_numpy().astype(bool)
        feats["fts_left"] = np.where(one_and_one, 4, fts) if lg.n_fts > 4 else fts
        fh, fa = p.foul_h.to_numpy(), p.foul_a.to_numpy()
        fouls_off, fouls_def = np.where(pos == 0, fh, fa), np.where(pos == 0, fa, fh)
        feats["fouls_off"] = np.minimum(fouls_off, lg.n_fouls - 1)
        feats["fouls_def"] = np.minimum(fouls_def, lg.n_fouls - 1)
    # legal masks, mirroring engine.rules.legal_* / elapsed_range
    to_off, to_def = np.where(pos == 0, toh, toa), np.where(pos == 0, toa, toh)
    tmask = PHASE_TYPES[p.phase.to_numpy()].copy()
    tmask[:, PT["timeout"]] &= (to_off > 0) | (to_def > 0)
    tmask[:, PT["run_out"]] &= qsec <= lg.max_elapsed
    rmask = type_results(lg)[t_idx].copy()
    is_to = t_idx == PT["timeout"]
    rmask[is_to, lg.rs["timeout_offense"]] &= to_off[is_to] > 0
    rmask[is_to, lg.rs["timeout_defense"]] &= to_def[is_to] > 0
    is_ft = t_idx == PT["free_throw"]
    more = (p.fts_left.to_numpy() > 1) & ~one_and_one
    rmask[is_ft & more, lg.rs["missed_off"]] = False
    rmask[is_ft & more, lg.rs["missed_def"]] = False
    rmask[is_ft & ~more, lg.rs["missed"]] = False
    if lg.bonus:  # a foul in the bonus always sends the shooter to the line
        ot = p.qtr.to_numpy() > lg.regulation_periods
        in_bonus = np.array([lg.bonus.level(int(f) + 1, bool(o)) > 0 for f, o in zip(fouls_def, ot)])
        rmask[(t_idx == PT["foul"]) & in_bonus, lg.rs["fts_0"]] = False
    hi = np.minimum(np.maximum(qsec, 0), lg.max_elapsed)
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

    names = [k[2:] for k in games[0] if k.startswith("f_") and k != "f_cont"]
    feats = {f: pad(f"f_{f}") for f in names}
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
    rm = pad("result_mask", torch.bool, (games[0]["result_mask"].shape[1],))
    tm[valid == 0] = True
    rm[valid == 0] = True
    ar = torch.arange(N_ELAPSED)
    em = (ar >= pad("el_lo")[..., None]) & (ar <= pad("el_hi")[..., None])
    em[valid == 0] = True
    return dict(feats=feats, valid=valid, type=pad("type"), result=pad("result"),
                elapsed=pad("elapsed"), type_mask=tm, result_mask=rm, elapsed_mask=em)
