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
from .lineups import N_COACH_CONT, N_GROUPS, N_PERS, N_SLOTS, SIDE, SLOT_GROUP, WHO_NONE, LineupData, other_token

LINEUP_PLAY_KEYS = ("f_off_coach", "f_def_coach", "f_coach_cont", "pers", "pers_grp")

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


def encode_lineups(p: pd.DataFrame, pos: np.ndarray, home: np.ndarray, away: np.ndarray,
                   ld: LineupData) -> dict[str, np.ndarray]:
    """Per-play coach features and on-field personnel (possession side first)."""
    n = len(p)
    gids = p.game_id.to_numpy()
    coach = {}
    for gid in pd.unique(gids):
        coach[gid] = (ld.coach(gid, "home"), ld.coach(gid, "away"))
    hc = [coach[g][0] for g in gids]
    ac = [coach[g][1] for g in gids]
    is_home = pos == 0
    off_coach = np.where(is_home, [c[0] for c in hc], [c[0] for c in ac]).astype(np.int32)
    def_coach = np.where(is_home, [c[0] for c in ac], [c[0] for c in hc]).astype(np.int32)
    hcv, acv = np.array([c[1] for c in hc], np.float32), np.array([c[1] for c in ac], np.float32)
    coach_cont = np.concatenate([np.where(is_home[:, None], hcv, acv), np.where(is_home[:, None], acv, hcv)], 1)

    pers = np.zeros((n, N_PERS), dtype=np.int32)
    grp = np.zeros((n, N_PERS), dtype=np.int8)
    slots = np.full((n, N_PERS), -2, dtype=np.int16)  # -2 = no personnel for this play
    who = np.full(n, -1, dtype=np.int8)  # 0..10 = offensive player on the field, 11 = nobody, -1 = unknown
    ptype = p.play_type.to_numpy()
    if "play_id" in p.columns and len(ld.personnel):
        pe = ld.personnel.set_index(["game_id", "play_id"])
        keys = pd.MultiIndex.from_arrays([p.game_id, p.play_id])
        hit = pe.reindex(keys)
        found = hit.home_ids.notna().to_numpy() & (p.play_type != "timeout").to_numpy()
        for i in np.flatnonzero(found):
            gid = gids[i]
            sides = (hit.home_ids.iat[i], hit.away_ids.iat[i]) if pos[i] == 0 else \
                (hit.away_ids.iat[i], hit.home_ids.iat[i])
            ball = hit.ball_id.iat[i] if "ball_id" in hit.columns else None
            off_ids = list(sides[0])[:SIDE]
            if ptype[i] in ("run", "pass"):
                if isinstance(ball, str):
                    who[i] = off_ids.index(ball) if ball in off_ids else -1
                elif ptype[i] == "pass":
                    who[i] = WHO_NONE
            for k, ids in enumerate(sides):
                t, g, sl = ld.personnel_tokens(gid, list(ids))
                m = len(t)
                pers[i, k * SIDE: k * SIDE + m] = t
                grp[i, k * SIDE: k * SIDE + m] = g
                slots[i, k * SIDE: k * SIDE + m] = sl
    return dict(f_off_coach=off_coach, f_def_coach=def_coach, f_coach_cont=coach_cont.astype(np.float32),
                pers=pers, pers_grp=grp, pers_slot=slots, who=who)


def encode_games(plays: pd.DataFrame, games: pd.DataFrame, ts: TeamSeasonVocab, max_len: int,
                 ratings: pd.DataFrame | None = None, lineups: LineupData | None = None,
                 lineup_source: str = "game"):
    """Returns a list of dicts of numpy arrays, one per game. `ratings` adds the "cont" features;
    `lineups` adds depth charts, coaches and per-play personnel. lineup_source="game" uses who
    played in the game itself (training), "previous" each team's previous game (what a user would
    enter before kickoff) and "none" leaves depth charts unknown."""
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
    if lineups is not None:
        arrays |= encode_lineups(p, pos, home, away, lineups)
        if lineup_source != "game":  # per-play personnel is only known after the fact
            arrays["pers"][:] = 0
            arrays["pers_grp"][:] = 0
            arrays["who"][:] = -1
    bounds = np.flatnonzero(np.r_[first, True])
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        b = min(b, a + max_len)
        item = {k: v[a:b] for k, v in arrays.items()} | {"game_id": p.game_id.iloc[a]}
        if lineups is not None:
            gid, row = item["game_id"], g.loc[item["game_id"]]
            if lineup_source == "game":
                lu = [lineups.lineup(gid, row.home), lineups.lineup(gid, row.away)]
                item["lineup_prev"] = np.stack([lineups.previous_lineup(gid, t) for t in (row.home, row.away)])
            elif lineup_source == "previous":
                lu = [lineups.previous_lineup(gid, t, int(row.season), int(row.week)) for t in (row.home, row.away)]
            else:
                lu = [np.zeros(N_SLOTS, np.int32)] * 2
            item["lineup"] = np.stack(lu)
        out.append(item)
    return out


def collate(games: list[dict], unk_ts_prob: float = 0.0, rng: np.random.Generator | None = None,
            unk_cont_prob: float = 0.0, unk_lineup_prob: float = 0.0, unk_pers_prob: float = 0.0,
            other_player_prob: float = 0.0, prev_lineup_prob: float = 0.0) -> dict:
    """Pads games into a batch. During training the unk_* probabilities blank out team-seasons,
    ratings, a team's whole depth chart or a game's personnel (so the model can run without them),
    other_player_prob swaps single players for their group's OTHER token, and prev_lineup_prob shows
    a team's previous-game depth chart instead (what a user starts from before kickoff)."""
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
    if "lineup" in games[0]:
        feats |= _collate_lineups(games, pad, rng, unk_lineup_prob, unk_pers_prob, other_player_prob,
                                  prev_lineup_prob)
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


def _collate_lineups(games, pad, rng, unk_lineup_prob, unk_pers_prob, other_player_prob,
                     prev_lineup_prob=0.0) -> dict:
    B = len(games)
    lineup = np.stack([g["lineup"] for g in games]).astype(np.int64)        # (B, 2, S)
    if rng is not None and prev_lineup_prob > 0 and "lineup_prev" in games[0]:
        prev = np.stack([g["lineup_prev"] for g in games]).astype(np.int64)
        use = (rng.random((B, 2)) < prev_lineup_prob) & prev.any(-1) & lineup.any(-1)
        lineup[use] = prev[use]
    pers = pad("pers", torch.long, (N_PERS,)).numpy()
    grp = pad("pers_grp", torch.long, (N_PERS,)).numpy()
    if rng is not None:
        if unk_lineup_prob > 0:
            lineup[rng.random((B, 2)) < unk_lineup_prob] = 0
        if unk_pers_prob > 0:
            pers[rng.random(B) < unk_pers_prob] = 0
        if other_player_prob > 0:
            swap = (rng.random(lineup.shape) < other_player_prob) & (lineup > 2 * N_GROUPS)
            lineup = np.where(swap, other_token(0) + SLOT_GROUP[None, None, :], lineup)
            swap = (rng.random(pers.shape) < other_player_prob) & (pers > 2 * N_GROUPS)
            pers = np.where(swap, other_token(0) + grp - 1, pers)
    grp = np.where(pers > 0, grp, 0)
    T = pers.shape[1]
    who = np.full((B, T), -1, dtype=np.int64)
    for i, g in enumerate(games):
        if "who" in g:
            who[i, : len(g["who"])] = g["who"]
    who[(pers[..., :SIDE] == 0).all(-1)] = -1  # personnel hidden or unknown: no target
    return {"who": torch.as_tensor(who),"lineup": torch.as_tensor(lineup), "pers": torch.as_tensor(pers, dtype=torch.long),
            "pers_grp": torch.as_tensor(grp), "off_coach": pad("f_off_coach"), "def_coach": pad("f_def_coach"),
            "coach_cont": pad("f_coach_cont", torch.float32, (N_COACH_CONT,))}
