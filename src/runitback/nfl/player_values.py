"""Player values in points per game over a replacement-level player, at every position.

A regularized adjusted plus-minus on play-by-play data: each scrimmage play's EPA is regressed on
the 22 players on the field (an offense coefficient for each of the possession team's players, a
defense coefficient for each of the other team's) plus situation controls, with ridge shrinkage
toward 0 and recent seasons weighted more. Players with too few plays in the window share a
replacement-level coefficient for their side and position group; a player's value is his
coefficient minus that (for defenders, with the sign flipped so that higher is better), times the
plays he is expected to be on the field for. Kickers and punters: shrunk EPA per attempt over the
league average, times attempts per game.

Every fit uses only plays before the game it is for (``fit_values(table, before=season*100+week)``),
so values are leak-free and update weekly; ``game_values`` sums a lineup's values per side.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from .data.players import GROUPS

log = logging.getLogger(__name__)

OFF_GROUPS = ("QB", "RB", "WR", "TE", "OL")
DEF_GROUPS = ("DL", "LB", "DB")
WINDOW = 4             # previous seasons in each fit (plus the current one up to the game)
DECAY = 0.7            # weight of a play from k seasons back: DECAY ** k
MIN_PLAYS = 150        # plays on the field (in the window) for a player's own coefficient
LAMBDA = 3000.0        # ridge penalty on player coefficients (tuned on next-season play EPA)
CTRL_SCALE = 30.0      # controls are scaled up, i.e. barely penalized
EPA_CLIP = 6.0
# prior width of each group's per-play effect, relative to the others: a column scaled by c is
# penalized c**2 less (quarterbacks touch every pass, so their effects are far larger)
GROUP_SCALE = {"QB": 4.0, "OL": 0.7, "DL": 0.7}
K_SHRINK, P_SHRINK = 25.0, 40.0   # pseudo-attempts at the league mean for kickers / punters
TEAM_PLAYS = 63.0      # scrimmage plays per team-game (offense, and faced on defense)
FG_PER_GAME, PUNTS_PER_GAME = 1.7, 4.0


# -- data -------------------------------------------------------------------------------------------

@dataclass
class PlayData:
    """Scrimmage plays (passes, runs) with known personnel, players as integer codes."""
    plays: pd.DataFrame       # game_id, t (season * 100 + week), season, down, ytg, yl, epa
    off: np.ndarray           # (n, W) codes of the possession team's players, -1 = none
    dfn: np.ndarray           # (n, W) the other team's
    ids: np.ndarray           # code -> player_id
    group: np.ndarray         # code -> group index in GROUPS (-1 = unknown)


def play_table(data_dir: Path, raw_dir: Path, group_of: dict[str, str], width: int = 13) -> PlayData:
    plays = pd.read_parquet(data_dir / "plays.parquet",
                            columns=["game_id", "play_id", "pos", "down", "ytg", "yl", "play_type"])
    plays = plays[plays.play_type.isin(["pass", "run"])]
    pers = pd.read_parquet(data_dir / "personnel.parquet", columns=["game_id", "play_id", "home_ids", "away_ids"])
    games = pd.read_parquet(data_dir / "games.parquet", columns=["game_id", "season", "week"])
    df = plays.merge(pers, on=["game_id", "play_id"]).merge(games, on="game_id")
    epa = []
    for s in sorted(df.season.unique()):
        r = pd.read_parquet(raw_dir / f"pbp_{s}.parquet", columns=["game_id", "play_id", "epa"])
        epa.append(r.assign(play_id=r.play_id.astype("int64")))
    df = df.merge(pd.concat(epa), on=["game_id", "play_id"]).dropna(subset=["epa"])
    df = df.sort_values(["season", "week", "game_id", "play_id"]).reset_index(drop=True)
    home = df.pos.to_numpy() == 0
    off_l = np.where(home, df.home_ids, df.away_ids)
    def_l = np.where(home, df.away_ids, df.home_ids)
    ids = np.array(sorted({p for col in (df.home_ids, df.away_ids) for lst in col for p in lst}))
    code = {p: i for i, p in enumerate(ids)}

    def to_codes(lists):
        out = np.full((len(lists), width), -1, dtype=np.int32)
        for i, lst in enumerate(lists):
            c = [code[p] for p in lst[:width]]
            out[i, :len(c)] = c
        return out

    gi = {g: i for i, g in enumerate(GROUPS)}
    group = np.array([gi.get(group_of.get(p), -1) for p in ids], dtype=np.int32)
    df["t"] = df.season * 100 + df.week
    return PlayData(df[["game_id", "t", "season", "down", "ytg", "yl", "epa"]], to_codes(off_l), to_codes(def_l),
                    ids, group)


def special_table(raw_dir: Path, seasons: list[int], weeks: pd.DataFrame) -> pd.DataFrame:
    """Field goals and punts: ``t, kind ('K'/'P'), player_id, epa`` (EPA of the kicking team)."""
    out = []
    for s in seasons:
        r = pd.read_parquet(raw_dir / f"pbp_{s}.parquet",
                            columns=["game_id", "play_type", "epa", "kicker_player_id", "punter_player_id"])
        fg = r[r.play_type == "field_goal"].dropna(subset=["kicker_player_id", "epa"])
        pt = r[r.play_type == "punt"].dropna(subset=["punter_player_id", "epa"])
        out += [fg.assign(kind="K", player_id=fg.kicker_player_id)[["game_id", "kind", "player_id", "epa"]],
                pt.assign(kind="P", player_id=pt.punter_player_id)[["game_id", "kind", "player_id", "epa"]]]
    df = pd.concat(out).merge(weeks, on="game_id")
    return df.assign(t=df.season * 100 + df.week)[["t", "season", "kind", "player_id", "epa"]]


def slot_plays(pg: pd.DataFrame) -> dict[tuple[str, int], float]:
    """Expected plays on the field per game for each depth-chart slot (group, rank), from the
    average snap share of the rank-th player of each group (by snaps) in played games."""
    off = pg.groupby(["game_id", "team"]).offense_snaps.transform("max")
    dfn = pg.groupby(["game_id", "team"]).defense_snaps.transform("max")
    share = np.where(pg.group.isin(DEF_GROUPS), pg.defense_snaps / dfn.clip(lower=1),
                     pg.offense_snaps / off.clip(lower=1))
    d = pg.assign(share=share)
    d = d[d.group.isin(OFF_GROUPS + DEF_GROUPS)]
    m = d.groupby(["group", "depth"]).share.mean()
    return {(g, int(k)): float(v) * TEAM_PLAYS for (g, k), v in m.items()}


# -- fits -------------------------------------------------------------------------------------------

@dataclass
class Values:
    before: int
    table: pd.DataFrame      # side ('off'/'def'/'K'/'P'), player_id, group, plays, value_play


def _controls(df: pd.DataFrame) -> sp.csr_matrix:
    d = df.down.clip(1, 4).astype(int) - 1
    ytg = np.digitize(df.ytg, [4, 7, 11])
    yl = np.clip(df.yl // 10, 0, 9).astype(int)
    seas = (df.season - df.season.min()).astype(int).to_numpy()
    n = len(df)
    rows = np.repeat(np.arange(n), 4)
    cols = np.stack([np.zeros(n, int), 1 + d * 4 + ytg, 17 + yl, 27 + seas], 1).ravel()
    return sp.csr_matrix((np.full(len(rows), CTRL_SCALE), (rows, cols)), shape=(n, 27 + seas.max() + 1))


def fit_values(data: PlayData, special: pd.DataFrame, before: int, lam: float = LAMBDA) -> Values:
    """Player values from the plays before ``before`` (season * 100 + week)."""
    cur = before // 100
    pl = data.plays
    sel = ((pl.t < before) & (pl.season >= cur - WINDOW)).to_numpy()
    win = pl[sel]
    w = DECAY ** (cur - win.season.to_numpy())
    n_ids, G = len(data.ids), len(GROUPS)
    tabs, blocks, scales = [], [], []
    offset = 0
    for side, codes in (("off", data.off[sel]), ("def", data.dfn[sel])):
        valid = (codes >= 0)
        valid &= data.group[np.where(valid, codes, 0)] >= 0
        cnt = np.bincount(codes[valid], minlength=n_ids)
        own = (cnt >= MIN_PLAYS) & (data.group >= 0)
        own_col = np.full(n_ids, -1)
        own_col[own] = np.arange(own.sum())
        n_own = int(own.sum())
        # a player without his own column uses his side's replacement column for his group
        col_of = np.where(own, own_col, n_own + np.maximum(data.group, 0))
        r, c = np.nonzero(valid)
        cols = col_of[codes[r, c]]
        scale = np.array([GROUP_SCALE.get(g, 1.0) for g in GROUPS])
        col_scale = np.concatenate([scale[data.group[own]], scale])
        blocks.append(sp.csr_matrix((col_scale[cols], (r, cols)), shape=(len(win), n_own + G)))
        scales.append(col_scale)
        tabs.append(pd.DataFrame({"side": side, "player_id": data.ids[own], "group": np.array(GROUPS)[data.group[own]],
                                  "plays": cnt[own], "_col": offset + own_col[own],
                                  "_repl": offset + n_own + data.group[own]}))
        offset += n_own + G
    X = sp.hstack([*blocks, _controls(win)]).tocsr()
    y = np.clip(win.epa.to_numpy(), -EPA_CLIP, EPA_CLIP)
    sw = np.sqrt(w)
    A = sp.diags(sw) @ X
    b = lsqr(A, sw * (y - np.average(y, weights=w)), damp=np.sqrt(lam), atol=1e-6, btol=1e-6, iter_lim=3000)[0]
    b = b[:offset] * np.concatenate(scales)          # back to EPA per play
    tab = pd.concat(tabs, ignore_index=True)
    d = b[tab._col.to_numpy()] - b[tab._repl.to_numpy()]
    tab["value_play"] = np.where(tab.side == "off", d, -d)
    tab = tab.drop(columns=["_col", "_repl"])
    st = special[(special.t < before) & (special.season >= cur - WINDOW)]
    sts = []
    for kind, k, per_game in (("K", K_SHRINK, FG_PER_GAME), ("P", P_SHRINK, PUNTS_PER_GAME)):
        s = st[st.kind == kind]
        if s.empty:
            continue
        sw_ = DECAY ** (cur - s.season.to_numpy())
        mu = np.average(s.epa, weights=sw_)
        g = s.assign(w=sw_, we=sw_ * (s.epa - mu)).groupby("player_id")[["w", "we"]].sum()
        v = g.we / (g.w + k)
        sts.append(pd.DataFrame({"side": kind, "player_id": g.index, "group": kind,
                                 "plays": g.w.to_numpy(), "value_play": v.to_numpy()}))
    return Values(before, pd.concat([tab, *sts], ignore_index=True))


# -- lineups -> team values -------------------------------------------------------------------------

def lineup_values(rows: pd.DataFrame, vals: pd.DataFrame, plays_at: dict[tuple[str, int], float]) -> dict:
    """One team's depth chart (``group, rank, player_id``) -> points per game over replacement:
    ``off``, ``def``, ``st``, ``qb``."""
    vp = vals.set_index(["side", "player_id"]).value_play
    out = {"off": 0.0, "def": 0.0, "st": 0.0, "qb": 0.0}
    for r in rows.itertuples():
        if r.group in OFF_GROUPS:
            v = vp.get(("off", r.player_id), 0.0) * plays_at.get((r.group, int(r.rank)), 0.0)
            out["off"] += v
            if r.group == "QB" and int(r.rank) == 0:
                out["qb"] = v
        elif r.group in DEF_GROUPS:
            out["def"] += vp.get(("def", r.player_id), 0.0) * plays_at.get((r.group, int(r.rank)), 0.0)
        elif r.group in ("K", "P") and int(r.rank) == 0:
            per = FG_PER_GAME if r.group == "K" else PUNTS_PER_GAME
            out["st"] += vp.get((r.group, r.player_id), 0.0) * per
    return out
