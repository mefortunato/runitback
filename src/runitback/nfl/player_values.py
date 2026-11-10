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

from ..wager.nfl.lineup import LINEUP
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


# -- per-game lineup values (the wager model's ``_pv_*`` columns) ----------------------------------

# the layout is the wager model's (Sport.lineup, runitback.wager.nfl.lineup)
GROUP_COLUMNS = LINEUP.groups   # _pv_0 .. _pv_8 (K and P summed into ST)
RECENT = LINEUP.recent          # games behind the per-group changes (about the wager model's history)
PV_COLUMNS = LINEUP.columns     # _pv_0 .. _pv_19


@dataclass
class GameValues:
    pv: pd.DataFrame          # game_id, team, _pv_0 .. _pv_19 (every game from first_season, upcoming too)
    lineups: pd.DataFrame     # upcoming games' projected charts with backups and each player's value
    slots: dict               # points per unit of value_play at each slot: {group: [rank 0, rank 1, ...]}
    latest: Values            # the most recent fit


def slot_multipliers(plays_at: dict[tuple[str, int], float], depth: dict[str, int] | None = None
                     ) -> dict[str, list[float]]:
    """{group: [multiplier at rank 0, 1, ...]}: a player's points per game at a starting slot are his
    ``value_play`` times this (plays on the field; field goals / punts per game for K / P). With
    ``depth``, only the first ``depth[group]`` ranks (a projected chart's starting slots)."""
    out = {}
    for g in OFF_GROUPS + DEF_GROUPS:
        n = max((k + 1 for gg, k in plays_at if gg == g), default=0)
        n = min(n, depth[g]) if depth else n
        out[g] = [float(plays_at.get((g, k), 0.0)) for k in range(n)]
    return out | {"K": [FG_PER_GAME], "P": [PUNTS_PER_GAME]}


def value_key(group: str, player_id: str) -> tuple[str, str]:
    """The (side, player_id) a player's value is kept under for a lineup slot in ``group``."""
    return ("off" if group in OFF_GROUPS else "def" if group in DEF_GROUPS else group, player_id)


def side_values(rows, vp: dict, slots: dict[str, list[float]]) -> dict[str, float]:
    """A lineup's (``(group, rank, player_id)`` rows) points over replacement per group column."""
    out = dict.fromkeys(GROUP_COLUMNS, 0.0)
    for g, k, pid in rows:
        mult = slots.get(g)
        if mult is None or k >= len(mult):
            continue
        out["ST" if g in ("K", "P") else g] += vp.get(value_key(g, pid), 0.0) * mult[k]
    return out


def game_table(data_dir: Path, raw_dir: Path, first_season: int = 2016, jobs: int = 4,
               download: bool = False, upcoming: bool = True) -> GameValues:
    """Leak-free weekly values for every week from ``first_season`` (each fit on the plays before
    it) and each team-game's ``_pv_*`` columns from its projected pre-game lineup (published depth
    chart minus players ruled out), in points per game over replacement:

        _pv_0 .. _pv_8  QB, RB, WR, TE, OL, DL, LB, DB, kicker + punter (each group's sum)
        _pv_9  offense change vs the previous game's lineup   _pv_10 defense change
        _pv_11 .. _pv_19  each group's change vs the mean of the team's last 8 games' lineups

    With ``upcoming``, the schedule's unplayed games are included (their charts and injury reports
    as published so far; ``download`` refreshes them), and ``lineups`` holds their projected charts
    with every backup, for the lineup editor."""
    from joblib import Parallel, delayed

    from .data.players import DEPTH
    from .lineup_features import (
        MOVES,
        load_availability,
        load_depth_charts,
        projected_lineups,
    )

    players = pd.read_parquet(data_dir / "players.parquet")
    group_of = dict(zip(players.player_id, players.group))
    games = pd.read_parquet(data_dir / "games.parquet")
    pg = pd.read_parquet(data_dir / "player_games.parquet")
    data = play_table(data_dir, raw_dir, group_of)
    seasons = sorted(set(games.season))
    special = special_table(raw_dir, [s for s in seasons if (raw_dir / f"pbp_{s}.parquet").exists()],
                            games[["game_id", "season", "week"]])
    plays_at = slot_plays(pg[pg.season < 2016])   # snap shares by slot (before every fit; stable)
    slots = slot_multipliers(plays_at)
    target = games[games.season >= first_season][["game_id", "season", "week", "home", "away"]]
    future = target.iloc[:0]
    sched_path = raw_dir / "players" / "schedules.parquet"
    if upcoming and sched_path.exists():
        sc = pd.read_parquet(sched_path)
        sc = sc[(sc.season >= max(seasons)) & sc.result.isna() & ~sc.game_id.isin(games.game_id)]
        future = pd.DataFrame({"game_id": sc.game_id, "season": sc.season.astype(int), "week": sc.week.astype(int),
                               "home": sc.home_team.replace(MOVES), "away": sc.away_team.replace(MOVES)})
        target = pd.concat([target, future], ignore_index=True)
    weeks = sorted({s * 100 + w for s, w in zip(target.season, target.week)})
    weeks = [t for t in weeks if (data.plays.t < t).any()]
    log.info("%d plays; fitting %d weeks", len(data.plays), len(weeks))
    fits = Parallel(n_jobs=jobs)(delayed(fit_values)(data, special, t) for t in weeks)
    vals = {f.before: dict(zip(zip(f.table.side, f.table.player_id), f.table.value_play)) for f in fits}

    off_sum = lambda v: sum(v[g] for g in OFF_GROUPS)
    def_sum = lambda v: sum(v[g] for g in DEF_GROUPS)

    tseasons = sorted(set(target.season))
    out_players, active = load_availability(raw_dir, tseasons, download=download)
    charts = load_depth_charts(raw_dir, tseasons, download=download)
    proj = projected_lineups(pg, target[~target.game_id.isin(future.game_id)], out_players, active, charts)
    bench = projected_lineups(pg, future, out_players, active, charts, bench=True) if len(future) else proj.iloc[:0]
    starting = bench[(bench["rank"] >= 0) & (bench["rank"] < bench.group.map(DEPTH).fillna(0))]
    proj_rows = {k: list(zip(d.group, d["rank"].astype(int), d.player_id))
                 for k, d in pd.concat([proj, starting]).groupby(["game_id", "team"])}
    # each team's played games by week: the previous one and the last RECENT before any game
    order = pg[["game_id", "team", "season", "week"]].drop_duplicates()
    played_by = {team: (d.season.to_numpy() * 100 + d.week.to_numpy(), d.game_id.to_numpy())
                 for team, d in order.sort_values(["season", "week"]).groupby("team")}
    actual = {k: list(zip(d.group, d.depth.astype(int), d.player_id)) for k, d in pg.groupby(["game_id", "team"])}
    rows = []
    for g in target.itertuples():
        t = g.season * 100 + g.week
        vp = vals.get(t)
        if vp is None:
            continue
        for team in (g.home, g.away):
            pr = proj_rows.get((g.game_id, team))
            if pr is None:
                continue
            cur = side_values(pr, vp, slots)
            ts, ids = played_by.get(team, (np.array([], int), np.array([], object)))
            before = ids[ts < t]
            prev = before[-1] if len(before) else None
            pv_prev = side_values(actual[(prev, team)], vp, slots) if (prev, team) in actual else None
            row = {"game_id": g.game_id, "team": team} | {f"_pv_{i}": cur[k] for i, k in enumerate(GROUP_COLUMNS)}
            row["_pv_9"] = off_sum(cur) - off_sum(pv_prev) if pv_prev else 0.0
            row["_pv_10"] = def_sum(cur) - def_sum(pv_prev) if pv_prev else 0.0
            recent = [side_values(actual[(h, team)], vp, slots) for h in before[-RECENT:] if (h, team) in actual]
            for i, k in enumerate(GROUP_COLUMNS):
                row[f"_pv_{11 + i}"] = cur[k] - float(np.mean([r[k] for r in recent])) if recent else 0.0
            rows.append(row)
    pv = pd.DataFrame(rows, columns=["game_id", "team", *PV_COLUMNS])
    pv[PV_COLUMNS] = pv[PV_COLUMNS].astype(np.float32)

    names = players.set_index("player_id")
    lineups = bench.merge(future[["game_id", "season", "week"]], on="game_id")
    lineups["value_play"] = [vals.get(s * 100 + w, {}).get(value_key(g, p), 0.0)
                             for s, w, g, p in zip(lineups.season, lineups.week, lineups.group, lineups.player_id)]
    lineups["name"] = lineups.player_id.map(names.name)
    lineups["position"] = lineups.player_id.map(names.position)
    lineups = lineups[["game_id", "team", "group", "rank", "player_id", "name", "position", "value_play", "source"]]
    return GameValues(pv, lineups.reset_index(drop=True), slot_multipliers(plays_at, DEPTH), fits[-1])


def write_game_table(gv: GameValues, out: Path, players: pd.DataFrame | None = None) -> None:
    """``<out>/pv.parquet`` (the model's lineup table), ``lineups.parquet`` and ``slots.json`` (the
    lineup editor's), ``values_latest.csv`` (the latest fit, every player's value per play and as
    a starter)."""
    import json

    out.mkdir(parents=True, exist_ok=True)
    gv.pv.to_parquet(out / "pv.parquet", index=False)
    gv.lineups.to_parquet(out / "lineups.parquet", index=False)
    (out / "slots.json").write_text(json.dumps(gv.slots))
    last = gv.latest.table.copy()
    if players is not None:
        last["name"] = last.player_id.map(players.set_index("player_id").name)
    starter = {g: m[0] for g, m in gv.slots.items()}
    last["pts_as_starter"] = last.value_play * last.group.map(starter).fillna(0.0)
    last.sort_values(["group", "pts_as_starter"], ascending=[True, False]).to_csv(
        out / "values_latest.csv", index=False, float_format="%.4f")
