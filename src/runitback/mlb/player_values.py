"""MLB player values: plate-appearance ridge adjusted plus-minus on RE24 run value, in runs per game
over replacement.

The MLB adapter of the sport-agnostic fitter (:mod:`runitback.core.player_values`):

- **Events: plate appearances** (M1's ``pa_<S>`` tables). The target is the PA's RE24 run value
  (``run_value``: the change in run expectancy over the PA's own event plus the runs it scored; the
  run-expectancy matrix of season S is fit on season S-1, so the target is leak-free too).
- **The fit.** Run value on a batting coefficient for the batter and a pitching coefficient for the
  pitcher (optionally a fielding coefficient for each of the eight fielders behind him,
  ``fielding=True``), plus controls: the park (venue), the batter / pitcher handedness matchup, the
  base / out state before the event, home batting, the pitcher's role (a starter's first, second,
  third, fourth+ time through the order, or a reliever) and the season. Ridge shrinkage toward 0,
  recent seasons weighted more.
- **Role-specific players.** A player's batting, pitching and fielding coefficients are separate
  columns with separate replacement levels (a two-way player has both). Groups: ``B`` batters,
  ``PB`` pitchers batting (primary position P; mostly pre-2022 National League), ``P`` pitchers,
  ``F`` / ``C`` fielders (catchers apart). Players under ``MIN_PA`` plate appearances in a role (in
  the window) share that role's replacement-level coefficient; a value is the coefficient over it
  (pitching and fielding with the sign flipped, so higher is better).
- **Runs per game, through usage.** The lineup table's groups are the wager layout's
  (:data:`runitback.wager.mlb.lineup.LINEUP`): ``BAT`` the nine starters' batting values times
  their batting slot's plate appearances per game (the league's starters in that slot, previous
  season); ``SP`` the starter's pitching value times his expected batters faced (his last
  ``SP_STARTS`` starts, shrunk toward the league's mean start); ``BP`` the bullpen, the relievers
  the team used in its last ``BULLPEN_GAMES`` games weighted by their share of its relief batters
  faced, times the league's relief batters faced per game.
- **Leak-free and weekly.** Each week (Monday) is fit on the plate appearances before it, and every
  game uses its week's fit; usages come from earlier games or the previous season.

:func:`game_table` writes each team-game's ``_pv_*`` columns. Until the MLB projected lineups (work
package M3) exist, the projection is a stand-in: **the team's previous game's starting nine**
(batting order kept), **the game's actual starting pitcher** (announced before the game; the Stats
API keeps no history of the announcement, see M1) and the bullpen above. Pass ``lineups`` to use
another projection.
"""
from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..core.player_values import APMConfig, Events, fit_apm, fit_weekly
from ..wager.mlb.lineup import LINEUP

log = logging.getLogger(__name__)

GROUPS = ("B", "PB", "P", "F", "C")
BAT_SIDE, PIT_SIDE, FLD_SIDE = ("bat", 1.0), ("pit", -1.0), ("fld", -1.0)
WINDOW = 3                      # previous seasons in each fit (plus the current one)
DECAY = 0.7                     # weight of a PA from k seasons back: DECAY ** k
MIN_PA = 300                    # PAs in a role (in the window) for a player's own coefficient
LAMBDA = 1000.0                 # ridge penalty (tuned on the next season's PAs; see experiments/mlb)
CTRL_SCALE = 30.0               # controls are scaled up, i.e. barely penalized
TARGET = "run_value"
FIELDERS = [f"fielder_{p}" for p in range(2, 10)]
SLOTS = 9
SP_STARTS = 10                  # a starter's previous starts behind his expected batters faced
SP_SHRINK = 3.0                 # pseudo-starts at the league's mean batters faced
BULLPEN_GAMES = 15              # team games behind the bullpen projection
N_ROLE = 5                      # starter's 1st / 2nd / 3rd / 4th+ time through the order, reliever
HANDS = {("L", "L"): 0, ("L", "R"): 1, ("R", "L"): 2, ("R", "R"): 3}   # (batter, pitcher); else 4
N_HAND = 5

PA_COLUMNS = ["game_pk", "season", "at_bat_index", "bat_team_id", "fld_team_id", "bat_home", "batter_id",
              "bat_side", "spot", "pitcher_id", "pitch_hand", "pre_bases", "pre_outs", TARGET, *FIELDERS]


def day_number(date) -> int:
    """Days since 1970-01-01 of a ``YYYY-MM-DD`` date (the fits' time key ``t``)."""
    return (datetime.date.fromisoformat(str(date)[:10]) - datetime.date(1970, 1, 1)).days


def week_start(t: int) -> int:
    """The Monday on or before day ``t`` (1970-01-01 was a Thursday)."""
    return t - (t + 3) % 7


# -- loading ----------------------------------------------------------------------------------------

def _read_seasons(proc: Path, name: str, seasons, columns=None) -> pd.DataFrame:
    parts = []
    for s in seasons:
        p = Path(proc) / f"{name}_{s}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p, columns=columns))
        else:
            log.warning("%s missing; skipped", p.name)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


@dataclass
class Tables:
    pa: pd.DataFrame         # plate appearances with run values, in game order (see :func:`prepare_pa`)
    games: pd.DataFrame      # one row per game, with ``t`` and the game order
    lineups: pd.DataFrame    # M1's lineups
    players: pd.DataFrame    # player_id, name, bats, throws, position


def load(proc: Path, seasons) -> Tables:
    """M1's processed tables (``runitback mlb process``) for ``seasons``, prepared for the fits."""
    games = _read_seasons(proc, "games", seasons)
    pa = _read_seasons(proc, "pa", seasons, PA_COLUMNS)
    lineups = _read_seasons(proc, "lineups", seasons)
    players = _read_seasons(proc, "players", seasons).drop_duplicates("player_id", keep="last")
    games = prepare_games(games)
    return Tables(prepare_pa(pa, games), games, lineups, players.reset_index(drop=True))


def prepare_games(games: pd.DataFrame) -> pd.DataFrame:
    """Finished games with ``t`` (day number) and ``order`` (game order: date, start time, game_pk)."""
    g = games[games.final.astype(bool)].copy() if "final" in games else games.copy()
    g["t"] = g.date.map(day_number).astype(np.int32)
    g = g.sort_values(["t", "start", "game_pk"] if "start" in g else ["t", "game_pk"], kind="stable")
    g["order"] = np.arange(len(g))
    return g.reset_index(drop=True)


def prepare_pa(pa: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """PAs with a run value in the ``games`` given, in game order, with ``t``, ``venue``, ``hand``
    (batter / pitcher matchup code), ``state`` (base / out state before the event, 0-23), ``is_sp``
    (the pitcher started the game) and ``role`` (a starter's time through the order 0-3, 4 a
    reliever)."""
    g = games.set_index("game_pk")
    pa = pa[pa.game_pk.isin(g.index) & pa[TARGET].notna() & pa.batter_id.notna() & pa.pitcher_id.notna()]
    pa = pa.assign(_order=pa.game_pk.map(g.order)).sort_values(["_order", "at_bat_index"], kind="stable")
    pa = pa.drop(columns="_order").reset_index(drop=True)
    pa["batter_id"] = pa.batter_id.astype(np.int64)
    pa["pitcher_id"] = pa.pitcher_id.astype(np.int64)
    pa["t"] = pa.game_pk.map(g.t).astype(np.int32)
    pa["venue"] = pa.game_pk.map(g.venue_id).fillna(-1).astype(np.int64)
    # the fielding team's starter: the home team's when the away team bats
    sp_id = np.where(pa.bat_home.to_numpy() == 1, pa.game_pk.map(g.away_sp), pa.game_pk.map(g.home_sp))
    pa["is_sp"] = pa.pitcher_id.to_numpy() == sp_id
    bf = pa.groupby(["game_pk", "pitcher_id"]).cumcount().to_numpy()
    pa["role"] = np.where(pa.is_sp, np.minimum(bf // 9, 3), 4).astype(np.int8)
    pa["hand"] = [HANDS.get(k, 4) for k in zip(pa.bat_side, pa.pitch_hand)]
    pa["hand"] = pa.hand.astype(np.int8)
    pa["state"] = (pa.pre_bases.astype(int) * 3 + pa.pre_outs.clip(0, 2).astype(int)).astype(np.int8)
    return pa


# -- fits -------------------------------------------------------------------------------------------

@dataclass
class MLBEvents(Events):
    """:class:`~runitback.core.player_values.Events` whose codes are role-specific: each side's codes
    index ``ids[start:stop]`` of its ``segments`` entry (sorted player ids)."""
    segments: dict[str, tuple[int, int]] = None

    def codes(self, side: str, player_ids) -> np.ndarray:
        """Player ids -> this side's codes (-1 for ids not in it)."""
        a, b = self.segments[side]
        seg, pid = self.ids[a:b], np.asarray(player_ids)
        if not len(seg):
            return np.full(len(pid), -1)
        i = np.searchsorted(seg, pid).clip(0, len(seg) - 1)
        return np.where(seg[i] == pid, a + i, -1)


def sides(fielding: bool = False) -> tuple[tuple[str, float], ...]:
    return (BAT_SIDE, PIT_SIDE, FLD_SIDE) if fielding else (BAT_SIDE, PIT_SIDE)


def events_from(pa: pd.DataFrame, players: pd.DataFrame, fielding: bool = False) -> MLBEvents:
    """PAs as the generic fitter's events: sides ``bat`` (the batter) and ``pit`` (the pitcher), and
    with ``fielding`` ``fld`` (fielders 2-9). ``pa`` must be sorted by ``t`` (as :func:`prepare_pa`
    leaves it)."""
    position = dict(zip(players.player_id, players.position))
    gi = {g: i for i, g in enumerate(GROUPS)}
    arrays = {"bat": pa[["batter_id"]].to_numpy(), "pit": pa[["pitcher_id"]].to_numpy()}
    if fielding:
        arrays["fld"] = pa[FIELDERS].to_numpy(dtype="float64", na_value=np.nan)
    ids, groups, codes, segments = [], [], {}, {}
    start = 0
    for side, arr in arrays.items():
        vals = arr[np.isfinite(arr)] if arr.dtype.kind == "f" else arr.ravel()
        seg = np.unique(vals.astype(np.int64))
        if side == "bat":
            grp = [gi["PB"] if position.get(int(p)) == "P" else gi["B"] for p in seg]
        elif side == "pit":
            grp = [gi["P"]] * len(seg)
        else:
            grp = [gi["C"] if position.get(int(p)) == "C" else gi["F"] for p in seg]
        filled = np.where(np.isfinite(arr), arr, -1).astype(np.int64) if arr.dtype.kind == "f" else arr
        c = np.searchsorted(seg, filled).astype(np.int32) + start
        codes[side] = np.where(filled >= 0, c, -1).astype(np.int32)
        segments[side] = (start, start + len(seg))
        ids.append(seg)
        groups += grp
        start += len(seg)
    frame = pa[["game_pk", "t", "season", "bat_home", "venue", "hand", "state", "role", TARGET]]
    return MLBEvents(frame, codes, np.concatenate(ids), np.array(groups, dtype=np.int32), segments)


def _controls(df: pd.DataFrame) -> sp.csr_matrix:
    n = len(df)
    venue = pd.factorize(df.venue)[0]
    seas = (df.season - df.season.min()).astype(int).to_numpy()
    blocks = [(df.bat_home.to_numpy().astype(int), 2), (venue, venue.max() + 1 if n else 1),
              (df.hand.to_numpy().astype(int), N_HAND), (df.state.to_numpy().astype(int), 24),
              (df.role.to_numpy().astype(int), N_ROLE), (seas, seas.max() + 1 if n else 1)]
    cols, offset = [], 0
    for codes, width in blocks:
        cols.append(offset + codes)
        offset += int(width)
    cols = np.stack(cols, 1).ravel()
    rows = np.repeat(np.arange(n), len(blocks))
    return sp.csr_matrix((np.full(len(rows), CTRL_SCALE), (rows, cols)), shape=(n, offset))


def apm_config(lam: float = LAMBDA, window: int = WINDOW, decay: float = DECAY, min_pa: int = MIN_PA,
               fielding: bool = False, group_scale: dict | None = None) -> APMConfig:
    """The MLB's adjusted plus-minus settings for the generic fitter."""
    return APMConfig(groups=GROUPS, target=TARGET, controls=_controls, sides=sides(fielding),
                     group_scale=group_scale or {}, min_events=min_pa, decay=decay, window=window, lam=lam,
                     iter_lim=1000)


def fit_values(events: MLBEvents, before: int, season: int, cfg: APMConfig | None = None) -> pd.DataFrame:
    """``side, player_id, group, pa, value_pa`` from the PAs with ``t < before``: runs per plate
    appearance over replacement (higher is better on every side)."""
    tab = fit_apm(events, cfg or apm_config(), before, season)
    return tab.rename(columns={"events": "pa", "value": "value_pa"})


def predict(events: MLBEvents, tab: pd.DataFrame, rows: np.ndarray, cfg: APMConfig) -> np.ndarray:
    """Each PA's run value over the league's from the players' values in ``tab`` (players without a
    value: replacement, 0)."""
    pred = np.zeros(len(rows))
    for side, sign in cfg.sides:
        v = np.zeros(len(events.ids) + 1)           # last slot: empty fielder slots (-1)
        t = tab[tab.side == side]
        v[events.codes(side, t.player_id.to_numpy())] = t.value.to_numpy()
        pred += sign * v[events.sides[side][rows]].sum(1)
    return pred


def holdout_score(events: MLBEvents, test_season: int, cfg: APMConfig) -> dict:
    """Fit on everything before ``test_season`` and predict its PAs' run values from the players'
    values: the out-of-sample reduction in mean squared error vs a constant, and the correlation of
    the prediction with the run value. For tuning ``cfg``."""
    fr = events.frame
    test = np.flatnonzero((fr.season == test_season).to_numpy())
    tab = fit_apm(events, cfg, int(fr.t.iloc[test].min()), test_season)
    pred = predict(events, tab, test, cfg)
    y = fr[TARGET].to_numpy()[test]
    pred = pred - pred.mean()        # values are over replacement: the constant is the intercept's
    base = ((y - y.mean()) ** 2).mean()
    resid = y - y.mean() - pred
    return {"season": test_season, "mse_gain": float(base - (resid ** 2).mean()),
            "corr": float(np.corrcoef(pred, y)[0, 1])}


# -- usage ------------------------------------------------------------------------------------------

def _shift_by_season(own: dict) -> dict:
    """A season's games use the previous season's usage (the first season its own)."""
    seasons = sorted(own)
    return {s: own[seasons[max(i - 1, 0)]] for i, s in enumerate(seasons)}


def starters(lineups: pd.DataFrame) -> pd.DataFrame:
    """Each team-game's starting nine: ``game_pk, team_id, player_id, spot`` (1-9)."""
    s = lineups[lineups.starter.astype(bool) & lineups.spot.between(1, SLOTS)]
    return s[["game_pk", "team_id", "player_id", "spot"]].astype({"spot": int}).reset_index(drop=True)


def slot_pa(pa: pd.DataFrame, lineups: pd.DataFrame) -> dict[int, list[float]]:
    """{season: [plate appearances per game of the starter in batting slot 1 .. 9]} (previous
    season's league means)."""
    n = pa.groupby(["game_pk", "batter_id"]).size().rename("n").reset_index()
    st = starters(lineups).merge(n.rename(columns={"batter_id": "player_id"}), on=["game_pk", "player_id"],
                                 how="left").fillna({"n": 0})
    st = st[st.game_pk.isin(set(pa.game_pk))]
    st = st.merge(pa.drop_duplicates("game_pk")[["game_pk", "season"]], on="game_pk")
    by = st.groupby(["season", "spot"]).n.mean()
    own = {int(s): [float(by.get((s, k), 0.0)) for k in range(1, SLOTS + 1)]
           for s in by.index.get_level_values(0).unique()}
    return _shift_by_season(own)


def pitcher_games(pa: pd.DataFrame) -> pd.DataFrame:
    """Batters faced per pitcher per game: ``game_pk, t, season, team_id, pitcher_id, is_sp, bf``."""
    d = pa.groupby(["game_pk", "fld_team_id", "pitcher_id"], sort=False).agg(
        t=("t", "first"), season=("season", "first"), is_sp=("is_sp", "first"), bf=("t", "size")).reset_index()
    return d.rename(columns={"fld_team_id": "team_id"})


def league_bf(pg: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """{season: (mean batters faced by a starter, mean relief batters faced per team-game)}
    (previous season's)."""
    own = {}
    for s, d in pg.groupby("season"):
        total = d.groupby(["game_pk", "team_id"]).bf.sum()
        starter = d[d.is_sp].groupby(["game_pk", "team_id"]).bf.sum().reindex(total.index, fill_value=0)
        own[int(s)] = (float(d.bf[d.is_sp].mean()), float((total - starter).mean()))
    return _shift_by_season(own)


def expected_sp_bf(pg: pd.DataFrame, league: dict) -> pd.Series:
    """Each start's expected batters faced, from the pitcher's previous ``SP_STARTS`` starts (any
    season) shrunk toward the league mean by ``SP_SHRINK`` pseudo-starts; indexed like ``pg``'s
    starts (``game_pk, pitcher_id``)."""
    st = pg[pg.is_sp].sort_values(["pitcher_id", "t", "game_pk"], kind="stable")
    g = st.groupby("pitcher_id").bf
    prev_sum = g.transform(lambda x: x.shift(1).rolling(SP_STARTS, min_periods=1).sum()).fillna(0)
    prev_n = g.transform(lambda x: x.shift(1).rolling(SP_STARTS, min_periods=1).count()).fillna(0)
    mean = st.season.map(lambda s: league[s][0] if s in league else np.nan)
    exp = (prev_sum + SP_SHRINK * mean) / (prev_n + SP_SHRINK)
    return pd.Series(exp.to_numpy(), index=pd.MultiIndex.from_frame(st[["game_pk", "pitcher_id"]]))


def bullpens(pg: pd.DataFrame, games: pd.DataFrame) -> dict[tuple[int, int], list[tuple[int, float]]]:
    """{(game_pk, team_id): [(pitcher_id, share), ...]}: the relievers the team used in its previous
    ``BULLPEN_GAMES`` games, weighted by their share of its relief batters faced there (the game's
    own starter excluded)."""
    from collections import Counter, deque

    rel = pg[~pg.is_sp]
    used = {k: dict(zip(d.pitcher_id, d.bf)) for k, d in rel.groupby(["game_pk", "team_id"])}
    sps = pg[pg.is_sp].set_index(["game_pk", "team_id"]).pitcher_id.to_dict()
    tg = pd.concat([games[["game_pk", "order", "home_id"]].rename(columns={"home_id": "team_id"}),
                    games[["game_pk", "order", "away_id"]].rename(columns={"away_id": "team_id"})])
    out = {}
    for team, d in tg.sort_values(["team_id", "order"], kind="stable").groupby("team_id"):
        window: deque = deque()
        total: Counter = Counter()
        for g in d.game_pk:
            key = (int(g), int(team))
            w = {p: b for p, b in total.items() if b > 0 and p != sps.get(key)}
            s = sum(w.values())
            if s > 0:
                out[key] = [(int(p), b / s) for p, b in sorted(w.items())]
            game = used.get(key, {})
            window.append(game)
            total.update(game)
            if len(window) > BULLPEN_GAMES:
                total.subtract(window.popleft())
    return out


# -- per-game lineup values (the wager model's ``_pv_*`` columns) ----------------------------------

GROUP_COLUMNS = LINEUP.groups           # BAT, SP, BP
OFF_COLUMNS = ("BAT",)
PIT_COLUMNS = ("SP", "BP")
RECENT = LINEUP.recent
PV_COLUMNS = LINEUP.columns             # _pv_0 .. _pv_7


def previous_lineup(lineups: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """The stand-in projected batting lineup: each team-game's previous game's starting nine (known
    before the game). ``game_pk, team_id, player_id, spot``."""
    st = starters(lineups)
    order = games.set_index("game_pk").order
    tg = pd.concat([games[["game_pk", "home_id"]].rename(columns={"home_id": "team_id"}),
                    games[["game_pk", "away_id"]].rename(columns={"away_id": "team_id"})])
    tg = tg.assign(order=tg.game_pk.map(order)).sort_values(["team_id", "order"], kind="stable")
    tg["prev"] = tg.groupby("team_id").game_pk.shift(1)
    pr = tg.dropna(subset=["prev"]).astype({"prev": np.int64}).merge(
        st.rename(columns={"game_pk": "prev"}), on=["prev", "team_id"])
    return pr[["game_pk", "team_id", "player_id", "spot"]].reset_index(drop=True)


def side_values(bat_rows, sp_id, sp_bf, bullpen, vp: dict, mult: list[float], relief_bf: float
                ) -> dict[str, float]:
    """Runs per game over replacement per group: ``bat_rows`` ``(spot, player_id)``, the starter and
    his expected batters faced, the bullpen ``(pitcher_id, share)``."""
    bat = sum(vp.get(("bat", p), 0.0) * mult[k - 1] for k, p in bat_rows if 1 <= k <= len(mult))
    spv = vp.get(("pit", sp_id), 0.0) * sp_bf if sp_id is not None else 0.0
    bp = sum(vp.get(("pit", p), 0.0) * w for p, w in (bullpen or ())) * relief_bf
    return {"BAT": bat, "SP": spv, "BP": bp}


@dataclass
class GameValues:
    pv: pd.DataFrame          # game_pk, team_id, _pv_0 .. _pv_7
    slots: dict               # {season: PAs per game at batting slot 1-9}
    league: dict              # {season: (SP batters faced, relief batters faced per game)}
    fits: dict                # {before (day): values table}
    latest: pd.DataFrame      # the most recent fit


def game_table(tb: Tables, first_season: int = 2016, jobs: int = 4, lineups: pd.DataFrame | None = None,
               cfg: APMConfig | None = None) -> GameValues:
    """Leak-free weekly values for every week from ``first_season`` (each fit on the PAs before its
    Monday) and each team-game's ``_pv_*`` columns, in runs per game over replacement:

        _pv_0 BAT   _pv_1 SP   _pv_2 BP            the projection's value per group
        _pv_3  lineup change vs the previous game   _pv_4 pitching (SP + BP) change
        _pv_5 .. _pv_7  each group's change vs the mean of the team's last ``RECENT`` games

    ``lineups`` (``game_pk, team_id, player_id, spot``; ``spot`` 1-9 bats, 0 the starting pitcher):
    the projections; default :func:`previous_lineup` and the game's actual starter. The previous
    games are the actual starting nine and starter, with their projected bullpen."""
    cfg = cfg or apm_config()
    pa, games = tb.pa, tb.games
    events = events_from(pa, tb.players, fielding=any(s == "fld" for s, _ in cfg.sides))
    slots = slot_pa(pa, tb.lineups)
    pg = pitcher_games(pa)
    league = league_bf(pg)
    sp_bf = expected_sp_bf(pg, league).to_dict()
    pens = bullpens(pg, games)
    sel = games[games.season >= first_season]
    weeks = sorted(set(sel.t.map(week_start)))
    first_t = int(events.frame.t.min())
    weeks = [w for w in weeks if w > first_t]
    season_at = sel.assign(w=sel.t.map(week_start)).groupby("w").season.min().to_dict()
    log.info("%d plate appearances; fitting %d weeks", len(events.frame), len(weeks))
    fits = fit_weekly(events, cfg, weeks, lambda w: season_at[w], jobs)
    fits = {w: f.rename(columns={"events": "pa", "value": "value_pa"}) for w, f in zip(weeks, fits)}
    vals = {w: dict(zip(zip(f.side, f.player_id), f.value_pa)) for w, f in fits.items()}

    actual_sp = {}
    for side in ("home", "away"):
        actual_sp |= dict(zip(zip(games.game_pk, games[f"{side}_id"]), games[f"{side}_sp"]))
    actual_bat = {k: list(zip(d.spot, d.player_id)) for k, d in starters(tb.lineups).groupby(["game_pk", "team_id"])}
    if lineups is None:
        proj_bat = {k: list(zip(d.spot, d.player_id))
                    for k, d in previous_lineup(tb.lineups, games).groupby(["game_pk", "team_id"])}
        proj_sp = actual_sp
    else:
        b = lineups[lineups.spot >= 1]
        proj_bat = {k: list(zip(d.spot.astype(int), d.player_id)) for k, d in b.groupby(["game_pk", "team_id"])}
        s = lineups[lineups.spot == 0]
        proj_sp = dict(zip(zip(s.game_pk, s.team_id), s.player_id))

    def bf_of(g, sp_id, season):
        e = sp_bf.get((g, sp_id))
        if e is None or not np.isfinite(e):
            e = league.get(season, (np.nan,))[0]
        return float(e) if np.isfinite(e) else 0.0

    tg = pd.concat([games[["game_pk", "season", "t", "order", "home_id"]].rename(columns={"home_id": "team_id"}),
                    games[["game_pk", "season", "t", "order", "away_id"]].rename(columns={"away_id": "team_id"})])
    tg = tg.sort_values(["team_id", "order"], kind="stable")
    rows = []
    n = len(GROUP_COLUMNS)
    for team, d in tg.groupby("team_id"):
        hist: list[tuple[int, int]] = []          # (game_pk, season) of the team's earlier games
        for g in d.itertuples():
            key = (g.game_pk, team)
            vp = vals.get(week_start(g.t)) if g.season >= first_season else None
            if vp is not None and g.season in slots and g.season in league and key in proj_bat:
                mult, relief = slots[g.season], league[g.season][1]
                sp_id = proj_sp.get(key)
                sp_id = int(sp_id) if sp_id is not None and pd.notna(sp_id) else None
                cur = side_values(proj_bat[key], sp_id, bf_of(g.game_pk, sp_id, g.season), pens.get(key), vp,
                                  mult, relief)

                def actual(h):
                    hk = (h, team)
                    if hk not in actual_bat:
                        return None
                    s_id = actual_sp.get(hk)
                    s_id = int(s_id) if s_id is not None and pd.notna(s_id) else None
                    return side_values(actual_bat[hk], s_id, bf_of(h, s_id, g.season), pens.get(hk), vp, mult,
                                       relief)

                prev = actual(hist[-1]) if hist else None
                recent = [r for r in (actual(h) for h in hist[-RECENT:]) if r is not None]
                row = {"game_pk": g.game_pk, "team_id": team} | {f"_pv_{j}": cur[k] for j, k in enumerate(GROUP_COLUMNS)}
                row[f"_pv_{n}"] = sum(cur[k] - prev[k] for k in OFF_COLUMNS) if prev else 0.0
                row[f"_pv_{n + 1}"] = sum(cur[k] - prev[k] for k in PIT_COLUMNS) if prev else 0.0
                for j, k in enumerate(GROUP_COLUMNS):
                    row[f"_pv_{n + 2 + j}"] = cur[k] - float(np.mean([r[k] for r in recent])) if recent else 0.0
                rows.append(row)
            hist.append(g.game_pk)
    pv = pd.DataFrame(rows, columns=["game_pk", "team_id", *PV_COLUMNS])
    pv[PV_COLUMNS] = pv[PV_COLUMNS].astype(np.float32)
    pv = pv.sort_values(["game_pk", "team_id"]).reset_index(drop=True)
    return GameValues(pv, slots, league, fits, fits[weeks[-1]])


def to_wager_keys(pv: pd.DataFrame, team_keys: dict[int, str]) -> pd.DataFrame:
    """Add the wager model's keys: ``game_id`` (the game_pk as a string) and ``team`` (MLB's current
    abbreviation of the team id, ``runitback.wager.mlb.data.download.team_keys``)."""
    out = pv.copy()
    out.insert(0, "game_id", out.game_pk.astype(str))
    out.insert(1, "team", out.team_id.map(lambda i: team_keys.get(int(i))))
    return out


def write_game_table(gv: GameValues, out: Path, players: pd.DataFrame | None = None) -> None:
    """``<out>/pv.parquet`` (the wager model's lineup table), ``usage.json`` (PAs per batting slot,
    starter and relief batters faced, by season) and ``values_latest.csv`` (the latest fit: every
    player's value per PA and per game in his role)."""
    out.mkdir(parents=True, exist_ok=True)
    gv.pv.to_parquet(out / "pv.parquet", index=False)
    (out / "usage.json").write_text(json.dumps({"slot_pa": {str(k): v for k, v in gv.slots.items()},
                                                "sp_bf_relief_bf": {str(k): v for k, v in gv.league.items()}}))
    last = gv.latest.copy()
    if players is not None:
        last["name"] = last.player_id.map(players.set_index("player_id").name)
    s = max(gv.slots)
    per_game = {"bat": float(np.mean(gv.slots[s])), "pit": gv.league[s][0], "fld": float(sum(gv.slots[s]))}
    last["runs_per_game"] = last.value_pa * last.side.map(per_game)
    last.sort_values(["side", "runs_per_game"], ascending=[True, False]).to_csv(
        out / "values_latest.csv", index=False, float_format="%.4f")
