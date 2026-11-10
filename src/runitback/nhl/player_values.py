"""NHL player values: expected-goals adjusted plus-minus per shift segment, plus a goalie term.

The NHL adapter of the sport-agnostic fitter (:mod:`runitback.core.player_values`):

- **Shift segments.** H1's events table (``events_<season>.parquet``) is cut into segments
  (:func:`season_segments`): runs of events in one period with the same ten skaters and two goalies
  on ice. A segment lasts from its first event to the next segment's first event (or the period's
  last event). Segments with a goalie pulled, more than five or fewer than three skaters a side, or
  no length are not fit. The shootout is dropped.
- **The fit.** Each segment gives two rows, one per team attacking: the attacking team's expected
  goals per 60 minutes (the release's ``xg`` summed over shots, missed shots and goals) on an offense
  coefficient for each attacking skater and a defense coefficient for each defending skater, weighted
  by the segment's length (minutes). Controls: home offense, strength from the attacking side (3-5
  skaters each), how the segment started (on the fly, or a faceoff in the offensive / neutral /
  defensive zone), the score from the attacking side (clipped at 3), and season. A ridge pulls the
  player coefficients toward 0, and recent seasons weigh more.
- **Groups and replacement level.** Skaters are F (C / L / R) or D by their NHL position. Skaters
  under ``MIN_TOI`` minutes on ice in the fitted segments of the window share a replacement
  coefficient per side and group. A player's value is his coefficient over it (for defense with the sign flipped, so higher is better), in expected goals per 60.
- **Goalies** (:func:`goalie_values`): goals saved above expected per unblocked shot faced (``xg``
  minus goals, each season's league mean removed), recency-weighted and shrunk toward the league
  mean by ``GOALIE_SHRINK`` pseudo-shots. Unknown goalies are at the league mean: a replacement level
  below it (from low-usage goalies) predicted the next season worse. Empty-net shots are not a
  goalie's.
- **Goals per game.** A skater's value per 60 times his slot's minutes on ice (the k-th forward or
  defenseman of a team by time on ice, the league's mean in the previous season), offense plus
  defense. A goalie's value per shot times the shots a starter faces a game (previous season).
- **Leak-free and weekly.** Each week (Monday) is fit on the segments and shots before it; every game
  uses its week's fit, so it never sees itself or later games.

``game_table`` sums each team-game's projected lineup into the wager model's ``_pv_*`` columns
(layout :data:`runitback.wager.nhl.lineup.LINEUP`). The projection is H3's
(:func:`from_projected`): the team's previous game's skaters, ranked by time on ice, with the
starting goalie either (a) the previous game's starter (leak-free; ``pv.parquet``) or (b) the actual
starter, known at puck drop (H3's documented leak; ``pv_actual.parquet``). Without H3's table the
same previous-game lineup is rebuilt from the segments (:func:`previous_lineup`).
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

from ..core.player_values import APMConfig, Events, fit_apm, fit_weekly, shrunk_rates
from ..wager.nhl.lineup import GOALIE, LINEUP, POSITIONS
from .data.events import LINEUP_COLS

log = logging.getLogger(__name__)

GROUPS = POSITIONS                  # F / D from the NHL position
SIDES = (("off", 1.0), ("def", -1.0))
WINDOW = 4                          # previous seasons in each fit (plus the current one)
DECAY = 0.7                         # weight of a segment from k seasons back: DECAY ** k
MIN_TOI = 100                       # minutes on ice (in the window) for a skater's own coefficient
LAMBDA = 300.0                      # ridge penalty (tuned on the next season's segments)
CTRL_SCALE = 30.0                   # controls are scaled up, i.e. barely penalized
GOALIE_SHRINK = 6000.0              # pseudo-shots at the league mean (tuned on the next season)
SLOTS = {"F": 14, "D": 8}           # usage slots per group (beyond: 0)
DEFAULT_GROUP = "F"                 # skaters with no NHL position
MAX_SKATERS = 5                     # per side, in the fit
SHOTS = ("SHOT", "MISSED_SHOT", "GOAL")

# segment starts (from the attacking side)
START_FLY, START_OZ, START_NZ, START_DZ = range(4)
N_START = 4
ZONE_LINE = 25.0                    # blue lines at x = +-25 ft (``x_fixed``: home attacks +x)
SCORE_CLIP = 3
N_STRENGTH = 9                      # (3..5 attackers) x (3..5 defenders)

SEG_COLUMNS = (["game_id", "season", "game_type", "game_date", "t", "period", "seconds", "start_x", "home_lead",
                "xg_home", "xg_away", "home_goalie", "away_goalie"]
               + [f"home_{i}" for i in range(1, MAX_SKATERS + 1)] + [f"away_{i}" for i in range(1, MAX_SKATERS + 1)])
SHOT_COLUMNS = ["game_id", "season", "t", "goalie", "xg", "goal"]


def day_number(date) -> int:
    """Days since 1970-01-01 of a ``YYYY-MM-DD`` date (the fits' time key ``t``)."""
    return (datetime.date.fromisoformat(str(date)[:10]) - datetime.date(1970, 1, 1)).days


def week_start(t: int) -> int:
    """The Monday on or before day ``t`` (1970-01-01 was a Thursday)."""
    return t - (t + 3) % 7


# -- segments ---------------------------------------------------------------------------------------

def season_segments(ev: pd.DataFrame) -> pd.DataFrame:
    """One season's events (H1's table) -> shift segments, one row each, with the home and away
    skaters (sorted, ``-1`` = empty), goalies, length, each side's expected goals, the home lead at
    the segment's first event and the ``x_fixed`` of a faceoff at its first second (NaN: on the fly).
    Only segments that are fit are kept (see the module docstring)."""
    ev = ev[ev.period_type.astype(str) != "SHOOTOUT"].sort_values(["game_id", "event_idx"], kind="stable")
    ev = ev.reset_index(drop=True)
    pers = ev[LINEUP_COLS].astype("float64").to_numpy(na_value=np.nan)
    key = np.nan_to_num(pers, nan=-1.0)
    gid = ev.game_id.to_numpy()
    per = ev.period.to_numpy()
    new = np.r_[True, (gid[1:] != gid[:-1]) | (per[1:] != per[:-1]) | (key[1:] != key[:-1]).any(1)]
    seg = np.cumsum(new) - 1
    first = np.flatnonzero(new)
    secs = ev.game_seconds.to_numpy()
    start = secs[first]
    # a segment ends where the next one in its period starts, else at the period's last event
    gp = pd.Series(list(zip(gid, per)))
    period_end = pd.Series(secs).groupby(gp.factorize()[0]).transform("max").to_numpy()
    same = np.r_[(gid[first[1:]] == gid[first[:-1]]) & (per[first[1:]] == per[first[:-1]]), False]
    end = np.where(same, np.r_[start[1:], 0], period_end[first])
    et = ev.event_type.astype(str).to_numpy()
    team = ev.event_team.astype(object).to_numpy()
    xg = pd.to_numeric(ev.xg, errors="coerce").fillna(0.0).to_numpy()
    shot = np.isin(et, SHOTS)
    n = len(first)
    xg_home = np.bincount(seg, weights=np.where(shot & (team == "home"), xg, 0.0), minlength=n)
    xg_away = np.bincount(seg, weights=np.where(shot & (team == "away"), xg, 0.0), minlength=n)
    # a faceoff at the segment's first second
    fo = (et == "FACEOFF") & (secs == start[seg])
    fo_x = pd.Series(ev.x_fixed.astype("float64").to_numpy()).where(fo).groupby(seg).first().reindex(range(n))
    p = pers[first]
    cols = {c: p[:, i] for i, c in enumerate(LINEUP_COLS)}
    n_home = np.isfinite(p[:, :6]).sum(1)
    n_away = np.isfinite(p[:, 6:12]).sum(1)
    out = pd.DataFrame({
        "game_id": gid[first], "season": ev.season.to_numpy()[first],
        "game_type": ev.game_type.astype(str).to_numpy()[first],
        "game_date": ev.game_date.astype(str).str[:10].to_numpy()[first], "period": per[first],
        "seconds": (end - start).astype(np.int32), "start_x": fo_x.to_numpy(),
        "home_lead": (ev.home_score.to_numpy() - ev.away_score.to_numpy())[first].astype(np.int16),
        "xg_home": xg_home, "xg_away": xg_away,
        "home_goalie": cols["home_goalie"], "away_goalie": cols["away_goalie"]})
    for side in ("home", "away"):
        for i in range(1, MAX_SKATERS + 1):
            out[f"{side}_{i}"] = np.nan_to_num(cols[f"{side}_{i}"], nan=-1).astype(np.int64)
    ok = ((out.seconds > 0) & np.isfinite(out.home_goalie) & np.isfinite(out.away_goalie)
          & (n_home >= 3) & (n_home <= MAX_SKATERS) & (n_away >= 3) & (n_away <= MAX_SKATERS)).to_numpy()
    out = out[ok].astype({"home_goalie": "int64", "away_goalie": "int64"})
    out["t"] = out.game_date.map(day_number).astype(np.int32)
    return out[SEG_COLUMNS].reset_index(drop=True)


def season_shots(ev: pd.DataFrame) -> pd.DataFrame:
    """Unblocked shots (shots, misses and goals, not the shootout) at a goalie in net, with their
    ``xg``: ``game_id, season, t, goalie, xg, goal``."""
    e = ev[ev.event_type.astype(str).isin(SHOTS) & (ev.period_type.astype(str) != "SHOOTOUT")
           & ev.event_team.isin(["home", "away"])]
    goalie = np.where(e.event_team == "home", e.away_goalie.astype("float64"), e.home_goalie.astype("float64"))
    xg = pd.to_numeric(e.xg, errors="coerce").to_numpy(dtype=float)
    empty = e.empty_net.fillna(False).astype(bool).to_numpy()
    ok = np.isfinite(goalie) & np.isfinite(xg) & ~empty
    e = e[ok]
    out = pd.DataFrame({"game_id": e.game_id.to_numpy(), "season": e.season.to_numpy(),
                        "t": e.game_date.astype(str).str[:10].map(day_number).astype(np.int32).to_numpy(),
                        "goalie": goalie[ok].astype(np.int64), "xg": xg[ok],
                        "goal": (e.event_type.astype(str) == "GOAL").to_numpy()})
    return out[SHOT_COLUMNS].reset_index(drop=True)


def build_segments(seasons: list[int], processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``segments.parquet`` and ``shots.parquet`` (processed_dir) from H1's ``events_<season>.parquet``.
    Logs the segments, minutes and expected goals per team-game."""
    segs, shots = [], []
    for s in seasons:
        path = processed_dir / f"events_{s}.parquet"
        if not path.exists():
            log.warning("season %d: %s missing; skipped", s, path.name)
            continue
        ev = pd.read_parquet(path)
        g = season_segments(ev)
        sh = season_shots(ev)
        games = g.game_id.nunique()
        log.info("season %d: %d games, %d segments (%.1f s each), %.1f fitted minutes and %.2f xG per team-game, "
                 "%d shots at a goalie", s, games, len(g), g.seconds.mean(), g.seconds.sum() / 60 / games,
                 (g.xg_home.sum() + g.xg_away.sum()) / 2 / games, len(sh))
        segs.append(g)
        shots.append(sh)
    seg, shot = pd.concat(segs, ignore_index=True), pd.concat(shots, ignore_index=True)
    seg.to_parquet(processed_dir / "segments.parquet", index=False)
    shot.to_parquet(processed_dir / "shots.parquet", index=False)
    return seg, shot


# -- skater fits ------------------------------------------------------------------------------------

def player_groups(players: pd.DataFrame) -> dict[int, str]:
    """{player_id: F / D / G} from the player directory; no NHL position: ``DEFAULT_GROUP``."""
    return {int(p): g if g in ("F", "D", "G") else DEFAULT_GROUP for p, g in zip(players.player_id, players.group)}


def _skaters(seg: pd.DataFrame, side: str) -> np.ndarray:
    return seg[[f"{side}_{i}" for i in range(1, MAX_SKATERS + 1)]].to_numpy()


def segment_rows(seg: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Each segment as two rows (home attacking, then away attacking): the frame (``t, season, game_id,
    off_home, strength, start, score, minutes, xg60``) and the attacking / defending skater ids."""
    home, away = _skaters(seg, "home"), _skaters(seg, "away")
    nh, na = (home >= 0).sum(1), (away >= 0).sum(1)
    x = seg.start_x.to_numpy()
    zone_home = np.where(np.isnan(x), START_FLY, np.where(x > ZONE_LINE, START_OZ,
                                                          np.where(x < -ZONE_LINE, START_DZ, START_NZ)))
    flip = {START_FLY: START_FLY, START_OZ: START_DZ, START_NZ: START_NZ, START_DZ: START_OZ}
    zone_away = np.vectorize(flip.get, otypes=[np.int64])(zone_home) if len(x) else zone_home
    lead = seg.home_lead.to_numpy()
    minutes = seg.seconds.to_numpy() / 60.0
    parts = []
    for off_home, n_off, n_def, zone, score, xg in ((True, nh, na, zone_home, lead, seg.xg_home.to_numpy()),
                                                    (False, na, nh, zone_away, -lead, seg.xg_away.to_numpy())):
        parts.append(pd.DataFrame({
            "t": seg.t.to_numpy(), "season": seg.season.to_numpy(), "game_id": seg.game_id.to_numpy(),
            "off_home": off_home, "strength": ((n_off - 3) * 3 + (n_def - 3)).astype(np.int8),
            "start": zone.astype(np.int8),
            "score": (np.clip(score, -SCORE_CLIP, SCORE_CLIP) + SCORE_CLIP).astype(np.int8),
            "minutes": minutes, "xg60": xg * 60.0 / minutes}))
    frame = pd.concat(parts, ignore_index=True)
    off = np.concatenate([home, away])
    dfn = np.concatenate([away, home])
    return frame, off, dfn


def events_from(seg: pd.DataFrame, group_of: dict[int, str]) -> Events:
    """Segments as the generic fitter's events (sides ``off`` / ``def``; target ``xg60``, weight
    ``minutes``). Goalies are not skaters: one found among the skaters is dropped from the slot."""
    frame, off, dfn = segment_rows(seg)
    order = np.lexsort((frame.game_id.to_numpy(), frame.t.to_numpy()))
    frame, off, dfn = frame.iloc[order].reset_index(drop=True), off[order], dfn[order]
    ids = np.unique(np.concatenate([off.ravel(), dfn.ravel()]))
    ids = ids[ids >= 0]
    gi = {g: i for i, g in enumerate(GROUPS)}
    group = np.array([gi.get(group_of.get(int(p), DEFAULT_GROUP), -1) for p in ids], dtype=np.int32)

    def codes(a):
        c = np.searchsorted(ids, np.maximum(a, 0)).astype(np.int32)
        return np.where(a >= 0, c, -1).astype(np.int32)

    return Events(frame, {"off": codes(off), "def": codes(dfn)}, ids, group)


def _controls(df: pd.DataFrame) -> sp.csr_matrix:
    n = len(df)
    seas = (df.season - df.season.min()).astype(int).to_numpy()
    base = [0, 1, 3, 3 + N_STRENGTH, 3 + N_STRENGTH + N_START, 3 + N_STRENGTH + N_START + 2 * SCORE_CLIP + 1]
    cols = np.stack([np.zeros(n, int), base[1] + df.off_home.to_numpy().astype(int),
                     base[2] + df.strength.to_numpy().astype(int), base[3] + df.start.to_numpy().astype(int),
                     base[4] + df.score.to_numpy().astype(int), base[5] + seas], 1).ravel()
    rows = np.repeat(np.arange(n), 6)
    return sp.csr_matrix((np.full(len(rows), CTRL_SCALE), (rows, cols)), shape=(n, base[5] + seas.max() + 1))


def apm_config(lam: float = LAMBDA, window: int = WINDOW, decay: float = DECAY, min_toi: float = MIN_TOI
               ) -> APMConfig:
    """The NHL's adjusted plus-minus settings for the generic fitter (``min_toi``: minutes on ice)."""
    return APMConfig(groups=GROUPS, target="xg60", weight="minutes", count_weight=True, controls=_controls,
                     sides=SIDES, min_events=min_toi, decay=decay, window=window, lam=lam, iter_lim=1000)


def fit_values(events: Events, before: int, season: int, cfg: APMConfig | None = None) -> pd.DataFrame:
    """``side, player_id, group, toi, value_60`` (minutes on ice in the window; expected goals per 60
    over replacement) from the segments with ``t < before``."""
    tab = fit_apm(events, cfg or apm_config(), before, season)
    return tab.rename(columns={"events": "toi", "value": "value_60"})


def holdout_score(events: Events, test_season: int, cfg: APMConfig) -> dict:
    """Fit on everything before ``test_season`` and predict its segments' expected goals per 60 from
    the ten skaters' values (unknown skaters at replacement): the out-of-sample weighted mean
    squared error reduction vs a constant, and the weighted correlation. For tuning ``cfg``."""
    fr = events.frame
    test = (fr.season == test_season).to_numpy()
    before = int(fr.t[test].min())
    tab = fit_apm(events, cfg, before, test_season)
    pred = np.zeros(test.sum())
    for side, sign in cfg.sides:
        v = np.zeros(len(events.ids) + 1)          # the last entry: empty slots
        t = tab[tab.side == side]
        v[np.searchsorted(events.ids, t.player_id.to_numpy())] = t.value.to_numpy()
        c = events.sides[side][test]
        pred += sign * v[np.where(c >= 0, c, len(events.ids))].sum(1)
    y = fr.xg60.to_numpy()[test]
    w = fr.minutes.to_numpy()[test]
    mu = np.average(y, weights=w)
    pred = pred - np.average(pred, weights=w)
    base = np.average((y - mu) ** 2, weights=w)
    resid = np.average((y - mu - pred) ** 2, weights=w)
    cov = np.average(pred * (y - mu), weights=w)
    corr = cov / np.sqrt(np.average(pred ** 2, weights=w) * base)
    return {"season": test_season, "mse_gain": float(base - resid), "corr": float(corr)}


# -- goalies ----------------------------------------------------------------------------------------

def goalie_values(shots: pd.DataFrame, before: int, season: int, shrink: float = GOALIE_SHRINK,
                  window: int = WINDOW, decay: float = DECAY) -> pd.DataFrame:
    """``player_id, shots, value_shot``: each goalie's goals saved above expected per shot, from the
    shots with ``t < before`` in seasons ``season - window .. season``. Each season's league mean of
    (xG - goals) per shot is removed first (the release's xG is not calibrated the same way every
    season); the recency-weighted rate is shrunk toward 0 (the league) by ``shrink`` pseudo-shots.
    A goalie with no shots in the window is at 0."""
    d = shots[(shots.t < before) & (shots.season >= season - window)]
    if d.empty:
        return pd.DataFrame(columns=["player_id", "shots", "value_shot"])
    r = d.xg - d.goal.astype(float)
    d = d.assign(saved=r - r.groupby(d.season).transform("mean"))
    out = shrunk_rates(d.rename(columns={"goalie": "player_id"}), "player_id", "saved", season, shrink, decay)
    n = d.groupby("goalie").size()
    return pd.DataFrame({"player_id": out.player_id.astype(np.int64).to_numpy(),
                         "shots": n.reindex(out.player_id).to_numpy(), "value_shot": out.value.to_numpy()})


def goalie_holdout(shots: pd.DataFrame, test_season: int, shrink: float, window: int = WINDOW,
                   decay: float = DECAY) -> dict:
    """Fit before ``test_season``; predict its shots' (xG - goal) minus the season mean from the goalie's
    value (others at 0): squared-error reduction per shot vs 0 (x 1e4) and the correlation of the
    goalies' predicted values with their realized rates (goalies with 500+ test shots)."""
    test = shots[shots.season == test_season]
    tab = goalie_values(shots, int(test.t.min()), test_season, shrink, window, decay)
    v = dict(zip(tab.player_id, tab.value_shot))
    y = test.xg - test.goal.astype(float)
    y = (y - y.mean()).to_numpy()
    pred = test.goalie.map(v).fillna(0.0).to_numpy()
    pred = pred - pred.mean()
    gain = float(np.mean(y ** 2) - np.mean((y - pred) ** 2))
    real = pd.DataFrame({"g": test.goalie.to_numpy(), "y": y, "p": pred}).groupby("g").agg(
        n=("y", "size"), y=("y", "mean"), p=("p", "first"))
    real = real[real.n >= 500]
    return {"season": test_season, "mse_gain": 1e4 * gain, "corr": float(real.y.corr(real.p)), "goalies": len(real)}


# -- lineups and usage ------------------------------------------------------------------------------

def team_keys(games: pd.DataFrame) -> pd.DataFrame:
    """``game_id, home, away, game_date, season`` from H1's games table."""
    out = games[["game_id", "season", "game_date", "home", "away"]]
    return out.assign(game_date=out.game_date.astype(str).str[:10])


def player_games(seg: pd.DataFrame, games: pd.DataFrame, group_of: dict[int, str]) -> pd.DataFrame:
    """Each team-game's players from its fitted segments: ``game_id, season, game_date, team,
    player_id, group, toi`` (minutes), ``rank`` (within the group by time on ice; goalies: 0 = the
    starter, the goalie in net at the first segment)."""
    keys = team_keys(games).set_index("game_id")
    parts = []
    for side in ("home", "away"):
        sk = _skaters(seg, side)
        m = pd.DataFrame({"game_id": np.repeat(seg.game_id.to_numpy(), MAX_SKATERS), "player_id": sk.ravel(),
                          "toi": np.repeat(seg.seconds.to_numpy() / 60.0, MAX_SKATERS)})
        m = m[m.player_id >= 0].groupby(["game_id", "player_id"], as_index=False).toi.sum()
        m["group"] = m.player_id.map(lambda p: group_of.get(int(p), DEFAULT_GROUP)).replace({"G": DEFAULT_GROUP})
        gk = pd.DataFrame({"game_id": seg.game_id.to_numpy(), "player_id": seg[f"{side}_goalie"].to_numpy(),
                           "toi": seg.seconds.to_numpy() / 60.0, "order": np.arange(len(seg))})
        gk = gk.groupby(["game_id", "player_id"], as_index=False).agg(toi=("toi", "sum"), order=("order", "min"))
        gk = gk.sort_values(["game_id", "order"]).drop(columns="order").assign(group=GOALIE)
        d = pd.concat([m, gk], ignore_index=True)
        d["team"] = d.game_id.map(keys[side])
        parts.append(d)
    pg = pd.concat(parts, ignore_index=True)
    pg = pg.merge(keys[["season", "game_date"]], left_on="game_id", right_index=True)
    sk = pg.group != GOALIE
    pg = pd.concat([pg[sk].sort_values(["game_id", "team", "group", "toi", "player_id"],
                                       ascending=[True, True, True, False, True], kind="stable"), pg[~sk]])
    pg["rank"] = pg.groupby(["game_id", "team", "group"]).cumcount()
    cols = ["game_id", "season", "game_date", "team", "player_id", "group", "toi", "rank"]
    return pg[cols].sort_values(["game_date", "game_id", "team", "group", "rank"], kind="stable").reset_index(drop=True)


@dataclass
class Usage:
    toi: dict        # {season: {"F": [minutes at slot 0, 1, ...], "D": [...]}}
    shots: dict      # {season: unblocked shots a starting goalie faces a game}


def usage(pg: pd.DataFrame, shots: pd.DataFrame) -> Usage:
    """Mean minutes on ice per slot and shots faced per starting goalie's game, by season. A season's
    games use the previous season's (the first season its own)."""
    sk = pg[pg.group != GOALIE]
    team_games = pg.groupby("season").apply(lambda d: d[["game_id", "team"]].drop_duplicates().shape[0],
                                            include_groups=False)
    own_toi = {}
    for s, d in sk.groupby("season"):
        own_toi[int(s)] = {g: [float(d[(d.group == g) & (d["rank"] == k)].toi.sum() / team_games[s])
                               for k in range(SLOTS[g])] for g in GROUPS}
    st = pg[(pg.group == GOALIE) & (pg["rank"] == 0)][["game_id", "player_id", "season"]]
    faced = shots.groupby(["game_id", "goalie"]).size().rename("n").reset_index()
    st = st.merge(faced, left_on=["game_id", "player_id"], right_on=["game_id", "goalie"], how="left").fillna({"n": 0})
    own_shots = {int(s): float(v) for s, v in st.groupby("season").n.mean().items()}
    seasons = sorted(own_toi)
    prev = {s: seasons[max(i - 1, 0)] for i, s in enumerate(seasons)}
    return Usage({s: own_toi[prev[s]] for s in seasons}, {s: own_shots.get(prev[s], 0.0) for s in seasons})


def previous_lineup(pg: pd.DataFrame, target: pd.DataFrame | None = None) -> pd.DataFrame:
    """The stand-in projected lineup: for each team-game, the team's previous game's skaters (ranked
    by time on ice within F / D) and its starting goalie (rank 0), known at puck drop. ``target``:
    (game_id, team) rows (default every team-game in ``pg``)."""
    order = pg[["game_id", "team", "game_date"]].drop_duplicates(["game_id", "team"]).sort_values(
        ["team", "game_date", "game_id"], kind="stable")
    order["prev"] = order.groupby("team").game_id.shift(1)
    if target is not None:
        order = target[["game_id", "team"]].merge(order[["game_id", "team", "prev"]], on=["game_id", "team"])
    cols = ["team", "player_id", "group", "rank"]
    pr = order.dropna(subset=["prev"]).astype({"prev": pg.game_id.dtype}).merge(
        pg[["game_id", *cols]].rename(columns={"game_id": "prev"}), on=["prev", "team"])
    pr = pr[(pr.group != GOALIE) | (pr["rank"] == 0)]
    return pr[["game_id", *cols]].reset_index(drop=True)


# -- per-game lineup values (the wager model's ``_pv_*`` columns) ----------------------------------

GROUP_COLUMNS = LINEUP.groups        # F_OFF, D_OFF, F_DEF, D_DEF, G
OFF_COLUMNS = tuple(f"{g}_OFF" for g in GROUPS)
DEF_COLUMNS = tuple(f"{g}_DEF" for g in GROUPS)
RECENT = LINEUP.recent
PV_COLUMNS = LINEUP.columns          # _pv_0 .. _pv_12


def side_values(rows, vals: dict, gvals: dict, toi: dict, shots_pg: float) -> dict[str, float]:
    """A lineup (``(group, rank, player_id)`` rows) -> goals per game over replacement per group column.
    ``vals``: {(side, player_id): xG per 60}; ``gvals``: {goalie: goals saved above expected per shot}."""
    out = dict.fromkeys(GROUP_COLUMNS, 0.0)
    for g, k, pid in rows:
        if g == GOALIE:
            if k == 0:
                out[GOALIE] += gvals.get(pid, 0.0) * shots_pg
            continue
        mult = toi[g][k] / 60.0 if k < len(toi[g]) else 0.0
        out[f"{g}_OFF"] += vals.get(("off", pid), 0.0) * mult
        out[f"{g}_DEF"] += vals.get(("def", pid), 0.0) * mult
    return out


@dataclass
class GameValues:
    pv: pd.DataFrame          # game_id, team, _pv_0 .. _pv_12
    usage: Usage
    fits: dict                # {before (day): skater values table}
    goalie_fits: dict         # {before (day): goalie values table}
    latest: pd.DataFrame      # the most recent skater fit
    latest_goalies: pd.DataFrame


def _rows(df: pd.DataFrame) -> dict:
    return {k: list(zip(d.group, d["rank"].astype(int), d.player_id.astype(int)))
            for k, d in df.groupby(["game_id", "team"])}


def from_projected(proj: pd.DataFrame, goalie: str = "prev") -> pd.DataFrame:
    """H3's ``projected_lineups.parquet`` (:mod:`runitback.nhl.data.lineups`) -> ``lineups`` for
    :func:`game_table`: the projected skaters ranked by their slot (time on ice in the previous game)
    and one starting goalie, (a) ``goalie="prev"`` the previous game's starter (leak-free) or (b)
    ``"actual"`` the game's actual starter (known at puck drop: H3's documented leak)."""
    from .data.lineups import lineup

    lu = lineup(proj, goalie)
    rank = np.where(lu.group == GOALIE, 0, lu.slot.fillna(99).astype(int) - 1)
    return pd.DataFrame({"game_id": lu.game_id.to_numpy(), "team": lu.team.to_numpy(),
                         "player_id": lu.player_id.astype(np.int64).to_numpy(), "group": lu.group.to_numpy(),
                         "rank": rank})


def game_table(seg: pd.DataFrame, shots: pd.DataFrame, games: pd.DataFrame, players: pd.DataFrame,
               first_season: int = 2013, jobs: int = 4, lineups: pd.DataFrame | None = None,
               cfg: APMConfig | None = None, goalie_shrink: float = GOALIE_SHRINK,
               fitted: GameValues | None = None) -> GameValues:
    """Leak-free weekly values for every week from ``first_season`` (each fit on the segments and shots
    before its Monday) and each team-game's ``_pv_*`` columns from its projected lineup, in goals per
    game over replacement:

        _pv_0 .. _pv_4   F, D offense; F, D defense; starting goalie
        _pv_5  offense change vs the previous game's lineup   _pv_6 defense change   _pv_7 goalie change
        _pv_8 .. _pv_12  each group's change vs the mean of the team's last ``RECENT`` lineups

    ``lineups`` (``game_id, team, player_id, group, rank``; group F / D / G, the goalie with rank 0 is
    the starter): the projected lineups, e.g. :func:`from_projected` (H3's); default
    :func:`previous_lineup`. ``team`` is the games table's team code. ``fitted``: an earlier result
    whose weekly fits and usage to reuse (another projection of the same games)."""
    cfg = cfg or apm_config()
    group_of = player_groups(players)
    events = events_from(seg, group_of)
    pg = player_games(seg, games, group_of)
    use = usage(pg, shots)
    tg = pg[pg.season >= first_season][["game_id", "season", "game_date", "team"]].drop_duplicates(["game_id", "team"])
    tg = tg.assign(t=tg.game_date.map(day_number))
    tg["before"] = tg.t.map(week_start)
    first_t = int(events.frame.t.min())
    weeks = sorted(w for w in set(tg.before) if w > first_t)
    season_at = tg.groupby("before").season.min().to_dict()
    if fitted is not None:
        fits, gfits = fitted.fits, fitted.goalie_fits
    else:
        log.info("%d segment rows; fitting %d weeks", len(events.frame), len(weeks))
        fits = fit_weekly(events, cfg, weeks, lambda w: season_at[w], jobs)
        fits = {w: f.rename(columns={"events": "toi", "value": "value_60"}) for w, f in zip(weeks, fits)}
        gfits = {w: goalie_values(shots, w, season_at[w], goalie_shrink, cfg.window, cfg.decay)
                 for w in weeks}
    vals = {w: dict(zip(zip(f.side, f.player_id), f.value_60)) for w, f in fits.items()}
    gvals = {w: dict(zip(f.player_id.astype(int), f.value_shot)) for w, f in gfits.items()}

    proj = previous_lineup(pg, tg) if lineups is None else lineups
    proj_rows = _rows(proj)
    actual = _rows(pg[(pg.group != GOALIE) | (pg["rank"] == 0)])
    order = pg[["game_id", "team", "game_date"]].drop_duplicates(["game_id", "team"]).sort_values(
        ["game_date", "game_id"], kind="stable")
    played_by = {team: d.game_id.to_numpy() for team, d in order.groupby("team")}
    pos_in = {(g, team): i for team, ids in played_by.items() for i, g in enumerate(ids)}
    seasons = sorted(use.toi)
    n = len(GROUP_COLUMNS)
    summaries = [cols for _, cols in LINEUP.summaries]
    rows = []
    for g in tg.itertuples():
        vp, gv = vals.get(g.before), gvals.get(g.before)
        pr = proj_rows.get((g.game_id, g.team))
        if vp is None or pr is None:
            continue
        s = g.season if g.season in use.toi else seasons[-1]
        args = (vp, gv, use.toi[s], use.shots[s])
        cur = side_values(pr, *args)
        i = pos_in[(g.game_id, g.team)]
        before = played_by[g.team][:i]
        prev = before[-1] if len(before) else None
        pv_prev = side_values(actual[(prev, g.team)], *args) if (prev, g.team) in actual else None
        row = {"game_id": g.game_id, "team": g.team} | {f"_pv_{j}": cur[k] for j, k in enumerate(GROUP_COLUMNS)}
        for j, cols in enumerate(summaries):
            row[f"_pv_{n + j}"] = sum(cur[k] - pv_prev[k] for k in cols) if pv_prev else 0.0
        recent = [side_values(actual[(h, g.team)], *args) for h in before[-RECENT:] if (h, g.team) in actual]
        for j, k in enumerate(GROUP_COLUMNS):
            row[f"_pv_{n + len(summaries) + j}"] = cur[k] - float(np.mean([r[k] for r in recent])) if recent else 0.0
        rows.append(row)
    pv = pd.DataFrame(rows, columns=["game_id", "team", *PV_COLUMNS])
    pv[PV_COLUMNS] = pv[PV_COLUMNS].astype(np.float32)
    return GameValues(pv, use, fits, gfits, fits[weeks[-1]], gfits[weeks[-1]])


def to_wager_teams(pv: pd.DataFrame) -> pd.DataFrame:
    """Re-key to the wager model's games table: ``game_id`` as a string and ``team`` the franchise's
    current code (``runitback.wager.nhl.data.download.MOVES``: Atlanta is Winnipeg, Arizona Utah)."""
    from ..wager.nhl.data.download import MOVES

    return pv.assign(game_id=pv.game_id.astype(str), team=pv.team.map(lambda t: MOVES.get(t, t)))


def write_game_table(gv: GameValues, out: Path, players: pd.DataFrame | None = None) -> None:
    """``<out>/pv.parquet`` (the wager model's lineup table), ``fits.parquet`` / ``goalie_fits.parquet``
    (every weekly fit, keyed by ``before``: the day it was fit before), ``usage.json`` (minutes per slot and a
    starter's shots faced, by season) and ``values_latest.csv`` (the latest fit: every skater's value per
    60 and per game in the top slot of his group, offense and defense, and every goalie's per shot and
    per game)."""
    out.mkdir(parents=True, exist_ok=True)
    gv.pv.to_parquet(out / "pv.parquet", index=False)
    pd.concat([f.assign(before=w) for w, f in gv.fits.items()], ignore_index=True).to_parquet(
        out / "fits.parquet", index=False)
    pd.concat([f.assign(before=w) for w, f in gv.goalie_fits.items()], ignore_index=True).to_parquet(
        out / "goalie_fits.parquet", index=False)
    (out / "usage.json").write_text(json.dumps({"toi": {str(k): v for k, v in gv.usage.toi.items()},
                                                "shots": {str(k): v for k, v in gv.usage.shots.items()}}))
    last = max(gv.usage.toi)
    sk = gv.latest.copy()
    sk["goals_per_game_top_slot"] = sk.value_60 * sk.group.map(lambda g: gv.usage.toi[last][g][0] / 60)
    gk = gv.latest_goalies.assign(side="goalie", group=GOALIE)
    gk["goals_per_game_top_slot"] = gk.value_shot * gv.usage.shots[last]
    tab = pd.concat([sk, gk], ignore_index=True)
    if players is not None:
        tab["name"] = tab.player_id.map(players.set_index("player_id").name)
    tab.sort_values(["side", "goals_per_game_top_slot"], ascending=[True, False]).to_csv(
        out / "values_latest.csv", index=False, float_format="%.4f")
