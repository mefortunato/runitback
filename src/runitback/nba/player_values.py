"""NBA player values: possession-level ridge adjusted plus-minus, in points per game over replacement.

The NBA adapter of the sport-agnostic fitter (:mod:`runitback.core.player_values`):

- **Possessions.** The play-by-play is cut into possessions (:func:`game_possessions`): a possession
  is a run of one team's actions (shots, non-technical free throws, turnovers, offensive rebounds)
  ended by a made shot (unless an and-one free throw follows), a made last free throw, a turnover,
  the other team's defensive rebound or action, or the end of the period. Its target is the
  offense's points (technical free throws excluded), and its players are the ten on court at its
  first action (``credit_lineups`` of :mod:`runitback.nba.data.players`: the five on court when the
  clock stopped, so a free-throw trip counts for the five that drew the foul).
- **The fit.** Points per possession on an offense coefficient for each of the five offensive
  players and a defense coefficient for each of the five defenders, plus controls (home offense,
  how the possession started: period start / after a make / after a defensive rebound / after a
  turnover / other, and season), with ridge shrinkage toward 0 and recent seasons weighted more.
- **Groups and replacement level.** Players are grouped G / F / C by their ESPN position (the
  choice recorded for work package N2; rotation slots are used for usage instead, below). Players
  under ``MIN_MINUTES`` minutes on court in the window (counted as possessions, ``POSS_PER_MIN`` per
  minute) share a replacement-level coefficient for their side and group; a player's value is his
  coefficient over it (for defense with the sign flipped, so higher is better).
- **Points per game.** A player's value per possession times the possessions on court of his
  rotation slot (the k-th player of a team by minutes; the league's mean in the previous season).
- **Leak-free and weekly.** Each week (Monday) is fit on the possessions before it; every game uses
  its week's fit, so it never sees itself or later games.

``game_table`` sums each team-game's projected rotation into the wager model's ``_pv_*`` columns
(layout :data:`runitback.wager.nba.lineup.LINEUP`). The projection is N3's projected lineups
(:func:`projected_rotation` of ``projected_lineups.parquet``; ``runitback nba player-values
--lineups``), or by default the N2 stand-in known at tip-off: **the team's previous game's rotation**
(players who played, ranked by minutes).
"""
from __future__ import annotations

import datetime
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..core.player_values import APMConfig, Events, fit_apm, fit_weekly
from ..wager.nba.lineup import LINEUP, POSITIONS
from .data.players import LINEUP_COLS, credit_lineups

log = logging.getLogger(__name__)

GROUPS = POSITIONS                  # G / F / C from the ESPN position
SIDES = (("off", 1.0), ("def", -1.0))
WINDOW = 4                          # previous seasons in each fit (plus the current one)
DECAY = 0.7                         # weight of a possession from k seasons back: DECAY ** k
POSS_PER_MIN = 100 / 48             # possessions per side per minute on court (about the league's pace)
MIN_MINUTES = 250                   # minutes on court (in the window) for a player's own coefficient
MIN_POSS = int(round(MIN_MINUTES * POSS_PER_MIN))
LAMBDA = 5000.0                     # ridge penalty (tuned on the next season's possessions)
CTRL_SCALE = 30.0                   # controls are scaled up, i.e. barely penalized
POINTS_CLIP = 4.0
SLOTS = 13                          # rotation slots with a usage (the 13th man and beyond: 0)
DEFAULT_GROUP = "F"                 # players with no ESPN position

# how a possession started (the previous one's end)
START_PERIOD, START_MADE, START_DREB, START_TO, START_OTHER = range(5)
N_START = 5
FT_OF = re.compile(r"(\d)\s*of\s*(\d)")
POSS_COLUMNS = ["game_id", "season", "game_date", "t", "period", "off_home", "start", "points",
                *[f"off_{i}" for i in range(1, 6)], *[f"def_{i}" for i in range(1, 6)]]

# action kinds
OTHER, SHOT, FT, TO, OREB, DREB = range(6)


def day_number(date) -> int:
    """Days since 1970-01-01 of a ``YYYY-MM-DD`` date (the fits' time key ``t``)."""
    return (datetime.date.fromisoformat(str(date)[:10]) - datetime.date(1970, 1, 1)).days


def week_start(t: int) -> int:
    """The Monday on or before day ``t`` (1970-01-01 was a Thursday)."""
    return t - (t + 3) % 7


# -- possessions ------------------------------------------------------------------------------------

def classify(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per play-by-play row: ``side`` (0 home / 1 away / -1 neither), ``kind`` (action kind), ``made``,
    ``last_ft`` (the last free throw of a trip that ends the possession if made), ``pts`` (points the
    row scores for ``side``; technical free throws count 0)."""
    tt = pbp.type_text.fillna("").astype(str)
    text = pbp.text.fillna("").astype(str).str.lower() if "text" in pbp else pd.Series("", index=pbp.index)
    side = np.where(pbp.team_id == pbp.home_team_id, 0, np.where(pbp.team_id == pbp.away_team_id, 1, -1))
    is_ft = tt.str.startswith("Free Throw")
    ft = tt.str.extract(FT_OF)
    k, n = pd.to_numeric(ft[0], errors="coerce"), pd.to_numeric(ft[1], errors="coerce")
    tech = is_ft & (tt.str.contains("Technical") | k.isna())
    retains = tt.str.contains("Flagrant") | tt.str.contains("Clear Path")  # the fouled team keeps the ball
    scoring = pbp.scoring_play.fillna(False).astype(bool)
    value = pd.to_numeric(pbp.score_value, errors="coerce").fillna(0)
    shot = pbp.shooting_play.fillna(False).astype(bool) & ~tt.str.contains("Foul") & ~is_ft
    to = tt.str.contains("Turnover") & (tt != "No Turnover")
    kind = np.select([is_ft & ~tech, shot, to, tt == "Offensive Rebound", tt == "Defensive Rebound"],
                     [FT, SHOT, TO, OREB, DREB], OTHER)
    made = np.where(is_ft, scoring | text.str.contains("makes"), scoring & (value > 0))
    pts = np.where(((kind == SHOT) | (kind == FT)) & made, np.where(kind == FT, 1, value.clip(0, 3)), 0)
    return pd.DataFrame({"side": side.astype(np.int8), "kind": kind.astype(np.int8), "made": made.astype(bool),
                         "last_ft": ((kind == FT) & (k == n) & ~retains).to_numpy(), "pts": pts.astype(np.int8)},
                        index=pbp.index)


def game_possessions(period, side, kind, made, last_ft, pts) -> list[tuple[int, int, int, int]]:
    """Cut one game's rows (in game order) into possessions: ``(first action row, side, start kind,
    points)`` per possession."""
    out: list[list[int]] = []
    cur = None                  # [first row, side, start, points]
    ended, pending_made = False, False
    start_next = START_PERIOD
    cur_period = None
    for i in range(len(period)):
        if period[i] != cur_period:
            cur_period, cur, ended, pending_made, start_next = period[i], None, False, False, START_PERIOD
        s, k = int(side[i]), int(kind[i])
        if s < 0 or k == OTHER:
            continue
        if k == DREB:                          # the rebounder's possession starts
            new, start = True, START_DREB
        elif cur is None:
            new, start = True, start_next
        elif s != cur[1]:
            new, start = True, (start_next if ended else START_MADE if pending_made else START_OTHER)
        elif ended:
            new, start = True, start_next
        elif pending_made and k != FT:          # a make, then the same team again: a new possession
            new, start = True, START_MADE
        else:
            new, start = False, None
        if new:
            cur = [i, s, start, 0]
            out.append(cur)
            ended, pending_made = False, False
        cur[3] += int(pts[i]) if s == cur[1] else 0
        if k == TO:
            ended, start_next = True, START_TO
        elif k == FT and last_ft[i] and made[i]:
            ended, pending_made, start_next = True, False, START_MADE
        elif k == SHOT and made[i]:
            pending_made = True
        elif k == SHOT or k == OREB:
            pending_made = False
    return [tuple(c) for c in out]


PBP_COLUMNS = ["game_id", "game_play_number", "sequence_number", "season", "season_type", "game_date",
               "period_number", "type_text", "text", "team_id", "home_team_id", "away_team_id", "scoring_play",
               "shooting_play", "score_value"]


def season_possessions(pbp: pd.DataFrame, oc: pd.DataFrame) -> pd.DataFrame:
    """Possessions of every game in both ``pbp`` (raw ESPN rows) and ``oc`` (its on-court rows, in game
    order), with the ten players at each possession's first action. Possessions without five known
    players on each side are dropped."""
    pbp = pbp.assign(game_id=pbp.game_id.astype(str))
    gpn = pd.to_numeric(pbp.game_play_number, errors="coerce")
    seq = pd.to_numeric(pbp.sequence_number, errors="coerce")
    use_gpn = gpn.notna().groupby(pbp.game_id).transform("all")
    pbp = pbp.assign(game_play_number=np.where(use_gpn, gpn, seq).astype("float64"))
    pbp = pbp.drop_duplicates(["game_id", "game_play_number"]).drop(columns=["period_number"], errors="ignore")
    oc = oc[oc.game_id.isin(set(pbp.game_id))].reset_index(drop=True)
    oc = oc.assign(game_play_number=oc.game_play_number.astype("float64"))
    credit = credit_lineups(oc)
    oc = pd.concat([oc[["game_id", "game_play_number", "period"]], credit], axis=1)
    df = oc.merge(pbp, on=["game_id", "game_play_number"], how="left", sort=False)
    cl = classify(df)
    period = df.period.to_numpy()
    side, kind, made, last_ft, pts = (cl[c].to_numpy() for c in ("side", "kind", "made", "last_ft", "pts"))
    rows, sides, starts, points = [], [], [], []
    bounds = np.flatnonzero(np.r_[True, df.game_id.to_numpy()[1:] != df.game_id.to_numpy()[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        for r, s, st, p in game_possessions(period[a:b], side[a:b], kind[a:b], made[a:b], last_ft[a:b], pts[a:b]):
            rows.append(a + r)
            sides.append(s)
            starts.append(st)
            points.append(p)
    rows = np.array(rows, dtype=np.int64)
    sides = np.array(sides, dtype=np.int8)
    sub = df.iloc[rows]
    home = sub[[f"home_{i}" for i in range(1, 6)]].to_numpy(dtype="float64", na_value=np.nan)
    away = sub[[f"away_{i}" for i in range(1, 6)]].to_numpy(dtype="float64", na_value=np.nan)
    off = np.where(sides[:, None] == 0, home, away)
    dfn = np.where(sides[:, None] == 0, away, home)
    out = pd.DataFrame({
        "game_id": sub.game_id.to_numpy(), "season": sub.season.to_numpy(),
        "game_date": sub.game_date.astype(str).str[:10].to_numpy(), "period": sub.period.to_numpy(),
        "off_home": sides == 0, "start": np.array(starts, dtype=np.int8),
        "points": np.array(points, dtype=np.int8)})
    out["season"] = out.groupby("game_id").season.transform("first")
    out["game_date"] = out.groupby("game_id").game_date.transform("first")
    out["t"] = out.game_date.map(day_number).astype(np.int32)
    for i in range(5):
        out[f"off_{i + 1}"] = off[:, i]
        out[f"def_{i + 1}"] = dfn[:, i]
    full = np.isfinite(off).all(1) & np.isfinite(dfn).all(1)
    out = out[full].astype({c: "int64" for c in POSS_COLUMNS if c[:4] in ("off_", "def_")})
    return out[POSS_COLUMNS].reset_index(drop=True)


def franchise_games(pg: pd.DataFrame) -> set[str]:
    """Games between two of the 30 franchises (ESPN's box scores also have All-Star, Rising Stars and
    exhibition games, which are no evidence of how players play)."""
    from .vocab import TEAMS

    ok = pg.assign(f=pg.team.isin(TEAMS) & pg.opponent.isin(TEAMS)).groupby("game_id").f.all()
    return set(ok.index[ok])


def franchise_only(poss: pd.DataFrame, pg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``poss`` and ``pg`` restricted to :func:`franchise_games`."""
    keep = franchise_games(pg)
    return poss[poss.game_id.isin(keep)].reset_index(drop=True), pg[pg.game_id.isin(keep)].reset_index(drop=True)


def build_possessions(seasons: list[int], raw_dir: Path, processed_dir: Path) -> pd.DataFrame:
    """``possessions.parquet`` (processed_dir) for the seasons with play-by-play in raw_dir, from
    N1's ``on_court.parquet``. Logs the points per possession and possessions per team-game."""
    from .data.download import numpy_dtypes
    from .data.players import _read

    oc = pd.read_parquet(processed_dir / "on_court.parquet",
                         columns=["game_id", "game_play_number", "period", "clock", *LINEUP_COLS])
    parts = []
    for s in seasons:
        path = raw_dir / f"pbp_{s}.parquet"
        if not path.exists():
            log.warning("season %d: %s missing; skipped", s, path.name)
            continue
        # pyarrow-backed columns (pd.NA ids) as the release files' numpy dtypes, as load_raw does
        pbp = numpy_dtypes(_read(path, PBP_COLUMNS))
        p = season_possessions(pbp, oc[oc.game_id.isin(set(pbp.game_id.astype(str)))])
        per_game = p.groupby(["game_id", "off_home"]).size().mean() if len(p) else float("nan")
        log.info("season %d: %d possessions, %.1f per team-game, %.3f points each", s, len(p), per_game,
                 p.points.mean() if len(p) else float("nan"))
        parts.append(p)
    poss = pd.concat(parts, ignore_index=True)
    pg_path = processed_dir / "player_games.parquet"
    if pg_path.exists():
        poss, _ = franchise_only(poss, pd.read_parquet(pg_path, columns=["game_id", "team", "opponent"]))
    poss.to_parquet(processed_dir / "possessions.parquet", index=False)
    return poss


# -- fits -------------------------------------------------------------------------------------------

def player_groups(players: pd.DataFrame) -> dict[int, str]:
    """{player_id: G / F / C} from the player directory; no ESPN position: ``DEFAULT_GROUP``."""
    return {int(p): g if g in GROUPS else DEFAULT_GROUP for p, g in zip(players.player_id, players.group)}


def events_from(poss: pd.DataFrame, group_of: dict[int, str]) -> Events:
    """Possessions as the generic fitter's events (sides ``off`` and ``def``; target ``points``)."""
    poss = poss.sort_values(["t", "game_id"], kind="stable").reset_index(drop=True)
    off = poss[[f"off_{i}" for i in range(1, 6)]].to_numpy()
    dfn = poss[[f"def_{i}" for i in range(1, 6)]].to_numpy()
    ids = np.unique(np.concatenate([off.ravel(), dfn.ravel()]))
    gi = {g: i for i, g in enumerate(GROUPS)}
    group = np.array([gi.get(group_of.get(int(p)), gi[DEFAULT_GROUP]) for p in ids], dtype=np.int32)
    frame = poss[["game_id", "t", "season", "off_home", "start", "points"]]
    return Events(frame, {"off": np.searchsorted(ids, off).astype(np.int32),
                          "def": np.searchsorted(ids, dfn).astype(np.int32)}, ids, group)


def _controls(df: pd.DataFrame) -> sp.csr_matrix:
    n = len(df)
    seas = (df.season - df.season.min()).astype(int).to_numpy()
    cols = np.stack([np.zeros(n, int), 1 + df.off_home.to_numpy().astype(int),
                     3 + df.start.to_numpy().astype(int), 3 + N_START + seas], 1).ravel()
    rows = np.repeat(np.arange(n), 4)
    return sp.csr_matrix((np.full(len(rows), CTRL_SCALE), (rows, cols)), shape=(n, 3 + N_START + seas.max() + 1))


def apm_config(lam: float = LAMBDA, window: int = WINDOW, decay: float = DECAY, min_poss: int = MIN_POSS
               ) -> APMConfig:
    """The NBA's adjusted plus-minus settings for the generic fitter."""
    return APMConfig(groups=GROUPS, target="points", controls=_controls, sides=SIDES, min_events=min_poss,
                     decay=decay, window=window, lam=lam, target_clip=POINTS_CLIP, iter_lim=1000)


def fit_values(events: Events, before: int, season: int, cfg: APMConfig | None = None) -> pd.DataFrame:
    """``side, player_id, group, possessions, value_poss`` from the possessions with ``t < before``."""
    tab = fit_apm(events, cfg or apm_config(), before, season)
    return tab.rename(columns={"events": "possessions", "value": "value_poss"})


def holdout_score(events: Events, test_season: int, cfg: APMConfig) -> dict:
    """Fit on everything before ``test_season`` and predict its possessions' points from the ten
    players' values (unknown players at replacement): the out-of-sample mean squared error vs a
    constant, and the correlation of the prediction with the points. For tuning ``cfg``."""
    fr = events.frame
    test = (fr.season == test_season).to_numpy()
    before = int(fr.t[test].min())
    tab = fit_apm(events, cfg, before, test_season)
    pred = np.zeros(test.sum())
    for side, _ in cfg.sides:
        v = np.zeros(len(events.ids))
        t = tab[tab.side == side]
        v[np.searchsorted(events.ids, t.player_id.to_numpy())] = t.value.to_numpy()
        pred += (1 if side == "off" else -1) * v[events.sides[side][test]].sum(1)
    y = np.clip(fr.points.to_numpy()[test], -cfg.target_clip, cfg.target_clip) if cfg.target_clip else \
        fr.points.to_numpy()[test]
    base = ((y - y.mean()) ** 2).mean()
    pred = pred - pred.mean()        # values are over replacement: the constant is the intercept's
    resid = y - y.mean() - pred
    return {"season": test_season, "mse_gain": float(base - (resid ** 2).mean()),
            "corr": float(np.corrcoef(pred, y)[0, 1])}


# -- usage: rotation slots --------------------------------------------------------------------------

def rotations(pg: pd.DataFrame) -> pd.DataFrame:
    """Each team-game's players who played, ranked by minutes: ``game_id, team, player_id, rank``."""
    d = pg[~pg.dnp & (pg.minutes > 0)].sort_values(["game_id", "team", "minutes"], ascending=[True, True, False],
                                                   kind="stable")
    d = d.assign(rank=d.groupby(["game_id", "team"]).cumcount())
    return d[["game_id", "team", "player_id", "rank"]].reset_index(drop=True)


def slot_possessions(poss: pd.DataFrame, pg: pd.DataFrame) -> dict[int, list[float]]:
    """{season: [possessions on court per side at rotation slot 0, 1, ...]}: the mean over the
    season's team-games of the k-th player by minutes's possessions on court (offense and defense
    averaged). A season's games use the previous season's slots (the first season its own)."""
    parts = []
    for side in ("off", "def"):
        cols = [f"{side}_{i}" for i in range(1, 6)]
        m = poss[["game_id", *cols]].melt(id_vars="game_id", value_name="player_id")
        parts.append(m.groupby(["game_id", "player_id"]).size().rename(side))
    cnt = pd.concat(parts, axis=1).fillna(0).reset_index()
    cnt["n"] = (cnt["off"] + cnt["def"]) / 2
    rot = rotations(pg).merge(pg[["game_id", "player_id", "season"]], on=["game_id", "player_id"])
    rot = rot.merge(cnt[["game_id", "player_id", "n"]], on=["game_id", "player_id"], how="left").fillna({"n": 0})
    rot = rot[rot.game_id.isin(set(poss.game_id))]
    by = rot[rot["rank"] < SLOTS].groupby(["season", "rank"]).n.mean()
    own = {int(s): [float(by.get((s, k), 0.0)) for k in range(SLOTS)] for s in by.index.get_level_values(0).unique()}
    seasons = sorted(own)
    return {s: own[seasons[max(i - 1, 0)]] for i, s in enumerate(seasons)}


# -- per-game lineup values (the wager model's ``_pv_*`` columns) ----------------------------------

GROUP_COLUMNS = LINEUP.groups        # G_OFF, F_OFF, C_OFF, G_DEF, F_DEF, C_DEF
OFF_COLUMNS = tuple(f"{g}_OFF" for g in GROUPS)
DEF_COLUMNS = tuple(f"{g}_DEF" for g in GROUPS)
RECENT = LINEUP.recent
PV_COLUMNS = LINEUP.columns          # _pv_0 .. _pv_13


def side_values(rows, vp: dict, group_of: dict, mult: list[float]) -> dict[str, float]:
    """A rotation (``(rank, player_id)`` rows) -> points per game over replacement per group column."""
    out = dict.fromkeys(GROUP_COLUMNS, 0.0)
    for k, pid in rows:
        if k >= len(mult):
            continue
        g = group_of.get(pid, DEFAULT_GROUP)
        out[f"{g}_OFF"] += vp.get(("off", pid), 0.0) * mult[k]
        out[f"{g}_DEF"] += vp.get(("def", pid), 0.0) * mult[k]
    return out


@dataclass
class GameValues:
    pv: pd.DataFrame          # game_id, team, _pv_0 .. _pv_13
    slots: dict               # {season: possessions per side at each rotation slot}
    fits: dict                # {before (day): values table}
    latest: pd.DataFrame      # the most recent fit


def previous_rotation(pg: pd.DataFrame, target: pd.DataFrame | None = None) -> pd.DataFrame:
    """The stand-in projected rotation: for each team-game, the team's previous game's players who
    played, ranked by minutes (known at tip-off). ``target``: (game_id, team) rows (default every
    team-game in ``pg``)."""
    rot = rotations(pg)
    order = pg[["game_id", "team", "game_date"]].drop_duplicates(["game_id", "team"]).sort_values(
        ["team", "game_date", "game_id"], kind="stable")
    order["prev"] = order.groupby("team").game_id.shift(1)
    if target is not None:
        order = target[["game_id", "team"]].merge(order[["game_id", "team", "prev"]], on=["game_id", "team"])
    pr = order.dropna(subset=["prev"]).merge(rot.rename(columns={"game_id": "prev"}), on=["prev", "team"])
    return pr[["game_id", "team", "player_id", "rank"]].reset_index(drop=True)


def projected_rotation(proj: pd.DataFrame) -> pd.DataFrame:
    """N3's projected lineups (``projected_lineups.parquet``, :mod:`runitback.nba.data.lineups`) as
    :func:`game_table`'s ``lineups``: each team-game's available players, ranked by projected
    minutes (players ruled out, ``rank == -1``, are left out)."""
    d = proj[(proj.status == "available") & (proj["rank"] >= 0)]
    d = d.assign(game_id=d.game_id.astype(str), rank=d["rank"].astype(int))
    return d[["game_id", "team", "player_id", "rank"]].reset_index(drop=True)


def game_table(poss: pd.DataFrame, pg: pd.DataFrame, players: pd.DataFrame, first_season: int = 2016,
               jobs: int = 4, lineups: pd.DataFrame | None = None, cfg: APMConfig | None = None) -> GameValues:
    """Leak-free weekly values for every week from ``first_season`` (each fit on the possessions
    before its Monday) and each team-game's ``_pv_*`` columns from its projected rotation, in points
    per game over replacement:

        _pv_0 .. _pv_5   G, F, C offense; G, F, C defense (each group's sum)
        _pv_6  offense change vs the previous game's rotation   _pv_7 defense change
        _pv_8 .. _pv_13  each group's change vs the mean of the team's last ``RECENT`` rotations

    ``lineups`` (``game_id, team, player_id, rank``): the projected rotations; default
    :func:`previous_rotation`. ``team`` is ``player_games``' team key."""
    cfg = cfg or apm_config()
    group_of = player_groups(players)
    events = events_from(poss, group_of)
    slots = slot_possessions(poss, pg)
    games = pg[pg.season >= first_season][["game_id", "season", "game_date", "team"]].drop_duplicates(
        ["game_id", "team"])
    games = games.assign(t=games.game_date.map(day_number))
    games["before"] = games.t.map(week_start)
    weeks = sorted(set(games.before))
    first_t = int(events.frame.t.min())
    weeks = [w for w in weeks if w > first_t]
    season_at = games.groupby("before").season.min().to_dict()
    log.info("%d possessions; fitting %d weeks", len(events.frame), len(weeks))
    fits = fit_weekly(events, cfg, weeks, lambda w: season_at[w], jobs)
    fits = {w: f.rename(columns={"events": "possessions", "value": "value_poss"}) for w, f in zip(weeks, fits)}
    vals = {w: dict(zip(zip(f.side, f.player_id), f.value_poss)) for w, f in fits.items()}

    proj = previous_rotation(pg, games) if lineups is None else lineups
    proj_rows = {k: list(zip(d["rank"].astype(int), d.player_id)) for k, d in proj.groupby(["game_id", "team"])}
    actual = {k: list(zip(d["rank"].astype(int), d.player_id)) for k, d in rotations(pg).groupby(["game_id", "team"])}
    order = pg[["game_id", "team", "game_date"]].drop_duplicates(["game_id", "team"]).sort_values(
        ["game_date", "game_id"], kind="stable")
    played_by = {team: d.game_id.to_numpy() for team, d in order.groupby("team")}
    pos_in = {(g, team): i for team, ids in played_by.items() for i, g in enumerate(ids)}
    seasons = sorted(slots)
    rows = []
    for g in games.itertuples():
        vp = vals.get(g.before)
        pr = proj_rows.get((g.game_id, g.team))
        if vp is None or pr is None:
            continue
        mult = slots.get(g.season) or slots[seasons[-1]]
        cur = side_values(pr, vp, group_of, mult)
        i = pos_in[(g.game_id, g.team)]
        before = played_by[g.team][:i]
        prev = before[-1] if len(before) else None
        pv_prev = side_values(actual[(prev, g.team)], vp, group_of, mult) if (prev, g.team) in actual else None
        row = {"game_id": g.game_id, "team": g.team} | {f"_pv_{j}": cur[k] for j, k in enumerate(GROUP_COLUMNS)}
        n = len(GROUP_COLUMNS)
        row[f"_pv_{n}"] = sum(cur[k] - pv_prev[k] for k in OFF_COLUMNS) if pv_prev else 0.0
        row[f"_pv_{n + 1}"] = sum(cur[k] - pv_prev[k] for k in DEF_COLUMNS) if pv_prev else 0.0
        recent = [side_values(actual[(h, g.team)], vp, group_of, mult) for h in before[-RECENT:]
                  if (h, g.team) in actual]
        for j, k in enumerate(GROUP_COLUMNS):
            row[f"_pv_{n + 2 + j}"] = cur[k] - float(np.mean([r[k] for r in recent])) if recent else 0.0
        rows.append(row)
    pv = pd.DataFrame(rows, columns=["game_id", "team", *PV_COLUMNS])
    pv[PV_COLUMNS] = pv[PV_COLUMNS].astype(np.float32)
    return GameValues(pv, slots, fits, fits[weeks[-1]])


def to_wager_teams(pv: pd.DataFrame, raw_dir: Path, seasons: list[int]) -> pd.DataFrame:
    """Re-key ``team`` from ``player_games``' abbreviations to the wager model's (the franchise's
    current abbreviation from its ESPN id, ``runitback.wager.nba.data.download.team_key``)."""
    from ..wager.nba.data.download import team_key
    from .data.players import _read
    from .vocab import canon_team

    parts = []
    for s in seasons:
        path = raw_dir / f"player_box_{s}.parquet"
        if path.exists():
            parts.append(_read(path, ["game_id", "team_id", "team_abbreviation"]).drop_duplicates())
    box = pd.concat(parts, ignore_index=True).dropna()
    keys = pd.DataFrame({"game_id": box.game_id.astype(str),
                         "team": box.team_abbreviation.map(lambda x: canon_team(str(x))),
                         "wager_team": [team_key(i, a) for i, a in zip(box.team_id, box.team_abbreviation)]})
    out = pv.merge(keys.drop_duplicates(["game_id", "team"]), on=["game_id", "team"], how="left")
    out["team"] = out.pop("wager_team").fillna(out.team)
    return out


def write_game_table(gv: GameValues, out: Path, players: pd.DataFrame | None = None) -> None:
    """``<out>/pv.parquet`` (the wager model's lineup table), ``slots.json`` (possessions per side at
    each rotation slot, by season) and ``values_latest.csv`` (the latest fit: every player's value
    per possession and as a starter, offense and defense)."""
    out.mkdir(parents=True, exist_ok=True)
    gv.pv.to_parquet(out / "pv.parquet", index=False)
    (out / "slots.json").write_text(json.dumps({str(k): v for k, v in gv.slots.items()}))
    last = gv.latest.copy()
    if players is not None:
        last["name"] = last.player_id.map(players.set_index("player_id").name)
    starter = gv.slots[max(gv.slots)][0]
    last["pts_as_starter"] = last.value_poss * starter
    last.sort_values(["side", "pts_as_starter"], ascending=[True, False]).to_csv(
        out / "values_latest.csv", index=False, float_format="%.4f")


# -- the swap test (experiments/nba, scripts/nba_lineup_swap.py) --------------------------------------

SWAPS = ("best", "worst", "replacement", "out")
REPLACEMENT_ID = -1                 # a player with no value of his own: replacement level (0)


def swap_rotation(rows: list[tuple[int, int]], group_of: dict, group: str, kind: str,
                  ref: int | None = None) -> list[tuple[int, int]] | None:
    """A projected rotation (``(rank, player_id)`` rows) with its top-ranked ``group`` player (the
    group's starter) changed: to player ``ref`` in the same slot (``best`` / ``worst``: a regular from
    another team), to a replacement-level player (``replacement``), or ruled out (``out``: everyone
    below him moves up a slot). None when the team has no player in ``group`` or ``ref`` is already
    in its rotation."""
    rows = sorted(rows)
    starter = next(((k, p) for k, p in rows if group_of.get(p, DEFAULT_GROUP) == group), None)
    if starter is None:
        return None
    if kind == "out":
        rest = [p for k, p in rows if (k, p) != starter]
        return list(enumerate(rest))
    if kind == "replacement":
        new = REPLACEMENT_ID
    elif kind in ("best", "worst"):
        if ref is None or any(p == ref for _, p in rows):
            return None
        new = ref
    else:
        raise ValueError(f"unknown swap {kind!r}; choose from {SWAPS}")
    return [(k, new if (k, p) == starter else p) for k, p in rows]


def swap_refs(fit: pd.DataFrame, min_poss: int) -> dict[tuple[str, str], int]:
    """``{(group, "best" | "worst"): player_id}``: each group's best and worst regular by offense plus
    defense value per possession in ``fit`` (:func:`fit_values`), among players with their own
    coefficient on both sides and at least ``min_poss`` possessions on court in the fit's window."""
    tab = fit.pivot_table(index=["player_id", "group"], columns="side", values="value_poss", aggfunc="sum")
    poss = fit.groupby("player_id").possessions.min()
    tab = tab.dropna()
    tab = tab[poss.reindex(tab.index.get_level_values(0)).to_numpy() >= min_poss]
    value = tab.sum(axis=1)
    out = {}
    for g in GROUPS:
        v = value[value.index.get_level_values(1) == g]
        if len(v):
            out[(g, "best")], out[(g, "worst")] = int(v.idxmax()[0]), int(v.idxmin()[0])
    return out
