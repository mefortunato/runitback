"""College basketball player values and the per-game lineup table (``pv.parquet``).

Following C1 (``experiments/cbb/lineup_probe.md``) and the decision recorded in
``experiments/cbb/README.md`` (no CollegeBasketballData, for its monthly call limit):

- **Women's (``wcbb``): box-score values** throughout (:mod:`.box_values`).
- **Men's (``cbb``): box-score values, plus adjusted plus-minus where lineups exist** (ESPN's
  substitutions, from mid-February 2025; :mod:`.espn_lineups`). The adjusted plus-minus is the
  NBA's possession-level fit (:mod:`runitback.nba.player_values`, on the generic
  :mod:`runitback.core.player_values`), **shrunk toward the box-score values**: each possession's
  target is its points minus what the ten players' box values predict, so the ridge pulls every
  player toward his box value rather than toward replacement. A player's value is his box value plus
  his own coefficient over his group's replacement coefficient (players under ``MIN_MINUTES`` on the
  floor keep their box value). Weeks with no possessions before them (every week before
  February 2025) use the box values alone, and the first weeks after stay close to them.

Values are in points per possession over replacement (offense: points the team scores; defense:
points it saves; higher is better on both). A player's **points per game** are his value times the
league's possessions per game (the previous season's mean) times his share of 40 minutes, with his
projected minutes (:mod:`.lineups`), for the game being predicted and for the team's earlier games
behind the change columns (so those measure who is in or out, not the projection's smoothing).

Every week (Monday) from ``first_season`` is fit on the games (and possessions) before it, and each
game uses its week's fit, so no game ever sees itself or later games.

``pv.parquet``: one row per team-game, ``game_id, season, team_id, team`` and the wager model's
``_pv_*`` columns in the layout of :data:`runitback.wager.cbb.lineup.LINEUP`:

    _pv_0 .. _pv_5   G, F, C offense; G, F, C defense: the projected rotation's points per game
    _pv_6, _pv_7     offense / defense change vs the team's previous game's rotation
    _pv_8 .. _pv_13  each group's change vs the mean of the team's last ``RECENT`` rotations
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..core.player_values import APMConfig, Events, fit_apm
from ..wager.cbb.lineup import LINEUP, POSITIONS
from .box_values import BoxConfig, BoxModel, week_start
from .data import DEFAULT_GROUP, REGULATION_MINUTES

log = logging.getLogger(__name__)

GROUPS = POSITIONS
GROUP_COLUMNS = LINEUP.groups          # G_OFF, F_OFF, C_OFF, G_DEF, F_DEF, C_DEF
OFF_COLUMNS = tuple(f"{g}_OFF" for g in GROUPS)
DEF_COLUMNS = tuple(f"{g}_DEF" for g in GROUPS)
RECENT = LINEUP.recent
PV_COLUMNS = LINEUP.columns            # _pv_0 .. _pv_13
FIRST_SEASON = {"cbb": 2012, "wcbb": 2017}   # first season with a full box-value window behind it

# men's adjusted plus-minus (on top of the box prior)
APM_WINDOW = 2                  # previous seasons in each fit (plus the current one)
APM_DECAY = 0.7
POSS_PER_MIN = 70 / 40          # possessions per side per minute on the floor (college pace)
MIN_MINUTES = 500               # minutes on the floor (in the window) for a player's own coefficient
APM_LAMBDA = 3000.0             # ridge penalty (tuned on later possessions: experiments/cbb/README.md)
CTRL_SCALE = 30.0
POINTS_CLIP = 4.0
N_START = 5


# -- men's adjusted plus-minus around the box prior ------------------------------------------------

def events_from(poss: pd.DataFrame, group_of: dict[int, str]) -> Events:
    """Men's possessions (:mod:`.espn_lineups`) as the generic fitter's events (sides ``off`` / ``def``; target ``points``)."""
    poss = poss.sort_values(["t", "game_id"], kind="stable").reset_index(drop=True)
    off = poss[[f"off_{i}" for i in range(1, 6)]].to_numpy(dtype=np.int64)
    dfn = poss[[f"def_{i}" for i in range(1, 6)]].to_numpy(dtype=np.int64)
    ids = np.unique(np.concatenate([off.ravel(), dfn.ravel()]))
    gi = {g: i for i, g in enumerate(GROUPS)}
    group = np.array([gi.get(group_of.get(int(p), DEFAULT_GROUP), gi[DEFAULT_GROUP]) for p in ids], dtype=np.int32)
    frame = poss[["game_id", "t", "season", "off_home", "start", "points"]].copy()
    return Events(frame, {"off": np.searchsorted(ids, off).astype(np.int32),
                          "def": np.searchsorted(ids, dfn).astype(np.int32)}, ids, group)


def _controls(df: pd.DataFrame) -> sp.csr_matrix:
    n = len(df)
    seas = (df.season - df.season.min()).astype(int).to_numpy()
    cols = np.stack([np.zeros(n, int), 1 + df.off_home.to_numpy().astype(int),
                     3 + df.start.to_numpy().astype(int), 3 + N_START + seas], 1).ravel()
    rows = np.repeat(np.arange(n), 4)
    return sp.csr_matrix((np.full(len(rows), CTRL_SCALE), (rows, cols)), shape=(n, 3 + N_START + seas.max() + 1))


def apm_config(lam: float = APM_LAMBDA, window: int = APM_WINDOW, decay: float = APM_DECAY,
               min_minutes: float = MIN_MINUTES) -> APMConfig:
    """The men's adjusted plus-minus settings for the generic fitter (target: points over the box
    prior, ``resid``)."""
    return APMConfig(groups=GROUPS, target="resid", controls=_controls, min_events=round(min_minutes * POSS_PER_MIN),
                     decay=decay, window=window, lam=lam, target_clip=POINTS_CLIP, iter_lim=1000)


def _value_arrays(events: Events, prior: pd.DataFrame) -> dict[str, np.ndarray]:
    """{side: per-code prior value per possession} from a values table (missing players: 0)."""
    out = {}
    for side in ("off", "def"):
        v = np.zeros(len(events.ids))
        t = prior[prior.side == side]
        idx = np.searchsorted(events.ids, t.player_id.to_numpy())
        ok = (idx < len(events.ids)) & (events.ids[np.minimum(idx, len(events.ids) - 1)] == t.player_id.to_numpy())
        v[idx[ok]] = t.value.to_numpy()[ok]
        out[side] = v
    return out


def fit_apm_prior(events: Events, prior: pd.DataFrame, before: int, season: int,
                  cfg: APMConfig | None = None) -> pd.DataFrame:
    """Adjusted plus-minus shrunk toward ``prior`` (a values table: ``side, player_id, group, minutes,
    value`` per possession): the fit's possessions before ``before`` with each one's points minus its
    ten players' prior values, then every prior player's value plus his own coefficient over
    replacement. Returns ``side, player_id, group, minutes, value, apm_possessions``."""
    cfg = cfg or apm_config()
    pv = _value_arrays(events, prior)
    fr = events.frame
    expected = pv["off"][events.sides["off"]].sum(1) - pv["def"][events.sides["def"]].sum(1)
    ev = Events(fr.assign(resid=fr.points.to_numpy() - expected), events.sides, events.ids, events.group)
    tab = fit_apm(ev, cfg, before, season)
    dev = {(s, p): v for s, p, v in zip(tab.side, tab.player_id, tab.value)}
    n = {(s, p): e for s, p, e in zip(tab.side, tab.player_id, tab.events)}
    out = prior.copy()
    keys = list(zip(out.side, out.player_id))
    out["value"] = out.value.to_numpy() + np.array([dev.get(k, 0.0) for k in keys])
    out["apm_possessions"] = np.array([n.get(k, 0) for k in keys], dtype=np.int64)
    # players with their own coefficient but no box history (rare): coefficient alone
    extra = tab[[k not in set(keys) for k in zip(tab.side, tab.player_id)]]
    if len(extra):
        out = pd.concat([out, pd.DataFrame({"side": extra.side, "player_id": extra.player_id, "group": extra.group,
                                            "minutes": 0.0, "value": extra.value,
                                            "apm_possessions": extra.events})], ignore_index=True)
    return out


def predict_points(events: Events, values: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    """Each of ``rows``' possessions: its offense's values minus its defense's (points over the
    average, before the intercept)."""
    v = _value_arrays(events, values)
    return v["off"][events.sides["off"][rows]].sum(1) - v["def"][events.sides["def"][rows]].sum(1)


def holdout_score(events: Events, prior: pd.DataFrame, before: int, season: int, test: np.ndarray,
                  cfg: APMConfig | None = None) -> dict:
    """Fit at ``before`` and predict the ``test`` possessions' points (clipped as the fit's target)
    from the box prior alone and from the adjusted plus-minus around it: the squared-error reduction
    per possession vs a constant (x 1e-4) and the correlation, for each."""
    cfg = cfg or apm_config()
    post = fit_apm_prior(events, prior, before, season, cfg)
    rows = np.flatnonzero(test)
    y = np.clip(events.frame.points.to_numpy()[rows].astype(float), -cfg.target_clip, cfg.target_clip)
    yc = y - y.mean()
    out = {}
    for name, tab in (("box", prior), ("apm", post)):
        pred = predict_points(events, tab, rows)
        pred = pred - pred.mean()
        out[f"{name}_gain"] = float(1e4 * ((yc ** 2).mean() - ((yc - pred) ** 2).mean()))
        out[f"{name}_corr"] = float(np.corrcoef(pred, y)[0, 1])
    return out


# -- weekly values -----------------------------------------------------------------------------------

def week_cutoffs(pg: pd.DataFrame, first_season: int) -> pd.DataFrame:
    """Every team-game from ``first_season``: ``game_id, season, game_date, team_id, team, before``
    (the Monday its values are fit before)."""
    g = pg[pg.season >= first_season][["game_id", "season", "game_date", "team_id", "team"]].drop_duplicates(
        ["game_id", "team_id"])
    t = pd.to_datetime(g.game_date).sub(pd.Timestamp("1970-01-01")).dt.days.astype(np.int64)
    return g.assign(before=week_start(t).to_numpy()).reset_index(drop=True)


def weekly_values(box: BoxModel, games: pd.DataFrame, poss: pd.DataFrame | None = None,
                  players: pd.DataFrame | None = None, cfg: APMConfig | None = None,
                  apm_from: int | None = None) -> dict[int, pd.DataFrame]:
    """{Monday: values table} for every week in ``games``: the box values, and with ``poss`` (men's
    possessions), from season ``apm_from`` on, the adjusted plus-minus around them wherever
    possessions exist before the Monday."""
    season_at = games.groupby("before").season.min().to_dict()
    events = None
    if poss is not None and len(poss):
        group_of = dict(zip(players.player_id, players.group)) if players is not None else {}
        events = events_from(poss, group_of)
        first_t = int(events.frame.t.min())
    out = {}
    for b in sorted(season_at):
        s = season_at[b]
        v = box.values(b, s)
        if events is not None and (apm_from is None or s >= apm_from) and b > first_t:
            v = fit_apm_prior(events, v, b, s, cfg)
        out[b] = v
    return out


# -- per-game lineup values (the wager model's ``_pv_*`` columns) ----------------------------------

def possessions_per_game(tg: pd.DataFrame) -> dict[int, float]:
    """{season: the previous season's mean possessions per team-game} (the first season: its own)."""
    ok = tg[(tg.poss >= 40) & (tg.minutes >= 190)]
    by = ok.groupby("season").poss.mean()
    seasons = sorted(by.index)
    return {int(s): float(by[seasons[max(i - 1, 0)]]) for i, s in enumerate(seasons)}


def side_values(rows, vals: dict, group_of: dict, ppg: float) -> dict[str, float]:
    """A rotation (``(player_id, minutes)`` rows) -> points per game over replacement per group column."""
    out = dict.fromkeys(GROUP_COLUMNS, 0.0)
    for pid, minutes in rows:
        g = group_of.get(pid, DEFAULT_GROUP)
        w = ppg * minutes / REGULATION_MINUTES
        out[f"{g}_OFF"] += vals.get(("off", pid), 0.0) * w
        out[f"{g}_DEF"] += vals.get(("def", pid), 0.0) * w
    return out


@dataclass
class GameValues:
    pv: pd.DataFrame          # game_id, season, team_id, team, _pv_0 .. _pv_13
    ppg: dict                 # {season: possessions per team-game}
    fits: dict                # {Monday (day number): values table}

    @property
    def latest(self) -> pd.DataFrame:
        return self.fits[max(self.fits)]


def game_table(pg: pd.DataFrame, tg: pd.DataFrame, proj: pd.DataFrame, fits: dict[int, pd.DataFrame],
               games: pd.DataFrame) -> GameValues:
    """Each team-game's ``_pv_*`` columns (see the module docstring) from its week's values ``fits``,
    its projected rotation ``proj`` (:func:`runitback.college.lineups.projected_lineups`) and the
    team's earlier games' projected rotations (as played where a game has none)."""
    group_of = {int(p): g for p, g in zip(pg.player_id, pg.group)}
    ppg = possessions_per_game(tg)
    seasons = sorted(ppg)
    vals = {b: dict(zip(zip(f.side, f.player_id), f.value)) for b, f in fits.items()}
    proj_rows = {k: list(zip(d.player_id, d.proj_minutes)) for k, d in
                 proj[["game_id", "team_id", "player_id", "proj_minutes"]].groupby(["game_id", "team_id"])}
    # the team's earlier games as projected (the same minutes model as the current game's, so the
    # change columns measure who is in or out rather than the projection's smoothing); a game
    # without a projection: as played
    played = pg[~pg.dnp]
    actual = {k: list(zip(d.player_id, d.minutes)) for k, d in
              played[["game_id", "team_id", "player_id", "minutes"]].groupby(["game_id", "team_id"])}
    actual |= proj_rows
    order = pg[["game_id", "team_id", "game_date"]].drop_duplicates(["game_id", "team_id"]).sort_values(
        ["game_date", "game_id"], kind="stable")
    played_by = {team: d.game_id.to_numpy() for team, d in order.groupby("team_id")}
    pos_in = {(g, team): i for team, ids in played_by.items() for i, g in enumerate(ids)}
    n = len(GROUP_COLUMNS)
    rows = []
    for g in games.itertuples():
        vp = vals.get(g.before)
        pr = proj_rows.get((g.game_id, g.team_id))
        if vp is None or pr is None:
            continue
        rate = ppg.get(g.season) or ppg[seasons[-1]]
        cur = side_values(pr, vp, group_of, rate)
        i = pos_in[(g.game_id, g.team_id)]
        before = played_by[g.team_id][:i]
        prev = before[-1] if len(before) else None
        pv_prev = side_values(actual[(prev, g.team_id)], vp, group_of, rate) if (prev, g.team_id) in actual else None
        row = {"game_id": g.game_id, "season": g.season, "team_id": g.team_id, "team": g.team}
        row |= {f"_pv_{j}": cur[k] for j, k in enumerate(GROUP_COLUMNS)}
        row[f"_pv_{n}"] = sum(cur[k] - pv_prev[k] for k in OFF_COLUMNS) if pv_prev else 0.0
        row[f"_pv_{n + 1}"] = sum(cur[k] - pv_prev[k] for k in DEF_COLUMNS) if pv_prev else 0.0
        recent = [side_values(actual[(h, g.team_id)], vp, group_of, rate) for h in before[-RECENT:]
                  if (h, g.team_id) in actual]
        for j, k in enumerate(GROUP_COLUMNS):
            row[f"_pv_{n + 2 + j}"] = cur[k] - float(np.mean([r[k] for r in recent])) if recent else 0.0
        rows.append(row)
    pv = pd.DataFrame(rows, columns=["game_id", "season", "team_id", "team", *PV_COLUMNS])
    pv[PV_COLUMNS] = pv[PV_COLUMNS].astype(np.float32)
    return GameValues(pv, ppg, fits)


def build(pg: pd.DataFrame, tg: pd.DataFrame, proj: pd.DataFrame, first_season: int,
          poss: pd.DataFrame | None = None, players: pd.DataFrame | None = None,
          box_cfg: BoxConfig | None = None, apm_cfg: APMConfig | None = None, apm_from: int | None = None
          ) -> GameValues:
    """Weekly values and ``pv`` for every team-game from ``first_season`` (``poss``: men's
    possessions for the adjusted plus-minus)."""
    box = BoxModel.build(pg, tg, box_cfg)
    games = week_cutoffs(pg, first_season)
    log.info("%d team-games, %d weeks", len(games), games.before.nunique())
    fits = weekly_values(box, games, poss, players, apm_cfg, apm_from)
    return game_table(pg, tg, proj, fits, games)


def write_game_table(gv: GameValues, out: Path, players: pd.DataFrame | None = None) -> None:
    """``<out>/pv.parquet``, ``ppg.json`` (possessions per team-game by season) and
    ``values_latest.csv`` (the latest fit: every player's value per possession and per 40 minutes)."""
    out.mkdir(parents=True, exist_ok=True)
    gv.pv.to_parquet(out / "pv.parquet", index=False)
    (out / "ppg.json").write_text(json.dumps({str(k): v for k, v in gv.ppg.items()}))
    last = gv.latest.copy()
    if players is not None:
        d = players.set_index("player_id")
        last["name"] = last.player_id.map(d.name)
        last["team"] = last.player_id.map(d.team)
    last["pts_per_40"] = last.value * gv.ppg[max(gv.ppg)]
    last.sort_values(["side", "pts_per_40"], ascending=[True, False]).to_csv(
        out / "values_latest.csv", index=False, float_format="%.4f")
