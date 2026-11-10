"""College football quarterback values: EPA per quarterback play, adjusted and shrunk, in points per
game over replacement, and each team-game's starting-quarterback terms for the wager model.

The college adapter of the sport-agnostic fitter (:mod:`runitback.core.player_values`). College
play-by-play has no on-field personnel, so only the quarterback is valued:

- **Plays.** Every dropback (pass attempt or sack) is the passer's and, by default, every run by a
  quarterback is the rusher's (:func:`runitback.cfb.data.players.qb_plays`). Scrambles are logged as
  runs, so leaving runs out would credit a scrambling quarterback only with his failed dropbacks.
- **The fit.** Each play's EPA on one coefficient for its quarterback, plus controls: down and
  distance, field position, home / away / neutral, score difference, dropback vs run, whether the
  defense is an FBS team (FCS defenses give up far more, and every FCS game is in the fit, so FCS
  quarterbacks and FBS transfers from FCS are on the FBS scale) and season; and the **defense's
  team-season** (ridge columns with :data:`DEF_SCALE` times a quarterback's prior width): college
  schedules differ far more in strength than the NFL's. Ridge shrinkage toward 0, recent seasons
  weighted more (``experiments/cfb/README.md`` has the tuning).
- **Replacement level.** Quarterbacks with fewer than :data:`MIN_PLAYS` plays in the window share
  a replacement coefficient; a quarterback's value is his coefficient over it. Ids are ESPN's
  athlete ids, which carry over when a player transfers, so a transfer keeps his value.
- **Points per game.** Value per play times the plays a starter makes per game (the league's mean
  in the previous season).
- **Leak-free and weekly.** Each week (Monday) is fit on the plays before it; every game uses its
  week's fit.

``game_table`` gives each team-game the wager model's ``_pv_*`` columns
(:data:`runitback.wager.cfb.lineup.LINEUP`):

    _pv_0  the projected starter's points per game over replacement
    _pv_1  change vs the previous game's starter
    _pv_2  change vs the mean of the team's last ``RECENT`` starters (the starting-QB-change term)

There is no public pre-game depth chart or injury history for college football, so the projected
starter is a stand-in known at kickoff: **the team's previous game's starter** (:data:`RULES`). It is
right in 81% of 2014-2025 team-games (41% of season openers); the previous game's dropback leader is
right slightly more often (83%) but its values track the margin and the line slightly less
(``experiments/cfb/README.md``). With the default ``_pv_1`` is 0; ``_pv_2`` moves when the starter
changed in the last few games. Pass ``starters`` to use another projection (e.g. a posted depth chart
for upcoming games).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..core.player_values import APMConfig, Events, fit_apm, fit_weekly
from ..wager.cfb.data.download import FCS
from ..wager.cfb.lineup import LINEUP

log = logging.getLogger(__name__)

GROUPS = ("QB",)
WINDOW = 4                  # previous seasons in each fit (plus the current one)
DECAY = 0.5                 # weight of a play from k seasons back: DECAY ** k (tuned; 0.7 was worse)
MIN_PLAYS = 50              # quarterback plays (in the window) for his own coefficient (tuned)
LAMBDA = 300.0              # ridge penalty on the quarterback coefficients (tuned week by week)
DEF_SCALE = 1.0             # defense team-season columns' prior width relative to a quarterback's (0: none;
                            # a tie play by play, better at game level)
CTRL_SCALE = 30.0           # situation controls are scaled up, i.e. barely penalized
EPA_CLIP = 6.0
RECENT = LINEUP.recent
PV_COLUMNS = LINEUP.columns  # _pv_0 .. _pv_2
RULES = ("prev_starter", "prev_leader")

SCORE_BINS = [-21, -14, -7, 0, 1, 8, 15, 22]


def week_start(t: int) -> int:
    """The Monday on or before day ``t`` (1970-01-01 was a Thursday)."""
    return t - (t + 3) % 7


# -- fits -------------------------------------------------------------------------------------------

def events_from(plays: pd.DataFrame) -> Events:
    """Quarterback plays as the generic fitter's events: one side, ``off``, of width 1 (the
    quarterback); the defense team-season is a control (``dkey``)."""
    plays = plays.sort_values(["t", "game_id", "order"], kind="stable").reset_index(drop=True)
    ids = np.unique(plays.qb.to_numpy())
    frame = plays[["game_id", "t", "season", "home", "kind", "down", "distance", "ytg", "score_diff", "def_fbs",
                   "epa"]].copy()
    frame["dkey"] = plays.season.to_numpy().astype(np.int64) * 1_000_000 + plays["def"].to_numpy()
    codes = np.searchsorted(ids, plays.qb.to_numpy()).astype(np.int32)[:, None]
    return Events(frame, {"off": codes}, ids, np.zeros(len(ids), dtype=np.int32))


def controls(def_scale: float = DEF_SCALE):
    """The control columns' builder: situation one-hots scaled by :data:`CTRL_SCALE` and, if
    ``def_scale`` is not 0, one column per defense team-season scaled by ``def_scale``."""
    def build(df: pd.DataFrame) -> sp.csr_matrix:
        n = len(df)
        d = df.down.to_numpy().astype(int).clip(1, 4) - 1
        dist = np.digitize(df.distance.to_numpy(), [4, 7, 11])
        yl = np.clip(df.ytg.to_numpy() // 10, 0, 9).astype(int)
        home = df.home.to_numpy().astype(int) + 1
        score = np.digitize(df.score_diff.to_numpy(), SCORE_BINS)
        kind = df.kind.to_numpy().astype(int)
        level = df.def_fbs.to_numpy().astype(int)
        seas = (df.season - df.season.min()).astype(int).to_numpy()
        offs = np.cumsum([0, 1, 16, 10, 3, len(SCORE_BINS) + 1, 2, 2])
        cols = np.stack([np.zeros(n, int), offs[1] + d * 4 + dist, offs[2] + yl, offs[3] + home,
                         offs[4] + score, offs[5] + kind, offs[6] + level, offs[7] + seas], 1)
        width = offs[7] + seas.max() + 1
        rows = np.repeat(np.arange(n), cols.shape[1])
        X = sp.csr_matrix((np.full(rows.size, CTRL_SCALE), (rows, cols.ravel())), shape=(n, width))
        if not def_scale:
            return X
        dk, _ = pd.factorize(df.dkey.to_numpy())
        D = sp.csr_matrix((np.full(n, def_scale), (np.arange(n), dk)), shape=(n, dk.max() + 1))
        return sp.hstack([X, D]).tocsr()
    return build


def apm_config(lam: float = LAMBDA, min_plays: int = MIN_PLAYS, window: int = WINDOW, decay: float = DECAY,
               def_scale: float = DEF_SCALE) -> APMConfig:
    """College football's quarterback settings for the generic fitter."""
    return APMConfig(groups=GROUPS, target="epa", controls=controls(def_scale), sides=(("off", 1.0),),
                     min_events=min_plays, decay=decay, window=window, lam=lam, target_clip=EPA_CLIP,
                     iter_lim=1000)


def fit_values(events: Events, before: int, season: int, cfg: APMConfig | None = None) -> pd.DataFrame:
    """``player_id, plays, value_play`` from the plays with ``t < before``."""
    tab = fit_apm(events, cfg or apm_config(), before, season)
    return tab.rename(columns={"events": "plays", "value": "value_play"})[["player_id", "plays", "value_play"]]


def holdout_score(events: Events, test_season: int, cfg: APMConfig) -> dict:
    """Fit on everything before ``test_season`` and predict its plays' EPA from the quarterback's
    value alone (unknown quarterbacks at replacement): the out-of-sample squared-error reduction per
    play vs a constant, and the prediction's correlation with the EPA. For tuning ``cfg``."""
    fr = events.frame
    test = (fr.season == test_season).to_numpy()
    tab = fit_apm(events, cfg, int(fr.t[test].min()), test_season)
    v = np.zeros(len(events.ids))
    v[np.searchsorted(events.ids, tab.player_id.to_numpy())] = tab.value.to_numpy()
    pred = v[events.sides["off"][test, 0]]
    y = np.clip(fr.epa.to_numpy()[test], -cfg.target_clip, cfg.target_clip)
    pred = pred - pred.mean()
    base = ((y - y.mean()) ** 2).mean()
    return {"season": test_season, "mse_gain": float(base - ((y - y.mean() - pred) ** 2).mean()),
            "corr": float(np.corrcoef(pred, y)[0, 1])}


def holdout_weeks(events: Events, test_season: int, cfg: APMConfig, jobs: int = 1) -> dict:
    """As the values are used: each week of ``test_season`` is predicted from the fit on the plays
    before its Monday (in-season plays included). Pooled over the season's plays: the squared-error
    reduction per play vs each week's constant, the correlation, and ``slope`` (EPA per point of
    predicted EPA; 1 is calibrated, under 1 means the values are too spread)."""
    fr = events.frame
    test = (fr.season == test_season).to_numpy()
    wk = fr.t[test].map(week_start).to_numpy()
    weeks = sorted(set(wk))
    fits = fit_weekly(events, cfg, weeks, lambda _: test_season, jobs)
    codes = events.sides["off"][test, 0]
    pred = np.zeros(test.sum())
    for w, tab in zip(weeks, fits):
        v = np.zeros(len(events.ids))
        v[np.searchsorted(events.ids, tab.player_id.to_numpy())] = tab.value.to_numpy()
        sel = wk == w
        pred[sel] = v[codes[sel]] - v[codes[sel]].mean()
    y = np.clip(fr.epa.to_numpy()[test], -cfg.target_clip, cfg.target_clip)
    yc = y - pd.Series(y).groupby(wk).transform("mean").to_numpy()
    return {"season": test_season, "mse_gain": float((yc ** 2).mean() - ((yc - pred) ** 2).mean()),
            "corr": float(np.corrcoef(pred, yc)[0, 1]), "slope": float(np.polyfit(pred, yc, 1)[0])}


# -- usage ------------------------------------------------------------------------------------------

def starter_plays(plays: pd.DataFrame, tg: pd.DataFrame) -> dict[int, float]:
    """{season: quarterback plays a starter makes per game}: the mean over the season's team-games
    of the starter's plays, used for the next season (the first season uses its own)."""
    n = plays.groupby(["game_id", "off", "qb"]).size().rename("n").reset_index()
    s = tg.merge(n, left_on=["game_id", "team_id", "starter"], right_on=["game_id", "off", "qb"], how="left")
    own = s.groupby("season").n.mean().to_dict()
    seasons = sorted(own)
    return {int(x): float(own[seasons[max(i - 1, 0)]]) for i, x in enumerate(seasons)}


# -- per-game lineup values (the wager model's ``_pv_*`` columns) ----------------------------------

def schedule_team_games(schedule: pd.DataFrame, days: pd.DataFrame, first_season: int) -> pd.DataFrame:
    """The wager model's team-games (every scheduled game with an FBS team, upcoming ones included):
    ``game_id, season, t, team_id, team`` with ``team`` the wager key (the school, or
    :data:`~runitback.wager.cfb.data.download.FCS` for an FCS side)."""
    s = schedule.assign(game_id=schedule.game_id.astype(str))
    s = s[(s.home_division == "fbs") | (s.away_division == "fbs")]
    s = s[s.season >= first_season].merge(days[["game_id", "t"]], on="game_id")
    parts = []
    for side in ("home", "away"):
        parts.append(pd.DataFrame({
            "game_id": s.game_id, "season": s.season.astype(int), "t": s.t,
            "team_id": pd.to_numeric(s[f"{side}_id"], errors="coerce"),
            "team": s[f"{side}_team"].where(s[f"{side}_division"] == "fbs", FCS)}))
    out = pd.concat(parts, ignore_index=True).dropna(subset=["team_id"]).astype({"team_id": "int64"})
    return out.drop_duplicates(["game_id", "team_id"]).sort_values(["t", "game_id", "team_id"],
                                                                   kind="stable").reset_index(drop=True)


def project_starters(tg: pd.DataFrame, games: pd.DataFrame, rule: str = "prev_starter") -> pd.DataFrame:
    """The projected starter of each of ``games`` (``game_id, team_id, t``), known at kickoff: the
    starter (``prev_starter``) or the dropback leader (``prev_leader``) of the team's last game before
    it. Teams with no earlier game have none. Returns ``game_id, team_id, qb``."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    col = "starter" if rule == "prev_starter" else "leader"
    hist = tg[["team_id", "t", col]].rename(columns={col: "qb"}).sort_values("t", kind="stable")
    g = games[["game_id", "team_id", "t"]].sort_values("t", kind="stable")
    out = pd.merge_asof(g, hist, on="t", by="team_id", allow_exact_matches=False)
    return out.dropna(subset=["qb"]).astype({"qb": "int64"})[["game_id", "team_id", "qb"]]


@dataclass
class GameValues:
    pv: pd.DataFrame          # game_id, team, _pv_0 .. _pv_2
    starters: pd.DataFrame    # per team-game: projected / actual / previous starter and their values
    usage: dict               # {season: quarterback plays per game of a starter}
    fits: dict                # {before (day): values table}
    latest: pd.DataFrame      # the most recent fit


def game_table(plays: pd.DataFrame, tg: pd.DataFrame, games: pd.DataFrame, jobs: int = 4,
               starters: pd.DataFrame | None = None, cfg: APMConfig | None = None,
               rule: str = "prev_starter") -> GameValues:
    """Leak-free weekly values (each fit on the plays before its Monday) and each of ``games``'
    (:func:`schedule_team_games`) ``_pv_*`` columns in points per game over replacement (see the
    module docstring). ``starters`` (``game_id, team_id, qb``): the projected starters; default
    :func:`project_starters` with ``rule``. ``tg``: :func:`~runitback.cfb.data.players.team_games`."""
    cfg = cfg or apm_config()
    events = events_from(plays)
    usage = starter_plays(plays, tg)
    games = games.assign(before=games.t.map(week_start))
    first_t = int(events.frame.t.min())
    weeks = sorted(w for w in set(games.before) if w > first_t)
    season_at = games.groupby("before").season.min().to_dict()
    log.info("%d quarterback plays; fitting %d weeks", len(events.frame), len(weeks))
    fits = fit_weekly(events, cfg, weeks, lambda w: season_at[w], jobs)
    fits = {w: f.rename(columns={"events": "plays", "value": "value_play"})[["player_id", "plays", "value_play"]]
            for w, f in zip(weeks, fits)}
    vals = {w: dict(zip(f.player_id, f.value_play)) for w, f in fits.items()}

    proj = project_starters(tg, games, rule) if starters is None else starters
    proj = dict(zip(zip(proj.game_id.astype(str), proj.team_id), proj.qb))
    actual = dict(zip(zip(tg.game_id, tg.team_id), tg.starter))
    hist = {team: d.sort_values(["t", "game_id"], kind="stable") for team, d in tg.groupby("team_id")}
    hist = {team: (d.t.to_numpy(), d.starter.to_numpy()) for team, d in hist.items()}
    seasons = sorted(usage)
    rows = []
    for g in games.itertuples():
        vp = vals.get(g.before)
        qb = proj.get((g.game_id, g.team_id))
        if vp is None or qb is None:
            continue
        mult = usage.get(g.season) or usage[seasons[-1]]
        ts, past = hist.get(g.team_id, (np.array([]), np.array([])))
        past = past[ts < g.t]
        recent = past[-RECENT:]
        prev = past[-1] if len(past) else None
        cur = vp.get(qb, 0.0) * mult
        v_prev = vp.get(prev, 0.0) * mult if prev is not None else cur
        v_recent = float(np.mean([vp.get(q, 0.0) for q in recent])) * mult if len(recent) else cur
        rows.append({"game_id": g.game_id, "team": g.team, "team_id": g.team_id, "season": g.season, "t": g.t,
                     "projected": qb, "starter": actual.get((g.game_id, g.team_id)), "prev_starter": prev,
                     "_pv_0": cur, "_pv_1": cur - v_prev, "_pv_2": cur - v_recent,
                     # the actual starter (evaluation only; unknown quarterbacks at replacement)
                     "starter_value": vp.get(actual[(g.game_id, g.team_id)], 0.0) * mult
                     if (g.game_id, g.team_id) in actual else np.nan})
    st = pd.DataFrame(rows)
    pv = st[["game_id", "team", *PV_COLUMNS]].copy()
    pv[PV_COLUMNS] = pv[PV_COLUMNS].astype(np.float32)
    st = st.astype({"starter": "Int64", "prev_starter": "Int64"})
    return GameValues(pv, st, usage, fits, fits[weeks[-1]])


def players_table(box: pd.DataFrame) -> pd.DataFrame:
    """``player_id, name, team_id`` (the player's latest team) from the player box scores."""
    b = box.dropna(subset=["athlete_id"]).assign(player_id=lambda d: pd.to_numeric(d.athlete_id, errors="coerce"))
    b = b.dropna(subset=["player_id"]).sort_values(["season", "game_id"], kind="stable")
    b = b.drop_duplicates("player_id", keep="last")
    return pd.DataFrame({"player_id": b.player_id.astype("int64"), "name": b.athlete_name,
                         "team_id": pd.to_numeric(b.team_id, errors="coerce")}).reset_index(drop=True)


def write_game_table(gv: GameValues, out: Path, players: pd.DataFrame | None = None) -> None:
    """``<out>/pv.parquet`` (the wager model's lineup table), ``starters.parquet`` (each team-game's
    projected, actual and previous starter, with values), ``usage.json`` (plays per game of a starter,
    by season) and ``values_latest.csv`` (the latest fit: every quarterback's value per play and per
    game as a starter)."""
    out.mkdir(parents=True, exist_ok=True)
    gv.pv.to_parquet(out / "pv.parquet", index=False)
    gv.starters.to_parquet(out / "starters.parquet", index=False)
    (out / "usage.json").write_text(json.dumps({str(k): v for k, v in gv.usage.items()}))
    last = gv.latest.copy()
    if players is not None:
        last["name"] = last.player_id.map(players.set_index("player_id").name)
    last["pts_as_starter"] = last.value_play * gv.usage[max(gv.usage)]
    last.sort_values("pts_as_starter", ascending=False).to_csv(out / "values_latest.csv", index=False,
                                                               float_format="%.4f")

