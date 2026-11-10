"""NBA projected lineups: who is expected to play each game, and for how long, from what is known
at tip-off.

There is no history of NBA injury reports, so the historical projection uses a documented
**stand-in** built from the ESPN player box score of the game itself (`player_games.parquet`, N1):

- **Ruled out:** players whose did-not-play `reason` is an injury (`dnp_kind == "injury"`: body
  parts, illness, surgery, protocols, reconditioning). This is what the final injury report would
  have said, as known at tip-off: it includes late scratches and misses nothing a team announced.
  The leak: a player ruled out after the line was set counts as known, and players listed out but
  for whom ESPN gives no injury reason (rest, personal, suspension, coach's decision) stay in.
- **Roster** (`roster=True`, the default): the candidates are the players listed in the game's box
  score for the team (actives and listed inactives), i.e. the roster at tip-off. Traded, waived and
  two-way players sent down drop out, new arrivals come in. ESPN leaves many injured inactives out
  of the box score altogether, so this also removes most absences the `reason` cannot see (see
  `evaluate`). With `roster=False`, the candidates are the team's previous `TEAM_GAMES` games'
  rotation (everyone in those box scores for the team) and only the injury reasons are removed:
  the plan's rule taken literally.

Nothing else about the game is used: its minutes, starters and results are not. Each candidate's
**expected minutes** are a recency-weighted mean (`DECAY` per game) of his minutes over his last
`PLAYER_GAMES` available games before the game's date, for any team (available = played or
coach's decision; injured, resting, ... games are skipped). They are scaled to the team's 240
minutes (no player over `MAX_MINUTES`), so a missing starter's minutes go to the others in
proportion. The projected **starters** are the five with the highest recency-weighted share of
starts in those games (ties by expected minutes). `rank` orders the available players by
projected minutes (the rotation slots); ruled-out players are kept at `rank == -1`.

For **upcoming games** (`live_lineups`), the roster is ESPN's current team roster, players listed
Out or Doubtful on ESPN's `/injuries` report are ruled out, and the starters are the posted ones
once ESPN publishes them (about half an hour before tip-off) instead of the projection. The minutes
come from the same history, so the live and historical projections are built the same way.

Output (`projected_lineups.parquet` in the processed directory), one row per (game, team, player):
`game_id, season, game_date, team, player_id, group, status ("available" | "out"), out_reason,
exp_minutes (before scaling), proj_minutes, proj_starter, rank, recent_games, source`
(`box` | `rotation` | `live` | `posted`).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..vocab import canon_team
from .players import position_group

log = logging.getLogger(__name__)

PLAYER_GAMES = 10     # a player's recent available games behind his expected minutes
DECAY = 0.85          # weight of his k-th most recent of them: DECAY ** k
TEAM_GAMES = 5        # roster=False: the rotation is everyone in the team's last TEAM_GAMES box scores
TEAM_MINUTES = 240.0  # regulation
MAX_MINUTES = 42.0
STARTERS = 5
OUT_KINDS = ("injury",)
OUT_STATUSES = {"Out", "Doubtful"}    # ESPN /injuries statuses that rule a player out (as the NFL's)
AVAILABLE_KINDS = (None, "coach")     # dnp_kind of a game in which the player was available

COLUMNS = ["game_id", "season", "game_date", "team", "player_id", "group", "status", "out_reason",
           "exp_minutes", "proj_minutes", "proj_starter", "rank", "recent_games", "source"]


# ------------------------------------------------------------------------------- player history

def player_history(pg: pd.DataFrame) -> pd.DataFrame:
    """Per player and available game (played, or a coach's decision), the recency-weighted minutes,
    start share and count of his last `PLAYER_GAMES` available games *up to and including* it:
    `player_id, date, w_minutes, w_start, n` (look these up strictly before a game's date)."""
    av = pg[~pg.dnp | pg.dnp_kind.eq("coach")]
    av = av.assign(date=pd.to_datetime(av.game_date)).sort_values(["player_id", "date"], kind="stable")
    av = av.drop_duplicates(["player_id", "date"], keep="last")
    g = av.groupby("player_id", sort=False)
    mins, start = av.minutes.astype(float), av.starter.astype(float)
    num_m = np.zeros(len(av))
    num_s = np.zeros(len(av))
    den = np.zeros(len(av))
    n = np.zeros(len(av))
    for k in range(PLAYER_GAMES):
        m_k = (mins if k == 0 else g.minutes.shift(k)).to_numpy(float)
        s_k = (start if k == 0 else g.starter.shift(k).astype(float)).to_numpy(float)
        ok = ~np.isnan(m_k)
        w = DECAY ** k
        num_m += np.where(ok, w * m_k, 0)
        num_s += np.where(ok, w * np.nan_to_num(s_k), 0)
        den += np.where(ok, w, 0)
        n += ok
    return pd.DataFrame({"player_id": av.player_id.to_numpy(), "date": av.date.to_numpy(),
                         "w_minutes": num_m / den, "w_start": num_s / den, "n": n.astype(np.int16)})


def history_before(hist: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """`keys` (player_id, game_date) joined with each player's history from strictly before that
    date: adds `exp_minutes`, `start_share`, `recent_games` (0 / 0 / 0 for a player with none)."""
    k = keys.assign(_i=np.arange(len(keys)), date=pd.to_datetime(keys.game_date))
    k = k.astype({"player_id": "int64"}).sort_values("date", kind="stable")
    h = hist.astype({"player_id": "int64"}).sort_values("date", kind="stable")
    m = pd.merge_asof(k, h, on="date", by="player_id", allow_exact_matches=False).sort_values("_i")
    out = keys.copy()
    out["exp_minutes"] = m.w_minutes.fillna(0.0).to_numpy()
    out["start_share"] = m.w_start.fillna(0.0).to_numpy()
    out["recent_games"] = m.n.fillna(0).astype(int).to_numpy()
    return out


# ----------------------------------------------------------------------------------- projection

def scale_minutes(exp: np.ndarray, total: float = TEAM_MINUTES, cap: float = MAX_MINUTES) -> np.ndarray:
    """Scale expected minutes to the team's `total`, no player over `cap` (the excess goes to the
    others in proportion). All zeros (no history at all) stay zeros."""
    m = np.asarray(exp, dtype=float).copy()
    fixed = np.zeros(len(m), dtype=bool)
    for _ in range(len(m) + 1):
        free = ~fixed
        tot = m[free].sum()
        if tot <= 0:
            break
        m[free] *= (total - m[fixed].sum()) / tot
        over = free & (m > cap)
        if not over.any():
            break
        m[over] = cap
        fixed |= over
    return m


def _scale_groups(exp: np.ndarray, key: np.ndarray, total: float = TEAM_MINUTES,
                  cap: float = MAX_MINUTES) -> np.ndarray:
    """`scale_minutes` within each group of `key` (one team-game each), vectorized."""
    m = np.asarray(exp, dtype=float).copy()
    fixed = np.zeros(len(m), dtype=bool)
    key = pd.Series(key)
    for _ in range(STARTERS + 3):   # each round caps at least one more player
        free_sum = pd.Series(np.where(fixed, 0.0, m)).groupby(key).transform("sum").to_numpy()
        fixed_sum = pd.Series(np.where(fixed, m, 0.0)).groupby(key).transform("sum").to_numpy()
        f = np.divide(total - fixed_sum, free_sum, out=np.zeros_like(m), where=free_sum > 0)
        m = np.where(fixed | (free_sum <= 0), m, m * f)
        over = ~fixed & (m > cap)
        if not over.any():
            break
        m[over] = cap
        fixed |= over
    return m


def _finish(cands: pd.DataFrame, hist: pd.DataFrame, posted: dict | None = None) -> pd.DataFrame:
    """Candidates (`game_id, team, player_id, game_date, status, ...`) -> projected minutes, starters
    (`posted`: (game_id, team) -> the posted five, which replace the projection) and rotation rank."""
    if cands.empty:
        return pd.DataFrame(columns=COLUMNS)
    d = history_before(hist, cands.reset_index(drop=True))
    av = d.status.eq("available").to_numpy()
    key = pd.factorize(pd.Series(list(zip(d.game_id, d.team))))[0]
    d["proj_minutes"] = np.where(av, _scale_groups(np.where(av, d.exp_minutes, 0.0), key), 0.0)
    d["_k"], d["_av"] = key, av
    s = d.sort_values(["_k", "_av", "start_share", "exp_minutes"], ascending=[True, False, False, False],
                      kind="stable")
    d["proj_starter"] = False
    d.loc[s.index, "proj_starter"] = (s.groupby("_k").cumcount() < STARTERS).to_numpy() & s._av.to_numpy()
    if posted:
        pk = [(g, t) in posted for g, t in zip(d.game_id, d.team)]
        is_posted = [p in posted.get((g, t), ()) for g, t, p in zip(d.game_id, d.team, d.player_id)]
        d["proj_starter"] = np.where(pk, np.array(is_posted) & av, d.proj_starter)
    r = d.sort_values(["_k", "_av", "proj_minutes", "proj_starter"], ascending=[True, False, False, False],
                      kind="stable")
    d.loc[r.index, "rank"] = np.where(r._av, r.groupby("_k").cumcount(), -1)
    d["proj_minutes"] = d.proj_minutes.round(2)
    d["exp_minutes"] = d.exp_minutes.round(2)
    d["rank"] = d["rank"].astype(np.int16)
    d["proj_starter"] = d.proj_starter.astype(bool)
    return d.sort_values(["game_date", "game_id", "team", "status", "rank"], kind="stable")[COLUMNS] \
        .reset_index(drop=True)


def projected_lineups(pg: pd.DataFrame, roster: bool = True, out_kinds: tuple[str, ...] = OUT_KINDS,
                      games: set[str] | None = None) -> pd.DataFrame:
    """Projected lineups of every (game, team) in `pg` (player_games.parquet), or of `games` only.

    See the module docstring: candidates are the game's box-score roster (`roster=True`) or the
    team's last `TEAM_GAMES` games' rotation; players with a DNP `reason` of a kind in `out_kinds`
    are ruled out; minutes and starters come only from games before the game's date."""
    pg = pg.assign(game_id=pg.game_id.astype(str))
    hist = player_history(pg)
    target = pg if games is None else pg[pg.game_id.isin(set(map(str, games)))]
    out_now = target.dnp & target.dnp_kind.isin(out_kinds)
    reason = dict(zip(zip(target.game_id, target.team, target.player_id), target.dnp_reason.where(out_now)))
    group_of = pg.dropna(subset=["group"]).sort_values("game_date").groupby("player_id").group.last()
    if roster:
        c = target[["game_id", "season", "game_date", "team", "player_id"]].copy()
        source = "box"
    else:  # the team's previous TEAM_GAMES box scores (its rotation), plus nothing from this game
        order = pg[["game_id", "season", "game_date", "team"]].drop_duplicates(["game_id", "team"]) \
            .sort_values(["team", "game_date", "game_id"], kind="stable")
        roster_of = {k: set(d.player_id) for k, d in pg.groupby(["game_id", "team"], sort=False)}
        want = None if games is None else set(map(str, games))
        rows = []
        for team, o in order.groupby("team", sort=False):
            ids = o.game_id.tolist()
            for i, r in enumerate(o.itertuples(index=False)):
                if want is not None and r.game_id not in want:
                    continue
                prev = set().union(*(roster_of[(h, team)] for h in ids[max(0, i - TEAM_GAMES): i]))
                rows += [(r.game_id, r.season, r.game_date, team, p) for p in prev]
        c = pd.DataFrame(rows, columns=["game_id", "season", "game_date", "team", "player_id"])
        source = "rotation"
    keys = list(zip(c.game_id, c.team, c.player_id))
    why = [reason.get(k) for k in keys]
    c["out_reason"] = [w if isinstance(w, str) else None for w in why]
    c["status"] = np.where(c.out_reason.notna(), "out", "available")
    c["group"] = c.player_id.map(group_of)
    c["source"] = source
    return _finish(c, hist)


# ------------------------------------------------------------------------------------- evaluation

def evaluate(proj: pd.DataFrame, pg: pd.DataFrame) -> pd.DataFrame:
    """Per season, the projection against what happened (the game's box score), next to the naive
    baseline of the team's previous game as played (`prev_*`):

    - `starters`: share of the actual starters among the projected five; `all5`: share of
      team-games with all five right
    - `minutes_mae`: mean absolute error of projected vs actual minutes over every player who was
      projected to play or played (per player-game)
    - `missed`: regulars (20+ minutes in the team's previous game) who did not play this one
      (DNP or not in the box score); `missed_flagged`: the share of them the projection did not
      count on (ruled out, or not a candidate)"""
    pg = pg.assign(game_id=pg.game_id.astype(str))
    act = pg[["game_id", "season", "game_date", "team", "player_id", "starter", "minutes", "dnp"]]
    act = act.assign(minutes=act.minutes.where(~act.dnp, 0.0), starter=act.starter & ~act.dnp)
    tg = act[["game_id", "season", "game_date", "team"]].drop_duplicates(["game_id", "team"]) \
        .sort_values(["team", "game_date", "game_id"], kind="stable")
    tg["prev"] = tg.groupby("team").game_id.shift(1)
    projected = set(zip(proj.game_id, proj.team))
    tg = tg[[k in projected for k in zip(tg.game_id, tg.team)]]
    key = ["game_id", "team", "player_id"]
    av = proj[proj.status.eq("available")][key + ["proj_minutes", "proj_starter"]]
    cur = act[key + ["starter", "minutes"]]
    prev = act.rename(columns={"game_id": "prev", "starter": "prev_starter", "minutes": "prev_minutes"})[
        ["prev", "team", "player_id", "prev_starter", "prev_minutes"]]
    prev = tg[["game_id", "team", "prev"]].dropna().merge(prev, on=["prev", "team"]).drop(columns="prev")
    m = tg[["game_id", "team", "season"]].merge(
        cur.merge(av, on=key, how="outer").merge(prev, on=key, how="outer"), on=["game_id", "team"])
    for c in ("proj_minutes", "minutes", "prev_minutes"):
        m[c] = m[c].fillna(0.0)
    for c in ("proj_starter", "starter", "prev_starter"):
        m[c] = m[c].astype("boolean").fillna(False).astype(bool)
    m["has_prev"] = m.game_id.isin(set(prev.game_id))
    st = m[m.starter].groupby(["season", "game_id", "team"]).agg(
        n=("starter", "size"), hit=("proj_starter", "sum"), prev_hit=("prev_starter", "sum"),
        has_prev=("has_prev", "first"))
    st = st[st.n == STARTERS].reset_index()
    missed = m[m.has_prev & (m.prev_minutes >= 20) & (m.minutes <= 0)]
    rows = {}
    for s, g in m.groupby("season"):
        h, mi = st[st.season == s], missed[missed.season == s]
        hp, gp = h[h.has_prev], g[g.has_prev]
        rows[int(s)] = dict(
            team_games=len(h), starters=h.hit.sum() / max(1, STARTERS * len(h)),
            all5=float((h.hit == STARTERS).mean()), minutes_mae=float((g.proj_minutes - g.minutes).abs().mean()),
            missed=len(mi), missed_flagged=float((mi.proj_minutes <= 0).mean()) if len(mi) else np.nan,
            prev_starters=hp.prev_hit.sum() / max(1, STARTERS * len(hp)),
            prev_all5=float((hp.prev_hit == STARTERS).mean()),
            prev_minutes_mae=float((gp.prev_minutes - gp.minutes).abs().mean()))
    return pd.DataFrame(rows).T.rename_axis("season")


# ------------------------------------------------------------------------------- live (upcoming)

def _espn(path: str, **params) -> dict:
    from ...wager.espn import SITE, get_json
    return get_json(SITE.format(sport="basketball", league="nba") + "/" + path, **params)


def _athlete_id(a: dict) -> int | None:
    if a.get("id"):
        return int(a["id"])
    for link in a.get("links") or []:   # /injuries gives the id only in the player-card links
        parts = str(link.get("href", "")).split("/id/")
        if len(parts) > 1 and parts[1].split("/")[0].isdigit():
            return int(parts[1].split("/")[0])
    return None


def fetch_upcoming(dates: list[str]) -> pd.DataFrame:
    """ESPN's not-yet-started games on `dates` ("YYYYMMDD"): game_id, game_date (US Eastern),
    season, home, away, home_id, away_id."""
    rows = []
    for day in dates:
        data = _espn("scoreboard", dates=day)
        season = (data.get("season") or {}).get("year")
        for e in data.get("events") or []:
            if ((e.get("status") or {}).get("type") or {}).get("state") != "pre":
                continue
            comp = (e.get("competitions") or [{}])[0]
            side = {c.get("homeAway"): c.get("team") or {} for c in comp.get("competitors") or []}
            start = pd.Timestamp(e["date"]).tz_convert("America/New_York")
            rows.append(dict(game_id=str(e["id"]), game_date=start.strftime("%Y-%m-%d"),
                             season=int(season or start.year + (start.month >= 8)),
                             home=canon_team(side["home"].get("abbreviation", "")),
                             away=canon_team(side["away"].get("abbreviation", "")),
                             home_id=str(side["home"].get("id")), away_id=str(side["away"].get("id"))))
    return pd.DataFrame(rows, columns=["game_id", "game_date", "season", "home", "away", "home_id", "away_id"])


def parse_injuries(data: dict) -> pd.DataFrame:
    """ESPN `/injuries` -> team, player_id, status, date, comment."""
    rows = []
    for t in data.get("injuries") or []:
        for i in t.get("injuries") or []:
            a = i.get("athlete") or {}
            team = (a.get("team") or {}).get("abbreviation")
            pid = _athlete_id(a)
            if pid is None or not team:
                continue
            rows.append((canon_team(team), pid, i.get("status"), i.get("date"), i.get("shortComment")))
    return pd.DataFrame(rows, columns=["team", "player_id", "status", "date", "comment"])


def parse_roster(data: dict) -> list[tuple[int, str | None]]:
    """ESPN `/teams/<id>/roster` -> [(player_id, position abbreviation)]."""
    return [(int(a["id"]), (a.get("position") or {}).get("abbreviation")) for a in data.get("athletes") or []
            if a.get("id")]


def parse_posted_starters(data: dict) -> dict[str, list[int]]:
    """ESPN `/summary?event=<id>` -> team -> the five posted starters (empty until ESPN posts them)."""
    out = {}
    for t in (data.get("boxscore") or {}).get("players") or []:
        team = canon_team((t.get("team") or {}).get("abbreviation", ""))
        ids = [int(a["athlete"]["id"]) for s in t.get("statistics") or [] for a in s.get("athletes") or []
               if a.get("starter") and (a.get("athlete") or {}).get("id")]
        if len(ids) == STARTERS:
            out[team] = ids
    return out


def live_candidates(games: pd.DataFrame, rosters: dict[str, list[tuple[int, str | None]]],
                    injuries: pd.DataFrame) -> pd.DataFrame:
    """Candidates of upcoming `games` from current rosters (team -> [(player_id, position)]) and
    the injury report (Out / Doubtful ruled out)."""
    out = {(t, p): s for t, p, s in zip(injuries.team, injuries.player_id, injuries.status)
           if s in OUT_STATUSES}
    rows = []
    for g in games.itertuples(index=False):
        for team in (g.home, g.away):
            for pid, pos in rosters.get(team, []):
                st = out.get((team, pid))
                rows.append((g.game_id, g.season, g.game_date, team, pid, position_group(pos),
                             st, "out" if st else "available"))
    return pd.DataFrame(rows, columns=["game_id", "season", "game_date", "team", "player_id", "group",
                                       "out_reason", "status"])


def live_lineups(pg: pd.DataFrame, dates: list[str] | None = None, days: int = 2) -> pd.DataFrame:
    """Projected lineups of ESPN's upcoming games (today and the next `days - 1` days, or `dates`),
    from the current rosters, the `/injuries` report and posted starters where ESPN has them."""
    if dates is None:
        today = pd.Timestamp.now(tz="America/New_York").normalize()
        dates = [(today + pd.Timedelta(days=k)).strftime("%Y%m%d") for k in range(days)]
    games = fetch_upcoming(dates)
    if games.empty:
        return pd.DataFrame(columns=COLUMNS)
    injuries = parse_injuries(_espn("injuries"))
    rosters, posted = {}, {}
    for g in games.itertuples(index=False):
        for team, tid in ((g.home, g.home_id), (g.away, g.away_id)):
            if team not in rosters:
                rosters[team] = parse_roster(_espn(f"teams/{tid}/roster"))
        for team, ids in parse_posted_starters(_espn("summary", event=g.game_id)).items():
            posted[(g.game_id, team)] = ids
    return project_live(pg, games, rosters, injuries, posted)


def project_live(pg: pd.DataFrame, games: pd.DataFrame, rosters: dict, injuries: pd.DataFrame,
                 posted: dict[tuple[str, str], list[int]] | None = None) -> pd.DataFrame:
    """`live_lineups` without the network: the projection from already fetched ESPN data."""
    c = live_candidates(games, rosters, injuries)
    group_of = pg.dropna(subset=["group"]).sort_values("game_date").groupby("player_id").group.last()
    c["group"] = c.group.fillna(c.player_id.map(group_of))
    posted = posted or {}
    c["source"] = ["posted" if (g, t) in posted else "live" for g, t in zip(c.game_id, c.team)]
    return _finish(c, player_history(pg), posted)


# ------------------------------------------------------------------------------------------ build

def build_lineups(processed: Path, roster: bool = True, live: bool = False) -> dict:
    """Write `projected_lineups.parquet` (every game in player_games.parquet) and, with `live`,
    `projected_lineups_upcoming.parquet`; returns `evaluate`'s summary per season."""
    pg = pd.read_parquet(processed / "player_games.parquet")
    proj = projected_lineups(pg, roster=roster)
    proj.to_parquet(processed / "projected_lineups.parquet")
    ev = evaluate(proj, pg)
    log.info("projected lineups: %d rows, %d team-games\n%s", len(proj),
             proj[["game_id", "team"]].drop_duplicates().shape[0], ev.round(4).to_string())
    if live:
        up = live_lineups(pg)
        up.to_parquet(processed / "projected_lineups_upcoming.parquet")
        log.info("upcoming: %d games", up.game_id.nunique())
    return {int(s): {k: (round(float(v), 4) if pd.notna(v) else None) for k, v in r.items()}
            for s, r in ev.iterrows()}
