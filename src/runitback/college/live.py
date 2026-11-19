"""College basketball projected rotations of **upcoming games** (work package C4, step 1 of the production
switch for ``RUNITBACK_WCBB_PLAYER_VALUES``).

The history's rotations (:mod:`.lineups`, rule ``dnp``) know who played; an upcoming game has no box
score, so its candidates come from what ESPN has before tip-off:

- **Candidates:** the team's current ESPN roster, minus anyone ESPN's ``/injuries`` lists as Out or
  Doubtful (ESPN has listed none for college teams in the months checked, so in practice nobody is
  removed), and **only players with a played game in the box-score history** (any team, so a transfer
  keeps his). A walk-on who never played, or a freshman who has not played yet, has no minutes to
  project and would only dilute the rotation; the history's candidates are likewise players who got in.
- **Minutes and starters:** exactly :func:`runitback.college.lineups.project`'s: each player's
  recency-weighted minutes and starts over the games *before the game's date*.

No leak: every input is known before the game (the roster and injury report as of the run, earlier
games), unlike the history's ``dnp`` rule, which takes the box score's roster. The upcoming games'
rotations are therefore a little less informed than the training rows' (a player who sits for a
coach's decision is still on the roster).

Output (``projected_lineups_upcoming.parquet``): :data:`runitback.college.lineups.COLUMNS` plus
``home`` and ``source`` (``live``), one row per (game, team, player), for ESPN's not-yet-started
Division I games on the dates asked. ``team`` is the wager model's team key (the school's latest name,
:func:`runitback.college.data.latest_team_names`); games with a team missing from the box-score history
(non-Division I opponents) are skipped.
"""
from __future__ import annotations

import logging

import pandas as pd

from ..wager.espn import SITE, get_json
from .data import GROUP
from .lineups import COLUMNS, player_history, project

log = logging.getLogger(__name__)

LEAGUES = {"cbb": "mens-college-basketball", "wcbb": "womens-college-basketball"}
OUT_STATUSES = ("Out", "Doubtful")
DIVISION_ONE = 50           # the scoreboard's group of every Division I game (the default lists featured games)
OUTPUT_COLUMNS = [*COLUMNS, "home"]


def _espn(gender: str, path: str, **params) -> dict:
    return get_json(SITE.format(sport="basketball", league=LEAGUES[gender]) + "/" + path, **params)


def parse_scoreboard(data: dict) -> pd.DataFrame:
    """ESPN's scoreboard -> its not-yet-started games: ``game_id, game_date`` (US Eastern), ``season``,
    ``home_id, away_id`` (ESPN team ids)."""
    season = (data.get("leagues") or [{}])[0].get("season", {}).get("year")
    rows = []
    for e in data.get("events") or []:
        if ((e.get("status") or {}).get("type") or {}).get("state") != "pre":
            continue
        comp = (e.get("competitions") or [{}])[0]
        side = {c.get("homeAway"): c.get("team") or {} for c in comp.get("competitors") or []}
        if "home" not in side or "away" not in side:
            continue
        start = pd.Timestamp(e["date"]).tz_convert("America/New_York")
        rows.append(dict(game_id=str(e["id"]), game_date=start.strftime("%Y-%m-%d"),
                         season=int(season or start.year + (start.month >= 8)),
                         home_id=int(side["home"]["id"]), away_id=int(side["away"]["id"])))
    return pd.DataFrame(rows, columns=["game_id", "game_date", "season", "home_id", "away_id"])


def fetch_upcoming(gender: str, dates: list[str]) -> pd.DataFrame:
    """The not-yet-started Division I games on ``dates`` (``YYYYMMDD``): :func:`parse_scoreboard`'s columns."""
    parts = [parse_scoreboard(_espn(gender, "scoreboard", dates=d, groups=DIVISION_ONE, limit=1000)) for d in dates]
    parts = [p for p in parts if len(p)]
    if not parts:
        return parse_scoreboard({})
    return pd.concat(parts, ignore_index=True).drop_duplicates("game_id")


def _athlete_id(a: dict) -> int | None:
    if a.get("id"):
        return int(a["id"])
    for link in a.get("links") or []:    # /injuries gives the id only in the player-card links
        parts = str(link.get("href", "")).split("/id/")
        if len(parts) > 1 and parts[1].split("/")[0].isdigit():
            return int(parts[1].split("/")[0])
    return None


def parse_injuries(data: dict) -> set[int]:
    """ESPN ``/injuries`` -> the ids of players listed Out or Doubtful."""
    out = set()
    for t in data.get("injuries") or []:
        for i in t.get("injuries") or []:
            pid = _athlete_id(i.get("athlete") or {})
            if pid is not None and i.get("status") in OUT_STATUSES:
                out.add(pid)
    return out


def parse_roster(data: dict) -> list[tuple[int, str | None]]:
    """ESPN ``/teams/<id>/roster`` -> ``[(player_id, position group or None)]``."""
    return [(int(a["id"]), GROUP.get((a.get("position") or {}).get("abbreviation", ""))) for a in
            data.get("athletes") or [] if a.get("id")]


def live_candidates(games: pd.DataFrame, rosters: dict[int, list[tuple[int, str | None]]], out: set[int],
                    names: pd.Series, known: set[int]) -> pd.DataFrame:
    """Candidates of the upcoming ``games`` (:func:`parse_scoreboard`): each team's roster (ESPN team id
    -> players) minus ``out`` and the players not in ``known`` (those with a played game). A team with no
    name in ``names`` (not in the box-score history) drops its game. Columns of
    :func:`runitback.college.lineups.candidates` plus ``home``."""
    rows = []
    for g in games.itertuples(index=False):
        if g.home_id not in names.index or g.away_id not in names.index:
            continue
        for tid, home in ((g.home_id, True), (g.away_id, False)):
            for pid, group in rosters.get(tid, []):
                if pid in out or pid not in known:
                    continue
                rows.append((g.game_id, g.season, g.game_date, tid, names[tid], pid, group, "live", home))
    return pd.DataFrame(rows, columns=["game_id", "season", "game_date", "team_id", "team", "player_id", "group",
                                       "source", "home"])


def project_live(pg: pd.DataFrame, games: pd.DataFrame, rosters: dict, out: set[int]) -> pd.DataFrame:
    """:func:`live_lineups` without the network: the projection from already fetched ESPN data."""
    from .data import latest_team_names

    names = latest_team_names(pg)
    hist = player_history(pg)
    c = live_candidates(games, rosters, out, names, set(hist.player_id))
    if c.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    group_of = pg.dropna(subset=["group"]).sort_values("game_date").groupby("player_id").group.last()
    c["group"] = c.group.fillna(c.player_id.map(group_of))
    home = c.set_index(["game_id", "team_id"]).home.groupby(level=[0, 1]).first()
    p = project(c.drop(columns="home"), hist)
    p["home"] = [bool(home[(g, t)]) for g, t in zip(p.game_id, p.team_id)]
    return p[OUTPUT_COLUMNS]


def live_lineups(gender: str, pg: pd.DataFrame, dates: list[str] | None = None, days: int = 2) -> pd.DataFrame:
    """Projected rotations of ESPN's upcoming Division I games today and the next ``days - 1`` days (US
    Eastern), or on ``dates`` (``YYYYMMDD``): :data:`OUTPUT_COLUMNS`."""
    if dates is None:
        today = pd.Timestamp.now(tz="America/New_York").normalize()
        dates = [(today + pd.Timedelta(days=k)).strftime("%Y%m%d") for k in range(days)]
    games = fetch_upcoming(gender, dates)
    log.info("%d upcoming games on %s", len(games), dates)
    if games.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    out = parse_injuries(_espn(gender, "injuries"))
    rosters = {}
    for tid in sorted(set(games.home_id) | set(games.away_id)):
        rosters[tid] = parse_roster(_espn(gender, f"teams/{tid}/roster"))
    return project_live(pg, games, rosters, out)
