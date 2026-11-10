"""NHL games and team box scores from the NHL's stats API, betting lines from
ESPN.

Every season is four requests to the NHL's own stats API
(``api.nhle.com/stats/rest``): the schedule (``/game``, upcoming games
included) and three per-game team reports (``/team/summary``,
``/team/realtime``, ``/team/powerplay``: shots, hits, blocks, faceoffs,
power plays, ...). The schedule is the source of truth for which games
exist; past games that were never played (unneeded playoff games) are
dropped.

Betting lines come from ESPN's odds API, one request per game, from the
2019-20 season on (see :mod:`runitback.wager.espn`). ESPN's game ids come from its
scoreboard, one request per month, matched to the NHL's games on date and
teams. ESPN's "spread" is the puck line (a 1.5-goal handicap priced like a
moneyline); it is kept as ``puck_line`` with its prices, while the market's
win probability comes from the moneylines. Finished seasons' responses and
finished games' lines are cached on disk (``cache_dir``).

Team keys are the NHL's three-letter codes, a relocated franchise's current
one for its whole history (Atlanta -> Winnipeg, Phoenix / Arizona -> Utah,
whose team took over Arizona's roster); the name it played under stays in
``team_name`` / ``opp_name``. Seasons are keyed by their ending year
(2025-26 is 2026). Goals include the shootout's deciding goal, so ``result``
is the final score's margin.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ... import espn

logger = logging.getLogger(__name__)

API = "https://api.nhle.com/stats/rest/en"

# NHL game types kept: 1 preseason, 19 / 20 (4 Nations, All-Star) dropped.
GAME_TYPES = {2: "regular", 3: "playoffs"}

# gameStateId of finished games.
FINAL_STATES = {6, 7}

# Relocated franchises -> their current code.
MOVES = {"ATL": "WPG", "PHX": "UTA", "ARI": "UTA"}

# ESPN abbreviations that differ from the NHL's codes.
ESPN_ABBREVIATIONS = {"NJ": "NJD", "TB": "TBL", "SJ": "SJS", "LA": "LAK",
                      "UTAH": "UTA", "MON": "MTL"}  # fmt: skip

# Per-game team reports, with the columns kept and their games-table names.
REPORTS = {
    "summary": {
        "shotsForPerGame": "shots",
        "faceoffWinPct": "faceoff_pct",
        "powerPlayPct": "pp_pct",
        "penaltyKillPct": "pk_pct",
    },
    "realtime": {
        "hits": "hits",
        "blockedShots": "blocked_shots",
        "giveaways": "giveaways",
        "takeaways": "takeaways",
        "missedShots": "missed_shots",
        "emptyNetGoals": "empty_net_goals",
    },
    "powerplay": {
        "ppOpportunities": "pp_opportunities",
        "powerPlayGoalsFor": "pp_goals",
        "ppTimeOnIcePerGame": "pp_toi",
        "shGoalsAgainst": "sh_goals_against",
    },
}
BOX_COLUMNS = [col for report in REPORTS.values() for col in report.values()]

# The first season (by ending year) with betting lines on ESPN.
FIRST_LINES_SEASON = 2020

# Longest rest counted, in days (season openers get it too).
MAX_REST = 7

# Odds lookahead: upcoming games starting within this many days get lines.
LINES_DAYS = 3


def season_of(day: datetime.date) -> int:
    """The season a date belongs to, by its ending year (2025-26 is 2026)."""
    return day.year + 1 if day.month >= 8 else day.year


def season_complete(season: int, today: datetime.datetime) -> bool:
    """Whether nothing in ``season`` can change (the Final ends in June; the
    2019-20 bubble playoffs in late September)."""
    last = (
        datetime.date(season, 10, 31)
        if season == 2020
        else datetime.date(season, 7, 31)
    )
    return today.date() > last


def season_id(season: int) -> str:
    return f"{season - 1}{season}"


def months(season: int) -> list[str]:
    """``YYYYMM`` of every month a season can have games in (the 2020-21
    season ran January to July)."""
    start = datetime.date(season - 1, 9, 1)
    end = datetime.date(season, 10 if season == 2020 else 7, 1)
    out, day = [], start
    while day <= end:
        out.append(day.strftime("%Y%m"))
        day = (day.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)
    return out


def team_codes(records: list[dict]) -> dict[int, str]:
    """NHL ``/team`` -> team id: games-table key."""
    return {
        int(team["id"]): MOVES.get(team["triCode"], team["triCode"])
        for team in records
        if team.get("triCode")
    }


def team_names(records: list[dict]) -> dict[int, str]:
    return {int(team["id"]): team["fullName"] for team in records}


# ----------------------------------------------------------------- schedule
def schedule_frame(records: list[dict], codes: dict[int, str]) -> pd.DataFrame:
    """NHL ``/game`` -> one row per game (home / away columns), typed.

    Preseason, All-Star and international exhibition games are dropped.
    """
    games = pd.DataFrame(records)
    if games.empty:
        return games
    games["game_type"] = pd.to_numeric(games["gameType"]).map(GAME_TYPES)
    games = games.dropna(subset=["game_type"]).copy()
    games["game_id"] = games["id"].astype(str)
    games["season"] = games["season"].astype(str).str[4:].astype(int)
    eastern = pd.to_datetime(games["easternStartTime"], errors="coerce")
    games["start"] = eastern.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="shift_forward"
    ).dt.tz_convert("UTC")
    games["start_date"] = games["start"].dt.strftime("%Y-%m-%dT%H:%MZ")
    games["completed"] = games["gameStateId"].isin(FINAL_STATES)
    games["periods"] = pd.to_numeric(games.get("period"), errors="coerce")
    for side, col in (("home", "homeTeamId"), ("away", "visitingTeamId")):
        games[f"{side}_id"] = pd.to_numeric(games[col]).astype(int)
        games[f"{side}_team"] = games[f"{side}_id"].map(codes)
    games["home_score"] = pd.to_numeric(games["homeScore"], errors="coerce")
    games["away_score"] = pd.to_numeric(games["visitingScore"], errors="coerce")
    games["game_date"] = games["gameDate"]
    games["neutral_site"] = False
    return games.dropna(subset=["home_team", "away_team"]).drop_duplicates("game_id")


def box_frame(reports: dict[str, list[dict]]) -> pd.DataFrame:
    """The per-game team reports -> ``game_id``, ``id`` (team) and the
    :data:`BOX_COLUMNS` stats."""
    out = None
    for report, columns in REPORTS.items():
        frame = pd.DataFrame(reports.get(report) or [])
        if frame.empty:
            frame = pd.DataFrame(columns=["gameId", "teamId", *columns])
        frame = frame.reindex(columns=["gameId", "teamId", *columns])
        frame = frame.rename(columns={"gameId": "game_id", "teamId": "id", **columns})
        frame["game_id"] = frame["game_id"].astype(str)
        frame["id"] = pd.to_numeric(frame["id"], errors="coerce")
        frame = frame.drop_duplicates(["game_id", "id"])
        out = frame if out is None else out.merge(frame, on=["game_id", "id"],
                                                    how="outer")  # fmt: skip
    out[BOX_COLUMNS] = out[BOX_COLUMNS].apply(pd.to_numeric, errors="coerce")
    for pct in ("faceoff_pct", "pp_pct", "pk_pct"):
        out[pct] = 100 * out[pct]
    return out


# -------------------------------------------------------------------- games
def build_games(
    schedule: list[dict],
    reports: dict[str, list[dict]],
    teams: list[dict],
    lines: dict[str, dict] | None = None,
    today: datetime.datetime | None = None,
) -> pd.DataFrame:
    """One row per team per game from the NHL's schedule and team reports
    and ESPN's lines (keyed by the NHL's game id).

    Past games that were never completed are dropped; future ones are kept
    as upcoming games with missing results and stats.
    """
    from .features import STATS_COLUMNS

    today = today or datetime.datetime.now(datetime.timezone.utc)
    codes = team_codes(teams)
    games = schedule_frame(schedule, codes)
    if games.empty:
        return games
    upcoming = ~games["completed"] | games["home_score"].isna()
    stale = upcoming & (games["start"] < pd.Timestamp(today) - pd.Timedelta(days=1))
    games = games[~stale].copy()
    games.loc[upcoming, ["home_score", "away_score"]] = np.nan
    games["game_date"], games["day"], games["week"] = espn.day_numbers(games)
    games["postseason"] = games["game_type"].ne("regular").astype(int)
    names = team_names(teams)
    for side in ("home", "away"):
        games[f"{side}_display_name"] = games[f"{side}_id"].map(names)
    games = espn.join_lines(games, lines)
    games = games.rename(columns={"spread_line": "puck_line"})
    for side in ("home", "away"):
        games = games.rename(columns={f"{side}_spread_odds": f"{side}_puck_line_odds"})

    keep = ["game_id", "season", "day", "week", "game_date", "start_date",
            "game_type", "postseason", "neutral_site", "periods", "puck_line",
            "total_line", "ou_over", "ou_under"]  # fmt: skip
    sides = ["id", "team", "display_name", "score", "moneyline", "puck_line_odds"]
    cols = keep + [f"{s}_{c}" for s in ("home", "away") for c in sides]
    games = games.reindex(columns=cols)
    rows = espn.invert(games, negated=("puck_line",))
    rows = rows.rename(
        columns={
            "display_name": "team_name",
            "opp_display_name": "opp_name",
            "score": "goals",
            "opp_score": "opp_goals",
            "ou_over": "over_odds",
            "ou_under": "under_odds",
        }
    )
    rows["result"] = rows["goals"] - rows["opp_goals"]
    rows["total"] = rows["goals"] + rows["opp_goals"]
    rows["rest"] = espn.rest_days(rows, MAX_REST)
    rows = espn.with_opp(rows, "rest")

    box = box_frame(reports)
    stats = [c for c in STATS_COLUMNS if c != "goals"]
    box = box.reindex(columns=["game_id", "id", *stats])
    rows = rows.merge(box, on=["game_id", "id"], how="left")
    rows = rows.merge(
        box.rename(columns={"id": "opp_id", **{c: f"opp_{c}" for c in stats}}),
        on=["game_id", "opp_id"],
        how="left",
    )
    # finished games' stats only: a stray box score cannot leak into a preview
    rows.loc[rows["result"].isna(), stats + [f"opp_{c}" for c in stats]] = np.nan
    rows = rows.drop(columns=["id", "opp_id", "neutral_site"])
    return rows.sort_values(["season", "day", "game_id", "location"]).reset_index(
        drop=True
    )


# ----------------------------------------------------------------- download
def espn_ids(
    games: pd.DataFrame, season: int, final: bool, cache: espn.Cache
) -> pd.Series:
    """ESPN's id of each of a season's games (``schedule_frame`` rows), from
    its monthly scoreboards (finished seasons' cached)."""
    events = []
    for month in months(season):
        name = f"scoreboard_{month}.json" if final else None
        data = espn.get_json(
            espn.SITE.format(sport="hockey", league="nhl") + "/scoreboard",
            name,
            cache,
            dates=month,
            limit=1000,
        )
        events.append(espn.events_frame(data.get("events") or []))
    events = pd.concat(events, ignore_index=True)
    local = games["start"].dt.tz_convert("America/New_York")
    frame = games.assign(game_date=local.dt.strftime("%Y-%m-%d"))
    return espn.match_events(frame, events, espn_keys(events))


def espn_keys(events: pd.DataFrame) -> dict[str, str]:
    """ESPN abbreviation -> games-table key, for every team in ``events``."""
    out = {}
    for abbreviation in set(events["home"]) | set(events["away"]):
        if isinstance(abbreviation, str):
            code = ESPN_ABBREVIATIONS.get(abbreviation, abbreviation)
            out[abbreviation] = MOVES.get(code, code)
    return out


def fetch_season(
    season: int,
    today: datetime.datetime,
    cache: espn.Cache,
    teams: list[dict],
    lines_days: int = LINES_DAYS,
) -> tuple[list[dict], dict[str, list[dict]], dict]:
    """``(schedule, team reports, lines)`` of one season."""
    final = season_complete(season, today)
    sid = season_id(season)
    schedule = espn.get_json(
        f"{API}/game",
        f"game_{season}.json" if final else None,
        cache,
        cayenneExp=f"season={sid}",
    )["data"]
    reports = {
        report: espn.get_json(
            f"{API}/team/{report}",
            f"{report}_{season}.json" if final else None,
            cache,
            isAggregate="false",
            isGame="true",
            start=0,
            limit=-1,
            cayenneExp=f"seasonId={sid}",
        )["data"]
        for report in REPORTS
    }
    lines = {}
    games = schedule_frame(schedule, team_codes(teams))
    if season >= FIRST_LINES_SEASON and len(games):
        games["espn_id"] = espn_ids(games, season, final, cache)
        games = games.dropna(subset=["espn_id"])
        finished, soon = espn.lines_to_fetch(games, today, lines_days)
        to_espn = dict(zip(games["game_id"], games["espn_id"]))
        by_espn = espn.season_lines(
            season,
            [to_espn[g] for g in finished],
            [to_espn[g] for g in soon],
            cache,
            lambda g: espn.fetch_odds("hockey", "nhl", g),
        )
        lines = {g: by_espn[e] for g, e in to_espn.items() if e in by_espn}
    return schedule, reports, lines


def download(
    output_file: str | None = None,
    first_season: int = 2006,
    last_season: int | None = None,
    cache_dir: str | None = "data/nhl/nhl_cache",
) -> pd.DataFrame:
    """Download every season's games, box scores and lines.

    Args:
        output_file: Also write the games table here.
        first_season / last_season: Seasons to download, by ending year
            (default: 2005-06, the first after the lockout and the first
            with every report, through the current season). ESPN's betting
            lines start in 2019-20.
        cache_dir: Reuse finished seasons' responses and finished games'
            lines from here (``None``: always download). A full download
            makes about 9,000 odds requests; a daily refresh a few dozen.
    """
    today = datetime.datetime.now(datetime.timezone.utc)
    last_season = last_season or season_of(today.date())
    cache = espn.Cache(cache_dir)
    teams = espn.get_json(f"{API}/team")["data"]
    schedule, reports, lines = [], {r: [] for r in REPORTS}, {}
    for season in range(first_season, last_season + 1):
        s, r, season_lines = fetch_season(season, today, cache, teams)
        schedule += s
        for report, rows in r.items():
            reports[report] += rows
        lines |= season_lines
        logger.info("%d: %d games, %d with lines", season, len(s), len(season_lines))
    games = build_games(schedule, reports, teams, lines, today=today)
    if output_file is not None:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        games.to_csv(output_file, index=False)
    return games
