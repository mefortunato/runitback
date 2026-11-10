"""NBA games, box scores and betting lines from ESPN.

The schedule and team box scores come from sportsdataverse's ESPN NBA
release files (https://github.com/sportsdataverse/sportsdataverse-data, one
parquet file per season, refreshed daily around 11:45 UTC with the previous
night's games; see :mod:`runitback.wager.espn`). The schedule is the source of
truth for which games exist, so upcoming games are kept with a missing
result; postponed games are dropped (ESPN gives the makeup game a new id).

Betting lines come from ESPN's odds API, one request per game: every
book ESPN lists, from the 2012-13 season on. Each market comes from the most
preferred book that has it (:data:`PROVIDERS`). Lines of finished games
never change, so they are cached per season on disk (``cache_dir``) with the
finished seasons' parquet files; a daily refresh only requests the newest
games and the next few days' upcoming ones (ESPN posts lines about a day
ahead).

Team keys are the franchise's current abbreviation (``GSW``, ``NYK``, ...,
as The Odds API's ``odds_map`` uses), from ESPN's franchise id, so a team
that moved or was renamed (Seattle -> Oklahoma City, New Jersey ->
Brooklyn) keeps one history; the name it played under stays in
``team_name`` / ``opp_name``.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

import pandas as pd

from ... import espn
from ...espn import BOX_COLUMNS, Cache, parse_odds  # noqa: F401

logger = logging.getLogger(__name__)

SCHEDULE_URL = espn.RELEASES + "/espn_nba_schedules/nba_schedule_{season}.parquet"
TEAM_BOX_URL = espn.RELEASES + "/espn_nba_team_boxscores/team_box_{season}.parquet"

# ESPN abbreviations that differ from the standard (and The Odds API's).
ABBREVIATIONS = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA",
                 "WSH": "WAS"}  # fmt: skip

# ESPN's franchise ids; higher ids are All-Star and exhibition teams.
FRANCHISE_IDS = range(1, 31)

# ESPN's season types: 1 preseason and 4 All-Star are dropped.
SEASON_TYPES = {2: "regular", 3: "playoffs", 5: "play_in"}

# Longest rest counted, in days (season openers get it too).
MAX_REST = 10

# Odds lookahead: upcoming games starting within this many days get lines.
LINES_DAYS = 3

# The first season (by ending year) with betting lines.
FIRST_LINES_SEASON = 2013


def season_of(day: datetime.date) -> int:
    """The season a date belongs to, by its ending year (2025-26 is 2026)."""
    return day.year + 1 if day.month >= 8 else day.year


def season_complete(season: int, today: datetime.datetime) -> bool:
    """Whether nothing in ``season`` can change (the Finals end in June)."""
    return today.date() > datetime.date(season, 7, 31)


def team_key(team_id, abbreviation) -> str:
    """A franchise's games-table key: its current standard abbreviation."""
    return CURRENT.get(int(team_id)) or ABBREVIATIONS.get(abbreviation, abbreviation)


# ESPN franchise id -> current abbreviation (fixed since the 2014-15 renames).
CURRENT = {
    1: "ATL", 2: "BOS", 3: "NOP", 4: "CHI", 5: "CLE", 6: "DAL", 7: "DEN",
    8: "DET", 9: "GSW", 10: "HOU", 11: "IND", 12: "LAC", 13: "LAL", 14: "MIA",
    15: "MIL", 16: "MIN", 17: "BKN", 18: "NYK", 19: "ORL", 20: "PHI", 21: "PHX",
    22: "POR", 23: "SAC", 24: "SAS", 25: "OKC", 26: "UTA", 27: "WAS", 28: "TOR",
    29: "MEM", 30: "CHA",
}  # fmt: skip


def fetch_odds(game_id: str) -> dict:
    return espn.fetch_odds("basketball", "nba", game_id)


def season_lines(
    season: int, final_ids: list[str], upcoming_ids: list[str], cache: Cache
) -> dict[str, dict]:
    """``{game_id: lines}`` for a season's finished games (cached) and the
    given upcoming ones (fetched every time)."""
    return espn.season_lines(
        season, final_ids, upcoming_ids, cache, lambda g: fetch_odds(g)
    )


def _franchises(games: pd.DataFrame) -> pd.Series:
    return games["home_id"].isin(FRANCHISE_IDS) & games["away_id"].isin(FRANCHISE_IDS)


def _team_keys(rows: pd.DataFrame) -> tuple[list[str], list[str]]:
    return (
        [team_key(i, a) for i, a in zip(rows["id"], rows["abbreviation"])],
        [team_key(i, a) for i, a in zip(rows["opp_id"], rows["opp_abbreviation"])],
    )


def build_games(
    schedule: pd.DataFrame,
    team_box: pd.DataFrame,
    lines: dict[str, dict] | None = None,
    today: datetime.datetime | None = None,
) -> pd.DataFrame:
    """One row per team per game from ESPN's schedule, box scores and lines.

    Past games that were never completed (postponed) are dropped; future
    ones are kept as upcoming games with missing results and stats.
    """
    return espn.build_basketball_games(
        schedule,
        team_box,
        lines,
        today or datetime.datetime.now(datetime.timezone.utc),
        season_types=SEASON_TYPES,
        team_keys=_team_keys,
        max_rest=MAX_REST,
        keep=_franchises,
    )


# ----------------------------------------------------------------- download
def fetch_season(
    season: int,
    today: datetime.datetime,
    cache: Cache,
    lines_days: int = LINES_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """``(schedule, team box scores, lines)`` of one season."""
    final = season_complete(season, today)
    schedule = espn.season_file(
        SCHEDULE_URL.format(season=season), f"schedule_{season}.parquet", final, cache
    )
    box = espn.season_file(
        TEAM_BOX_URL.format(season=season), f"team_box_{season}.parquet", final, cache
    )
    lines = {}
    games = espn.schedule_frame(schedule, SEASON_TYPES, _franchises)
    if season >= FIRST_LINES_SEASON and len(games):
        finished, soon = espn.lines_to_fetch(games, today, lines_days)
        lines = season_lines(season, finished, soon, cache)
    return schedule, box, lines


def download(
    output_file: str | None = None,
    first_season: int = 2004,
    last_season: int | None = None,
    cache_dir: str | None = "data/nba/espn_cache",
) -> pd.DataFrame:
    """Download every season's games, box scores and lines.

    Args:
        output_file: Also write the games table here.
        first_season / last_season: Seasons to download, by ending year
            (default: 2003-04 through the current season). ESPN's betting
            lines start in 2012-13.
        cache_dir: Reuse finished seasons' files and finished games' lines
            from here (``None``: always download). A full download makes
            about 18,000 odds requests; a daily refresh a few dozen.
    """
    today = datetime.datetime.now(datetime.timezone.utc)
    last_season = last_season or season_of(today.date())
    cache = Cache(cache_dir)
    schedules, boxes, lines = [], [], {}
    for season in range(first_season, last_season + 1):
        schedule, box, season_lines_ = fetch_season(season, today, cache)
        schedules.append(schedule)
        boxes.append(box)
        lines |= season_lines_
        logger.info("%d: %d games, %d with lines", season, len(schedule),
                    len(season_lines_))  # fmt: skip
    games = build_games(
        pd.concat(schedules, ignore_index=True),
        pd.concat(boxes, ignore_index=True),
        lines,
        today=today,
    )
    if output_file is not None:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        games.to_csv(output_file, index=False)
    return games
