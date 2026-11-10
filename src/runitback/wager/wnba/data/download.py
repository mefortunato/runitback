"""WNBA games, box scores and betting lines from ESPN.

As the NBA's (:mod:`runitback.wager.nba.data.download`): the schedule and team box
scores come from sportsdataverse's ESPN WNBA release files (one parquet file
per season, refreshed daily), the betting lines from ESPN's odds API, one
request per game, from 2018 on (see :mod:`runitback.wager.espn`). Finished seasons'
files and finished games' lines are cached on disk (``cache_dir``).

Seasons are calendar years (May to October). Team keys are the abbreviation
a franchise plays under in its latest season, from ESPN's team id, so a team
that moved (Detroit -> Tulsa -> Dallas, San Antonio -> Las Vegas) keeps one
history; the name it played under stays in ``team_name`` / ``opp_name``.
All-Star and exhibition teams are dropped.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

import pandas as pd

from ... import espn

logger = logging.getLogger(__name__)

LEAGUE = espn.League(
    sport="basketball",
    league="wnba",
    schedule_url=espn.RELEASES + "/espn_wnba_schedules/wnba_schedule_{season}.parquet",
    team_box_url=espn.RELEASES + "/espn_wnba_team_boxscores/team_box_{season}.parquet",
    # ESPN's season types: 1 preseason is dropped (All-Star games are
    # regular season ones, dropped with their teams)
    season_types={2: "regular", 3: "playoffs"},
    first_lines_season=2018,
)

# Teams with fewer games in a season are All-Star or exhibition teams.
MIN_GAMES = 5

# Longest rest counted, in days (season openers get it too).
MAX_REST = 10

# Odds lookahead: upcoming games starting within this many days get lines.
LINES_DAYS = 3


def season_of(day: datetime.date) -> int:
    return day.year


def season_complete(season: int, today: datetime.datetime) -> bool:
    """Whether nothing in ``season`` can change (the Finals end in October)."""
    return today.date() > datetime.date(season, 11, 30)


def team_keys(rows: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """``(team, opp_team)``: each franchise's latest abbreviation, by ESPN id."""
    names = espn.latest_names(rows, "abbreviation")
    return rows["id"].map(names), rows["opp_id"].map(names)


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
        season_types=LEAGUE.season_types,
        team_keys=team_keys,
        max_rest=MAX_REST,
        keep=espn.keep_league_teams(MIN_GAMES),
    )


def download(
    output_file: str | None = None,
    first_season: int = 2003,
    last_season: int | None = None,
    cache_dir: str | None = "data/wnba/espn_cache",
) -> pd.DataFrame:
    """Download every season's games, box scores and lines.

    Args:
        output_file: Also write the games table here.
        first_season / last_season: Seasons to download (default: 2003, the
            first with box scores, through the current season). ESPN's
            betting lines start in 2018.
        cache_dir: Reuse finished seasons' files and finished games' lines
            from here (``None``: always download). A full download makes
            about 2,500 odds requests; a daily refresh a few.
    """
    today = datetime.datetime.now(datetime.timezone.utc)
    last_season = last_season or season_of(today.date())
    cache = espn.Cache(cache_dir)
    schedules, boxes, lines = [], [], {}
    for season in range(first_season, last_season + 1):
        schedule, box, season_lines = espn.basketball_season(
            LEAGUE, season, season_complete(season, today), today, cache, LINES_DAYS
        )
        schedules.append(schedule)
        boxes.append(box)
        lines |= season_lines
        logger.info("%d: %d games, %d with lines", season, len(schedule),
                    len(season_lines))  # fmt: skip
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
