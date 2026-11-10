"""College basketball games, box scores and betting lines from ESPN.

Shared by men's (``cbb``) and women's (``wcbb``) college basketball, like the
NBA's :mod:`~runitback.wager.nba.data.download`: the schedule and team box scores
come from sportsdataverse's ESPN release files (one parquet file per season,
refreshed daily), the betting lines from ESPN's odds API, one request per
game (see :mod:`runitback.wager.espn`), except the men's finished games', which
come from CollegeBasketballData by the season when ``CBBD_API_KEY`` is set
(see :mod:`.cbbd`). Finished seasons' files and finished games' lines are
cached on disk (``cache_dir``).

Only Division I teams have a team state of their own. ESPN lists every
opponent of a Division I team, and a school from Division II, III or the
NAIA plays one or two of these games a year, so they are pooled into one
team, :data:`NON_D1` (as college football pools FCS teams); the school's own
name stays in ``team_name`` / ``opp_name``. A Division I team is one that
ESPN files under a conference.

Team keys are the school (ESPN's team "location": ``Duke``, ``Miami (OH)``)
it plays under in its latest season, from ESPN's team id, so a school that
changed its name keeps one history.
"""

from __future__ import annotations

import datetime
import functools
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ... import espn
from . import cbbd

logger = logging.getLogger(__name__)

# Team key of every non-Division I opponent.
NON_D1 = "Non-D1"

# ESPN's season types: 1 preseason (exhibitions) is dropped.
SEASON_TYPES = {2: "regular", 3: "postseason"}

# ESPN's tournament id of the NCAA tournament (men's and women's).
NCAA_TOURNAMENT = 22

# Longest rest counted, in days (season openers get it too).
MAX_REST = 10

# Odds lookahead: upcoming games starting within this many days get lines.
LINES_DAYS = 3


@dataclass(frozen=True)
class College:
    """One of the college basketball leagues on ESPN."""

    name: str
    league: espn.League
    first_season: int


MEN = College(
    name="cbb",
    league=espn.League(
        sport="basketball",
        league="mens-college-basketball",
        schedule_url=espn.RELEASES
        + "/espn_mens_college_basketball_schedules/mbb_schedule_{season}.parquet",
        team_box_url=espn.RELEASES
        + "/espn_mens_college_basketball_team_boxscores/team_box_{season}.parquet",
        season_types=SEASON_TYPES,
        # 2012-13 has a consensus line; every major book from 2013-14
        first_lines_season=2013,
    ),
    # sportsdataverse's box scores start in 2002-03
    first_season=2003,
)

WOMEN = College(
    name="wcbb",
    league=espn.League(
        sport="basketball",
        league="womens-college-basketball",
        schedule_url=espn.RELEASES
        + "/espn_womens_college_basketball_schedules/wbb_schedule_{season}.parquet",
        team_box_url=espn.RELEASES
        + "/espn_womens_college_basketball_team_boxscores/team_box_{season}.parquet",
        season_types=SEASON_TYPES,
        # books' lines on women's games start in 2021-22
        first_lines_season=2022,
    ),
    first_season=2006,
)


def season_of(day: datetime.date) -> int:
    """The season a date belongs to, by its ending year (2025-26 is 2026)."""
    return day.year + 1 if day.month >= 8 else day.year


def season_complete(season: int, today: datetime.datetime) -> bool:
    """Whether nothing in ``season`` can change (the Final Four is in April)."""
    return today.date() > datetime.date(season, 5, 31)


def division_one(games: pd.DataFrame) -> pd.Series:
    """Games between two Division I teams (both filed under a conference)."""
    return games["home_conference_id"].notna() & games["away_conference_id"].notna()


def college_keys(rows: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """``(team, opp_team)``: each Division I team's latest school name, by
    ESPN id (:func:`~runitback.wager.espn.latest_names`); :data:`NON_D1` for the
    rest."""
    names = espn.latest_names(rows[rows["conference_id"].notna()], "school")

    def keys(ids: pd.Series, conference: pd.Series) -> pd.Series:
        return ids.map(names).where(conference.notna(), NON_D1).fillna(NON_D1)

    return (
        keys(rows["id"], rows["conference_id"]),
        keys(rows["opp_id"], rows["opp_conference_id"]),
    )


def game_types(rows: pd.DataFrame) -> pd.Series:
    """``regular``, ``postseason`` (NIT, CBI, ...) or ``ncaa`` (the NCAA
    tournament)."""
    ncaa = pd.to_numeric(rows["tournament_id"], errors="coerce").eq(NCAA_TOURNAMENT)
    return rows["game_type"].where(~(ncaa & rows["game_type"].eq("postseason")), "ncaa")


def build_games(
    schedule: pd.DataFrame,
    team_box: pd.DataFrame,
    lines: dict[str, dict] | None = None,
    today: datetime.datetime | None = None,
) -> pd.DataFrame:
    """One row per team per game from ESPN's schedule, box scores and lines.

    Games without a Division I team are dropped (ESPN lists none, but a
    stray one would have no team state at all); non-Division I sides are
    pooled into :data:`NON_D1`. Past games that were never completed are
    dropped; future ones are kept as upcoming games.
    """
    rows = espn.build_basketball_games(
        schedule,
        team_box,
        lines,
        today or datetime.datetime.now(datetime.timezone.utc),
        season_types=SEASON_TYPES,
        team_keys=college_keys,
        max_rest=MAX_REST,
        keep=lambda g: (
            g["home_conference_id"].notna() | g["away_conference_id"].notna()
        ),
        extra=("conference_competition", "tournament_id"),
        side_extra=("conference_id",),
    )
    if rows.empty:
        return rows
    rows["game_type"] = game_types(rows)
    rows["conference_game"] = (
        rows.pop("conference_competition").fillna(False).astype(bool).astype(int)
    )
    for prefix in ("", "opp_"):
        conference = rows.pop(f"{prefix}conference_id")
        rows[f"{prefix}conference"] = (
            pd.to_numeric(conference, errors="coerce")
            .map(lambda c: NON_D1 if np.isnan(c) else str(int(c)))
            .astype(str)
        )
    return rows.drop(columns=["tournament_id"])


def fetch_season(
    college: College,
    season: int,
    today: datetime.datetime,
    cache: espn.Cache,
    lines_days: int = LINES_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """``(schedule, team box scores, lines)`` of one season; lines only of
    games between two Division I teams. Men's finished games' lines come
    from CollegeBasketballData when ``CBBD_API_KEY`` is set (see
    :mod:`.cbbd`); upcoming games', and finished ones CBBD has none for,
    from ESPN."""
    return espn.basketball_season(
        college.league,
        season,
        season_complete(season, today),
        today,
        cache,
        lines_days,
        wants_lines=division_one,
        bulk_lines=bulk_lines(college),
    )


@functools.cache
def _no_key() -> None:
    logger.warning(
        "%s not set: men's finished games' lines from ESPN, one request a game",
        cbbd.KEY_ENV,
    )


def bulk_lines(college: College):
    """CBBD's lines of finished games for the men with a key, else ``None``
    (every game's lines from ESPN). Games CBBD has no lines for still come
    from ESPN."""
    key = cbbd.api_key()
    if college is not MEN:
        return None
    if key is None:
        _no_key()
        return None
    return lambda season, games: cbbd.season_lines(season, games, key)


def download_college(
    college: College,
    output_file: str | None = None,
    first_season: int | None = None,
    last_season: int | None = None,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    today = datetime.datetime.now(datetime.timezone.utc)
    last_season = last_season or season_of(today.date())
    cache = espn.Cache(cache_dir)
    schedules, boxes, lines = [], [], {}
    for season in range(first_season or college.first_season, last_season + 1):
        schedule, box, season_lines = fetch_season(college, season, today, cache)
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


def download(
    output_file: str | None = None,
    first_season: int = MEN.first_season,
    last_season: int | None = None,
    cache_dir: str | None = "data/cbb/espn_cache",
) -> pd.DataFrame:
    """Download every season's men's games, box scores and lines.

    Args:
        output_file: Also write the games table here.
        first_season / last_season: Seasons to download, by ending year
            (default: 2002-03 through the current season). ESPN's betting
            lines start in 2012-13.
        cache_dir: Reuse finished seasons' files and finished games' lines
            from here (``None``: always download). A full download makes
            about 5,000 odds requests a season from ESPN, or about a dozen
            from CollegeBasketballData with ``CBBD_API_KEY``; a daily
            refresh a few hundred (ESPN, upcoming games).
    """
    return download_college(MEN, output_file, first_season, last_season, cache_dir)
