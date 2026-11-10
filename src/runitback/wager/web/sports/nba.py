import pandas as pd

from ...nba.data.teams import load_teams
from ..views import (
    SportView,
    Stat,
    bracket_playoffs,
    calendar_date,
    column,
    eastern_kickoff,
    ratio,
    total,
)


def playoffs(games: pd.DataFrame) -> dict[int, dict[str, str]]:
    """NBA champion and runner-up, and the conference finals losers.

    ESPN's playoff notes changed wording over the years (and are blank in
    some), so the rounds come from the bracket: the Finals are the series of
    the season's last playoff game, and each finalist's previous playoff
    opponent lost its conference finals. A season is marked once its
    champion has won four Finals games.
    """
    return bracket_playoffs(games, games["game_type"].eq("playoffs"), lambda s: 4)


kickoff = eastern_kickoff
date_name = calendar_date


VIEW = SportView(
    sport="nba",
    label="NBA",
    date_label="Day",
    load_teams=load_teams,
    score=("points", "opp_points"),
    kickoff=kickoff,
    playoffs=playoffs,
    playoff_labels={
        "champion": "Won NBA Finals",
        "runner_up": "Lost NBA Finals",
        "semifinal": "Lost conference finals",
    },
    date_name=date_name,
    upcoming_dates=7,
    recent_dates=14,
    split_season=True,
    recent_games=8,
    stats=(
        Stat("points", "Points scored", column("points")),
        Stat("points_allowed", "Points allowed", column("opp_points"), 1, False),
        Stat("fg_pct", "Field goal %", ratio("fgm", "fga")),
        Stat("fg3_pct", "Three-point %", ratio("fg3m", "fg3a")),
        Stat("fg3a", "Three-point attempts", column("fg3a")),
        Stat("ft_pct", "Free throw %", ratio("ftm", "fta")),
        Stat("rebounds", "Rebounds", total("oreb", "dreb")),
        Stat("oreb", "Offensive rebounds", column("oreb")),
        Stat("assists", "Assists", column("ast")),
        Stat("turnovers", "Turnovers", column("tov"), 1, False),
        Stat("steals", "Steals", column("stl")),
        Stat("blocks", "Blocks", column("blk")),
    ),
)
