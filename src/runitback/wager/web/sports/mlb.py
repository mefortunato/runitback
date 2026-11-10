import pandas as pd

from ...mlb.data.teams import load_teams
from ..views import (
    SportView,
    Stat,
    bracket_playoffs,
    calendar_date,
    column,
    eastern_kickoff,
    ratio,
)


def playoffs(games: pd.DataFrame) -> dict[int, dict[str, str]]:
    """World Series winner and loser, and the League Championship Series
    losers, from the bracket."""
    return bracket_playoffs(games, games["game_type"].ne("regular"), lambda s: 4)


def on_base_pct(rows: pd.DataFrame) -> pd.Series:
    on = rows["hits"] + rows["walks"] + rows["hit_by_pitch"]
    chances = on - rows["hits"] + rows["at_bats"] + rows["sac_flies"]
    return on / chances.where(chances > 0)


VIEW = SportView(
    sport="mlb",
    label="MLB",
    date_label="Day",
    load_teams=load_teams,
    score=("runs", "opp_runs"),
    kickoff=eastern_kickoff,
    playoffs=playoffs,
    playoff_labels={
        "champion": "Won World Series",
        "runner_up": "Lost World Series",
        "semifinal": "Lost League Championship Series",
    },
    date_name=calendar_date,
    upcoming_dates=7,
    recent_dates=14,
    recent_games=10,
    stats=(
        Stat("runs", "Runs scored", column("runs"), 2),
        Stat("runs_allowed", "Runs allowed", column("opp_runs"), 2, False),
        Stat("hits", "Hits", column("hits")),
        Stat("home_runs", "Home runs", column("home_runs"), 2),
        Stat("obp", "On-base %", on_base_pct, 3),
        Stat("slg", "Slugging %", ratio("total_bases", "at_bats", 1.0), 3),
        Stat("walks", "Walks", column("walks")),
        Stat("strikeouts", "Strikeouts", column("strikeouts"), 1, False),
        Stat("hits_allowed", "Hits allowed", column("opp_hits"), 1, False),
        Stat(
            "home_runs_allowed", "Home runs allowed", column("opp_home_runs"), 2, False
        ),  # fmt: skip
        Stat(
            "pitching_strikeouts", "Strikeouts by pitchers", column("opp_strikeouts")
        ),  # fmt: skip
        Stat("errors", "Errors", column("errors"), 2, False),
    ),
)
