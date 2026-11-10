import pandas as pd

from ...cbb.data.teams import load_teams
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
    """NCAA champion and runner-up, and the Final Four losers (from the
    NCAA tournament's bracket: its last game is the final)."""
    return bracket_playoffs(games, games["game_type"].eq("ncaa"), lambda s: 1)


STATS = (
    Stat("points", "Points scored", column("points")),
    Stat("points_allowed", "Points allowed", column("opp_points"), 1, False),
    Stat("fg_pct", "Field goal %", ratio("fgm", "fga")),
    Stat("fg3_pct", "Three-point %", ratio("fg3m", "fg3a")),
    Stat("fg3a", "Three-point attempts", column("fg3a")),
    Stat("ft_pct", "Free throw %", ratio("ftm", "fta")),
    Stat(
        "opp_fg_pct", "Opponent field goal %", ratio("opp_fgm", "opp_fga"), 1, False
    ),  # fmt: skip
    Stat("rebounds", "Rebounds", total("oreb", "dreb")),
    Stat("oreb", "Offensive rebounds", column("oreb")),
    Stat("assists", "Assists", column("ast")),
    Stat("turnovers", "Turnovers", column("tov"), 1, False),
    Stat("steals", "Steals", column("stl")),
)

PLAYOFF_LABELS = {
    "champion": "Won national championship",
    "runner_up": "Lost national championship",
    "semifinal": "Lost in the Final Four",
}

VIEW = SportView(
    sport="cbb",
    label="CBB",
    date_label="Day",
    load_teams=load_teams,
    score=("points", "opp_points"),
    kickoff=eastern_kickoff,
    playoffs=playoffs,
    playoff_labels=PLAYOFF_LABELS,
    date_name=calendar_date,
    upcoming_dates=7,
    recent_dates=10,
    split_season=True,
    recent_games=8,
    # weekly: ~360 teams make ~130,000 pairs to rate per game day
    ranking_every=7,
    stats=STATS,
)
