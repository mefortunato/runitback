import dataclasses

import pandas as pd

from ...wnba.data.teams import load_teams
from ..views import bracket_playoffs
from .nba import VIEW as NBA


def finals_wins(season: int) -> int:
    """Wins that take the WNBA Finals: best of 3 until 2004, of 5 until
    2024, of 7 since 2025."""
    return 2 if season <= 2004 else 3 if season <= 2024 else 4


def playoffs(games: pd.DataFrame) -> dict[int, dict[str, str]]:
    """WNBA champion and runner-up, and the semifinal (conference finals
    until 2015) losers, from the bracket."""
    return bracket_playoffs(games, games["game_type"].eq("playoffs"), finals_wins)


# The NBA's box score and pages; seasons are calendar years.
VIEW = dataclasses.replace(
    NBA,
    sport="wnba",
    label="WNBA",
    load_teams=load_teams,
    playoffs=playoffs,
    playoff_labels={
        "champion": "Won WNBA Finals",
        "runner_up": "Lost WNBA Finals",
        "semifinal": "Lost in the semifinals",
    },
    split_season=False,
)
