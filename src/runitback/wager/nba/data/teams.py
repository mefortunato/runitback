import pandas as pd

from ... import espn
from .download import team_key


def teams_frame(teams: list[dict], groups: list[dict]) -> pd.DataFrame:
    """ESPN ``/teams`` and ``/groups`` -> web team metadata.

    Keys are the games table's (``team_key``); the conference is ``East`` /
    ``West`` and the division its name (``Atlantic``, ...).
    """
    return espn.teams_frame(
        teams, groups, key=lambda team: team_key(team["id"], team["abbreviation"])
    )


def load_teams() -> pd.DataFrame:
    """The 30 teams' names, colors, logos, conferences and divisions from ESPN."""
    teams = espn.site_json("basketball", "nba", "teams")
    groups = espn.site_json("basketball", "nba", "groups")
    return teams_frame(teams["sports"][0]["leagues"][0]["teams"], groups["groups"])
