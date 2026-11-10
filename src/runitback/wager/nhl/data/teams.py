import pandas as pd

from ... import espn
from .download import ESPN_ABBREVIATIONS, MOVES


def team_key(abbreviation: str) -> str:
    """ESPN abbreviation -> games-table key (the NHL's code)."""
    code = ESPN_ABBREVIATIONS.get(abbreviation, abbreviation)
    return MOVES.get(code, code)


def teams_frame(teams: list[dict], groups: list[dict]) -> pd.DataFrame:
    """ESPN ``/teams`` and ``/groups`` -> web team metadata: the conference
    is ``East`` / ``West`` and the division its name."""
    frame = espn.teams_frame(teams, groups, key=lambda t: team_key(t["abbreviation"]))
    frame["division"] = frame["division"].str.removesuffix(" Division")
    return frame


def load_teams() -> pd.DataFrame:
    """The 32 teams' names, colors, logos, conferences and divisions (ESPN)."""
    teams = espn.site_json("hockey", "nhl", "teams")
    groups = espn.site_json("hockey", "nhl", "groups")
    return teams_frame(teams["sports"][0]["leagues"][0]["teams"], groups["groups"])
