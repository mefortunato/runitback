import pandas as pd

from ... import espn
from .download import ESPN_ABBREVIATIONS


def teams_frame(teams: list[dict], groups: list[dict]) -> pd.DataFrame:
    """ESPN ``/teams`` and ``/groups`` -> web team metadata: the conference
    is the league (``AL`` / ``NL``) and the division its name."""
    frame = espn.teams_frame(
        teams,
        groups,
        key=lambda t: ESPN_ABBREVIATIONS.get(t["abbreviation"], t["abbreviation"]),
    )
    frame["division"] = frame["division"].str.replace(
        r"^(American|National) League", lambda m: m[1][0] + "L", regex=True
    )
    return frame


def load_teams() -> pd.DataFrame:
    """The 30 teams' names, colors, logos, leagues and divisions (ESPN)."""
    teams = espn.site_json("baseball", "mlb", "teams")
    groups = espn.site_json("baseball", "mlb", "groups")
    return teams_frame(teams["sports"][0]["leagues"][0]["teams"], groups["groups"])
