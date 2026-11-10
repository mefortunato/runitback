import datetime

import pandas as pd

from . import download
from .download import FCS, season_of

# The pooled FCS opponent (see download.FCS).
FCS_TEAM = {
    "team": FCS,
    "name": "FCS opponent",
    "nickname": "FCS",
    "abbr": "FCS",
    "color": "#6b7280",
    "color2": "#d1d5db",
    "logo": None,
    "conference": FCS,
    "division": None,
}


def _color(value) -> str | None:
    if not isinstance(value, str) or not value.strip("#"):
        return None
    return value if value.startswith("#") else f"#{value}"


def _text(value) -> str | None:
    return value if isinstance(value, str) and value else None


def teams_frame(info: pd.DataFrame) -> pd.DataFrame:
    """sportsdataverse ``cfb_team_info`` -> web team metadata of the FBS
    teams, plus the pooled FCS team.

    Keys are school names, as in the games table. A conference's division
    (where it still has them) is shown as ``"<conference> <division>"``.
    """
    rows = []
    if "classification" in info:
        info = info[info["classification"].eq("fbs")]
    for team in info.to_dict("records"):
        conference = _text(team.get("conference"))
        division = _text(team.get("division"))
        rows.append(
            {
                "team": team["school"],
                "name": team["school"],
                "nickname": _text(team.get("mascot")) or team["school"],
                "abbr": _text(team.get("abbreviation")) or team["school"][:4].upper(),
                "color": _color(team.get("color")),
                "color2": _color(team.get("alt_color")),
                "logo": _text(team.get("logo")),
                "conference": conference,
                "division": f"{conference} {division}"
                if conference and division
                else None,
            }
        )
    rows.append(FCS_TEAM)
    return pd.DataFrame(rows).drop_duplicates("team").reset_index(drop=True)


def load_teams(season: int | None = None) -> pd.DataFrame:
    """This season's FBS teams (name, colors, logo, conference) from
    sportsdataverse (last season's until this season's file is out)."""
    today = datetime.datetime.now(datetime.timezone.utc).date()
    season = season or season_of(today)
    cache = download.Cache(None)
    for year in (season, season - 1):
        url = download.TEAM_INFO_URL.format(season=year)
        info = download.season_file(url, "", final=False, cache=cache)
        if len(info):
            return teams_frame(info)
    raise LookupError(f"no college football team info for {season}")
