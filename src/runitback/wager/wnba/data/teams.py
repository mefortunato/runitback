import httpx
import pandas as pd

from ... import espn

STANDINGS = "https://site.api.espn.com/apis/v2/sports/basketball/wnba/standings"


def conferences(standings: dict) -> list[dict]:
    """ESPN's standings -> each conference (``East`` / ``West``) with its
    teams, as :func:`espn.teams_frame` takes them."""
    return [
        {
            "shortName": (child.get("name") or "").split()[0][:4],
            "teams": [
                {"id": entry["team"]["id"]}
                for entry in (child.get("standings") or {}).get("entries") or []
            ],
        }
        for child in standings.get("children") or []
    ]


def teams_frame(teams: list[dict], groups: list[dict]) -> pd.DataFrame:
    """ESPN ``/teams`` -> web team metadata; keys are ESPN's (current)
    abbreviations, as in the games table."""
    return espn.teams_frame(
        teams, groups, key=lambda team: team["abbreviation"], nested=False
    )


def load_teams() -> pd.DataFrame:
    """The current teams' names, colors, logos and conferences from ESPN."""
    teams = espn.site_json("basketball", "wnba", "teams")
    standings = httpx.get(STANDINGS, timeout=30)
    standings.raise_for_status()
    return teams_frame(
        teams["sports"][0]["leagues"][0]["teams"], conferences(standings.json())
    )
