import concurrent.futures
import datetime
import logging

import httpx
import pandas as pd

from ... import espn
from .download import MEN, NON_D1, WOMEN, College, season_of

CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/{league}"

logger = logging.getLogger(__name__)

# The pooled non-Division I opponent (see download.NON_D1).
NON_D1_TEAM = {
    "team": NON_D1,
    "name": "Non-Division I opponent",
    "nickname": "Non-D1",
    "abbr": "N-D1",
    "color": "#6b7280",
    "color2": "#d1d5db",
    "logo": None,
    "conference": NON_D1,
    "division": None,
}


def _core(url: str) -> dict:
    response = espn._get(url, limit=100)
    response.raise_for_status()
    return response.json()


def conferences(
    league: str, season: int, cache: espn.Cache | None = None
) -> list[dict]:
    """This season's Division I conferences from ESPN's core API, each with
    its ``shortName`` and its ``teams`` (ids), as :func:`espn.teams_frame`
    takes them (the site API's ``/groups`` lists only some of them).

    Each fetch is kept in ``cache``. When ESPN refuses (it answers 403 to a
    runner after a burst of odds requests), the cached copy is used, or none:
    teams then have no conference rather than failing the build.
    """
    name = f"conferences_{season}.json"
    try:
        groups = _fetch_conferences(league, season)
    except (httpx.HTTPStatusError, httpx.TransportError) as error:
        cached = cache.read_json(name) if cache is not None else {}
        logger.warning(
            "ESPN conferences of %s %d unavailable (%s); using %s",
            league, season, error, "the cached copy" if cached else "none",
        )  # fmt: skip
        return cached.get("conferences", [])
    if cache is not None:
        cache.write_json(name, {"conferences": groups})
    return groups


def _fetch_conferences(league: str, season: int) -> list[dict]:
    base = f"{CORE.format(league=league)}/seasons/{season}/types/2/groups"
    refs = [item["$ref"] for item in _core(f"{base}/50/children")["items"]]

    def conference(ref: str) -> dict:
        group = _core(ref)
        teams = _core(f"{base}/{group['id']}/teams")["items"]
        ids = [item["$ref"].split("/teams/")[1].split("?")[0] for item in teams]
        return {
            "name": group.get("name"),
            "shortName": group.get("shortName") or group.get("abbreviation"),
            "teams": [{"id": team} for team in ids],
        }

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        return list(pool.map(conference, refs))


def teams_frame(teams: list[dict], groups: list[dict]) -> pd.DataFrame:
    """ESPN's Division I ``/teams`` and :func:`conferences` -> web team
    metadata, plus the pooled non-Division I team. Keys are schools (ESPN's
    team ``location``), as in the games table."""
    frame = espn.teams_frame(
        teams, groups, key=lambda team: team["location"], nested=False, abbr=True
    )
    return pd.concat([frame, pd.DataFrame([NON_D1_TEAM])], ignore_index=True)


def load_college_teams(college: College) -> pd.DataFrame:
    """This season's Division I teams (names, colors, logos, conferences)."""
    today = datetime.datetime.now(datetime.timezone.utc).date()
    league = college.league.league
    teams = espn.site_json("basketball", league, "teams", groups=50, limit=500)
    cache = espn.Cache(f"data/{college.name}/espn_cache")
    return teams_frame(
        teams["sports"][0]["leagues"][0]["teams"],
        conferences(league, season_of(today), cache),
    )


def load_teams() -> pd.DataFrame:
    return load_college_teams(MEN)


def load_women_teams() -> pd.DataFrame:
    return load_college_teams(WOMEN)
