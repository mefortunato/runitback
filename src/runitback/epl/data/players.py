"""Premier League player appearances from the Fantasy Premier League data.

One row per player per game he played (``minutes > 0``), with the game's id
in the games table (:mod:`runitback.wager.epl.data.download`) (``<season>_<home>_<away>``), the player's
team that game and the Fantasy Premier League (FPL) match statistics: minutes,
starts, goals, assists, clean sheets, cards, saves, bonus and BPS, the
influence / creativity / threat indexes and, from 2022-23, expected goals,
assists, goal involvements and goals conceded.

Sources:

- Finished seasons come from the community archive
  `vaastav/Fantasy-Premier-League
  <https://github.com/vaastav/Fantasy-Premier-League>`_ (``gws/merged_gw.csv``,
  ``players_raw.csv``, ``teams.csv``, ``fixtures.csv`` per season), the FPL
  API's own responses saved after every gameweek. The archive's files for
  2016-17 to 2018-19 lack ``teams.csv`` (and 2016-17 / 2017-18 ``fixtures.csv``) and
  its player files lack the team before 2020-21, so the first season is
  2019-20, whose team ids the fixtures give.
- The current season comes from the live FPL API
  (``fantasy.premierleague.com/api``: ``bootstrap-static``, ``fixtures`` and
  one ``element-summary`` per player who has played), which the archive
  trails by days to weeks.

A player's team in a game is the fixture's home or away team (his ``was_home``
flag), not his current club, so transfers are right; ``player_id`` is the
player's ``code``, which stays the same across seasons (the FPL ``element`` id
does not). ``position`` is the player's FPL position that season (``GKP`` /
``DEF`` / ``MID`` / ``FWD``). A fixture postponed and replayed appears twice in
the FPL data (a no-minutes row at the original gameweek); the played row is
kept. Seasons are keyed by ending year, as in the games table.
"""

from __future__ import annotations

import datetime
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ...wager import espn
from ...wager.epl.data.download import season_complete, season_of
from ...wager.epl.data.teams import CLUBS

logger = logging.getLogger(__name__)

ARCHIVE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
API = "https://fantasy.premierleague.com/api"

# The first season the archive can resolve to a game (see the module docs).
FIRST_SEASON = 2020

POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# FPL history column -> table column; columns an older season lacks are NaN.
STATS = {
    "minutes": "minutes",
    "starts": "starts",
    "goals_scored": "goals",
    "assists": "assists",
    "clean_sheets": "clean_sheets",
    "goals_conceded": "goals_conceded",
    "own_goals": "own_goals",
    "penalties_saved": "penalties_saved",
    "penalties_missed": "penalties_missed",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "saves": "saves",
    "bonus": "bonus",
    "bps": "bps",
    "influence": "influence",
    "creativity": "creativity",
    "threat": "threat",
    "ict_index": "ict_index",
    "expected_goals": "xg",
    "expected_assists": "xa",
    "expected_goal_involvements": "xgi",
    "expected_goals_conceded": "xgc",
    "total_points": "fpl_points",
    "selected": "selected",
}
STAT_COLUMNS = list(STATS.values())
KEY_COLUMNS = ["season", "game_id", "gameweek", "kickoff_time", "team", "opp_team",
               "location", "player_id", "player", "position"]  # fmt: skip
COLUMNS = KEY_COLUMNS + STAT_COLUMNS + ["price"]


def archive_season_name(season: int) -> str:
    """``2026`` -> ``"2025-26"``, the archive's directory name."""
    return f"{season - 1}-{season % 100:02d}"


def fix_text(value):
    """Undo the UTF-8-read-as-latin-1 text of some archive files."""
    if isinstance(value, str) and ("Ã" in value or "Â" in value):
        try:
            return value.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return value
    return value


def read_csv(data: bytes, **kwargs) -> pd.DataFrame:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    return pd.read_csv(
        io.StringIO(text), on_bad_lines="skip", engine="python", **kwargs
    )


# ------------------------------------------------------------------- build
def team_short_names(teams: pd.DataFrame) -> dict[int, str]:
    """FPL team id -> games-table team key (the FPL short name is the code)."""
    codes = dict(zip(teams["id"].astype(int), teams["short_name"]))
    unknown = sorted(set(codes.values()) - {code for code, _ in CLUBS.values()})
    if unknown:
        raise ValueError(
            f"FPL team codes {unknown} are not in runitback.wager.epl.data.teams.CLUBS"
        )
    return codes


def build_players(
    history: pd.DataFrame,
    elements: pd.DataFrame,
    teams: pd.DataFrame,
    fixtures: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """Player appearances of one season (see the module docs).

    Args:
        history: One row per player per fixture (``element``, ``fixture``,
            ``round``, ``was_home``, ``kickoff_time``, the :data:`STATS`
            columns and ``value``): the archive's ``merged_gw.csv`` or the
            live ``element-summary`` histories.
        elements: ``id``, ``code``, ``first_name``, ``second_name`` and
            ``element_type`` of every player.
        teams: ``id`` and ``short_name`` of the season's teams.
        fixtures: ``id``, ``team_h`` and ``team_a`` of the season's fixtures.
    """
    codes = team_short_names(teams)
    # the archive's own name / position / team columns would shadow the merged ones
    needed = [
        "element",
        "fixture",
        "round",
        "was_home",
        "kickoff_time",
        "value",
        *STATS,
    ]
    history = history[[c for c in needed if c in history]].copy()
    for col in ("element", "fixture", "round", "minutes"):
        history[col] = pd.to_numeric(history[col], errors="coerce")
    history = history.dropna(subset=["element", "fixture", "round"])
    history = history[history["minutes"] > 0]
    history["was_home"] = history["was_home"].astype(str).str.lower().eq("true")
    # a postponed-and-replayed game, or a transfer's repeated line: one row
    history = history.sort_values("round", kind="stable")
    history = history.drop_duplicates(["element", "fixture"], keep="last")

    fix = fixtures[["id", "team_h", "team_a"]].rename(columns={"id": "fixture"})
    rows = history.merge(fix, on="fixture", how="left")
    lost = rows["team_h"].isna()
    if lost.any():
        logger.warning("%d: dropping %d rows of unknown fixtures", season, lost.sum())
        rows = rows[~lost]
    home, away = rows["team_h"].map(codes), rows["team_a"].map(codes)
    out = pd.DataFrame(index=rows.index)
    out["season"] = season
    out["game_id"] = f"{season}_" + home + "_" + away
    out["gameweek"] = rows["round"].astype(int)
    out["kickoff_time"] = rows["kickoff_time"]
    out["team"] = home.where(rows["was_home"], away)
    out["opp_team"] = away.where(rows["was_home"], home)
    out["location"] = np.where(rows["was_home"], 1, -1)

    info = elements.rename(columns={"id": "element"}).copy()
    info["player"] = (
        info["first_name"].map(fix_text) + " " + info["second_name"].map(fix_text)
    )
    info["position"] = pd.to_numeric(info["element_type"]).map(POSITIONS)
    info = info.rename(columns={"code": "player_id"})
    rows = rows.merge(info[["element", "player_id", "player", "position"]],
                      on="element", how="left")  # fmt: skip
    rows.index = out.index
    unknown = rows["player_id"].isna()
    if unknown.any():
        logger.warning("%d: dropping %d rows of unknown players", season, unknown.sum())
    out["player_id"] = rows["player_id"]
    out["player"], out["position"] = rows["player"], rows["position"]
    for source, name in STATS.items():
        col = rows[source] if source in rows else pd.Series(np.nan, index=rows.index)
        out[name] = pd.to_numeric(col, errors="coerce")
    out["price"] = pd.to_numeric(rows.get("value"), errors="coerce") / 10
    out = out[~unknown]
    out["player_id"] = out["player_id"].astype(int)
    return (
        out[COLUMNS]
        .sort_values(["gameweek", "game_id", "team", "player_id"])
        .reset_index(drop=True)
    )


# ----------------------------------------------------------------- sources
def fetch(url: str, name: str | None, cache: espn.Cache) -> bytes:
    data = cache.read_bytes(name) if name else None
    if data is None:
        response = espn._get(url)
        response.raise_for_status()
        data = response.content
        if name:
            cache.write_bytes(name, data)
    return data


def archive_players(season: int, cache: espn.Cache) -> pd.DataFrame:
    """A finished season from the archive (cached)."""
    folder = archive_season_name(season)

    def table(path: str) -> pd.DataFrame:
        name = f"{folder}_{path.replace('/', '_')}"
        return read_csv(fetch(f"{ARCHIVE}/{folder}/{path}", name, cache))

    return build_players(
        table("gws/merged_gw.csv"), table("players_raw.csv"),
        table("teams.csv"), table("fixtures.csv"), season,
    )  # fmt: skip


def live_players(season: int, workers: int = 8) -> pd.DataFrame:
    """The current season from the live FPL API: one request for the
    players, one for the fixtures and one per player who has played."""
    bootstrap = espn._get(f"{API}/bootstrap-static/").json()
    fixtures = pd.DataFrame(espn._get(f"{API}/fixtures/").json())
    elements = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])
    played = elements.loc[elements["minutes"] > 0, "id"].tolist()

    def summary(element: int) -> list[dict]:
        response = espn._get(f"{API}/element-summary/{element}/")
        response.raise_for_status()
        return response.json()["history"]

    with ThreadPoolExecutor(workers) as pool:
        history = pd.DataFrame(
            [row for rows in pool.map(summary, played) for row in rows]
        )
    if history.empty:
        return pd.DataFrame(columns=COLUMNS)
    return build_players(history, elements, teams, fixtures, season)


# ------------------------------------------------------------------- checks
def check(players: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Per season: games without any player rows, rows whose game is not in
    ``games``, and the share of team-games whose players' goals (plus the
    opponent's own goals) differ from the game's score."""
    played = games[games["result"].notna()]
    out = []
    for season, p in players.groupby("season"):
        g = played[played["season"] == season]
        scored = (
            p.groupby(["game_id", "team"])
            .agg(g=("goals", "sum"), og=("own_goals", "sum"))
            .reset_index()
        )
        og = scored.rename(columns={"team": "opp_team", "og": "opp_og"})[
            ["game_id", "opp_team", "opp_og"]
        ]
        both = g.merge(
            scored[["game_id", "team", "g"]], on=["game_id", "team"], how="left"
        )
        both = both.merge(og, on=["game_id", "opp_team"], how="left")
        implied = both["g"].fillna(0) + both["opp_og"].fillna(0)
        out.append({
            "season": season,
            "games_without_players": int(g["game_id"].nunique() - p["game_id"].nunique()
                                         + len(set(p["game_id"]) - set(g["game_id"]))),
            "rows_without_game": int((~p["game_id"].isin(games["game_id"])).sum()),
            "goal_mismatch": float((implied != both["goals"]).mean()),
        })  # fmt: skip
    return pd.DataFrame(out)


def download(
    output_file: str | None = None,
    first_season: int = FIRST_SEASON,
    last_season: int | None = None,
    cache_dir: str | None = "data/epl/fpl_cache",
    games: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Download every season's player appearances.

    Args:
        output_file: Also write the table here.
        first_season / last_season: Seasons by ending year (default
            2019-20 through the current one).
        cache_dir: Reuse finished seasons' archive files from here.
        games: The games table of :func:`runitback.wager.epl.data.download.download`;
            given, every season is checked against it (see :func:`check`) and
            mismatches logged.
    """
    today = datetime.datetime.now(datetime.timezone.utc)
    current = season_of(today.date())
    last_season = last_season or current
    cache = espn.Cache(cache_dir)
    frames = []
    for season in range(first_season, last_season + 1):
        if season_complete(season, today):
            frame = archive_players(season, cache)
        else:
            frame = live_players(season)
        logger.info("%d: %d appearances", season, len(frame))
        frames.append(frame)
    players = pd.concat(frames, ignore_index=True)
    if games is not None:
        report = check(players, games)
        logger.info("checked against the games table:\n%s", report.to_string())
    if output_file is not None:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        players.to_csv(output_file, index=False)
    return players
