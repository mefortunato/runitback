"""Download NHL data, cached under the raw directory.

- **Play-by-play**: sportsdataverse's `nhl_pbp_full` release (NHL API events with the skaters and
  goalies on ice, strength and `xg` merged in). `season` here is the year the season ends
  (2025 = 2024-25), as in the rest of runitback. The release's file names do not follow one rule:
  `play_by_play_2011` holds 2010-11 but `play_by_play_2015` holds 2015-16, `play_by_play_2019`
  repeats 2018-19, and no file holds 2014-15 (checked 2026-11-10). So every file is checked by its
  `season` column and the neighbouring file names are tried; a season no file holds is skipped
  with a warning.
- **Schedule** (`api.nhle.com/stats/rest/en/game`): game dates and teams. The release has no
  `game_date` column before 2022-23.
- **Players** (`api.nhle.com/stats/rest/en/{skater,goalie}/summary`): names and positions
  (C / L / R / D / G) of everyone who played that season. Goalie ids tell the goalie apart from
  the skaters in the shift charts.
- **Shift charts** (`api.nhle.com/stats/rest/en/shiftcharts`): every player's shifts in a game, the
  fallback for games whose on-ice personnel is missing from the release (`events.gap_games`).
  They start with 2010-11.

Files: `pbp_<season>.parquet`, `games_<season>.parquet`, `players_<season>.parquet`,
`shifts/<game_id>.parquet`.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

import pandas as pd

log = logging.getLogger(__name__)
RELEASE_URL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
               "nhl_pbp_full/play_by_play_{n}.parquet")
STATS_URL = "https://api.nhle.com/stats/rest/en/"
# release file name per season where it is not the season itself (checked 2026-11-10); others are
# found by trying the neighbouring names
RELEASE_FILE = {2016: 2015, 2017: 2016, 2018: 2017, 2019: 2018}
GAME_TYPES = {2: "REG", 3: "POST"}
SHIFT = 517  # shift-chart typeCode of a shift (505 is a goal)


def season_code(season: int) -> str:
    """2025 -> '20242025' (the NHL's season id)."""
    return f"{season - 1}{season}"


def release_candidates(season: int) -> list[int]:
    first = RELEASE_FILE.get(season, season)
    return list(dict.fromkeys([first, season, season - 1, season + 1]))


def download_season(season: int, raw_dir: Path, force: bool = False,
                    read: Callable[[str], pd.DataFrame] = pd.read_parquet) -> Path | None:
    """Cache the release's rows for `season` as pbp_<season>.parquet. None if no file holds it."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"pbp_{season}.parquet"
    if path.exists() and not force:
        return path
    want = season_code(season)
    for n in release_candidates(season):
        url = RELEASE_URL.format(n=n)
        try:
            df = read(url)
        except Exception as e:  # missing file (404) or a network error
            log.info("%s: %s", url, e)
            continue
        got = sorted({str(int(s)) if not isinstance(s, str) else s for s in df["season"].dropna().unique()})
        if want in got:
            df = df[df["season"].astype(str).str.replace(r"\.0$", "", regex=True) == want]
            df.reset_index(drop=True).to_parquet(path)
            log.info("season %s: %s (%d rows)", season, url, len(df))
            return path
        log.info("%s holds season %s, not %s", url, got, want)
    log.warning("season %s (%s): no nhl_pbp_full release file holds it; skipped", season, want)
    return None


def fetch_json(url: str, retries: int = 4, pause: float = 0.2) -> dict:
    """GET a JSON document, retrying with exponential backoff."""
    for k in range(retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "runitback"}),
                                        timeout=60) as r:
                out = json.load(r)
            time.sleep(pause)  # stay well under the API's rate limits
            return out
        except Exception as e:
            if k == retries:
                raise
            log.info("%s: %s; retrying", url, e)
            time.sleep(2 ** (k + 1))
    raise AssertionError  # unreachable


def stats_url(endpoint: str, cayenne: str, **params) -> str:
    q = {"cayenneExp": cayenne, **params}
    return STATS_URL + endpoint + "?" + urllib.parse.urlencode(q)


def download_schedule(season: int, raw_dir: Path, force: bool = False, fetch=fetch_json) -> Path:
    """Regular-season and playoff games of `season` (games_<season>.parquet)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"games_{season}.parquet"
    if path.exists() and not force:
        return path
    games = pd.DataFrame(fetch(stats_url("game", f"season={season_code(season)}"))["data"])
    teams = pd.DataFrame(fetch(STATS_URL + "team")["data"]).set_index("id")["triCode"]
    games = games[games["gameType"].isin(GAME_TYPES)]
    out = pd.DataFrame({
        "game_id": games["id"].astype("int64"), "season": season,
        "game_type": games["gameType"].map(GAME_TYPES), "game_date": games["gameDate"].astype(str),
        "home_team_id": games["homeTeamId"], "away_team_id": games["visitingTeamId"],
        "home": games["homeTeamId"].map(teams), "away": games["visitingTeamId"].map(teams),
        "home_final": games["homeScore"], "away_final": games["visitingScore"],
        "state": games["gameStateId"],  # 7 = final
    }).sort_values("game_id").reset_index(drop=True)
    out.to_parquet(path)
    return path


def download_players(season: int, raw_dir: Path, force: bool = False, fetch=fetch_json) -> Path:
    """Everyone who played in `season` with their position (players_<season>.parquet)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"players_{season}.parquet"
    if path.exists() and not force:
        return path
    rows = []
    for kind, name_col in (("skater", "skaterFullName"), ("goalie", "goalieFullName")):
        for game_type in GAME_TYPES:
            data = fetch(stats_url(f"{kind}/summary", f"seasonId={season_code(season)} and gameTypeId={game_type}",
                                   limit=-1, start=0))["data"]
            for r in data:
                rows.append(dict(player_id=int(r["playerId"]), name=r.get(name_col),
                                 position="G" if kind == "goalie" else r.get("positionCode"),
                                 shoots=r.get("shootsCatches"), games=int(r.get("gamesPlayed") or 0),
                                 game_type=GAME_TYPES[game_type]))
    df = pd.DataFrame(rows, columns=["player_id", "name", "position", "shoots", "games", "game_type"])
    df.insert(0, "season", season)
    df.to_parquet(path)
    return path


def parse_clock(s) -> float:
    """'MM:SS' elapsed in the period -> seconds."""
    if not isinstance(s, str) or ":" not in s:
        return float("nan")
    m, sec = s.split(":")
    return int(m) * 60 + int(sec)


def shifts_frame(data: list[dict]) -> pd.DataFrame:
    """Shift-chart rows -> one row per shift: player_id, team_id, team, period, start, end (seconds)."""
    cols = ["player_id", "team_id", "team", "period", "start", "end"]
    rows = [r for r in data if r.get("typeCode") == SHIFT]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    return pd.DataFrame({
        "player_id": df["playerId"].astype("int64"), "team_id": df["teamId"].astype("int64"),
        "team": df["teamAbbrev"], "period": df["period"].astype("int64"),
        "start": df["startTime"].map(parse_clock), "end": df["endTime"].map(parse_clock),
    })[cols]


def download_shifts(game_ids, raw_dir: Path, force: bool = False, fetch=fetch_json) -> int:
    """Cache the shift charts of `game_ids` (shifts/<game_id>.parquet). Returns how many were fetched.
    A game the API has no shifts for is cached empty."""
    out = raw_dir / "shifts"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for g in game_ids:
        path = out / f"{int(g)}.parquet"
        if path.exists() and not force:
            continue
        shifts_frame(fetch(stats_url("shiftcharts", f"gameId={int(g)}"))["data"]).to_parquet(path)
        n += 1
    return n


def load_shifts(game_id: int, raw_dir: Path) -> pd.DataFrame | None:
    path = raw_dir / "shifts" / f"{int(game_id)}.parquet"
    return pd.read_parquet(path) if path.exists() else None
