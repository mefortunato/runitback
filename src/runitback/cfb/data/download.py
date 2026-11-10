"""Download college football data from the public SportsDataverse releases, one parquet file per
season under ``raw_dir`` (``season`` is the year the season starts in, as in the data):

- ``pbp_<season>.parquet``: ``espn_cfb_pbp`` (ESPN play-by-play with cfbfastR's EPA, 2004+), trimmed
  to :data:`PBP_COLUMNS`, the columns the engine preprocess (``runitback.cfb.data.preprocess``, F1)
  and the quarterback tables (``runitback.cfb.data.players``, F2) read. The full file has ~500
  columns and ~50 MB a season. Columns a season's file does not have (2004-2013 have no
  ``pointAfterAttempt`` or ``extra_point_result``) are left out; ``load_raw`` adds them as missing.
- ``player_box_<season>.parquet``: ``espn_cfb_player_box`` (one row per player, game and stat
  category: passing, rushing, receiving, ...), with ESPN athlete ids.
- ``schedule_<season>.parquet``: ``cfb_schedules`` (CFBD's schedule: kickoff times, ESPN team ids,
  school names and divisions), the wager model's source of game ids and team keys.

A file the release does not have (a season not started yet) is skipped with a warning.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Callable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ...wager.cfb.data.download import RELEASES, SCHEDULE_URL

log = logging.getLogger(__name__)

PBP_URL = RELEASES + "/espn_cfb_pbp/play_by_play_{season}.parquet"
PLAYER_BOX_URL = RELEASES + "/espn_cfb_player_box/player_box_{season}.parquet"

PBP_COLUMNS = [
    # games and teams
    "season", "game_id", "game_play_number", "id", "seasonType", "week", "status_type_completed",
    "homeTeamId", "awayTeamId", "homeTeamName", "awayTeamName", "homeFinalScore", "awayFinalScore",
    "homeTeamSpread", "gameSpreadAvailable", "overUnder", "pos_team_id", "def_pos_team_id", "kicking_team",
    # state
    "period", "clock.minutes", "clock.seconds", "down", "distance", "start.down", "start.distance",
    "start.yardsToEndzone", "end.yardsToEndzone", "pos_score_diff_start", "start.homeScore", "start.awayScore",
    "end.homeScore", "end.awayScore", "homeTimeoutCalled", "awayTimeoutCalled",
    # the play
    "type.text", "text", "statYardage", "EPA", "rush", "pass", "sack", "int", "fumble_lost", "completion",
    "kneel_down", "scrimmage_play", "pointAfterAttempt.text", "extra_point_result", "two_point_conv_result",
    "wallclock",
    # players
    "passer_player_name", "passer_player_id", "rusher_player_name", "rusher_player_id",
    "receiver_player_name", "receiver_player_id",
]
COLUMNS = PBP_COLUMNS  # the engine preprocess's name for them
BOX_COLUMNS = ["game_id", "season", "team_id", "athlete_id", "athlete_name", "category",
               "completions/passingAttempts", "rushingAttempts"]


def _read_url(url: str) -> pa.Table:
    import httpx

    r = httpx.get(url, timeout=300, follow_redirects=True)
    r.raise_for_status()
    return pq.read_table(io.BytesIO(r.content))


def _fetch(url: str, path: Path, columns: list[str] | None, force: bool,
           read: Callable[[str], pa.Table | pd.DataFrame] | None = None) -> Path | None:
    """Cache ``url`` at ``path`` (only ``columns``, those it has). None if the release has no file."""
    if path.exists() and not force:
        return path
    try:
        table = (read or _read_url)(url)
    except Exception as e:  # no file for the season (e.g. one not started yet)
        log.warning("%s: not downloaded (%s)", path.name, e)
        return None
    if isinstance(table, pd.DataFrame):
        table = pa.Table.from_pandas(table, preserve_index=False)
    if columns is not None:
        table = table.select([c for c in columns if c in table.column_names])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    log.info("%s: %d rows", path.name, table.num_rows)
    return path


def download_season(season: int, raw_dir: Path, force: bool = False, players: bool = True,
                    read: Callable[[str], pa.Table | pd.DataFrame] | None = None) -> list[Path]:
    """The season's trimmed play-by-play and, with ``players``, its player box scores and schedule.
    Returns the files written or already cached; a file the release lacks is left out."""
    files = [(PBP_URL, "pbp", PBP_COLUMNS)]
    if players:
        files += [(PLAYER_BOX_URL, "player_box", BOX_COLUMNS), (SCHEDULE_URL, "schedule", None)]
    paths = [_fetch(url.format(season=season), raw_dir / f"{kind}_{season}.parquet", cols, force, read)
             for url, kind, cols in files]
    return [p for p in paths if p is not None]


def load(kind: str, seasons: list[int], raw_dir: Path) -> pd.DataFrame:
    """``pbp`` / ``player_box`` / ``schedule`` rows of the seasons whose files are in ``raw_dir``."""
    parts = []
    for s in seasons:
        path = raw_dir / f"{kind}_{s}.parquet"
        if path.exists():
            parts.append(pd.read_parquet(path))
        else:
            log.warning("season %d: %s missing; skipped", s, path.name)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_raw(season: int, raw_dir: Path) -> pd.DataFrame:
    """One season's play-by-play with every column of :data:`PBP_COLUMNS` (missing ones as NA)."""
    path = raw_dir / f"pbp_{season}.parquet"
    names = set(pq.read_schema(path).names)
    df = pd.read_parquet(path, columns=[c for c in PBP_COLUMNS if c in names])
    for c in PBP_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df
