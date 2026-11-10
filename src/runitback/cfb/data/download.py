"""Download college football play-by-play, player box scores and schedules from the public
SportsDataverse releases, one parquet file per season under ``raw_dir``.

- ``pbp_<season>.parquet``: ``espn_cfb_pbp`` (ESPN play-by-play with cfbfastR's EPA), trimmed to
  :data:`PBP_COLUMNS` (the full file has ~500 columns and ~50 MB a season).
- ``player_box_<season>.parquet``: ``espn_cfb_player_box`` (one row per player, game and stat
  category: passing, rushing, receiving, ...), with ESPN athlete ids.
- ``schedule_<season>.parquet``: ``cfb_schedules`` (CFBD's schedule: kickoff times, ESPN team ids,
  school names and divisions), the wager model's source of game ids and team keys.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ...wager.cfb.data.download import RELEASES, SCHEDULE_URL

log = logging.getLogger(__name__)

PBP_URL = RELEASES + "/espn_cfb_pbp/play_by_play_{season}.parquet"
PLAYER_BOX_URL = RELEASES + "/espn_cfb_player_box/player_box_{season}.parquet"

PBP_COLUMNS = [
    "season", "week", "seasonType", "game_id", "game_play_number", "id", "pos_team_id", "def_pos_team_id",
    "homeTeamId", "awayTeamId", "homeFinalScore", "awayFinalScore", "homeTeamSpread", "gameSpreadAvailable",
    "overUnder", "period", "down", "distance", "start.yardsToEndzone", "pos_score_diff_start", "EPA", "pass",
    "rush", "sack", "scrimmage_play", "kneel_down", "passer_player_name", "passer_player_id",
    "rusher_player_name", "rusher_player_id", "receiver_player_name", "receiver_player_id", "type.text",
]
BOX_COLUMNS = ["game_id", "season", "team_id", "athlete_id", "athlete_name", "category",
               "completions/passingAttempts", "rushingAttempts"]


def _fetch(url: str, path: Path, columns: list[str] | None, force: bool) -> Path:
    if path.exists() and not force:
        return path
    import io

    import httpx
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    r = httpx.get(url, timeout=300, follow_redirects=True)
    r.raise_for_status()
    table = pq.read_table(io.BytesIO(r.content))
    if columns is not None:
        table = table.select([c for c in columns if c in table.column_names])
    pq.write_table(table, path)
    log.info("%s: %d rows", path.name, table.num_rows)
    return path


def download_season(season: int, raw_dir: Path, force: bool = False) -> list[Path]:
    """The season's trimmed play-by-play, player box scores and schedule (``season`` is the year the
    season starts in, as in the data)."""
    return [
        _fetch(PBP_URL.format(season=season), raw_dir / f"pbp_{season}.parquet", PBP_COLUMNS, force),
        _fetch(PLAYER_BOX_URL.format(season=season), raw_dir / f"player_box_{season}.parquet", BOX_COLUMNS, force),
        _fetch(SCHEDULE_URL.format(season=season), raw_dir / f"schedule_{season}.parquet", None, force),
    ]


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
