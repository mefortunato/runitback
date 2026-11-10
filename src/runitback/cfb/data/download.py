"""Download college football play-by-play: sportsdataverse's ``espn_cfb_pbp`` release (ESPN's
play-by-play with sportsdataverse's EPA, one file per season from 2004), cached as parquet.

Only the columns the preprocess reads are kept (a full season file has ~500 columns and is
~50 MB). Columns a season's file does not have (2004-2013 have no ``pointAfterAttempt`` or
``extra_point_result``) are left out; ``load_raw`` adds them as missing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import pandas as pd
import pyarrow.parquet as pq

log = logging.getLogger(__name__)
RELEASE_URL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
               "espn_cfb_pbp/play_by_play_{season}.parquet")

COLUMNS = [
    "season", "game_id", "game_play_number", "seasonType", "week", "status_type_completed",
    "homeTeamId", "awayTeamId", "homeTeamName", "awayTeamName", "homeFinalScore", "awayFinalScore",
    "pos_team_id", "def_pos_team_id", "kicking_team", "period", "clock.minutes", "clock.seconds",
    "start.down", "start.distance", "start.yardsToEndzone", "end.yardsToEndzone", "type.text",
    "statYardage", "start.homeScore", "start.awayScore", "end.homeScore", "end.awayScore",
    "homeTimeoutCalled", "awayTimeoutCalled", "pointAfterAttempt.text", "extra_point_result",
    "two_point_conv_result", "rush", "pass", "sack", "int", "fumble_lost", "completion",
    "kneel_down", "scrimmage_play", "EPA", "text", "wallclock",
]


def download_season(season: int, raw_dir: Path, force: bool = False,
                    read: Callable[[str], pd.DataFrame] | None = None) -> Path | None:
    """Cache `season`'s play-by-play as pbp_<season>.parquet. None if the release has no file."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"pbp_{season}.parquet"
    if path.exists() and not force:
        return path
    url = RELEASE_URL.format(season=season)
    try:
        if read is not None:
            df = read(url)
        else:
            import io
            import urllib.request
            with urllib.request.urlopen(url, timeout=300) as r:
                buf = io.BytesIO(r.read())
            names = set(pq.read_schema(buf).names)
            buf.seek(0)
            df = pd.read_parquet(buf, columns=[c for c in COLUMNS if c in names])
    except Exception as e:  # no file for the season (e.g. one not started yet)
        log.warning("cfb %d: no play-by-play (%s)", season, e)
        return None
    df = df[[c for c in COLUMNS if c in df.columns]]
    df.to_parquet(path)
    log.info("cfb %d: %d plays, %d games", season, len(df), df.game_id.nunique())
    return path


def load_raw(season: int, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"pbp_{season}.parquet"
    names = set(pq.read_schema(path).names)
    df = pd.read_parquet(path, columns=[c for c in COLUMNS if c in names])
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df
