"""Download NFL play-by-play via SportsDataverse (nflverse releases), cached as parquet."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
NFLVERSE_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"

COLUMNS = [
    "game_id", "season", "season_type", "week", "home_team", "away_team", "posteam", "defteam",
    "qtr", "quarter_seconds_remaining", "down", "ydstogo", "yardline_100", "play_type",
    "yards_gained", "complete_pass", "interception", "sack", "fumble_lost", "field_goal_result",
    "extra_point_result", "two_point_conv_result", "two_point_attempt", "timeout", "timeout_team",
    "total_home_score", "total_away_score", "desc", "home_score", "away_score", "epa", "game_date",
]


def download_season(season: int, raw_dir: Path, force: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"pbp_{season}.parquet"
    if path.exists() and not force:
        return path
    try:
        from sportsdataverse.nfl import load_nfl_pbp
        df = load_nfl_pbp(seasons=[season], return_as_pandas=True)
    except Exception as e:  # package missing or API change: read the release directly
        log.warning("sportsdataverse failed for %s (%s); falling back to nflverse URL", season, e)
        df = pd.read_parquet(NFLVERSE_URL.format(season=season))
    df.to_parquet(path)
    return path


def load_raw(season: int, raw_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(raw_dir / f"pbp_{season}.parquet", columns=COLUMNS)
    return df
