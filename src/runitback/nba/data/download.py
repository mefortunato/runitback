"""Download NBA play-by-play via SportsDataverse (ESPN data, hoopR releases), cached as parquet."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
ESPN_NBA_URL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
                "espn_nba_pbp/play_by_play_{season}.parquet")

COLUMNS = [
    "game_id", "game_play_number", "sequence_number", "season", "season_type", "game_date",
    "period_number", "start_quarter_seconds_remaining", "type_text", "text", "team_id",
    "home_team_id", "away_team_id", "home_team_abbrev", "away_team_abbrev", "home_score", "away_score",
    "score_value", "scoring_play", "shooting_play", "home_team_spread",
]


def download_season(season: int, raw_dir: Path, force: bool = False) -> Path:
    """`season` is the year the season ends in (2025 = 2024-25)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"pbp_{season}.parquet"
    if path.exists() and not force:
        return path
    try:
        from sportsdataverse.nba import load_nba_pbp
        df = load_nba_pbp(seasons=[season], return_as_pandas=True)
        if df is None or len(df) == 0:
            raise ValueError("no rows")
    except Exception as e:  # package missing or API change: read the release directly
        log.warning("sportsdataverse failed for %s (%s); falling back to the release URL", season, e)
        df = pd.read_parquet(ESPN_NBA_URL.format(season=season))
    df.to_parquet(path)
    return path


def load_raw(season: int, raw_dir: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq
    path = raw_dir / f"pbp_{season}.parquet"
    available = set(pq.read_schema(path).names)
    return pd.read_parquet(path, columns=[c for c in COLUMNS if c in available])
