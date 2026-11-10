"""Download NBA play-by-play via SportsDataverse (ESPN data, hoopR releases), cached as parquet."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
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


def numpy_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Plain numpy columns in place of pandas' nullable or pyarrow-backed ones (``Int64``,
    ``boolean``, ``string``, ``int32[pyarrow]``, ... with ``pd.NA``), as
    ``sportsdataverse.nba.load_nba_pbp`` can return: missing numbers become NaN, missing flags
    False and missing strings None, as in the release files, so preprocessing sees the same values
    whichever way a season was downloaded."""
    out = {}
    for c in df.columns:
        col = df[c]
        ext = pd.api.types.is_extension_array_dtype(col.dtype)
        if ext and pd.api.types.is_bool_dtype(col.dtype):
            col = col.fillna(False).astype(bool)
        elif ext and pd.api.types.is_numeric_dtype(col.dtype):
            col = col.astype("float64") if col.isna().any() else col.to_numpy(dtype=np.float64).astype(
                np.int64 if pd.api.types.is_integer_dtype(col.dtype) else np.float64)
            col = pd.Series(col, index=df.index)
        elif (ext or col.dtype == object) and col.isna().any():
            vals = col.astype(object).to_numpy(copy=True)
            vals[col.isna().to_numpy()] = None
            col = pd.Series(vals, index=df.index, dtype=object)
        out[c] = col
    return pd.DataFrame(out, index=df.index)


def load_raw(season: int, raw_dir: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq
    path = raw_dir / f"pbp_{season}.parquet"
    available = set(pq.read_schema(path).names)
    return numpy_dtypes(pd.read_parquet(path, columns=[c for c in COLUMNS if c in available]))
