"""Default MLB filesystem locations shared by the CLI.

Relative defaults resolve against the working directory; set the environment variable to pin them
elsewhere. ``data/mlb/raw.csv`` and ``data/mlb/mlb_cache`` belong to the wager loader
(``runitback-wager mlb download``); this package uses ``raw/feeds`` and ``processed`` beside them.
"""
from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    """Root of downloaded and processed data (RUNITBACK_MLB_DATA, default data/mlb)."""
    return Path(os.environ.get("RUNITBACK_MLB_DATA", "data/mlb"))


def feeds_dir() -> Path:
    """Cached live feeds of finished games, one gzipped JSON per game."""
    return data_dir() / "raw" / "feeds"


def processed_dir() -> Path:
    """Season tables: ``pa_<S>``, ``games_<S>``, ``lineups_<S>``, ``substitutions_<S>``, ``players_<S>``."""
    return data_dir() / "processed"
