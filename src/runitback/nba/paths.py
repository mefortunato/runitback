"""Default NBA filesystem locations shared by the CLI and the server.

Relative defaults resolve against the working directory; set the environment variables to pin
them elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    """Root of downloaded and processed data (RUNITBACK_NBA_DATA, default data/nba)."""
    return Path(os.environ.get("RUNITBACK_NBA_DATA", "data/nba"))


def runs_dir() -> Path:
    """Root of training runs (RUNITBACK_NBA_RUNS, default runs/nba)."""
    return Path(os.environ.get("RUNITBACK_NBA_RUNS", "runs/nba"))


def default_checkpoint() -> Path:
    return runs_dir() / "prod" / "model.pt"


def checkpoint_path() -> Path:
    """Checkpoint the server loads (RUNITBACK_NBA_CKPT, default runs/nba/prod/model.pt)."""
    return Path(os.environ.get("RUNITBACK_NBA_CKPT", str(default_checkpoint())))
