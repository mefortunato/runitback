"""Default filesystem locations shared by the CLI and the server.

Relative defaults resolve against the working directory; set the environment variables to pin
them elsewhere (the Docker image sets RUNITBACK_NFL_CKPT, for example).
"""
from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    """Root of downloaded, processed and synthetic data (RUNITBACK_NFL_DATA, default data/nfl)."""
    return Path(os.environ.get("RUNITBACK_NFL_DATA", "data/nfl"))


def runs_dir() -> Path:
    """Root of training runs (RUNITBACK_NFL_RUNS, default runs/nfl)."""
    return Path(os.environ.get("RUNITBACK_NFL_RUNS", "runs/nfl"))


def default_checkpoint() -> Path:
    return runs_dir() / "prod" / "model.pt"


def checkpoint_path() -> Path:
    """Checkpoint the server loads (RUNITBACK_NFL_CKPT, default runs/nfl/prod/model.pt)."""
    return Path(os.environ.get("RUNITBACK_NFL_CKPT", str(default_checkpoint())))
