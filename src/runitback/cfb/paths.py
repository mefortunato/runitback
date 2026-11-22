"""Default college football filesystem locations shared by the CLI.

Relative defaults resolve against the working directory; set the environment variables to pin them
elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    """Root of downloaded and processed data (RUNITBACK_CFB_DATA, default data/cfb)."""
    return Path(os.environ.get("RUNITBACK_CFB_DATA", "data/cfb"))


def runs_dir() -> Path:
    """Root of training runs (RUNITBACK_CFB_RUNS, default runs/cfb)."""
    return Path(os.environ.get("RUNITBACK_CFB_RUNS", "runs/cfb"))


def default_checkpoint() -> Path:
    return runs_dir() / "prod" / "model.pt"


def checkpoint_path() -> Path:
    """Checkpoint the server loads (RUNITBACK_CFB_CKPT, default runs/cfb/prod/model.pt)."""
    return Path(os.environ.get("RUNITBACK_CFB_CKPT", str(default_checkpoint())))
