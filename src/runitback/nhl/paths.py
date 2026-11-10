"""Default NHL filesystem locations shared by the CLI.

Relative defaults resolve against the working directory; set the environment variable to pin it
elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    """Root of downloaded and processed data (RUNITBACK_NHL_DATA, default data/nhl)."""
    return Path(os.environ.get("RUNITBACK_NHL_DATA", "data/nhl"))
