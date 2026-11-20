"""Default Premier League filesystem locations shared by the CLI.

Relative defaults resolve against the working directory; set the environment variable to pin it
elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    """Root of downloaded data (RUNITBACK_EPL_DATA, default data/epl)."""
    return Path(os.environ.get("RUNITBACK_EPL_DATA", "data/epl"))
