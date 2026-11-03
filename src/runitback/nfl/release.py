"""Download published NFL model checkpoints from GitHub releases."""
from __future__ import annotations

from pathlib import Path

from ..core.release import REPO_URL, fetch_release, sha256  # noqa: F401  (re-exported)

# version -> (release tag, asset file name, sha256)
MODELS = {
    "v0.1.0": ("v0.1.0", "runitback-nfl-model-v0.1.0.pt",
               "319e99aacf2fd7531968f260bc6c34e342382ca88be3872f99f0dc57eb071cf4"),
}
LATEST = "v0.1.0"


def fetch_model(dest: Path, version: str = LATEST, force: bool = False, url: str | None = None) -> Path:
    """Download a released checkpoint to `dest`, verifying its checksum. Skips if already present."""
    return fetch_release(dest, version, MODELS, force, url)
