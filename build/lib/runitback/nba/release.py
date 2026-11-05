"""Download published NBA model checkpoints from GitHub releases.

To publish a model: attach the checkpoint to a GitHub release tagged `nba-<version>` as
`runitback-nba-model-<version>.pt`, add it to MODELS with its SHA-256 (`sha256sum model.pt`), and
point LATEST at it. The Docker image then bakes it in via `runitback nba fetch-model --optional`.
"""
from __future__ import annotations

from pathlib import Path

from ..core.release import REPO_URL, fetch_release, sha256  # noqa: F401  (re-exported)

# version -> (release tag, asset file name, sha256)
MODELS: dict[str, tuple[str, str, str]] = {
    "v0.1.0": ("nba-v0.1.0", "runitback-nba-model-v0.1.0.pt",
               "eaa9ca750fb11558edab51b817a98c77d1622deb20136e6acce619efa64813a6"),
}
LATEST: str | None = "v0.1.0"


def fetch_model(dest: Path, version: str | None = None, force: bool = False, url: str | None = None) -> Path:
    """Download a released checkpoint (default: LATEST) to `dest`, verifying its checksum."""
    version = version or LATEST
    if version is None:
        raise ValueError("no NBA model has been released yet")
    return fetch_release(dest, version, MODELS, force, url)
