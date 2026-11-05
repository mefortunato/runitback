"""Published NFL model checkpoints (GitHub releases).

The list lives in releases.json next to this file: {"latest": version, "models": {version: {tag,
asset, sha256}}}. The Docker image bakes in `latest` (`runitback nfl fetch-model`); the training and
refresh workflows add entries with `register_release`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..core.release import REPO_URL, fetch_release, sha256  # noqa: F401  (re-exported)

RELEASES_FILE = Path(__file__).with_name("releases.json")


def load_releases(path: Path = RELEASES_FILE) -> dict:
    return json.loads(Path(path).read_text())


_REL = load_releases()
# version -> (release tag, asset file name, sha256)
MODELS = {v: (m["tag"], m["asset"], m["sha256"]) for v, m in _REL["models"].items()}
LATEST = _REL["latest"]


def asset_name(version: str) -> str:
    return f"runitback-nfl-model-{version}.pt"


def next_patch_version(version: str) -> str:
    """v0.2.0 -> v0.2.1"""
    m = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", version)
    if not m:
        raise ValueError(f"not a vMAJOR.MINOR.PATCH version: {version}")
    return f"v{m[1]}.{m[2]}.{int(m[3]) + 1}"


def register_release(version: str, checkpoint: Path, path: Path = RELEASES_FILE, make_latest: bool = True) -> dict:
    """Add a published checkpoint (tag = version) to releases.json and, by default, make it the latest."""
    if not re.fullmatch(r"v\d+\.\d+\.\d+", version):
        raise ValueError(f"not a vMAJOR.MINOR.PATCH version: {version}")
    rel = load_releases(path)
    if version in rel["models"]:
        raise ValueError(f"{version} is already registered")
    rel["models"][version] = {"tag": version, "asset": asset_name(version), "sha256": sha256(checkpoint)}
    if make_latest:
        rel["latest"] = version
    Path(path).write_text(json.dumps(rel, indent=2) + "\n")
    return rel


def fetch_model(dest: Path, version: str | None = None, force: bool = False, url: str | None = None) -> Path:
    """Download a released checkpoint (default: the latest) to `dest`, verifying its checksum."""
    return fetch_release(dest, version or LATEST, MODELS, force, url)
