"""Published college football model checkpoints (GitHub releases).

The list lives in releases.json next to this file: {"latest": version or null, "models": {version:
{tag, asset, sha256}}}. The Docker image bakes in `latest` (`runitback cfb fetch-model --optional`; until
a model is listed the college generator page says how to get one). The `CFB model` workflow trains and
publishes a model (release tag `cfb-<version>`) and opens the PR that registers it here.
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
LATEST: str | None = _REL["latest"]


def tag_name(version: str) -> str:
    return f"cfb-{version}"


def asset_name(version: str) -> str:
    return f"runitback-cfb-model-{version}.pt"


def register_release(version: str, checkpoint: Path, path: Path = RELEASES_FILE, make_latest: bool = True) -> dict:
    """Add a published checkpoint (tag ``cfb-<version>``) to releases.json and, by default, make it the latest."""
    if not re.fullmatch(r"v\d+\.\d+\.\d+", version):
        raise ValueError(f"not a vMAJOR.MINOR.PATCH version: {version}")
    rel = load_releases(path)
    if version in rel["models"]:
        raise ValueError(f"{version} is already registered")
    rel["models"][version] = {"tag": tag_name(version), "asset": asset_name(version), "sha256": sha256(checkpoint)}
    if make_latest:
        rel["latest"] = version
    Path(path).write_text(json.dumps(rel, indent=2) + "\n")
    return rel


def fetch_model(dest: Path, version: str | None = None, force: bool = False, url: str | None = None) -> Path:
    """Download a released checkpoint (default: the latest) to `dest`, verifying its checksum."""
    version = version or LATEST
    if version is None:
        raise ValueError("no college football model has been released yet")
    return fetch_release(dest, version, MODELS, force, url)
