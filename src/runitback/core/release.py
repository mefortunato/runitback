"""Download published model checkpoints from GitHub releases, verifying their SHA-256."""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Mapping

REPO_URL = "https://github.com/mefortunato/runitback"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_release(dest: Path, version: str, models: Mapping[str, tuple[str, str, str]],
                  force: bool = False, url: str | None = None) -> Path:
    """Download release `version` from `models` ({version: (tag, asset name, sha256)}) to `dest`.

    Skips the download when `dest` already holds that exact model."""
    if version not in models:
        known = ", ".join(models) or "none"
        raise ValueError(f"unknown model version {version}; known: {known}")
    tag, name, expected = models[version]
    if dest.exists() and not force:
        if sha256(dest) == expected:
            return dest
        raise FileExistsError(f"{dest} exists but is not the {version} model; pass force=True to replace it")
    url = url or f"{REPO_URL}/releases/download/{tag}/{name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        got = sha256(tmp)
        if got != expected:
            raise ValueError(f"checksum mismatch for {url}: expected {expected}, got {got}")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest
