"""Download published NFL model checkpoints from GitHub releases."""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

REPO_URL = "https://github.com/mefortunato/runitback"
# version -> (asset file name, sha256)
MODELS = {
    "v0.1.0": ("runitback-nfl-model-v0.1.0.pt",
               "319e99aacf2fd7531968f260bc6c34e342382ca88be3872f99f0dc57eb071cf4"),
}
LATEST = "v0.1.0"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_model(dest: Path, version: str = LATEST, force: bool = False, url: str | None = None) -> Path:
    """Download a released checkpoint to `dest`, verifying its checksum. Skips if already present."""
    if version not in MODELS:
        raise ValueError(f"unknown model version {version}; known: {', '.join(MODELS)}")
    name, expected = MODELS[version]
    if dest.exists() and not force:
        if sha256(dest) == expected:
            return dest
        raise FileExistsError(f"{dest} exists but is not the {version} model; pass force=True to replace it")
    url = url or f"{REPO_URL}/releases/download/{version}/{name}"
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
