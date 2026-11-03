import hashlib

import pytest

from runitback.nba import release


def test_fetch_latest_downloads_and_verifies(tmp_path, monkeypatch):
    src = tmp_path / "asset.pt"
    src.write_bytes(b"nba weights")
    monkeypatch.setitem(release.MODELS, "vtest", ("nba-vtest", "asset.pt", hashlib.sha256(b"nba weights").hexdigest()))
    monkeypatch.setattr(release, "LATEST", "vtest")
    dest = tmp_path / "runs" / "model.pt"
    assert release.fetch_model(dest, url=src.as_uri()) == dest
    assert dest.read_bytes() == b"nba weights"


def test_fetch_without_release(tmp_path, monkeypatch):
    monkeypatch.setattr(release, "LATEST", None)
    with pytest.raises(ValueError, match="no NBA model has been released"):
        release.fetch_model(tmp_path / "model.pt")
