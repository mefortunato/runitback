import hashlib

import pytest

from runitback.nfl import release


@pytest.fixture
def fake_release(tmp_path, monkeypatch):
    src = tmp_path / "asset.pt"
    src.write_bytes(b"weights")
    monkeypatch.setitem(release.MODELS, "vtest", ("asset.pt", hashlib.sha256(b"weights").hexdigest()))
    return src.as_uri()


def test_fetch_downloads_and_verifies(tmp_path, fake_release):
    dest = tmp_path / "runs" / "model.pt"
    assert release.fetch_model(dest, "vtest", url=fake_release) == dest
    assert dest.read_bytes() == b"weights"
    assert release.fetch_model(dest, "vtest", url="file:///nonexistent") == dest  # already present: no download


def test_fetch_rejects_bad_checksum_and_foreign_file(tmp_path, fake_release, monkeypatch):
    monkeypatch.setitem(release.MODELS, "vbad", ("asset.pt", "0" * 64))
    dest = tmp_path / "model.pt"
    with pytest.raises(ValueError, match="checksum mismatch"):
        release.fetch_model(dest, "vbad", url=fake_release)
    assert not dest.exists() and not (tmp_path / "model.pt.part").exists()
    dest.write_bytes(b"something else")
    with pytest.raises(FileExistsError):
        release.fetch_model(dest, "vtest", url=fake_release)
    with pytest.raises(ValueError, match="unknown model version"):
        release.fetch_model(dest, "v9.9")
