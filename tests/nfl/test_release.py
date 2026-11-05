import hashlib

import pytest

from runitback.nfl import release


@pytest.fixture
def fake_release(tmp_path, monkeypatch):
    src = tmp_path / "asset.pt"
    src.write_bytes(b"weights")
    monkeypatch.setitem(release.MODELS, "vtest", ("vtest", "asset.pt", hashlib.sha256(b"weights").hexdigest()))
    return src.as_uri()


def test_fetch_downloads_and_verifies(tmp_path, fake_release):
    dest = tmp_path / "runs" / "model.pt"
    assert release.fetch_model(dest, "vtest", url=fake_release) == dest
    assert dest.read_bytes() == b"weights"
    assert release.fetch_model(dest, "vtest", url="file:///nonexistent") == dest  # already present: no download


def test_fetch_rejects_bad_checksum_and_foreign_file(tmp_path, fake_release, monkeypatch):
    monkeypatch.setitem(release.MODELS, "vbad", ("vbad", "asset.pt", "0" * 64))
    dest = tmp_path / "model.pt"
    with pytest.raises(ValueError, match="checksum mismatch"):
        release.fetch_model(dest, "vbad", url=fake_release)
    assert not dest.exists() and not (tmp_path / "model.pt.part").exists()
    dest.write_bytes(b"something else")
    with pytest.raises(FileExistsError):
        release.fetch_model(dest, "vtest", url=fake_release)
    with pytest.raises(ValueError, match="unknown model version"):
        release.fetch_model(dest, "v9.9")


def test_register_release_and_next_version(tmp_path):
    import json
    path = tmp_path / "releases.json"
    path.write_text((release.RELEASES_FILE).read_text())
    ck = tmp_path / "model.pt"
    ck.write_bytes(b"new weights")
    rel = release.register_release("v0.2.0", ck, path)
    assert rel["latest"] == "v0.2.0"
    assert rel["models"]["v0.2.0"] == {"tag": "v0.2.0", "asset": "runitback-nfl-model-v0.2.0.pt",
                                       "sha256": hashlib.sha256(b"new weights").hexdigest()}
    assert json.loads(path.read_text())["models"]["v0.1.0"]  # earlier releases stay listed
    with pytest.raises(ValueError, match="already registered"):
        release.register_release("v0.2.0", ck, path)
    with pytest.raises(ValueError):
        release.register_release("0.3", ck, path)
    assert release.next_patch_version("v0.2.0") == "v0.2.1"
    assert release.next_patch_version("v1.9.9") == "v1.9.10"


def test_fetch_defaults_to_latest(tmp_path, monkeypatch, fake_release):
    monkeypatch.setattr(release, "LATEST", "vtest")
    dest = tmp_path / "model.pt"
    assert release.fetch_model(dest, url=fake_release) == dest  # no version: the latest release
