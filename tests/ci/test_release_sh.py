"""scripts/ci/release.sh against a fake `gh`: absent vs failing releases, required vs optional folds."""
import os
import subprocess
import tarfile
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "ci" / "release.sh"

# a gh that serves $FAKE_RELEASES/<tag>/<asset> and fails like gh does for a missing release
FAKE_GH = """#!/usr/bin/env bash
set -e
[ -z "$FAKE_GH_FAIL" ] || { echo "HTTP 502: bad gateway" >&2; exit 1; }
tag=$3
[ "$1 $2" = "release view" ] || [ "$1 $2" = "release download" ] || exit 2
[ -d "$FAKE_RELEASES/$tag" ] || { echo "release not found" >&2; exit 1; }
if [ "$2" = view ]; then ls "$FAKE_RELEASES/$tag"; exit 0; fi
while [ $# -gt 0 ]; do case $1 in --pattern) p=$2;; --dir) d=$2;; esac; shift; done
mkdir -p "$d" && cp "$FAKE_RELEASES/$tag/$p" "$d/"
"""


def run(tmp_path, body, **env):
    gh = tmp_path / "bin" / "gh"
    gh.parent.mkdir(exist_ok=True)
    gh.write_text(FAKE_GH)
    gh.chmod(0o755)
    e = {**os.environ, "PATH": f"{gh.parent}:{os.environ['PATH']}", "FAKE_RELEASES": str(tmp_path / "rel"), **env}
    return subprocess.run(["bash", "-c", f"source {SCRIPT}; {body}"], cwd=tmp_path, env=e, capture_output=True, text=True)


def release(tmp_path, tag, seasons):
    d = tmp_path / "rel" / tag
    d.mkdir(parents=True)
    for s in seasons:
        marker = tmp_path / f"m{s}"
        marker.write_text(str(s))
        with tarfile.open(d / f"fold{s}.tar.gz", "w:gz") as t:
            t.add(marker, arcname="model.pt")


def test_folds_land_per_season_and_per_sport(tmp_path):
    release(tmp_path, "foundation-cfb", [2024, 2025])
    r = run(tmp_path, "fetch_folds cfb out/cfb 2024 2025")
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "out/cfb/fold2025/model.pt").read_text() == "2025"
    assert (tmp_path / "dl/cfb/fold2024.tar.gz").exists()  # not in a directory another sport uses


def test_required_season_missing_fails(tmp_path):
    release(tmp_path, "foundation-cfb", [2024])
    r = run(tmp_path, "fetch_folds cfb out 2024 2025")
    assert r.returncode == 1 and "no fold2025" in r.stderr


def test_optional_season_may_be_missing(tmp_path):
    release(tmp_path, "foundation-cfb", [2024])
    r = run(tmp_path, "fetch_folds cfb out 2024 -- 2025")
    assert r.returncode == 0 and "no fold2025 yet" in r.stdout
    assert (tmp_path / "out/fold2024/model.pt").exists() and not (tmp_path / "out/fold2025").exists()


def test_missing_release_is_empty_but_not_an_error(tmp_path):
    (tmp_path / "rel").mkdir()
    r = run(tmp_path, "release_assets foundation-cfb")
    assert r.returncode == 0 and r.stdout == ""
    assert run(tmp_path, "fetch_folds cfb out 2025").returncode == 1  # the folds are required


def test_other_gh_failures_fail_the_step(tmp_path):
    release(tmp_path, "foundation-cfb", [2025])
    r = run(tmp_path, "release_assets foundation-cfb", FAKE_GH_FAIL="1")
    assert r.returncode == 1 and "cannot read release" in r.stderr
    assert run(tmp_path, "fetch_folds cfb out 2025", FAKE_GH_FAIL="1").returncode == 1


def test_fetch_asset(tmp_path):
    d = tmp_path / "rel" / "cfb-espn-cache"
    d.mkdir(parents=True)
    (d / "cfb-espn-cache.tar.gz").write_text("x")
    assert run(tmp_path, "fetch_asset cfb-espn-cache cfb-espn-cache.tar.gz dl/lines").returncode == 0
    assert (tmp_path / "dl/lines/cfb-espn-cache.tar.gz").exists()
    assert run(tmp_path, "fetch_asset cfb-espn-cache nope.tar.gz dl/lines").returncode == 1
