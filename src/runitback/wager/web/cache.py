"""A store of finished work shared by every machine that builds the web data.

``runitback wager web build`` backtests five seasons of each sport and trains the
model for upcoming games. Both are slow on a CPU and both are pure
functions of their inputs, so each result is stored under a key that says
exactly what produced it (:func:`~runitback.wager.web.build.backtest_key`): any
machine that builds the same thing, a GPU box, a laptop or the Train
workflow, can reuse it instead of computing it again.

Stores:

- a directory (``path/to/cache``), for one machine;
- a GitHub release (``release:<tag>``, or ``release:<owner>/<repo>@<tag>``),
  read and written with the ``gh`` CLI, so every machine with access to the
  repository shares it. One asset per entry.

Entries:

- a backtest season: ``<sport>-<season>-<key>.pool.csv.gz`` (its games'
  out-of-sample predictions) and ``.rankings.csv.gz`` (power ratings before
  each date);
- a trained model: ``<sport>-model-<key>.tar.gz`` (the MLflow model
  directory and when it was trained).
"""

from __future__ import annotations

import datetime
import io
import json
import logging
import pathlib
import shutil
import subprocess
import tarfile
import tempfile

import pandas as pd

logger = logging.getLogger(__name__)

# Characters of a key used in entry names (64 bits: no collisions in
# practice, short enough to read).
KEY_CHARS = 16


def season_names(sport: str, season: int, key: str) -> tuple[str, str]:
    stem = f"{sport}-{int(season)}-{key[:KEY_CHARS]}"
    return f"{stem}.pool.csv.gz", f"{stem}.rankings.csv.gz"


def model_name(sport: str, key: str) -> str:
    return f"{sport}-model-{key[:KEY_CHARS]}.tar.gz"


class Store:
    """Named binary entries: :meth:`names`, :meth:`get`, :meth:`put`."""

    def names(self) -> set[str]:
        raise NotImplementedError

    def get(self, name: str) -> bytes | None:
        raise NotImplementedError

    def put(self, name: str, data: bytes) -> None:
        raise NotImplementedError

    # ---------------------------------------------------------- seasons
    def get_season(
        self, sport: str, season: int, key: str
    ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        """``(pool, rankings)`` of a stored backtest season, or ``None``."""
        pool_name, rankings_name = season_names(sport, season, key)
        if not {pool_name, rankings_name} <= self.names():
            return None
        pool, rankings = self.get(pool_name), self.get(rankings_name)
        if pool is None or rankings is None:
            return None
        read = lambda data: pd.read_csv(io.BytesIO(data), compression="gzip")  # noqa: E731
        return read(pool), read(rankings)

    def put_season(
        self,
        sport: str,
        season: int,
        key: str,
        pool: pd.DataFrame,
        rankings: pd.DataFrame,
    ) -> None:
        for name, frame in zip(season_names(sport, season, key), (pool, rankings)):
            buffer = io.BytesIO()
            frame.to_csv(
                buffer, index=False, compression={"method": "gzip", "mtime": 0}
            )
            self.put(name, buffer.getvalue())

    def has_season(self, sport: str, season: int, key: str) -> bool:
        return set(season_names(sport, season, key)) <= self.names()

    # ------------------------------------------------------------ models
    def get_model(
        self, sport: str, key: str, directory: pathlib.Path, max_age_days: float
    ) -> dict | None:
        """Unpack a stored model trained at most ``max_age_days`` ago into
        ``directory``; returns its metadata (``trained_at``, ``metrics``)."""
        name = model_name(sport, key)
        if name not in self.names():
            return None
        data = self.get(name)
        if data is None:
            return None
        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                tar.extractall(tmp, filter="data")
            meta = json.loads((pathlib.Path(tmp) / "model_meta.json").read_text())
            trained = datetime.datetime.fromisoformat(meta["trained_at"])
            age = datetime.datetime.now(datetime.timezone.utc) - trained
            if age > datetime.timedelta(days=max_age_days):
                logger.info("stored %s model is %s old: retraining", sport, age)
                return None
            if directory.exists():
                shutil.rmtree(directory)
            shutil.copytree(pathlib.Path(tmp) / "model", directory)
        return meta

    def put_model(self, sport: str, key: str, directory: pathlib.Path, meta: dict):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            tar.add(directory, arcname="model")
            info = json.dumps(meta, default=float).encode()
            entry = tarfile.TarInfo("model_meta.json")
            entry.size = len(info)
            tar.addfile(entry, io.BytesIO(info))
        self.put(model_name(sport, key), buffer.getvalue())


class DirectoryStore(Store):
    """Entries as files in a directory."""

    def __init__(self, directory: str | pathlib.Path):
        self.directory = pathlib.Path(directory)

    def names(self) -> set[str]:
        if not self.directory.is_dir():
            return set()
        return {p.name for p in self.directory.iterdir() if p.is_file()}

    def get(self, name: str) -> bytes | None:
        path = self.directory / name
        return path.read_bytes() if path.is_file() else None

    def put(self, name: str, data: bytes) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / name).write_bytes(data)


class ReleaseStore(Store):
    """Entries as the assets of a GitHub release, through the ``gh`` CLI
    (authenticated by ``GH_TOKEN`` in Actions, ``gh auth login`` elsewhere).
    The release is created on the first upload."""

    def __init__(self, tag: str, repo: str | None = None, run=subprocess.run):
        self.tag, self.repo, self._run = tag, repo, run
        self._names: set[str] | None = None

    def _gh(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        repo = ("--repo", self.repo) if self.repo else ()
        return self._run(
            ["gh", *args, *repo], check=check, capture_output=True, text=False
        )

    def names(self) -> set[str]:
        if self._names is None:
            done = self._gh("release", "view", self.tag, "--json", "assets",
                            check=False)  # fmt: skip
            if done.returncode:
                self._names = set()  # no release yet
            else:
                assets = json.loads(done.stdout)["assets"]
                self._names = {asset["name"] for asset in assets}
        return self._names

    def get(self, name: str) -> bytes | None:
        with tempfile.TemporaryDirectory() as tmp:
            done = self._gh("release", "download", self.tag, "--pattern", name,
                            "--dir", tmp, check=False)  # fmt: skip
            path = pathlib.Path(tmp) / name
            return path.read_bytes() if not done.returncode and path.exists() else None

    def put(self, name: str, data: bytes) -> None:
        if self._gh("release", "view", self.tag, check=False).returncode:
            # another machine may create it at the same time: that is fine
            self._gh("release", "create", self.tag, "--latest=false",
                     "--title", "Backtest cache",
                     "--notes", "Backtest seasons and trained models by key "
                     "(runitback.wager.web.cache); any build can reuse them.",
                     check=False)  # fmt: skip
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / name
            path.write_bytes(data)
            self._gh("release", "upload", self.tag, str(path), "--clobber")
        self.names().add(name)


def open_store(spec: str | None) -> Store | None:
    """``release:<tag>`` / ``release:<owner>/<repo>@<tag>`` -> a release
    store, any other string a directory store, ``None`` no store."""
    if not spec:
        return None
    if spec.startswith("release:"):
        target = spec.removeprefix("release:")
        repo, _, tag = target.rpartition("@")
        return ReleaseStore(tag, repo or None)
    return DirectoryStore(spec)
