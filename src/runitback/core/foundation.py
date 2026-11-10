"""Foundation folds for any sport's play-by-play generator (the wager models' ``pbp32`` layer).

A wager model's history tokens can carry each team-game's embedding from a play-by-play
generator (experiments/nfl/README.md). The embedding of a game a wager model sees must come from a
generator that never trained on the season it predicts, so there is one generator per test season
("fold"):

    <root>/fold<S>/model.pt          trained on games before season S (validated inside S-1, as the
                                     sport's generator decides); never sees S or later
    <root>/fold<S>/pca_<dim>.npz     the projection, fit on games before S
    <root>/fold<S>/pbp_<dim>.parquet every played game's embedding (game_id, team, season, _pbp_0..)

A fold is trained once (``prepare``); afterwards it only gets the games played since
(``update_fold``), projected with the stored PCA, so the numbers a wager model was fit on never change.

What is sport-specific sits behind :class:`Generator`: how to train a fold's model, how to turn a
checkpoint and the processed data into raw per-team-game embeddings, and which games have been
played. Everything else (the fold layout, the walk-forward PCA, incremental updates) lives here.
The NFL's generator is ``runitback.nfl.foundation.NFL``.
"""
from __future__ import annotations

import logging
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PREFIX = "_pbp_"
RAW_COLUMNS = r"^(off|def)_\d+$"
KEYS = ["game_id", "team", "season"]
CKPT = "model.pt"


@runtime_checkable
class Generator(Protocol):
    """A sport's play-by-play generator, as the fold code needs it."""

    sport: str

    def train(self, data_dir: Path, work_dir: Path, val_season: int, threads: int | None = None) -> Path:
        """Train a fold's model on ``data_dir`` in ``work_dir`` without seeing any season after
        ``val_season`` (the fold's test season minus 1). Returns the checkpoint written."""
        ...

    def embed(self, ckpt: Path, data_dir: Path, game_ids: Iterable[str] | None = None) -> pd.DataFrame:
        """Raw per-team-game embeddings of every played game (or of ``game_ids``): one row per team
        and game with ``game_id``, ``team``, ``season`` and float columns ``off_<i>`` / ``def_<i>``
        (the states pooled over the team's offensive and defensive events)."""
        ...

    def played_games(self, data_dir: Path) -> set[str]:
        """Ids of the games in ``data_dir`` that have events to embed."""
        ...

    def current_season(self, data_dir: Path) -> int:
        """The latest season with events: the season the model for upcoming games tests on."""
        ...


def fold_dir(root: Path, season: int) -> Path:
    return Path(root) / f"fold{int(season)}"


def fold_files(dim: int) -> tuple[str, str, str]:
    """The files that make a fold ready for the wager model."""
    return (CKPT, f"pca_{dim}.npz", f"pbp_{dim}.parquet")


def existing_folds(root: Path) -> list[int]:
    """Test seasons of the folds under ``root`` that have a trained model."""
    return sorted(int(d.name[4:]) for d in Path(root).glob("fold*") if (d / CKPT).exists())


def default_seasons(gen: Generator, data_dir: Path, backtest: int = 5) -> list[int]:
    """The backtest's last ``backtest`` complete seasons plus the current one."""
    now = gen.current_season(data_dir)
    return list(range(now - backtest, now + 1))


def reduce(raw: pd.DataFrame, before_season: int, dim: int = 32) -> tuple[pd.DataFrame, dict]:
    """Project raw embeddings onto the top `dim` principal components of the games before
    `before_season` (standardized per column on those games). Returns the ``_pbp_*`` table and the
    projection (mean, std, components, explained variance)."""
    x = raw.filter(regex=RAW_COLUMNS).to_numpy(np.float64)
    fit = (raw.season < before_season).to_numpy()
    if fit.sum() <= dim:
        raise ValueError(f"{int(fit.sum())} team-games before season {before_season}: too few to fit a "
                         f"{dim}-dimension projection")
    mean, std = x[fit].mean(0), x[fit].std(0) + 1e-6
    z = (x - mean) / std
    _, s, vt = np.linalg.svd(z[fit], full_matrices=False)
    comp = vt[:dim]
    var = (s ** 2) / (s ** 2).sum()
    fitted = {"mean": mean, "std": std, "components": comp, "explained": var[:dim]}
    return project(raw, fitted), fitted


def project(raw: pd.DataFrame, proj) -> pd.DataFrame:
    """Raw embeddings onto a fold's stored projection (``reduce``'s, or its ``pca_<dim>.npz``)."""
    x = raw.filter(regex=RAW_COLUMNS).to_numpy(np.float64)
    z = ((x - proj["mean"]) / proj["std"]) @ np.asarray(proj["components"]).T
    out = raw[KEYS].copy().reset_index(drop=True)
    for j in range(z.shape[1]):
        out[f"{PREFIX}{j}"] = z[:, j].astype(np.float32)
    return out


def load_projection(out: Path, dim: int) -> dict:
    with np.load(Path(out) / f"pca_{dim}.npz") as f:
        return {k: f[k] for k in f.files}


def embed_fold(gen: Generator, ckpt: Path, data_dir: Path, out: Path, before_season: int, dim: int = 32) -> Path:
    """Write <out>/raw.parquet (per team-game raw embeddings; reused when present),
    <out>/pbp_<dim>.parquet (PCA fit on games before `before_season`) and <out>/pca_<dim>.npz."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw.parquet"
    if raw_path.exists():
        raw = pd.read_parquet(raw_path)
    else:
        raw = gen.embed(Path(ckpt), Path(data_dir))
        raw.to_parquet(raw_path)
    red, proj = reduce(raw, before_season, dim)
    path = out / f"pbp_{dim}.parquet"
    red.to_parquet(path)
    np.savez(out / f"pca_{dim}.npz", **proj)
    log.info("fold %d: %d team-games, %d dims explain %.1f%% of the variance", before_season, len(red), dim,
             100 * proj["explained"].sum())
    return path


def update_fold(gen: Generator, ckpt: Path, data_dir: Path, out: Path, dim: int = 32) -> tuple[Path, int]:
    """Add the played games missing from <out>/pbp_<dim>.parquet (e.g. this week's) with the
    fold's stored projection: earlier rows and the projection stay as they are, so a wager model
    fit on the table sees the same numbers. Returns the table and how many games were added."""
    out = Path(out)
    path = out / f"pbp_{dim}.parquet"
    table = pd.read_parquet(path)
    proj = load_projection(out, dim)
    new = sorted(set(gen.played_games(Path(data_dir))) - set(table.game_id))
    if new:
        add = project(gen.embed(Path(ckpt), Path(data_dir), game_ids=new), proj)
        table = pd.concat([table, add], ignore_index=True)
        table.to_parquet(path)
    log.info("%s: %d team-games, %d games added", path, len(table), len(new))
    return path, len(new)


def train_fold(gen: Generator, root: Path, season: int, data_dir: Path, threads: int | None = None) -> Path:
    """Train fold ``season``'s generator (validated on ``season - 1``; never sees ``season``)."""
    out = fold_dir(root, season)
    work = out / "train"
    ckpt = gen.train(Path(data_dir), work, int(season) - 1, threads)
    out.mkdir(parents=True, exist_ok=True)
    shutil.move(str(ckpt), out / CKPT)
    shutil.rmtree(work, ignore_errors=True)
    return out / CKPT


def prepare(gen: Generator, root: Path, data_dir: Path, seasons: Iterable[int], train: bool = True,
            threads: int | None = None, dim: int = 32) -> dict[int, str]:
    """Make each fold in ``seasons`` ready: train its model if it has none (``train``), embed every
    played game once, then add the games played since. Returns {season: what was done}."""
    done = {}
    for s in sorted({int(x) for x in seasons}):
        d = fold_dir(root, s)
        ckpt = d / CKPT
        steps = []
        if not ckpt.exists():
            if not train:
                raise FileNotFoundError(f"{ckpt} is missing (run with training on, or fetch the fold)")
            train_fold(gen, root, s, data_dir, threads)
            steps.append("trained")
        if not (d / f"pca_{dim}.npz").exists():
            embed_fold(gen, ckpt, Path(data_dir), d, before_season=s, dim=dim)
            (d / "raw.parquet").unlink(missing_ok=True)  # large; new games are projected, not refit
            steps.append("embedded")
        else:
            _, added = update_fold(gen, ckpt, Path(data_dir), d, dim)
            steps.append(f"{added} new games")
        done[s] = ", ".join(steps)
        log.info("%s fold %d: %s", gen.sport, s, done[s])
    return done
