"""Foundation folds for the NHL wager model (``runitback nhl foundation``): ``pbp32``.

The NHL wager model's history tokens can carry each team-game's embedding from the event
generator (docs/multisport/PLAN.md, layer A; work package H5). The fold layout, the walk-forward
PCA and the incremental updates are sport-agnostic (``runitback.core.foundation``); this module
is the NHL's :class:`~runitback.core.foundation.Generator`:

    <root>/fold<S>/model.pt        ``runitback.nhl.train`` on games before the last 40% (by date) of
                                   season S-1, validated on that 40%; never sees S or later
    <root>/fold<S>/pca_32.npz      the projection, fit on games before S
    <root>/fold<S>/pbp_32.parquet  every played game's embedding (game_id, team, season, _pbp_0..31),
                                   pooled over each team's own and its opponent's events
                                   (``runitback.nhl.embed``)

Seasons are named by the year they end in (2026 = 2025-26). The data directory is
``runitback nhl preprocess``'s (``data/nhl/plays``): games, the replayed plays and pre-game ratings.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ..core import foundation as core

DIM = 32
# The fold's training settings, kept in one place so the CLI and
# .github/workflows/nhl-experiment-inputs.yml train the same model. The NBA's recipe (d_model 192,
# 4 layers, pre-game ratings) with more epochs: the NHL has about a third of the NBA's training
# events (2011+, ~320 events a game), and the NBA folds were still improving at 6 epochs.
TRAIN = dict(val_fraction=0.4, epochs=10, d_model=192, n_layers=4, n_heads=6, batch_size=32, lr=1e-3,
             dropout=0.1, min_season=2011, unk_ts_prob=0.1, unk_cont_prob=0.05, use_ratings=True, seed=0)
FILES = core.fold_files(DIM)
fold_dir = core.fold_dir


def _games(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / "games.parquet", columns=["game_id", "season"])


class NHLGenerator:
    """The NHL event transformer (``runitback.nhl.train``) as a fold generator."""

    sport = "nhl"

    def __init__(self, **overrides):
        self.settings = TRAIN | overrides

    def train(self, data_dir: Path, work_dir: Path, val_season: int, threads: int | None = None) -> Path:
        from .train import train

        return train(Path(data_dir), Path(work_dir), val_season=int(val_season), threads=threads, **self.settings)

    def embed(self, ckpt: Path, data_dir: Path, game_ids: Iterable[str] | None = None) -> pd.DataFrame:
        from .embed import game_embeddings

        return game_embeddings(ckpt, data_dir, game_ids=game_ids)

    def played_games(self, data_dir: Path) -> set[str]:
        # games.parquet holds only games the replay finished (every one has events)
        return set(_games(data_dir).game_id)

    def current_season(self, data_dir: Path) -> int:
        return int(_games(data_dir).season.max())


NHL = NHLGenerator()


def current_season(data_dir: Path) -> int:
    """The latest season with events: the season the model for upcoming games tests on."""
    return NHL.current_season(data_dir)


def train_fold(root: Path, season: int, data_dir: Path, threads: int | None = None,
               gen: NHLGenerator = NHL) -> Path:
    """Train fold ``season``'s generator (validated on the end of ``season - 1``; never sees ``season``)."""
    return core.train_fold(gen, root, season, data_dir, threads)


def prepare(root: Path, data_dir: Path, seasons: Iterable[int], train: bool = True,
            threads: int | None = None, gen: NHLGenerator = NHL) -> dict[int, str]:
    """Make each fold in ``seasons`` ready: train its model if it has none (``train``), embed every
    played game once, then add the games played since. Returns {season: what was done}."""
    return core.prepare(gen, root, data_dir, seasons, train=train, threads=threads, dim=DIM)
