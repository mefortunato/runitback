"""Foundation folds for the production wager model (``runitback nfl foundation``).

The NFL wager model's history tokens carry each team-game's play-by-play embedding
(experiments/nfl/README.md: the frozen 32-dimension embedding, ``pbp32``). The fold layout, the
walk-forward PCA and the incremental updates are sport-agnostic (``runitback.core.foundation``);
this module is the NFL's :class:`~runitback.core.foundation.Generator`:

    <root>/fold<S>/model.pt        trained on games before season S-1 week 15 (validated on the rest
                                   of S-1); never sees S or later
    <root>/fold<S>/pca_32.npz      the projection, fit on games before S
    <root>/fold<S>/pbp_32.parquet  every played game's embedding (game_id, team, _pbp_0..31)

The walk-forward backtest of season S uses fold S, and the model for upcoming games (validated
on the last complete season, testing the current one) uses the current season's fold. A fold is
trained once; afterwards ``prepare`` only adds the games played since (``embed.update_fold``),
with the stored projection, so the numbers a wager model was fit on never change.

The wager side reads the folds through ``RUNITBACK_NFL_FOUNDATION=<root>`` (runitback.wager.nfl).
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ..core import foundation as core

DIM = 32
# the experiment's fold settings (experiments/nfl/foundation.yaml, .github/workflows/foundation-experiment.yml)
# (`runitback nfl train --val-week 15 --epochs 10 --who --prev-lineup-prob 0.3` with the CLI's defaults)
TRAIN = dict(val_week=15, epochs=10, d_model=192, n_layers=4, n_heads=6, batch_size=32, lr=1e-3,
             use_lineups=True, player_dim=32, lineup_dropout=0.2, unk_lineup_prob=0.2,
             other_player_prob=0.05, prev_lineup_prob=0.3, box_weight=0.0, who_head=True)
FILES = core.fold_files(DIM)
fold_dir = core.fold_dir


class NFLGenerator:
    """The NFL play-by-play transformer (``runitback.nfl.train``) as a fold generator."""

    sport = "nfl"

    def train(self, data_dir: Path, work_dir: Path, val_season: int, threads: int | None = None) -> Path:
        from .train import train

        train([Path(data_dir)], Path(work_dir), val_season=int(val_season), threads=threads, **TRAIN)
        return Path(work_dir) / "model.pt"

    def embed(self, ckpt: Path, data_dir: Path, game_ids: Iterable[str] | None = None) -> pd.DataFrame:
        from .embed import game_embeddings

        return game_embeddings(ckpt, data_dir, game_ids=game_ids)

    def played_games(self, data_dir: Path) -> set[str]:
        games = pd.read_parquet(Path(data_dir) / "games.parquet", columns=["game_id"])
        played = pd.read_parquet(Path(data_dir) / "plays.parquet", columns=["game_id"]).game_id.unique()
        return set(games.game_id) & set(played)

    def current_season(self, data_dir: Path) -> int:
        plays = pd.read_parquet(Path(data_dir) / "plays.parquet", columns=["game_id"])
        games = pd.read_parquet(Path(data_dir) / "games.parquet", columns=["game_id", "season"])
        return int(games[games.game_id.isin(plays.game_id.unique())].season.max())


NFL = NFLGenerator()


def current_season(data_dir: Path) -> int:
    """The latest season with play-by-play: the season the model for upcoming games tests on."""
    return NFL.current_season(data_dir)


def train_fold(root: Path, season: int, data_dir: Path, threads: int | None = None) -> Path:
    """Train fold ``season``'s foundation model (never sees ``season``)."""
    return core.train_fold(NFL, root, season, data_dir, threads)


def prepare(root: Path, data_dir: Path, seasons: list[int], train: bool = True,
            threads: int | None = None) -> dict[int, str]:
    """Make each fold in ``seasons`` ready: train its model if it has none (``train``), embed every
    played game once, then add the games played since. Returns {season: what was done}."""
    return core.prepare(NFL, root, data_dir, seasons, train=train, threads=threads, dim=DIM)
