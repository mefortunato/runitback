"""Foundation folds for the basketball wager models (``runitback nba|cbb|wcbb foundation``): ``pbp32``.

The NBA wager model's history tokens can carry each team-game's embedding from the event
generator (docs/multisport/PLAN.md, layer A; work package N4). The fold layout, the walk-forward
PCA and the incremental updates are sport-agnostic (``runitback.core.foundation``); this module
is the NBA's :class:`~runitback.core.foundation.Generator`:

    <root>/fold<S>/model.pt        ``runitback.nba.train`` on games before the last 40% (by date) of
                                   season S-1, validated on that 40%; never sees S or later
    <root>/fold<S>/pca_32.npz      the projection, fit on games before S
    <root>/fold<S>/pbp_32.parquet  every played game's embedding (game_id, team, season, _pbp_0..31),
                                   pooled per possession (``runitback.nba.embed``)

Seasons are named by the year they end in (2026 = 2025-26). The walk-forward backtest of season S
uses fold S, and the model for upcoming games uses the current season's fold. A fold is trained
once; afterwards ``prepare`` only adds the games played since, with the stored projection, so the
numbers a wager model was fit on never change.

The same generator code runs the college leagues (C2): :class:`BasketballGenerator` takes a
:class:`~runitback.nba.league.League` (rules, vocabulary, team list) and the fold recipe. The college
folds train on the five seasons before the test season (``League.train_seasons``): 360+ teams and
five seasons of 5,000-plus games each is already as many events as the NBA's two decades.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ..core import foundation as core
from .league import CBB, NBA as NBA_LEAGUE, WCBB, League

DIM = 32
# The fold's training settings: the production NBA recipe (`runitback nba train`'s defaults, README
# "NBA pipeline"), validated on the end of the season before the fold's test season. Kept in one
# place so the CLI and .github/workflows/nba-experiment-inputs.yml train the same model.
TRAIN = dict(val_fraction=0.4, epochs=6, d_model=192, n_layers=4, n_heads=6, batch_size=32, lr=1e-3,
             dropout=0.1, min_season=2002, unk_ts_prob=0.1, unk_cont_prob=0.05, use_ratings=True, seed=0)
FILES = core.fold_files(DIM)
fold_dir = core.fold_dir


def _played(data_dir: Path) -> pd.DataFrame:
    games = pd.read_parquet(Path(data_dir) / "games.parquet", columns=["game_id", "season"])
    played = pd.read_parquet(Path(data_dir) / "plays.parquet", columns=["game_id"]).game_id.unique()
    return games[games.game_id.isin(played)]


class BasketballGenerator:
    """The basketball event transformer (``runitback.nba.train``) as a fold generator, for one league."""

    def __init__(self, league: League = NBA_LEAGUE, **overrides):
        self.league = league
        self.sport = league.name
        self.settings = TRAIN | overrides

    def train(self, data_dir: Path, work_dir: Path, val_season: int, threads: int | None = None) -> Path:
        from .train import train

        settings = dict(self.settings)
        if self.league.train_seasons:  # the seasons up to and including the validation season
            settings["min_season"] = max(settings["min_season"], int(val_season) - self.league.train_seasons + 1)
        return train([Path(data_dir)], Path(work_dir), val_season=int(val_season), threads=threads,
                     league=self.league, **settings)

    def embed(self, ckpt: Path, data_dir: Path, game_ids: Iterable[str] | None = None) -> pd.DataFrame:
        from .embed import game_embeddings

        return game_embeddings(ckpt, data_dir, game_ids=game_ids)

    def played_games(self, data_dir: Path) -> set[str]:
        return set(_played(data_dir).game_id)

    def current_season(self, data_dir: Path) -> int:
        return int(_played(data_dir).season.max())


class NBAGenerator(BasketballGenerator):
    """The NBA event transformer as a fold generator."""

    def __init__(self, **overrides):
        super().__init__(NBA_LEAGUE, **overrides)


NBA = NBAGenerator()
CBB_GENERATOR = BasketballGenerator(CBB)
WCBB_GENERATOR = BasketballGenerator(WCBB)
GENERATORS = {"nba": NBA, "cbb": CBB_GENERATOR, "wcbb": WCBB_GENERATOR}


def current_season(self, data_dir: Path) -> int:
        return int(_played(data_dir).season.max())


NBA = NBAGenerator()


def current_season(data_dir: Path) -> int:
    """The latest season with events: the season the model for upcoming games tests on."""
    return NBA.current_season(data_dir)


def train_fold(root: Path, season: int, data_dir: Path, threads: int | None = None,
               gen: BasketballGenerator = NBA) -> Path:
    """Train fold ``season``'s generator (validated on the end of ``season - 1``; never sees ``season``)."""
    return core.train_fold(gen, root, season, data_dir, threads)


def prepare(root: Path, data_dir: Path, seasons: Iterable[int], train: bool = True,
            threads: int | None = None, gen: BasketballGenerator = NBA) -> dict[int, str]:
    """Make each fold in ``seasons`` ready: train its model if it has none (``train``), embed every
    played game once, then add the games played since. Returns {season: what was done}."""
    return core.prepare(gen, root, data_dir, seasons, train=train, threads=threads, dim=DIM)
