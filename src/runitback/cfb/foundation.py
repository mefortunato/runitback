"""College football foundation folds (``runitback cfb foundation``): the ``pbp32`` layer for the
CFB wager model (docs/multisport/PLAN.md, F1/F3).

The generator is the NFL play transformer (``runitback.nfl.train``) trained on college plays
(``runitback.cfb.data.preprocess``, the NFL's schema), team-level: there is no college personnel
data, so no lineups, coaches or who head. Teams are ESPN team ids; team strength comes from the
team-season tokens and the pre-game ratings (EWMA EPA/play and Elo). The fold layout, the
walk-forward PCA and the incremental updates are sport-agnostic (``runitback.core.foundation``):

    <root>/fold<S>/model.pt        trained on games before season S-1 week 10 (validated on the rest
                                   of S-1, bowls included); never sees S or later
    <root>/fold<S>/pca_32.npz      the projection, fit on games before S
    <root>/fold<S>/pbp_32.parquet  every played game's embedding (game_id, team, season, _pbp_0..31)

``team`` is the ESPN team id (a string): the CFB wager model keys teams by school name, with FCS
schools pooled, so its integration (F3) joins through the schedule's ``home_team_id`` /
``away_team_id``.
"""
from __future__ import annotations

from pathlib import Path

from ..core import foundation as core
from ..nfl.foundation import NFLGenerator

DIM = 32
# fold settings (.github/workflows/cfb-foundation.yml): the NFL fold's trunk and epochs without its
# lineup inputs. College seasons have ~880 games, so a fold trains on up to ~18,000 games (2.5x the
# NFL's); about 5 minutes an epoch on a 4-core runner.
EPOCHS = 10
TRAIN = dict(val_week=10, epochs=EPOCHS, d_model=192, n_layers=4, n_heads=6, batch_size=32, lr=1e-3,
             use_lineups=False, who_head=False, box_weight=0.0, min_season=2004)
FILES = core.fold_files(DIM)
fold_dir = core.fold_dir


class CFBGenerator(NFLGenerator):
    """The play transformer trained on college football plays, as a fold generator. Embedding and
    the played-games bookkeeping are the NFL generator's (same processed-data schema)."""

    sport = "cfb"

    def __init__(self, **overrides):
        self.settings = TRAIN | overrides

    def train(self, data_dir: Path, work_dir: Path, val_season: int | None, threads: int | None = None) -> Path:
        """Train on every game before ``val_season``'s week ``val_week`` (None: on every game, with no
        validation)."""
        from ..nfl.train import train

        train([Path(data_dir)], Path(work_dir), val_season=None if val_season is None else int(val_season),
              threads=threads, **self.settings)
        return Path(work_dir) / "model.pt"


CFB = CFBGenerator()


def prepare(root: Path, data_dir: Path, seasons: list[int], train: bool = True, threads: int | None = None,
            generator: CFBGenerator = CFB) -> dict[int, str]:
    """Make each fold in ``seasons`` ready: train its model if it has none (``train``), embed every
    played game once, then add the games played since. Returns {season: what was done}."""
    return core.prepare(generator, root, data_dir, seasons, train=train, threads=threads, dim=DIM)
