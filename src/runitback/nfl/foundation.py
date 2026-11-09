"""Foundation folds for the production wager model (``runitback nfl foundation``).

The NFL wager model's history tokens carry each team-game's play-by-play embedding
(experiments/nfl/README.md: the frozen 32-dimension embedding, ``pbp32``). The embedding of a
game a wager model sees must come from a play-by-play model that never trained on the season
it predicts, so there is one foundation model per test season ("fold"):

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

import logging
import shutil
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

DIM = 32
# the experiment's fold settings (experiments/nfl/foundation.yaml, .github/workflows/foundation-experiment.yml)
# (`runitback nfl train --val-week 15 --epochs 10 --who --prev-lineup-prob 0.3` with the CLI's defaults)
TRAIN = dict(val_week=15, epochs=10, d_model=192, n_layers=4, n_heads=6, batch_size=32, lr=1e-3,
             use_lineups=True, player_dim=32, lineup_dropout=0.2, unk_lineup_prob=0.2,
             other_player_prob=0.05, prev_lineup_prob=0.3, box_weight=0.0, who_head=True)
FILES = ("model.pt", f"pca_{DIM}.npz", f"pbp_{DIM}.parquet")


def fold_dir(root: Path, season: int) -> Path:
    return Path(root) / f"fold{int(season)}"


def current_season(data_dir: Path) -> int:
    """The latest season with play-by-play: the season the model for upcoming games tests on."""
    plays = pd.read_parquet(Path(data_dir) / "plays.parquet", columns=["game_id"])
    games = pd.read_parquet(Path(data_dir) / "games.parquet", columns=["game_id", "season"])
    return int(games[games.game_id.isin(plays.game_id.unique())].season.max())


def train_fold(root: Path, season: int, data_dir: Path, threads: int | None = None) -> Path:
    """Train fold ``season``'s foundation model (never sees ``season``)."""
    from .train import train

    out = fold_dir(root, season)
    work = out / "train"
    train([Path(data_dir)], work, val_season=int(season) - 1, threads=threads, **TRAIN)
    out.mkdir(parents=True, exist_ok=True)
    shutil.move(str(work / "model.pt"), out / "model.pt")
    shutil.rmtree(work, ignore_errors=True)
    return out / "model.pt"


def prepare(root: Path, data_dir: Path, seasons: list[int], train: bool = True,
            threads: int | None = None) -> dict[int, str]:
    """Make each fold in ``seasons`` ready: train its model if it has none (``train``), embed every
    played game once, then add the games played since. Returns {season: what was done}."""
    from .embed import embed_fold, update_fold

    done = {}
    for s in sorted({int(x) for x in seasons}):
        d = fold_dir(root, s)
        ckpt = d / "model.pt"
        steps = []
        if not ckpt.exists():
            if not train:
                raise FileNotFoundError(f"{ckpt} is missing (run with training on, or fetch the fold)")
            train_fold(root, s, data_dir, threads)
            steps.append("trained")
        if not (d / f"pca_{DIM}.npz").exists():
            embed_fold(ckpt, Path(data_dir), d, before_season=s, dim=DIM)
            (d / "raw.parquet").unlink(missing_ok=True)  # 30 MB; new games are projected, not refit
            steps.append("embedded")
        else:
            _, added = update_fold(ckpt, Path(data_dir), d, DIM)
            steps.append(f"{added} new games")
        done[s] = ", ".join(steps)
        log.info("fold %d: %s", s, done[s])
    return done
