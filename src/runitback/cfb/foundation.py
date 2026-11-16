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

import pandas as pd

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


def wager_keys(emb: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """A fold's embeddings (``game_id, team`` = ESPN team id, ``season``, ``_pbp_*``) keyed as the CFB
    wager model keys its team-games: ``team`` is the school name, or ``FCS`` for an FCS side (the
    pooled team), through the schedule's ``home_id`` / ``away_id`` and divisions. Games without an
    FBS team have no wager row and are dropped, as are team ids the schedule doesn't know."""
    from ..wager.cfb.data.download import FCS

    s = schedule.assign(game_id=schedule.game_id.astype(str))
    s = s[(s.home_division == "fbs") | (s.away_division == "fbs")]
    names = pd.concat([
        pd.DataFrame({"game_id": s.game_id, "team_id": pd.to_numeric(s[f"{side}_id"], errors="coerce"),
                      "wager_team": s[f"{side}_team"].where(s[f"{side}_division"] == "fbs", FCS)})
        for side in ("home", "away")]).dropna(subset=["team_id"]).astype({"team_id": "int64"})
    names = names.drop_duplicates(["game_id", "team_id"])
    out = emb.assign(game_id=emb.game_id.astype(str), team_id=pd.to_numeric(emb.team, errors="coerce"))
    out = out.dropna(subset=["team_id"]).astype({"team_id": "int64"})
    out = out.merge(names, on=["game_id", "team_id"], how="inner")
    out["team"] = out.pop("wager_team")
    return out.drop(columns="team_id").reset_index(drop=True)


def write_wager_keys(root: Path, seasons: list[int], schedule: pd.DataFrame) -> dict[int, int]:
    """Write ``<root>/fold<S>/pbp_32_wager.parquet`` (:func:`wager_keys`) for each fold; returns the
    team-games kept per season."""
    kept = {}
    for s in seasons:
        d = fold_dir(Path(root), s)
        out = wager_keys(pd.read_parquet(d / FILES[2]), schedule)
        out.to_parquet(d / "pbp_32_wager.parquet", index=False)
        kept[s] = len(out)
    return kept
