"""Per-game, per-team embeddings from a trained play transformer, for downstream game models.

Each finished game's real plays are run through the frozen model (``encode``) and the hidden
states are averaged over the plays where each team had the ball. A team's vector is
``[its offense, its defense]`` (its defense is the opponent's offense), so it has
``2 * d_model`` dimensions and describes how that game went, in context.

Team-season embeddings are set to UNK (index 0) for every play: they are learned from a whole
season's games, so they would carry later results into an early-season game's vector. The model
already sees UNK on about 10% of training games (``unk_ts_prob``), so this is in distribution.
Pre-game team-strength ratings are kept: they only use earlier games.

The output table has one row per team per game, ``emb_0 .. emb_{2d-1}``, plus ``checkpoint``
(the checkpoint's sha256) and ``in_sample`` (whether the checkpoint was trained on the game).
A downstream backtest must only use vectors from a checkpoint that was not trained on the season
it tests; ``in_sample`` lets it check.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from ..core.embedding import pool_by_possession, team_vectors
from ..core.training import batches
from .model.dataset import collate, encode_games
from .model.features import TeamSeasonVocab
from .model.transformer import ModelConfig, PlayTransformer
from .release import sha256
from .train import load_data

log = logging.getLogger(__name__)

KEY_COLUMNS = ["game_id", "season", "week", "team", "opp", "is_home", "checkpoint", "in_sample"]


def load_model(ckpt: Path) -> tuple[PlayTransformer, dict]:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = PlayTransformer(ModelConfig(**ck["config"]))
    model.load_state_dict(ck["state_dict"])
    return model.eval(), {k: v for k, v in ck.items() if k not in ("state_dict", "ratings")}


def trained_on(games: pd.DataFrame, meta: dict) -> np.ndarray:
    """Whether each game is in the checkpoint's training range.

    Uses ``train_last`` ((season, week) of the last training game, saved by ``train``) and falls
    back to the validation split for older checkpoints: training used every game before
    (``val_season``, ``val_week``), or every game when there was no validation season.
    """
    when = list(zip(games.season.astype(int), games.week.astype(int)))
    if meta.get("train_last"):
        last = tuple(meta["train_last"])
        return np.array([w <= last for w in when])
    if meta.get("val_season") is not None:
        cut = (int(meta["val_season"]), int(meta.get("val_week") or 1))
        return np.array([w < cut for w in when])
    return np.ones(len(games), dtype=bool)


@torch.inference_mode()
def embed_items(model: PlayTransformer, items: list[dict], batch_size: int = 64
                ) -> tuple[np.ndarray, np.ndarray]:
    """``(home, away)`` team vectors, each ``[len(items), 2 * d_model]``, in ``items`` order."""
    d = model.cfg.d_model
    home = np.zeros((len(items), 2 * d), dtype=np.float32)
    away = np.zeros_like(home)
    for b in batches(items, batch_size, np.random.default_rng(0), shuffle=False):
        batch = collate([items[i] for i in b])
        h = model.encode(batch["feats"])
        home_off, away_off = pool_by_possession(h, batch["valid"], batch["feats"]["is_home"])
        hv, av = team_vectors(home_off, away_off)
        home[b], away[b] = hv.numpy(), av.numpy()
    return home, away


def game_embeddings(ckpt: Path, data_dirs: Path | Sequence[Path], seasons: Sequence[int] | None = None,
                    batch_size: int = 64, threads: int | None = None) -> pd.DataFrame:
    """Embed every game in ``data_dirs`` (optionally only ``seasons``) with checkpoint ``ckpt``.

    Returns one row per team per game: ``KEY_COLUMNS`` then ``emb_0 .. emb_{2d-1}``.
    """
    if threads:
        torch.set_num_threads(threads)
    model, meta = load_model(ckpt)
    games, plays, ratings, _ = load_data(data_dirs, use_ratings=model.cfg.n_cont > 0)
    if seasons is not None:
        games = games[games.season.isin(list(seasons))]
    games = games[games.game_id.isin(plays.game_id)]
    plays = plays[plays.game_id.isin(games.game_id)]
    if games.empty:
        raise ValueError("no games with plays to embed")
    # an empty vocabulary maps every team-season to UNK (0)
    items = encode_games(plays, games, TeamSeasonVocab([]), model.cfg.max_len, ratings)
    log.info("embedding %d games with %s", len(items), ckpt)
    home, away = embed_items(model, items, batch_size)

    g = games.set_index("game_id").loc[[it["game_id"] for it in items]].reset_index()
    base = pd.DataFrame({"game_id": g.game_id.astype(str), "season": g.season.astype(int),
                         "week": g.week.astype(int), "checkpoint": sha256(Path(ckpt)),
                         "in_sample": trained_on(g, meta)})
    cols = [f"emb_{i}" for i in range(home.shape[1])]
    sides = [
        base.assign(team=g.home.values, opp=g.away.values, is_home=True).join(pd.DataFrame(home, columns=cols)),
        base.assign(team=g.away.values, opp=g.home.values, is_home=False).join(pd.DataFrame(away, columns=cols)),
    ]
    out = pd.concat(sides, ignore_index=True)
    return out[KEY_COLUMNS + cols].sort_values(["season", "week", "game_id", "is_home"],
                                                ascending=[True, True, True, False],
                                                ignore_index=True)
