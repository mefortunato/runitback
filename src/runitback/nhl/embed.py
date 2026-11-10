"""Game embeddings from the NHL event model: the ``pbp32`` foundation for the NHL wager model.

Each played game is run through the event transformer's trunk under teacher forcing (the real
events and pre-game ratings), which gives one hidden state per event. Hockey has no possessions, so
events are pooled by the team that made them: a team-game's embedding is the mean state over the
events credited to the team (its shot attempts, faceoff wins, hits, giveaways, takeaways,
penalties, goalie pulls: ``off_*``) next to the mean over the events credited to its opponent
(``def_*``), one row per team and game. Events with no team (stoppages, period ends) are left out.
The state at an event is the model's view just before it (it has not seen the event), as in the
NFL's per-play and the NBA's per-possession pooling.

The trunk's inputs are from the home team's side (score difference home - away, home and away team
embeddings), so the same team's rows differ a little between home and away games; the PCA sees both.

The fold layout, the walk-forward PCA (fit on games before the fold's test season) and the
incremental updates are sport-agnostic (``runitback.core.foundation``); ``embed_fold`` /
``update_fold`` run the NHL generator (``runitback.nhl.foundation.NHL``) through them.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..core import foundation
from ..core.foundation import PREFIX, project, reduce  # noqa: F401  (re-exported: the fold's PCA)
from .model.dataset import collate, encode_games
from .model.features import TeamSeasonVocab
from .model.transformer import EventTransformer, ModelConfig

log = logging.getLogger(__name__)


def load_foundation(ckpt: Path) -> tuple[EventTransformer, TeamSeasonVocab, dict]:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = EventTransformer(ModelConfig(**ck["config"]))
    model.load_state_dict(ck["state_dict"])
    return model.eval(), TeamSeasonVocab.from_keys(ck["team_seasons"]), ck


def encode_all(ckpt: Path, data_dir: Path, game_ids=None) -> tuple[EventTransformer, list[dict], pd.DataFrame]:
    """The model and every (or the given) game with events, as model inputs, plus the games table
    indexed by game_id. Pre-game ratings come from the data directory (they use only earlier games)."""
    model, ts, _ = load_foundation(ckpt)
    data_dir = Path(data_dir)
    games = pd.read_parquet(data_dir / "games.parquet")
    if game_ids is not None:
        games = games[games.game_id.isin({str(g) for g in game_ids})]
    plays = pd.read_parquet(data_dir / "plays.parquet")
    plays = plays[plays.game_id.isin(games.game_id)]
    rpath = data_dir / "ratings.parquet"
    ratings = pd.read_parquet(rpath) if (rpath.exists() and model.cfg.n_cont) else None
    items = encode_games(plays, games, ts, model.cfg.max_len, ratings) if len(plays) else []
    return model, items, games.set_index("game_id")


def pool_sides(h: torch.Tensor, side: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """(B, T, d) event states -> (B, 2, 2d): [home, away] x [own events, opponent's events], each the
    mean state over the events credited to that team (side 0 = home, 1 = away; 2 = no team). A team
    with no events pools to zeros (cannot happen in a full game)."""
    home = ((side == 0) & (valid > 0)).float()
    away = ((side == 1) & (valid > 0)).float()

    def mean(w):
        m = (h * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1.0)
        return torch.where(w.sum(1, keepdim=True) > 0, m, torch.zeros_like(m))

    mh, ma = mean(home), mean(away)
    return torch.stack([torch.cat([mh, ma], -1), torch.cat([ma, mh], -1)], 1)


@torch.no_grad()
def game_embeddings(ckpt: Path, data_dir: Path, batch_size: int = 32, game_ids=None) -> pd.DataFrame:
    """Raw per-team-game embeddings: game_id, team, season, off_0.., def_0.. (2 * d_model), of every
    played game (or of the given ones)."""
    model, items, games = encode_all(ckpt, data_dir, game_ids)
    items = [it for it in items if len(it["type"])]
    d = model.cfg.d_model
    rows, vecs = [], []
    for i in range(0, len(items), batch_size):
        chunk = items[i: i + batch_size]
        b = collate(chunk)
        v = pool_sides(model.encode(b["feats"]), b["side"], b["valid"])
        for k, it in enumerate(chunk):
            g = games.loc[it["game_id"]]
            rows += [(it["game_id"], g.home, int(g.season)), (it["game_id"], g.away, int(g.season))]
            vecs += [v[k, 0].numpy(), v[k, 1].numpy()]
        if (i // batch_size) % 50 == 0:
            log.info("embedded %d/%d games", min(i + batch_size, len(items)), len(items))
    cols = [f"off_{j}" for j in range(d)] + [f"def_{j}" for j in range(d)]
    out = pd.DataFrame(np.stack(vecs) if vecs else np.zeros((0, 2 * d), np.float32), columns=cols)
    return pd.concat([pd.DataFrame(rows, columns=["game_id", "team", "season"]), out], axis=1)


def embed_fold(ckpt: Path, data_dir: Path, out: Path, before_season: int, dim: int = 32) -> Path:
    """Write <out>/raw.parquet, <out>/pbp_<dim>.parquet (PCA fit on games before `before_season`)
    and <out>/pca_<dim>.npz (runitback.core.foundation.embed_fold)."""
    from .foundation import NHL
    return foundation.embed_fold(NHL, ckpt, data_dir, out, before_season, dim)


def update_fold(ckpt: Path, data_dir: Path, out: Path, dim: int = 32) -> tuple[Path, int]:
    """Add the played games missing from <out>/pbp_<dim>.parquet with the fold's stored projection."""
    from .foundation import NHL
    return foundation.update_fold(NHL, ckpt, data_dir, out, dim)
