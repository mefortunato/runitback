"""Game embeddings from the NBA event model: the ``pbp32`` foundation for the NBA wager model.

Each played game is run through the event transformer's trunk under teacher forcing (the real
events and pre-game ratings), which gives one hidden state per event. Events are grouped into
**possessions**: maximal runs of consecutive events in the same period with the same team on
offense (``pos``). A possession's state is the mean over its events, so a trip to the line or a
timeout doesn't count for more than a one-shot possession. A team-game's embedding is the mean
possession state over the possessions the team had the ball (``off_*``) next to the mean over the
possessions it defended (``def_*``), one row per team and game, from its own side (as the NFL's
``runitback.nfl.embed``, which pools per play).

The fold layout, the walk-forward PCA (fit on games before the fold's test season) and the
incremental updates are sport-agnostic (``runitback.core.foundation``); ``embed_fold`` /
``update_fold`` run the NBA generator (``runitback.nba.foundation.NBA``) through them.
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
    ts = TeamSeasonVocab([])
    ts.keys = list(ck["team_seasons"])
    ts.idx = {k: i + 1 for i, k in enumerate(ts.keys)}
    return model.eval(), ts, ck


def encode_all(ckpt: Path, data_dir: Path, game_ids=None) -> tuple[EventTransformer, list[dict], pd.DataFrame]:
    """The model and every (or the given) processed game with events, as model inputs, plus the
    games table indexed by game_id."""
    model, ts, _ = load_foundation(ckpt)
    data_dir = Path(data_dir)
    games = pd.read_parquet(data_dir / "games.parquet")
    plays = pd.read_parquet(data_dir / "plays.parquet")
    if game_ids is not None:
        games = games[games.game_id.isin(set(game_ids))]
    rpath = data_dir / "ratings.parquet"
    ratings = pd.read_parquet(rpath) if (rpath.exists() and model.cfg.n_cont) else None
    plays = plays[plays.game_id.isin(games.game_id)]
    items = encode_games(plays, games, ts, model.cfg.max_len, ratings) if len(plays) else []
    return model, items, games.set_index("game_id")


def possession_ids(is_home: np.ndarray, qtr: np.ndarray) -> np.ndarray:
    """Per event, the index of its possession within the game: a new one starts whenever the team
    on offense or the period changes."""
    if len(is_home) == 0:
        return np.zeros(0, dtype=np.int64)
    new = np.r_[True, (is_home[1:] != is_home[:-1]) | (qtr[1:] != qtr[:-1])]
    return np.cumsum(new) - 1


def possession_weights(is_home: torch.Tensor, qtr: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """(B, T) event weights that make each possession count once: 1 / (its number of events) on
    valid events, 0 on padding."""
    w = torch.zeros_like(valid, dtype=torch.float32)
    for i in range(valid.shape[0]):
        n = int(valid[i].sum())
        pid = possession_ids(is_home[i, :n].numpy(), qtr[i, :n].numpy())
        size = np.bincount(pid)
        w[i, :n] = torch.as_tensor(1.0 / size[pid], dtype=torch.float32)
    return w


def pool_sides(h: torch.Tensor, is_home: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """(B, T, d) event states -> (B, 2, 2d): [home, away] x [own offense, own defense], each the
    ``weight``-ed mean over the events of that side's possessions. With ``possession_weights``
    that is the mean over possessions of each possession's mean state. A side with no
    possessions (cannot happen in a full game) pools to zeros."""
    home = is_home.float() * weight
    away = (1 - is_home.float()) * weight

    def mean(w):
        return (h * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1e-9)

    mh, ma = mean(home), mean(away)
    mh = torch.where(home.sum(1, keepdim=True) > 0, mh, torch.zeros_like(mh))
    ma = torch.where(away.sum(1, keepdim=True) > 0, ma, torch.zeros_like(ma))
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
        f = b["feats"]
        w = possession_weights(f["is_home"], f["qtr"], b["valid"])
        v = pool_sides(model.encode(f), f["is_home"], w)
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
    """Write <out>/raw.parquet (per team-game raw embeddings), <out>/pbp_<dim>.parquet (PCA fit on
    games before `before_season`) and <out>/pca_<dim>.npz (runitback.core.foundation.embed_fold)."""
    from .foundation import NBA
    return foundation.embed_fold(NBA, ckpt, data_dir, out, before_season, dim)


def update_fold(ckpt: Path, data_dir: Path, out: Path, dim: int = 32) -> tuple[Path, int]:
    """Add the played games missing from <out>/pbp_<dim>.parquet with the fold's stored projection
    (runitback.core.foundation.update_fold). Returns the table and how many games were added."""
    from .foundation import NBA
    return foundation.update_fold(NBA, ckpt, data_dir, out, dim)
