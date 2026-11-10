"""Game embeddings from the play-by-play model: the foundation the wager models build on.

Each played game is run through the play transformer's trunk under teacher forcing (the real plays,
real lineups and personnel), which gives one hidden state per play. A team-game's embedding is the
mean hidden state over the plays the team had the ball (``off_*``) next to the mean over the plays
it defended (``def_*``), so both teams of a game get their own row, from their own side.

``reduce`` (sport-agnostic, ``runitback.core.foundation``) projects the raw embeddings onto principal
components fit only on games before a given season (the fold's test season), so a walk-forward wager
model never sees a projection fit on the games it predicts. The result is a table keyed by
(game_id, team) with ``_pbp_<i>`` columns that the wager transformer joins onto its history tokens
(``runitback.wager``, ``pbp_embeddings``). ``embed_fold`` / ``update_fold`` run the NFL generator
through the generic fold code (``runitback.nfl.foundation.NFL``).

``lower_states`` returns each game's per-play states after the first ``n_layers - k`` blocks, so the
top ``k`` blocks can be fine-tuned on the wager loss without recomputing the frozen ones.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..core import foundation
from ..core.foundation import (  # noqa: F401  (re-exported: the fold's PCA)
    PREFIX,
    project,
    reduce,
)
from .model.dataset import TeamSeasonVocab, collate, encode_games
from .model.lineups import CoachVocab, LineupData, PlayerVocab
from .model.transformer import ModelConfig, PlayTransformer

log = logging.getLogger(__name__)


def load_foundation(ckpt: Path) -> tuple[PlayTransformer, TeamSeasonVocab, dict]:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = PlayTransformer(ModelConfig(**ck["config"]))
    model.load_state_dict(ck["state_dict"])
    ts = TeamSeasonVocab([])
    ts.keys = list(ck["team_seasons"])
    ts.idx = {k: i + 1 for i, k in enumerate(ts.keys)}
    return model.eval(), ts, ck


def encode_all(ckpt: Path, data_dir: Path, game_ids=None) -> tuple[PlayTransformer, list[dict], pd.DataFrame]:
    """The foundation model and every (or the given) processed game as model inputs."""
    model, ts, ck = load_foundation(ckpt)
    games = pd.read_parquet(data_dir / "games.parquet")
    plays = pd.read_parquet(data_dir / "plays.parquet")
    if game_ids is not None:
        games = games[games.game_id.isin(set(game_ids))]
    rpath = data_dir / "ratings.parquet"
    ratings = pd.read_parquet(rpath) if (rpath.exists() and model.cfg.n_cont) else None
    ld = None
    lm = ck.get("lineup_meta")
    if lm and model.lineup is not None and (data_dir / "player_games.parquet").exists():
        ld = LineupData.load(data_dir, [], PlayerVocab(lm["player_vocab"]), CoachVocab(lm["coach_vocab"]))
    items = encode_games(plays[plays.game_id.isin(games.game_id)], games, ts, model.cfg.max_len, ratings, ld)
    return model, items, games.set_index("game_id")


def _run_blocks(model: PlayTransformer, feats: dict, upto: int | None = None) -> torch.Tensor:
    """model.encode, stopping after `upto` blocks (None: all blocks and the final LayerNorm)."""
    x = sum(model.emb[f](feats[f]) for f in model.emb)
    if model.cont is not None:
        x = x + model.cont(feats["cont"])
    if model.lineup is not None:
        x = x + model.lineup_context(feats)
    T = x.shape[1]
    x = x + model.pos(torch.arange(T).clamp(max=model.cfg.max_len - 1))
    x = model.drop(model.in_ln(x))
    blocks = model.blocks if upto is None else model.blocks[:upto]
    for i, b in enumerate(blocks):
        x = b(x, None, i)
    return model.ln_f(x) if upto is None else x


def pool_sides(h: torch.Tensor, is_home: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """(B, T, d) play states -> (B, 2, 2d): [home, away] x [mean over own offense, own defense].
    A side with no plays (cannot happen in a full game) pools to zeros."""
    home = is_home.float() * valid
    away = (1 - is_home.float()) * valid

    def mean(w):
        return (h * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1)

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
        f = b["feats"]
        v = pool_sides(_run_blocks(model, f), f["is_home"], b["valid"])
        for k, it in enumerate(chunk):
            g = games.loc[it["game_id"]]
            rows += [(it["game_id"], g.home, int(g.season)), (it["game_id"], g.away, int(g.season))]
            vecs += [v[k, 0].numpy(), v[k, 1].numpy()]
        if (i // batch_size) % 50 == 0:
            log.info("embedded %d/%d games", min(i + batch_size, len(items)), len(items))
    cols = [f"off_{j}" for j in range(d)] + [f"def_{j}" for j in range(d)]
    out = pd.DataFrame(np.stack(vecs), columns=cols)
    return pd.concat([pd.DataFrame(rows, columns=["game_id", "team", "season"]), out], axis=1)


@torch.no_grad()
def lower_states(model: PlayTransformer, items: list[dict], trainable: int, batch_size: int = 32
                 ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """{game_id: (states after the frozen blocks (T, d) as float16, is_home (T,))}."""
    upto = len(model.blocks) - trainable
    out = {}
    for i in range(0, len(items), batch_size):
        chunk = items[i: i + batch_size]
        b = collate(chunk)
        h = _run_blocks(model, b["feats"], upto)
        for k, it in enumerate(chunk):
            n = len(it["type"])
            out[it["game_id"]] = (h[k, :n].half(), b["feats"]["is_home"][k, :n].bool())
    return out


def embed_fold(ckpt: Path, data_dir: Path, out: Path, before_season: int, dim: int = 32) -> Path:
    """Write <out>/raw.parquet (per team-game raw embeddings), <out>/pbp_<dim>.parquet (PCA fit on
    games before `before_season`) and <out>/pca_<dim>.npz (runitback.core.foundation.embed_fold)."""
    from .foundation import NFL
    return foundation.embed_fold(NFL, ckpt, data_dir, out, before_season, dim)


def update_fold(ckpt: Path, data_dir: Path, out: Path, dim: int = 32) -> tuple[Path, int]:
    """Add the played games missing from <out>/pbp_<dim>.parquet (e.g. this week's) with the
    fold's stored projection (runitback.core.foundation.update_fold). Returns the table and how
    many games were added."""
    from .foundation import NFL
    return foundation.update_fold(NFL, ckpt, data_dir, out, dim)
