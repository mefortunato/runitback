"""Fine-tune the play-by-play foundation model on the wager loss.

The frozen arm (``pbp`` token option) reads fixed game embeddings: the foundation trunk's per-play
states, pooled per team-game and projected onto the fold's principal components
(:mod:`runitback.nfl.embed`). Here the top ``layers`` blocks of the trunk (and its final LayerNorm)
are trained further, together with the already fitted wager model, so the game embeddings can
change to suit the spread / moneyline / total prediction:

1. the frozen lower blocks' per-play states of every game are computed once (``lower_states``);
2. each step, the history games of the batch run through the trainable top blocks, are pooled and
   projected exactly like the frozen embeddings (so step 0 reproduces the frozen arm), and replace
   the ``_pbp_*`` token columns before the wager model's forward;
3. optionally (``aux_weight``) the foundation's own next-play loss on a sample of those games keeps
   the trunk usable as a play-by-play generator: its heads are frozen, so the top blocks must stay
   compatible with them;
4. after training, every game's embedding is recomputed and written to a new table, which the
   returned featurizer reads: serving and state refreshes work exactly as for the frozen arm. The
   fine-tuned foundation checkpoint is saved next to it for the generator.

Settings (``pbp_finetune`` dict): ``foundation`` (checkpoint path with ``{season}``), ``embed_dir``
(the fold's embedding directory with ``{season}``, holding ``pca_<dim>.npz``), ``data_dir``
(processed play-by-play), ``layers`` (top blocks to train, default 1), ``epochs`` (4),
``patience`` (2), ``lr`` (foundation learning rate, 3e-5), ``wager_lr`` (wager model, 5e-5),
``aux_weight`` (0.0), ``aux_games`` (16), ``recent_seasons`` (train on this many latest training
seasons, default 8), ``max_batches`` (cap per epoch, default none), ``out`` (tag for output names).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import PBP_PREFIX, SequenceData, SequenceFeatures, _PBP_CACHE, sequence

log = logging.getLogger(__name__)

DEFAULTS = dict(layers=1, epochs=4, patience=2, lr=3e-5, wager_lr=5e-5, aux_weight=0.0, aux_games=16,
                recent_seasons=8, max_batches=None, batch_size=128, out=None)


def wager_loss(module, batch) -> torch.Tensor:
    """The wager model's training loss (MarginModule._step without logging)."""
    if getattr(module, "distribution", False):
        raise NotImplementedError("pbp fine-tuning supports point (non-distribution) models")
    y = module.targets(batch)
    pred = module(batch)
    if y.dim() == 1:
        y, pred = y[:, None], pred[:, None]
    z = (y - module.target_mean) / module.target_std
    known = ~torch.isnan(z)
    error = (pred - torch.nan_to_num(z)) ** 2 * known
    per_target = error.sum(0) / known.sum(0).clamp(min=1)
    weights = getattr(module, "loss_weights", None)
    return per_target[0] if weights is None else (per_target * weights).sum()


class PBPTokens(torch.nn.Module):
    """Top trunk blocks + pooling + the fold's fixed PCA projection: game states -> _pbp_* values."""

    def __init__(self, foundation, layers: int, pca: dict, states: dict, game_ids: list[str]):
        super().__init__()
        from runitback.nfl.embed import pool_sides

        self.pool = pool_sides
        self.top = copy.deepcopy(foundation.blocks[len(foundation.blocks) - layers:])
        self.first = len(foundation.blocks) - layers
        self.ln_f = copy.deepcopy(foundation.ln_f)
        self.register_buffer("mean", torch.tensor(pca["mean"], dtype=torch.float32))
        self.register_buffer("std", torch.tensor(pca["std"], dtype=torch.float32))
        self.register_buffer("comp", torch.tensor(pca["components"], dtype=torch.float32))
        self.states = [states[g] for g in game_ids]
        for p in self.parameters():  # copied from the frozen foundation
            p.requires_grad_(True)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        for i, b in enumerate(self.top):
            x = b(x, None, self.first + i)
        return self.ln_f(x)

    def forward(self, games: torch.Tensor, sides: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        """games: (G,) indices into the state list; sides: (G,) 0 home / 1 away -> (G, dim)."""
        out = []
        for i in range(0, len(games), chunk):
            idx = games[i: i + chunk].tolist()
            T = max(len(self.states[g][0]) for g in idx)
            d = self.states[idx[0]][0].shape[-1]
            x = torch.zeros(len(idx), T, d)
            home = torch.zeros(len(idx), T, dtype=torch.bool)
            valid = torch.zeros(len(idx), T)
            for k, g in enumerate(idx):
                h, is_home = self.states[g]
                x[k, : len(h)] = h.float()
                home[k, : len(h)] = is_home
                valid[k, : len(h)] = 1
            v = self.pool(self.trunk(x), home, valid)                       # (n, 2, 2d)
            v = v[torch.arange(len(idx)), sides[i: i + chunk]]
            out.append(((v - self.mean) / self.std) @ self.comp.T)
        return torch.cat(out)


def _row_games(games: pd.DataFrame, schema, pbp_games: pd.DataFrame, game_ids: list[str]):
    """Each wager row's index into the foundation's game list (-1: no play-by-play) and side."""
    pos = {g: i for i, g in enumerate(game_ids)}
    meta = pbp_games.reindex(games[schema.game_id].to_numpy())
    gidx = np.array([pos.get(g, -1) for g in games[schema.game_id]], dtype=np.int64)
    side = np.where(meta["away"].to_numpy() == games[schema.team].to_numpy(), 1, 0)
    gidx[meta["home"].isna().to_numpy()] = -1
    return torch.from_numpy(gidx), torch.from_numpy(side.astype(np.int64))


def finetune(module, data: SequenceData, features: SequenceFeatures, games: pd.DataFrame, schema,
             rows: dict, settings: dict, fold: int, extra_targets: tuple):
    """Returns (module, data, features) with fine-tuned embeddings (see the module docstring)."""
    from runitback.nfl.embed import encode_all, lower_states

    cfg = DEFAULTS | settings
    tag = cfg["out"] or hashlib.sha1(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:8]
    ckpt = Path(cfg["foundation"].format(season=fold))
    embed_dir = Path(cfg["embed_dir"].format(season=fold))
    out_dir = embed_dir.parent / f"ft_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dim = features.pbp_dim
    pca = dict(np.load(embed_dir / f"pca_{dim}.npz"))
    t0 = time.time()
    foundation, items, pbp_games = encode_all(ckpt, Path(cfg["data_dir"]),
                                              game_ids=set(games[schema.game_id]))
    for p in foundation.parameters():
        p.requires_grad_(False)
    states = lower_states(foundation, items, cfg["layers"])
    game_ids = [it["game_id"] for it in items]
    item_of = {g: it for g, it in zip(game_ids, items)}
    tokens = PBPTokens(foundation, cfg["layers"], pca, states, game_ids)
    gidx, side = _row_games(games, schema, pbp_games, game_ids)
    log.info("fold %d: %d games' lower states in %.0fs", fold, len(states), time.time() - t0)

    cols = torch.tensor([features.token_columns.index(f"{PBP_PREFIX}{j}") for j in range(dim)])
    fmean = torch.tensor([features.mean[f"{PBP_PREFIX}{j}"] for j in range(dim)], dtype=torch.float32)
    fstd = torch.tensor([features.std[f"{PBP_PREFIX}{j}"] for j in range(dim)], dtype=torch.float32)

    def batch_for(positions: torch.Tensor, aux: bool = False):
        r = positions
        hist, ohist = data.history[r], data.opp_history[r]
        need = torch.unique(torch.cat([hist.flatten(), ohist.flatten()]))
        need = need[(need >= 0)]
        need = need[gidx[need] >= 0]
        tok = data.tokens.clone()
        if len(need):
            v = (tokens(gidx[need], side[need]) - fmean) / fstd
            tok[need[:, None], cols[None, :]] = v
        season = data.seasons[r]
        batch = {"team": sequence(tok, data.seasons, hist, season),
                 "opp": sequence(tok, data.seasons, ohist, season),
                 "context": data.context[r], "opp_context": data.opp_context[r], "y": data.y[r]}
        return batch, need

    def aux_loss(need: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        from runitback.nfl.model.dataset import collate

        g = np.unique(gidx[need].numpy())
        pick = rng.choice(g, size=min(cfg["aux_games"], len(g)), replace=False)
        b = collate([item_of[game_ids[i]] for i in pick])
        lower = torch.zeros(len(pick), b["valid"].shape[1], tokens.states[0][0].shape[-1])
        for k, i in enumerate(pick):
            h = tokens.states[i][0]
            lower[k, : len(h)] = h.float()
        top = tokens.trunk(lower)
        foundation.encode = lambda feats, cache=None: top  # the trunk's output for these games
        try:
            loss, _ = foundation.loss(b)
        finally:
            del foundation.encode
        return loss

    # train on the latest seasons' rows (their history games reach further back)
    season_of = data.seasons.numpy()
    train_seasons = sorted(set(season_of[rows["train"]].tolist()))[-cfg["recent_seasons"]:]
    train_rows = torch.as_tensor(rows["train"][np.isin(season_of[rows["train"]], train_seasons)])
    val_rows = torch.as_tensor(rows["val"])
    opt = torch.optim.AdamW([
        {"params": tokens.parameters(), "lr": cfg["lr"]},
        {"params": module.parameters(), "lr": cfg["wager_lr"]},
    ], weight_decay=getattr(module, "weight_decay", 1e-4))
    rng = np.random.default_rng(0)

    @torch.no_grad()
    def val_loss() -> float:
        module.eval(); tokens.eval()
        total, n = 0.0, 0
        for i in range(0, len(val_rows), 512):
            r = data.rows[val_rows[i: i + 512]]
            batch, _ = batch_for(r)
            total += float(wager_loss(module, batch)) * len(r)
            n += len(r)
        return total / max(n, 1)

    best = val_loss()
    log.info("fold %d: fine-tune start, val loss %.4f (frozen)", fold, best)
    best_state = (copy.deepcopy(module.state_dict()), copy.deepcopy(tokens.state_dict()))
    history, bad = [{"epoch": -1, "val": best}], 0
    for epoch in range(cfg["epochs"]):
        module.train(); tokens.train()
        perm = train_rows[torch.as_tensor(rng.permutation(len(train_rows)))]
        n_batches = (len(perm) + cfg["batch_size"] - 1) // cfg["batch_size"]
        if cfg["max_batches"]:
            n_batches = min(n_batches, cfg["max_batches"])
        for k in range(n_batches):
            r = data.rows[perm[k * cfg["batch_size"]: (k + 1) * cfg["batch_size"]]]
            batch, need = batch_for(r)
            loss = wager_loss(module, batch)
            if cfg["aux_weight"] > 0 and len(need):
                loss = loss + cfg["aux_weight"] * aux_loss(need, rng)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*tokens.parameters(), *module.parameters()], 1.0)
            opt.step()
            if k % 10 == 0:
                log.info("fold %d ep %d batch %d/%d loss %.4f (%.0fs)", fold, epoch, k, n_batches,
                         float(loss), time.time() - t0)
        v = val_loss()
        history.append({"epoch": epoch, "val": v})
        log.info("fold %d: epoch %d val loss %.4f", fold, epoch, v)
        if v < best:
            best, bad = v, 0
            best_state = (copy.deepcopy(module.state_dict()), copy.deepcopy(tokens.state_dict()))
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    module.load_state_dict(best_state[0])
    tokens.load_state_dict(best_state[1])
    module.eval(); tokens.eval()

    # every game's fine-tuned embedding, read by the returned featurizer
    with torch.no_grad():
        have = torch.nonzero(gidx >= 0).flatten()
        v = tokens(gidx[have], side[have]).numpy()
    table = games.iloc[have.numpy()][[schema.game_id, schema.team]].rename(
        columns={schema.game_id: "game_id", schema.team: "team"}).reset_index(drop=True)
    for j in range(dim):
        table[f"{PBP_PREFIX}{j}"] = v[:, j]
    path = out_dir / f"pbp_{dim}.parquet"
    table.to_parquet(path)
    _PBP_CACHE.pop(str(path), None)
    # the fine-tuned foundation (for the play-by-play generator)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = dict(ck["state_dict"])
    for i, b in enumerate(tokens.top):
        for k, w in b.state_dict().items():
            sd[f"blocks.{tokens.first + i}.{k}"] = w.clone()
    for k, w in tokens.ln_f.state_dict().items():
        sd[f"ln_f.{k}"] = w.clone()
    torch.save({**ck, "state_dict": sd, "finetune": {**{k: v for k, v in cfg.items()}, "history": history}},
               out_dir / "model.pt")
    (out_dir / "history.json").write_text(json.dumps({"settings": cfg, "history": history}, indent=1, default=str))
    log.info("fold %d: fine-tuned in %.0fs, best val %.4f -> %s", fold, time.time() - t0, best, out_dir)

    features = copy.copy(features)
    features.pbp = str(path)
    data = SequenceData(games, schema, features, extra_targets=extra_targets)
    return module, data, features
