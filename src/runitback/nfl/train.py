"""Train the play transformer."""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data.ratings import RatingsTable
from .model.dataset import build_team_seasons, collate, encode_games
from .model.features import CONT_FEATURES
from .model.features import feature_sizes
from .model.transformer import ModelConfig, PlayTransformer

log = logging.getLogger(__name__)


def split_games(games: pd.DataFrame, val_season: int | None, val_week: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validation = late-season games of `val_season` (and anything after); train = everything before."""
    if val_season is None:
        return games, games.iloc[:0]
    is_val = (games.season > val_season) | ((games.season == val_season) & (games.week >= val_week))
    return games[~is_val], games[(games.season == val_season) & (games.week >= val_week)]


def batches(items: list[dict], batch_size: int, rng: np.random.Generator, shuffle: bool = True):
    order = rng.permutation(len(items)) if shuffle else np.arange(len(items))
    # bucket by length to reduce padding
    chunks = [order[i: i + batch_size * 20] for i in range(0, len(order), batch_size * 20)]
    out = []
    for c in chunks:
        c = sorted(c, key=lambda i: len(items[i]["type"]))
        out += [c[i: i + batch_size] for i in range(0, len(c), batch_size)]
    if shuffle:
        rng.shuffle(out)
    return out


@torch.no_grad()
def evaluate(model: PlayTransformer, items: list[dict], batch_size: int = 32) -> dict[str, float]:
    model.eval()
    sums, n = {}, 0.0
    for b in batches(items, batch_size, np.random.default_rng(0), shuffle=False):
        batch = collate([items[i] for i in b])
        _, logs = model.loss(batch)
        w = batch["valid"].sum().item()
        for k, v in logs.items():
            sums[k] = sums.get(k, 0.0) + v * w
        n += w
    model.train()
    return {k: v / n for k, v in sums.items()} | {"total": sum(sums.values()) / n}


def train(data_dir: Path, out_dir: Path, *, val_season: int | None = 2025, val_week: int = 10,
          epochs: int = 8, batch_size: int = 32, lr: float = 1e-3, d_model: int = 256, n_layers: int = 6,
          n_heads: int = 8, dropout: float = 0.1, min_season: int = 1999, max_games: int | None = None,
          unk_ts_prob: float = 0.1, unk_cont_prob: float = 0.05, use_ratings: bool = True, threads: int | None = None, seed: int = 0) -> Path:
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    games = pd.read_parquet(data_dir / "games.parquet")
    plays = pd.read_parquet(data_dir / "plays.parquet")
    games = games[games.season >= min_season]
    train_g, val_g = split_games(games, val_season, val_week)
    if max_games:
        train_g = train_g.sample(min(max_games, len(train_g)), random_state=seed)
    ts = build_team_seasons(train_g)
    rpath = data_dir / "ratings.parquet"
    ratings = pd.read_parquet(rpath) if use_ratings and rpath.exists() else None
    cfg = ModelConfig(feature_sizes=feature_sizes(len(ts)), d_model=d_model, n_layers=n_layers,
                      n_heads=n_heads, dropout=dropout, n_cont=len(CONT_FEATURES) if ratings is not None else 0)
    train_items = encode_games(plays[plays.game_id.isin(train_g.game_id)], train_g, ts, cfg.max_len, ratings)
    val_items = encode_games(plays[plays.game_id.isin(val_g.game_id)], val_g, ts, cfg.max_len, ratings) \
        if len(val_g) else []
    log.info("train games %d, val games %d", len(train_items), len(val_items))

    model = PlayTransformer(cfg)
    log.info("params: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.98))
    steps_per_epoch = math.ceil(len(train_items) / batch_size)
    total_steps, warmup = epochs * steps_per_epoch, min(300, steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))

    meta = dict(config=cfg.to_dict(), team_seasons=ts.keys, val_season=val_season, val_week=val_week,
                ratings=RatingsTable(ratings).to_records() if ratings is not None else None)
    best, ckpt = float("inf"), out_dir / "model.pt"
    history = []
    step, t0 = 0, time.time()
    for epoch in range(epochs):
        for b in batches(train_items, batch_size, rng):
            batch = collate([train_items[i] for i in b], unk_ts_prob, rng, unk_cont_prob)
            loss, logs = model.loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                log.info("ep %d step %d/%d loss %.4f %s (%.0fs)", epoch, step, total_steps, loss.item(),
                         " ".join(f"{k}={v:.3f}" for k, v in logs.items()), time.time() - t0)
        val = evaluate(model, val_items) if val_items else {}
        history.append({"epoch": epoch, **{f"val_{k}": v for k, v in val.items()}})
        log.info("epoch %d val %s", epoch, json.dumps({k: round(v, 4) for k, v in val.items()}))
        score = val.get("total", -epoch)
        if score < best or not val_items:
            best = score
            torch.save({"state_dict": model.state_dict(), **meta, "history": history}, ckpt)
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    return ckpt
