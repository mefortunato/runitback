"""Train the NHL event transformer on the replayed plays table (`runitback nhl preprocess`)."""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..core.training import autocast, batches, cpu_state_dict, resolve_device, to_device
from .data.preprocess import load
from .data.ratings import RatingsTable
from .model.dataset import build_team_seasons, collate, encode_games
from .model.features import CONT_FEATURES, TeamSeasonVocab, feature_sizes
from .model.transformer import EventTransformer, ModelConfig

log = logging.getLogger(__name__)


def split_games(games: pd.DataFrame, val_season: int | None, val_fraction: float = 0.4
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validation = the last `val_fraction` of `val_season` (by date, playoffs included); train =
    every game before it. Nothing from after `val_season` is used."""
    if val_season is None:
        return games, games.iloc[:0]
    dates = games.loc[games.season == val_season, "game_date"].sort_values()
    if dates.empty:
        return games[games.season < val_season], games.iloc[:0]
    cutoff = dates.iloc[int(len(dates) * (1 - val_fraction))]
    is_val = (games.season == val_season) & (games.game_date >= cutoff)
    is_later = (games.season > val_season) | is_val
    return games[~is_later], games[is_val]


@torch.no_grad()
def evaluate(model: EventTransformer, items: list[dict], batch_size: int = 32, amp: bool = False) -> dict[str, float]:
    """Mean next-event negative log-likelihood per head (nats per event) and their sum."""
    model.eval()
    device = next(model.parameters()).device
    sums, n = {}, 0.0
    for b in batches(items, batch_size, np.random.default_rng(0), shuffle=False):
        batch = to_device(collate([items[i] for i in b]), device)
        with autocast(device, amp):
            _, logs = model.loss(batch)
        w = batch["valid"].sum().item()
        for k, v in logs.items():
            sums[k] = sums.get(k, 0.0) + v * w
        n += w
    model.train()
    return {k: v / n for k, v in sums.items()} | {"total": sum(sums.values()) / n}


def train(data_dir: Path, out_dir: Path, *, val_season: int | None = None, val_fraction: float = 0.4,
          epochs: int = 6, batch_size: int = 32, lr: float = 1e-3, d_model: int = 192, n_layers: int = 4,
          n_heads: int = 6, dropout: float = 0.1, min_season: int = 2011, max_games: int | None = None,
          unk_ts_prob: float = 0.1, unk_cont_prob: float = 0.05, use_ratings: bool = True,
          threads: int | None = None, seed: int = 0, device: str | None = "auto", amp: bool = True) -> Path:
    """Train a model on the games before the validation cut; keep the epoch with the lowest
    validation loss (the last one without validation games). Returns the checkpoint."""
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    dev = resolve_device(device)
    amp = amp and dev.type == "cuda" and torch.cuda.is_bf16_supported()
    log.info("training on %s%s", dev, " with bf16 autocast" if amp else "")
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    games, plays, ratings = load(data_dir)
    if not use_ratings:
        ratings = None
    games = games[games.season >= min_season]
    train_g, val_g = split_games(games, val_season, val_fraction)
    if max_games:
        train_g = train_g.sample(min(max_games, len(train_g)), random_state=seed)
        val_g = val_g.sample(min(max(max_games // 4, 1), len(val_g)), random_state=seed) if len(val_g) else val_g
    ts = build_team_seasons(train_g)
    cfg = ModelConfig(feature_sizes=feature_sizes(len(ts)), d_model=d_model, n_layers=n_layers,
                      n_heads=n_heads, dropout=dropout, n_cont=len(CONT_FEATURES) if ratings is not None else 0)
    train_items = encode_games(plays[plays.game_id.isin(train_g.game_id)], train_g, ts, cfg.max_len, ratings)
    val_items = encode_games(plays[plays.game_id.isin(val_g.game_id)], val_g, ts, cfg.max_len, ratings)
    log.info("train games %d (seasons %s-%s), val games %d", len(train_items), train_g.season.min(),
             train_g.season.max(), len(val_items))

    model = EventTransformer(cfg).to(dev)
    log.info("params: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.98))
    steps_per_epoch = math.ceil(len(train_items) / batch_size)
    total_steps, warmup = max(epochs * steps_per_epoch, 1), min(300, steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))

    # pre-game ratings of the training games only, so a checkpoint carries nothing from later
    kept_ratings = ratings[ratings.game_id.isin(train_g.game_id)] if ratings is not None else None
    meta = dict(sport="nhl", config=cfg.to_dict(), team_seasons=ts.keys, val_season=val_season,
                val_fraction=val_fraction, min_season=min_season,
                ratings=RatingsTable(kept_ratings).to_records() if kept_ratings is not None else None)
    best, ckpt = float("inf"), out_dir / "model.pt"
    history = []
    step, t0 = 0, time.time()
    for epoch in range(epochs):
        for b in batches(train_items, batch_size, rng):
            batch = to_device(collate([train_items[i] for i in b], unk_ts_prob, rng, unk_cont_prob), dev)
            with autocast(dev, amp):
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
        val = evaluate(model, val_items, amp=amp) if val_items else {}
        history.append({"epoch": epoch, **{f"val_{k}": v for k, v in val.items()}})
        log.info("epoch %d val %s", epoch, json.dumps({k: round(v, 4) for k, v in val.items()}))
        score = val.get("total", -epoch)
        if score < best:
            best = score
            torch.save({"state_dict": cpu_state_dict(model), **meta, "history": history}, ckpt)
        else:  # keep the history complete in the saved checkpoint
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            ck["history"] = history
            torch.save(ck, ckpt)
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    return ckpt
