"""Train the play transformer."""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from ..core.training import autocast, batches, cpu_state_dict, resolve_device, to_device
from .data.ratings import RatingsTable
from .model.dataset import build_team_seasons, collate, encode_games
from .model.features import CONT_FEATURES
from .model.features import TeamSeasonVocab, feature_sizes
from .model.lineups import (
    CoachVocab, LineupData, PersonnelSampler, PlayerVocab, inference_meta, signature, situation_key,
)
from .model.lineups import N_USAGE
from .vocab import PLAY_TYPES
from .model.transformer import ModelConfig, PlayTransformer

log = logging.getLogger(__name__)


def split_games(games: pd.DataFrame, val_season: int | None, val_week: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validation = late-season games of `val_season` (and anything after); train = everything before."""
    if val_season is None:
        return games, games.iloc[:0]
    is_val = (games.season > val_season) | ((games.season == val_season) & (games.week >= val_week))
    return games[~is_val], games[(games.season == val_season) & (games.week >= val_week)]


@torch.no_grad()
def evaluate(model: PlayTransformer, items: list[dict], batch_size: int = 32, amp: bool = False) -> dict[str, float]:
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
    play = ("type", "result", "yards", "elapsed")  # "total" stays comparable with models without a who head
    return {k: v / n for k, v in sums.items()} | {"total": sum(v for k, v in sums.items() if k in play) / n}


def fit_personnel_sampler(items: list[dict]) -> PersonnelSampler:
    """Count which depth-chart slots were on the field, per side and situation."""
    from collections import Counter, defaultdict
    samples: dict[str, Counter] = defaultdict(Counter)
    for it in items:
        if "pers_slot" not in it:
            continue
        has = np.flatnonzero((it["pers_slot"] > -2).any(1) & (it["pers"] > 0).all(1))
        for i in has:
            pt = PLAY_TYPES[it["type"][i]]
            key = situation_key(pt, int(it["f_down"][i]), int(it["f_ytg"][i]), int(it["f_yl"][i]))
            for role, sl in (("off", slice(0, 11)), ("def", slice(11, 22))):
                sig = signature(it["pers_slot"][i, sl].tolist(), it["pers_grp"][i, sl].tolist())
                samples[f"{role}|{key}"][sig] += 1
                samples[f"{role}|{pt}"][sig] += 1
    return PersonnelSampler.fit(samples)


def box_sigma(items: list[dict], batch_size: int = 64) -> torch.Tensor:
    """Spread of each real box-score stat across training games (scales the box-score loss)."""
    from .model.boxscore import game_box, true_play_stats
    from .vocab import YARDS_MIN
    rows = []
    for i in range(0, len(items), batch_size):
        b = collate(items[i: i + batch_size])
        o, d = true_play_stats(b["type"], b["result"], b["yards"] + YARDS_MIN, b["feats"]["yl"])
        rows.append(game_box(o, d, b["feats"]["is_home"], b["valid"]))
    return torch.cat(rows).float().std(0).clamp(min=1e-3)


@torch.no_grad()
def evaluate_box(model: PlayTransformer, items: list[dict], sigma: torch.Tensor, batch_size: int = 32) -> dict:
    """Teacher-forced box-score error on held-out games: RMSE of each stat (in its own units)."""
    from .model.boxscore import GAME_STATS, batch_box
    model.eval()
    sq, n = 0.0, 0
    device = next(model.parameters()).device
    for b in batches(items, batch_size, np.random.default_rng(0), shuffle=False):
        batch = to_device(collate([items[i] for i in b]), device)
        h = model.encode(batch["feats"])
        pred, real = batch_box(model, h, model.personnel(batch["feats"]), batch)
        sq, n = sq + ((pred - real) ** 2).sum(0).float().cpu(), n + len(b)
    model.train()
    return {k: float(v) for k, v in zip(GAME_STATS, (sq / n) ** 0.5)}


def _lineup_dir(dirs: Sequence[Path]) -> Path | None:
    return next((Path(d) for d in dirs if (Path(d) / "player_games.parquet").exists()), None)


def _is_synthetic(data_dir: Path) -> bool:
    meta = data_dir / "generation.json"
    return meta.is_file() and bool(json.loads(meta.read_text()).get("synthetic"))


def load_data(data_dirs: Path | Sequence[Path], use_ratings: bool = True
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Concatenate games/plays/ratings from one or more directories (e.g. processed + synthetic).

    Returns (games, plays, ratings, lookup_ratings). `ratings` covers every game and feeds the
    model's rating features; `lookup_ratings` is what the checkpoint stores for future pre-game
    lookups, and leaves out synthetic seasons so simulated results never stand in for real ones.
    """
    dirs = [Path(data_dirs)] if isinstance(data_dirs, (str, Path)) else [Path(d) for d in data_dirs]
    if not dirs:
        raise ValueError("no training data directories given")
    games, plays, ratings, real_ratings = [], [], [], []
    for d in dirs:
        games.append(pd.read_parquet(d / "games.parquet"))
        plays.append(pd.read_parquet(d / "plays.parquet"))
        rpath = d / "ratings.parquet"
        if use_ratings and rpath.exists():
            r = pd.read_parquet(rpath)
            ratings.append(r)
            if not _is_synthetic(d):
                real_ratings.append(r)
    all_games = pd.concat(games, ignore_index=True)
    duplicated = all_games.game_id[all_games.game_id.duplicated()]
    if len(duplicated):
        raise ValueError(f"game_id {duplicated.iloc[0]!r} appears in more than one training directory")
    all_ratings = pd.concat(ratings, ignore_index=True) if ratings else None
    lookup = pd.concat(real_ratings, ignore_index=True) if real_ratings else None
    return all_games, pd.concat(plays, ignore_index=True), all_ratings, lookup


def train(data_dir: Path | Sequence[Path], out_dir: Path, *, val_season: int | None = 2025, val_week: int = 10,
          epochs: int = 8, batch_size: int = 32, lr: float = 1e-3, d_model: int = 256, n_layers: int = 6,
          n_heads: int = 8, dropout: float = 0.1, min_season: int = 1999, max_games: int | None = None,
          unk_ts_prob: float = 0.1, unk_cont_prob: float = 0.05, use_ratings: bool = True, threads: int | None = None,
          seed: int = 0, init: Path | None = None, device: str | None = "auto", amp: bool = True,
          use_lineups: bool = True, unk_lineup_prob: float = 0.1, unk_pers_prob: float = 0.15,
          other_player_prob: float = 0.03, player_dim: int = 0, lineup_dropout: float = 0.0,
          prev_lineup_prob: float = 0.0, box_weight: float = 0.0, who_head: bool = False,
          qb_weight: float = 0.0) -> Path:
    """Train a model on one or more data directories. With `init`, fine-tune that checkpoint: its
    architecture and team-season vocabulary are kept (team-seasons it has not seen use the unknown
    embedding) and d_model/n_layers/n_heads/dropout are ignored."""
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    dev = resolve_device(device)
    amp = amp and dev.type == "cuda" and torch.cuda.is_bf16_supported()
    log.info("training on %s%s", dev, " with bf16 autocast" if amp else "")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    games, plays, ratings, lookup_ratings = load_data(data_dir, use_ratings)
    games = games[games.season >= min_season]
    train_g, val_g = split_games(games, val_season, val_week)
    if max_games:
        train_g = train_g.sample(min(max_games, len(train_g)), random_state=seed)
    base = torch.load(init, map_location="cpu", weights_only=False) if init else None
    dirs = [Path(data_dir)] if isinstance(data_dir, (str, Path)) else [Path(d) for d in data_dir]
    ld = None
    if use_lineups and (base is None or base.get("lineup_meta")):
        ldir = _lineup_dir(dirs)
        if ldir is None:
            log.warning("no player data found (runitback nfl preprocess builds it); training without lineups")
        elif base is not None:
            lm = base["lineup_meta"]
            ld = LineupData.load(ldir, [], PlayerVocab(lm["player_vocab"]), CoachVocab(lm["coach_vocab"]))
        else:
            ld = LineupData.load(ldir, train_g.game_id)
        if ld is not None:
            log.info("lineups: %d players with embeddings, %d coaches", len(ld.pvocab.ids), len(ld.cvocab.names))
    if base is not None:
        cfg = ModelConfig(**base["config"])
        ts = TeamSeasonVocab([])
        ts.keys = list(base["team_seasons"])
        ts.idx = {k: i + 1 for i, k in enumerate(ts.keys)}
        if lookup_ratings is None and base.get("ratings"):
            lookup_ratings = pd.DataFrame(base["ratings"])
        log.info("fine-tuning %s (d_model=%d, n_layers=%d)", init, cfg.d_model, cfg.n_layers)
    else:
        ts = build_team_seasons(train_g)
        cfg = ModelConfig(feature_sizes=feature_sizes(len(ts)), d_model=d_model, n_layers=n_layers,
                          n_heads=n_heads, dropout=dropout, n_cont=len(CONT_FEATURES) if ratings is not None else 0,
                          n_players=len(ld.pvocab) if ld else 0, n_coaches=len(ld.cvocab) if ld else 0,
                          player_dim=player_dim, lineup_dropout=lineup_dropout, who_head=who_head and ld is not None,
                          n_usage=N_USAGE if (who_head and ld is not None and ld.usage) else 0,
                          qb_weight=qb_weight if ld is not None else 0.0)
    if lookup_ratings is None:
        lookup_ratings = ratings  # only synthetic data: better than no pre-game ratings at all
    train_items = encode_games(plays[plays.game_id.isin(train_g.game_id)], train_g, ts, cfg.max_len, ratings, ld)
    val_items = encode_games(plays[plays.game_id.isin(val_g.game_id)], val_g, ts, cfg.max_len, ratings, ld) \
        if len(val_g) else []
    log.info("train games %d, val games %d", len(train_items), len(val_items))

    model = PlayTransformer(cfg)
    if base is not None:
        model.load_state_dict(base["state_dict"])
    model.to(dev)
    log.info("params: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.98))
    steps_per_epoch = math.ceil(len(train_items) / batch_size)
    total_steps, warmup = epochs * steps_per_epoch, min(300, steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))

    sigma = box_sigma(train_items)
    sigma_dev = sigma.to(dev)
    log.info("box-score scales: %s", [round(float(x), 1) for x in sigma])
    meta = dict(config=cfg.to_dict(), team_seasons=ts.keys, val_season=val_season, val_week=val_week,
                ratings=RatingsTable(lookup_ratings).to_records() if lookup_ratings is not None else None,
                init_checkpoint=str(init) if init else None, box_weight=box_weight, box_sigma=sigma.tolist())
    if ld is not None:
        lm = inference_meta(ld, pd.read_parquet(_lineup_dir(dirs) / "coach_games.parquet"))
        lm["personnel_sampler"] = fit_personnel_sampler(train_items).to_dict()
        from .model.lineups import fit_role_inclusion
        lm["role_inclusion"] = fit_role_inclusion(ld, plays[plays.game_id.isin(train_g.game_id)])
        meta["lineup_meta"] = lm
        log.info("personnel sampler: %d situations", len(lm["personnel_sampler"]))
    best, ckpt = float("inf"), out_dir / "model.pt"
    history = []
    step, t0 = 0, time.time()
    for epoch in range(epochs):
        for b in batches(train_items, batch_size, rng):
            batch = to_device(collate([train_items[i] for i in b], unk_ts_prob, rng, unk_cont_prob,
                                      unk_lineup_prob, unk_pers_prob, other_player_prob, prev_lineup_prob), dev)
            with autocast(dev, amp):
                loss, logs = model.loss(batch, box_weight, sigma_dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                log.info("ep %d step %d/%d loss %.4f %s (%.0fs)", epoch, step, total_steps, loss.item(),
                         " ".join(f"{k}={v:.3f}" for k, v in logs.items() if not k.startswith("box_")),
                         time.time() - t0)
        val = evaluate(model, val_items, amp=amp) if val_items else {}
        vbox = evaluate_box(model, val_items, sigma) if val_items and box_weight > 0 else {}
        history.append({"epoch": epoch, **{f"val_{k}": v for k, v in val.items()},
                        **{f"val_box_rmse_{k}": v for k, v in vbox.items()}})
        log.info("epoch %d val %s", epoch, json.dumps({k: round(v, 4) for k, v in val.items()}))
        if vbox:
            log.info("epoch %d val box-score RMSE %s", epoch, json.dumps({k: round(v, 2) for k, v in vbox.items()}))
        score = val.get("total", -epoch)
        if score < best or not val_items:
            best = score
            torch.save({"state_dict": cpu_state_dict(model), **meta, "history": history}, ckpt)
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    return ckpt
