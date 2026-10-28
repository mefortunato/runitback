"""NFL play transformer: shared causal trunk (core) with factorized next-play outcome heads."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.transformer import Block, KVCache, mlp
from ..vocab import N_ELAPSED, N_YARDS, PLAY_TYPES, RESULTS
from .features import FEATURES

NEG = -1e9


@dataclass
class ModelConfig:
    feature_sizes: dict
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    max_len: int = 400
    dropout: float = 0.1
    n_cont: int = 0  # continuous team-strength inputs (0 = model without ratings)

    def to_dict(self) -> dict:
        return asdict(self)


class PlayTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.emb = nn.ModuleDict({f: nn.Embedding(cfg.feature_sizes[f], d) for f in FEATURES})
        self.pos = nn.Embedding(cfg.max_len, d)
        self.cont = nn.Linear(cfg.n_cont, d) if cfg.n_cont else None
        self.in_ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(d)
        nT, nR = len(PLAY_TYPES), len(RESULTS)
        self.e_type, self.e_res, self.e_yards = nn.Embedding(nT, d), nn.Embedding(nR, d), nn.Embedding(N_YARDS, d)
        self.type_head = mlp(d, d, nT)
        self.res_head = mlp(d, d, nR)
        self.yards_head = mlp(d, d, N_YARDS)
        self.el_head = mlp(d, d, N_ELAPSED)

    # -- trunk
    def encode(self, feats: dict[str, torch.Tensor], cache: KVCache | None = None) -> torch.Tensor:
        x = sum(self.emb[f](feats[f]) for f in FEATURES)
        if self.cont is not None:
            x = x + self.cont(feats["cont"])
        T = x.shape[1]
        start = cache.len if cache is not None else 0
        x = x + self.pos(torch.arange(start, start + T, device=x.device).clamp(max=self.cfg.max_len - 1))
        x = self.drop(self.in_ln(x))
        for i, b in enumerate(self.blocks):
            x = b(x, cache, i)
        if cache is not None:
            cache.len += T
        return self.ln_f(x)

    def new_cache(self, batch: int) -> KVCache:
        c = self.cfg
        return KVCache(c.n_layers, batch, c.n_heads, c.max_len, c.d_model // c.n_heads,
                       device=next(self.parameters()).device)

    # -- heads (each conditioned on the fields chosen before it)
    def type_logits(self, h):
        return self.type_head(h)

    def result_logits(self, h, t):
        return self.res_head(h + self.e_type(t))

    def yards_logits(self, h, t, r):
        return self.yards_head(h + self.e_type(t) + self.e_res(r))

    def elapsed_logits(self, h, t, r, y):
        return self.el_head(h + self.e_type(t) + self.e_res(r) + self.e_yards(y))

    def loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        h = self.encode(batch["feats"])
        valid = batch["valid"]
        t, r, y, e = batch["type"], batch["result"], batch["yards"], batch["elapsed"]
        parts = {
            "type": (self.type_logits(h), t, batch["type_mask"]),
            "result": (self.result_logits(h, t), r, batch["result_mask"]),
            "yards": (self.yards_logits(h, t, r), y, batch["yards_mask"]),
            "elapsed": (self.elapsed_logits(h, t, r, y), e, batch["elapsed_mask"]),
        }
        total, logs = 0.0, {}
        n = valid.sum().clamp(min=1)
        for name, (logits, tgt, mask) in parts.items():
            logits = logits.masked_fill(~mask, NEG)
            nll = F.cross_entropy(logits.flatten(0, 1), tgt.flatten(), reduction="none").view_as(tgt)
            l = (nll * valid).sum() / n
            total = total + l
            logs[name] = l.item()
        return total, logs
