"""Sport-agnostic transformer building blocks: causal block with a preallocated KV cache."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCache:
    def __init__(self, n_layers: int, batch: int, n_heads: int, max_len: int, d_head: int, device="cpu"):
        shape = (batch, n_heads, max_len, d_head)
        self.k = [torch.zeros(shape, device=device) for _ in range(n_layers)]
        self.v = [torch.zeros(shape, device=device) for _ in range(n_layers)]
        self.len = 0

    def expand(self, batch: int) -> "KVCache":
        """Copy a batch-1 cache to `batch` independent rows."""
        new = KVCache.__new__(KVCache)
        new.k = [k[:1].expand(batch, -1, -1, -1).clone() for k in self.k]
        new.v = [v[:1].expand(batch, -1, -1, -1).clone() for v in self.v]
        new.len = self.len
        return new


class Block(nn.Module):
    def __init__(self, d: int, h: int, dropout: float):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)
        self.p = dropout

    def forward(self, x, cache: KVCache | None = None, layer: int = 0, pad_mask=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if cache is not None:
            s = cache.len
            cache.k[layer][:, :, s:s + T] = k
            cache.v[layer][:, :, s:s + T] = v
            k, v = cache.k[layer][:, :, : s + T], cache.v[layer][:, :, : s + T]
            if T == 1:
                a = F.scaled_dot_product_attention(q, k, v)
            else:
                mask = torch.ones(T, s + T, dtype=torch.bool, device=x.device).tril(s)
                a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.mlp(self.ln2(x)))


def mlp(d_in: int, d: int, n_out: int) -> nn.Module:
    return nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, n_out))
