"""Sport-agnostic training helpers."""
from __future__ import annotations

import numpy as np


def batches(items: list[dict], batch_size: int, rng: np.random.Generator, shuffle: bool = True,
            length_key: str = "type") -> list[list[int]]:
    """Index batches over per-game items, bucketed by length to reduce padding."""
    order = rng.permutation(len(items)) if shuffle else np.arange(len(items))
    chunks = [order[i: i + batch_size * 20] for i in range(0, len(order), batch_size * 20)]
    out = []
    for c in chunks:
        c = sorted(c, key=lambda i: len(items[i][length_key]))
        out += [c[i: i + batch_size] for i in range(0, len(c), batch_size)]
    if shuffle:
        rng.shuffle(out)
    return out
