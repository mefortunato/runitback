"""Sport-agnostic summaries of simulated games."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class SimResult:
    home: str
    away: str
    n: int
    home_win: float
    away_win: float
    tie: float
    home_score_mean: float
    away_score_mean: float
    margin_mean: float          # home - away
    total_mean: float
    home_score_quantiles: dict
    away_score_quantiles: dict
    margin_quantiles: dict
    total_quantiles: dict
    margin_hist: dict
    scores: list = field(repr=False)  # [(home, away), ...]

    def to_dict(self, include_scores: bool = False) -> dict:
        d = asdict(self)
        if not include_scores:
            d.pop("scores")
        return d


def _q(a: np.ndarray) -> dict:
    return {str(q): float(np.quantile(a, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}


def summarize_scores(home: str, away: str, scores: list[tuple[int, int]]) -> SimResult:
    """Summarize simulated final scores [(home, away), ...] into win/score distributions."""
    h = np.array([s[0] for s in scores])
    a = np.array([s[1] for s in scores])
    m, t = h - a, h + a
    vals, counts = np.unique(np.clip(m, -40, 40), return_counts=True)
    return SimResult(
        home=home, away=away, n=len(scores),
        home_win=float((m > 0).mean()), away_win=float((m < 0).mean()), tie=float((m == 0).mean()),
        home_score_mean=float(h.mean()), away_score_mean=float(a.mean()),
        margin_mean=float(m.mean()), total_mean=float(t.mean()),
        home_score_quantiles=_q(h), away_score_quantiles=_q(a), margin_quantiles=_q(m), total_quantiles=_q(t),
        margin_hist={int(v): int(c) for v, c in zip(vals, counts)},
        scores=list(zip(h.tolist(), a.tolist())),
    )
