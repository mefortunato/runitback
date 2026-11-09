from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats


def win_probability(margin, sigma: float) -> np.ndarray:
    """P(team wins) assuming the realized margin ~ Normal(margin, sigma)."""
    return scipy.stats.norm.cdf(np.asarray(margin, dtype=float) / sigma)


def over_probability(total, line, sigma: float) -> np.ndarray:
    """P(combined points > line) assuming the total ~ Normal(total, sigma)."""
    total, line = np.asarray(total, dtype=float), np.asarray(line, dtype=float)
    return scipy.stats.norm.sf((line - total) / sigma)


def symmetrize(rows: pd.DataFrame, pred, game_id: str) -> np.ndarray:
    """Average each game's two perspectives of a symmetric quantity (e.g.
    the total), so pred(A, B) == pred(B, A)."""
    pred = pd.Series(np.asarray(pred, dtype=float), index=rows.index)
    return pred.groupby(rows[game_id].values).transform("mean").to_numpy()


def fit_sigma(y, margin) -> float:
    """Spread of realized margins around predictions (residual RMSE)."""
    y, margin = np.asarray(y, dtype=float), np.asarray(margin, dtype=float)
    return float(np.sqrt(np.mean((y - margin) ** 2)))


def antisymmetrize(rows: pd.DataFrame, pred, game_id: str) -> np.ndarray:
    """Average each game's two perspectives so pred(A, B) == -pred(B, A).

    ``rows`` holds one row per team per game; games with a single row in
    ``rows`` keep their raw prediction.
    """
    pred = pd.Series(np.asarray(pred, dtype=float), index=rows.index)
    groups = pred.groupby(rows[game_id].values)
    total, count = groups.transform("sum"), groups.transform("count")
    other = total - pred
    return np.where(count == 2, (pred - other) / 2, pred)


def evaluate(y, margin, win_prob=None) -> dict[str, float]:
    """Margin metrics plus winner / probability metrics on decided games."""
    y, margin = np.asarray(y, dtype=float), np.asarray(margin, dtype=float)
    metrics = {
        "n": float(len(y)),
        "rmse": float(np.sqrt(np.mean((y - margin) ** 2))),
        "mae": float(np.mean(np.abs(y - margin))),
        "r2": float(1 - np.sum((y - margin) ** 2) / np.sum((y - y.mean()) ** 2)),
        "pearson": float(scipy.stats.pearsonr(y, margin)[0]),
    }
    decided = y != 0
    won = y[decided] > 0
    metrics["accuracy"] = float(np.mean(won == (margin[decided] > 0)))
    if win_prob is not None:
        p = np.clip(np.asarray(win_prob, dtype=float)[decided], 1e-6, 1 - 1e-6)
        metrics["brier"] = float(np.mean((p - won) ** 2))
        metrics["log_loss"] = float(-np.mean(won * np.log(p) + ~won * np.log(1 - p)))
    return metrics


def format_metrics(metrics: dict[str, dict[str, float]]) -> str:
    """Render ``{split: {metric: value}}`` as a small table."""
    frame = pd.DataFrame(metrics).T
    if "n" in frame:
        frame["n"] = frame["n"].astype(int)
    return frame.to_string(float_format=lambda v: f"{v:.4f}")
