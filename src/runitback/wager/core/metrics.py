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


def outcome_class(y) -> np.ndarray:
    """Each game's outcome from the team's side: 0 a loss, 1 a draw, 2 a win
    (the order the ranked probability score uses)."""
    return (np.sign(np.asarray(y, dtype=float)) + 1).astype(int)


def three_way_scores(y, win_prob, draw_prob) -> dict[str, np.ndarray]:
    """Per-game scores of win / draw / loss probabilities (``y`` is the
    margin; the loss probability is what is left).

    - ``rps``: the ranked probability score, the football standard, with the
      outcomes ordered loss < draw < win:
      ``1/2 [(F_L - O_L)^2 + (F_L + F_D - O_L - O_D)^2]`` (F the forecast,
      O the outcome). It rewards a near miss: a draw forecast for a win
      scores better than a loss forecast.
    - ``brier3``: the multiclass Brier score ``sum_i (p_i - o_i)^2``.
    - ``log_loss3``: the negative log probability of what happened.
    - ``correct3``: whether the likeliest outcome happened (1.0 / 0.0).
    """
    p_win = np.asarray(win_prob, dtype=float)
    p_draw = np.asarray(draw_prob, dtype=float)
    p = np.clip(np.stack([1 - p_win - p_draw, p_draw, p_win], axis=-1), 0.0, 1.0)
    actual = outcome_class(y)
    o = np.eye(3)[actual]
    cum_p, cum_o = np.cumsum(p, axis=-1), np.cumsum(o, axis=-1)
    return {
        "rps": 0.5 * ((cum_p - cum_o)[..., :2] ** 2).sum(axis=-1),
        "brier3": ((p - o) ** 2).sum(axis=-1),
        "log_loss3": -np.log(np.clip(p[np.arange(len(actual)), actual], 1e-6, 1.0)),
        "correct3": (p.argmax(axis=-1) == actual).astype(float),
    }


def evaluate(y, margin, win_prob=None, draw_prob=None) -> dict[str, float]:
    """Margin metrics plus winner / probability metrics on decided games.

    With ``draw_prob`` (a three-outcome sport; ``win_prob`` is then the
    unconditional probability of a win) every game is scored, draws included:
    the binary ``accuracy`` / ``brier`` / ``log_loss`` give way to
    ``accuracy3``, ``rps``, ``brier3`` and ``log_loss3``
    (:func:`three_way_scores`).
    """
    y, margin = np.asarray(y, dtype=float), np.asarray(margin, dtype=float)
    metrics = {
        "n": float(len(y)),
        "rmse": float(np.sqrt(np.mean((y - margin) ** 2))),
        "mae": float(np.mean(np.abs(y - margin))),
        "r2": float(1 - np.sum((y - margin) ** 2) / np.sum((y - y.mean()) ** 2)),
        "pearson": float(scipy.stats.pearsonr(y, margin)[0]),
    }
    if draw_prob is not None:
        scores = three_way_scores(y, win_prob, draw_prob)
        metrics["accuracy3"] = float(scores["correct3"].mean())
        for name in ("rps", "brier3", "log_loss3"):
            metrics[name] = float(scores[name].mean())
        return metrics
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
