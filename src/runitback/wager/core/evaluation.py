from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .metrics import (
    antisymmetrize,
    evaluate,
    fit_sigma,
    format_metrics,
    symmetrize,
    win_probability,
)
from .predictor import MatchupModel, Predictor, save_predictor
from .schema import GameSchema
from .splits import SeasonSplit

logger = logging.getLogger(__name__)


def calibrate_and_evaluate(
    rows: pd.DataFrame, raw_pred, schema: GameSchema, split: SeasonSplit
) -> tuple[float, dict[str, dict[str, float]], np.ndarray]:
    """Shared post-fit step for every model family.

    Antisymmetrizes per-row predictions of played games, fits the win
    probability spread ``sigma`` on the validation seasons (falling back to
    test, then train) and scores every split.

    Args:
        rows: Played rows (both perspectives of each game) that were predicted.
        raw_pred: Raw margin predictions aligned with ``rows``.

    Returns:
        ``(sigma, metrics_by_split, antisymmetric_predictions)``
    """
    pred = antisymmetrize(rows, raw_pred, schema.game_id)
    y = rows[schema.target].to_numpy(dtype=float)
    seasons = rows[schema.season].to_numpy()
    masks = {
        name: np.isin(seasons, getattr(split, name)) & ~np.isnan(pred)
        for name in ("train", "val", "test")
    }

    source = next((n for n in ("val", "test", "train") if masks[n].any()), None)
    if source is None:
        raise ValueError("no predictions to calibrate on")
    if source == "train":
        logger.warning("no val/test seasons: win probabilities fit in-sample")
    sigma = fit_sigma(y[masks[source]], pred[masks[source]])

    metrics = {
        name: evaluate(y[mask], pred[mask], win_probability(pred[mask], sigma))
        for name, mask in masks.items()
        if mask.any()
    }
    return sigma, metrics, pred


def calibrate_total(
    rows: pd.DataFrame, raw_total, schema: GameSchema, split: SeasonSplit, metrics
) -> float:
    """Residual spread of the (symmetrized) predicted total on the
    validation seasons (else test, else train); adds each split's
    ``total_rmse`` / ``total_mae`` to ``metrics`` in place."""
    pred = symmetrize(rows, raw_total, schema.game_id)
    y = rows[schema.total].to_numpy(dtype=float)
    seasons = rows[schema.season].to_numpy()
    ok = ~np.isnan(pred) & ~np.isnan(y)
    masks = {
        name: np.isin(seasons, getattr(split, name)) & ok
        for name in ("train", "val", "test")
    }
    for name, mask in masks.items():
        if mask.any() and name in metrics:
            err = y[mask] - pred[mask]
            metrics[name]["total_rmse"] = float(np.sqrt(np.mean(err**2)))
            metrics[name]["total_mae"] = float(np.mean(np.abs(err)))
    source = next(n for n in ("val", "test", "train") if masks[n].any())
    return fit_sigma(y[masks[source]], pred[masks[source]])


def finish_fit(
    model: MatchupModel,
    games: pd.DataFrame,
    rows: pd.DataFrame,
    raw_pred,
    split: SeasonSplit,
    sport: str,
    save_dir: str | None = None,
    print_metrics: bool = True,
    raw_total=None,
) -> dict:
    """Common tail of every ``fit_<family>``.

    Calibrates and scores ``raw_pred`` (pre-game predictions for ``rows``)
    and, for models that predict it, ``raw_total``; refreshes ``model``'s
    state from ``games``, wraps it in a :class:`Predictor` and saves it.

    Returns:
        ``{"predictor", "metrics", "split"}``
    """
    sigma, metrics, _ = calibrate_and_evaluate(rows, raw_pred, model.schema, split)
    total_sigma = None
    if raw_total is not None:
        total_sigma = calibrate_total(rows, raw_total, model.schema, split, metrics)
    if print_metrics:
        print(format_metrics(metrics))
    model.update_state(games)
    predictor = Predictor(
        model, sigma=sigma, sport=sport, metrics=metrics, total_sigma=total_sigma
    )
    if save_dir is not None:
        save_predictor(predictor, save_dir)
    return {"predictor": predictor, "metrics": metrics, "split": split}
