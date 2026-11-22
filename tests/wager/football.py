"""A toy football league (Poisson goals, a quarter of games drawn) for the draws-mode tests."""

import dataclasses

import numpy as np
import pandas as pd
import scipy.stats

from .conftest import TOY_SCHEMA, make_games

HOME_RATE, AWAY_RATE = 1.45, 1.1  # expected goals, every game

DRAWS_SCHEMA = dataclasses.replace(
    TOY_SCHEMA,
    draws=True,
    market_draw="draw_moneyline",
    market_average=("avg_moneyline", "avg_draw_moneyline", "opp_avg_moneyline"),
)


def american(implied):
    """American odds of an implied probability (margin included)."""
    implied = np.asarray(implied, dtype=float)
    return np.where(
        implied >= 0.5, -100 * implied / (1 - implied), 100 * (1 - implied) / implied
    )


def true_probabilities() -> tuple[float, float, float]:
    """P(home win), P(draw), P(away win) of a game, from the goals' Skellam distribution."""
    win = scipy.stats.skellam.sf(0, HOME_RATE, AWAY_RATE)
    draw = scipy.stats.skellam.pmf(0, HOME_RATE, AWAY_RATE)
    return float(win), float(draw), float(1 - win - draw)


def football_games(seed: int = 3, weeks: int = 40) -> pd.DataFrame:
    """The toy league with Poisson goals and three-way market prices: the
    preferred bookmaker's (``moneyline``, ``draw_moneyline``, ``opp_moneyline``,
    5% margin) and the closing average's (``avg_*``, 3% margin), both quoting the
    true probabilities."""
    games = make_games(seed=seed, persistence=0.7, weeks=weeks)
    rng = np.random.default_rng(seed)
    game = games.groupby("game_id").ngroup().to_numpy()
    first = games.groupby("game_id").cumcount().to_numpy() == 0
    # one draw of goals per game, from the first row's side, mirrored onto the second
    home = rng.poisson(HOME_RATE, game.max() + 1)[game]
    away = rng.poisson(AWAY_RATE, game.max() + 1)[game]
    pts, opts = np.where(first, home, away), np.where(first, away, home)
    unplayed = games["margin"].isna()
    games["points"], games["opp_points"] = pts, opts
    games["margin"], games["total"] = pts - opts, pts + opts
    games.loc[unplayed, ["margin", "total", *TOY_SCHEMA.stats]] = np.nan
    win, draw, loss = true_probabilities()
    own, other = np.where(first, win, loss), np.where(first, loss, win)
    for prefix, margin in (("", 1.05), ("avg_", 1.03)):
        games[f"{prefix}moneyline"] = american(own * margin)
        games[f"opp_{prefix}moneyline"] = american(other * margin)
        games[f"{prefix}draw_moneyline"] = american(draw * margin)
    games["spread"] = np.where(first, 1.0, -1.0) * (HOME_RATE - AWAY_RATE)
    return games
