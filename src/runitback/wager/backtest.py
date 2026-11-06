"""Walk-forward backtests of any model family, scored against the market.

For each test season a model is fit on earlier seasons only (the latest of
them validates / calibrates). The season is then replayed date by date: team
state is refreshed from the games played so far and that date's games are
predicted, the same retrain-occasionally, refresh-weekly loop as production.

Where the sport's schema names market columns, the market's spread and
moneylines are scored on the same games, and the model's disagreements with
the market are scored as bets against it: against the spread, on the
moneyline (at the posted odds) and, for models that predict the total, on
the over/under. Sports without a point spread (hockey's puck line and
baseball's run line are fixed handicaps priced like moneylines) are scored
against the moneylines alone, and the market's margin is the one its win
probability implies (:func:`implied_sigma`).
"""

from __future__ import annotations

import json
import pathlib
import time
import typing
from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
import scipy.stats
import sklearn.base

from .core.io import Table, read_table
from .core.metrics import evaluate, fit_sigma, win_probability
from .core.predictor import Predictor
from .core.schema import GameSchema, one_row_per_game, played, upcoming
from .core.splits import SeasonSplit, split_seasons
from .core.sport import get_sport

# Fits a model on the games it is given (all before the test season).
Fitter = Callable[[pd.DataFrame, SeasonSplit], Predictor]
# called with the predictor's state just before each test date's games
OnDate = Callable[[Predictor, int, typing.Any], None]

# Win rate needed to break even on bets at -110.
BREAKEVEN = 110 / 210


def sklearn_fitter(
    sport: str, model: sklearn.base.BaseEstimator | None = None, **kwargs
) -> Fitter:
    from .models.sklearn import fit_sklearn

    def fit(games: pd.DataFrame, split: SeasonSplit) -> Predictor:
        return fit_sklearn(
            games,
            sport,
            model=model,
            train_seasons=split.train,
            val_seasons=split.val,
            test_seasons=split.test,
            print_metrics=False,
            **kwargs,
        )["predictor"]

    return fit


def gnn_fitter(
    sport: str, max_epochs: int = 100, model: dict | None = None, **kwargs
) -> Fitter:
    """``model``: :func:`~runitback.wager.models.gnn.default_model` arguments, e.g.
    ``{"total_weight": 1.0}``; ``kwargs`` go to ``fit_gnn``."""
    from .models.gnn import default_model, default_trainer, fit_gnn

    def fit(games: pd.DataFrame, split: SeasonSplit) -> Predictor:
        trainer = default_trainer(
            max_epochs=max_epochs,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        return fit_gnn(
            games,
            sport,
            model=default_model(sport, kwargs.get("context"), **(model or {})),
            trainer=trainer,
            train_seasons=split.train,
            val_seasons=split.val,
            test_seasons=split.test,
            print_metrics=False,
            **kwargs,
        )["predictor"]

    return fit


def transformer_fitter(
    sport: str,
    max_epochs: int = 100,
    model: dict | None = None,
    patience: int = 10,
    **kwargs,
) -> Fitter:
    """``model``: encoder/config overrides for :func:`default_model`, e.g.
    ``{"history": 24, "season_decay": True, "market_tokens": True,
    "antisymmetric": True}``; ``kwargs`` go to :func:`fit_transformer`
    (e.g. ``ensemble=5``, ``batch_size=128``)."""
    from .models.lightning import default_trainer
    from .models.transformer import default_model, fit_transformer

    def trainer():
        return default_trainer(
            max_epochs=max_epochs,
            patience=patience,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

    def fit(games: pd.DataFrame, split: SeasonSplit) -> Predictor:
        return fit_transformer(
            games,
            sport,
            model=default_model(
                sport,
                kwargs.get("context"),
                kwargs.get("opponent_strength"),
                kwargs.get("elo_margin"),
                **(model or {}),
            ),
            trainer=trainer,
            train_seasons=split.train,
            val_seasons=split.val,
            test_seasons=split.test,
            print_metrics=False,
            **kwargs,
        )["predictor"]

    return fit


def baseline_fitter(sport: str, name: str) -> Fitter:
    from .models import baselines

    fit_fn = {
        "home": baselines.fit_home,
        "point_diff": baselines.fit_point_diff,
        "elo": baselines.fit_elo,
    }[name]

    def fit(games: pd.DataFrame, split: SeasonSplit) -> Predictor:
        return fit_fn(
            games, sport, print_metrics=False, **baselines.split_kwargs(split)
        )["predictor"]

    return fit


Family = Literal["home", "point_diff", "elo", "sklearn", "gnn", "transformer"]
FAMILIES: tuple[str, ...] = typing.get_args(Family)


def family_fitter(
    family: str,
    sport: str,
    model: sklearn.base.BaseEstimator | None = None,
    max_epochs: int = 100,
) -> Fitter:
    """The :data:`Fitter` of a built-in model family.

    ``model`` is the sklearn regressor for ``sklearn``; ``max_epochs`` caps
    training of the neural families, which get a fresh default model and
    trainer for every fit.
    """
    if family in ("home", "point_diff", "elo"):
        return baseline_fitter(sport, family)
    if family == "sklearn":
        return sklearn_fitter(sport, model=model)
    if family == "gnn":
        return gnn_fitter(sport, max_epochs=max_epochs)
    if family == "transformer":
        return transformer_fitter(sport, max_epochs=max_epochs)
    raise ValueError(f"unknown model family {family!r}; choose from {FAMILIES}")


def no_vig_probability(moneyline, opp_moneyline) -> np.ndarray:
    """P(team wins) implied by American moneylines, bookmaker margin removed."""

    def implied(line):
        line = np.asarray(line, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(line > 0, 100 / (line + 100), -line / (100 - line))

    team, opp = implied(moneyline), implied(opp_moneyline)
    return team / (team + opp)


def payout(odds) -> np.ndarray:
    """Profit per unit staked on a winning bet at American ``odds``."""
    odds = np.asarray(odds, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(odds > 0, odds / 100, 100 / -odds)


def has_market(schema: GameSchema) -> bool:
    """Whether the schema names a market to score against (a spread or
    moneylines)."""
    return bool(schema.market_spread or schema.market_moneyline)


def implied_sigma(y, prob) -> float:
    """Residual spread ``s`` of margins around the ones win probabilities
    imply, ``s * z`` with ``z = Phi^-1(prob)``: the ``s`` whose implied
    margins leave residuals of spread ``s`` (``mean((y - s z)^2) = s^2``)."""
    y = np.asarray(y, dtype=float)
    z = scipy.stats.norm.ppf(np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6))
    ok = np.isfinite(y) & np.isfinite(z)
    y, z = y[ok], z[ok]
    if not len(y):
        return np.nan
    a, b, c = 1 - np.mean(z**2), 2 * np.mean(y * z), -np.mean(y**2)
    if a <= 0:
        return float(np.sqrt(-c))
    return float((-b + np.sqrt(b * b - 4 * a * c)) / (2 * a))


def market_sigma(history: pd.DataFrame, schema: GameSchema) -> float:
    """Residual spread of results around the market's margin, from earlier
    seasons' lined games: around the spread where there is one, else around
    the margin the moneylines imply (:func:`implied_sigma`)."""
    if schema.market_spread:
        lined = history.dropna(subset=[schema.market_spread])
        return fit_sigma(lined[schema.target], lined[schema.market_spread])
    team, opp = (history[col] for col in schema.market_moneyline)
    sigma = implied_sigma(history[schema.target], no_vig_probability(team, opp))
    return sigma if np.isfinite(sigma) else float(history[schema.target].std())


def market_predictions(
    rows: pd.DataFrame, schema: GameSchema, sigma: float
) -> pd.DataFrame:
    """The market's margin (spread) and win probability for each row, and
    (with a total line and its odds) ``market_over_prob``.

    Probabilities come from the moneylines where both exist, else from the
    spread with residual spread ``sigma``; the over probability comes from
    the no-vig over/under odds. Without a spread the margin is the one the
    moneylines imply with residual spread ``sigma``.
    """
    if schema.market_spread:
        margin = rows[schema.market_spread].astype(float)
        prob = win_probability(margin, sigma)
    else:
        margin = prob = np.full(len(rows), np.nan)
    if schema.market_moneyline:
        team, opp = (rows[col] for col in schema.market_moneyline)
        no_vig = no_vig_probability(team, opp)
        prob = np.where(np.isfinite(no_vig), no_vig, prob)
    if not schema.market_spread:
        z = scipy.stats.norm.ppf(np.clip(prob, 1e-6, 1 - 1e-6))
        margin = sigma * z
    out = pd.DataFrame({"market_margin": margin, "market_prob": prob}, index=rows.index)
    odds = schema.market_total_odds
    if odds and all(col in rows for col in odds):
        out["market_over_prob"] = no_vig_probability(rows[odds[0]], rows[odds[1]])
    return out


def walk_forward(
    games: pd.DataFrame,
    sport: str,
    fitter: Fitter,
    seasons: list[int],
    verbose: bool = False,
    on_date: OnDate | None = None,
    predictors: dict[int, Predictor] | None = None,
) -> pd.DataFrame:
    """Out-of-sample predictions for every played game of ``seasons``.

    ``on_date(predictor, season, date)`` is called once per test date, after
    the team state is refreshed and before that date's games are predicted.
    ``predictors`` maps seasons to an already fitted predictor to use instead
    of fitting one (it must not have been trained on that season or later).

    Returns one row per game (home team's perspective) with the result, the
    model's ``pred_margin`` / ``win_prob`` (and ``pred_total`` /
    ``over_prob`` from models that predict the total) and, if the schema
    has market columns, ``market_margin`` / ``market_prob``.
    """
    schema = get_sport(sport).schema
    games = played(games, schema)
    out = []
    for season in seasons:
        start = time.time()
        history = games[games[schema.season] < season]
        split = split_seasons(history[schema.season].unique(), test=[])
        if predictors and season in predictors:
            predictor = predictors[season]
        else:
            predictor = fitter(history, split)

        current = games[games[schema.season] == season]
        season_preds = []
        for date in sorted(current[schema.date].unique()):
            known = pd.concat([history, current[current[schema.date] < date]])
            predictor.update_state(known)
            if on_date is not None:
                on_date(predictor, season, date)
            day = one_row_per_game(current[current[schema.date] == date], schema)
            teams = set(predictor.model.teams)
            has_state = day[schema.team].isin(teams) & day[schema.opp].isin(teams)
            if has_state.any():
                season_preds.append(predictor.predict(day[has_state]))
        season_preds = pd.concat(season_preds, ignore_index=True)

        if has_market(schema):
            sigma = market_sigma(history, schema)
            market = market_predictions(season_preds, schema, sigma)
            season_preds = pd.concat([season_preds, market], axis=1)
        out.append(season_preds)
        if verbose:
            print(
                f"{season}: {len(season_preds)} games predicted "
                f"({time.time() - start:.0f}s)",
                flush=True,
            )

    results = pd.concat(out, ignore_index=True)
    keep = [*schema.key_columns, schema.location, schema.target, schema.total]
    keep += [schema.team_name, schema.opp_name, *schema.display]
    keep += [schema.market_spread, schema.market_total]
    keep += [*(schema.market_moneyline or ()), *(schema.market_total_odds or ())]
    keep += ["pred_margin", "win_prob", "market_margin", "market_prob"]
    keep += ["cover_prob", "pred_total", "over_prob", "market_over_prob"]
    return results[[c for c in dict.fromkeys(keep) if c and c in results]]


def _record(rows: list, t, bet, outcome, profit) -> None:
    """One betting-record row: ``outcome`` +1 win, -1 loss, 0 push per bet."""
    wins, losses = int((outcome > 0).sum()), int((outcome < 0).sum())
    decided = wins + losses
    pct = wins / decided if decided else np.nan
    half = 1.96 * np.sqrt(pct * (1 - pct) / decided) if decided else np.nan
    bets = int(bet.sum())
    rows.append(
        {
            "min_edge": t,
            "bets": bets,
            "wins": wins,
            "losses": losses,
            "pushes": int((outcome == 0).sum()),
            "win_pct": pct,
            "ci95_low": pct - half,
            "ci95_high": pct + half,
            "units": float(profit.sum()),
            "roi": float(profit.sum()) / max(bets, 1),
        }
    )


def ats_record(
    results: pd.DataFrame, schema: GameSchema, thresholds=(0, 1, 2, 3, 4, 5)
) -> pd.DataFrame:
    """Bet the side the model favors vs the spread when it disagrees by > t.

    A bet wins when that side covers; pushes return the stake. ROI assumes
    -110 on every bet.
    """
    edge = (results["pred_margin"] - results["market_margin"]).to_numpy()
    cover = (results[schema.target] - results["market_margin"]).to_numpy()
    rows = []
    for t in thresholds:
        bet = np.abs(edge) > t
        outcome = np.sign(edge[bet]) * np.sign(cover[bet])
        profit = np.where(outcome > 0, 100 / 110, np.where(outcome < 0, -1, 0))
        _record(rows, t, bet, outcome, profit)
    return pd.DataFrame(rows)


def moneyline_record(
    results: pd.DataFrame, schema: GameSchema, thresholds=(0, 0.02, 0.05, 0.1)
) -> pd.DataFrame:
    """Bet the side whose model win probability beats the market's no-vig
    probability by more than ``t``, at that side's posted moneyline.

    ``units`` / ``roi`` are profit at the posted odds (1 unit per bet); a
    tie returns the stake. Games without both moneylines are skipped.
    """
    team_col, opp_col = schema.market_moneyline
    lined = results.dropna(subset=[team_col, opp_col, "market_prob"])
    edge = (lined["win_prob"] - lined["market_prob"]).to_numpy()
    margin = lined[schema.target].to_numpy()
    odds = np.where(edge > 0, lined[team_col], lined[opp_col])
    rows = []
    for t in thresholds:
        bet = np.abs(edge) > t
        outcome = np.sign(edge[bet]) * np.sign(margin[bet])
        profit = np.where(outcome > 0, payout(odds[bet]), np.where(outcome < 0, -1, 0))
        _record(rows, t, bet, outcome, profit)
    return pd.DataFrame(rows)


def totals_record(
    results: pd.DataFrame, schema: GameSchema, thresholds=(0, 1, 2, 3, 4, 5)
) -> pd.DataFrame:
    """Bet the over when the model's total beats the line by more than
    ``t`` points, the under when it is below by more than ``t``.

    Profit is at the posted over/under odds where the schema has them (else
    -110); a total landing on the line returns the stake.
    """
    lined = results.dropna(subset=["pred_total", schema.market_total, schema.total])
    edge = (lined["pred_total"] - lined[schema.market_total]).to_numpy()
    went = (lined[schema.total] - lined[schema.market_total]).to_numpy()
    odds = np.full(len(lined), -110.0)
    if schema.market_total_odds and all(c in lined for c in schema.market_total_odds):
        over, under = (lined[c].to_numpy(dtype=float) for c in schema.market_total_odds)
        odds = np.where(edge > 0, over, under)
        odds = np.where(np.isfinite(odds), odds, -110.0)
    rows = []
    for t in thresholds:
        bet = np.abs(edge) > t
        outcome = np.sign(edge[bet]) * np.sign(went[bet])
        profit = np.where(outcome > 0, payout(odds[bet]), np.where(outcome < 0, -1, 0))
        _record(rows, t, bet, outcome, profit)
    return pd.DataFrame(rows)


def totals_summary(results: pd.DataFrame, schema: GameSchema) -> dict | None:
    """The model's total vs the market's line on the same games: RMSE,
    over/under Brier (on games that did not land on the line), the
    correlation of (model - line) with (total - line), and the bet record.
    """
    if "pred_total" not in results or not schema.market_total:
        return None
    lined = results.dropna(subset=["pred_total", schema.market_total, schema.total])
    if lined.empty:
        return None
    y = lined[schema.total].to_numpy(dtype=float)
    line = lined[schema.market_total].to_numpy(dtype=float)
    pred = lined["pred_total"].to_numpy(dtype=float)
    out = {
        "n": len(lined),
        "rmse": {
            "model": float(np.sqrt(np.mean((y - pred) ** 2))),
            "market": float(np.sqrt(np.mean((y - line) ** 2))),
        },
        "record": totals_record(lined, schema).to_dict("records"),
    }
    decided = y != line
    went_over = (y > line)[decided]
    brier = {}
    if "over_prob" in lined:
        brier["model"] = float(np.mean((lined["over_prob"][decided] - went_over) ** 2))
    if "market_over_prob" in lined:
        market = lined["market_over_prob"].to_numpy(dtype=float)[decided]
        ok = np.isfinite(market)
        if ok.any():
            brier["market"] = float(np.mean((market[ok] - went_over[ok]) ** 2))
    out["brier"] = brier
    edge, cover = pred - line, y - line
    r = float(np.corrcoef(edge, cover)[0, 1])
    half = 1.96 / np.sqrt(len(edge) - 3)
    out["edge_correlation"] = {
        "r": r,
        "ci95_low": float(np.tanh(np.arctanh(r) - half)),
        "ci95_high": float(np.tanh(np.arctanh(r) + half)),
    }
    return out


def edge_correlation(results: pd.DataFrame, schema: GameSchema) -> dict:
    """Correlation of (model - spread) with (result - spread), with 95% CI.

    Above zero means the model knows something the spread does not.
    """
    edge = results["pred_margin"] - results["market_margin"]
    cover = results[schema.target] - results["market_margin"]
    r = float(np.corrcoef(edge, cover)[0, 1])
    half = 1.96 / np.sqrt(len(edge) - 3)
    low, high = np.tanh(np.arctanh(r) - half), np.tanh(np.arctanh(r) + half)
    return {"r": r, "ci95_low": float(low), "ci95_high": float(high)}


def summarize(results: pd.DataFrame, schema: GameSchema, early_dates: int = 4):
    """Model (and market) metrics overall, by season and by phase of season.

    With market columns, every metric is computed on the games that have a
    market line, so model and market are scored on the same games.
    """
    sources = {"model": ("pred_margin", "win_prob")}
    if "market_margin" in results:
        sources["market"] = ("market_margin", "market_prob")
        results = results.dropna(subset=["market_margin", "market_prob"])

    def score(frame):
        y = frame[schema.target]
        return {
            name: evaluate(y, frame[margin], frame[prob])
            for name, (margin, prob) in sources.items()
        }

    date_rank = results.groupby(schema.season)[schema.date].rank(method="dense")
    phase = np.where(date_rank <= early_dates, f"first {early_dates} dates", "rest")
    summary = {
        "overall": score(results),
        "by_season": {int(s): score(f) for s, f in results.groupby(schema.season)},
        "by_phase": {p: score(f) for p, f in results.groupby(phase)},
    }
    if "market" in sources:
        if schema.market_spread:
            summary["ats"] = ats_record(results, schema).to_dict("records")
            summary["edge_correlation"] = edge_correlation(results, schema)
        if schema.market_moneyline and set(schema.market_moneyline) <= set(results):
            summary["moneyline"] = moneyline_record(results, schema).to_dict("records")
        totals = totals_summary(results, schema)
        if totals:
            summary["totals"] = totals
    return summary


def format_summary(summary: dict) -> str:
    cols = ["n", "rmse", "mae", "accuracy", "brier", "log_loss"]

    def table(scores: dict) -> pd.DataFrame:
        frame = pd.DataFrame(
            {(k, src): m for k, by_src in scores.items() for src, m in by_src.items()}
        ).T[cols]
        frame["n"] = frame["n"].astype(int)
        return frame

    fmt = {"float_format": lambda v: f"{v:.4f}"}
    parts = [
        "Overall\n" + table({"all": summary["overall"]}).droplevel(0).to_string(**fmt),
        "By season\n" + table(summary["by_season"]).to_string(**fmt),
        "By phase\n" + table(summary["by_phase"]).to_string(**fmt),
    ]
    if "ats" in summary:
        ats = pd.DataFrame(summary["ats"]).set_index("min_edge")
        parts.append(
            f"Against the spread (break-even {BREAKEVEN:.2%} at -110)\n"
            + ats.to_string(**fmt)
        )
        corr = summary["edge_correlation"]
        parts.append(
            "Edge vs cover correlation: "
            f"{corr['r']:+.4f} (95% CI {corr['ci95_low']:+.4f} to "
            f"{corr['ci95_high']:+.4f})"
        )
    if "moneyline" in summary:
        ml = pd.DataFrame(summary["moneyline"]).set_index("min_edge")
        parts.append(
            "Moneyline (model win probability minus the market's, profit at "
            "the posted odds)\n" + ml.to_string(**fmt)
        )
    if "totals" in summary:
        totals = summary["totals"]
        record = pd.DataFrame(totals["record"]).set_index("min_edge")
        brier = ", ".join(f"{k} {v:.4f}" for k, v in totals["brier"].items())
        corr = totals["edge_correlation"]
        parts.append(
            f"Over/under ({totals['n']} games): total RMSE model "
            f"{totals['rmse']['model']:.3f}, line {totals['rmse']['market']:.3f}; "
            f"over Brier {brier}; edge vs result correlation {corr['r']:+.4f} "
            f"(95% CI {corr['ci95_low']:+.4f} to {corr['ci95_high']:+.4f})\n"
            + record.to_string(**fmt)
        )
    return "\n\n".join(parts)


def complete_seasons(games: pd.DataFrame, schema: GameSchema) -> list[int]:
    """Fully played seasons with at least two earlier seasons (train + val)."""
    seasons = sorted(played(games, schema)[schema.season].unique())
    unfinished = set(upcoming(games, schema)[schema.season])
    return [int(s) for s in seasons[2:] if s not in unfinished]


def backtest(
    games: Table,
    sport: str,
    family: Family = "sklearn",
    seasons: list[int] | None = None,
    n_seasons: int = 5,
    model: sklearn.base.BaseEstimator | None = None,
    max_epochs: int = 100,
    output_dir: str | None = None,
    print_summary: bool = True,
    on_date: OnDate | None = None,
    predictors: dict[int, Predictor] | None = None,
) -> dict:
    """Walk-forward backtest of a model family against the market.

    Args:
        games: Processed games table.
        sport: Sport name, e.g. ``"nfl"``.
        family: Model family: a naive baseline (``home``, ``point_diff``,
            ``elo``) or ``sklearn`` / ``gnn`` / ``transformer``.
        seasons: Test seasons; default the last ``n_seasons`` complete ones.
        model: sklearn regressor (``family="sklearn"``); default random forest.
        max_epochs: Training epochs per season for ``gnn`` / ``transformer``.
        output_dir: Write ``<family>_games.csv`` (one row per game) and
            ``<family>_summary.json`` here.
        on_date / predictors: Passed to :func:`walk_forward`.

    Returns:
        ``{"results": per-game DataFrame, "summary": dict}``
    """
    games = read_table(games)
    schema = get_sport(sport).schema
    if seasons is None:
        seasons = complete_seasons(games, schema)[-n_seasons:]
    fitter = family_fitter(family, sport, model=model, max_epochs=max_epochs)

    results = walk_forward(
        games,
        sport,
        fitter,
        seasons,
        verbose=print_summary,
        on_date=on_date,
        predictors=predictors,
    )
    summary = {
        "sport": sport,
        "family": family,
        "seasons": [int(s) for s in seasons],
        **summarize(results, schema),
    }
    if output_dir:
        path = pathlib.Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        results.to_csv(path / f"{family}_games.csv", index=False)
        (path / f"{family}_summary.json").write_text(
            json.dumps(summary, indent=2, default=float)
        )
    if print_summary:
        print(format_summary(summary))
    return {"results": results, "summary": summary}


def compare(
    results: dict[str, pd.DataFrame], schema: GameSchema, early_dates: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score several families' walk-forward results on their common games.

    Returns:
        ``(table, games)``: one row per source (each family, plus the market
        when there is one) with RMSE, accuracy, Brier, log loss, early-season
        Brier and, against the market, the paired Brier difference with its
        95% CI, ATS win rate and edge correlation; and the per-game table
        with every source's ``<source>_margin`` / ``<source>_prob``.
    """
    common = set.intersection(*(set(r[schema.game_id]) for r in results.values()))
    first = next(iter(results.values()))
    base_cols = [*schema.key_columns, schema.location, schema.target]
    base_cols += ["market_margin", "market_prob"]
    games = first[first[schema.game_id].isin(common)]
    games = games[[c for c in dict.fromkeys(base_cols) if c and c in games]]
    games = games.dropna(subset=[c for c in ("market_margin",) if c in games])
    games = games.set_index(schema.game_id)

    sources = {}
    for name, frame in results.items():
        frame = frame.set_index(schema.game_id).reindex(games.index)
        games[f"{name}_margin"] = frame["pred_margin"]
        games[f"{name}_prob"] = frame["win_prob"]
        sources[name] = (f"{name}_margin", f"{name}_prob")
    has_market = "market_margin" in games
    if has_market:
        sources["market"] = ("market_margin", "market_prob")
    games = games.reset_index()

    y = games[schema.target].to_numpy(dtype=float)
    decided = y != 0
    won = y > 0
    date_rank = games.groupby(schema.season)[schema.date].rank(method="dense")
    early = (date_rank <= early_dates).to_numpy()

    def brier(prob):
        return (prob - won) ** 2

    rows = []
    for name, (margin_col, prob_col) in sources.items():
        margin = games[margin_col].to_numpy(dtype=float)
        prob = games[prob_col].to_numpy(dtype=float)
        row = {"source": name, **evaluate(y, margin, prob)}
        row["early_brier"] = float(brier(prob)[decided & early].mean())
        if has_market and name != "market":
            diff = (brier(prob) - brier(games["market_prob"].to_numpy()))[decided]
            half = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
            row |= {
                "brier_vs_market": float(diff.mean()),
                "brier_vs_market_ci95": float(half),
            }
            if schema.market_spread:
                one = games.assign(pred_margin=margin)
                ats = ats_record(one, schema, thresholds=(0,)).iloc[0]
                row["ats_win_pct"] = float(ats["win_pct"])
                row["edge_corr"] = edge_correlation(one, schema)["r"]
        rows.append(row)
    table = pd.DataFrame(rows).set_index("source").sort_values("brier")
    table["n"] = table["n"].astype(int)
    return table, games


def benchmark(
    games: Table,
    sport: str,
    families: list[Family] | None = None,
    seasons: list[int] | None = None,
    n_seasons: int = 5,
    model: sklearn.base.BaseEstimator | None = None,
    max_epochs: int = 100,
    output_dir: str | None = None,
    print_summary: bool = True,
) -> dict:
    """Walk-forward backtest of several families, compared on common games.

    Args:
        families: Families to run; default all (naive baselines included).
        seasons / n_seasons / model / max_epochs: As in :func:`backtest`.
        output_dir: Write ``benchmark.csv`` (the comparison table),
            ``benchmark.json`` and ``benchmark_games.csv`` (every source's
            prediction for every game).

    Returns:
        ``{"table", "games", "results"}``; ``results`` maps each family to its
        :func:`walk_forward` output.
    """
    games = read_table(games)
    schema = get_sport(sport).schema
    families = list(families or FAMILIES)
    if seasons is None:
        seasons = complete_seasons(games, schema)[-n_seasons:]

    results = {}
    for family in families:
        if print_summary:
            print(f"== {family}", flush=True)
        fitter = family_fitter(family, sport, model=model, max_epochs=max_epochs)
        results[family] = walk_forward(
            games, sport, fitter, seasons, verbose=print_summary
        )
    table, per_game = compare(results, schema)

    if output_dir:
        path = pathlib.Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        table.to_csv(path / "benchmark.csv")
        per_game.to_csv(path / "benchmark_games.csv", index=False)
        (path / "benchmark.json").write_text(
            json.dumps(
                {
                    "sport": sport,
                    "seasons": [int(s) for s in seasons],
                    "table": table.reset_index().to_dict("records"),
                },
                indent=2,
                default=float,
            )
        )
    if print_summary:
        columns = ["n", "rmse", "accuracy", "brier", "log_loss", "early_brier"]
        columns += [
            c
            for c in ("brier_vs_market", "brier_vs_market_ci95", "ats_win_pct")
            if c in table
        ]
        print(
            f"\nSeasons {seasons}, {len(per_game)} common games, sorted by Brier\n"
            + table[columns].to_string(float_format=lambda v: f"{v:.4f}")
        )
    return {"table": table, "games": per_game, "results": results}
